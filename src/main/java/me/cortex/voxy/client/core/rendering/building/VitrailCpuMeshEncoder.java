package me.cortex.voxy.client.core.rendering.building;

import me.cortex.voxy.common.util.MemoryBuffer;
import me.cortex.voxy.common.world.WorldEngine;
import org.lwjgl.system.MemoryUtil;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.function.IntBinaryOperator;
import java.util.function.IntUnaryOperator;

/** Expands a visible Voxy packed section into Vitrail's indexed 16-byte CPU mesh layout. */
public final class VitrailCpuMeshEncoder {
    public static final int VERTEX_STRIDE = 16;

    private VitrailCpuMeshEncoder() {}

    @FunctionalInterface
    public interface TintLookup {
        int colour(int modelId, int biomeId, int face, boolean opaque);
    }

    /** One opaque or translucent tile; caller owns both buffers and must close the mesh. */
    public record Mesh(int x, int y, int z, boolean opaque,
                       MemoryBuffer vertices, MemoryBuffer indices,
                       int vertexCount, int indexCount) implements AutoCloseable {
        @Override
        public void close() {
            this.vertices.free();
            this.indices.free();
        }
    }

    /** Encodes a section with white tint and Vitrail material zero. */
    public static List<Mesh> encode(BuiltSection section, IntBinaryOperator faceDataLookup) {
        return encode(section, faceDataLookup, null);
    }

    /** Encodes only one opacity half when the caller is building a single render pass. */
    public static List<Mesh> encode(BuiltSection section, IntBinaryOperator faceDataLookup, Boolean opaqueOnly) {
        return encode(section, faceDataLookup, (modelId, biomeId, face, opaque) -> 0xffff_ffff, opaqueOnly);
    }

    public static List<Mesh> encode(BuiltSection section, IntBinaryOperator faceDataLookup,
            TintLookup tintLookup, Boolean opaqueOnly) {
        return encode(section, faceDataLookup, tintLookup, ignored -> 0, opaqueOnly);
    }

    public static List<Mesh> encode(BuiltSection section, IntBinaryOperator faceDataLookup,
            TintLookup tintLookup, IntUnaryOperator materialLookup, Boolean opaqueOnly) {
        if (section == null || section.isEmpty() || section.geometryBuffer == null) return List.of();
        if (faceDataLookup == null) throw new NullPointerException("faceDataLookup");
        if (tintLookup == null) throw new NullPointerException("tintLookup");
        if (materialLookup == null) throw new NullPointerException("materialLookup");
        if (section.offsets == null || section.offsets.length < 8) {
            throw new IllegalArgumentException("BuiltSection does not contain Voxy quad-group offsets");
        }

        long totalLong = section.geometryBuffer.size / 8L;
        if (totalLong > Integer.MAX_VALUE) throw new IllegalArgumentException("Section contains too many quads");
        int totalQuads = (int) totalLong;
        Map<TileKey, MeshBuilder> builders = new LinkedHashMap<>();
        ArrayList<Mesh> meshes = new ArrayList<>();
        try {
            for (int group = 0; group < 8; group++) {
                boolean opaque = group != 0;
                if (opaqueOnly != null && opaqueOnly != opaque) continue;
                int from = section.offsets[group];
                int to = group == 7 ? totalQuads : section.offsets[group + 1];
                if (from < 0 || to < from || to > totalQuads) {
                    throw new IllegalArgumentException("Invalid quad offsets in BuiltSection");
                }
                for (int quadIndex = from; quadIndex < to; quadIndex++) {
                    long packed = MemoryUtil.memGetLong(section.geometryBuffer.address + quadIndex * 8L);
                    int face = (int) packed & 7;
                    int modelId = (int) ((packed >>> 26) & 0xffffL);
                    int biomeId = (int) ((packed >>> 46) & 0x1ffL);
                    int faceData = faceDataLookup.applyAsInt(modelId, face);
                    VitrailQuadEncoder.encode(packed, section.position,
                            WorldEngine.getLevel(section.position), faceData,
                            tintLookup.colour(modelId, biomeId, face, opaque),
                            materialLookup.applyAsInt(modelId),
                            (sx, sy, sz, v0, v1, v2, v3, reverse) -> {
                                TileKey key = new TileKey(sx, sy, sz, opaque);
                                MeshBuilder builder = builders.computeIfAbsent(key, ignored -> new MeshBuilder());
                                builder.append(v0, v1, v2, v3, reverse);
                            });
                }
            }

            meshes.ensureCapacity(builders.size());
            for (var entry : builders.entrySet()) {
                MeshBuilder builder = entry.getValue();
                TileKey key = entry.getKey();
                meshes.add(builder.finish(key));
            }
            return List.copyOf(meshes);
        } catch (RuntimeException | Error e) {
            meshes.forEach(Mesh::close);
            builders.values().forEach(MeshBuilder::free);
            throw e;
        }
    }

    private record TileKey(int x, int y, int z, boolean opaque) {}

    private static final class MeshBuilder {
        private MemoryBuffer vertices = new MemoryBuffer(1024);
        private MemoryBuffer indices = new MemoryBuffer(1536);
        private int vertexCount;
        private int indexCount;

        void append(VitrailQuadEncoder.Vertex v0, VitrailQuadEncoder.Vertex v1,
                VitrailQuadEncoder.Vertex v2, VitrailQuadEncoder.Vertex v3, boolean reverse) {
            ensureVertexCapacity(vertexCount + 4);
            ensureIndexCapacity(indexCount + 6);
            long ptr = vertices.address + (long) vertexCount * VERTEX_STRIDE;
            writeVertex(ptr, v0); writeVertex(ptr + VERTEX_STRIDE, v1);
            writeVertex(ptr + VERTEX_STRIDE * 2L, v2); writeVertex(ptr + VERTEX_STRIDE * 3L, v3);

            int base = vertexCount;
            if (reverse) {
                writeIndex(indexCount++, base); writeIndex(indexCount++, base + 2); writeIndex(indexCount++, base + 1);
                writeIndex(indexCount++, base + 2); writeIndex(indexCount++, base + 3); writeIndex(indexCount++, base + 1);
            } else {
                writeIndex(indexCount++, base); writeIndex(indexCount++, base + 1); writeIndex(indexCount++, base + 2);
                writeIndex(indexCount++, base + 2); writeIndex(indexCount++, base + 1); writeIndex(indexCount++, base + 3);
            }
            vertexCount += 4;
        }

        Mesh finish(TileKey key) {
            MemoryBuffer exactVertices = this.vertices.subSize((long) vertexCount * VERTEX_STRIDE);
            this.vertices = null;
            try {
                MemoryBuffer exactIndices = this.indices.subSize((long) indexCount * Integer.BYTES);
                this.indices = null;
                return new Mesh(key.x, key.y, key.z, key.opaque,
                        exactVertices, exactIndices, vertexCount, indexCount);
            } catch (RuntimeException | Error e) {
                exactVertices.free();
                throw e;
            }
        }

        void free() {
            if (this.vertices != null) this.vertices.free();
            if (this.indices != null) this.indices.free();
        }

        private void ensureVertexCapacity(int requiredVertices) {
            long required = (long) requiredVertices * VERTEX_STRIDE;
            if (required <= this.vertices.size) return;
            long capacity = Math.max(this.vertices.size * 2, required);
            MemoryBuffer expanded = new MemoryBuffer(capacity);
            this.vertices.cpyTo(expanded.address);
            this.vertices.free();
            this.vertices = expanded;
        }

        private void ensureIndexCapacity(int requiredIndices) {
            long required = (long) requiredIndices * Integer.BYTES;
            if (required <= this.indices.size) return;
            long capacity = Math.max(this.indices.size * 2, required);
            MemoryBuffer expanded = new MemoryBuffer(capacity);
            this.indices.cpyTo(expanded.address);
            this.indices.free();
            this.indices = expanded;
        }

        private void writeIndex(int index, int value) {
            MemoryUtil.memPutInt(this.indices.address + (long) index * Integer.BYTES, value);
        }
    }

    private static void writeVertex(long ptr, VitrailQuadEncoder.Vertex vertex) {
        MemoryUtil.memPutShort(ptr, (short) vertex.x());
        MemoryUtil.memPutShort(ptr + 2, (short) vertex.y());
        MemoryUtil.memPutShort(ptr + 4, (short) vertex.z());
        MemoryUtil.memPutShort(ptr + 6, (short) (vertex.light() & 0xff));
        int rgba = vertex.colour();
        MemoryUtil.memPutByte(ptr + 8, (byte) (rgba >>> 24));
        MemoryUtil.memPutByte(ptr + 9, (byte) (rgba >>> 16));
        MemoryUtil.memPutByte(ptr + 10, (byte) (rgba >>> 8));
        MemoryUtil.memPutByte(ptr + 11, (byte) rgba);
        MemoryUtil.memPutByte(ptr + 12, (byte) vertex.material());
        MemoryUtil.memPutByte(ptr + 13, (byte) vertex.normal());
        MemoryUtil.memPutShort(ptr + 14, (short) 0);
    }
}

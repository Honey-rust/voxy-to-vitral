package me.cortex.voxy.client.core.rendering.building;

import me.cortex.voxy.common.util.MemoryBuffer;
import me.cortex.voxy.common.util.UnsafeUtil;
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
    /** 2048 models x 6 faces fit in a 256x12288 atlas, normally leaving only two pages. */
    public static final int MODELS_PER_ATLAS_PAGE = 2048;

    private VitrailCpuMeshEncoder() {}

    @FunctionalInterface
    public interface TintLookup {
        int colour(int modelId, int biomeId, int face, boolean opaque);
    }

    /** One opaque or translucent tile; caller owns all three buffers and must close the mesh. */
    public record Mesh(int x, int y, int z, boolean opaque,
                       MemoryBuffer vertices, MemoryBuffer indices,
                       int vertexCount, int indexCount, MemoryBuffer detail, int atlasPage) implements AutoCloseable {
        @Override
        public void close() {
            this.vertices.free();
            this.indices.free();
            this.detail.free();
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
        return encode(section, faceDataLookup, tintLookup, materialLookup, ignored -> 0,
                (m, f) -> -1, opaqueOnly);
    }

    public static List<Mesh> encode(BuiltSection section, IntBinaryOperator faceDataLookup,
            TintLookup tintLookup, IntUnaryOperator materialLookup, IntUnaryOperator nearSuppressionLookup,
            IntBinaryOperator averageLookup, Boolean opaqueOnly) {
        if (section == null || section.isEmpty() || section.geometryBuffer == null) return List.of();
        if (faceDataLookup == null) throw new NullPointerException("faceDataLookup");
        if (tintLookup == null) throw new NullPointerException("tintLookup");
        if (materialLookup == null) throw new NullPointerException("materialLookup");
        if (nearSuppressionLookup == null) throw new NullPointerException("nearSuppressionLookup");
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
                    int nearSuppression = nearSuppressionLookup.applyAsInt(modelId) != 0 ? 0x8000 : 0;
                    VitrailQuadEncoder.encode(packed, section.position,
                            WorldEngine.getLevel(section.position), faceData,
                            tintLookup.colour(modelId, biomeId, face, opaque),
                            materialLookup.applyAsInt(modelId), opaque
                                    ? VitrailQuadEncoder.OPAQUE_TILE_BLOCKS
                                    : VitrailQuadEncoder.TRANSLUCENT_TILE_BLOCKS,
                            (sx, sy, sz, v0, v1, v2, v3, reverse) -> {
                                TileKey key = new TileKey(sx, sy, sz, opaque,
                                        modelId / MODELS_PER_ATLAS_PAGE);
                                MeshBuilder builder = builders.computeIfAbsent(key, ignored -> new MeshBuilder());
                                builder.append(v0, v1, v2, v3, reverse,
                                        (modelId % MODELS_PER_ATLAS_PAGE) * 6 + face,
                                        WorldEngine.getLevel(section.position), averageLookup.applyAsInt(modelId, face),
                                        nearSuppression);
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

    private record TileKey(int x, int y, int z, boolean opaque, int atlasPage) {}

    /** Combines meshes sharing one tile, pass and atlas page into a single draw buffer. */
    public static Mesh combine(List<Mesh> sources) {
        if (sources.isEmpty()) throw new IllegalArgumentException("No meshes to combine");
        Mesh first = sources.getFirst();
        long vertexBytes = 0;
        long detailBytes = 0;
        long indexBytes = 0;
        int vertexCount = 0;
        int indexCount = 0;
        for (Mesh source : sources) {
            if (source.x() != first.x() || source.y() != first.y() || source.z() != first.z()
                    || source.opaque() != first.opaque() || source.atlasPage() != first.atlasPage()) {
                throw new IllegalArgumentException("Cannot combine meshes from different tiles or atlas pages");
            }
            vertexBytes = Math.addExact(vertexBytes, source.vertices().size);
            detailBytes = Math.addExact(detailBytes, source.detail().size);
            indexBytes = Math.addExact(indexBytes, source.indices().size);
            vertexCount = Math.addExact(vertexCount, source.vertexCount());
            indexCount = Math.addExact(indexCount, source.indexCount());
        }
        MemoryBuffer vertices = new MemoryBuffer(vertexBytes);
        MemoryBuffer details = new MemoryBuffer(detailBytes);
        MemoryBuffer indices = new MemoryBuffer(indexBytes);
        long vertexOffset = 0;
        long detailOffset = 0;
        int writtenIndices = 0;
        int baseVertex = 0;
        try {
            for (Mesh source : sources) {
                UnsafeUtil.memcpy(source.vertices().address, vertices.address + vertexOffset, source.vertices().size);
                UnsafeUtil.memcpy(source.detail().address, details.address + detailOffset, source.detail().size);
                for (int i = 0; i < source.indexCount(); i++) {
                    int index = MemoryUtil.memGetInt(source.indices().address + (long)i * Integer.BYTES);
                    MemoryUtil.memPutInt(indices.address + (long)writtenIndices++ * Integer.BYTES,
                            Math.addExact(index, baseVertex));
                }
                vertexOffset += source.vertices().size;
                detailOffset += source.detail().size;
                baseVertex = Math.addExact(baseVertex, source.vertexCount());
            }
            return new Mesh(first.x(), first.y(), first.z(), first.opaque(),
                    vertices, indices, vertexCount, indexCount, details, first.atlasPage());
        } catch (RuntimeException | Error e) {
            vertices.free();
            indices.free();
            details.free();
            throw e;
        }
    }

    /**
     * Combines meshes from nearby tiles by rebasing their unsigned-short positions onto one
     * shared origin. Voxy normally submits these through one indirect draw stream; Vitrail's
     * provider API uses ordinary indexed draws, so doing the equivalent aggregation on the CPU
     * avoids one render pass draw for every source tile.
     */
    public static Mesh combineRebased(List<Mesh> sources) {
        if (sources.isEmpty()) throw new IllegalArgumentException("No meshes to combine");
        Mesh first = sources.getFirst();
        int originX = first.x();
        int originY = first.y();
        int originZ = first.z();
        long vertexBytes = 0;
        long detailBytes = 0;
        long indexBytes = 0;
        int vertexCount = 0;
        int indexCount = 0;
        for (Mesh source : sources) {
            if (source.opaque() != first.opaque() || source.atlasPage() != first.atlasPage()) {
                throw new IllegalArgumentException("Cannot combine different passes or atlas pages");
            }
            originX = Math.min(originX, source.x());
            originY = Math.min(originY, source.y());
            originZ = Math.min(originZ, source.z());
            vertexBytes = Math.addExact(vertexBytes, source.vertices().size);
            detailBytes = Math.addExact(detailBytes, source.detail().size);
            indexBytes = Math.addExact(indexBytes, source.indices().size);
            vertexCount = Math.addExact(vertexCount, source.vertexCount());
            indexCount = Math.addExact(indexCount, source.indexCount());
        }

        MemoryBuffer vertices = new MemoryBuffer(vertexBytes);
        MemoryBuffer details = new MemoryBuffer(detailBytes);
        MemoryBuffer indices = new MemoryBuffer(indexBytes);
        long vertexOffset = 0;
        long detailOffset = 0;
        int writtenIndices = 0;
        int baseVertex = 0;
        try {
            for (Mesh source : sources) {
                UnsafeUtil.memcpy(source.vertices().address, vertices.address + vertexOffset,
                        source.vertices().size);
                int dx = Math.subtractExact(source.x(), originX);
                int dy = Math.subtractExact(source.y(), originY);
                int dz = Math.subtractExact(source.z(), originZ);
                for (int i = 0; i < source.vertexCount(); i++) {
                    long sourcePtr = source.vertices().address + (long)i * VERTEX_STRIDE;
                    long targetPtr = vertices.address + vertexOffset + (long)i * VERTEX_STRIDE;
                    putUnsignedShort(targetPtr,
                            Math.addExact(Short.toUnsignedInt(MemoryUtil.memGetShort(sourcePtr)), dx));
                    putUnsignedShort(targetPtr + 2,
                            Math.addExact(Short.toUnsignedInt(MemoryUtil.memGetShort(sourcePtr + 2)), dy));
                    putUnsignedShort(targetPtr + 4,
                            Math.addExact(Short.toUnsignedInt(MemoryUtil.memGetShort(sourcePtr + 4)), dz));
                }
                UnsafeUtil.memcpy(source.detail().address, details.address + detailOffset,
                        source.detail().size);
                for (int i = 0; i < source.indexCount(); i++) {
                    int index = MemoryUtil.memGetInt(source.indices().address + (long)i * Integer.BYTES);
                    MemoryUtil.memPutInt(indices.address + (long)writtenIndices++ * Integer.BYTES,
                            Math.addExact(index, baseVertex));
                }
                vertexOffset += source.vertices().size;
                detailOffset += source.detail().size;
                baseVertex = Math.addExact(baseVertex, source.vertexCount());
            }
            return new Mesh(originX, originY, originZ, first.opaque(), vertices, indices,
                    vertexCount, indexCount, details, first.atlasPage());
        } catch (RuntimeException | Error e) {
            vertices.free();
            indices.free();
            details.free();
            throw e;
        }
    }

    private static void putUnsignedShort(long address, int value) {
        if ((value & ~0xffff) != 0) {
            throw new IllegalArgumentException("Rebased Vitrail vertex exceeds 16-bit position range: " + value);
        }
        MemoryUtil.memPutShort(address, (short)value);
    }

    private static final class MeshBuilder {
        private MemoryBuffer vertices = new MemoryBuffer(1024);
        private MemoryBuffer indices = new MemoryBuffer(1536);
        private MemoryBuffer detail = new MemoryBuffer(1024);
        private int vertexCount;
        private int indexCount;

        void append(VitrailQuadEncoder.Vertex v0, VitrailQuadEncoder.Vertex v1,
                VitrailQuadEncoder.Vertex v2, VitrailQuadEncoder.Vertex v3, boolean reverse,
                int tile, int lod, int average, int nearSuppression) {
            ensureVertexCapacity(vertexCount + 4);
            ensureIndexCapacity(indexCount + 6);
            long ptr = vertices.address + (long) vertexCount * VERTEX_STRIDE;
            writeVertex(ptr, v0, nearSuppression); writeVertex(ptr + VERTEX_STRIDE, v1, nearSuppression);
            writeVertex(ptr + VERTEX_STRIDE * 2L, v2, nearSuppression); writeVertex(ptr + VERTEX_STRIDE * 3L, v3, nearSuppression);
            VitrailQuadEncoder.Vertex[] corners = {v0, v1, v2, v3};
            for (int i = 0; i < 4; i++) {
                var v = corners[i];
                int axis = v.normal() >> 1;
                float scale = 1 << lod;
                float u = (axis == 2 ? v.y() : v.x()) / scale;
                float w = (axis == 1 ? v.y() : v.z()) / scale;
                long d = detail.address + (long)(vertexCount + i) * 16;
                MemoryUtil.memPutFloat(d, u);
                MemoryUtil.memPutFloat(d + 4, w);
                MemoryUtil.memPutInt(d + 8, tile);
                MemoryUtil.memPutByte(d + 12, (byte)(average >>> 16));
                MemoryUtil.memPutByte(d + 13, (byte)(average >>> 8));
                MemoryUtil.memPutByte(d + 14, (byte)average);
                MemoryUtil.memPutByte(d + 15, (byte)(average >>> 24));
            }

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
            MemoryBuffer exactIndices = null;
            try {
                exactIndices = this.indices.subSize((long) indexCount * Integer.BYTES);
                this.indices = null;
                return new Mesh(key.x, key.y, key.z, key.opaque,
                        exactVertices, exactIndices, vertexCount, indexCount, takeDetail(), key.atlasPage);
            } catch (RuntimeException | Error e) {
                exactVertices.free();
                if (exactIndices != null) exactIndices.free();
                throw e;
            }
        }

        void free() {
            if (this.vertices != null) this.vertices.free();
            if (this.indices != null) this.indices.free();
            if (this.detail != null) this.detail.free();
        }

        private MemoryBuffer takeDetail() {
            MemoryBuffer result = this.detail.subSize((long)vertexCount * 16);
            this.detail = null;
            return result;
        }

        private void ensureVertexCapacity(int requiredVertices) {
            long required = (long) requiredVertices * VERTEX_STRIDE;
            if (required <= this.vertices.size) return;
            long capacity = Math.max(this.vertices.size * 2, required);
            MemoryBuffer expanded = new MemoryBuffer(capacity);
            this.vertices.cpyTo(expanded.address);
            this.vertices.free();
            this.vertices = expanded;
            MemoryBuffer expandedDetail = new MemoryBuffer(capacity);
            this.detail.cpyTo(expandedDetail.address);
            this.detail.free();
            this.detail = expandedDetail;
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

    private static void writeVertex(long ptr, VitrailQuadEncoder.Vertex vertex, int nearSuppression) {
        MemoryUtil.memPutShort(ptr, (short) vertex.x());
        MemoryUtil.memPutShort(ptr + 2, (short) vertex.y());
        MemoryUtil.memPutShort(ptr + 4, (short) vertex.z());
        // Vitrail's DH-compatible nudge decoder uses bits 8-13; bit 15 is free.
        MemoryUtil.memPutShort(ptr + 6, (short) ((vertex.light() & 0xff) | nearSuppression));
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

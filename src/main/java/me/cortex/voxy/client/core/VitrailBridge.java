package me.cortex.voxy.client.core;

import com.mojang.blaze3d.buffers.GpuBuffer;
import com.mojang.blaze3d.GpuFormat;
import com.mojang.blaze3d.textures.GpuTexture;
import com.mojang.blaze3d.textures.GpuTextureView;
import com.mojang.blaze3d.systems.RenderSystem;
import me.cortex.voxy.client.core.rendering.building.VitrailCpuMeshEncoder;
import me.cortex.voxy.common.Logger;
import me.cortex.voxy.common.world.WorldEngine;
import org.lwjgl.system.MemoryUtil;

import java.lang.reflect.Constructor;
import java.lang.reflect.InvocationHandler;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.lang.reflect.Proxy;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

/** Optional reflective bridge into Vitrail, keeping Voxy loadable without the Vitrail mod. */
public final class VitrailBridge {
    private static final String RENDERER = "dev.vitrail.api.render.DistantTerrainRenderer";
    private static final String PROVIDER = "dev.vitrail.api.render.DistantTerrainProvider";
    private static final String SECTION = "dev.vitrail.api.render.DistantTerrainSection";
    private static final String PIECE = "dev.vitrail.api.render.DistantTerrainSection$Piece";

    private static volatile Method drawWithProviders;
    private static volatile Method providerInvocationCount;
    private static volatile Method registerProvider;
    private static volatile Method unregisterProvider;
    private static volatile Constructor<?> sectionConstructor;
    private static volatile Constructor<?> pieceConstructor;
    private static volatile boolean unavailable;
    private static volatile boolean warned;
    private static Object registeredProvider;
    private static long lastOpaqueInvocationCount;
    private static long lastTranslucentInvocationCount;
    private static AggregatePass opaqueAggregate;
    private static AggregatePass translucentAggregate;
    private static CombinedAggregateBuild aggregateBuild;
    private static List<NodeStamp> pendingStamps = List.of();
    private static long pendingSinceNanos;
    private static double lastCameraX = Double.NaN;
    private static double lastCameraZ = Double.NaN;
    private static final long INITIAL_STREAM_DELAY_NANOS = 100_000_000L;
    private static final long AGGREGATE_SETTLE_NANOS = 1_000_000_000L;
    private static final long AGGREGATE_FRAME_BUDGET_NANOS = 2_000_000L;
    private static final int MAX_SOURCE_MESHES_PER_UPLOAD = 24;
    private static final int STREAM_REGION_BLOCKS = 256;
    private static final double CAMERA_MOTION_EPSILON_SQUARED = 0.00000001;
    private static final Map<Integer, AtlasPage> atlasPages = new HashMap<>();

    private VitrailBridge() {}

    /** Registers Voxy as a provider during Vitrail's DH pass, or for direct calls without DH. */
    public static synchronized void registerProvider(VoxyRenderSystem renderer) {
        if (!RenderBackend.isVitrailVulkanActive() || unavailable || registeredProvider != null) return;
        try {
            ClassLoader loader = VitrailBridge.class.getClassLoader();
            Class<?> api = Class.forName(RENDERER, true, loader);
            Class<?> providerType = Class.forName(PROVIDER, true, loader);
            sectionConstructor = Class.forName(SECTION, true, loader)
                    .getConstructor(int.class, int.class, int.class, List.class);
            pieceConstructor = Class.forName(PIECE, true, loader)
                    .getConstructor(GpuBuffer.class, GpuBuffer.class, int.class, GpuBuffer.class, GpuTextureView.class);
            drawWithProviders = api.getMethod("drawWithProviders", boolean.class, List.class);
            registerProvider = api.getMethod("registerProvider", providerType);
            unregisterProvider = api.getMethod("unregisterProvider", providerType);
            providerInvocationCount = api.getMethod("providerInvocationCount", providerType, boolean.class);

            InvocationHandler handler = (proxy, method, args) -> {
                if (method.getName().equals("getSections") && args != null && args.length == 1) {
                    return buildSections(renderer, (Boolean) args[0]);
                }
                if (method.getName().equals("toString")) return "Voxy Vitrail terrain provider";
                if (method.getName().equals("hashCode")) return System.identityHashCode(proxy);
                if (method.getName().equals("equals")) return proxy == (args == null ? null : args[0]);
                return null;
            };
            registeredProvider = Proxy.newProxyInstance(providerType.getClassLoader(),
                    new Class<?>[]{providerType}, handler);
            registerProvider.invoke(null, registeredProvider);
            Logger.info("Registered Voxy's CPU terrain provider with Vitrail");
        } catch (ReflectiveOperationException | LinkageError e) {
            unavailable = true;
            warnOnce("Vitrail's distant-terrain provider API is unavailable", e);
        }
    }

    /** Called at Sodium's terrain stages if DH has not supplied this pass during the current frame. */
    public static void drawFromSodium(boolean opaque) {
        if (!RenderBackend.isVitrailVulkanActive() || unavailable) {
            return;
        }
        Method method = drawWithProviders;
        if (method == null || providerInvocationCount == null || registeredProvider == null) return;
        try {
            long calls = (long) providerInvocationCount.invoke(null, registeredProvider, opaque);
            long previous = opaque ? lastOpaqueInvocationCount : lastTranslucentInvocationCount;
            if (calls != previous) {
                if (opaque) lastOpaqueInvocationCount = calls;
                else lastTranslucentInvocationCount = calls;
                return;
            }
            method.invoke(null, opaque, List.of());
            calls = (long) providerInvocationCount.invoke(null, registeredProvider, opaque);
            if (opaque) lastOpaqueInvocationCount = calls;
            else lastTranslucentInvocationCount = calls;
        } catch (IllegalAccessException | InvocationTargetException | LinkageError e) {
            warnOnce("Could not submit Voxy terrain to Vitrail", e);
        }
    }

    public static synchronized void unregisterProvider() {
        Object provider = registeredProvider;
        registeredProvider = null;
        if (provider != null && unregisterProvider != null) {
            try {
                unregisterProvider.invoke(null, provider);
            } catch (ReflectiveOperationException | LinkageError e) {
                warnOnce("Could not unregister Voxy's Vitrail terrain provider", e);
            }
        }
        releaseAggregate(opaqueAggregate);
        releaseAggregate(translucentAggregate);
        if (aggregateBuild != null) aggregateBuild.cancel();
        opaqueAggregate = null;
        translucentAggregate = null;
        aggregateBuild = null;
        pendingStamps = List.of();
        pendingSinceNanos = 0;
        lastCameraX = Double.NaN;
        lastCameraZ = Double.NaN;
        for (var page : atlasPages.values()) { page.view.close(); page.texture.close(); }
        atlasPages.clear();
    }

    private static synchronized List<Object> buildSections(VoxyRenderSystem renderer, boolean opaque) {
        var device = RenderSystem.tryGetDevice();
        if (device == null || sectionConstructor == null || pieceConstructor == null) return List.of();
        AggregatePass current = opaque ? opaqueAggregate : translucentAggregate;

        // The opaque pass owns streaming progress so both render layers publish the same regions.
        if (!opaque) return displayedSections(current, false);

        double cameraX = renderer.getVitrailCameraX();
        double cameraZ = renderer.getVitrailCameraZ();
        List<me.cortex.voxy.client.core.rendering.hierachical.NodeManager.GeometryNode> visible = renderer.getVitrailVisibleNodes(
                cameraX, cameraZ);
        List<NodeStamp> stamps = visible.stream()
                .map(node -> new NodeStamp(node.position(), node.geometryId(), node.geometryVersion()))
                .sorted(Comparator.comparingLong(NodeStamp::position)
                        .thenComparingInt(NodeStamp::geometryId)
                        .thenComparingLong(NodeStamp::version))
                .toList();
        long now = System.nanoTime();
        double cameraDeltaX = cameraX - lastCameraX;
        double cameraDeltaZ = cameraZ - lastCameraZ;
        boolean cameraMoved = Double.isFinite(lastCameraX)
                && cameraDeltaX * cameraDeltaX + cameraDeltaZ * cameraDeltaZ > CAMERA_MOTION_EPSILON_SQUARED;
        lastCameraX = cameraX;
        lastCameraZ = cameraZ;

        if (aggregateBuild == null && opaqueAggregate != null && translucentAggregate != null
                && opaqueAggregate.stamps.equals(stamps) && translucentAggregate.stamps.equals(stamps)) {
            pendingStamps = List.of();
            pendingSinceNanos = 0;
            return current.sections;
        }

        if (aggregateBuild == null) {
            boolean initialStream = opaqueAggregate == null || opaqueAggregate.sections.isEmpty();
            if (initialStream) {
                // The initial Voxy population changes on nearly every frame. Resetting the normal
                // settle timer for each batch made us wait until generation was almost completely
                // quiet before showing anything. Start once the first useful snapshot has had a
                // very short chance to fill, then let the existing streamed rebuilds catch up.
                if (visible.isEmpty()) return current == null ? List.of() : current.sections;
                if (pendingSinceNanos == 0L) {
                    pendingSinceNanos = now;
                    pendingStamps = stamps;
                    return current == null ? List.of() : current.sections;
                }
                if (now - pendingSinceNanos < INITIAL_STREAM_DELAY_NANOS) {
                    return current == null ? List.of() : current.sections;
                }
            } else if (current != null && (cameraMoved || !pendingStamps.equals(stamps))) {
                pendingStamps = stamps;
                pendingSinceNanos = now;
                return current.sections;
            } else if (current != null && now - pendingSinceNanos < AGGREGATE_SETTLE_NANOS) {
                return current.sections;
            }
            aggregateBuild = new CombinedAggregateBuild(renderer, device, visible, stamps, cameraX, cameraZ);
        }

        try {
            CombinedAggregateBuild build = aggregateBuild;
            AggregatePair rebuilt = build.advance(AGGREGATE_FRAME_BUDGET_NANOS);
            if (rebuilt == null) return displayedSections(current, true);

            AggregatePass previousOpaque = opaqueAggregate;
            AggregatePass previousTranslucent = translucentAggregate;
            opaqueAggregate = rebuilt.opaque;
            translucentAggregate = rebuilt.translucent;
            aggregateBuild = null;
            pendingStamps = List.of();
            pendingSinceNanos = 0;
            releaseAggregate(previousOpaque);
            releaseAggregate(previousTranslucent);
            long rebuildMillis = (System.nanoTime() - build.startedNanos) / 1_000_000L;
            Logger.info("Streamed Voxy's Vitrail aggregate: " + visible.size() + " nodes across "
                    + build.regions.size() + " near-to-far regions, " + rebuilt.opaque.pieces.size()
                    + " opaque / " + rebuilt.translucent.pieces.size() + " translucent meshes over "
                    + build.frames + " frames / " + rebuildMillis + " ms");
            return rebuilt.opaque.sections;
        } catch (ReflectiveOperationException | RuntimeException | Error e) {
            if (aggregateBuild != null) aggregateBuild.cancel();
            aggregateBuild = null;
            throw new IllegalStateException("Failed to build Voxy's Vitrail meshes", e);
        }
    }

    private static List<Object> displayedSections(AggregatePass current, boolean opaque) {
        if (aggregateBuild != null && (current == null || current.sections.isEmpty())) {
            try {
                List<Object> preview = aggregateBuild.previewSections(opaque);
                if (!preview.isEmpty()) return preview;
            } catch (ReflectiveOperationException e) {
                throw new IllegalStateException("Failed to publish a streamed Voxy region", e);
            }
        }
        return current == null ? List.of() : current.sections;
    }

    /** Streams complete regions from the camera outwards, publishing opaque and translucent pieces together. */
    private static final class CombinedAggregateBuild {
        private final VoxyRenderSystem renderer;
        private final com.mojang.blaze3d.systems.GpuDevice device;
        private final List<NodeStamp> stamps;
        private final List<RegionBatch> regions;
        private final Map<GroupKey, List<VitrailCpuMeshEncoder.Mesh>> opaqueGroups = new LinkedHashMap<>();
        private final Map<GroupKey, List<VitrailCpuMeshEncoder.Mesh>> translucentGroups = new LinkedHashMap<>();
        private final ArrayList<VitrailCpuMeshEncoder.Mesh> sourceMeshes = new ArrayList<>();
        private final ArrayList<UploadedPiece> opaqueUploaded = new ArrayList<>();
        private final ArrayList<UploadedPiece> translucentUploaded = new ArrayList<>();
        private final long startedNanos = System.nanoTime();
        private List<UploadTask> uploadTasks;
        private int regionIndex;
        private int nodeIndex;
        private int uploadIndex;
        private int committedOpaquePieces;
        private int committedTranslucentPieces;
        private int frames;
        private boolean transferred;

        private CombinedAggregateBuild(VoxyRenderSystem renderer,
                com.mojang.blaze3d.systems.GpuDevice device,
                List<me.cortex.voxy.client.core.rendering.hierachical.NodeManager.GeometryNode> visible,
                List<NodeStamp> stamps, double cameraX, double cameraZ) {
            this.renderer = renderer;
            this.device = device;
            this.stamps = stamps;
            this.regions = createRegions(visible, cameraX, cameraZ);
        }

        private AggregatePair advance(long budgetNanos) throws ReflectiveOperationException {
            this.frames++;
            long deadline = System.nanoTime() + budgetNanos;
            boolean didWork = false;

            while (this.regionIndex < this.regions.size()) {
                RegionBatch region = this.regions.get(this.regionIndex);
                while (this.nodeIndex < region.nodes.size() && (!didWork || System.nanoTime() < deadline)) {
                    var node = region.nodes.get(this.nodeIndex++);
                    var section = this.renderer.getVitrailCpuSectionSnapshot(node.geometryId());
                    if (section != null) {
                        try {
                            if (section.position == node.position()) {
                                List<VitrailCpuMeshEncoder.Mesh> meshes = VitrailCpuMeshEncoder.encode(section,
                                        this.renderer::getVitrailModelFaceData, this.renderer::getVitrailFaceColour,
                                        this.renderer::getVitrailModelMaterial, this.renderer::getVitrailFaceAverage,
                                        null);
                                this.sourceMeshes.addAll(meshes);
                                for (var mesh : meshes) {
                                    GroupKey key = new GroupKey(new TileKey(mesh.x(), mesh.y(), mesh.z()), mesh.atlasPage());
                                    Map<GroupKey, List<VitrailCpuMeshEncoder.Mesh>> groups = mesh.opaque()
                                            ? this.opaqueGroups : this.translucentGroups;
                                    groups.computeIfAbsent(key, ignored -> new ArrayList<>()).add(mesh);
                                }
                            }
                        } finally {
                            section.free();
                        }
                    }
                    didWork = true;
                }
                if (this.nodeIndex < region.nodes.size()) return null;

                if (this.uploadTasks == null) {
                    ArrayList<UploadTask> tasks = new ArrayList<>();
                    addUploadTasks(tasks, this.opaqueGroups);
                    addUploadTasks(tasks, this.translucentGroups);
                    this.uploadTasks = List.copyOf(tasks);
                }

                while (this.uploadIndex < this.uploadTasks.size() && (!didWork || System.nanoTime() < deadline)) {
                    upload(this.uploadTasks.get(this.uploadIndex++));
                    didWork = true;
                }
                if (this.uploadIndex < this.uploadTasks.size()) return null;

                this.sourceMeshes.forEach(VitrailCpuMeshEncoder.Mesh::close);
                this.sourceMeshes.clear();
                this.opaqueGroups.clear();
                this.translucentGroups.clear();
                this.uploadTasks = null;
                this.uploadIndex = 0;
                this.nodeIndex = 0;
                this.regionIndex++;
                this.committedOpaquePieces = this.opaqueUploaded.size();
                this.committedTranslucentPieces = this.translucentUploaded.size();
                if (System.nanoTime() >= deadline) return null;
            }

            AggregatePass opaque = new AggregatePass(this.stamps,
                    makeSections(this.opaqueUploaded, this.opaqueUploaded.size()), List.copyOf(this.opaqueUploaded));
            AggregatePass translucent = new AggregatePass(this.stamps,
                    makeSections(this.translucentUploaded, this.translucentUploaded.size()),
                    List.copyOf(this.translucentUploaded));
            this.opaqueUploaded.clear();
            this.translucentUploaded.clear();
            this.transferred = true;
            return new AggregatePair(opaque, translucent);
        }

        private List<Object> previewSections(boolean opaque) throws ReflectiveOperationException {
            List<UploadedPiece> pieces = opaque ? this.opaqueUploaded : this.translucentUploaded;
            int committed = opaque ? this.committedOpaquePieces : this.committedTranslucentPieces;
            return committed == 0 ? List.of() : makeSections(pieces, committed);
        }

        private List<Object> makeSections(List<UploadedPiece> pieces, int limit) throws ReflectiveOperationException {
            LinkedHashMap<TileKey, List<Object>> piecesByTile = new LinkedHashMap<>();
            for (int i = 0; i < limit; i++) {
                UploadedPiece piece = pieces.get(i);
                piecesByTile.computeIfAbsent(piece.tile, ignored -> new ArrayList<>()).add(piece.apiPiece);
            }
            ArrayList<Object> sections = new ArrayList<>(piecesByTile.size());
            for (var entry : piecesByTile.entrySet()) {
                TileKey key = entry.getKey();
                sections.add(sectionConstructor.newInstance(key.x, key.y, key.z, List.copyOf(entry.getValue())));
            }
            return List.copyOf(sections);
        }

        private void upload(UploadTask task) throws ReflectiveOperationException {
            List<VitrailCpuMeshEncoder.Mesh> parts = task.parts;
            VitrailCpuMeshEncoder.Mesh mesh = parts.size() == 1
                    ? parts.getFirst() : VitrailCpuMeshEncoder.combine(parts);
            boolean combined = parts.size() != 1;
            GpuBuffer vertex = null;
            GpuBuffer index = null;
            GpuBuffer detail = null;
            try {
                vertex = this.device.createBuffer(() -> "Voxy Vitrail aggregate vertices", GpuBuffer.USAGE_VERTEX,
                        MemoryUtil.memByteBuffer(mesh.vertices().address, Math.toIntExact(mesh.vertices().size)));
                index = this.device.createBuffer(() -> "Voxy Vitrail aggregate indices", GpuBuffer.USAGE_INDEX,
                        MemoryUtil.memByteBuffer(mesh.indices().address, Math.toIntExact(mesh.indices().size)));
                var atlas = atlasFor(this.renderer, this.device, mesh);
                detail = this.device.createBuffer(() -> "Voxy aggregate LOD texture coordinates", GpuBuffer.USAGE_VERTEX,
                        MemoryUtil.memByteBuffer(mesh.detail().address, Math.toIntExact(mesh.detail().size)));
                Object apiPiece = pieceConstructor.newInstance(vertex, index, mesh.indexCount(), detail, atlas);
                ArrayList<UploadedPiece> uploaded = mesh.opaque() ? this.opaqueUploaded : this.translucentUploaded;
                uploaded.add(new UploadedPiece(task.key.tile, apiPiece, vertex, index, detail));
            } catch (RuntimeException | ReflectiveOperationException | Error e) {
                if (vertex != null) vertex.close();
                if (index != null) index.close();
                if (detail != null) detail.close();
                throw e;
            } finally {
                if (combined) mesh.close();
            }
        }

        private void cancel() {
            if (this.transferred) return;
            this.sourceMeshes.forEach(VitrailCpuMeshEncoder.Mesh::close);
            this.sourceMeshes.clear();
            releasePieces(this.opaqueUploaded);
            releasePieces(this.translucentUploaded);
            this.opaqueUploaded.clear();
            this.translucentUploaded.clear();
        }

        private static void addUploadTasks(List<UploadTask> tasks,
                Map<GroupKey, List<VitrailCpuMeshEncoder.Mesh>> groups) {
            for (var entry : groups.entrySet()) {
                List<VitrailCpuMeshEncoder.Mesh> parts = entry.getValue();
                for (int from = 0; from < parts.size(); from += MAX_SOURCE_MESHES_PER_UPLOAD) {
                    int to = Math.min(parts.size(), from + MAX_SOURCE_MESHES_PER_UPLOAD);
                    tasks.add(new UploadTask(entry.getKey(), parts.subList(from, to)));
                }
            }
        }

        private static List<RegionBatch> createRegions(
                List<me.cortex.voxy.client.core.rendering.hierachical.NodeManager.GeometryNode> visible,
                double cameraX, double cameraZ) {
            Map<RegionKey, ArrayList<me.cortex.voxy.client.core.rendering.hierachical.NodeManager.GeometryNode>> grouped
                    = new HashMap<>();
            for (var node : visible) {
                long position = node.position();
                double size = 32.0 * (1L << WorldEngine.getLevel(position));
                double centerX = (WorldEngine.getX(position) + 0.5) * size;
                double centerZ = (WorldEngine.getZ(position) + 0.5) * size;
                RegionKey key = new RegionKey((int)Math.floor(centerX / STREAM_REGION_BLOCKS),
                        (int)Math.floor(centerZ / STREAM_REGION_BLOCKS));
                grouped.computeIfAbsent(key, ignored -> new ArrayList<>()).add(node);
            }
            ArrayList<RegionBatch> result = new ArrayList<>(grouped.size());
            for (var entry : grouped.entrySet()) result.add(new RegionBatch(entry.getKey(), List.copyOf(entry.getValue())));
            result.sort(Comparator
                    .comparingDouble((RegionBatch region) -> regionDistanceSquared(region.key, cameraX, cameraZ))
                    .thenComparingInt(region -> region.key.x)
                    .thenComparingInt(region -> region.key.z));
            return List.copyOf(result);
        }

        private static double regionDistanceSquared(RegionKey region, double cameraX, double cameraZ) {
            double centerX = (region.x + 0.5) * STREAM_REGION_BLOCKS;
            double centerZ = (region.z + 0.5) * STREAM_REGION_BLOCKS;
            double dx = centerX - cameraX;
            double dz = centerZ - cameraZ;
            return dx * dx + dz * dz;
        }
    }

    private static void releaseAggregate(AggregatePass aggregate) {
        if (aggregate != null) releasePieces(aggregate.pieces);
    }

    private static void releasePieces(List<UploadedPiece> pieces) {
        for (UploadedPiece piece : pieces) {
            try { piece.vertex.close(); } catch (RuntimeException e) { Logger.warn("Could not release Vitrail vertex buffer", e); }
            try { piece.index.close(); } catch (RuntimeException e) { Logger.warn("Could not release Vitrail index buffer", e); }
            try { piece.detail.close(); } catch (RuntimeException e) { Logger.warn("Could not release Vitrail detail buffer", e); }
        }
    }

    private static void warnOnce(String message, Throwable error) {
        if (!warned) {
            synchronized (VitrailBridge.class) {
                if (!warned) {
                    warned = true;
                    Logger.warn(message + ": " + error);
                }
            }
        }
    }

    private record TileKey(int x, int y, int z) {}
    private record GroupKey(TileKey tile, int atlasPage) {}
    private record UploadTask(GroupKey key, List<VitrailCpuMeshEncoder.Mesh> parts) {}
    private record RegionKey(int x, int z) {}
    private record RegionBatch(RegionKey key,
            List<me.cortex.voxy.client.core.rendering.hierachical.NodeManager.GeometryNode> nodes) {}
    private record NodeStamp(long position, int geometryId, long version) {}
    private record UploadedPiece(TileKey tile, Object apiPiece, GpuBuffer vertex, GpuBuffer index, GpuBuffer detail) {}
    private record AggregatePass(List<NodeStamp> stamps, List<Object> sections, List<UploadedPiece> pieces) {}
    private record AggregatePair(AggregatePass opaque, AggregatePass translucent) {}
    private record AtlasPage(GpuTexture texture, GpuTextureView view, Set<Integer> uploaded) {}

    private static GpuTextureView atlasFor(VoxyRenderSystem renderer, com.mojang.blaze3d.systems.GpuDevice device,
            VitrailCpuMeshEncoder.Mesh mesh) {
        AtlasPage page = atlasPages.computeIfAbsent(mesh.atlasPage(), ignored -> {
            GpuTexture texture = device.createTexture(() -> "Voxy baked LOD atlas", GpuTexture.USAGE_TEXTURE_BINDING | GpuTexture.USAGE_COPY_DST,
                    GpuFormat.RGBA8_UNORM, 256, 1536, 1, 1);
            try { return new AtlasPage(texture, device.createTextureView(texture), new HashSet<>()); }
            catch (RuntimeException | Error e) { texture.close(); throw e; }
        });
        var encoder = device.createCommandEncoder();
        ByteBuffer pixels = MemoryUtil.memAlloc(16 * 16 * 4);
        try {
            for (int i = 0; i < mesh.vertexCount(); i += 4) {
                int tile = MemoryUtil.memGetInt(mesh.detail().address + (long)i * 16 + 8);
                if (page.uploaded.contains(tile)) continue;
                int[] source = renderer.getVitrailFacePixels(mesh.atlasPage() * 256 + tile / 6, tile % 6);
                pixels.clear();
                for (int p = 0; p < 256; p++) {
                    int abgr = source == null ? -1 : source[p];
                    pixels.put((byte)abgr).put((byte)(abgr >>> 8)).put((byte)(abgr >>> 16)).put((byte)(abgr >>> 24));
                }
                pixels.flip();
                encoder.writeToTexture(page.texture, pixels, 0, 0, (tile % 16) * 16, (tile / 16) * 16, 16, 16);
                page.uploaded.add(tile);
            }
        } finally { MemoryUtil.memFree(pixels); }
        return page.view;
    }
}

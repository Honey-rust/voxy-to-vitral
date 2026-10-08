package me.cortex.voxy.client.core;

import com.mojang.blaze3d.buffers.GpuBuffer;
import com.mojang.blaze3d.GpuFormat;
import com.mojang.blaze3d.textures.GpuTexture;
import com.mojang.blaze3d.textures.GpuTextureView;
import com.mojang.blaze3d.systems.RenderSystem;
import me.cortex.voxy.client.core.rendering.building.VitrailCpuMeshEncoder;
import me.cortex.voxy.client.core.rendering.compat.PersistentGeometryCache;
import me.cortex.voxy.client.config.VoxyConfig;
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
    private static volatile Method usesPlainRenderer;
    private static volatile Method capturePlainView;
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
    private static long pendingFirstDirtyNanos;
    private static long pendingSettleNanos = 1_000_000_000L;
    private static double lastCameraX = Double.NaN;
    private static double lastCameraZ = Double.NaN;
    private static double lastProjectionScalePixels = Double.NaN;
    private static final long INITIAL_STREAM_DELAY_NANOS = 100_000_000L;
    private static final long AGGREGATE_SETTLE_NANOS = 220_000_000L;
    private static final long MAX_AGGREGATE_DEFER_NANOS = 900_000_000L;
    private static final long INITIAL_AGGREGATE_FRAME_BUDGET_NANOS = 8_000_000L;
    private static final long UPDATE_AGGREGATE_FRAME_BUDGET_NANOS = 3_000_000L;
    // Voxy's native indirect renderer submits thousands of nodes in very few GPU commands. The
    // provider path has ordinary indexed draws, so aggregate more source nodes per piece and use
    // a larger streaming region to keep draw-call count close to the native design.
    private static final int MAX_SOURCE_MESHES_PER_UPLOAD = 192;
    /** Keeps rebased unsigned-short positions comfortably inside their 65535-block range. */
    private static final int AGGREGATE_CELL_BLOCKS = 32768;
    private static final int ATLAS_WIDTH = 256;
    private static final int ATLAS_HEIGHT = VitrailCpuMeshEncoder.MODELS_PER_ATLAS_PAGE * 6;
    private static final int STREAM_REGION_BLOCKS = 1024;
    private static final double CAMERA_MOTION_EPSILON_SQUARED = 0.0025;
    private static final Comparator<NodeStamp> NODE_STAMP_ORDER = Comparator
            .comparingLong(NodeStamp::position)
            .thenComparingInt(NodeStamp::geometryId)
            .thenComparingLong(NodeStamp::version);
    private static final Map<Integer, AtlasPage> atlasPages = new HashMap<>();
    private static final PersistentGeometryCache geometryCache = new PersistentGeometryCache();

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
            usesPlainRenderer = api.getMethod("usesPlainRenderer");
            capturePlainView = Class.forName("dev.vitrail.render.PlainDistantDraw", true, loader)
                    .getMethod("captureTerrainView", org.joml.Matrix4fc.class, org.joml.Matrix4fc.class,
                            double.class, double.class, double.class);

            InvocationHandler handler = (proxy, method, args) -> {
                if (method.getName().equals("getSections") && args != null && args.length == 1) {
                    return buildSections(renderer, (Boolean) args[0]);
                }
                if (method.getName().equals("renderDistanceBlocks")) {
                    return Math.max(1, Math.round(VoxyConfig.CONFIG.sectionRenderDistance * 32.0F * 16.0F));
                }
                if (method.getName().equals("nearPlaneBlocks")) {
                    // Native Voxy resolves the overlap with world depth/Hi-Z. A radial fragment
                    // cut exposes caves when looking down, so keep the complete cut behind vanilla.
                    return 0.0F;
                }
                if (method.getName().equals("farPlaneBlocks")) {
                    return (VoxyConfig.CONFIG.sectionRenderDistance * 32.0F + 2.0F)
                            * (float) Math.sqrt(3.0) * 16.0F;
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

    public static void drawPlainFromSodium(org.joml.Matrix4fc projection, org.joml.Matrix4fc view,
            double x, double y, double z) {
        if (capturePlainView == null) return;
        try {
            capturePlainView.invoke(null, projection, view, x, y, z);
            drawFromSodium(true);
        } catch (ReflectiveOperationException | LinkageError e) {
            warnOnce("Could not capture Sodium's current terrain view", e);
        }
    }

    public static boolean usesPlainRenderer() {
        Method method = usesPlainRenderer;
        if (method == null) return false;
        try {
            return (boolean) method.invoke(null);
        } catch (ReflectiveOperationException | LinkageError e) {
            warnOnce("Could not determine Vitrail's distant renderer", e);
            return false;
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
        pendingFirstDirtyNanos = 0;
        pendingSettleNanos = AGGREGATE_SETTLE_NANOS;
        lastCameraX = Double.NaN;
        lastCameraZ = Double.NaN;
        lastProjectionScalePixels = Double.NaN;
        for (var page : atlasPages.values()) { page.view.close(); page.texture.close(); }
        atlasPages.clear();
        geometryCache.close();
    }

    private static synchronized List<Object> buildSections(VoxyRenderSystem renderer, boolean opaque) {
        var device = RenderSystem.tryGetDevice();
        if (device == null || sectionConstructor == null || pieceConstructor == null) return List.of();
        AggregatePass current = opaque ? opaqueAggregate : translucentAggregate;

        // The opaque pass owns streaming progress so both render layers publish the same regions.
        if (!opaque) return displayedSections(current, false);

        double cameraX = renderer.getVitrailCameraX();
        double cameraZ = renderer.getVitrailCameraZ();
        boolean hasCompleteBaseline = opaqueAggregate != null && !opaqueAggregate.sections.isEmpty();
        List<me.cortex.voxy.client.core.rendering.hierachical.NodeManager.GeometryNode> visible =
                hasCompleteBaseline ? renderer.getVitrailVisibleNodes(cameraX, cameraZ)
                        : renderer.getVitrailCoarseVisibleNodes();
        List<NodeStamp> stamps = visible.stream()
                .map(node -> new NodeStamp(node.position(), node.geometryId(), node.geometryVersion()))
                .sorted(NODE_STAMP_ORDER)
                .toList();
        long now = System.nanoTime();
        double cameraDeltaX = cameraX - lastCameraX;
        double cameraDeltaZ = cameraZ - lastCameraZ;
        boolean cameraMoved = Double.isFinite(lastCameraX)
                && cameraDeltaX * cameraDeltaX + cameraDeltaZ * cameraDeltaZ > CAMERA_MOTION_EPSILON_SQUARED;
        double projectionScalePixels = renderer.getVitrailProjectionScalePixels();
        boolean projectionChanged = Double.isFinite(lastProjectionScalePixels)
                && Math.abs(projectionScalePixels - lastProjectionScalePixels)
                > Math.max(1.0, Math.abs(lastProjectionScalePixels) * 0.005);
        lastCameraX = cameraX;
        lastCameraZ = cameraZ;
        lastProjectionScalePixels = projectionScalePixels;

        // Finish an in-flight complete cut. Restarting it whenever the camera turns can starve
        // every direction indefinitely; the persistent traversal history below prevents a later
        // cut from deliberately downgrading detail that the player has already seen.
        if (aggregateBuild == null
                && opaqueAggregate != null && translucentAggregate != null
                && opaqueAggregate.stamps.equals(stamps) && translucentAggregate.stamps.equals(stamps)) {
            pendingStamps = List.of();
            pendingSinceNanos = 0;
            pendingFirstDirtyNanos = 0;
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
                boolean cutChanged = !pendingStamps.equals(stamps);
                pendingStamps = stamps;
                if (pendingFirstDirtyNanos == 0L) pendingFirstDirtyNanos = now;
                if (pendingSinceNanos == 0L || cameraMoved || cutChanged) pendingSinceNanos = now;
                if (projectionChanged) pendingSettleNanos = INITIAL_STREAM_DELAY_NANOS;
                // Let rapid turns settle so we refine the view the player actually stopped on.
                // A hard deadline still publishes progress while chunks continue arriving.
                if (now - pendingSinceNanos < pendingSettleNanos
                        && now - pendingFirstDirtyNanos < MAX_AGGREGATE_DEFER_NANOS) {
                    return current.sections;
                }
            }
            aggregateBuild = new CombinedAggregateBuild(renderer, device, visible, stamps, cameraX, cameraZ);
        }

        try {
            CombinedAggregateBuild build = aggregateBuild;
            // Fill an empty view promptly. Once a complete cut is already displayed, spend a much
            // smaller slice of each frame refining it, matching Voxy's background-work policy and
            // avoiding a permanent 2 ms tax while the player moves or turns.
            long frameBudget = current == null || current.sections.isEmpty()
                    ? INITIAL_AGGREGATE_FRAME_BUDGET_NANOS
                    : UPDATE_AGGREGATE_FRAME_BUDGET_NANOS;
            AggregatePair rebuilt = build.advance(frameBudget);
            if (rebuilt == null) return displayedSections(current, true);

            // The cut may have become finer while this build was running. The completed cut still
            // has full coverage, so publish it as the next coarse-to-fine step and queue the latest.
            boolean stale = !build.stamps.equals(stamps);

            AggregatePass previousOpaque = opaqueAggregate;
            AggregatePass previousTranslucent = translucentAggregate;
            opaqueAggregate = rebuilt.opaque;
            translucentAggregate = rebuilt.translucent;
            aggregateBuild = null;
            if (stale) {
                // Keep progressing through complete cuts instead of throwing away encoded work.
                pendingStamps = stamps;
                pendingSinceNanos = now;
                pendingFirstDirtyNanos = now;
                pendingSettleNanos = INITIAL_STREAM_DELAY_NANOS;
            } else {
                pendingStamps = List.of();
                pendingSinceNanos = 0;
                pendingFirstDirtyNanos = 0;
                pendingSettleNanos = AGGREGATE_SETTLE_NANOS;
            }
            releaseAggregate(previousOpaque);
            releaseAggregate(previousTranslucent);
            long rebuildMillis = (System.nanoTime() - build.startedNanos) / 1_000_000L;
            PersistentGeometryCache.Stats cacheStats = geometryCache.stats();
            Logger.info("Streamed Voxy's Vitrail aggregate: " + build.stamps.size() + " nodes across "
                    + build.regions.size() + " near-to-far regions, " + rebuilt.opaque.pieces.size()
                    + " opaque / " + rebuilt.translucent.pieces.size() + " translucent meshes over "
                    + build.frames + " frames / " + rebuildMillis + " ms; exact hierarchy cut, effective subdivision "
                    + Math.round(renderer.getVitrailEffectiveSubdivisionPixels() * 10.0) / 10.0 + " px; geometry cache "
                    + build.cacheHits + " hit / " + build.cacheMisses + " miss, " + cacheStats.entries()
                    + " entries / " + (cacheStats.residentBytes() / (1024L * 1024L)) + " MiB");
            return rebuilt.opaque.sections;
        } catch (ReflectiveOperationException | RuntimeException | Error e) {
            if (aggregateBuild != null) aggregateBuild.cancel();
            aggregateBuild = null;
            throw new IllegalStateException("Failed to build Voxy's Vitrail meshes", e);
        }
    }

    private static List<Object> displayedSections(AggregatePass current, boolean opaque) {
        if (aggregateBuild != null) {
            // Every streaming region owns whole level-4 hierarchy roots.  It is therefore safe to
            // replace a finished region while retaining the old cut in unfinished regions: a
            // parent and all of its descendants can never be split across the boundary.  This is
            // the CPU/provider equivalent of Voxy retaining a parent until its child cut is ready.
            try {
                List<Object> preview = aggregateBuild.previewSections(opaque, current);
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
        private final ArrayList<PersistentGeometryCache.Lease> geometryLeases = new ArrayList<>();
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
        private int cacheHits;
        private int cacheMisses;
        private boolean transferred;
        private final long rendererEpoch;

        private CombinedAggregateBuild(VoxyRenderSystem renderer,
                com.mojang.blaze3d.systems.GpuDevice device,
                List<me.cortex.voxy.client.core.rendering.hierachical.NodeManager.GeometryNode> visible,
                List<NodeStamp> stamps, double cameraX, double cameraZ) {
            this.renderer = renderer;
            this.device = device;
            this.stamps = stamps;
            this.rendererEpoch = renderer.getVitrailHierarchyEpoch();
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
                    PersistentGeometryCache.Lease lease = geometryCache.acquire(
                            new PersistentGeometryCache.Key(this.rendererEpoch,
                                    node.position(), node.geometryId(), node.geometryVersion()),
                            () -> encodeNode(node));
                    if (lease != null) {
                        this.geometryLeases.add(lease);
                        if (lease.hit()) this.cacheHits++; else this.cacheMisses++;
                        List<VitrailCpuMeshEncoder.Mesh> meshes = lease.meshes();
                        this.sourceMeshes.addAll(meshes);
                        for (var mesh : meshes) {
                            GroupKey key = new GroupKey(Math.floorDiv(mesh.x(), AGGREGATE_CELL_BLOCKS),
                                    Math.floorDiv(mesh.y(), AGGREGATE_CELL_BLOCKS),
                                    Math.floorDiv(mesh.z(), AGGREGATE_CELL_BLOCKS), mesh.atlasPage());
                            Map<GroupKey, List<VitrailCpuMeshEncoder.Mesh>> groups = mesh.opaque()
                                    ? this.opaqueGroups : this.translucentGroups;
                            groups.computeIfAbsent(key, ignored -> new ArrayList<>()).add(mesh);
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

                this.sourceMeshes.clear();
                this.opaqueGroups.clear();
                this.translucentGroups.clear();
                this.geometryLeases.forEach(PersistentGeometryCache.Lease::close);
                this.geometryLeases.clear();
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

        private List<Object> previewSections(boolean opaque, AggregatePass previous)
                throws ReflectiveOperationException {
            List<UploadedPiece> pieces = opaque ? this.opaqueUploaded : this.translucentUploaded;
            int committed = opaque ? this.committedOpaquePieces : this.committedTranslucentPieces;
            if (this.regionIndex == 0) return previous == null ? List.of() : previous.sections;

            ArrayList<UploadedPiece> displayed = new ArrayList<>(committed
                    + (previous == null ? 0 : previous.pieces.size()));
            displayed.addAll(pieces.subList(0, committed));
            if (previous != null) {
                HashSet<RegionKey> replaced = new HashSet<>();
                for (int index = 0; index < this.regionIndex; index++) {
                    replaced.add(this.regions.get(index).key);
                }
                for (UploadedPiece old : previous.pieces) {
                    if (!replaced.contains(old.region)) displayed.add(old);
                }
            }
            return makeSections(displayed, displayed.size());
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
                    ? parts.getFirst() : VitrailCpuMeshEncoder.combineRebased(parts);
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
                uploaded.add(new UploadedPiece(new TileKey(mesh.x(), mesh.y(), mesh.z()),
                        this.regions.get(this.regionIndex).key,
                        apiPiece, vertex, index, detail));
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
            this.sourceMeshes.clear();
            this.geometryLeases.forEach(PersistentGeometryCache.Lease::close);
            this.geometryLeases.clear();
            releasePieces(this.opaqueUploaded);
            releasePieces(this.translucentUploaded);
            this.opaqueUploaded.clear();
            this.translucentUploaded.clear();
        }

        private List<VitrailCpuMeshEncoder.Mesh> encodeNode(
                me.cortex.voxy.client.core.rendering.hierachical.NodeManager.GeometryNode node) {
            var section = this.renderer.getVitrailCpuSectionSnapshot(node.geometryId());
            if (section == null) return null;
            try {
                if (section.position != node.position()) return null;
                return VitrailCpuMeshEncoder.encode(section,
                        this.renderer::getVitrailModelFaceData, this.renderer::getVitrailFaceColour,
                        this.renderer::getVitrailModelMaterial, this.renderer::getVitrailNearSuppression,
                        this.renderer::getVitrailFaceAverage,
                        null);
            } finally {
                section.free();
            }
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
                int level = WorldEngine.getLevel(position);
                int rootShift = WorldEngine.MAX_LOD_LAYER - level;
                int rootX = WorldEngine.getX(position) >> rootShift;
                int rootZ = WorldEngine.getZ(position) >> rootShift;
                int rootsPerRegion = STREAM_REGION_BLOCKS
                        / (32 << WorldEngine.MAX_LOD_LAYER);
                RegionKey key = new RegionKey(Math.floorDiv(rootX, rootsPerRegion),
                        Math.floorDiv(rootZ, rootsPerRegion));
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
            double minX = (double) region.x * STREAM_REGION_BLOCKS;
            double minZ = (double) region.z * STREAM_REGION_BLOCKS;
            double maxX = minX + STREAM_REGION_BLOCKS;
            double maxZ = minZ + STREAM_REGION_BLOCKS;
            // Distance to the region's nearest edge, not its centre. This gives a true expanding
            // ring around the player even while they stand close to a 256-block region boundary.
            double dx = cameraX < minX ? minX - cameraX : cameraX > maxX ? cameraX - maxX : 0.0;
            double dz = cameraZ < minZ ? minZ - cameraZ : cameraZ > maxZ ? cameraZ - maxZ : 0.0;
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
    private record GroupKey(int cellX, int cellY, int cellZ, int atlasPage) {}
    private record UploadTask(GroupKey key, List<VitrailCpuMeshEncoder.Mesh> parts) {}
    private record RegionKey(int x, int z) {}
    private record RegionBatch(RegionKey key,
            List<me.cortex.voxy.client.core.rendering.hierachical.NodeManager.GeometryNode> nodes) {}
    private record NodeStamp(long position, int geometryId, long version) {}
    private record UploadedPiece(TileKey tile, RegionKey region, Object apiPiece,
            GpuBuffer vertex, GpuBuffer index, GpuBuffer detail) {}
    private record AggregatePass(List<NodeStamp> stamps, List<Object> sections, List<UploadedPiece> pieces) {}
    private record AggregatePair(AggregatePass opaque, AggregatePass translucent) {}
    private record AtlasPage(GpuTexture texture, GpuTextureView view, Set<Integer> uploaded) {}

    private static GpuTextureView atlasFor(VoxyRenderSystem renderer, com.mojang.blaze3d.systems.GpuDevice device,
            VitrailCpuMeshEncoder.Mesh mesh) {
        AtlasPage page = atlasPages.computeIfAbsent(mesh.atlasPage(), ignored -> {
            GpuTexture texture = device.createTexture(() -> "Voxy baked LOD atlas", GpuTexture.USAGE_TEXTURE_BINDING | GpuTexture.USAGE_COPY_DST,
                    GpuFormat.RGBA8_UNORM, ATLAS_WIDTH, ATLAS_HEIGHT, 1, 1);
            try { return new AtlasPage(texture, device.createTextureView(texture), new HashSet<>()); }
            catch (RuntimeException | Error e) { texture.close(); throw e; }
        });
        var encoder = device.createCommandEncoder();
        ByteBuffer pixels = MemoryUtil.memAlloc(16 * 16 * 4);
        try {
            for (int i = 0; i < mesh.vertexCount(); i += 4) {
                int tile = MemoryUtil.memGetInt(mesh.detail().address + (long)i * 16 + 8);
                if (page.uploaded.contains(tile)) continue;
                int[] source = renderer.getVitrailFacePixels(
                        mesh.atlasPage() * VitrailCpuMeshEncoder.MODELS_PER_ATLAS_PAGE + tile / 6,
                        tile % 6);
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

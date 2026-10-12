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
    private static volatile Constructor<?> boundsConstructor;
    private static final org.joml.Matrix4f schedulingViewProjection = new org.joml.Matrix4f();
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
    private static final LoadPacing loadPacing = new LoadPacing();
    private static VoxyRenderSystem selectionCacheRenderer;
    private static long selectionCacheSerial = -1;
    private static boolean selectionCacheBaseline;
    private static List<me.cortex.voxy.client.core.rendering.hierachical.NodeManager.GeometryNode> selectionCacheNodes = List.of();
    private static List<NodeStamp> selectionCacheStamps = List.of();
    // Voxy's native indirect renderer submits thousands of nodes in very few GPU commands. The
    // provider path has ordinary indexed draws, so aggregate more source nodes per piece and use
    // a larger streaming region to keep draw-call count close to the native design.
    private static final int MAX_SOURCE_MESHES_PER_UPLOAD = 192;
    // Combining and GPU buffer creation cannot yield halfway through a task.
    private static final long MAX_UPLOAD_BYTES = 2L * 1024L * 1024L;
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

    // Render-thread-only LRU. Each entry owns one reference; queued cache hits own
    // another so eviction cannot retire a buffer before it reaches the active cut.
    private static final long GPU_REUSE_BYTES = 256L * 1024L * 1024L;
    private static final LinkedHashMap<GpuMeshKey, CachedGpuPiece> gpuMeshCache = new LinkedHashMap<>(128, 0.75f, true);
    private static long gpuMeshCacheBytes;
    private static final java.util.ArrayDeque<UploadedPiece> retiredPieces = new java.util.ArrayDeque<>();

    // Only immutable snapshots and leased CPU meshes enter this worker; GPU calls stay here.
    private static final java.util.concurrent.ExecutorService preparationWorker =
            java.util.concurrent.Executors.newSingleThreadExecutor(task -> {
                Thread thread = new Thread(task, "Voxy Vitrail mesh preparation");
                thread.setDaemon(true);
                return thread;
            });

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
                    .getConstructor(GpuBuffer.class, GpuBuffer.class, int.class, GpuBuffer.class, GpuTextureView.class,
                            Class.forName(SECTION + "$Bounds", true, loader));
            boundsConstructor = Class.forName(SECTION + "$Bounds", true, loader)
                    .getConstructor(float.class, float.class, float.class, float.class, float.class, float.class);
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
            schedulingViewProjection.set(projection).mul(view);
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
        frontierAggregate = null;
        frontierBuild = null;
        frontierRegion = -1;
        presentationFrontier = Set.of();
        pendingSinceNanos = 0;
        pendingFirstDirtyNanos = 0;
        pendingSettleNanos = AGGREGATE_SETTLE_NANOS;
        lastCameraX = Double.NaN;
        lastCameraZ = Double.NaN;
        lastProjectionScalePixels = Double.NaN;
        loadPacing.reset();
        providerSamples = providerNanos = providerPeak = providerOver8ms = providerReport = 0;
        for (var page : atlasPages.values()) { page.view.close(); page.texture.close(); }
        atlasPages.clear();
        for (CachedGpuPiece cached : gpuMeshCache.values()) releasePieces(List.of(cached.piece));
        gpuMeshCache.clear();
        gpuMeshCacheBytes = 0;
        drainRetiredPieces(Long.MAX_VALUE);
        geometryCache.close();
        selectionCacheRenderer = null;
        selectionCacheSerial = -1;
        selectionCacheNodes = List.of();
        selectionCacheStamps = List.of();
    }

    private static long providerSamples, providerNanos, providerPeak, providerOver8ms, providerReport;
    private static synchronized List<Object> buildSections(VoxyRenderSystem renderer, boolean opaque) {
        long start = System.nanoTime();
        try { return buildSectionsInternal(renderer, opaque); }
        finally {
            if (opaque) {
                long elapsed = System.nanoTime() - start;
                providerSamples++; providerNanos += elapsed;
                providerPeak = Math.max(providerPeak, elapsed);
                if (elapsed > 8_000_000L) providerOver8ms++;
                if (providerReport == 0) providerReport = start + 5_000_000_000L;
                if (start >= providerReport) {
                    Logger.info("Voxy provider CPU: " + providerSamples + " frames, avg "
                            + providerNanos / providerSamples / 1000L + " us, peak " + providerPeak / 1000L
                            + " us, over8ms " + providerOver8ms + ", streaming " + (aggregateBuild != null));
                    providerSamples = providerNanos = providerPeak = providerOver8ms = 0;
                    providerReport = start + 5_000_000_000L;
                }
            }
        }
    }

    private static List<Object> buildSectionsInternal(VoxyRenderSystem renderer, boolean opaque) {
        var device = RenderSystem.tryGetDevice();
        if (device == null || sectionConstructor == null || pieceConstructor == null) return List.of();
        AggregatePass current = opaque ? opaqueAggregate : translucentAggregate;
        if (opaque) {
            loadPacing.frame(System.nanoTime());
            drainRetiredPieces(250_000L);
            refreshPendingAtlas(renderer, device);
        }

        // The opaque pass owns streaming progress so both render layers publish the same regions.
        if (!opaque) return displayedSections(current, false);

        double cameraX = renderer.getVitrailCameraX();
        double cameraZ = renderer.getVitrailCameraZ();
        boolean hasCompleteBaseline = opaqueAggregate != null && !opaqueAggregate.sections.isEmpty();
        long selectionSerial = renderer.getVitrailSelectionSerial();
        if (selectionCacheRenderer != renderer || selectionCacheSerial != selectionSerial
                || selectionCacheBaseline != hasCompleteBaseline) {
            selectionCacheRenderer = renderer;
            selectionCacheSerial = selectionSerial;
            selectionCacheBaseline = hasCompleteBaseline;
            selectionCacheNodes = renderer.getVitrailVisibleNodes(cameraX, cameraZ);
            selectionCacheStamps = selectionCacheNodes.stream()
                .map(node -> new NodeStamp(node.position(), node.geometryId(), node.geometryVersion()))
                .sorted(NODE_STAMP_ORDER)
                .toList();
        }
        var visible = selectionCacheNodes;
        List<NodeStamp> stamps = selectionCacheStamps;
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
        double previousProjectionScalePixels = lastProjectionScalePixels;
        lastProjectionScalePixels = projectionScalePixels;
        if (projectionChanged) loadPacing.responseUntil = now + 750_000_000L;

        // Finish published regions even during smooth zoom; cancellation rolls them back.

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
            } else if (current != null) {
                // A pending update must keep waiting even on frames with identical input.
                // Otherwise the next render frame bypasses the settle interval entirely;
                // selections refresh less frequently than drawing and cause rebuild bursts.
                boolean cutChanged = !pendingStamps.equals(stamps);
                pendingStamps = stamps;
                if (pendingFirstDirtyNanos == 0L) pendingFirstDirtyNanos = now;
                if (pendingSinceNanos == 0L || cameraMoved || cutChanged) pendingSinceNanos = now;
                if (projectionChanged) pendingSettleNanos = INITIAL_STREAM_DELAY_NANOS;
                // A released zoom no longer needs its denser cut. Start the replacement promptly.
                if (projectionChanged && projectionScalePixels < previousProjectionScalePixels) pendingSinceNanos = now - pendingSettleNanos;
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
            long frameBudget = loadPacing.budget(!hasCompleteBaseline || now < loadPacing.responseUntil);
            long workStarted = System.nanoTime();
            AggregatePair rebuilt;
            try { rebuilt = build.advance(frameBudget); }
            finally { loadPacing.workNanos = System.nanoTime() - workStarted; }
            build.minBudgetNanos = Math.min(build.minBudgetNanos, frameBudget);
            build.maxBudgetNanos = Math.max(build.maxBudgetNanos, frameBudget);
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
                    + " opaque / " + rebuilt.translucent.pieces.size() + " translucent meshes, "
                    + build.reusedRegions + " unchanged regions reused, over "
                    + build.frames + " frames / " + rebuildMillis + " ms; exact hierarchy cut, effective subdivision "
                    + Math.round(renderer.getVitrailEffectiveSubdivisionPixels() * 10.0) / 10.0 + " px; geometry cache "
                    + build.cacheHits + " hit / " + build.cacheMisses + " miss, " + cacheStats.entries()
                    + " entries / " + (cacheStats.residentBytes() / (1024L * 1024L)) + " MiB; selected levels " + java.util.Arrays.toString(build.selectedLevels)
                    + "; GPU batch reuse " + build.gpuReuseHits + " hit (" + build.activeGpuReuseHits + " active) / " + (gpuMeshCacheBytes >> 20) + " MiB; adaptive budget " + (build.minBudgetNanos / 1000L) + "-" + (build.maxBudgetNanos / 1000L) + " us; async waits " + build.workerWaitFrames + "; peak encode/upload " + (build.maxEncodeNanos / 1000L) + "/" + (build.maxUploadNanos / 1000L) + " us; worker encode/prepare " + (build.maxWorkerEncodeNanos / 1000L) + "/" + (build.maxPrepareNanos / 1000L) + " us");
            return rebuilt.opaque.sections;
        } catch (ReflectiveOperationException | RuntimeException | Error e) {
            if (aggregateBuild != null) aggregateBuild.cancel();
            aggregateBuild = null;
            throw new IllegalStateException("Failed to build Voxy's Vitrail meshes", e);
        }
    }

    private static AggregatePass frontierAggregate;
    private static CombinedAggregateBuild frontierBuild;
    private static int frontierRegion = -1;
    private static Set<Long> presentationFrontier = Set.of();

    /** Published parents and their ancestors can refine by one layer on the next build. */
    public static Set<Long> getPresentationRefinementFrontier() {
        int completed = aggregateBuild == null ? 0 : aggregateBuild.regionIndex;
        if (frontierAggregate == opaqueAggregate && frontierBuild == aggregateBuild
                && frontierRegion == completed) return presentationFrontier;
        Set<RegionKey> replaced = new HashSet<>();
        Set<Long> positions = new HashSet<>();
        if (aggregateBuild != null) {
            for (int i = 0; i < completed; i++) {
                RegionKey key = aggregateBuild.regions.get(i).key;
                replaced.add(key);
                for (NodeStamp stamp : aggregateBuild.regionStamps.get(key)) addPresentationAncestors(positions, stamp.position());
            }
        }
        if (opaqueAggregate != null) {
            for (var entry : opaqueAggregate.regionStamps.entrySet()) {
                if (!replaced.contains(entry.getKey())) {
                    for (NodeStamp stamp : entry.getValue()) addPresentationAncestors(positions, stamp.position());
                }
            }
        }
        frontierAggregate = opaqueAggregate;
        frontierBuild = aggregateBuild;
        frontierRegion = completed;
        presentationFrontier = Set.copyOf(positions);
        return presentationFrontier;
    }

    private static void addPresentationAncestors(Set<Long> positions, long position) {
        positions.add(position);
        for (int level = WorldEngine.getLevel(position); level < WorldEngine.MAX_LOD_LAYER; level++) {
            position = WorldEngine.getWorldSectionId(level + 1, WorldEngine.getX(position) >> 1,
                    WorldEngine.getY(position) >> 1, WorldEngine.getZ(position) >> 1);
            if (!positions.add(position)) break;
        }
    }

    public static boolean hasCompleteBaseline() {
        return opaqueAggregate != null && !opaqueAggregate.sections.isEmpty();
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

    private static final class LoadPacing {
        private long previousFrame, workNanos, responseUntil, bootstrapUntil;
        private double baseFrameNanos = 8_333_333.0, recentFrameNanos = 8_333_333.0;
        private void reset() {
            previousFrame = workNanos = responseUntil = bootstrapUntil = 0;
            baseFrameNanos = recentFrameNanos = 8_333_333.0;
        }
        private void frame(long now) {
            if (previousFrame == 0) bootstrapUntil = now + 20_000_000_000L;
            if (previousFrame != 0) {
                long elapsed = now - previousFrame;
                // Ignore menus/world transitions, but react immediately to ordinary spikes.
                if (elapsed > 500_000L && elapsed < 250_000_000L) {
                    recentFrameNanos = elapsed;
                    double base = Math.max(500_000.0, elapsed - workNanos);
                    baseFrameNanos += (base - baseFrameNanos) * (base > baseFrameNanos ? 0.35 : 0.08);
                }
            }
            previousFrame = now;
            workNanos = 0;
        }
        private long budget(boolean responsive) {
            responsive |= System.nanoTime() < bootstrapUntil;
            int limit = net.minecraft.client.Minecraft.getInstance().options.framerateLimit().get();
            int target = limit > 0 && limit < 120 ? limit : 120;
            double frameTarget = 1_000_000_000.0 / Math.max(30, target);
            // A 120 FPS target must not starve initial coverage on a slower shader frame.
            // Allow bounded progress, while backing off again after a genuine long frame.
            long floor = responsive && recentFrameNanos < 20_000_000.0 ? 650_000L : 150_000L;
            if (recentFrameNanos > frameTarget * 1.20) return floor;
            double spare = frameTarget * 0.94 - baseFrameNanos;
            return (long)Math.max(Math.max(200_000L, floor), Math.min(2_000_000.0, spare * 0.65));
        }
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
        private final double projectionScalePixels;
        private List<UploadTask> uploadTasks;
        private int regionIndex;
        private int nodeIndex;
        private int uploadIndex;
        private int committedOpaquePieces;
        private int committedTranslucentPieces;
        private int frames;
        private int cacheHits;
        private int cacheMisses;
        private int gpuReuseHits;
        private int activeGpuReuseHits;
        private final Map<GpuMeshKey, UploadedPiece> activeGpuPieces = new HashMap<>();
        private long maxEncodeNanos, maxUploadNanos;
        private long minBudgetNanos = Long.MAX_VALUE, maxBudgetNanos;
        private int workerWaitFrames;
        private double uploadNanosPerByte = 0.25;
        private GpuBuffer uploadVertex, uploadIndices, uploadDetail;
        private GpuTextureView uploadAtlas;
        private int uploadStage;
        private volatile long maxWorkerEncodeNanos, maxPrepareNanos;
        private final Map<NodeStamp, java.util.concurrent.CompletableFuture<List<VitrailCpuMeshEncoder.Mesh>>> encodings = new LinkedHashMap<>();
        private final java.util.ArrayDeque<java.util.concurrent.CompletableFuture<PreparedUpload>> preparedQueue = new java.util.ArrayDeque<>();
        private int prepareIndex;
        private final int[] selectedLevels = new int[WorldEngine.MAX_LOD_LAYER + 1];
        private boolean transferred;
        private final long rendererEpoch;
        private final Map<RegionKey, List<NodeStamp>> regionStamps = new HashMap<>();
        private final AggregatePass previousOpaque = opaqueAggregate;
        private final AggregatePass previousTranslucent = translucentAggregate;
        private final Map<RegionKey, List<UploadedPiece>> previousOpaqueByRegion = indexPieces(previousOpaque);
        private final Map<RegionKey, List<UploadedPiece>> previousTranslucentByRegion = indexPieces(previousTranslucent);
        private int reusedRegions;
        private int opaquePreviewRegion = -1, translucentPreviewRegion = -1;
        private List<Object> opaquePreview = List.of(), translucentPreview = List.of();

        private CombinedAggregateBuild(VoxyRenderSystem renderer,
                com.mojang.blaze3d.systems.GpuDevice device,
                List<me.cortex.voxy.client.core.rendering.hierachical.NodeManager.GeometryNode> visible,
                List<NodeStamp> stamps, double cameraX, double cameraZ) {
            this.renderer = renderer;
            // The published aggregate owns these independently of the history LRU.
            // Index active batches even when the 256 MiB history has evicted them.
            if (this.previousOpaque != null) for (UploadedPiece piece : this.previousOpaque.pieces) {
                if (piece.key != null) this.activeGpuPieces.put(piece.key, piece);
            }
            if (this.previousTranslucent != null) for (UploadedPiece piece : this.previousTranslucent.pieces) {
                if (piece.key != null) this.activeGpuPieces.put(piece.key, piece);
            }
            this.projectionScalePixels = renderer.getVitrailProjectionScalePixels();
            this.device = device;
            this.stamps = stamps;
            for (var node : visible) this.selectedLevels[WorldEngine.getLevel(node.position())]++;
            this.rendererEpoch = renderer.getVitrailHierarchyEpoch();
            this.regions = createRegions(visible, cameraX, cameraZ);
            for (RegionBatch region : this.regions) {
                this.regionStamps.put(region.key, region.nodes.stream()
                        .map(node -> new NodeStamp(node.position(), node.geometryId(), node.geometryVersion()))
                        .sorted(NODE_STAMP_ORDER).toList());
            }
        }

        private AggregatePair advance(long budgetNanos) throws ReflectiveOperationException {
            this.frames++;
            long deadline = System.nanoTime() + budgetNanos;
            boolean didWork = false;

            while (this.regionIndex < this.regions.size()) {
                RegionBatch region = this.regions.get(this.regionIndex);
                if (this.nodeIndex == 0 && this.uploadTasks == null && canReuse(region.key)) {
                    reusePieces(this.previousOpaqueByRegion, region.key, this.opaqueUploaded);
                    reusePieces(this.previousTranslucentByRegion, region.key, this.translucentUploaded);
                    this.reusedRegions++;
                    this.regionIndex++;
                    this.committedOpaquePieces = this.opaqueUploaded.size();
                    this.committedTranslucentPieces = this.translucentUploaded.size();
                    didWork = true;
                    if (System.nanoTime() >= deadline) return null;
                    continue;
                }
                // Cross-region look-ahead avoids a frame round trip for every small region.
                // Skip reused regions and cap snapshots at eight; GPU work still obeys pacing.
                // Once uploads are pending, spend this frame on them first. Repeated look-ahead
                // must never mark idle polling as work and starve a ready upload indefinitely.
                if (this.uploadTasks == null) {
                for (int aheadRegion = this.regionIndex; aheadRegion < Math.min(this.regions.size(), this.regionIndex + 4)
                        && this.encodings.size() < 8 && System.nanoTime() < deadline; aheadRegion++) {
                    var aheadBatch = this.regions.get(aheadRegion);
                    if (canReuse(aheadBatch.key)) continue;
                    int from = aheadRegion == this.regionIndex ? this.nodeIndex : 0;
                    for (int ahead = from; ahead < aheadBatch.nodes.size() && this.encodings.size() < 8
                            && System.nanoTime() < deadline; ahead++) {
                        var candidate = aheadBatch.nodes.get(ahead);
                        var key = new PersistentGeometryCache.Key(this.rendererEpoch, candidate.position(),
                                candidate.geometryId(), candidate.geometryVersion());
                        NodeStamp candidateStamp = new NodeStamp(candidate.position(), candidate.geometryId(), candidate.geometryVersion());
                        if (!geometryCache.contains(key) && !this.encodings.containsKey(candidateStamp)) {
                            long captureStarted = System.nanoTime();
                            encodeNode(candidate, true);
                            // An unavailable snapshot queues nothing and consumes no progress.
                            didWork |= this.encodings.containsKey(candidateStamp);
                            this.maxEncodeNanos = Math.max(this.maxEncodeNanos, System.nanoTime() - captureStarted);
                        }
                    }
                }
                }
                while (this.nodeIndex < region.nodes.size() && (!didWork || System.nanoTime() < deadline)) {
                    var node = region.nodes.get(this.nodeIndex++);
                    long encodeStarted = System.nanoTime();
                    PersistentGeometryCache.Lease lease = geometryCache.acquire(
                            new PersistentGeometryCache.Key(this.rendererEpoch,
                                    node.position(), node.geometryId(), node.geometryVersion()),
                            () -> encodeNode(node));
                    this.maxEncodeNanos = Math.max(this.maxEncodeNanos, System.nanoTime() - encodeStarted);
                    if (lease == null && this.encodings.containsKey(new NodeStamp(node.position(), node.geometryId(), node.geometryVersion()))) {
                        // Do not publish an incomplete region while its node is still being encoded.
                        this.nodeIndex--;
                        this.workerWaitFrames++;
                        return null;
                    }
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
                    queuePreparations();
                    var future = this.preparedQueue.peekFirst();
                    if (!future.isDone()) { this.workerWaitFrames++; return null; }
                    PreparedUpload prepared = future.join();
                    long bytes = prepared.cached != null ? 0 : switch (this.uploadStage) {
                        case 0 -> prepared.mesh.vertices().size;
                        case 1 -> prepared.mesh.indices().size;
                        case 2 -> prepared.mesh.detail().size;
                        default -> 0;
                    };
                    long estimate = prepared.cached == null && this.uploadStage < 3 ? (long)Math.max(20_000.0, bytes * this.uploadNanosPerByte) : 5_000L;
                    if (didWork && deadline - System.nanoTime() < estimate) return null;
                    // With pending uploads, the next advance skips prefetch and starts with
                    // didWork=false, guaranteeing forward progress for one indivisible task.
                    long uploadStarted = System.nanoTime();
                    if (uploadStep(prepared)) {
                        this.preparedQueue.removeFirst();
                        this.uploadIndex++;
                    }
                    long elapsed = System.nanoTime() - uploadStarted;
                    this.maxUploadNanos = Math.max(this.maxUploadNanos, elapsed);
                    if (bytes > 0) this.uploadNanosPerByte += (elapsed / (double)bytes - this.uploadNanosPerByte) * 0.20;
                    didWork = true;
                }
                if (this.uploadIndex < this.uploadTasks.size()) return null;

                this.sourceMeshes.clear();
                this.opaqueGroups.clear();
                this.translucentGroups.clear();
                geometryCache.releaseLeases(this.geometryLeases);
                this.geometryLeases.clear();
                this.uploadTasks = null;
                this.uploadIndex = 0;
                this.prepareIndex = 0;
                this.nodeIndex = 0;
                this.regionIndex++;
                this.committedOpaquePieces = this.opaqueUploaded.size();
                this.committedTranslucentPieces = this.translucentUploaded.size();
                if (System.nanoTime() >= deadline) return null;
            }

            AggregatePass opaque = new AggregatePass(this.stamps,
                    makeSections(this.opaqueUploaded, this.opaqueUploaded.size()), List.copyOf(this.opaqueUploaded), Map.copyOf(this.regionStamps), this.rendererEpoch);
            AggregatePass translucent = new AggregatePass(this.stamps,
                    makeSections(this.translucentUploaded, this.translucentUploaded.size()),
                    List.copyOf(this.translucentUploaded), Map.copyOf(this.regionStamps), this.rendererEpoch);
            this.opaqueUploaded.clear();
            this.translucentUploaded.clear();
            this.transferred = true;
            return new AggregatePair(opaque, translucent);
        }

        private void queuePreparations() {
            // One worker prepares several uploads between frames. Never wait on it while drawing.
            while (this.prepareIndex < this.uploadTasks.size() && this.preparedQueue.size() < 4) {
                UploadTask task = this.uploadTasks.get(this.prepareIndex++);
                GpuMeshKey key = new GpuMeshKey(this.rendererEpoch, this.regions.get(this.regionIndex).key,
                        task.key, List.copyOf(task.parts));
                UploadedPiece active = this.activeGpuPieces.get(key);
                CachedGpuPiece cached = active == null ? gpuMeshCache.get(key) : null;
                UploadedPiece reusable = active != null ? active : cached != null ? cached.piece : null;
                if (reusable != null) {
                    reusable.owners++;
                    this.gpuReuseHits++;
                    if (active != null) this.activeGpuReuseHits++;
                    this.preparedQueue.addLast(java.util.concurrent.CompletableFuture.completedFuture(
                            new PreparedUpload(null, false, null, null, key, reusable)));
                } else {
                    this.preparedQueue.addLast(java.util.concurrent.CompletableFuture.supplyAsync(
                            () -> prepare(task, key), preparationWorker));
                }
            }
        }

        private boolean canReuse(RegionKey key) {
            return this.previousOpaque != null && this.previousTranslucent != null
                    && this.previousOpaque.epoch == this.rendererEpoch
                    && this.previousTranslucent.epoch == this.rendererEpoch
                    && this.regionStamps.get(key).equals(this.previousOpaque.regionStamps.get(key))
                    && this.regionStamps.get(key).equals(this.previousTranslucent.regionStamps.get(key));
        }

        private static Map<RegionKey, List<UploadedPiece>> indexPieces(AggregatePass previous) {
            if (previous == null) return Map.of();
            Map<RegionKey, List<UploadedPiece>> result = new HashMap<>();
            for (UploadedPiece piece : previous.pieces) {
                result.computeIfAbsent(piece.region, ignored -> new ArrayList<>()).add(piece);
            }
            return result;
        }

        private static void reusePieces(Map<RegionKey, List<UploadedPiece>> previous,
                RegionKey key, List<UploadedPiece> target) {
            for (UploadedPiece piece : previous.getOrDefault(key, List.of())) {
                piece.owners++;
                target.add(piece);
            }
        }

        private List<Object> previewSections(boolean opaque, AggregatePass previous)
                throws ReflectiveOperationException {
            List<UploadedPiece> pieces = opaque ? this.opaqueUploaded : this.translucentUploaded;
            int committed = opaque ? this.committedOpaquePieces : this.committedTranslucentPieces;
            if (this.regionIndex == 0) return previous == null ? List.of() : previous.sections;

            if (opaque && this.opaquePreviewRegion == this.regionIndex) return this.opaquePreview;
            if (!opaque && this.translucentPreviewRegion == this.regionIndex) return this.translucentPreview;
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
            List<Object> result = makeSections(displayed, displayed.size());
            if (opaque) { this.opaquePreviewRegion = this.regionIndex; this.opaquePreview = result; }
            else { this.translucentPreviewRegion = this.regionIndex; this.translucentPreview = result; }
            return result;
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

        private PreparedUpload prepare(UploadTask task, GpuMeshKey key) {
            long started = System.nanoTime();
            List<VitrailCpuMeshEncoder.Mesh> parts = task.parts;
            boolean owned = parts.size() != 1;
            VitrailCpuMeshEncoder.Mesh mesh = owned ? VitrailCpuMeshEncoder.combineRebased(parts) : parts.getFirst();
            try {
                int minX=65535, minY=65535, minZ=65535, maxX=0, maxY=0, maxZ=0;
                java.util.BitSet tiles = new java.util.BitSet();
                for (int i=0; i<mesh.vertexCount(); i++) {
                    long ptr = mesh.vertices().address + (long)i * VitrailCpuMeshEncoder.VERTEX_STRIDE;
                    int x=Short.toUnsignedInt(MemoryUtil.memGetShort(ptr));
                    int y=Short.toUnsignedInt(MemoryUtil.memGetShort(ptr+2));
                    int z=Short.toUnsignedInt(MemoryUtil.memGetShort(ptr+4));
                    minX=Math.min(minX,x); minY=Math.min(minY,y); minZ=Math.min(minZ,z);
                    int detailWord = MemoryUtil.memGetInt(mesh.detail().address + (long)i * 16 + 8);
                    int fraction = detailWord >>> 14;
                    maxX=Math.max(maxX,x + ((fraction & 63) != 0 ? 1 : 0));
                    maxY=Math.max(maxY,y + (((fraction >>> 6) & 63) != 0 ? 1 : 0));
                    maxZ=Math.max(maxZ,z + (((fraction >>> 12) & 63) != 0 ? 1 : 0));
                    if ((i & 3)==0) tiles.set(detailWord & 16383);
                }
                this.maxPrepareNanos = Math.max(this.maxPrepareNanos, System.nanoTime()-started);
                return new PreparedUpload(mesh, owned, new int[]{minX,minY,minZ,maxX,maxY,maxZ}, tiles, key, null);
            } catch (RuntimeException | Error e) {
                if (owned) mesh.close();
                throw e;
            }
        }

        /** One indivisible GPU operation at a time; only publish a fully uploaded piece. */
        private boolean uploadStep(PreparedUpload prepared) throws ReflectiveOperationException {
            if (prepared.cached != null) {
                (prepared.key.parts.getFirst().opaque() ? this.opaqueUploaded : this.translucentUploaded).add(prepared.cached);
                return true;
            }
            VitrailCpuMeshEncoder.Mesh mesh = prepared.mesh;
            switch (this.uploadStage) {
                case 0 -> this.uploadVertex = this.device.createBuffer(() -> "Voxy Vitrail aggregate vertices", GpuBuffer.USAGE_VERTEX,
                        MemoryUtil.memByteBuffer(mesh.vertices().address, Math.toIntExact(mesh.vertices().size)));
                case 1 -> this.uploadIndices = this.device.createBuffer(() -> "Voxy Vitrail aggregate indices", GpuBuffer.USAGE_INDEX,
                        MemoryUtil.memByteBuffer(mesh.indices().address, Math.toIntExact(mesh.indices().size)));
                case 2 -> this.uploadDetail = this.device.createBuffer(() -> "Voxy aggregate LOD texture coordinates", GpuBuffer.USAGE_VERTEX,
                        MemoryUtil.memByteBuffer(mesh.detail().address, Math.toIntExact(mesh.detail().size)));
                case 3 -> this.uploadAtlas = atlasFor(this.renderer, this.device, mesh, prepared.tiles);
                default -> {
                    int[] b = prepared.bounds;
                    Object bounds = boundsConstructor.newInstance((float)b[0], (float)b[1], (float)b[2],
                            (float)b[3], (float)b[4], (float)b[5]);
                    Object apiPiece = pieceConstructor.newInstance(this.uploadVertex, this.uploadIndices,
                            mesh.indexCount(), this.uploadDetail, this.uploadAtlas, bounds);
                    UploadedPiece piece = new UploadedPiece(new TileKey(mesh.x(), mesh.y(), mesh.z()),
                            this.regions.get(this.regionIndex).key, apiPiece, this.uploadVertex, this.uploadIndices,
                            this.uploadDetail, prepared.key);
                    (mesh.opaque() ? this.opaqueUploaded : this.translucentUploaded).add(piece);
                    // Ownership is now with the aggregate; cancellation must not close it twice.
                    this.uploadVertex = null; this.uploadIndices = null; this.uploadDetail = null;
                    this.uploadAtlas = null; this.uploadStage = 0;
                    rememberGpuPiece(prepared.key, piece, mesh.vertices().size + mesh.indices().size + mesh.detail().size);
                    if (prepared.owned) mesh.close();
                    return true;
                }
            }
            this.uploadStage++;
            return false;
        }

        private void cancel() {
            if (this.transferred) return;
            if (this.uploadVertex != null) this.uploadVertex.close();
            if (this.uploadIndices != null) this.uploadIndices.close();
            if (this.uploadDetail != null) this.uploadDetail.close();
            this.uploadVertex = null; this.uploadIndices = null; this.uploadDetail = null;
            this.uploadAtlas = null; this.uploadStage = 0;
            // Joining occurs only at teardown/failure. Leases cannot be freed under a worker.
            for (var future : this.encodings.values()) {
                try { future.join().forEach(VitrailCpuMeshEncoder.Mesh::close); }
                catch (java.util.concurrent.CompletionException ignored) { }
            }
            this.encodings.clear();
            for (var future : this.preparedQueue) {
                try {
                    PreparedUpload prepared = future.join();
                    if (prepared.cached != null) releasePieces(List.of(prepared.cached));
                    else if (prepared.owned) prepared.mesh.close();
                }
                catch (java.util.concurrent.CompletionException ignored) { }
            }
            this.preparedQueue.clear();
            this.sourceMeshes.clear();
            geometryCache.releaseLeases(this.geometryLeases);
            this.geometryLeases.clear();
            releasePieces(this.opaqueUploaded);
            releasePieces(this.translucentUploaded);
            this.opaqueUploaded.clear();
            this.translucentUploaded.clear();
        }

        private List<VitrailCpuMeshEncoder.Mesh> encodeNode(
                me.cortex.voxy.client.core.rendering.hierachical.NodeManager.GeometryNode node) {
            return encodeNode(node, false);
        }

        private List<VitrailCpuMeshEncoder.Mesh> encodeNode(
                me.cortex.voxy.client.core.rendering.hierachical.NodeManager.GeometryNode node, boolean prefetch) {
            NodeStamp stamp=new NodeStamp(node.position(),node.geometryId(),node.geometryVersion());
            var future=this.encodings.get(stamp);
            if (future != null) {
                if (prefetch || !future.isDone()) return null;
                List<VitrailCpuMeshEncoder.Mesh> result=future.join();
                this.encodings.remove(stamp);
                return result;
            }
            var section = this.renderer.getVitrailCpuSectionSnapshot(node.geometryId(), node.geometryVersion());
            if (section == null) return null;
            try {
                if (section.position != node.position()) { section.free(); return null; }
                if (section.isEmpty() || section.geometryBuffer == null) { section.free(); return List.of(); }
                // Freeze all model and biome lookups on the render thread. Neither Minecraft
                // lighting nor model-service memory is touched by the background encoder.
                var models = new it.unimi.dsi.fastutil.ints.Int2ObjectOpenHashMap<int[]>();
                var tints = new it.unimi.dsi.fastutil.ints.Int2IntOpenHashMap();
                int total = section.geometryBuffer == null ? 0 : Math.toIntExact(section.geometryBuffer.size / 8L);
                if (section.offsets == null || section.offsets.length < 8) {
                    throw new IllegalArgumentException("Missing quad group offsets");
                }
                for (int group=0; group<8; group++) {
                    int from=section.offsets[group], to=group==7 ? total : section.offsets[group+1];
                    if (from<0 || to<from || to>total) throw new IllegalArgumentException("Invalid quad group offsets");
                    boolean opaque=group!=0;
                    for (int i=from; i<to; i++) {
                        long packed=MemoryUtil.memGetLong(section.geometryBuffer.address+(long)i*8);
                        int model=(int)((packed>>>26)&0xffff), biome=(int)((packed>>>46)&0x1ff), face=(int)packed&7;
                        if (face>=6) throw new IllegalArgumentException("Invalid quad face");
                        if (!models.containsKey(model)) {
                            int id=model;
                            int[] data=new int[14];
                            for(int f=0; f<6; f++) {
                                data[f]=this.renderer.getVitrailModelFaceData(id,f);
                                data[6+f]=this.renderer.getVitrailFaceAverage(id,f);
                            }
                            data[12]=this.renderer.getVitrailModelMaterial(id);
                            data[13]=this.renderer.getVitrailNearSuppression(id);
                            models.put(model,data);
                        }
                        int tintKey=(model<<13)|(biome<<4)|(face<<1)|(opaque?1:0);
                        if (!tints.containsKey(tintKey)) tints.put(tintKey, this.renderer.getVitrailFaceColour(model,biome,face,opaque));
                    }
                }
                this.encodings.put(stamp, java.util.concurrent.CompletableFuture.supplyAsync(() -> {
                    long started=System.nanoTime();
                    try {
                        return VitrailCpuMeshEncoder.encode(section,
                                (model,face)->models.get(model)[face],
                                (model,biome,face,opaque)->tints.get((model<<13)|(biome<<4)|(face<<1)|(opaque?1:0)),
                                model->models.get(model)[12], model->models.get(model)[13],
                                (model,face)->models.get(model)[6+face], null);
                    } finally {
                        section.free();
                        this.maxWorkerEncodeNanos=Math.max(this.maxWorkerEncodeNanos,System.nanoTime()-started);
                    }
                }, preparationWorker));
                return null;
            } catch (RuntimeException | Error e) {
                section.free();
                throw e;
            }
        }

        private static void addUploadTasks(List<UploadTask> tasks,
                Map<GroupKey, List<VitrailCpuMeshEncoder.Mesh>> groups) {
            for (var entry : groups.entrySet()) {
                List<VitrailCpuMeshEncoder.Mesh> parts = entry.getValue();
                for (int from = 0; from < parts.size();) {
                    int to = from;
                    long bytes = 0;
                    while (to < parts.size() && to - from < MAX_SOURCE_MESHES_PER_UPLOAD) {
                        var part = parts.get(to);
                        long partBytes = part.vertices().size + part.indices().size + part.detail().size;
                        // A single source mesh is indivisible; always allow forward progress.
                        if (to > from && bytes + partBytes > MAX_UPLOAD_BYTES) break;
                        bytes += partBytes;
                        to++;
                    }
                    tasks.add(new UploadTask(entry.getKey(), parts.subList(from, to)));
                    from = to;
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
            Map<RegionKey, Integer> priorities = new HashMap<>();
            for (RegionBatch region : result) priorities.put(region.key, regionPriority(region, cameraX, cameraZ));
            result.sort(Comparator
                    .comparingInt((RegionBatch region) -> priorities.get(region.key))
                    .thenComparingDouble(region -> regionDistanceSquared(region.key, cameraX, cameraZ))
                    .thenComparingInt(region -> region.key.x)
                    .thenComparingInt(region -> region.key.z));
            return List.copyOf(result);
        }

        private static int regionPriority(RegionBatch region, double cameraX, double cameraZ) {
            if (regionDistanceSquared(region.key, cameraX, cameraZ) < 128.0 * 128.0) return 0;
            double cameraY = net.minecraft.client.Minecraft.getInstance().gameRenderer.mainCamera().position().y;
            var frustum = new org.joml.FrustumIntersection(schedulingViewProjection);
            for (var node : region.nodes) {
                long position = node.position();
                int size = 32 << WorldEngine.getLevel(position);
                float x = (float)((double)WorldEngine.getX(position) * size - cameraX);
                float y = (float)((double)WorldEngine.getY(position) * size - cameraY);
                float z = (float)((double)WorldEngine.getZ(position) * size - cameraZ);
                // Scheduling must reach beyond vanilla's far plane. Only rank by the four
                // side planes here; this priority never removes geometry from the cache.
                if (frustum.intersectAab(x, y, z, x + size, y + size, z + size, 0x0f) < 0) return 1;
            }
            return 2;
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

    private static void rememberGpuPiece(GpuMeshKey key, UploadedPiece piece, long bytes) {
        if (bytes > GPU_REUSE_BYTES) return;
        piece.owners++;
        CachedGpuPiece old = gpuMeshCache.put(key, new CachedGpuPiece(piece, bytes));
        gpuMeshCacheBytes += bytes;
        if (old != null) { gpuMeshCacheBytes -= old.bytes; releasePieces(List.of(old.piece)); }
        while (gpuMeshCacheBytes > GPU_REUSE_BYTES || gpuMeshCache.size() > 1024) {
            var iterator = gpuMeshCache.entrySet().iterator();
            CachedGpuPiece evicted = iterator.next().getValue();
            iterator.remove();
            gpuMeshCacheBytes -= evicted.bytes;
            releasePieces(List.of(evicted.piece));
        }
    }

    private static void releaseAggregate(AggregatePass aggregate) {
        if (aggregate != null) releasePieces(aggregate.pieces);
    }

    private static void releasePieces(List<UploadedPiece> pieces) {
        for (UploadedPiece piece : pieces) {
            if (--piece.owners > 0) continue;
            retiredPieces.addLast(piece);
        }
    }

    private static void drainRetiredPieces(long budgetNanos) {
        long start = System.nanoTime();
        int count = 0;
        while (!retiredPieces.isEmpty()) {
            UploadedPiece piece = retiredPieces.removeFirst();
            try { piece.vertex.close(); } catch (RuntimeException e) { Logger.warn("Could not release Vitrail vertex buffer", e); }
            try { piece.index.close(); } catch (RuntimeException e) { Logger.warn("Could not release Vitrail index buffer", e); }
            try { piece.detail.close(); } catch (RuntimeException e) { Logger.warn("Could not release Vitrail detail buffer", e); }
            if (budgetNanos != Long.MAX_VALUE && (++count >= 8 || System.nanoTime() - start >= budgetNanos)) break;
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
    private record GpuMeshKey(long epoch, RegionKey region, GroupKey group, List<VitrailCpuMeshEncoder.Mesh> parts) {}
    private record CachedGpuPiece(UploadedPiece piece, long bytes) {}
    private record PreparedUpload(VitrailCpuMeshEncoder.Mesh mesh, boolean owned, int[] bounds, java.util.BitSet tiles,
            GpuMeshKey key, UploadedPiece cached) {}
    private record UploadTask(GroupKey key, List<VitrailCpuMeshEncoder.Mesh> parts) {}
    private record RegionKey(int x, int z) {}
    private record RegionBatch(RegionKey key,
            List<me.cortex.voxy.client.core.rendering.hierachical.NodeManager.GeometryNode> nodes) {}
    private record NodeStamp(long position, int geometryId, long version) {}
    private static final class UploadedPiece {
        private final GpuMeshKey key;
        private final TileKey tile;
        private final RegionKey region;
        private final Object apiPiece;
        private final GpuBuffer vertex, index, detail;
        private int owners = 1;
        private UploadedPiece(TileKey tile, RegionKey region, Object apiPiece,
                GpuBuffer vertex, GpuBuffer index, GpuBuffer detail, GpuMeshKey key) {
            this.key = key;
            this.tile = tile; this.region = region; this.apiPiece = apiPiece;
            this.vertex = vertex; this.index = index; this.detail = detail;
        }
    }
    private record AggregatePass(List<NodeStamp> stamps, List<Object> sections, List<UploadedPiece> pieces,
            Map<RegionKey, List<NodeStamp>> regionStamps, long epoch) {}
    private record AggregatePair(AggregatePass opaque, AggregatePass translucent) {}
    private record AtlasPage(GpuTexture texture, GpuTextureView view, java.util.BitSet uploaded, java.util.BitSet pending) {}

    private static GpuTextureView atlasFor(VoxyRenderSystem renderer, com.mojang.blaze3d.systems.GpuDevice device,
            VitrailCpuMeshEncoder.Mesh mesh, java.util.BitSet required) {
        AtlasPage page = atlasPages.computeIfAbsent(mesh.atlasPage(), ignored -> {
            GpuTexture texture = device.createTexture(() -> "Voxy baked LOD atlas", GpuTexture.USAGE_TEXTURE_BINDING | GpuTexture.USAGE_COPY_DST,
                    GpuFormat.RGBA8_UNORM, ATLAS_WIDTH, ATLAS_HEIGHT, 1, 1);
            try { return new AtlasPage(texture, device.createTextureView(texture), new java.util.BitSet(), new java.util.BitSet()); }
            catch (RuntimeException | Error e) { texture.close(); throw e; }
        });
        // Face tile collection was already performed by the preparation worker.
        required.andNot(page.uploaded);
        if (required.isEmpty()) return page.view;
        ByteBuffer pixels = MemoryUtil.memAlloc(16 * 16 * 4);
        try {
            var encoder = device.createCommandEncoder();
            for (int tile = required.nextSetBit(0); tile >= 0; tile = required.nextSetBit(tile + 1)) {
                int[] source = renderer.getVitrailFacePixels(
                        mesh.atlasPage() * VitrailCpuMeshEncoder.MODELS_PER_ATLAS_PAGE + tile / 6, tile % 6);
                fillTilePixels(pixels, source);
                encoder.writeToTexture(page.texture, pixels, 0, 0, (tile % 16) * 16, (tile / 16) * 16, 16, 16);
                page.uploaded.set(tile);
                if (source == null) page.pending.set(tile);
            }
        } finally { MemoryUtil.memFree(pixels); }
        return page.view;
    }

    private static void fillTilePixels(ByteBuffer pixels, int[] source) {
        pixels.clear();
        for (int p = 0; p < 256; p++) {
            int abgr = source == null ? -1 : source[p];
            pixels.put((byte)abgr).put((byte)(abgr >>> 8)).put((byte)(abgr >>> 16)).put((byte)(abgr >>> 24));
        }
        pixels.flip();
    }

    private static void refreshPendingAtlas(VoxyRenderSystem renderer, com.mojang.blaze3d.systems.GpuDevice device) {
        int refreshed = 0;
        long started = System.nanoTime();
        for (var entry : atlasPages.entrySet()) {
            AtlasPage page = entry.getValue();
            for (int tile = page.pending.nextSetBit(0); tile >= 0; tile = page.pending.nextSetBit(tile + 1)) {
                int[] source = renderer.getVitrailFacePixels(
                        entry.getKey() * VitrailCpuMeshEncoder.MODELS_PER_ATLAS_PAGE + tile / 6, tile % 6);
                if (source != null) {
                    ByteBuffer pixels = MemoryUtil.memAlloc(1024);
                    try {
                        fillTilePixels(pixels, source);
                        device.createCommandEncoder().writeToTexture(page.texture, pixels, 0, 0,
                                (tile % 16) * 16, (tile / 16) * 16, 16, 16);
                        page.pending.clear(tile);
                    } finally { MemoryUtil.memFree(pixels); }
                    if (++refreshed >= 8) return;
                }
                if (System.nanoTime() - started >= 250_000L) return;
            }
        }
    }
}

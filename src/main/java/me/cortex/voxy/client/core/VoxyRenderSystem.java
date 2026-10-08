package me.cortex.voxy.client.core;

import com.mojang.blaze3d.opengl.GlConst;
import com.mojang.blaze3d.opengl.GlStateManager;
import me.cortex.voxy.client.TimingStatistics;
import me.cortex.voxy.client.VoxyClient;
import me.cortex.voxy.client.config.VoxyConfig;
import me.cortex.voxy.client.core.gl.GlBuffer;
import me.cortex.voxy.client.core.gl.GlTexture;
import me.cortex.voxy.client.core.model.ModelBakerySubsystem;
import me.cortex.voxy.client.core.rendering.RenderDistanceTracker;
import me.cortex.voxy.client.core.rendering.Viewport;
import me.cortex.voxy.client.core.rendering.ViewportSelector;
import me.cortex.voxy.client.core.rendering.bounding.BoundRenderer;
import me.cortex.voxy.client.core.rendering.bounding.ColumnStreamedBoundStore;
import me.cortex.voxy.client.core.rendering.bounding.StreamedBoundStore;
import me.cortex.voxy.client.core.rendering.compat.ExactCpuTraversal;
import me.cortex.voxy.client.core.rendering.compat.HierarchyView;
import me.cortex.voxy.client.core.rendering.building.RenderGenerationService;
import me.cortex.voxy.client.core.rendering.hierachical.AsyncNodeManager;
import me.cortex.voxy.client.core.rendering.hierachical.HierarchicalOcclusionTraverser;
import me.cortex.voxy.client.core.rendering.hierachical.NodeCleaner;
import me.cortex.voxy.client.core.rendering.section.IUsesMeshlets;
import me.cortex.voxy.client.core.rendering.section.backend.AbstractSectionRenderer;
import me.cortex.voxy.client.core.rendering.section.backend.mdic.MDICSectionRenderer;
import me.cortex.voxy.client.core.rendering.section.geometry.BasicSectionGeometryData;
import me.cortex.voxy.client.core.rendering.section.geometry.IGeometryData;
import me.cortex.voxy.client.core.rendering.util.DownloadStream;
import me.cortex.voxy.client.core.rendering.util.PrintfDebugUtil;
import me.cortex.voxy.client.core.rendering.util.UploadStream;
import me.cortex.voxy.client.core.util.GPUTiming;
import me.cortex.voxy.client.core.util.IrisUtil;
import me.cortex.voxy.common.Logger;
import me.cortex.voxy.common.thread.ServiceManager;
import me.cortex.voxy.common.util.GlobalCleaner;
import me.cortex.voxy.common.world.WorldEngine;
import me.cortex.voxy.commonImpl.VoxyCommon;
import net.caffeinemc.mods.sodium.client.util.FogParameters;
import net.fabricmc.loader.api.FabricLoader;
import net.minecraft.client.Minecraft;
import net.minecraft.network.chat.Component;
import org.jetbrains.annotations.Nullable;
import org.joml.Matrix4f;
import org.joml.Matrix4fc;
import org.lwjgl.opengl.GL11;

import java.lang.ref.Cleaner;
import java.util.Arrays;
import java.util.List;

import static org.lwjgl.opengl.ARBDirectStateAccess.glGetTextureLevelParameteri;
import static org.lwjgl.opengl.GL11.glGetIntegerv;
import static org.lwjgl.opengl.GL11C.*;
import static org.lwjgl.opengl.GL30C.*;
import static org.lwjgl.opengl.GL33.glBindSampler;
import static org.lwjgl.opengl.GL43.GL_SHADER_STORAGE_BUFFER;
import static org.lwjgl.opengl.GL43C.GL_SHADER_STORAGE_BUFFER_BINDING;

public class VoxyRenderSystem {
    private final WorldEngine worldIn;

    private final ModelBakerySubsystem modelService;
    private final RenderGenerationService renderGen;
    private final IGeometryData geometryData;
    private final AsyncNodeManager nodeManager;
    private final NodeCleaner nodeCleaner;
    private final HierarchicalOcclusionTraverser traversal;

    private final Cleaner.Cleanable geoRef;

    private final RenderDistanceTracker renderDistanceTracker;
    private final BoundRenderer boundOutlineRenderer;
    public final StreamedBoundStore visbleSectionStream;
    private @Nullable ColumnStreamedBoundStore columnStreamedBoundStore;

    private final ViewportSelector<?> viewportSelector;

    private final AbstractRenderPipeline pipeline;
    private final RenderProperties properties;
    private volatile double vitrailCameraX;
    private volatile double vitrailCameraY;
    private volatile double vitrailCameraZ;
    private volatile double vitrailProjectionScalePixels;
    private volatile double vitrailEffectiveSubdivisionPixels;
    /**
     * Persistent compatibility-path screen-space threshold.  Recomputing this from Voxy's base
     * setting every frame made the selected cut alternate between an oversized fine cut and a
     * heavily coarsened cut.  Native Voxy does not have that feedback loop: its configured
     * threshold is stable and the GPU render queue changes only with the hierarchy/view.
     */
    private double vitrailAdaptiveSubdivisionPixels = Double.NaN;
    private volatile Matrix4f vitrailViewProjection;
    private volatile Matrix4f vitrailCoverageViewProjection;
    private Matrix4f vitrailWidestProjection;
    private double vitrailWidestProjectionScalePixels = Double.POSITIVE_INFINITY;
    private int vitrailProjectionWidth = -1;
    private int vitrailProjectionHeight = -1;
    private final ExactCpuTraversal vitrailShadowTraversal = new ExactCpuTraversal();
    private final ExactCpuTraversal vitrailCoarseTraversal = new ExactCpuTraversal();
    private volatile ExactTraversalSnapshot vitrailExactSnapshot = ExactTraversalSnapshot.EMPTY;
    private volatile ExactTraversalSnapshot vitrailCoarseSnapshot = ExactTraversalSnapshot.EMPTY;
    private volatile ExactCpuTraversal.Stats vitrailShadowStats = ExactCpuTraversal.Stats.EMPTY;
    // The compatibility path submits ordinary indexed draws instead of Voxy's native GPU driven
    // indirect stream.  Keep distant coverage complete, but spend fewer nodes on detail that is
    // already beyond vanilla's effective view distance.
    private static final int VITRAIL_VISIBLE_NODE_BUDGET = 3_600;
    /**
     * The provider combines thousands of hierarchy nodes into roughly one hundred Vulkan draws.
     * A brief view-dependent node spike therefore does not justify permanently doubling the
     * screen-space threshold.  The previous unrestricted feedback reached 536 px in the reported
     * scene and visibly replaced an already fine cut with parents after a camera turn.
     */
    private static final double VITRAIL_MAX_ADAPTIVE_SUBDIVISION_PIXELS = 384.0;
    private static final boolean VITRAIL_LEGACY_SELECTOR =
            Boolean.getBoolean("voxy.vitrail.legacySelector");

    private record ExactTraversalSnapshot(HierarchyView hierarchy, ExactCpuTraversal.Result result) {
        private static final ExactTraversalSnapshot EMPTY =
                new ExactTraversalSnapshot(HierarchyView.EMPTY, ExactCpuTraversal.Result.EMPTY);
    }

    private static AbstractSectionRenderer.Factory<?,? extends IGeometryData> getRenderBackendFactory() {
        return MDICSectionRenderer.FACTORY;
    }

    public record VitrailVisibleSection(
            me.cortex.voxy.client.core.rendering.hierachical.NodeManager.GeometryNode node,
            me.cortex.voxy.client.core.rendering.building.BuiltSection geometry) implements AutoCloseable {
        @Override
        public void close() {
            this.geometry.free();
        }
    }

    public VoxyRenderSystem(WorldEngine world, ServiceManager sm) {
        world.acquireRef();
        Logger.info("Creating Voxy render system");

        System.gc();

        if (Minecraft.getInstance().options.renderDistance().get()<3) {
            String msg = "Voxy: Having a vanilla render distance of 2 can cause rare culling near the edge of your screen issues, please use 3 or more";
            Logger.warn(msg);
            Minecraft.getInstance().gui.chatListener().handleSystemMessage(Component.literal(msg), false);
        }

        boolean isVitrail = me.cortex.voxy.client.core.RenderBackend.isVitrailVulkanActive();

        // === Vitrail (Vulkan) 模式：启动纯 CPU 后台多线程数据生成引擎，跳过 OpenGL 着色器管线 ===
        if (isVitrail) {
            this.worldIn = world;
            this.properties = RenderProperties.getRenderProperties();
            this.visbleSectionStream = new StreamedBoundStore();
            var backendFactory = getRenderBackendFactory();
            {
                this.modelService = new ModelBakerySubsystem(world.getMapper());
                this.renderGen = new RenderGenerationService(world, this.modelService, sm, IUsesMeshlets.class.isAssignableFrom(backendFactory.clz()));

                this.geometryData = new BasicSectionGeometryData(1<<20, RenderResourceReuse.getOrCreateGeometryBuffer());

                if (((BasicSectionGeometryData)this.geometryData).isExternalGeometryBuffer) {
                    var buffer = ((BasicSectionGeometryData)this.geometryData).getGeometryBuffer();
                    this.geoRef = GlobalCleaner.CLEANER.register(this.geometryData,() -> RenderResourceReuse.giveBackGeometryBuffer(buffer));
                } else {
                    this.geoRef = null;
                }

                this.nodeManager = new AsyncNodeManager(1 << 21, this.geometryData, this.renderGen);
                this.nodeCleaner = new NodeCleaner(this.nodeManager);

                world.setDirtyCallback(this.nodeManager::worldEvent);

                Arrays.stream(world.getMapper().getBiomeEntries()).forEach(this.modelService::addBiome);
                world.getMapper().setBiomeCallback(this.modelService::addBiome);

                this.nodeManager.start();
            }

            {
                int minSec = Minecraft.getInstance().level.getMinSectionY() >> 5;
                int maxSec = (Minecraft.getInstance().level.getMaxSectionY() - 1) >> 5;

                if (VoxyCommon.IS_MINE_IN_ABYSS) {
                    minSec = -8;
                    maxSec = 7;
                }

                this.renderDistanceTracker = new RenderDistanceTracker(40,
                        minSec,
                        maxSec,
                        this.nodeManager::addTopLevel,
                        this.nodeManager::removeTopLevel);

                this.setRenderDistance(VoxyConfig.CONFIG.sectionRenderDistance);
            }

            this.traversal = null;
            this.pipeline = null;
            this.viewportSelector = null;
            this.boundOutlineRenderer = null;
            VitrailBridge.registerProvider(this);
            Logger.info("Vitrail Vulkan detected: Voxy's CPU LOD provider is connected to Vitrail's distant-terrain pass.");
            return;
        }

        int[] oldBufferBindings = new int[10];
        for (int i = 0; i < oldBufferBindings.length; i++) {
            oldBufferBindings[i] = glGetIntegeri(GL_SHADER_STORAGE_BUFFER_BINDING, i);
        }

        try {
            glFinish();
            glFinish();

            this.worldIn = world;

            this.properties = RenderProperties.getRenderProperties();
            this.visbleSectionStream = new StreamedBoundStore();
            var backendFactory = getRenderBackendFactory();
            {
                this.modelService = new ModelBakerySubsystem(world.getMapper());
                this.renderGen = new RenderGenerationService(world, this.modelService, sm, IUsesMeshlets.class.isAssignableFrom(backendFactory.clz()));

                this.geometryData = new BasicSectionGeometryData(1<<20, RenderResourceReuse.getOrCreateGeometryBuffer());

                if (((BasicSectionGeometryData)this.geometryData).isExternalGeometryBuffer) {
                    var buffer = ((BasicSectionGeometryData)this.geometryData).getGeometryBuffer();
                    this.geoRef = GlobalCleaner.CLEANER.register(this.geometryData,() -> RenderResourceReuse.giveBackGeometryBuffer(buffer));
                } else {
                    this.geoRef = null;
                }

                this.nodeManager = new AsyncNodeManager(1 << 21, this.geometryData, this.renderGen);
                this.nodeCleaner = new NodeCleaner(this.nodeManager);
                this.traversal = new HierarchicalOcclusionTraverser(this.nodeManager, this.nodeCleaner, this.renderGen);

                world.setDirtyCallback(this.nodeManager::worldEvent);

                Arrays.stream(world.getMapper().getBiomeEntries()).forEach(this.modelService::addBiome);
                world.getMapper().setBiomeCallback(this.modelService::addBiome);

                this.nodeManager.start();
            }

            this.pipeline = RenderPipelineFactory.createPipeline(this.properties, this.nodeManager, this.nodeCleaner, this.traversal, this::frexStillHasWork);
            this.pipeline.setupExtraModelBakeryData(this.modelService);

            this.traversal.lateStageCompile(this.pipeline);

            var sectionRenderer = backendFactory.create(this.pipeline, this.modelService.getStore(), this.geometryData);
            this.pipeline.setSectionRenderer(sectionRenderer);
            this.viewportSelector = new ViewportSelector<>(sectionRenderer::createViewport);

            {
                int minSec = Minecraft.getInstance().level.getMinSectionY() >> 5;
                int maxSec = (Minecraft.getInstance().level.getMaxSectionY() - 1) >> 5;

                if (VoxyCommon.IS_MINE_IN_ABYSS) {
                    minSec = -8;
                    maxSec = 7;
                }

                this.renderDistanceTracker = new RenderDistanceTracker(40,
                        minSec,
                        maxSec,
                        this.nodeManager::addTopLevel,
                        this.nodeManager::removeTopLevel);

                this.setRenderDistance(VoxyConfig.CONFIG.sectionRenderDistance);
            }

            this.boundOutlineRenderer = new BoundRenderer(this.pipeline);

            Logger.info("Voxy render system created with " + this.geometryData.getMaxCapacity() + " geometry capacity, using pipeline '" + this.pipeline.getClass().getSimpleName() + "' with renderer '" + sectionRenderer.getClass().getSimpleName() + "'");
        } catch (RuntimeException e) {
            world.releaseRef();
            throw e;
        }

        for (int i = 0; i < oldBufferBindings.length; i++) {
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, i, oldBufferBindings[i]);
        }

        for (int i = 0; i < 12; i++) {
            GlStateManager._activeTexture(GlConst.GL_TEXTURE0+i);
            GlStateManager._bindTexture(0);
            glBindSampler(i, 0);
        }
    }

    public Viewport<?> setupViewport(Matrix4fc vanillaProjection, Matrix4fc modelView, FogParameters fogParameters, int width, int height, double cameraX, double cameraY, double cameraZ) {
        if (me.cortex.voxy.client.core.RenderBackend.isVitrailVulkanActive()) {
            this.tickVitrail(vanillaProjection, modelView, width, height, cameraX, cameraY, cameraZ);
            return null;
        }

        var viewport = this.getViewport();
        if (viewport == null) {
            return null;
        }

        if (VoxyCommon.IS_MINE_IN_ABYSS) {
            int sector = (((int)Math.floor(cameraX)>>4)+512)>>10;
            cameraX -= sector<<14;
            cameraY += (16+(256-32-sector*30))*16;
        }

        float farPlaneChunks = 3000;
        if (this.pipeline instanceof IrisVoxyRenderPipeline ivrp) {
            if (ivrp._getData().useDynamicFarPlane) {
                farPlaneChunks = (VoxyConfig.CONFIG.sectionRenderDistance * 32 + 2) * ((float) Math.sqrt(3));
            }
        }
        var voxyProjection = computeProjectionMat(this.properties, vanillaProjection, farPlaneChunks*16);

        {
            var factor = this.pipeline.getRenderScalingFactor();
            if (factor != null) {
                int yIndex = 1;
                width = (int) (width * factor[0]);
                height = (int) (height * factor[yIndex]);
            }
        }
        if (width == 0 || height == 0) {
            Logger.error("Viewport width or height was zero, this is bad bad bad");
            return null;
        }

        viewport
                .setVanillaProjection(vanillaProjection)
                .setProjection(voxyProjection)
                .setModelView(new Matrix4f(modelView))
                .setCamera(cameraX, cameraY, cameraZ)
                .setScreenSize(width, height)
                .setFogParameters(fogParameters)
                .update();

        if (VoxyClient.getOcclusionDebugState()==0) {
            viewport.frameId++;
        }

        return viewport;
    }

    /** Advances Voxy's CPU-only world and geometry queues from the Vulkan terrain stage. */
    public void tickVitrail(double cameraX, double cameraZ) {
        this.tickVitrail(null, null, 0, 0, cameraX, 0.0, cameraZ);
    }

    /** Updates the Vulkan compatibility view without changing Voxy's original GL traversal. */
    public void tickVitrail(@Nullable Matrix4fc projection, @Nullable Matrix4fc modelView,
            int width, int height,
            double cameraX, double cameraY, double cameraZ) {
        if (!me.cortex.voxy.client.core.RenderBackend.isVitrailVulkanActive()) return;
        this.vitrailCameraX = cameraX;
        this.vitrailCameraY = cameraY;
        this.vitrailCameraZ = cameraZ;
        if (projection != null && height > 0) {
            double projectionScalePixels = Math.abs(projection.m11()) * height * 0.5;
            this.vitrailProjectionScalePixels = projectionScalePixels;
            if (modelView != null) {
                this.vitrailViewProjection = new Matrix4f(projection).mul(modelView);
                if (width != this.vitrailProjectionWidth || height != this.vitrailProjectionHeight) {
                    this.vitrailProjectionWidth = width;
                    this.vitrailProjectionHeight = height;
                    this.vitrailWidestProjection = null;
                    this.vitrailWidestProjectionScalePixels = Double.POSITIVE_INFINITY;
                }
                if (this.vitrailWidestProjection == null
                        || projectionScalePixels < this.vitrailWidestProjectionScalePixels) {
                    this.vitrailWidestProjection = new Matrix4f(projection);
                    this.vitrailWidestProjectionScalePixels = projectionScalePixels;
                }
                this.vitrailCoverageViewProjection = new Matrix4f(this.vitrailWidestProjection).mul(modelView);
            }
        }
        if (this.renderDistanceTracker != null) {
            this.renderDistanceTracker.setCenterAndProcess(cameraX, cameraZ);
        }
        if (this.nodeManager != null) {
            this.nodeManager.tickCpuOnly();
            double maximumDistance = VoxyConfig.CONFIG.sectionRenderDistance * 16.0 * 32.0;
            var geometryNodes = this.nodeManager.getCpuGeometryNodesSnapshot();
            this.nodeManager.synchronizeCpuRefinementRequests(geometryNodes);
            Matrix4f shadowMatrix = this.vitrailViewProjection;
            Matrix4f coverageMatrix = this.vitrailCoverageViewProjection;
            if (shadowMatrix != null && coverageMatrix != null && width > 0 && height > 0) {
                HierarchyView hierarchy = this.nodeManager.getCpuHierarchyViewSnapshot();
                double baseSubdivision = Math.max(1.0, VoxyConfig.CONFIG.subDivisionSize);
                if (!Double.isFinite(this.vitrailAdaptiveSubdivisionPixels)
                        || this.vitrailAdaptiveSubdivisionPixels < baseSubdivision) {
                    this.vitrailAdaptiveSubdivisionPixels = baseSubdivision;
                }
                this.vitrailAdaptiveSubdivisionPixels = Math.min(
                        Math.max(baseSubdivision, VITRAIL_MAX_ADAPTIVE_SUBDIVISION_PIXELS),
                        this.vitrailAdaptiveSubdivisionPixels);
                double subdivision = this.vitrailAdaptiveSubdivisionPixels;
                var exact = this.vitrailShadowTraversal.traverse(hierarchy,
                        new ExactCpuTraversal.Parameters(shadowMatrix, coverageMatrix, width, height,
                                cameraX, cameraY, cameraZ, maximumDistance, subdivision));
                // The native renderer can submit a very large exact cut through indirect draws.
                // The compatibility path uses ordinary indexed draws, so raise the screen-space
                // threshold just enough to keep its complete hierarchy cut within a practical
                // node budget. This coarsens coverage instead of truncating it.
                // One corrective pass is enough.  Keep the corrected value for the next frame so
                // traversal hysteresis sees one coherent threshold instead of several different
                // thresholds in a single frame.
                if (exact.selectedNodeIds().size() > VITRAIL_VISIBLE_NODE_BUDGET) {
                    double pressure = Math.sqrt((double) exact.selectedNodeIds().size()
                            / (VITRAIL_VISIBLE_NODE_BUDGET * 0.88));
                    subdivision = Math.min(
                            Math.max(baseSubdivision, VITRAIL_MAX_ADAPTIVE_SUBDIVISION_PIXELS),
                            subdivision * Math.min(1.20, Math.max(1.025, pressure)));
                    this.vitrailAdaptiveSubdivisionPixels = subdivision;
                    exact = this.vitrailShadowTraversal.traverse(hierarchy,
                            new ExactCpuTraversal.Parameters(shadowMatrix, coverageMatrix, width, height,
                                    cameraX, cameraY, cameraZ, maximumDistance, subdivision));
                } else if (exact.selectedNodeIds().size() < VITRAIL_VISIBLE_NODE_BUDGET * 0.68
                        && subdivision > baseSubdivision) {
                    // Recover detail slowly after hierarchy/request pressure falls.  The dead band
                    // prevents the full render list from being rebuilt for tiny count changes.
                    this.vitrailAdaptiveSubdivisionPixels = Math.max(baseSubdivision,
                            subdivision * 0.985);
                }
                this.vitrailExactSnapshot = new ExactTraversalSnapshot(hierarchy, exact);
                this.vitrailShadowStats = exact.stats();
                this.vitrailEffectiveSubdivisionPixels = subdivision;

                // A cheap, complete first stage mirrors Voxy's coarse-to-fine presentation. It is
                // built from the same hierarchy and coverage rules, only with a coarser threshold.
                var coarse = this.vitrailCoarseTraversal.traverse(hierarchy,
                        new ExactCpuTraversal.Parameters(shadowMatrix, coverageMatrix, width, height,
                                cameraX, cameraY, cameraZ, maximumDistance,
                                Math.max(256.0, subdivision * 4.0)));
                this.vitrailCoarseSnapshot = new ExactTraversalSnapshot(hierarchy, coarse);

                // traversal_dev advances one hierarchy layer per dispatch.  Preserve that visible
                // behaviour on CPU: establish broad coarse coverage first, then refine the roots
                // nearest the player.  Recursive traversal order alone is depth first and used to
                // spend the request budget on one distant patch while neighbouring roots stayed
                // coarse for a long time.
                java.util.ArrayList<Integer> requestedNodeIds = new java.util.ArrayList<>(
                        exact.requestedNodeIds());
                requestedNodeIds.sort(java.util.Comparator
                        .<Integer>comparingInt(nodeId -> {
                            HierarchyView.Node node = hierarchy.node(nodeId);
                            return node == null ? -1 : node.level();
                        }).reversed()
                        .thenComparingDouble(nodeId -> {
                            HierarchyView.Node node = hierarchy.node(nodeId);
                            if (node == null) return Double.POSITIVE_INFINITY;
                            double size = 32.0 * (1L << node.level());
                            double minX = WorldEngine.getX(node.packedPosition()) * size;
                            double minZ = WorldEngine.getZ(node.packedPosition()) * size;
                            double dx = cameraX < minX ? minX - cameraX
                                    : cameraX > minX + size ? cameraX - minX - size : 0.0;
                            double dz = cameraZ < minZ ? minZ - cameraZ
                                    : cameraZ > minZ + size ? cameraZ - minZ - size : 0.0;
                            return dx * dx + dz * dz;
                        }));
                java.util.ArrayList<Long> requests = new java.util.ArrayList<>(requestedNodeIds.size());
                for (int nodeId : requestedNodeIds) {
                    HierarchyView.Node node = hierarchy.node(nodeId);
                    if (node != null) requests.add(node.packedPosition());
                }
                this.nodeManager.submitCpuRefinementRequests(requests);
            } else if (VITRAIL_LEGACY_SELECTOR) {
                this.nodeManager.submitCpuRefinementRequests(
                        me.cortex.voxy.client.core.rendering.hierachical.VitrailLodSelector.refinementRequests(
                                geometryNodes, cameraX, cameraY, cameraZ,
                                0.0, maximumDistance,
                                this.vitrailProjectionScalePixels, this.getVitrailSubdivisionPixels(),
                                this.vitrailViewProjection));
            }
        }
        if (this.modelService != null) this.modelService.tick(900_000);
    }

    public double getVitrailCameraX() { return this.vitrailCameraX; }
    public double getVitrailCameraY() { return this.vitrailCameraY; }
    public double getVitrailCameraZ() { return this.vitrailCameraZ; }
    public double getVitrailProjectionScalePixels() { return this.vitrailProjectionScalePixels; }
    public double getVitrailEffectiveSubdivisionPixels() { return this.vitrailEffectiveSubdivisionPixels; }

    /** Returns caller-owned copies of Voxy's retained CPU LOD sections for Vitrail conversion. */
    public List<me.cortex.voxy.client.core.rendering.building.BuiltSection> getVitrailCpuSectionsSnapshot() {
        if (!me.cortex.voxy.client.core.RenderBackend.isVitrailVulkanActive() || this.nodeManager == null) {
            return List.of();
        }
        return this.nodeManager.getCpuSectionsSnapshot();
    }

    public List<me.cortex.voxy.client.core.rendering.hierachical.NodeManager.GeometryNode>
    getVitrailGeometryNodesSnapshot() {
        if (!me.cortex.voxy.client.core.RenderBackend.isVitrailVulkanActive() || this.nodeManager == null) {
            return List.of();
        }
        return this.nodeManager.getCpuGeometryNodesSnapshot();
    }

    /** Renderer-neutral hierarchy hook intended for the compatibility layer and future bridge mod. */
    public HierarchyView getHierarchyViewSnapshot() {
        return this.nodeManager == null ? HierarchyView.EMPTY
                : this.nodeManager.getCpuHierarchyViewSnapshot();
    }

    /** Latest atomic hierarchy/result pair used by the Vitrail render-list adapter. */
    public ExactCpuTraversal.Result getShadowTraversalSnapshot() {
        return this.vitrailExactSnapshot.result();
    }

    /** Stable renderer/world identity for bridge-owned persistent geometry. */
    public long getVitrailHierarchyEpoch() {
        return this.vitrailExactSnapshot.hierarchy().epoch();
    }

    public List<me.cortex.voxy.client.core.rendering.hierachical.NodeManager.GeometryNode>
    getVitrailVisibleNodes(double cameraX, double cameraZ) {
        return this.getVitrailVisibleNodes(cameraX, cameraZ, 0);
    }

    public List<me.cortex.voxy.client.core.rendering.hierachical.NodeManager.GeometryNode>
    getVitrailVisibleNodes(double cameraX, double cameraZ, int minimumLevel) {
        if (!me.cortex.voxy.client.core.RenderBackend.isVitrailVulkanActive() || this.nodeManager == null) {
            return List.of();
        }
        if (!VITRAIL_LEGACY_SELECTOR) {
            ExactTraversalSnapshot snapshot = this.vitrailExactSnapshot;
            if (snapshot.hierarchy().epoch() != 0) {
                return nodesFromExactSnapshot(snapshot, minimumLevel);
            }
        }
        // sectionRenderDistance is scaled in 1/16-chunk steps. Voxy's own render path
        // converts it to blocks with *16*32; using only *32 here rejects distant CPU LODs.
        // Keep a complete LOD cut below vanilla terrain. A horizontal radius cut exposes internal
        // faces from high camera positions; original Voxy hides overlap with depth/Hi-Z instead.
        double minimumDistance = 0.0;
        double maxDistance = VoxyConfig.CONFIG.sectionRenderDistance * 16.0 * 32.0;
        double baseSubdivision = Math.max(1.0, VoxyConfig.CONFIG.subDivisionSize);
        double subdivision = Math.max(baseSubdivision, this.getVitrailSubdivisionPixels());
        var geometryNodes = this.nodeManager.getCpuGeometryNodesSnapshot();
        List<me.cortex.voxy.client.core.rendering.hierachical.NodeManager.GeometryNode> selected =
                me.cortex.voxy.client.core.rendering.hierachical.VitrailLodSelector.select(
                geometryNodes, cameraX, this.vitrailCameraY, cameraZ,
                minimumDistance, maxDistance, minimumLevel,
                this.vitrailProjectionScalePixels, subdivision,
                this.vitrailViewProjection);
        // Voxy's native traversal removes hidden nodes with Hi-Z. The CPU compatibility path has
        // no cheap depth pyramid, so use the same screen-size control as a conservative node
        // budget. This preserves a complete hierarchical cut instead of truncating meshes.
        for (int pass = 0; pass < 4 && selected.size() > VITRAIL_VISIBLE_NODE_BUDGET; pass++) {
            double pressure = Math.sqrt((double) selected.size() / VITRAIL_VISIBLE_NODE_BUDGET);
            subdivision *= Math.min(2.0, Math.max(1.15, pressure * 1.05));
            selected = me.cortex.voxy.client.core.rendering.hierachical.VitrailLodSelector.select(
                    geometryNodes, cameraX, this.vitrailCameraY, cameraZ,
                    minimumDistance, maxDistance, minimumLevel,
                    this.vitrailProjectionScalePixels, subdivision, this.vitrailViewProjection);
        }
        if (selected.size() < VITRAIL_VISIBLE_NODE_BUDGET * 0.6 && subdivision > baseSubdivision) {
            this.vitrailEffectiveSubdivisionPixels = Math.max(baseSubdivision, subdivision * 0.92);
        } else {
            this.vitrailEffectiveSubdivisionPixels = subdivision;
        }
        return selected;
    }

    public List<me.cortex.voxy.client.core.rendering.hierachical.NodeManager.GeometryNode>
    getVitrailCoarseVisibleNodes() {
        return nodesFromExactSnapshot(this.vitrailCoarseSnapshot, 0);
    }

    private static List<me.cortex.voxy.client.core.rendering.hierachical.NodeManager.GeometryNode>
    nodesFromExactSnapshot(ExactTraversalSnapshot snapshot, int minimumLevel) {
        if (snapshot.hierarchy().epoch() == 0) return List.of();
        java.util.ArrayList<me.cortex.voxy.client.core.rendering.hierachical.NodeManager.GeometryNode>
                selected = new java.util.ArrayList<>(snapshot.result().selectedNodeIds().size());
        for (int nodeId : snapshot.result().selectedNodeIds()) {
            HierarchyView.Node node = snapshot.hierarchy().node(nodeId);
            if (node == null || !node.hasDrawableGeometry() || node.level() < minimumLevel) continue;
            selected.add(new me.cortex.voxy.client.core.rendering.hierachical.NodeManager.GeometryNode(
                    node.packedPosition(), node.geometryId(), node.geometryVersion(), node.level(),
                    node.childMask(), node.inner(), node.requestInFlight()));
        }
        return List.copyOf(selected);
    }

    private double getVitrailSubdivisionPixels() {
        double value = this.vitrailEffectiveSubdivisionPixels;
        return value > 0.0 && Double.isFinite(value)
                ? value : Math.max(1.0, VoxyConfig.CONFIG.subDivisionSize);
    }

    public me.cortex.voxy.client.core.rendering.building.BuiltSection
    getVitrailCpuSectionSnapshot(int geometryId) {
        if (!me.cortex.voxy.client.core.RenderBackend.isVitrailVulkanActive() || this.nodeManager == null) {
            return null;
        }
        return this.nodeManager.getCpuSectionSnapshot(geometryId);
    }

    /** Selects the current CPU LOD cut and returns caller-owned copies of those section meshes. */
    public List<VitrailVisibleSection> getVitrailVisibleSections(double cameraX, double cameraZ) {
        List<me.cortex.voxy.client.core.rendering.hierachical.NodeManager.GeometryNode> selected =
                this.getVitrailVisibleNodes(cameraX, cameraZ);
        java.util.ArrayList<VitrailVisibleSection> result = new java.util.ArrayList<>(selected.size());
        try {
            for (var node : selected) {
                var section = this.getVitrailCpuSectionSnapshot(node.geometryId());
                if (section != null) {
                    if (section.position != node.position()) {
                        section.free();
                        continue;
                    }
                    result.add(new VitrailVisibleSection(node, section));
                }
            }
            return List.copyOf(result);
        } catch (RuntimeException | Error e) {
            result.forEach(VitrailVisibleSection::close);
            throw e;
        }
    }

    /** Reads the CPU mirror of a model face record used to expand Voxy's packed quad. */
    public int getVitrailModelFaceData(int modelId, int face) {
        if (!me.cortex.voxy.client.core.RenderBackend.isVitrailVulkanActive() || this.modelService == null) {
            throw new IllegalStateException("Vitrail CPU model data is not available");
        }
        return this.modelService.factory.getFaceData(modelId, face);
    }

    /** DH mini-ID used by shader packs to distinguish distant terrain materials. */
    public int getVitrailModelMaterial(int modelId) {
        if (!me.cortex.voxy.client.core.RenderBackend.isVitrailVulkanActive() || this.modelService == null) {
            throw new IllegalStateException("Vitrail CPU model data is not available");
        }
        return this.modelService.factory.getVitrailDistantMaterial(modelId);
    }

    public int getVitrailNearSuppression(int modelId) {
        if (!me.cortex.voxy.client.core.RenderBackend.isVitrailVulkanActive() || this.modelService == null) {
            return 0;
        }
        long metadata = this.modelService.factory.getModelMetadataFromClientId(modelId);
        return !me.cortex.voxy.client.core.model.ModelQueries.isFluid(metadata)
                && !me.cortex.voxy.client.core.model.ModelQueries.isFullyOpaque(metadata) ? 1 : 0;
    }

    public int[] getVitrailFacePixels(int modelId, int face) {
        return this.modelService.factory.getVitrailFacePixels(modelId, face);
    }

    public int getVitrailFaceAverage(int modelId, int face) {
        return this.modelService.factory.getVitrailFaceAverage(modelId, face);
    }

    /** Model/biome tint and Voxy's directional face shade in Vitrail's RGBA byte order. */
    public int getVitrailFaceColour(int modelId, int biomeId, int face, boolean opaque) {
        if (!me.cortex.voxy.client.core.RenderBackend.isVitrailVulkanActive() || this.modelService == null) {
            return 0xffff_ffff;
        }
        int rgba = this.modelService.factory.getVitrailFaceColour(modelId, biomeId, face, opaque);
        var level = net.minecraft.client.Minecraft.getInstance().level;
        if (level == null || !this.modelService.factory.isModelShaded(modelId)) return rgba;
        var light = level.cardinalLighting();
        float shade = switch (face) {
            case 0 -> light.down();
            case 1 -> light.up();
            case 2, 3 -> light.north();
            case 4, 5 -> light.east();
            default -> 1.0f;
        };
        int red = Math.round(((rgba >>> 24) & 0xff) * shade);
        int green = Math.round(((rgba >>> 16) & 0xff) * shade);
        int blue = Math.round(((rgba >>> 8) & 0xff) * shade);
        return (red << 24) | (green << 16) | (blue << 8) | (rgba & 0xff);
    }

    public void renderOpaque(Viewport<?> viewport, int sourceDepthTexture, int sourceColourTexture) {
        if (me.cortex.voxy.client.core.RenderBackend.isVitrailVulkanActive()) {
            return;
        }
        if (viewport == null) {
            return;
        }

        if (viewport.width <= 0 || viewport.height <= 0) {
            Logger.error("Viewport width or height was zero, this is bad bad bad, exiting frame");
            return;
        }

        if (sourceDepthTexture == 0) {
            throw new IllegalStateException("Source depth texture cannot be 0");
        }

        TimingStatistics.resetSamplers();

        TimingStatistics.all.start();
        GPUTiming.INSTANCE.marker();
        TimingStatistics.main.start();

        int[] oldBufferBindings = new int[10];
        for (int i = 0; i < oldBufferBindings.length; i++) {
            oldBufferBindings[i] = glGetIntegeri(GL_SHADER_STORAGE_BUFFER_BINDING, i);
        }

        GlStateManager._enableDepthTest();
        GlStateManager._depthFunc(this.properties.closerEqualDepthCompare());
        GlStateManager._depthMask(true);
        GlStateManager._disablePolygonOffset();

        int oldFB = GL11.glGetInteger(GL_DRAW_FRAMEBUFFER_BINDING);

        int[] dims = new int[4];
        glGetIntegerv(GL_VIEWPORT, dims);

        glViewport(0, 0, viewport.width, viewport.height);

        int scrWidth  = glGetTextureLevelParameteri(sourceDepthTexture, 0, GL_TEXTURE_WIDTH);
        int scrHeight = glGetTextureLevelParameteri(sourceDepthTexture, 0, GL_TEXTURE_HEIGHT);

        this.pipeline.preSetup(viewport);

        TimingStatistics.E.start();
        if (this.visbleSectionStream != null && (!VoxyClient.disableSodiumChunkRender()) && !IrisUtil.irisShadowActive()) {
            if (VoxyClient.isFrexActive()!=(this.columnStreamedBoundStore!=null)) {
                if (this.columnStreamedBoundStore == null) {
                    this.columnStreamedBoundStore = new ColumnStreamedBoundStore();
                } else {
                    this.columnStreamedBoundStore.free();
                    this.columnStreamedBoundStore = null;
                }
            }
            this.boundOutlineRenderer.render(viewport, this.columnStreamedBoundStore==null?this.visbleSectionStream:this.columnStreamedBoundStore);
        } else {
            viewport.depthBoundingBuffer.clear(this.properties.inverseClearDepth());
        }
        TimingStatistics.E.stop();

        GPUTiming.INSTANCE.marker();
        this.pipeline.runPipeline(viewport, sourceDepthTexture, sourceColourTexture, scrWidth, scrHeight);
        GPUTiming.INSTANCE.marker();

        TimingStatistics.main.stop();
        TimingStatistics.postDynamic.start();

        PrintfDebugUtil.tick();

        {
            UploadStream.INSTANCE.tick();

            while (this.renderDistanceTracker.setCenterAndProcess(viewport.cameraX, viewport.cameraZ) && VoxyClient.isFrexActive());
            TimingStatistics.H.start();
            do { this.modelService.tick(900_000); } while (VoxyClient.isFrexActive() && !this.modelService.areQueuesEmpty());
            TimingStatistics.H.stop();
        }

        GPUTiming.INSTANCE.marker();
        TimingStatistics.postDynamic.stop();

        GPUTiming.INSTANCE.tick();

        int yIndex = 1;
        glBindFramebuffer(GlConst.GL_FRAMEBUFFER, oldFB);
        glViewport(dims[0], dims[yIndex], dims[2], dims[3]);

        {
            GlStateManager._glUseProgram(0);
            glUseProgram(0);
            GlStateManager._enableDepthTest();
            glEnable(GL_DEPTH_TEST);
            glDisable(GL_STENCIL_TEST);

            GlStateManager._glBindVertexArray(0);
            glBindVertexArray(0);

            GlStateManager._activeTexture(GlConst.GL_TEXTURE1);
            for (int i = 0; i < 12; i++) {
                GlStateManager._activeTexture(GlConst.GL_TEXTURE0+i);
                GlStateManager._bindTexture(0);
                glBindSampler(i, 0);
            }

            IrisUtil.clearIrisSamplers();

            for (int i = 0; i < oldBufferBindings.length; i++) {
                glBindBufferBase(GL_SHADER_STORAGE_BUFFER, i, oldBufferBindings[i]);
            }
            GlStateManager._blendEquationSeparate(GL_FUNC_ADD, GL_FUNC_ADD);
            glBlendEquation(GL_FUNC_ADD);
            GlStateManager._blendFuncSeparate(0,0, 0, 0);
            glBlendFunc(0, 0);
            GlStateManager._disableBlend(0);
            glDisable(GL_BLEND);
            GlStateManager._depthFunc(GL_LESS);
            glDepthFunc(GL_LESS);
        }

        TimingStatistics.all.stop();
    }

    private void autoBalanceSubDivSize() {
        boolean canDecreaseSize = this.renderGen.getTaskCount() < 300;
        int MIN_FPS = 55;
        int MAX_FPS = 65;
        float INCREASE_PER_SECOND = 60;
        float DECREASE_PER_SECOND = 30;
        if (Minecraft.getInstance().getFps() < MIN_FPS) {
            VoxyConfig.CONFIG.subDivisionSize = Math.min(VoxyConfig.CONFIG.subDivisionSize + INCREASE_PER_SECOND / Math.max(1f, Minecraft.getInstance().getFps()), 256);
        }

        if (MAX_FPS < Minecraft.getInstance().getFps() && canDecreaseSize) {
            VoxyConfig.CONFIG.subDivisionSize = Math.max(VoxyConfig.CONFIG.subDivisionSize - DECREASE_PER_SECOND / Math.max(1f, Minecraft.getInstance().getFps()), 28);
        }
    }

    public static float getVanillaRenderDistance() {
        return Minecraft.getInstance().options.getEffectiveRenderDistance()*16;
    }

    private static Matrix4f computeProjectionMat(RenderProperties properties, Matrix4fc base, float farPlane) {
        var rawMCProj = Minecraft.getInstance().gameRenderer.gameRenderState().levelRenderState.cameraRenderState.projectionMatrix;
        var extraProjection = rawMCProj.invert(new Matrix4f()).mul(base);

        float near = getVanillaRenderDistance()<=32.0f?8f:16f;
        near = VoxyClient.disableSodiumChunkRender()?0.1f:near;

        float far = farPlane;

        if (properties.isReverseZ()) {
            float tmp = near;
            near = far;
            far = tmp;
        }

        return extraProjection.mulLocal(
                new Matrix4f(rawMCProj)
                .m22((properties.isZero2One()?far:(far+near)) / (near - far))
                .m32((properties.isZero2One()?far:(far+far)) * near / (near - far))
        );
    }

    private boolean frexStillHasWork() {
        if (me.cortex.voxy.client.core.RenderBackend.isVitrailVulkanActive() || !VoxyClient.isFrexActive()) {
            return false;
        }
        UploadStream.INSTANCE.tick();
        this.modelService.tick(100_000_000);
        GL11.glFinish();
        return this.nodeManager.hasWork() || this.renderGen.getTaskCount()!=0 || !this.modelService.areQueuesEmpty();
    }

    public void setRenderDistance(float renderDistance) {
        if (this.renderDistanceTracker != null) {
            this.renderDistanceTracker.setRenderDistance((int) Math.ceil(renderDistance + 1));
        }
    }

    public Viewport<?> getViewport() {
        if (me.cortex.voxy.client.core.RenderBackend.isVitrailVulkanActive()) {
            return null;
        }
        if (IrisUtil.irisShadowActive()) {
            return null;
        }
        return this.viewportSelector.getViewport();
    }

    public void addDebugInfo(List<String> debug) {
        if (me.cortex.voxy.client.core.RenderBackend.isVitrailVulkanActive()) {
            debug.add("Voxy [Vitrail/Vulkan]: CPU LOD provider active");
            if (this.modelService != null) this.modelService.addDebugData(debug);
            if (this.renderGen != null) this.renderGen.addDebugData(debug);
            if (this.nodeManager != null) this.nodeManager.addDebug(debug);
            ExactCpuTraversal.Stats shadow = this.vitrailShadowStats;
            debug.add(String.format(java.util.Locale.ROOT,
                    "Exact %s: V/S/F/R/U/O %d/%d/%d/%d/%d/%d %.2fms G%d",
                    VITRAIL_LEGACY_SELECTOR ? "shadow" : "active",
                    shadow.visitedNodes(), shadow.selectedNodes(), shadow.fallbackCount(),
                    shadow.requestCount(), shadow.uncoveredBranches(), shadow.parentChildOverlap(),
                    shadow.traversalMillis(), shadow.hierarchyGeneration()));
            return;
        }
        debug.add("Buf/Tex [#/Mb]: [" + GlBuffer.getCount() + "/" + (GlBuffer.getTotalSize()/1_000_000) + "],[" + GlTexture.getCount() + "/" + (GlTexture.getEstimatedTotalSize()/1_000_000)+"]");
        {
            this.modelService.addDebugData(debug);
            this.renderGen.addDebugData(debug);
            this.nodeManager.addDebug(debug);
            this.pipeline.addDebug(debug);
        }
        {
            TimingStatistics.update();
            debug.add("Voxy frame runtime (millis): " + TimingStatistics.dynamic.pVal() + ", " + TimingStatistics.main.pVal()+ ", " + TimingStatistics.postDynamic.pVal()+ ", " + TimingStatistics.all.pVal());
            debug.add("Extra time: " + TimingStatistics.A.pVal() + ", " + TimingStatistics.B.pVal() + ", " + TimingStatistics.C.pVal() + ", " + TimingStatistics.D.pVal());
            debug.add("Extra 2 time: " + TimingStatistics.E.pVal() + ", " + TimingStatistics.F.pVal() + ", " + TimingStatistics.G.pVal() + ", " + TimingStatistics.H.pVal() + ", " + TimingStatistics.I.pVal());
        }
        debug.add(GPUTiming.INSTANCE.getDebug());
        PrintfDebugUtil.addToOut(debug);
    }

    public void shutdown() {
        if (me.cortex.voxy.client.core.RenderBackend.isVitrailVulkanActive()) {
            VitrailBridge.unregisterProvider();
            try {
                this.worldIn.setDirtyCallback(null);
                this.worldIn.getMapper().setBiomeCallback(null);
                this.worldIn.getMapper().setStateCallback(null);

                if (this.nodeManager != null) {
                    this.nodeManager.stop();
                }
                if (this.modelService != null) {
                    this.modelService.shutdown();
                }
                if (this.renderGen != null) {
                    this.renderGen.shutdown();
                }
                if (this.nodeCleaner != null) {
                    this.nodeCleaner.free();
                }
                if (this.geometryData != null) {
                    this.geometryData.free();
                }
                if (this.geoRef != null) {
                    this.geoRef.clean();
                }
                if (this.visbleSectionStream != null) {
                    this.visbleSectionStream.free();
                }
            } catch (Exception e) {
                Logger.error("Error shutting down renderer components in Vitrail mode", e);
            }
            if (this.worldIn != null) {
                this.worldIn.releaseRef();
            }
            Logger.info("Voxy render system: Vitrail bypass shutdown completed");
            return;
        }

        Logger.info("Flushing download stream");
        DownloadStream.INSTANCE.flushWaitClear();
        Logger.info("Shutting down rendering");
        try {
            this.worldIn.setDirtyCallback(null);
            this.worldIn.getMapper().setBiomeCallback(null);
            this.worldIn.getMapper().setStateCallback(null);

            this.nodeManager.stop();

            this.modelService.shutdown();
            this.renderGen.shutdown();
            this.traversal.free();
            this.nodeCleaner.free();
            this.geometryData.free();
            if (this.geoRef != null) {
                this.geoRef.clean();
            }

            this.boundOutlineRenderer.free();
            if (this.visbleSectionStream != null) {
                this.visbleSectionStream.free();
            }
            if (this.columnStreamedBoundStore != null) {
                this.columnStreamedBoundStore.free();
                this.columnStreamedBoundStore = null;
            }

            this.viewportSelector.free();
        } catch (Exception e) {Logger.error("Error shutting down renderer components", e);}
        Logger.info("Shutting down render pipeline");
        try {this.pipeline.free();} catch (Exception e){Logger.error("Error releasing render pipeline", e);}

        Logger.info("Flushing download stream");
        DownloadStream.INSTANCE.flushWaitClear();

        this.worldIn.releaseRef();
        Logger.info("Render shutdown completed");
    }

    public WorldEngine getEngine() {
        return this.worldIn;
    }
}

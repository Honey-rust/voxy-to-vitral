package me.cortex.voxy.client.core.rendering.compat;

import me.cortex.voxy.common.world.WorldEngine;
import org.joml.Matrix4fc;

import java.util.ArrayList;
import java.util.BitSet;
import it.unimi.dsi.fastutil.longs.Long2LongOpenHashMap;
import it.unimi.dsi.fastutil.longs.Long2ObjectOpenHashMap;
import it.unimi.dsi.fastutil.longs.LongOpenHashSet;
import java.util.List;

/**
 * CPU shadow implementation of Voxy's hierarchy traversal.
 *
 * <p>This class has no Vitrail dependency and produces node ids rather than draw commands. It is
 * deliberately conservative: an incomplete child cut falls back to a resident parent.</p>
 */
public final class ExactCpuTraversal {
    private static final double EXIT_HYSTERESIS = 0.85;
    private static final double CLIP_EPSILON = 1.0e-5;
    /**
     * Native Voxy only retains the current frustum and lets Hi-Z remove more of that cut.  The
     * bridge deliberately keeps an omnidirectional fallback because its provider aggregate is
     * replaced asynchronously.  Compensate for that larger working set by spending progressively
     * fewer child nodes on the far part of the render distance.
     */
    private static final double FAR_DETAIL_START_FRACTION = 0.20;
    private static final double FAR_DETAIL_MAX_MULTIPLIER = 4.0;

    private static final long ZOOM_DETAIL_NANOS = 400_000_000L;
    private static final long RETAINED_DETAIL_NANOS = 8_000_000_000L;
    private final Long2LongOpenHashMap lastViewedRefinementTime = new Long2LongOpenHashMap();
    private final Long2ObjectOpenHashMap<ZoomDetail> zoomDetails = new Long2ObjectOpenHashMap<>();
    private record ZoomDetail(double factor, long untilNanos) {}
    // Traversals are sequential on the render thread; reuse the audit scratch table.
    private final LongOpenHashSet selectedPositions = new LongOpenHashSet();
    private int frame;
    private long historyEpoch = Long.MIN_VALUE;
    private long zoomExitSince;

    public Result traverse(HierarchyView hierarchy, Parameters parameters) {
        return traverse(hierarchy, parameters, null);
    }

    /** A presentation cut may advance only one layer below the cut actually on screen. */
    public Result traverse(HierarchyView hierarchy, Parameters parameters, java.util.Set<Long> refinementFrontier) {
        long start = System.nanoTime();
        if (hierarchy.epoch() != this.historyEpoch) {
            this.lastViewedRefinementTime.clear();
            this.zoomDetails.clear();
            this.frame = 0;
            this.zoomExitSince = 0;
            this.historyEpoch = hierarchy.epoch();
        }
        this.frame++;

        State state = new State(hierarchy, parameters, this.lastViewedRefinementTime, this.zoomDetails, start, refinementFrontier);
        if (state.zoomFactor <= 1.20) {
            if (this.zoomExitSince == 0) this.zoomExitSince = start;
        } else {
            this.zoomExitSince = 0;
        }
        if (this.zoomExitSince != 0 && start - this.zoomExitSince >= 120_000_000L) {
            // Retain encoded meshes in the cache, but stop selecting extra zoom-only detail.
            for (long position : this.zoomDetails.keySet()) this.lastViewedRefinementTime.remove(position);
            this.zoomDetails.clear();
        }
        for (int rootId : hierarchy.topLevelNodeIds()) {
            state.visit(rootId);
        }
        // Distance-selected refinements are refreshed every traversal, regardless of yaw,
        // pitch or visibility. Only unvisited entries expire (unloaded/out-of-range nodes).
        if ((this.frame & 31) == 0) {
            var retained = this.lastViewedRefinementTime.long2LongEntrySet().fastIterator();
            while (retained.hasNext()) {
                if (start - retained.next().getLongValue() > RETAINED_DETAIL_NANOS) retained.remove();
            }
            var zoom = this.zoomDetails.long2ObjectEntrySet().fastIterator();
            while (zoom.hasNext()) {
                if (start > zoom.next().getValue().untilNanos()) zoom.remove();
            }
        }
        state.parentChildOverlap += state.countSelectedParentChildOverlaps(this.selectedPositions);

        long elapsed = System.nanoTime() - start;
        Stats stats = new Stats(
                state.visitedNodes,
                state.selectedNodeIds.size(),
                state.fallbackCount,
                state.requestedNodeIds.size(),
                elapsed,
                state.parentChildOverlap,
                state.uncoveredBranches,
                state.frustumCulled,
                state.distanceCulled,
                hierarchy.epoch(),
                hierarchy.generation());
        return new Result(List.copyOf(state.selectedNodeIds),
                List.copyOf(state.requestedNodeIds), stats);
    }

    public record Parameters(
            Matrix4fc viewProjection,
            Matrix4fc coverageViewProjection,
            int viewportWidth,
            int viewportHeight,
            double cameraX,
            double cameraY,
            double cameraZ,
            double renderDistanceBlocks,
            double subdivisionPixels) {
        public Parameters {
            if (viewProjection == null) throw new IllegalArgumentException("viewProjection");
            if (coverageViewProjection == null) throw new IllegalArgumentException("coverageViewProjection");
            if (viewportWidth <= 0 || viewportHeight <= 0) throw new IllegalArgumentException("viewport");
            if (!(subdivisionPixels > 0.0)) throw new IllegalArgumentException("subdivisionPixels");
        }
    }

    public record Result(List<Integer> selectedNodeIds, List<Integer> requestedNodeIds, Stats stats) {
        public static final Result EMPTY = new Result(List.of(), List.of(), Stats.EMPTY);
    }

    public record Stats(
            int visitedNodes,
            int selectedNodes,
            int fallbackCount,
            int requestCount,
            long traversalNanos,
            int parentChildOverlap,
            int uncoveredBranches,
            int frustumCulled,
            int distanceCulled,
            long hierarchyEpoch,
            long hierarchyGeneration) {
        public static final Stats EMPTY = new Stats(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
        public double traversalMillis() { return this.traversalNanos / 1_000_000.0; }
    }

    private static final class State {
        private final HierarchyView hierarchy;
        private final Parameters parameters;
        private final Long2LongOpenHashMap lastViewedRefinementTime;
        private final Long2ObjectOpenHashMap<ZoomDetail> zoomDetails;
        private final java.util.Set<Long> refinementFrontier;
        private final double zoomFactor;
        private final double coveragePixels;
        private final long nowNanos;
        private final BitSet visited = new BitSet();
        private final BitSet requested = new BitSet();
        private final ArrayList<Integer> selectedNodeIds = new ArrayList<>();
        private final ArrayList<Integer> requestedNodeIds = new ArrayList<>();
        // Projection is complete before recursion starts, so one scratch set serves the traversal.
        private final double[] clipX = new double[8];
        private final double[] clipY = new double[8];
        private final double[] clipZ = new double[8];
        private final double[] clipW = new double[8];
        private final double[] screenX = new double[8];
        private final double[] screenY = new double[8];

        private int visitedNodes;
        private int fallbackCount;
        private int parentChildOverlap;
        private int uncoveredBranches;
        private int frustumCulled;
        private int distanceCulled;

        private State(HierarchyView hierarchy, Parameters parameters,
                Long2LongOpenHashMap lastViewedRefinementTime, Long2ObjectOpenHashMap<ZoomDetail> zoomDetails, long nowNanos, java.util.Set<Long> refinementFrontier) {
            this.hierarchy = hierarchy;
            this.refinementFrontier = refinementFrontier;
            this.parameters = parameters;
            this.lastViewedRefinementTime = lastViewedRefinementTime;
            this.zoomDetails = zoomDetails;
            this.coveragePixels = focalScale(parameters.coverageViewProjection) * parameters.viewportHeight * 0.5;
            this.zoomFactor = Math.max(1.0, focalScale(parameters.viewProjection)
                    / Math.max(1.0e-6, focalScale(parameters.coverageViewProjection)));
            this.nowNanos = nowNanos;
        }

        /** Returns true if the visible part of this branch has complete geometry coverage. */
        private boolean visit(int nodeId) {
            HierarchyView.Node node = this.hierarchy.node(nodeId);
            if (node == null) {
                this.uncoveredBranches++;
                return false;
            }
            if (this.visited.get(nodeId)) {
                // A valid Voxy hierarchy is a tree. Treat duplicate/cyclic references as uncovered.
                this.parentChildOverlap++;
                return false;
            }
            this.visited.set(nodeId);
            this.visitedNodes++;

            if (!withinRenderDistance(node)) {
                this.distanceCulled++;
                return true;
            }

            // A persistent all-direction cut must use the same spatial metric in every
            // direction. Projected face areas change with yaw and pitch, causing asynchronous
            // fine/coarse replacements even when the camera has not moved.
            Projection projection = projectOmnidirectional(node);
            boolean retainedDetail = this.lastViewedRefinementTime.containsKey(node.packedPosition());
            double threshold = this.parameters.subdivisionPixels * distanceDetailMultiplier(node);
            if (retainedDetail) threshold *= EXIT_HYSTERESIS;
            // Sprint/speed FOV settling is not Zoomify. Do not renew a strong historical zoom
            // merely because the current normal FOV differs slightly from the widest one.
            boolean zoomVisible = this.zoomFactor > 1.25
                    && !project(node, this.parameters.viewProjection).outsideFrustum;
            ZoomDetail zoom = this.zoomDetails.get(node.packedPosition());
            double detailFactor = zoom != null && this.nowNanos <= zoom.untilNanos() ? zoom.factor() : 1.0;
            // Current visible zoom wins: partially zooming out must also reduce detail.
            if (zoomVisible) detailFactor = this.zoomFactor;
            threshold /= detailFactor;
            boolean refine = node.level() != 0 && projection.pixelArea > threshold * threshold;
            if (refine) {
                this.lastViewedRefinementTime.put(node.packedPosition(), this.nowNanos);
                if (zoomVisible) this.zoomDetails.put(node.packedPosition(),
                        new ZoomDetail(this.zoomFactor, this.nowNanos + ZOOM_DETAIL_NANOS));
            } else {
                this.lastViewedRefinementTime.remove(node.packedPosition());
            }

            if (refine) {
                // Request traversal remains unrestricted. Only presentation waits for the
                // previous layer to be published, preventing a ready hierarchy skipping levels.
                // Missing/empty parents cannot cover their children and must still descend.
                if (this.refinementFrontier != null && node.hasDrawableGeometry()
                        && !reachesCamera(node)
                        && !this.refinementFrontier.contains(node.packedPosition())) {
                    select(node);
                    return true;
                }
                boolean descended = hasUsableChildren(node);
                if (descended) {
                    int checkpoint = this.selectedNodeIds.size();
                    boolean complete = visitChildren(node);
                    if (complete) return true;
                    rollbackSelections(checkpoint);
                }
                // This matches traversal_dev: a leaf that should subdivide requests itself. An
                // already expanded node does not request itself merely because a child is late.
                if (!descended) request(node);
                if (node.hasGeometry()) {
                    select(node);
                    this.fallbackCount++;
                    return true;
                }
                this.uncoveredBranches++;
                return false;
            }

            if (node.hasGeometry()) {
                select(node);
                return true;
            }

            request(node);
            if (node.level() != 0 && hasUsableChildren(node)) {
                int checkpoint = this.selectedNodeIds.size();
                boolean complete = visitChildren(node);
                if (complete) return true;
                rollbackSelections(checkpoint);
            }
            this.uncoveredBranches++;
            return false;
        }

        private boolean visitChildren(HierarchyView.Node node) {
            if (!hasCompleteChildAllocation(node)) return false;
            boolean complete = true;
            for (int i = 0; i < node.childCount(); i++) {
                complete &= visit(node.childPointer() + i);
            }
            return complete;
        }

        private boolean hasUsableChildren(HierarchyView.Node node) {
            return node.hasChildren() && !node.childListEmpty() && node.childCount() > 0;
        }

        private boolean hasCompleteChildAllocation(HierarchyView.Node node) {
            if (!hasUsableChildren(node)) return false;
            int expected = Integer.bitCount(Byte.toUnsignedInt(node.childMask()));
            if (expected != 0 && expected != node.childCount()) return false;
            for (int i = 0; i < node.childCount(); i++) {
                if (this.hierarchy.node(node.childPointer() + i) == null) return false;
            }
            return true;
        }

        private void select(HierarchyView.Node node) {
            // EMPTY_MESH participates in coverage but intentionally emits no draw.
            if (node.hasDrawableGeometry()) this.selectedNodeIds.add(node.stableNodeId());
        }

        private void request(HierarchyView.Node node) {
            if (node.requestInFlight() || this.requested.get(node.stableNodeId())) return;
            this.requested.set(node.stableNodeId());
            this.requestedNodeIds.add(node.stableNodeId());
        }

        private void rollbackSelections(int checkpoint) {
            while (this.selectedNodeIds.size() > checkpoint) {
                this.selectedNodeIds.remove(this.selectedNodeIds.size() - 1);
            }
        }

        private int countSelectedParentChildOverlaps(LongOpenHashSet selectedPositions) {
            selectedPositions.clear();
            for (int nodeId : this.selectedNodeIds) {
                HierarchyView.Node node = this.hierarchy.node(nodeId);
                if (node != null) selectedPositions.add(node.packedPosition());
            }
            int overlaps = 0;
            for (int nodeId : this.selectedNodeIds) {
                HierarchyView.Node node = this.hierarchy.node(nodeId);
                if (node == null) continue;
                long position = node.packedPosition();
                for (int level = node.level() + 1; level <= WorldEngine.MAX_LOD_LAYER; level++) {
                    position = WorldEngine.getWorldSectionId(level,
                            WorldEngine.getX(position) >> 1,
                            WorldEngine.getY(position) >> 1,
                            WorldEngine.getZ(position) >> 1);
                    if (selectedPositions.contains(position)) {
                        overlaps++;
                        break;
                    }
                }
            }
            return overlaps;
        }

        private boolean reachesCamera(HierarchyView.Node node) {
            double size = 32.0 * (1L << node.level());
            double x = WorldEngine.getX(node.packedPosition()) * size;
            double y = WorldEngine.getY(node.packedPosition()) * size;
            double z = WorldEngine.getZ(node.packedPosition()) * size;
            double dx = distanceToInterval(this.parameters.cameraX, x, x + size);
            double dy = distanceToInterval(this.parameters.cameraY, y, y + size);
            double dz = distanceToInterval(this.parameters.cameraZ, z, z + size);
            // Match the bridge's loading guard: these coarse parents are not drawn,
            // so waiting for publication would prevent their children ever appearing.
            return dx * dx + dy * dy + dz * dz < 32.0 * 32.0;
        }

        private boolean withinRenderDistance(HierarchyView.Node node) {
            double distance = this.parameters.renderDistanceBlocks;
            if (distance < 0.0) return true;
            double size = 32.0 * (1L << node.level());
            double minX = (double) WorldEngine.getX(node.packedPosition()) * size;
            double minZ = (double) WorldEngine.getZ(node.packedPosition()) * size;
            double dx = distanceToInterval(this.parameters.cameraX, minX, minX + size);
            double dz = distanceToInterval(this.parameters.cameraZ, minZ, minZ + size);
            return dx * dx + dz * dz <= distance * distance;
        }

        private double distanceDetailMultiplier(HierarchyView.Node node) {
            double renderDistance = this.parameters.renderDistanceBlocks;
            if (!(renderDistance > 0.0)) return 1.0;

            double size = 32.0 * (1L << node.level());
            double minX = (double) WorldEngine.getX(node.packedPosition()) * size;
            double minZ = (double) WorldEngine.getZ(node.packedPosition()) * size;
            double dx = distanceToInterval(this.parameters.cameraX, minX, minX + size);
            double dz = distanceToInterval(this.parameters.cameraZ, minZ, minZ + size);
            double distance = Math.sqrt(dx * dx + dz * dz);
            double start = renderDistance * FAR_DETAIL_START_FRACTION;
            if (distance <= start || start >= renderDistance) return 1.0;

            double t = Math.min(1.0, (distance - start) / (renderDistance - start));
            // Ease in so middle-distance silhouettes retain detail, while the outer range reaches
            // a much smaller stable hierarchy cut.
            double eased = t * t * (3.0 - 2.0 * t);
            return 1.0 + (FAR_DETAIL_MAX_MULTIPLIER - 1.0) * eased;
        }

        private Projection project(HierarchyView.Node node, Matrix4fc matrix) {
            double size = 32.0 * (1L << node.level());
            double baseX = (double) WorldEngine.getX(node.packedPosition()) * size - this.parameters.cameraX;
            double baseY = (double) WorldEngine.getY(node.packedPosition()) * size - this.parameters.cameraY;
            double baseZ = (double) WorldEngine.getZ(node.packedPosition()) * size - this.parameters.cameraZ;

            double[] x = this.clipX;
            double[] y = this.clipY;
            double[] z = this.clipZ;
            double[] w = this.clipW;
            Matrix4fc m = matrix;
            for (int i = 0; i < 8; i++) {
                double px = baseX + (((i & 1) != 0) ? size : 0.0);
                double py = baseY + (((i & 2) != 0) ? size : 0.0);
                double pz = baseZ + (((i & 4) != 0) ? size : 0.0);
                x[i] = m.m00() * px + m.m10() * py + m.m20() * pz + m.m30();
                y[i] = m.m01() * px + m.m11() * py + m.m21() * pz + m.m31();
                z[i] = m.m02() * px + m.m12() * py + m.m22() * pz + m.m32();
                w[i] = m.m03() * px + m.m13() * py + m.m23() * pz + m.m33();
            }

            // This projection is only used for zoom refinement eligibility. Vanilla's finite
            // far plane must not reject the distant terrain that zoom is intended to inspect.
            if (allOutside(x, w, -1) || allOutside(x, w, 1)
                    || allOutside(y, w, -1) || allOutside(y, w, 1)) {
                return Projection.CULLED;
            }

            for (double clipW : w) {
                if (clipW <= CLIP_EPSILON) return Projection.FULL_SCREEN;
            }

            double[] sx = this.screenX;
            double[] sy = this.screenY;
            for (int i = 0; i < 8; i++) {
                sx[i] = (x[i] / w[i]) * 0.5 + 0.5;
                sy[i] = (y[i] / w[i]) * 0.5 + 0.5;
            }

            // Same six projected face parallelograms used by screenspace.glsl.
            double area = 0.0;
            area += cross(sx[1] - sx[0], sy[1] - sy[0], sx[2] - sx[0], sy[2] - sy[0]);
            area += cross(sx[1] - sx[0], sy[1] - sy[0], sx[4] - sx[0], sy[4] - sy[0]);
            area += cross(sx[4] - sx[0], sy[4] - sy[0], sx[2] - sx[0], sy[2] - sy[0]);
            area += cross(sx[6] - sx[7], sy[6] - sy[7], sx[5] - sx[7], sy[5] - sy[7]);
            area += cross(sx[6] - sx[7], sy[6] - sy[7], sx[3] - sx[7], sy[3] - sy[7]);
            area += cross(sx[3] - sx[7], sy[3] - sy[7], sx[5] - sx[7], sy[5] - sy[7]);
            area *= 0.5;
            return new Projection(false,
                    area * this.parameters.viewportWidth * (double) this.parameters.viewportHeight);
        }

        private Projection projectOmnidirectional(HierarchyView.Node node) {
            double size = 32.0 * (1L << node.level());
            double minX = (double) WorldEngine.getX(node.packedPosition()) * size;
            double minY = (double) WorldEngine.getY(node.packedPosition()) * size;
            double minZ = (double) WorldEngine.getZ(node.packedPosition()) * size;
            double dx = distanceToInterval(this.parameters.cameraX, minX, minX + size);
            double dy = distanceToInterval(this.parameters.cameraY, minY, minY + size);
            double dz = distanceToInterval(this.parameters.cameraZ, minZ, minZ + size);
            double distance = Math.sqrt(dx * dx + dy * dy + dz * dz);
            if (distance <= CLIP_EPSILON) return Projection.FULL_SCREEN;

            // Use the widest tracked FOV for the persistent spatial cut. Zoom changes drawing,
            // not the resident geometry, so zooming out cannot discard a completed fine cut.
            // Recover the rotation-independent focal length once per traversal, not per node.
            double pixelsPerBlock = this.coveragePixels / distance;
            double apparentSide = size * pixelsPerBlock;
            return new Projection(false, apparentSide * apparentSide);
        }

        private static boolean allOutside(double[] coordinate, double[] w, int side) {
            for (int i = 0; i < 8; i++) {
                boolean outside = side < 0 ? coordinate[i] < -w[i] : coordinate[i] > w[i];
                if (!outside) return false;
            }
            return true;
        }

        private static double focalScale(Matrix4fc matrix) {
            return Math.sqrt((double) matrix.m01() * matrix.m01()
                    + (double) matrix.m11() * matrix.m11()
                    + (double) matrix.m21() * matrix.m21());
        }

        private static boolean allOutsideNear(double[] z) {
            for (double value : z) if (value >= 0.0) return false;
            return true;
        }

        private static double cross(double ax, double ay, double bx, double by) {
            return Math.abs(ax * by - ay * bx);
        }

        private static double distanceToInterval(double point, double min, double max) {
            return point < min ? min - point : point > max ? point - max : 0.0;
        }
    }

    private record Projection(boolean outsideFrustum, double pixelArea) {
        private static final Projection CULLED = new Projection(true, 0.0);
        private static final Projection FULL_SCREEN = new Projection(false, Double.POSITIVE_INFINITY);
    }
}

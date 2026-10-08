package me.cortex.voxy.client.core.rendering.compat;

import me.cortex.voxy.common.world.WorldEngine;
import org.joml.Matrix4fc;

import java.util.ArrayList;
import java.util.BitSet;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;

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
     * Keep a complete coarse cut around the camera while only refining the current view at the
     * normal screen-space threshold.  Vitrail replaces its provider aggregate atomically, so
     * frustum-culling this base cut would otherwise make terrain disappear as the camera turns.
     */
    private static final double OFFSCREEN_SUBDIVISION_MULTIPLIER = 8.0;
    /**
     * Native Voxy only retains the current frustum and lets Hi-Z remove more of that cut.  The
     * bridge deliberately keeps an omnidirectional fallback because its provider aggregate is
     * replaced asynchronously.  Compensate for that larger working set by spending progressively
     * fewer child nodes on the far part of the render distance.
     */
    private static final double FAR_DETAIL_START_FRACTION = 0.20;
    private static final double FAR_DETAIL_MAX_MULTIPLIER = 4.0;

    private static final int RETAINED_DETAIL_FRAMES = 360;
    private final Map<Integer, Integer> lastViewedRefinementFrame = new HashMap<>();
    private int frame;
    private long historyEpoch = Long.MIN_VALUE;

    public Result traverse(HierarchyView hierarchy, Parameters parameters) {
        long start = System.nanoTime();
        if (hierarchy.epoch() != this.historyEpoch) {
            this.lastViewedRefinementFrame.clear();
            this.frame = 0;
            this.historyEpoch = hierarchy.epoch();
        }
        this.frame++;

        State state = new State(hierarchy, parameters, this.lastViewedRefinementFrame, this.frame);
        for (int rootId : hierarchy.topLevelNodeIds()) {
            state.visit(rootId);
        }
        // Keep recently viewed detail long enough for the asynchronous provider build and a quick
        // look back, then let it return to the cheaper omnidirectional cut. Permanently retaining
        // every direction visited grows the cut and rebuild cost for the lifetime of the world.
        for (int nodeId = state.viewedRefinements.nextSetBit(0); nodeId >= 0;
                nodeId = state.viewedRefinements.nextSetBit(nodeId + 1)) {
            this.lastViewedRefinementFrame.put(nodeId, this.frame);
        }
        if ((this.frame & 31) == 0) {
            this.lastViewedRefinementFrame.entrySet().removeIf(
                    entry -> this.frame - entry.getValue() > RETAINED_DETAIL_FRAMES);
        }
        state.parentChildOverlap += state.countSelectedParentChildOverlaps();

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
        private final Map<Integer, Integer> lastViewedRefinementFrame;
        private final int frame;
        private final BitSet viewedRefinements = new BitSet();
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
                Map<Integer, Integer> lastViewedRefinementFrame, int frame) {
            this.hierarchy = hierarchy;
            this.parameters = parameters;
            this.lastViewedRefinementFrame = lastViewedRefinementFrame;
            this.frame = frame;
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

            Projection detailProjection = project(node, this.parameters.viewProjection);
            Projection coverageProjection = this.parameters.coverageViewProjection == this.parameters.viewProjection
                    ? detailProjection : project(node, this.parameters.coverageViewProjection);
            boolean insideView = !coverageProjection.outsideFrustum;
            // A narrow FOV may request finer nodes in its centre. Outside the widest current view,
            // use a direction-independent projected size so the aggregate still contains a coarse
            // hierarchy cut in every direction. Turning can refine that cut but cannot erase it.
            Projection projection = !detailProjection.outsideFrustum
                    ? detailProjection
                    : !coverageProjection.outsideFrustum
                    ? coverageProjection
                    : projectOmnidirectional(node);

            Integer lastViewedFrame = this.lastViewedRefinementFrame.get(nodeId);
            boolean retainedDetail = lastViewedFrame != null
                    && this.frame - lastViewedFrame <= RETAINED_DETAIL_FRAMES;
            double threshold = this.parameters.subdivisionPixels
                    * (insideView || retainedDetail ? 1.0 : OFFSCREEN_SUBDIVISION_MULTIPLIER)
                    * distanceDetailMultiplier(node);
            if (retainedDetail) threshold *= EXIT_HYSTERESIS;
            boolean refine = node.level() != 0
                    && projection.pixelArea > threshold * threshold;
            if (refine) {
                if (insideView) this.viewedRefinements.set(nodeId);
            }

            if (refine) {
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

        private int countSelectedParentChildOverlaps() {
            HashSet<Long> selectedPositions = new HashSet<>(this.selectedNodeIds.size() * 2);
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

            if (allOutside(x, w, -1) || allOutside(x, w, 1)
                    || allOutside(y, w, -1) || allOutside(y, w, 1)
                    || allOutsideNear(z) || allOutside(z, w, 1)) {
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

            // m11 is cot(fovY/2). Use the widest tracked projection so zoom cannot inflate the
            // off-screen coverage budget. Squaring the apparent side gives a stable area metric
            // compatible with the normal projected-AABB subdivision test.
            double pixelsPerBlock = Math.abs(this.parameters.coverageViewProjection.m11())
                    * this.parameters.viewportHeight * 0.5 / distance;
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

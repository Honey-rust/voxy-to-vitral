package me.cortex.voxy.client.core.rendering.hierachical;

import me.cortex.voxy.common.world.WorldEngine;
import org.jetbrains.annotations.Nullable;
import org.joml.FrustumIntersection;
import org.joml.Matrix4fc;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/** CPU fallback for selecting a non-overlapping Voxy LOD cut when OpenGL compute is unavailable. */
public final class VitrailLodSelector {
    private static final double REFINE_DISTANCE_IN_NODE_WIDTHS = 4.0;
    private VitrailLodSelector() {}

    /** Selects drawable nodes by distance and hierarchy; returned ids refer to Voxy's geometry store. */
    public static List<NodeManager.GeometryNode> select(
            List<NodeManager.GeometryNode> nodes,
            double cameraX, double cameraZ,
            double minimumDistanceBlocks,
            double maximumDistanceBlocks) {
        return select(nodes, cameraX, cameraZ, minimumDistanceBlocks, maximumDistanceBlocks, 0);
    }

    /**
     * Selects a cut no finer than {@code minimumLevel}. This lets the Vulkan fallback publish a
     * complete coarse landscape first, then replace it with successively finer cuts.
     */
    public static List<NodeManager.GeometryNode> select(
            List<NodeManager.GeometryNode> nodes,
            double cameraX, double cameraZ,
            double minimumDistanceBlocks,
            double maximumDistanceBlocks,
            int minimumLevel) {
        return select(nodes, cameraX, 0.0, cameraZ, minimumDistanceBlocks, maximumDistanceBlocks,
                minimumLevel, 0.0, 0.0, null);
    }

    /**
     * CPU equivalent of Voxy's screen-space subdivision decision. Projection scale is the number
     * of vertical screen pixels represented by a one-radian tangent; passing zero retains the old
     * distance fallback for callers which do not have a render projection.
     */
    public static List<NodeManager.GeometryNode> select(
            List<NodeManager.GeometryNode> nodes,
            double cameraX, double cameraY, double cameraZ,
            double minimumDistanceBlocks,
            double maximumDistanceBlocks,
            int minimumLevel,
            double projectionScalePixels,
            double subdivisionPixels,
            @Nullable Matrix4fc viewProjection) {
        FrustumIntersection refinementFrustum = viewProjection == null
                ? null : new FrustumIntersection(viewProjection, false);
        Map<Long, NodeManager.GeometryNode> byPosition = new HashMap<>(nodes.size() * 2);
        Map<Long, List<NodeManager.GeometryNode>> children = new HashMap<>();
        for (NodeManager.GeometryNode node : nodes) byPosition.put(node.position(), node);
        for (NodeManager.GeometryNode node : nodes) {
            if (node.level() == WorldEngine.MAX_LOD_LAYER) continue;
            long parent = parentPosition(node.position());
            if (byPosition.containsKey(parent)) {
                children.computeIfAbsent(parent, ignored -> new ArrayList<>(8)).add(node);
            }
        }

        ArrayList<NodeManager.GeometryNode> selected = new ArrayList<>();
        for (NodeManager.GeometryNode node : nodes) {
            boolean root = node.level() == WorldEngine.MAX_LOD_LAYER
                    || !byPosition.containsKey(parentPosition(node.position()));
            if (root) {
                selectNode(node, children, selected, cameraX, cameraY, cameraZ,
                        minimumDistanceBlocks, maximumDistanceBlocks, minimumLevel,
                        projectionScalePixels, subdivisionPixels, refinementFrustum);
            }
        }
        return List.copyOf(selected);
    }

    /** Finds the currently visible coarse cut that needs child geometry from Voxy. */
    public static List<Long> refinementRequests(
            List<NodeManager.GeometryNode> nodes,
            double cameraX, double cameraZ,
            double maximumDistanceBlocks) {
        return refinementRequests(nodes, cameraX, 0.0, cameraZ, 0.0, maximumDistanceBlocks,
                0.0, 0.0, null);
    }

    public static List<Long> refinementRequests(
            List<NodeManager.GeometryNode> nodes,
            double cameraX, double cameraY, double cameraZ,
            double minimumDistanceBlocks,
            double maximumDistanceBlocks,
            double projectionScalePixels,
            double subdivisionPixels,
            @Nullable Matrix4fc viewProjection) {
        FrustumIntersection refinementFrustum = viewProjection == null
                ? null : new FrustumIntersection(viewProjection, false);
        ArrayList<NodeManager.GeometryNode> candidates = new ArrayList<>();
        for (NodeManager.GeometryNode node : select(
                nodes, cameraX, cameraY, cameraZ, minimumDistanceBlocks, maximumDistanceBlocks,
                0, projectionScalePixels, subdivisionPixels, viewProjection)) {
            if (node.level() > 0 && !node.inner() && !node.requestInFlight()
                    && !whollyInsideNearRange(node, cameraX, cameraZ, minimumDistanceBlocks)
                    && shouldRefine(node, cameraX, cameraY, cameraZ,
                            minimumDistanceBlocks, projectionScalePixels, subdivisionPixels,
                            refinementFrustum)) {
                candidates.add(node);
            }
        }
        candidates.sort(Comparator
                .comparing((NodeManager.GeometryNode node) ->
                        !crossesNearBoundary(node, cameraX, cameraZ, minimumDistanceBlocks))
                .thenComparingDouble((NodeManager.GeometryNode node) -> nodeDistance(node, cameraX, cameraZ))
                .thenComparing(Comparator.comparingInt(NodeManager.GeometryNode::level).reversed()));
        ArrayList<Long> positions = new ArrayList<>(Math.min(64, candidates.size()));
        for (NodeManager.GeometryNode node : candidates) {
            if (positions.size() == 64) break;
            positions.add(node.position());
        }
        return List.copyOf(positions);
    }

    private static void selectNode(NodeManager.GeometryNode node,
            Map<Long, List<NodeManager.GeometryNode>> children,
            List<NodeManager.GeometryNode> output,
            double cameraX, double cameraY, double cameraZ,
            double minimumDistanceBlocks, double maximumDistanceBlocks,
            int minimumLevel, double projectionScalePixels, double subdivisionPixels,
            @Nullable FrustumIntersection refinementFrustum) {
        double size = 32.0 * (1L << node.level());
        double minX = (double) WorldEngine.getX(node.position()) * size;
        double minZ = (double) WorldEngine.getZ(node.position()) * size;
        double dx = distanceToInterval(cameraX, minX, minX + size);
        double dz = distanceToInterval(cameraZ, minZ, minZ + size);
        double distance = Math.sqrt(dx * dx + dz * dz);
        if (maximumDistanceBlocks >= 0 && distance > maximumDistanceBlocks) return;

        List<NodeManager.GeometryNode> descendants = children.get(node.position());
        boolean hasChildren = descendants != null && !descendants.isEmpty();
        int expectedChildren = Byte.toUnsignedInt(node.childExistence());
        int presentChildren = 0;
        if (hasChildren) {
            for (NodeManager.GeometryNode child : descendants) {
                presentChildren |= 1 << childIndex(child.position());
            }
        }
        boolean childrenComplete = (presentChildren & expectedChildren) == expectedChildren;
        boolean refine = hasChildren && (node.geometryId() < 0
                || (childrenComplete && shouldRefineForScreen(node, cameraX, cameraY, cameraZ,
                        projectionScalePixels, subdivisionPixels, refinementFrustum)));
        double farX = Math.max(Math.abs(cameraX - minX), Math.abs(cameraX - (minX + size)));
        double farZ = Math.max(Math.abs(cameraZ - minZ), Math.abs(cameraZ - (minZ + size)));
        double farDistance = Math.hypot(farX, farZ);
        if (minimumDistanceBlocks > 0 && farDistance < minimumDistanceBlocks) {
            // The Vulkan compatibility path has no Voxy Hi-Z buffer with which to hide a coarse
            // proxy behind vanilla chunks. Exclude complete inner nodes instead; a per-fragment
            // near cut clips the part of boundary-crossing parents which lies inside this radius.
            return;
        }
        boolean crossesNearBoundary = minimumDistanceBlocks > 0
                && distance < minimumDistanceBlocks && farDistance >= minimumDistanceBlocks;
        boolean seamVisible = crossesNearBoundary
                && inRefinementView(node, cameraX, cameraY, cameraZ, refinementFrustum);
        if (seamVisible && hasChildren && childrenComplete) {
            refine = true;
        }
        // A node crossing the vanilla/LOD seam must descend even during an early coarse streaming
        // stage. Shader packs discard the near fragments independently; leaving a coarse node here
        // can expose its farther cave faces after its surface fragments have been discarded.
        if (!seamVisible && node.geometryId() >= 0 && node.level() <= minimumLevel) {
            refine = false;
        }
        if (hasChildren && !childrenComplete && node.geometryId() >= 0) {
            // Keep the parent's complete mesh until every existing child has arrived. Refining
            // into only the children already loaded would leave visible holes during streaming.
            output.add(node);
            return;
        }
        if (refine) {
            for (NodeManager.GeometryNode child : descendants) {
                selectNode(child, children, output, cameraX, cameraY, cameraZ,
                        minimumDistanceBlocks, maximumDistanceBlocks, minimumLevel,
                        projectionScalePixels, subdivisionPixels, refinementFrustum);
            }
        } else if (node.geometryId() >= 0) {
            output.add(node);
        }
    }

    private static double distanceToInterval(double point, double min, double max) {
        return point < min ? min - point : point > max ? point - max : 0.0;
    }

    private static double nodeDistance(NodeManager.GeometryNode node, double cameraX, double cameraZ) {
        double size = 32.0 * (1L << node.level());
        double minX = (double) WorldEngine.getX(node.position()) * size;
        double minZ = (double) WorldEngine.getZ(node.position()) * size;
        return Math.hypot(distanceToInterval(cameraX, minX, minX + size),
                distanceToInterval(cameraZ, minZ, minZ + size));
    }

    private static boolean shouldRefine(NodeManager.GeometryNode node,
            double cameraX, double cameraY, double cameraZ,
            double minimumDistanceBlocks,
            double projectionScalePixels, double subdivisionPixels,
            @Nullable FrustumIntersection refinementFrustum) {
        if (crossesNearBoundary(node, cameraX, cameraZ, minimumDistanceBlocks)
                && inRefinementView(node, cameraX, cameraY, cameraZ, refinementFrustum)) return true;
        return shouldRefineForScreen(node, cameraX, cameraY, cameraZ,
                projectionScalePixels, subdivisionPixels, refinementFrustum);
    }

    private static boolean shouldRefineForScreen(NodeManager.GeometryNode node,
            double cameraX, double cameraY, double cameraZ,
            double projectionScalePixels, double subdivisionPixels,
            @Nullable FrustumIntersection refinementFrustum) {
        double size = 32.0 * (1L << node.level());
        if (!(projectionScalePixels > 0.0) || !(subdivisionPixels > 0.0)) {
            return nodeDistance(node, cameraX, cameraZ) < size * REFINE_DISTANCE_IN_NODE_WIDTHS;
        }
        if (!inRefinementView(node, cameraX, cameraY, cameraZ, refinementFrustum)) return false;
        double minX = (double) WorldEngine.getX(node.position()) * size;
        double minY = (double) WorldEngine.getY(node.position()) * size;
        double minZ = (double) WorldEngine.getZ(node.position()) * size;
        double dx = distanceToInterval(cameraX, minX, minX + size);
        double dy = distanceToInterval(cameraY, minY, minY + size);
        double dz = distanceToInterval(cameraZ, minZ, minZ + size);
        double distance = Math.sqrt(dx * dx + dy * dy + dz * dz);
        double projectedWidthPixels = size * projectionScalePixels / Math.max(size, distance);
        return projectedWidthPixels > subdivisionPixels;
    }

    /**
     * Original Voxy applies screen-size subdivision after its frustum test. Keep a half-node guard
     * around that frustum so a slow aggregate rebuild does not reveal coarse terrain right at the
     * edge while the player turns.
     */
    private static boolean inRefinementView(NodeManager.GeometryNode node,
            double cameraX, double cameraY, double cameraZ,
            @Nullable FrustumIntersection frustum) {
        if (frustum == null) return true;
        double size = 32.0 * (1L << node.level());
        double pad = size * 0.5;
        double minX = (double) WorldEngine.getX(node.position()) * size - cameraX - pad;
        double minY = (double) WorldEngine.getY(node.position()) * size - cameraY - pad;
        double minZ = (double) WorldEngine.getZ(node.position()) * size - cameraZ - pad;
        return frustum.testAab((float) minX, (float) minY, (float) minZ,
                (float) (minX + size + pad * 2.0),
                (float) (minY + size + pad * 2.0),
                (float) (minZ + size + pad * 2.0));
    }

    private static boolean crossesNearBoundary(NodeManager.GeometryNode node,
            double cameraX, double cameraZ, double minimumDistanceBlocks) {
        if (!(minimumDistanceBlocks > 0.0)) return false;
        double size = 32.0 * (1L << node.level());
        double minX = (double) WorldEngine.getX(node.position()) * size;
        double minZ = (double) WorldEngine.getZ(node.position()) * size;
        double nearDistance = Math.hypot(distanceToInterval(cameraX, minX, minX + size),
                distanceToInterval(cameraZ, minZ, minZ + size));
        double farX = Math.max(Math.abs(cameraX - minX), Math.abs(cameraX - (minX + size)));
        double farZ = Math.max(Math.abs(cameraZ - minZ), Math.abs(cameraZ - (minZ + size)));
        return nearDistance < minimumDistanceBlocks
                && Math.hypot(farX, farZ) >= minimumDistanceBlocks;
    }

    private static boolean whollyInsideNearRange(NodeManager.GeometryNode node,
            double cameraX, double cameraZ, double minimumDistanceBlocks) {
        if (!(minimumDistanceBlocks > 0.0)) return false;
        double size = 32.0 * (1L << node.level());
        double minX = (double) WorldEngine.getX(node.position()) * size;
        double minZ = (double) WorldEngine.getZ(node.position()) * size;
        double farX = Math.max(Math.abs(cameraX - minX), Math.abs(cameraX - (minX + size)));
        double farZ = Math.max(Math.abs(cameraZ - minZ), Math.abs(cameraZ - (minZ + size)));
        return Math.hypot(farX, farZ) < minimumDistanceBlocks;
    }

    private static long parentPosition(long position) {
        int level = WorldEngine.getLevel(position);
        return WorldEngine.getWorldSectionId(level + 1,
                WorldEngine.getX(position) >> 1,
                WorldEngine.getY(position) >> 1,
                WorldEngine.getZ(position) >> 1);
    }

    private static int childIndex(long position) {
        return (WorldEngine.getX(position) & 1)
                | ((WorldEngine.getY(position) & 1) << 2)
                | ((WorldEngine.getZ(position) & 1) << 1);
    }
}

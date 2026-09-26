package me.cortex.voxy.client.core.rendering.hierachical;

import me.cortex.voxy.common.world.WorldEngine;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/** CPU fallback for selecting a non-overlapping Voxy LOD cut when OpenGL compute is unavailable. */
public final class VitrailLodSelector {
    private static final double REFINE_DISTANCE_IN_NODE_WIDTHS = 6.0;

    private VitrailLodSelector() {}

    /** Selects drawable nodes by distance and hierarchy; returned ids refer to Voxy's geometry store. */
    public static List<NodeManager.GeometryNode> select(
            List<NodeManager.GeometryNode> nodes,
            double cameraX, double cameraZ,
            double minimumDistanceBlocks,
            double maximumDistanceBlocks) {
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
                selectNode(node, children, selected, cameraX, cameraZ,
                        minimumDistanceBlocks, maximumDistanceBlocks);
            }
        }
        return List.copyOf(selected);
    }

    private static void selectNode(NodeManager.GeometryNode node,
            Map<Long, List<NodeManager.GeometryNode>> children,
            List<NodeManager.GeometryNode> output,
            double cameraX, double cameraZ,
            double minimumDistanceBlocks, double maximumDistanceBlocks) {
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
                || (childrenComplete && distance < size * REFINE_DISTANCE_IN_NODE_WIDTHS));
        // Exclude only nodes wholly covered by the near range. Testing the nearest
        // corner discards boundary tiles as well and opens a ring of missing terrain.
        double farX = Math.max(Math.abs(cameraX - minX), Math.abs(cameraX - (minX + size)));
        double farZ = Math.max(Math.abs(cameraZ - minZ), Math.abs(cameraZ - (minZ + size)));
        if (minimumDistanceBlocks > 0 && Math.hypot(farX, farZ) < minimumDistanceBlocks) return;
        if (minimumDistanceBlocks > 0 && distance < minimumDistanceBlocks && hasChildren && childrenComplete) {
            refine = true;
        }
        if (hasChildren && !childrenComplete && node.geometryId() >= 0) {
            // Keep the parent's complete mesh until every existing child has arrived. Refining
            // into only the children already loaded would leave visible holes during streaming.
            output.add(node);
            return;
        }
        if (refine) {
            for (NodeManager.GeometryNode child : descendants) {
                selectNode(child, children, output, cameraX, cameraZ,
                        minimumDistanceBlocks, maximumDistanceBlocks);
            }
        } else if (node.geometryId() >= 0) {
            output.add(node);
        }
    }

    private static double distanceToInterval(double point, double min, double max) {
        return point < min ? min - point : point > max ? point - max : 0.0;
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

package me.cortex.voxy.client.core.rendering.compat;

import java.util.Arrays;

/**
 * Immutable, renderer-neutral snapshot of Voxy's live hierarchy.
 *
 * <p>Node ids are stable for the lifetime of {@link #epoch()}. A new epoch means that consumers
 * must discard every cached id. Generation identifies one published snapshot within that epoch.</p>
 */
public final class HierarchyView {
    public static final int NULL_GEOMETRY_ID = -1;
    public static final int EMPTY_GEOMETRY_ID = -2;
    public static final int NULL_CHILD_POINTER = -1;
    public static final int EMPTY_CHILD_POINTER = -2;

    public static final int FLAG_INNER = 1;
    public static final int FLAG_REQUEST_IN_FLIGHT = 1 << 1;
    public static final int FLAG_GEOMETRY_IN_FLIGHT = 1 << 2;
    public static final int FLAG_ALL_CHILDREN_LEAF = 1 << 3;

    public static final HierarchyView EMPTY = new HierarchyView(0, 0, new int[0], new Node[0], 0);

    private final long epoch;
    private final long generation;
    private final int[] topLevelNodeIds;
    private final Node[] nodesById;
    private final int nodeCount;

    public HierarchyView(long epoch, long generation, int[] topLevelNodeIds,
                         Node[] nodesById, int nodeCount) {
        this.epoch = epoch;
        this.generation = generation;
        this.topLevelNodeIds = topLevelNodeIds.clone();
        this.nodesById = nodesById.clone();
        this.nodeCount = nodeCount;
    }

    public long epoch() { return this.epoch; }
    public long generation() { return this.generation; }
    public int nodeCount() { return this.nodeCount; }
    public int capacity() { return this.nodesById.length; }
    public int[] topLevelNodeIds() { return this.topLevelNodeIds.clone(); }

    public Node node(int stableNodeId) {
        return stableNodeId >= 0 && stableNodeId < this.nodesById.length
                ? this.nodesById[stableNodeId] : null;
    }

    @Override
    public String toString() {
        return "HierarchyView[epoch=" + this.epoch + ", generation=" + this.generation
                + ", roots=" + Arrays.toString(this.topLevelNodeIds)
                + ", nodes=" + this.nodeCount + ']';
    }

    /** Position remains in Voxy's packed world-section format and has no renderer dependency. */
    public record Node(
            int stableNodeId,
            long packedPosition,
            int level,
            int childPointer,
            int childCount,
            byte childMask,
            int flags,
            int geometryId,
            long geometryVersion,
            int requestId) {
        public boolean inner() { return (this.flags & FLAG_INNER) != 0; }
        public boolean requestInFlight() { return (this.flags & FLAG_REQUEST_IN_FLIGHT) != 0; }
        public boolean geometryInFlight() { return (this.flags & FLAG_GEOMETRY_IN_FLIGHT) != 0; }
        public boolean allChildrenLeaf() { return (this.flags & FLAG_ALL_CHILDREN_LEAF) != 0; }
        public boolean hasGeometry() { return this.geometryId != NULL_GEOMETRY_ID; }
        public boolean hasDrawableGeometry() { return this.geometryId >= 0; }
        public boolean emptyGeometry() { return this.geometryId == EMPTY_GEOMETRY_ID; }
        public boolean hasChildren() { return this.childPointer != NULL_CHILD_POINTER; }
        public boolean childListEmpty() { return this.childPointer == EMPTY_CHILD_POINTER; }
    }
}

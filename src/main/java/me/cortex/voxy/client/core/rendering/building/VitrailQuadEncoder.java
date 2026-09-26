package me.cortex.voxy.client.core.rendering.building;

import me.cortex.voxy.common.world.WorldEngine;

/** CPU expansion of Voxy's packed 64-bit LOD quad into Vitrail's distant-mesh vertices. */
public final class VitrailQuadEncoder {
    public static final int TILE_BLOCKS = 4096;
    private static final double EPSILON = 0.00005;

    private VitrailQuadEncoder() {}

    public record Vertex(int x, int y, int z, int light, int colour, int material, int normal) {}

    @FunctionalInterface
    public interface QuadConsumer {
        /** Receives four vertices belonging to the same 4096-block-aligned Vitrail section. */
        void accept(int sectionX, int sectionY, int sectionZ, Vertex v0, Vertex v1,
                    Vertex v2, Vertex v3, boolean reverseWinding);
    }

    /**
     * Expands one packed quad. Positions and dimensions mirror {@code lod/quad_util.glsl}; model
     * face data is the packed uint from Voxy's 64-byte model record. Large far quads are split at
     * Vitrail's section-coordinate tile boundaries, preserving the original plane and attributes.
     *
     * @param packed the eight-byte value emitted by Voxy's section mesher
     * @param nodePosition Voxy world-section id owning the quad
     * @param lodLevel level of that world section
     * @param faceData packed model face record for the quad's face
     * @param colour packed RGBA vertex tint; use {@code 0xFFFFFFFF} when no CPU tint is available
     * @param material Vitrail distant material id
     * @param output receives one or more tiled quads
     */
    public static void encode(long packed, long nodePosition, int lodLevel, int faceData,
            int colour, int material, QuadConsumer output) {
        if (lodLevel < 0 || lodLevel > 30) {
            throw new IllegalArgumentException("Invalid LOD level: " + lodLevel);
        }
        if (output == null) throw new NullPointerException("output");

        int face = (int) (packed & 7L);
        int axis = face >>> 1;
        int side = face & 1;
        int quadWidth = (int) ((packed >>> 3) & 15L) + 1;
        int quadHeight = (int) ((packed >>> 7) & 15L) + 1;
        double px = (packed >>> 21) & 31L;
        double py = (packed >>> 16) & 31L;
        double pz = (packed >>> 11) & 31L;

        double offsetX = (faceData & 15) / 16.0 - EPSILON;
        double sizeX = ((faceData >>> 4) & 15) / 16.0 + 1.0 / 16.0;
        double offsetZ = ((faceData >>> 8) & 15) / 16.0 - EPSILON;
        double sizeZ = ((faceData >>> 12) & 15) / 16.0 + 1.0 / 16.0;
        sizeX -= offsetX;
        sizeZ -= offsetZ;

        int encodedDepth = (faceData >>> 16) & 63;
        if (encodedDepth == 63) encodedDepth++;
        double depth = encodedDepth / 64.0;
        if (side == 1) depth = 1.0 - depth;

        // quad_util.glsl supplies vec3(faceSize.xz, depth): both in-plane offsets
        // precede the perpendicular depth before the axis swizzle.
        double[] start = swizzle(axis, offsetX, offsetZ, depth);
        long scale = 1L << lodLevel;
        double worldX = (double) WorldEngine.getX(nodePosition) * 32.0 * scale;
        double worldY = (double) WorldEngine.getY(nodePosition) * 32.0 * scale;
        double worldZ = (double) WorldEngine.getZ(nodePosition) * 32.0 * scale;
        double[] base = {worldX + (px + start[0]) * scale,
                worldY + (py + start[1]) * scale,
                worldZ + (pz + start[2]) * scale};

        // The two in-plane world axes in Voxy's swizzle order.
        // swizzle(axis, x, y, z) maps the packed in-plane extents to world axes.
        int firstAxis = axis == 2 ? 1 : 0;
        int secondAxis = axis == 1 ? 1 : 2;
        double firstExtent = (sizeX + quadWidth - 1.0) * scale;
        double secondExtent = (sizeZ + quadHeight - 1.0) * scale;

        int distantFace = switch (face) {
            case 0 -> 0; // -Y: DOWN
            case 1 -> 1; // +Y: UP
            case 2 -> 2; // -Z: NORTH
            case 3 -> 3; // +Z: SOUTH
            case 4 -> 4; // -X: WEST
            case 5 -> 5; // +X: EAST
            default -> throw new IllegalArgumentException("Invalid packed face: " + face);
        };
        int light = (int) ((packed >>> 55) & 0xffL);
        // Winding agrees with Vitrail's DOWN, UP, NORTH, SOUTH, WEST, EAST face convention.
        boolean reverse = face == 1 || face == 2 || face == 4;

        split(base, firstAxis, secondAxis, firstExtent, secondExtent,
                light, colour, material, distantFace, reverse, output);
    }

    private static void split(double[] base, int firstAxis, int secondAxis,
            double firstExtent, double secondExtent, int light, int colour, int material,
            int normal, boolean reverse, QuadConsumer output) {
        double firstAt = 0;
        while (firstAt < firstExtent - 1.0e-7) {
            double[] rowBase = base.clone();
            rowBase[firstAxis] += firstAt;
            double firstLength = Math.min(firstExtent - firstAt,
                    nextTileBoundary(rowBase[firstAxis]) - rowBase[firstAxis]);

            double secondAt = 0;
            while (secondAt < secondExtent - 1.0e-7) {
                double[] start = rowBase.clone();
                start[secondAxis] += secondAt;
                double secondLength = Math.min(secondExtent - secondAt,
                        nextTileBoundary(start[secondAxis]) - start[secondAxis]);

                int sectionX = Math.floorDiv((int) Math.floor(start[0]), TILE_BLOCKS) * TILE_BLOCKS;
                int sectionY = Math.floorDiv((int) Math.floor(start[1]), TILE_BLOCKS) * TILE_BLOCKS;
                int sectionZ = Math.floorDiv((int) Math.floor(start[2]), TILE_BLOCKS) * TILE_BLOCKS;
                double[] p0 = start;
                double[] p1 = start.clone(); p1[firstAxis] += firstLength;
                double[] p2 = start.clone(); p2[secondAxis] += secondLength;
                double[] p3 = p1.clone(); p3[secondAxis] += secondLength;

                // The DH format rounds to whole blocks. Epsilon-expanded faces can create
                // a sub-block sliver at a tile boundary; do not submit collapsed triangles.
                if (Math.round(p0[firstAxis]) != Math.round(p1[firstAxis])
                        && Math.round(p0[secondAxis]) != Math.round(p2[secondAxis])) {
                    output.accept(sectionX, sectionY, sectionZ,
                            vertex(p0, sectionX, sectionY, sectionZ, light, colour, material, normal),
                            vertex(p1, sectionX, sectionY, sectionZ, light, colour, material, normal),
                            vertex(p2, sectionX, sectionY, sectionZ, light, colour, material, normal),
                            vertex(p3, sectionX, sectionY, sectionZ, light, colour, material, normal),
                            reverse);
                }
                secondAt += secondLength;
            }
            firstAt += firstLength;
        }
    }

    private static double nextTileBoundary(double coordinate) {
        return (Math.floor(coordinate / TILE_BLOCKS) + 1.0) * TILE_BLOCKS;
    }

    private static Vertex vertex(double[] point, int sectionX, int sectionY, int sectionZ,
            int light, int colour, int material, int normal) {
        return new Vertex((int) Math.round(point[0] - sectionX),
                (int) Math.round(point[1] - sectionY),
                (int) Math.round(point[2] - sectionZ), light, colour, material, normal);
    }

    private static double[] swizzle(int axis, double x, double y, double z) {
        return switch (axis) {
            case 0 -> new double[] {x, z, y}; // Voxy's xzy
            case 1 -> new double[] {x, y, z};
            case 2 -> new double[] {z, x, y}; // Voxy's zxy
            default -> throw new IllegalArgumentException("Invalid axis: " + axis);
        };
    }
}

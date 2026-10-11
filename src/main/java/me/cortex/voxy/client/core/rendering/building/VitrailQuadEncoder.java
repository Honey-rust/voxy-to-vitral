package me.cortex.voxy.client.core.rendering.building;

import me.cortex.voxy.common.world.WorldEngine;

/** CPU expansion of Voxy's packed 64-bit LOD quad into Vitrail's distant-mesh vertices. */
public final class VitrailQuadEncoder {
    public static final int OPAQUE_TILE_BLOCKS = 4096;
    public static final int TRANSLUCENT_TILE_BLOCKS = 256;
    private static final double EPSILON = 0.00005;

    private VitrailQuadEncoder() {}

    public record Vertex(int x, int y, int z, int light, int colour, int material, int normal, int fraction) {
        public Vertex(int x, int y, int z, int light, int colour, int material, int normal) {
            this(x,y,z,light,colour,material,normal,0);
        }
        public double exactX() { return x + (fraction & 63) / 64.0; }
        public double exactY() { return y + ((fraction >>> 6) & 63) / 64.0; }
        public double exactZ() { return z + ((fraction >>> 12) & 63) / 64.0; }
    }

    @FunctionalInterface
    public interface QuadConsumer {
        /** Receives four vertices belonging to one aligned Vitrail section. */
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
        encode(packed, nodePosition, lodLevel, faceData, colour, material, OPAQUE_TILE_BLOCKS, output);
    }

    public static void encode(long packed, long nodePosition, int lodLevel, int faceData,
            int colour, int material, int tileBlocks, QuadConsumer output) {
        if (lodLevel < 0 || lodLevel > 30) {
            throw new IllegalArgumentException("Invalid LOD level: " + lodLevel);
        }
        if (output == null) throw new NullPointerException("output");
        if (tileBlocks <= 0 || tileBlocks > 65535) throw new IllegalArgumentException("Invalid tile size: " + tileBlocks);

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
                light, colour, material, distantFace, reverse, tileBlocks, output);
    }

    private static void split(double[] base, int firstAxis, int secondAxis,
            double firstExtent, double secondExtent, int light, int colour, int material,
            int normal, boolean reverse, int tileBlocks, QuadConsumer output) {
        double firstAt = 0;
        while (firstAt < firstExtent - 1.0e-7) {
            double firstCoordinate = base[firstAxis] + firstAt;
            double firstLength = Math.min(firstExtent - firstAt,
                    nextTileBoundary(firstCoordinate, tileBlocks) - firstCoordinate);
            double secondAt = 0;
            while (secondAt < secondExtent - 1.0e-7) {
                double secondCoordinate = base[secondAxis] + secondAt;
                double secondLength = Math.min(secondExtent - secondAt,
                        nextTileBoundary(secondCoordinate, tileBlocks) - secondCoordinate);
                double x = base[0] + (firstAxis == 0 ? firstAt : secondAxis == 0 ? secondAt : 0);
                double y = base[1] + (firstAxis == 1 ? firstAt : secondAxis == 1 ? secondAt : 0);
                double z = base[2] + (firstAxis == 2 ? firstAt : secondAxis == 2 ? secondAt : 0);
                int sectionX = Math.floorDiv((int)Math.floor(x), tileBlocks) * tileBlocks;
                int sectionY = Math.floorDiv((int)Math.floor(y), tileBlocks) * tileBlocks;
                int sectionZ = Math.floorDiv((int)Math.floor(z), tileBlocks) * tileBlocks;
                if (Math.round(firstCoordinate * 64.0) != Math.round((firstCoordinate + firstLength) * 64.0)
                        && Math.round(secondCoordinate * 64.0) != Math.round((secondCoordinate + secondLength) * 64.0)) {
                    double dx1 = firstAxis == 0 ? firstLength : 0, dy1 = firstAxis == 1 ? firstLength : 0, dz1 = firstAxis == 2 ? firstLength : 0;
                    double dx2 = secondAxis == 0 ? secondLength : 0, dy2 = secondAxis == 1 ? secondLength : 0, dz2 = secondAxis == 2 ? secondLength : 0;
                    output.accept(sectionX, sectionY, sectionZ,
                            vertex(x,y,z,sectionX,sectionY,sectionZ,light,colour,material,normal),
                            vertex(x+dx1,y+dy1,z+dz1,sectionX,sectionY,sectionZ,light,colour,material,normal),
                            vertex(x+dx2,y+dy2,z+dz2,sectionX,sectionY,sectionZ,light,colour,material,normal),
                            vertex(x+dx1+dx2,y+dy1+dy2,z+dz1+dz2,sectionX,sectionY,sectionZ,light,colour,material,normal), reverse);
                }
                secondAt += secondLength;
            }
            firstAt += firstLength;
        }
    }

    private static double nextTileBoundary(double coordinate, int tileBlocks) {
        return (Math.floor(coordinate / tileBlocks) + 1.0) * tileBlocks;
    }

    private static Vertex vertex(double px, double py, double pz, int sectionX, int sectionY, int sectionZ,
            int light, int colour, int material, int normal) {
        long x = Math.round((px - sectionX) * 64.0);
        long y = Math.round((py - sectionY) * 64.0);
        long z = Math.round((pz - sectionZ) * 64.0);
        int fraction = (int)(x & 63) | ((int)(y & 63) << 6) | ((int)(z & 63) << 12);
        return new Vertex((int)(x >> 6), (int)(y >> 6), (int)(z >> 6),
                light, colour, material, normal, fraction);
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

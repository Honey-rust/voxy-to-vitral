package me.cortex.voxy.client.core.model.bakery;

/** Full-resolution RGBA8 atlas readback, split into tightly packed whole-row batches. */
record AtlasReadbackLayout(int width, int height, int pixels, long bytes, int rowsPerBatch) {
    private static final long MAX_BATCH_BYTES = 64L * 1024 * 1024;

    static AtlasReadbackLayout of(int width, int height) {
        if (width <= 0 || height <= 0) {
            throw new IllegalArgumentException("Invalid block atlas dimensions: " + width + "x" + height);
        }
        long pixels = Math.multiplyExact((long) width, height);
        long bytes = Math.multiplyExact(pixels, Integer.BYTES);
        // The software sampler uses a single int[]. Never truncate its pixel count.
        if (pixels > Integer.MAX_VALUE - 8L) {
            throw new IllegalArgumentException("Block atlas exceeds the software sampler's int[] capacity: "
                    + width + "x" + height + ", pixels=" + pixels + ", bytes=" + bytes);
        }
        long rowBytes = Math.multiplyExact((long) width, Integer.BYTES);
        if (rowBytes > MAX_BATCH_BYTES) {
            throw new IllegalArgumentException("Block atlas row exceeds readback batch capacity: " + rowBytes);
        }
        int rows = (int) Math.min(height, MAX_BATCH_BYTES / rowBytes);
        return new AtlasReadbackLayout(width, height, (int) pixels, bytes, rows);
    }

    long bufferBytes() {
        return (long) width * rowsPerBatch * Integer.BYTES;
    }

    int rowsAt(int y) {
        if (y < 0 || y >= height) throw new IndexOutOfBoundsException(y);
        return Math.min(rowsPerBatch, height - y);
    }
}

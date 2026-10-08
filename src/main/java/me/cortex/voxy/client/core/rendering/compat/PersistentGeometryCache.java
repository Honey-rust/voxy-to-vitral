package me.cortex.voxy.client.core.rendering.compat;

import me.cortex.voxy.client.core.rendering.building.VitrailCpuMeshEncoder;

import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Supplier;

/**
 * Retains Vitrail-ready CPU geometry across render-list rebuilds.
 *
 * <p>The cache owns every mesh stored in it. A build borrows meshes through a lease, which keeps
 * an entry alive while meshes are being combined or uploaded. This deliberately sits between
 * Voxy's geometry snapshot and the Vitrail adapter so a canceled or superseded render-list build
 * does not throw away completed encoding work.</p>
 */
public final class PersistentGeometryCache implements AutoCloseable {
    // One GiB thrashed between adjacent views, while two GiB filled immediately and pushed the
    // 16-GiB test machine into memory pressure.  Keep a modest amount above 1.5 GiB so completed
    // detail survives a quick look-away/look-back cycle without returning to the old 2-GiB peak.
    // The system property remains available for machines that need a different trade-off.
    private static final long DEFAULT_LIMIT_BYTES = 1792L * 1024L * 1024L;

    private final LinkedHashMap<Key, Entry> entries = new LinkedHashMap<>(256, 0.75F, true);
    private final long limitBytes;
    private long residentBytes;
    private long hits;
    private long misses;

    public PersistentGeometryCache() {
        long configuredMiB = Math.max(64L, Long.getLong("voxy.vitrail.geometryCacheMiB",
                DEFAULT_LIMIT_BYTES / (1024L * 1024L)));
        this.limitBytes = Math.multiplyExact(configuredMiB, 1024L * 1024L);
    }

    /** Returns null when geometry is not resident in Voxy yet. */
    public synchronized Lease acquire(Key key, Supplier<List<VitrailCpuMeshEncoder.Mesh>> encoder) {
        Entry entry = this.entries.get(key);
        if (entry != null) {
            entry.references++;
            this.hits++;
            return new Lease(this, entry, true);
        }

        List<VitrailCpuMeshEncoder.Mesh> encoded = encoder.get();
        if (encoded == null) return null;
        encoded = List.copyOf(encoded);
        entry = new Entry(key, encoded, byteSize(encoded));
        entry.references = 1;
        this.entries.put(key, entry);
        this.residentBytes += entry.bytes;
        this.misses++;
        invalidateOlderVersions(key);
        evictToLimit();
        return new Lease(this, entry, false);
    }

    public synchronized Stats stats() {
        return new Stats(this.entries.size(), this.residentBytes, this.hits, this.misses);
    }

    private void invalidateOlderVersions(Key current) {
        Iterator<Map.Entry<Key, Entry>> iterator = this.entries.entrySet().iterator();
        while (iterator.hasNext()) {
            Entry entry = iterator.next().getValue();
            if (entry.key.equals(current)) continue;
            if (entry.key.rendererEpoch == current.rendererEpoch
                    && entry.key.geometryId == current.geometryId) {
                entry.stale = true;
                if (entry.references == 0) {
                    iterator.remove();
                    release(entry);
                }
            }
        }
    }

    private void evictToLimit() {
        if (this.residentBytes <= this.limitBytes) return;
        Iterator<Map.Entry<Key, Entry>> iterator = this.entries.entrySet().iterator();
        while (this.residentBytes > this.limitBytes && iterator.hasNext()) {
            Entry entry = iterator.next().getValue();
            if (entry.references != 0) continue;
            iterator.remove();
            release(entry);
        }
    }

    private synchronized void releaseLease(Entry entry) {
        if (entry.references <= 0) throw new IllegalStateException("Geometry cache lease underflow");
        entry.references--;
        if (entry.references == 0 && entry.stale) {
            if (this.entries.remove(entry.key, entry)) release(entry);
        }
        evictToLimit();
    }

    private void release(Entry entry) {
        this.residentBytes -= entry.bytes;
        entry.meshes.forEach(VitrailCpuMeshEncoder.Mesh::close);
    }

    @Override
    public synchronized void close() {
        for (Entry entry : this.entries.values()) {
            entry.meshes.forEach(VitrailCpuMeshEncoder.Mesh::close);
        }
        this.entries.clear();
        this.residentBytes = 0;
        this.hits = 0;
        this.misses = 0;
    }

    private static long byteSize(List<VitrailCpuMeshEncoder.Mesh> meshes) {
        long bytes = 0;
        for (VitrailCpuMeshEncoder.Mesh mesh : meshes) {
            bytes = Math.addExact(bytes, mesh.vertices().size);
            bytes = Math.addExact(bytes, mesh.indices().size);
            bytes = Math.addExact(bytes, mesh.detail().size);
        }
        return bytes;
    }

    public record Key(long rendererEpoch, long packedPosition, int geometryId, long geometryVersion) {}

    public record Stats(int entries, long residentBytes, long hits, long misses) {}

    public static final class Lease implements AutoCloseable {
        private final PersistentGeometryCache owner;
        private Entry entry;
        private final boolean hit;

        private Lease(PersistentGeometryCache owner, Entry entry, boolean hit) {
            this.owner = owner;
            this.entry = entry;
            this.hit = hit;
        }

        public List<VitrailCpuMeshEncoder.Mesh> meshes() {
            if (this.entry == null) throw new IllegalStateException("Geometry cache lease is closed");
            return this.entry.meshes;
        }

        public boolean hit() { return this.hit; }

        @Override
        public void close() {
            Entry held = this.entry;
            if (held == null) return;
            this.entry = null;
            this.owner.releaseLease(held);
        }
    }

    private static final class Entry {
        private final Key key;
        private final List<VitrailCpuMeshEncoder.Mesh> meshes;
        private final long bytes;
        private int references;
        private boolean stale;

        private Entry(Key key, List<VitrailCpuMeshEncoder.Mesh> meshes, long bytes) {
            this.key = key;
            this.meshes = meshes;
            this.bytes = bytes;
        }
    }
}

package me.cortex.voxy.client.core;

import com.mojang.blaze3d.buffers.GpuBuffer;
import com.mojang.blaze3d.systems.RenderSystem;
import me.cortex.voxy.client.core.rendering.building.VitrailCpuMeshEncoder;
import me.cortex.voxy.common.Logger;
import org.lwjgl.system.MemoryUtil;

import java.lang.reflect.Constructor;
import java.lang.reflect.InvocationHandler;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.lang.reflect.Proxy;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

/** Optional reflective bridge into Vitrail, keeping Voxy loadable without the Vitrail mod. */
public final class VitrailBridge {
    private static final String RENDERER = "dev.vitrail.api.render.DistantTerrainRenderer";
    private static final String PROVIDER = "dev.vitrail.api.render.DistantTerrainProvider";
    private static final String SECTION = "dev.vitrail.api.render.DistantTerrainSection";
    private static final String PIECE = "dev.vitrail.api.render.DistantTerrainSection$Piece";

    private static volatile Method drawWithProviders;
    private static volatile Method providerInvocationCount;
    private static volatile Method registerProvider;
    private static volatile Method unregisterProvider;
    private static volatile Constructor<?> sectionConstructor;
    private static volatile Constructor<?> pieceConstructor;
    private static volatile boolean unavailable;
    private static volatile boolean warned;
    private static Object registeredProvider;
    private static long lastOpaqueInvocationCount;
    private static long lastTranslucentInvocationCount;
    private static final Map<Integer, CachedGeometry> opaqueCache = new HashMap<>();
    private static final Map<Integer, CachedGeometry> translucentCache = new HashMap<>();

    private VitrailBridge() {}

    /** Registers Voxy as a provider during Vitrail's DH pass, or for direct calls without DH. */
    public static synchronized void registerProvider(VoxyRenderSystem renderer) {
        if (!RenderBackend.isVitrailVulkanActive() || unavailable || registeredProvider != null) return;
        try {
            ClassLoader loader = VitrailBridge.class.getClassLoader();
            Class<?> api = Class.forName(RENDERER, true, loader);
            Class<?> providerType = Class.forName(PROVIDER, true, loader);
            sectionConstructor = Class.forName(SECTION, true, loader)
                    .getConstructor(int.class, int.class, int.class, List.class);
            pieceConstructor = Class.forName(PIECE, true, loader)
                    .getConstructor(GpuBuffer.class, GpuBuffer.class, int.class);
            drawWithProviders = api.getMethod("drawWithProviders", boolean.class, List.class);
            registerProvider = api.getMethod("registerProvider", providerType);
            unregisterProvider = api.getMethod("unregisterProvider", providerType);
            providerInvocationCount = api.getMethod("providerInvocationCount", providerType, boolean.class);

            InvocationHandler handler = (proxy, method, args) -> {
                if (method.getName().equals("getSections") && args != null && args.length == 1) {
                    return buildSections(renderer, (Boolean) args[0]);
                }
                if (method.getName().equals("toString")) return "Voxy Vitrail terrain provider";
                if (method.getName().equals("hashCode")) return System.identityHashCode(proxy);
                if (method.getName().equals("equals")) return proxy == (args == null ? null : args[0]);
                return null;
            };
            registeredProvider = Proxy.newProxyInstance(providerType.getClassLoader(),
                    new Class<?>[]{providerType}, handler);
            registerProvider.invoke(null, registeredProvider);
            Logger.info("Registered Voxy's CPU terrain provider with Vitrail");
        } catch (ReflectiveOperationException | LinkageError e) {
            unavailable = true;
            warnOnce("Vitrail's distant-terrain provider API is unavailable", e);
        }
    }

    /** Called at Sodium's terrain stages if DH has not supplied this pass during the current frame. */
    public static void drawFromSodium(boolean opaque) {
        if (!RenderBackend.isVitrailVulkanActive() || unavailable) {
            return;
        }
        Method method = drawWithProviders;
        if (method == null || providerInvocationCount == null || registeredProvider == null) return;
        try {
            long calls = (long) providerInvocationCount.invoke(null, registeredProvider, opaque);
            long previous = opaque ? lastOpaqueInvocationCount : lastTranslucentInvocationCount;
            if (calls != previous) {
                if (opaque) lastOpaqueInvocationCount = calls;
                else lastTranslucentInvocationCount = calls;
                return;
            }
            method.invoke(null, opaque, List.of());
            calls = (long) providerInvocationCount.invoke(null, registeredProvider, opaque);
            if (opaque) lastOpaqueInvocationCount = calls;
            else lastTranslucentInvocationCount = calls;
        } catch (IllegalAccessException | InvocationTargetException | LinkageError e) {
            warnOnce("Could not submit Voxy terrain to Vitrail", e);
        }
    }

    public static synchronized void unregisterProvider() {
        Object provider = registeredProvider;
        registeredProvider = null;
        if (provider != null && unregisterProvider != null) {
            try {
                unregisterProvider.invoke(null, provider);
            } catch (ReflectiveOperationException | LinkageError e) {
                warnOnce("Could not unregister Voxy's Vitrail terrain provider", e);
            }
        }
        releaseCache(opaqueCache);
        releaseCache(translucentCache);
    }

    private static synchronized List<Object> buildSections(VoxyRenderSystem renderer, boolean opaque) {
        var device = RenderSystem.tryGetDevice();
        if (device == null || sectionConstructor == null || pieceConstructor == null) return List.of();

        Map<Integer, CachedGeometry> cache = opaque ? opaqueCache : translucentCache;
        LinkedHashMap<TileKey, List<Object>> piecesByTile = new LinkedHashMap<>();
        List<me.cortex.voxy.client.core.rendering.hierachical.NodeManager.GeometryNode> visible = renderer.getVitrailVisibleNodes(
                renderer.getVitrailCameraX(), renderer.getVitrailCameraZ());
        Set<Integer> selectedIds = new HashSet<>();
        try {
            for (var node : visible) {
                int geometryId = node.geometryId();
                selectedIds.add(geometryId);
                CachedGeometry cached = cache.get(geometryId);
                if (cached == null || cached.version != node.geometryVersion() || cached.position != node.position()) {
                    cache.remove(geometryId);
                    releaseGeometry(cached);
                    cached = uploadGeometry(renderer, device, node, opaque);
                    if (cached != null) cache.put(geometryId, cached);
                    else cache.remove(geometryId);
                }
                if (cached == null) continue;
                for (UploadedPiece piece : cached.pieces) {
                    piecesByTile.computeIfAbsent(piece.tile, ignored -> new ArrayList<>()).add(piece.apiPiece);
                }
            }
            for (var it = cache.entrySet().iterator(); it.hasNext();) {
                var entry = it.next();
                if (!selectedIds.contains(entry.getKey())) {
                    releaseGeometry(entry.getValue());
                    it.remove();
                }
            }

            ArrayList<Object> sections = new ArrayList<>(piecesByTile.size());
            for (var entry : piecesByTile.entrySet()) {
                TileKey key = entry.getKey();
                sections.add(sectionConstructor.newInstance(key.x, key.y, key.z, List.copyOf(entry.getValue())));
            }
            return List.copyOf(sections);
        } catch (ReflectiveOperationException | RuntimeException | Error e) {
            throw new IllegalStateException("Failed to build Voxy's Vitrail meshes", e);
        }
    }

    private static CachedGeometry uploadGeometry(VoxyRenderSystem renderer, com.mojang.blaze3d.systems.GpuDevice device,
            me.cortex.voxy.client.core.rendering.hierachical.NodeManager.GeometryNode node, boolean opaque)
            throws ReflectiveOperationException {
        var section = renderer.getVitrailCpuSectionSnapshot(node.geometryId());
        if (section == null) return null;
        ArrayList<UploadedPiece> uploaded = new ArrayList<>();
        try {
            if (section.position != node.position()) return null;
            List<VitrailCpuMeshEncoder.Mesh> meshes = VitrailCpuMeshEncoder.encode(section,
                    renderer::getVitrailModelFaceData, renderer::getVitrailFaceColour,
                    renderer::getVitrailModelMaterial, opaque);
            try {
                for (var mesh : meshes) {
                    if (mesh.opaque() != opaque) continue;
                    GpuBuffer vertex = null;
                    GpuBuffer index = null;
                    try {
                        ByteBuffer vertices = MemoryUtil.memByteBuffer(mesh.vertices().address, Math.toIntExact(mesh.vertices().size));
                        ByteBuffer indices = MemoryUtil.memByteBuffer(mesh.indices().address, Math.toIntExact(mesh.indices().size));
                        vertex = device.createBuffer(() -> "Voxy Vitrail distant vertices", GpuBuffer.USAGE_VERTEX, vertices);
                        index = device.createBuffer(() -> "Voxy Vitrail distant indices", GpuBuffer.USAGE_INDEX, indices);
                        Object apiPiece = pieceConstructor.newInstance(vertex, index, mesh.indexCount());
                        uploaded.add(new UploadedPiece(new TileKey(mesh.x(), mesh.y(), mesh.z()), apiPiece, vertex, index));
                    } catch (RuntimeException | ReflectiveOperationException | Error e) {
                        if (vertex != null) vertex.close();
                        if (index != null) index.close();
                        throw e;
                    }
                }
            } finally {
                meshes.forEach(VitrailCpuMeshEncoder.Mesh::close);
            }
            return new CachedGeometry(node.position(), node.geometryVersion(), List.copyOf(uploaded));
        } catch (RuntimeException | ReflectiveOperationException | Error e) {
            releasePieces(uploaded);
            throw e;
        } finally {
            section.free();
        }
    }

    private static void releaseCache(Map<Integer, CachedGeometry> cache) {
        cache.values().forEach(VitrailBridge::releaseGeometry);
        cache.clear();
    }

    private static void releaseGeometry(CachedGeometry geometry) {
        if (geometry != null) releasePieces(geometry.pieces);
    }

    private static void releasePieces(List<UploadedPiece> pieces) {
        for (UploadedPiece piece : pieces) {
            try { piece.vertex.close(); } catch (RuntimeException e) { Logger.warn("Could not release Vitrail vertex buffer", e); }
            try { piece.index.close(); } catch (RuntimeException e) { Logger.warn("Could not release Vitrail index buffer", e); }
        }
    }

    private static void warnOnce(String message, Throwable error) {
        if (!warned) {
            synchronized (VitrailBridge.class) {
                if (!warned) {
                    warned = true;
                    Logger.warn(message + ": " + error);
                }
            }
        }
    }

    private record TileKey(int x, int y, int z) {}
    private record UploadedPiece(TileKey tile, Object apiPiece, GpuBuffer vertex, GpuBuffer index) {}
    private record CachedGeometry(long position, long version, List<UploadedPiece> pieces) {}
}

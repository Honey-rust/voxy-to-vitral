package me.cortex.voxy.client.mixin.sodium;

import com.mojang.blaze3d.buffers.GpuBuffer;
import com.mojang.blaze3d.buffers.GpuBufferSlice;
import com.mojang.blaze3d.opengl.GlTextureView;
import com.mojang.blaze3d.textures.GpuSampler;
import me.cortex.voxy.client.VoxyClient;
import me.cortex.voxy.client.core.IVoxyRenderSystemHolder;
import me.cortex.voxy.client.core.rendering.Viewport;
import me.cortex.voxy.client.core.util.IrisUtil;
import net.caffeinemc.mods.sodium.client.render.chunk.ChunkRenderMatrices;
import net.caffeinemc.mods.sodium.client.render.chunk.DefaultChunkRenderer;
import net.caffeinemc.mods.sodium.client.render.chunk.ShaderChunkRenderer;
import net.caffeinemc.mods.sodium.client.render.chunk.lists.ChunkRenderListIterable;
import net.caffeinemc.mods.sodium.client.render.chunk.terrain.DefaultTerrainRenderPasses;
import net.caffeinemc.mods.sodium.client.render.chunk.terrain.TerrainRenderPass;
import net.caffeinemc.mods.sodium.client.render.chunk.vertex.format.ChunkVertexType;
import net.caffeinemc.mods.sodium.client.render.viewport.CameraTransform;
import net.caffeinemc.mods.sodium.client.util.FogParameters;
import net.fabricmc.loader.api.FabricLoader;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.Unique;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;

@Mixin(value = DefaultChunkRenderer.class, remap = false)
public abstract class MixinDefaultChunkRenderer extends ShaderChunkRenderer {

    public MixinDefaultChunkRenderer(ChunkVertexType vertexType) {
        super(vertexType);
    }

    @Inject(method = "render", at = @At(value = "HEAD"), cancellable = true)
    private void voxy$cancelThingie(ChunkRenderMatrices matrices, ChunkRenderListIterable renderLists, TerrainRenderPass renderPass, CameraTransform camera, FogParameters parameters, boolean indexedRenderingEnabled, GpuSampler terrainSampler, GpuBufferSlice uniformData, GpuBuffer sectionTimeInfo, CallbackInfo ci) {
        if (me.cortex.voxy.client.core.RenderBackend.isVitrailVulkanActive()) {
            // Match Voxy's composition order: distant terrain establishes the far background,
            // then Sodium's real chunks overwrite it. In particular, never let the coarse LOD
            // cover or modify the depth of nearby vanilla terrain.
            this.doRender(matrices, renderPass, camera, parameters);
            return;
        }
        if (VoxyClient.disableSodiumChunkRender()) {
            super.begin(renderPass, parameters, terrainSampler);
            this.doRender(matrices, renderPass, camera, parameters);
            super.end(renderPass);
            ci.cancel();
        }
    }

    @Inject(method = "render", at = @At(value = "INVOKE", target = "Lnet/caffeinemc/mods/sodium/client/render/chunk/ShaderChunkRenderer;end(Lnet/caffeinemc/mods/sodium/client/render/chunk/terrain/TerrainRenderPass;)V", shift = At.Shift.BEFORE))
    private void voxy$injectRender(ChunkRenderMatrices matrices, ChunkRenderListIterable renderLists, TerrainRenderPass renderPass, CameraTransform camera, FogParameters parameters, boolean indexedRenderingEnabled, GpuSampler terrainSampler, GpuBufferSlice uniformData, GpuBuffer sectionTimeInfo, CallbackInfo ci) {
        if (me.cortex.voxy.client.core.RenderBackend.isVitrailVulkanActive()) {
            return;
        }
        this.doRender(matrices, renderPass, camera, parameters);
    }

    @Unique
    private void doRender(ChunkRenderMatrices matrices, TerrainRenderPass renderPass, CameraTransform camera, FogParameters fogParameters) {
        if (me.cortex.voxy.client.core.RenderBackend.isVitrailVulkanActive()) {
            if (renderPass == DefaultTerrainRenderPasses.SOLID) {
                var renderer = IVoxyRenderSystemHolder.getNullable();
                if (renderer != null) {
                    var target = renderPass.getTarget();
                    renderer.tickVitrail(matrices.projection(), matrices.modelView(),
                            target.width, target.height, camera.x, camera.y, camera.z);
                }
                if (!me.cortex.voxy.client.core.VitrailBridge.usesPlainRenderer()) {
                    me.cortex.voxy.client.core.VitrailBridge.drawFromSodium(true);
                }
            } else if (renderPass == DefaultTerrainRenderPasses.TRANSLUCENT) {
                if (!me.cortex.voxy.client.core.VitrailBridge.usesPlainRenderer()) {
                    me.cortex.voxy.client.core.VitrailBridge.drawFromSodium(false);
                }
            }
            return;
        }
        if (renderPass == DefaultTerrainRenderPasses.CUTOUT) {
            var renderer = IVoxyRenderSystemHolder.getNullable();
            if (renderer != null) {
                Viewport<?> viewport = null;
                var target = renderPass.getTarget();
                if (IrisUtil.USED_IRIS_VIEWPORT) {
                    viewport = renderer.getViewport();
                    IrisUtil.USED_IRIS_VIEWPORT = false;
                } else {
                    viewport = renderer.setupViewport(matrices.projection(), matrices.modelView(), fogParameters, target.width, target.height, camera.x, camera.y, camera.z);
                }
                renderer.renderOpaque(viewport, ((GlTextureView)target.getDepthTextureView()).glId(), ((GlTextureView)target.getColorTextureView()).glId());
            }
        }
    }

    // Sodium has an early return when this terrain pass has no commands. At high altitude
    // the near cutout list can be empty even though Voxy still has distant terrain to draw.
    // TAIL only visits the final return; RETURN must cover both exits independently.
    @Inject(method = "render", at = @At("RETURN"), require = 2)
    private void voxy$drawPlainLodAfterCutout(ChunkRenderMatrices matrices,
            ChunkRenderListIterable renderLists, TerrainRenderPass renderPass, CameraTransform camera,
            FogParameters parameters, boolean indexedRenderingEnabled, GpuSampler terrainSampler,
            GpuBufferSlice uniformData, GpuBuffer sectionTimeInfo, CallbackInfo ci) {
        if (me.cortex.voxy.client.core.RenderBackend.isVitrailVulkanActive()
                && renderPass == DefaultTerrainRenderPasses.CUTOUT
                && me.cortex.voxy.client.core.VitrailBridge.usesPlainRenderer()) {
            me.cortex.voxy.client.core.VitrailBridge.drawPlainFromSodium(
                    matrices.projection(), matrices.modelView(), camera.x, camera.y, camera.z);
        }
    }
}

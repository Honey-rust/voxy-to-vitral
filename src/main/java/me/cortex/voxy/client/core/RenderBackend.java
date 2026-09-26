package me.cortex.voxy.client.core;

import com.mojang.blaze3d.systems.RenderSystem;
import net.fabricmc.loader.api.FabricLoader;

import java.util.Locale;

/** Runtime rendering backend checks. Vitrail can be installed while the game is using OpenGL. */
public final class RenderBackend {
    private RenderBackend() {}

    public static boolean isVitrailVulkanActive() {
        if (!FabricLoader.getInstance().isModLoaded("vitrail")) {
            return false;
        }

        var device = RenderSystem.tryGetDevice();
        if (device == null) {
            return false;
        }

        return device.getDeviceInfo().backendName().toLowerCase(Locale.ROOT).contains("vulkan");
    }
}

package me.cortex.voxy.client.core.gl;

import me.cortex.voxy.common.util.MemoryBuffer;
import me.cortex.voxy.common.util.TrackedObject;
import org.lwjgl.opengl.GL11;
import org.lwjgl.system.MemoryUtil;

import static org.lwjgl.opengl.ARBSparseBuffer.GL_SPARSE_STORAGE_BIT_ARB;
import static org.lwjgl.opengl.GL11.GL_UNSIGNED_BYTE;
import static org.lwjgl.opengl.GL15.glDeleteBuffers;
import static org.lwjgl.opengl.GL45C.*;

public class GlBuffer extends TrackedObject {
    public final int id;
    private final long size;
    private final int flags;

    private static int COUNT;
    private static long TOTAL_SIZE;

    public GlBuffer(long size) {
        this(size, 0);
    }

    public GlBuffer(MemoryBuffer buffer) {
        this(buffer.size, 0, false, buffer.address);
    }

    public GlBuffer(long size, boolean zero) {
        this(size, 0, zero);
    }

    public GlBuffer(long size, int flags) {
        this(size, flags, true);
    }

    public GlBuffer(long size, int flags, boolean zero) {
        this(size, flags, zero, 0);
    }

    private GlBuffer(long size, int flags, boolean zero, long data) {
        this.flags = flags;
        this.size = size;

        // === Vitrail (Vulkan) 模式：赋予虚拟缓冲句柄，不执行原生 GL 指令 ===
        if (net.fabricmc.loader.api.FabricLoader.getInstance().isModLoaded("vitrail")) {
            this.id = 0;
            COUNT++;
            TOTAL_SIZE += size;
            return;
        }

        this.id = glCreateBuffers();
        nglNamedBufferStorage(this.id, size, data, flags);
        if ((flags&GL_SPARSE_STORAGE_BIT_ARB)==0 && zero) {
            this.zero();
        }

        COUNT++;
        TOTAL_SIZE += size;
    }

    @Override
    public void free() {
        this.free0();
        if (!net.fabricmc.loader.api.FabricLoader.getInstance().isModLoaded("vitrail")) {
            glDeleteBuffers(this.id);
        }

        COUNT--;
        TOTAL_SIZE -= this.size;
    }

    public boolean isSparse() {
        return (this.flags&GL_SPARSE_STORAGE_BIT_ARB)!=0;
    }

    public long size() {
        return this.size;
    }

    public GlBuffer zero() {
        if (!net.fabricmc.loader.api.FabricLoader.getInstance().isModLoaded("vitrail")) {
            nglClearNamedBufferData(this.id, GL_R8UI, GL_RED_INTEGER, GL_UNSIGNED_BYTE, 0);
        }
        return this;
    }

    public GlBuffer zeroRange(long offset, long size) {
        if (!net.fabricmc.loader.api.FabricLoader.getInstance().isModLoaded("vitrail")) {
            nglClearNamedBufferSubData(this.id, GL_R8UI, offset, size, GL_RED_INTEGER, GL_UNSIGNED_BYTE, 0);
        }
        return this;
    }

    public GlBuffer fill(int data) {
        if (net.fabricmc.loader.api.FabricLoader.getInstance().isModLoaded("vitrail")) {
            return this;
        }
        //Clear unpack values
        //Fixed in mesa commit a5c3c452
        glPixelStorei(GL11.GL_UNPACK_SKIP_ROWS, 0);
        glPixelStorei(GL11.GL_UNPACK_SKIP_PIXELS, 0);

        MemoryUtil.memPutInt(SCRATCH, data);
        nglClearNamedBufferData(this.id, GL_R32UI, GL_RED_INTEGER, GL_UNSIGNED_INT, SCRATCH);
        return this;
    }

    public static int getCount() {
        return COUNT;
    }

    public static long getTotalSize() {
        return TOTAL_SIZE;
    }

    public GlBuffer name(String name) {
        if (net.fabricmc.loader.api.FabricLoader.getInstance().isModLoaded("vitrail")) {
            return this;
        }
        return GlDebug.name(name, this);
    }

    private static final long SCRATCH = MemoryUtil.nmemAlloc(4);
}
// MANUFACTURER=1 (nvidia), =0 (amd)
#define MANUFACTURER
#define REPORT_DIR
#define KERNEL_CAN_ABORT

#include "tool.h"
#include "kernel_limits.h"

int main(int argc, char **argv) {
    CHECK(hipSetDevice(0));

    uint32_t limit_threads_per_block;
    uint32_t limit_threads_per_grid;
    size_t limit_sharedmem_per_block;
#if MANUFACTURER == 1
    uint32_t limit_registers_per_thread;
#else
    uint32_t limit_registers_per_thread[2];
#endif
    uint32_t limit_registers_per_block;
    uint32_t warp_size;

    // get expected values
    hipDeviceProp_t prop;
    CHECK(hipGetDeviceProperties(&prop, 0));
    limit_threads_per_block = prop.maxThreadsPerBlock;
    limit_threads_per_grid = prop.maxGridSize[0];
    limit_sharedmem_per_block = prop.sharedMemPerBlock;
#if MANUFACTURER == 1
    limit_registers_per_thread = 255;
#else
    if (prop.gcnArch < 1000) {
        // 102 for Vega or lower
        limit_registers_per_thread[0] = 102;
    }
    else {
        // 106 for RDNA or higher
        limit_registers_per_thread[0] = 106;
    }
    limit_registers_per_thread[1] = 256;
#endif
    limit_registers_per_block = prop.regsPerBlock;
    warp_size = prop.warpSize;

    write_init("kernel_limits");
    write_line("# following values are from hipDeviceProp_t, not verified yet");
    write_value("warp_size", warp_size);
    write_value("limit_threads_per_block", limit_threads_per_block);
    write_value("limit_threads_per_grid", limit_threads_per_grid);
    write_value("limit_sharedmem_per_block", limit_sharedmem_per_block);
#if MANUFACTURER == 1
    write_value("limit_registers_per_thread", limit_registers_per_thread);
    write_value("limit_registers_per_block", limit_registers_per_block);
#else
    write_values("limit_registers_per_thread", limit_registers_per_thread, 2);
    uint32_t LRpB[2] = {limit_registers_per_block, limit_registers_per_block};
    write_values("limit_registers_per_block", LRpB, 2);
#endif
    

    return 0;
}
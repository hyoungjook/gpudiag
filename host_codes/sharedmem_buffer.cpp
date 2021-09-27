// MANUFACTURER=1 (nvidia), =0 (amd)
#define MANUFACTURER
#define REPORT_FILE
#define KERNEL_FILE

#define limit_sharedmem_per_block
#define num_mp
#define nslot_n1n2

#define ckpt_nslot_timeout_multiplier
#define ckpt_shared_memory_test_granularity

#include "gpudiag_runtime.h"
#include "measure_nslot.h"

bool shmem_data_consistent(uint64_t S, uint64_t s_min, int N, int *nbuf,
        int shmem_unit) {
    for (int i=0; i<N; i++) {
        if (nbuf[i] >= nslot_n1n2(0)) continue;
        uint64_t test_s = (i+1) * shmem_unit;
        test_s = (test_s + s_min - 1) / s_min * s_min; // ceil(test_s, s_min)
        if (nbuf[i] != S / test_s) return false;
    }
    return true;
}

int main(int argc, char **argv) {
    GDInit();
    write_init("sharedmem_buffer");

    const int shmem_unit = ckpt_shared_memory_test_granularity;
    const int iter_num = limit_sharedmem_per_block(0) / shmem_unit;
    int Nbuf[iter_num];
    float inv_s[iter_num];
    for (int i=0; i<iter_num; i++) {
        Nbuf[i] = measure_nslot(num_mp(0), 1, measure_shmem[i]);
        inv_s[i] = 1.0f / (float)((i+1)*shmem_unit);
    }
    // analyze
    uint64_t S = 0;
    for (int i=0; i<iter_num; i++) S = MAX(S, (i+1)*shmem_unit*Nbuf[i]);
    uint64_t s_min = shmem_unit;
    bool s_min_strange = false;
    while(true) {
        if (shmem_data_consistent(S, s_min, iter_num, Nbuf, shmem_unit)) {
            break;
        }
        s_min *= 2;
        if (s_min >= S) {s_min_strange = true; break;}
    }
    // write
    write_line("# 1. Raw data");
    write_graph_data_xs_with_line("Shared memory", iter_num,
        "1/s (1/Bytes)", inv_s, "Nslot", Nbuf, S, 0, "max slope");
    write_value("shared_memory_size", S);
    if (s_min_strange) write_line("## cannot measure s_min automatically..");
    else {
        if (s_min == shmem_unit) write_line("## s_min is less or equal than below!");
        write_value("shared_memory_alloc_unit", s_min);
    }

    return 0;
}
// MANUFACTURER=1 (nvidia), =0 (amd)
#define MANUFACTURER
#define REPORT_FILE
#define KERNEL_FILE

#define limit_sharedmem_per_block

#include "gpudiag_runtime.h"

int main(int argc, char **argv) {
    GDInit();
    write_init("dcache_hierarchy");

    // === L1D Linesize test ===
    const int num_linesize_repeat = 100; // should be consistent with -.py
    uint32_t *darr;
    uint64_t *hres, *dres;
    int resarrsize = num_linesize_repeat * sizeof(uint64_t);
    GDMalloc(&darr, num_linesize_repeat * sizeof(uint32_t));
    hres = (uint64_t*)malloc(resarrsize);
    GDMalloc(&dres, resarrsize);
    GDLaunchKernel(measure_l1d_linesize, dim3(1), dim3(1), 0, 0, darr, dres);
    GDSynchronize();
    GDMemcpyDToH(hres, dres, resarrsize);
    GDFree(darr); GDFree(dres);
    // analyze : ignore hres[0]
    uint64_t latency_min = hres[1], latency_max = hres[1];
    for (int i=1; i<num_linesize_repeat; i++) {
        latency_min = MIN(latency_min, hres[i]);
        latency_max = MAX(latency_max, hres[i]);
    }
    uint64_t latency_criterion = (latency_min + latency_max) / 2;
    int first_miss_idx = -1, second_miss_idx = -1, hit_cnt = 0;
    float l1d_hit_latency = 0;
    for (int i=1; i<num_linesize_repeat; i++) {
        if (hres[i] <= latency_criterion) {
            hit_cnt++; l1d_hit_latency += (float)hres[i];
        }
        else {
            if (first_miss_idx < 0) first_miss_idx = i;
            else if (second_miss_idx < 0) second_miss_idx = i;
        }
    }
    l1d_hit_latency /= (float)hit_cnt;
    int l1d_linesize = (second_miss_idx - first_miss_idx) * sizeof(uint32_t);
    write_line("# 1. L1D linesize test");
    write_graph_data("L1D linesize", num_linesize_repeat-1, "array index", 1, 1,
        "latency", hres+1);
    free(hres);
    if (l1d_linesize <= 0) {
        printf("Cannot resolve L1D linesize.");
        exit(1);
    }

    // == Dcache Hierarchy test ===
    resarrsize = NUM_DCACHE_REPEAT * sizeof(uint64_t);
    int testarrsize = NUM_DCACHE_REPEAT * l1d_linesize;
    GDMalloc(&darr, testarrsize);
    hres = (uint64_t*)malloc(resarrsize);
    GDMalloc(&dres, resarrsize);
    GDLaunchKernel(measure_dcache, dim3(1), dim3(1), 0, 0,
        darr, l1d_linesize / sizeof(uint32_t), dres);
    GDSynchronize();
    GDMemcpyDToH(hres, dres, resarrsize);
    GDFree(darr); GDFree(dres);
    // analyze : ignore hres[0], must be hit
    const int max_level = 10; int current_level = 0;
    int data_cnt[max_level]; for (int l=0; l<max_level; l++) data_cnt[l] = 0;
    float hit_latency[max_level]; hit_latency[0] = l1d_hit_latency; data_cnt[0]++;
    uint64_t capacity[max_level], linesize[max_level]; linesize[0] = l1d_linesize;
    for (int l=1; l<max_level; l++) linesize[l] = 0;
    int linesize_counting = -1;
    
    for (int i=1; i<NUM_DCACHE_REPEAT; i++) {
        float val = (float)hres[i];
        if (val > 1.5 * hit_latency[current_level]) {
            // new hierarchy found
            current_level++; hit_latency[current_level] = val;
            capacity[current_level-1] = (i) * l1d_linesize;
            linesize_counting = 0; continue;
        }
        // which level the data is hit?
        float min_err = val / hit_latency[0];
        int level = 0;
        for (int l=1; l<=current_level; l++) {
            float err = val / hit_latency[l];
            err = err<1.0f ? 1.0f-err : err-1.0f;
            if (err < min_err) {min_err = err; level = l;}
        }
        data_cnt[level]++;
        hit_latency[level] = (hit_latency[level] * (float)(data_cnt[level]-1) +
            val) / (float)data_cnt[level];
        if (linesize_counting >= 0) {
            linesize_counting++;
            if (linesize[current_level] == 0 && level == current_level) {
                linesize[current_level] = l1d_linesize * linesize_counting;
                linesize_counting = -1;
            }
        }
    }
    int max_observed_level = current_level;
    int num_observed_linesizes = 0;
    for (int l=0; l<max_level; l++) if (linesize[l]>0) num_observed_linesizes++;
    write_line("# 2. Dcache hierarchy test");
    write_graph_data("Dcache hierarchy", NUM_DCACHE_REPEAT-1,
        "data size(B)", 2*l1d_linesize, l1d_linesize, "latency", hres+1);
    if (max_observed_level > 0)
        write_values("dcache_capacities", capacity, max_observed_level);
    else {
        write_line("## Only 1 level is observed, L1D capacity is at least following.");
        write_value("dcache_capacities", NUM_DCACHE_REPEAT * l1d_linesize);
    }
    write_values("dcache_linesizes", linesize, num_observed_linesizes);
    write_values("dcache_latencies", hit_latency, max_observed_level+1);
    free(hres);

    return 0;
}
// MANUFACTURER=1 (nvidia), =0 (amd)
#define MANUFACTURER
#define REPORT_DIR
#define limit_registers_per_thread(i)

#include "tool.h"
#include "l1i_linesize.h"

int main(int argc, char **argv) {
    CHECK(hipSetDevice(0));

#if MANUFACTURER == 1
    const int N_measures = limit_registers_per_thread - 10;
#else
    const int N_measures = (int)(limit_registers_per_thread(0))/2-3;
#endif

    uint64_t *hres, *dres;
    const int arrsize = N_measures * sizeof(uint64_t);
    hres = (uint64_t*)malloc(arrsize);
    // initialize, because in CUDA clock data will be uint32_t
    hipMalloc(&dres, arrsize);
    for (int i=0; i<N_measures; i++) hres[i] = 0;
    hipMemcpy(dres, hres, arrsize, hipMemcpyHostToDevice);

    hipLaunchKernelGGL(measure_l1i_linesize, dim3(1), dim3(1), 0, 0, dres);
    hipStreamSynchronize(0);
    hipMemcpy(hres, dres, arrsize, hipMemcpyDeviceToHost);

    // raw data differences
    for (int i=0; i<N_measures-1; i++) {
        hres[i] = hres[i+1] - hres[i];
    }

    // goal
    uint64_t l1i_linesize;
    float l1i_hit_latency=0;
    
    // hit vs miss split criterion : average of min & max
    uint64_t latency_min, latency_max;
    latency_min = hres[0]; latency_max = hres[0];
    for (int i=0; i<N_measures-1; i++) {
        latency_min = latency_min < hres[i] ? latency_min : hres[i];
        latency_max = latency_max > hres[i] ? latency_max : hres[i];
    }
    uint64_t latency_criterion = (latency_min + latency_max) / 2;

    // split l1i hit vs miss by the criterion
    int first_miss_idx = -1, second_miss_idx = -1, hit_cnt = 0;
    float miss_latency_avg = 0; // to verify that the miss occurred
    for (int i=0; i<N_measures-1; i++) {
        if (hres[i] <= latency_criterion) {
            hit_cnt++;
            l1i_hit_latency += (float)hres[i];
        }
        else {
            if (first_miss_idx < 0) first_miss_idx = i;
            else {
                if (second_miss_idx < 0) second_miss_idx = i;
            }
            miss_latency_avg += (float)hres[i];
        }
    }
    l1i_hit_latency /= (float)hit_cnt;
    miss_latency_avg /= (float)(N_measures-1-hit_cnt);
    l1i_linesize = 16 * (second_miss_idx - first_miss_idx);  

    write_init("l1i_linesize");

    write_line("# 1. Raw data");
    write_graph_data("L1I linesize", N_measures-1, "inst. size(B)", 16, 16,
        "latency", hres);
    
    char buf[N_measures * 10 + 100];
    int bufptr;
    write_line("# 2. Hit/Miss split");
    sprintf(buf, "# criterion = (min+max)/2 = %lu", latency_criterion);
    write_line(buf);
    bufptr = sprintf(buf, "# ");
    for (int i=0; i<N_measures-1; i++) {
        bufptr += sprintf(
            buf+bufptr, "%s, ", hres[i]<=latency_criterion ? "hit" : "miss"
        );
    }
    write_line(buf);
    if (miss_latency_avg < 3 * l1i_hit_latency) {
        sprintf(
            buf, 
            "# CAUTION!! avg_miss_latency=%.2f not big enough "
            "than avg_hit_latency=%.2f",
            miss_latency_avg, l1i_hit_latency
        );
        write_line(buf);
    }
    if (second_miss_idx < 0) {
        sprintf(buf, "# CAUTION!! miss occurred only 1 time");
        write_line(buf);
    }

    write_line("# 3. Results");
    write_value("l1i_linesize", l1i_linesize);

    hipFree(dres);
    free(hres);

    return 0;
}
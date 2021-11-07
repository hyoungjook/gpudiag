// MANUFACTURER=1 (nvidia), =0 (amd)
#define MANUFACTURER
#define REPORT_FILE
#define KERNEL_FILE

#define warp_size
#define limit_threads_per_block

#include "gpudiag_runtime.h"

// consistent with warpsched_policy.py
#define Nrepeat 10

int main(int argc, char **argv) {
    GDInit();
    write_init("warpsched_policy");

    const int max_b = limit_threads_per_block(0) / warp_size(0);

    // measure the times
    uint64_t *hres, *dres;
    const size_t arrsize = Nrepeat * max_b * sizeof(uint64_t);
    hres = (uint64_t *)malloc(arrsize);
    for (int i=0; i<Nrepeat*max_b; i++) hres[i] = 0;
    GDMalloc(&dres, arrsize);
    GDMemcpyHToD(dres, hres, arrsize);
    GDLaunchKernel(test_warpsched, dim3(1), dim3(max_b*warp_size(0)), 0, 0, dres);
    GDSynchronize();
    GDMemcpyDToH(hres, dres, arrsize);

    // subtract minval
    uint64_t minval = (uint64_t)-1;
    for (int i=0; i<Nrepeat*max_b; i++) minval = (minval<hres[i])?minval:hres[i];
    for (int i=0; i<Nrepeat*max_b; i++) hres[i] -= minval;

    // convert to warp vs time
    uint64_t *data_warp = (uint64_t *)malloc(arrsize);
    uint64_t *data_time = (uint64_t *)malloc(arrsize);

    for (int i=0; i<Nrepeat*max_b; i++) {
        // find n'th minimum
        uint64_t minval = (uint64_t)(-1), minwarp = 0;
        int min_j = 0;
        for (int j=0; j<Nrepeat*max_b; j++) {
            if (hres[j] < minval) {
                minval = hres[j]; minwarp = j / Nrepeat; min_j = j;
            }
        }
        hres[min_j] = (uint64_t)(-1);
        // fill the data
        data_warp[i] = minwarp;
        data_time[i] = minval;
    }

    // print the result
    write_graph_data_xs("warp scheduling", Nrepeat*max_b,
        "time (clock)", data_time, "--scheduled warp id", data_warp);

    GDFree(dres);
    free(hres);
    return 0;
}
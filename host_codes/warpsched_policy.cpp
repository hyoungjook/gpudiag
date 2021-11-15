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

    // sort warp_ids by first scheduled time
    uint64_t *sorted_warp = (uint64_t *)malloc(max_b * sizeof(uint64_t));
    // sorted_warp[real_warpid] = sorted_warpid
    for (int i=0; i<max_b; i++) {
        int num_warps_earlier = 0;
        // real_warpid i's first scheduled time
        uint64_t first_time = hres[i * Nrepeat];
        // count the number of 'more early-started' warp ids
        for (int j=0; j<max_b; j++) {
            if (j==i) continue;
            if (hres[j*Nrepeat] < first_time ||
                hres[j*Nrepeat] == first_time && j < i) // tiebreaker by id
                    num_warps_earlier++;
        }
        // save to sorted
        sorted_warp[i] = num_warps_earlier;
    }

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
        data_warp[i] = sorted_warp[minwarp];
        data_time[i] = minval;
    }

    // print the result
    write_graph_data_xs("warp scheduling", Nrepeat*max_b,
        "time (clock)", data_time, "--scheduled warp id", data_warp);

    GDFree(dres);
    free(hres);
    free(data_warp);
    free(data_time);
    free(sorted_warp);
    return 0;
}
// MANUFACTURER=1 (nvidia), =0 (amd)
#define MANUFACTURER
#define REPORT_FILE
#define KERNEL_FILE

#define warp_size
#define limit_threads_per_block

#include "gpudiag_runtime.h"
#include <vector>

#define test_until_multiple 3

uint64_t test_once(uint64_t G, int B) {
    uint64_t htime, *dtime;
    GDMalloc(&dtime, sizeof(uint64_t));
    htime = 0;
    GDMemcpyHToD(dtime, &htime, sizeof(uint64_t));
    GDLaunchKernel(measure_width_verify, dim3(G), dim3(B), 0, 0, dtime);
    GDSynchronize();
    GDMemcpyDToH(&htime, dtime, sizeof(uint64_t));
    GDFree(dtime);
    return htime;
}

int main(int argc, char **argv) {
    GDInit();
    
    const int iter_num = limit_threads_per_block(0) / warp_size(0);
    std::vector<uint64_t> results[iter_num];

    uint64_t value = 0;
    for (int b=1; b<=iter_num; b++) {
        int testB = b * warp_size(0);
        uint64_t testG = 1;
        uint64_t ref_val = test_once(testG, testB);
        results[b-1].push_back(ref_val);
        testG++;
        while(1) {
            value = test_once(testG, testB);
            results[b-1].push_back(value);
            if (value > ref_val * test_until_multiple) break;
            testG++;
        }
    }

    // print
    for (int b=1; b<=iter_num; b++) {
        for (int i=0; i<results[b-1].size(); i++) {
            printf("%lu,", results[b-1][i]);
        }
        printf("\n");
    }
}
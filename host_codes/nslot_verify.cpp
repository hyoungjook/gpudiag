// MANUFACTURER=1 (nvidia), =0 (amd)
#define MANUFACTURER
#define REPORT_FILE
#define KERNEL_FILE

#define warp_size
#define limit_threads_per_block

#include "gpudiag_runtime.h"
#include <vector>

#define num_mp 1
#define test_until_multiple 7

uint64_t test_once(uint64_t G, int B) {
    uint64_t *htime, *dtime;
    size_t arrsize = 2 * G * sizeof(uint64_t);
    htime = (uint64_t*)malloc(arrsize);
    GDMalloc(&dtime, arrsize);
    for (int i=0; i<G; i++) htime[i] = 0;
    GDMemcpyHToD(dtime, htime, arrsize);
    GDLaunchKernel(measure_width_verify, dim3(G), dim3(B), 0, 0, dtime);
    GDSynchronize();
    GDMemcpyDToH(htime, dtime, arrsize);
    GDFree(dtime);
    uint64_t minsclk=(uint64_t)-1, maxeclk=0;
    for (int i=0; i<G; i++) {
        uint64_t sclk = htime[2*i], eclk = htime[2*i+1];
        minsclk = minsclk < sclk ? minsclk : sclk;
        maxeclk = maxeclk > eclk ? maxeclk : eclk;
    }
    free(htime);
    return maxeclk - minsclk;
}

int main(int argc, char **argv) {
    GDInit();
    
    const int iter_num = limit_threads_per_block(0) / warp_size(0);
    std::vector<float> results[iter_num];

    uint64_t value = 0;
    for (int b=1; b<=iter_num; b++) {
        int testB = b * warp_size(0);
        uint64_t testG = 1;
        uint64_t ref_val = test_once(testG, testB);
        results[b-1].push_back(1.0f);
        testG += num_mp;
        while(1) {
            value = test_once(testG, testB);
            results[b-1].push_back((float)value / (float)ref_val);
            if (value > ref_val * test_until_multiple) break;
            testG += num_mp;
        }
    }

    // print
    FILE *out = fopen("nslot_verify_result.txt", "w");
    for (int b=1; b<=iter_num; b++) {
        for (int i=0; i<results[b-1].size(); i++) {
            fprintf(out, "%f,", results[b-1][i]);
        }
        fprintf(out, "\n");
    }
    fclose(out);
}
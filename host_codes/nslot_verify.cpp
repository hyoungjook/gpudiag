// MANUFACTURER=1 (nvidia), =0 (amd)
#define MANUFACTURER
#define REPORT_FILE
#define KERNEL_FILE

#define warp_size
#define limit_threads_per_block

#include "gpudiag_runtime.h"
#include <vector>

#define num_mp 40
#define test_until_multiple 5

GDEvent_t start, stop;

float test_once(uint64_t G, int B) {
    
    GDEventRecord(start);
    GDLaunchKernel(measure_width_verify, dim3(G), dim3(B), 0, 0);
    GDEventRecord(stop);
    GDEventSynchronize(stop);
    float mili;
    GDEventElapsedTime(&mili, start, stop);
    return mili;
}

int main(int argc, char **argv) {
    GDInit();
    GDEventCreate(&start);
    GDEventCreate(&stop);
    
    const int iter_num = limit_threads_per_block(0) / warp_size(0);
    std::vector<float> results[iter_num];

    // test once for warmup icache in each mp!
    for(int i=0; i<20; i++) test_once(num_mp, 1);

    float value = 0;
    for (int b=1; b<=iter_num; b++) {
        int testB = b * warp_size(0);
        uint64_t testG = 1;
        float ref_val = test_once(testG, testB);
        results[b-1].push_back(ref_val);
        testG += num_mp;
        while(1) {
            value = test_once(testG, testB);
            if (value > ref_val * test_until_multiple) break;
            if (testG > num_mp * 40) break;
            results[b-1].push_back(value);
            testG += num_mp;
        }
    }

    // print
    FILE *out = fopen("nslot_verify_result.txt", "w");
    for (int b=1; b<=iter_num; b++) {
        for (int i=0; i<results[b-1].size(); i++) {
            // g, b, result
            fprintf(out, "%d, %d, %f\n", i+1, b, results[b-1][i]);
        }
    }
    fclose(out);
}
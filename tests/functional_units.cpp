// MANUFACTURER=1 (nvidia), =0 (amd)
#define MANUFACTURER
#define REPORT_DIR

#define warp_size
#define limit_threads_per_block

#define ckpt_eu_latency_repeats

#include "tool.h"
#include "exec_units.h"

void do_test(int test_idx, const char *name) {
    uint64_t htime, *dtime;
    hipMalloc(&dtime, sizeof(uint64_t));
    // measure period
    htime = 0;
    hipMemcpy(dtime, &htime, sizeof(uint64_t), hipMemcpyHostToDevice);
    hipLaunchKernelP(measure_latency[test_idx], dim3(1), dim3(1), 0, 0, dtime);
    hipStreamSynchronize(0);
    hipMemcpy(&htime, dtime, sizeof(uint64_t), hipMemcpyDeviceToHost);
    float period = (float)(htime - 1) / (ckpt_eu_latency_repeats - 1);
    char buf[50];
    sprintf(buf, "# %d. %s op. results", test_idx+1, name);
    write_line(buf);
    sprintf(buf, "# Period P = %.3f", period);
    write_line(buf);
    // measure n_bot
    if (measure_width[test_idx] != NULL) {
        const int iter_num = limit_threads_per_block / warp_size;
        uint64_t times_raw[iter_num];
        float T_at_c[iter_num];
        for (int b=1; b<=iter_num; b++) {
            htime = 0;
            hipMemcpy(dtime, &htime, sizeof(uint64_t), hipMemcpyHostToDevice);
            hipLaunchKernelP(measure_width[test_idx], dim3(1), dim3(b * warp_size),
                0, 0, dtime);
            hipStreamSynchronize(0);
            hipMemcpy(&htime, dtime, sizeof(uint64_t), hipMemcpyDeviceToHost);
            times_raw[b-1] = htime;
        }
        for (int c=0; c<iter_num; c++) {
            T_at_c[c] = (float)times_raw[c] / (float)times_raw[0];
        }
        float n_bot = 0;
        for (int c=0; c<iter_num; c++) {
            float inv_slope = (float)(c+1) / T_at_c[c];
            n_bot = n_bot>inv_slope?n_bot:inv_slope;
        }
        float bottleneck_thruput = n_bot / latency;
        sprintf(buf, "f(c) of %s op.", name);
        char buf2[50];
        sprintf(buf2, "n_bot=%.3f", n_bot);
        write_graph_data_with_line(
            buf, iter_num, "c", 1, 1, "f", T_at_c, 1.0f/n_bot, 0, buf2);
        sprintf(buf, "# n_bot = %.3f", n_bot);
        write_line(buf);
        sprintf(buf, "# bottleneck throughput = %.3f [simd insts/cyc/mp]",
            bottleneck_thruput);
        write_line(buf);
    }
}

int main(int argc, char **argv) {
    CHECK(hipSetDevice(0));
    write_init("functional_units");
    for (int i=0; i<NUM_TEST_OPS; i++) {
        printf("Running test for %s op..\n", op_names[i]);
        do_test(i, op_names[i]);
    }
    return 0;
}
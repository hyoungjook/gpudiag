// MANUFACTURER=1 (nvidia), =0 (amd)
#define MANUFACTURER
#define REPORT_FILE
#define KERNEL_FILE

#define warp_size
#define limit_threads_per_block

#define ckpt_fu_latency_repeats

#include "gpudiag_runtime.h"
#include <string.h>

void do_test(int test_idx, const char *name) {
    uint64_t htime, *dtime;
    GDMalloc(&dtime, sizeof(uint64_t));
    // measure period
    htime = 0;
    GDMemcpyHToD(dtime, &htime, sizeof(uint64_t));
    GDLaunchKernel(measure_latency[test_idx], dim3(1), dim3(1), 0, 0, dtime);
    GDSynchronize();
    GDMemcpyDToH(&htime, dtime, sizeof(uint64_t));
    float period = (float)(htime - 1) / (ckpt_fu_latency_repeats - 1);
    char *buf, *buf2;
    asprintf(&buf, "# %d. %s op. results", test_idx+1, name);
    write_line(buf); free(buf);
    asprintf(&buf, "# Period P = %.3f", period);
    write_line(buf); free(buf);
    // measure n_bot
    const int iter_num = limit_threads_per_block(0) / warp_size(0);
    uint64_t times_raw[iter_num];
    float T_at_c[iter_num];
    for (int b=1; b<=iter_num; b++) {
        htime = 0;
        GDMemcpyHToD(dtime, &htime, sizeof(uint64_t));
        GDLaunchKernel(measure_width[test_idx], dim3(1), dim3(b * warp_size(0)),
            0, 0, dtime);
        GDSynchronize();
        GDMemcpyDToH(&htime, dtime, sizeof(uint64_t));
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
    float bottleneck_thruput = n_bot / period;
    asprintf(&buf, "fu(c) of %s op.", name);
    asprintf(&buf2, "n_bot=%.3f", n_bot);
    write_graph_data_with_line(
        buf, iter_num, "c", 1, 1, "fu(c)", T_at_c, 1.0f/n_bot, 0, buf2);
    free(buf); free(buf2);
    asprintf(&buf, "# n_bot = %.3f", n_bot);
    write_line(buf); free(buf);
    asprintf(&buf, "# bottleneck throughput = %.3f [simd insts/cyc/mp]",
        bottleneck_thruput);
    write_line(buf); free(buf);

    // calculate c_br if branch
    if (strcmp(name, "br") == 0) {
        int c_br = -1;
        for (int c=0; c<iter_num; c++) {
            int index_of_2cbr = (c+1)*2-1;
            if (index_of_2cbr >= iter_num) break;
            if (T_at_c[index_of_2cbr] > 1.8 * T_at_c[c]) {
                c_br = c+1; break;
            }
        }
        if (c_br == -1) {
            printf("Failed to find c_br!\n");
            return;
        }
        write_value("c_br", c_br);
    }
}

int main(int argc, char **argv) {
    GDInit();
    write_init("functional_units");
    for (int i=0; i<NUM_TEST_OPS; i++) {
        printf("Running test for %s op..\n", op_names[i]);
        do_test(i, op_names[i]);
    }
    return 0;
}
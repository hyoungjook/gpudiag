// MANUFACTURER=1 (nvidia), =0 (amd)
#define MANUFACTURER
#define REPORT_FILE
#define KERNEL_FILE

#define warp_size
#define c_br

#define ckpt_nslot_timeout_multiplier

#include "gpudiag_runtime.h"
#include <vector>

int main(int argc, char **argv) {
    GDInit();
    write_init("num_mp");

    uint32_t hsync = 0, *dsync;
    GDMalloc(&dsync, sizeof(uint32_t));
    uint64_t *hmptime, *dmptime;
    uint32_t h_isto, *d_isto; // is_timeout
    GDMalloc(&d_isto, sizeof(uint32_t));
    uint64_t *htotclk, *dtotclk;

    std::vector<uint64_t> mp_times;
    uint32_t testG = 1;
    uint64_t mp_avg_until_G = 0;
    uint64_t mp_timeout = 0; // initially, no timeout (=0)
    while (true) {
        GDMemcpyHToD(dsync, &hsync, sizeof(uint32_t));
        hmptime = (uint64_t*)malloc(testG * sizeof(uint64_t));
        GDMalloc(&dmptime, testG * sizeof(uint64_t));
        h_isto = 0;
        GDMemcpyHToD(d_isto, &h_isto, sizeof(uint32_t));
        htotclk = (uint64_t*)malloc(testG * sizeof(uint64_t));
        GDMalloc(&dtotclk, testG * sizeof(uint64_t));

        GDLaunchKernel(measure_num_mp, dim3(testG), dim3(c_br(0) * warp_size(0)),
            0, 0, dsync, testG, dmptime, mp_timeout, d_isto, dtotclk);
        GDSynchronize();

        // decide timeout
        GDMemcpyDToH(htotclk, dtotclk, testG * sizeof(uint64_t));
        uint64_t max = 0;
        for (int i=0; i<testG; i++) max = max > htotclk[i] ? max : htotclk[i];
        max *= ckpt_nslot_timeout_multiplier;
        GDMemcpyDToH(&h_isto, d_isto, sizeof(uint32_t));
        if (testG > 1 && h_isto == 1) {
            GDFree(dmptime); free(hmptime);
            break; // if Nslot @ b=c_br is 1, deadlock will occur @ G=n_mp+1
        }
        mp_timeout = mp_timeout > max ? mp_timeout : max;

        // measure max time, detect sudden increase
        GDMemcpyDToH(hmptime, dmptime, testG * sizeof(uint64_t));
        max = 0;
        for (int i=0; i<testG; i++) max = max > hmptime[i] ? max : hmptime[i];
        mp_times.push_back(max);
        GDFree(dmptime); free(hmptime);
        
        if (testG == 1) mp_avg_until_G = max;
        else {
            if ((float)max > 1.5 * (float)mp_avg_until_G) {
                break; // if Nslot @ b=c_br is >=2, time will be x2 @ G=n_mp+1
            }
            mp_avg_until_G = (mp_avg_until_G * (testG-1) + max) / testG;
        }
        testG++;
    }
    uint32_t num_mp = testG - 1;
    write_line("# 1. Raw data");
    uint64_t *tmp_data = (uint64_t*)malloc(mp_times.size() * sizeof(uint64_t));
    for (int i=0; i<mp_times.size(); i++) tmp_data[i] = mp_times[i];
    write_graph_data("N_mp results", mp_times.size(),
        "GridDim", 1, 1, "latency", tmp_data);
    free(tmp_data);
    write_value("num_mp", num_mp);
    GDFree(dsync);
    GDFree(d_isto);

    return 0;
}
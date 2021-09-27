// MANUFACTURER=1 (nvidia), =0 (amd)
#define MANUFACTURER
#define REPORT_FILE
#define KERNEL_FILE

#define warp_size
#define limit_threads_per_block
#define num_mp

#define ckpt_nslot_timeout_multiplier

#include "gpudiag_runtime.h"
#include "measure_nslot.h"

int main(int argc, char **argv) {
    GDInit();
    write_init("warpstate_buffer");

    const int iter_num = limit_threads_per_block(0) / warp_size(0);
    int Nbuf[iter_num];
    float inv_b[iter_num];
    for (int b=1; b<=iter_num; b++) {
        Nbuf[b-1] = measure_nslot(num_mp(0), b*warp_size(0), measure_warpstatebuffer);
        inv_b[b-1] = 1.0f / (float)b;
    }
    // analyze
    bool is_N1_inf_at_1 = false;
    uint32_t N1, N2 = 0;
    for (int b=1; b<=iter_num; b++) N2 = MAX(N2, b*Nbuf[b-1]);
    if (Nbuf[0] == Nbuf[1]) {N1 = Nbuf[0];}
    else {
        if (Nbuf[1] == N2 / 2) {N1 = Nbuf[0];}
        else {N1 = Nbuf[1]; is_N1_inf_at_1 = true;}
    }
    // write
    write_line("# 1. Raw data");
    write_graph_data_xs_with_line("Warp state buffer", iter_num,
        "1/b", inv_b, "Nslot", Nbuf, N2, 0, "max slope");
    if (is_N1_inf_at_1) write_line("## N1 is infinity at b=1");
    write_value("barrier_buffer_size", N1);
    write_value("warp_state_buffer_size", N2);
    write_values("nslot_n1n2", Nbuf, iter_num);

    return 0;
}
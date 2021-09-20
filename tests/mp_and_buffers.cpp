// MANUFACTURER=1 (nvidia), =0 (amd)
#define MANUFACTURER
#define REPORT_DIR

#define warp_size
#define limit_threads_per_block
#define limit_sharedmem_per_block
#define limit_registers_per_thread
#define LRpB_test_info0
#define LRpB_test_data0
#define LRpB_test_info1
#define LRpB_test_data1

#define ckpt_nslot_timeout_multiplier
#define ckpt_shared_memory_test_granularity
#define ckpt_register_test_granularity
#define ckpt_mab_skip_num_mp_and_use
#define ckpt_mab_skip_wsb_and_use_n1
#define ckpt_mab_skip_wsb_and_use_n2
#define ckpt_mab_skip_wsb_and_use_nat1
#define ckpt_mab_skip_shmem
#define ckpt_mab_skip_regfile

#include "tool.h"
#include "mp_and_buffers.h"
#include <vector>

// returns num_mp
uint32_t num_mp_test();
// returns barrier_buffer_size
void warpstatebuffer_test(uint32_t num_mp, int *out_Nslots);
void sharedmemory_test(uint32_t num_mp, int Nslot_n1n2_at1);
template <int regtype>
void registerfile_test(uint32_t num_mp, int *in_Nslot_n1n2);

uint64_t test_nslot(uint32_t G, uint32_t B, uint64_t timeout,
    void (*kernel)(uint32_t*,uint32_t,uint64_t,uint32_t*,uint64_t*));
int measure_nslot(uint32_t num_mp, uint32_t B,
    void (*kernel)(uint32_t*,uint32_t,uint64_t,uint32_t*,uint64_t*));

int main(int argc, char **argv) {
    CHECK(hipSetDevice(0));
    write_init("mp_and_buffers");
    uint32_t num_mp = 0, barrier_buffer_size = 0;

    if (ckpt_mab_skip_num_mp_and_use <= 0) {
        printf("Running num_mp test..\n");
        num_mp = num_mp_test();
    }
    else num_mp = ckpt_mab_skip_num_mp_and_use;

    int Nslot_n1n2[limit_threads_per_block / warp_size];
    if (ckpt_mab_skip_wsb_and_use_n1 <= 0 ||
        ckpt_mab_skip_wsb_and_use_n2 <= 0 ||
        ckpt_mab_skip_wsb_and_use_nat1 <= 0) {
        printf("Running warpstatebuffer test..\n");
        warpstatebuffer_test(num_mp, Nslot_n1n2);
    }
    else { // fill in Nslot_n1n2 using ckpt values
        Nslot_n1n2[0] = ckpt_mab_skip_wsb_and_use_nat1;
        for (int b=2; b<=limit_threads_per_block/warp_size; b++) {
            Nslot_n1n2[b-1] = MIN(ckpt_mab_skip_wsb_and_use_n1,
                ckpt_mab_skip_wsb_and_use_n2 / b);
        }
    }

    if (ckpt_mab_skip_shmem <= 0) {
        printf("Running sharedmemory test..\n");
        sharedmemory_test(num_mp, Nslot_n1n2[0]);
    }

    if (ckpt_mab_skip_regfile <= 0) {
        printf("Running registerfile test..\n");
        registerfile_test<0>(num_mp, Nslot_n1n2);
#if MANUFACTURER == 0
        registerfile_test<1>(num_mp, Nslot_n1n2);
#endif
    }
    
    return 0;
}

uint32_t num_mp_test() {
    const int iter_num = limit_threads_per_block / warp_size;
    // Measure n_eff of BR
    uint64_t htime, *dtime;
    hipMalloc(&dtime, sizeof(uint64_t));
    uint64_t br_times_raw[iter_num];
    float T_br_at_c[iter_num];
    for (int b=1; b<=iter_num; b++) {
        htime = 0;
        hipMemcpy(dtime, &htime, sizeof(uint64_t), hipMemcpyHostToDevice);
        hipLaunchKernelP(
            measure_width_br, dim3(1), dim3(b * warp_size), 0, 0, dtime);
        hipStreamSynchronize(0);
        hipMemcpy(&htime, dtime, sizeof(uint64_t), hipMemcpyDeviceToHost);
        br_times_raw[b-1] = htime;
    }
    for (int c=0; c<iter_num; c++) {
        T_br_at_c[c] = (float)br_times_raw[c] / (float)br_times_raw[0];
    }
    write_line("# 1. Branch op. n_eff raw data");
    write_graph_data("f(c) of branch op.", iter_num, "c", 1, 1,
        "measured f", T_br_at_c);

    // Calculate c_br
    int c_br = -1;
    for (int c=0; c<iter_num; c++) {
        int index_of_2cbr = (c+1)*2-1;
        if (index_of_2cbr >= iter_num) break;
        if (T_br_at_c[index_of_2cbr] > 1.8 * T_br_at_c[c]) {
            c_br = c+1; break;
        }
    }
    if (c_br == -1) {
        printf("Failed to find c_br!\n");
        exit(1);
    }
    write_line("# 2. c_br result");
    char buf[30];
    sprintf(buf, "# c_br: %d", c_br);
    write_line(buf);
    hipFree(dtime);

    // Measure n_mp
    uint32_t hsync = 0, *dsync;
    hipMalloc(&dsync, sizeof(uint32_t));
    uint64_t *hmptime, *dmptime;
    uint32_t h_isto, *d_isto; // is_timeout
    hipMalloc(&d_isto, sizeof(uint32_t));
    uint64_t *htotclk, *dtotclk;

    std::vector<uint64_t> mp_times;
    uint32_t testG = 1;
    uint64_t mp_avg_until_G = 0;
    uint64_t mp_timeout = 0; // initially, no timeout (=0)
    while (true) {
        hipMemcpy(dsync, &hsync, sizeof(uint32_t), hipMemcpyHostToDevice);
        hmptime = (uint64_t*)malloc(testG * sizeof(uint64_t));
        hipMalloc(&dmptime, testG * sizeof(uint64_t));
        h_isto = 0;
        hipMemcpy(d_isto, &h_isto, sizeof(uint32_t), hipMemcpyHostToDevice);
        htotclk = (uint64_t*)malloc(testG * sizeof(uint64_t));
        hipMalloc(&dtotclk, testG * sizeof(uint64_t));

        hipLaunchKernelP(measure_num_mp, dim3(testG), dim3(c_br * warp_size),
            0, 0, dsync, testG, dmptime, mp_timeout, d_isto, dtotclk);
        hipStreamSynchronize(0);

        // decide timeout
        hipMemcpy(htotclk, dtotclk, testG * sizeof(uint64_t), hipMemcpyDeviceToHost);
        uint64_t max = 0;
        for (int i=0; i<testG; i++) max = max > htotclk[i] ? max : htotclk[i];
        max *= ckpt_nslot_timeout_multiplier;
        hipMemcpy(&h_isto, d_isto, sizeof(uint32_t), hipMemcpyDeviceToHost);
        if (testG > 1 && h_isto == 1) {
            hipFree(dmptime); free(hmptime);
            break; // if Nslot @ b=c_br is 1, deadlock will occur @ G=n_mp+1
        }
        mp_timeout = mp_timeout > max ? mp_timeout : max;

        // measure max time, detect sudden increase
        hipMemcpy(hmptime, dmptime, testG * sizeof(uint64_t), hipMemcpyDeviceToHost);
        max = 0;
        for (int i=0; i<testG; i++) max = max > hmptime[i] ? max : hmptime[i];
        mp_times.push_back(max);
        hipFree(dmptime); free(hmptime);
        
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
    write_line("# 3. num_mp raw data");
    uint64_t *tmp_data = (uint64_t*)malloc(mp_times.size() * sizeof(uint64_t));
    for (int i=0; i<mp_times.size(); i++) tmp_data[i] = mp_times[i];
    write_graph_data("N_mp measurements", mp_times.size(),
        "GridDim", 1, 1, "latency", tmp_data);
    free(tmp_data);
    write_value("num_mp", num_mp);
    hipFree(dsync);
    hipFree(d_isto);
    return num_mp;
}

void warpstatebuffer_test(uint32_t num_mp, int *out_Nslots) {
    const int iter_num = limit_threads_per_block / warp_size;
    int Nbuf[iter_num];
    float inv_b[iter_num];
    for (int b=1; b<=iter_num; b++) {
        Nbuf[b-1] = measure_nslot(num_mp, b*warp_size, measure_warpstatebuffer);
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
    write_line("# 4. Warp state buffer Nslot raw data");
    write_graph_data_xs_with_line("Warp state buffer", iter_num,
        "1/b", inv_b, "Nslot", Nbuf, N2, 0, "max slope");
    if (is_N1_inf_at_1) write_line("## N1 is infinity at b=1");
    write_value("barrier_buffer_size", N1);
    write_value("warp_state_buffer_size", N2);
    for (int i=0; i<iter_num; i++) out_Nslots[i] = Nbuf[i];
}

bool shmem_data_consistent(uint64_t S, uint64_t s_min, int N, int *nbuf,
        int shmem_unit, int Nslot_n1n2_at1) {
    for (int i=0; i<N; i++) {
        if (nbuf[i] >= Nslot_n1n2_at1) continue;
        uint64_t test_s = (i+1) * shmem_unit;
        test_s = (test_s + s_min - 1) / s_min * s_min; // ceil(test_s, s_min)
        if (nbuf[i] != S / test_s) return false;
    }
    return true;
}

void sharedmemory_test(uint32_t num_mp, int Nslot_n1n2_at1) {
    const int shmem_unit = ckpt_shared_memory_test_granularity;
    const int iter_num = limit_sharedmem_per_block / shmem_unit;
    int Nbuf[iter_num];
    float inv_s[iter_num];
    for (int i=0; i<iter_num; i++) {
        Nbuf[i] = measure_nslot(num_mp, 1, measure_shmem[i]);
        inv_s[i] = 1.0f / (float)((i+1)*shmem_unit);
    }
    // analyze
    uint64_t S = 0;
    for (int i=0; i<iter_num; i++) S = MAX(S, (i+1)*shmem_unit*Nbuf[i]);
    uint64_t s_min = shmem_unit;
    bool s_min_strange = false;
    while(true) {
        if (shmem_data_consistent(S, s_min, iter_num, Nbuf,
                shmem_unit, Nslot_n1n2_at1)) {
            break;
        }
        s_min *= 2;
        if (s_min >= S) {s_min_strange = true; break;}
    }
    // write
    write_line("# 5. Shared memory Nslot raw data");
    write_graph_data_xs_with_line("Shared memory", iter_num,
        "1/s (1/Bytes)", inv_s, "Nslot", Nbuf, S, 0, "max slope");
    write_value("shared_memory_size", S);
    if (s_min_strange) write_line("## cannot measure s_min automatically..");
    else {
        if (s_min == shmem_unit) write_line("## s_min is less or equal than below!");
        write_value("shared_memory_alloc_unit", s_min);
    }
}

bool regfile_data_consistent(uint64_t R, uint64_t r_min, int min_R, int reg_unit,
        int Nr, int Nb, int (*nbuf)[limit_threads_per_block/warp_size], int *Nslot_n1n2) {
    for (int i=0; i<Nr; i++) {
        for (int b=1; b<=Nb; b++) {
            if (nbuf[i][b-1] < 0) continue;
            if (nbuf[i][b-1] >= Nslot_n1n2[b-1]) continue;
            uint64_t test_r = min_R + i * reg_unit;
            test_r = (test_r + r_min - 1) / r_min * r_min; // ceil(test_r, r_min)
            if (nbuf[i][b-1] != R / (test_r * b)) return false;
        }
    }
    return true;
}

template <int regtype>
void registerfile_test(uint32_t num_mp, int *in_Nslot_n1n2) {
    const int num_regs = regtype==0?LRpB_test_info0(0):LRpB_test_info1(0);
    const int min_R = regtype==0?LRpB_test_info0(1):LRpB_test_info1(1);
    const int reg_unit = regtype==0?LRpB_test_info0(2):LRpB_test_info1(2);
    if (reg_unit != ckpt_register_test_granularity) {
        printf("register_test_granularity in config_ckpt.py not consistent with "
            "LRpB_test_info in result.txt!\n"
            "Please change ckpt value or run kernel_limits again.\n");
        exit(1);
    }
    const int max_possible_b = limit_threads_per_block / warp_size;
#if MANUFACTURER == 1
    uint32_t regkernel_min = REGKERNEL_REG_MIN_R;
#else
    uint32_t regkernel_min = regtype==0?REGKERNEL_SREG_MIN_R:REGKERNEL_VREG_MIN_R;
#endif
    int Nbuf[num_regs][max_possible_b];
    for (int i=0; i<num_regs; i++) for (int j=0; j<max_possible_b; j++) Nbuf[i][j]=-1;


    int num_nonzero_data = 0;
    int regkern_idx = 0;
    for (int i=0; i<num_regs; i++) {
        uint32_t testR = min_R + i * reg_unit;
        if (testR < regkernel_min) continue;
        uint32_t max_b_at_R = regtype==0?LRpB_test_data0(i):LRpB_test_data1(i);
        for (int b=1; b<=max_b_at_R; b++) {
#if MANUFACTURER == 1
            Nbuf[i][b-1] = measure_nslot(num_mp, b*warp_size, measure_reg[regkern_idx]);
#else
            if (regtype == 0)
                Nbuf[i][b-1] = measure_nslot(num_mp, b*warp_size, measure_sreg[regkern_idx]);
            else
                Nbuf[i][b-1] = measure_nslot(num_mp, b*warp_size, measure_vreg[regkern_idx]);
#endif
            num_nonzero_data++;
        }
        regkern_idx++;
    }

    // analyze
    uint64_t R = 0;
    int *serialNbuf = (int*)malloc(num_nonzero_data * sizeof(int));
    float *serialInvRB = (float*)malloc(num_nonzero_data * sizeof(float));
    int serialIdx = 0;
    for (int i=0; i<num_regs; i++) {
        uint32_t testR = min_R + i * reg_unit;
        if (testR < regkernel_min) continue;
        for (int b=1; b<=max_possible_b; b++) {
            if (Nbuf[i][b-1] < 0) continue;
            R = MAX(R, testR * b * Nbuf[i][b-1]);
            serialNbuf[serialIdx] = Nbuf[i][b-1];
            serialInvRB[serialIdx] = 1.0f / (float)(testR * b); serialIdx++;
        }
    }
    uint64_t r_min = reg_unit;
    bool r_min_strange = false;
    while(true) {
        if (regfile_data_consistent(R, r_min, min_R, reg_unit,
                num_regs, max_possible_b, Nbuf, in_Nslot_n1n2)) {
            break;
        }
        r_min *= 2;
        if (r_min >= R) {r_min_strange = true; break;}
    }
    // write
#if MANUFACTURER == 1
    write_line("# 6. Register file Nslot raw data");
    write_graph_data_xs_with_line("Register file size", num_nonzero_data,
        "1/(rb)", serialInvRB, "Nslot", serialNbuf, R, 0, "max slope");
    write_value("register_file_size", R);
    if (r_min_strange) write_line("## cannot measure r_min automatically..");
    else {
        if (r_min == reg_unit) write_line("## r_min is less or equal than below!");
        write_value("register_file_alloc_unit", r_min);
    }
#else
    if (regtype == 0) {
        write_line("# 6. Scalar register file Nslot raw data");
        write_graph_data_xs_with_line("Scalar register file size", num_nonzero_data,
            "1/(rb)", serialInvRB, "Nslot", serialNbuf, R, 0, "max slope");
        write_value("scalar_register_file_size", R);
        if (r_min_strange) write_line("## cannot measure r_s,min automatically..");
        else {
            if (r_min == reg_unit) write_line("## r_s,min is less or equal than below!");
            write_value("scalar_register_file_alloc_unit", r_min);
        }
    }
    else {
        write_line("# 7. Vector register file Nslot raw data");
        write_graph_data_xs_with_line("Vector register file size", num_nonzero_data,
            "1/(rb)", serialInvRB, "Nslot", serialNbuf, R, 0, "max_slope");
        write_value("vector_register_file_size", R);
        if (r_min_strange) write_line("## cannot measure r_v,min automatically..");
        else {
            if (r_min == reg_unit) write_line("## r_v,min is less or equal than below!");
            write_value("vectorr_register_file_alloc_unit", r_min);
        }
    }
#endif
    free(serialNbuf); free(serialInvRB);
}

uint64_t test_nslot(uint32_t G, uint32_t B, uint64_t timeout,
        void (*kernel)(uint32_t*,uint32_t,uint64_t,uint32_t*,uint64_t*)) {
    uint32_t hsync = 0, *dsync, h_isto = 0, *d_isto;
    hipMalloc(&dsync, sizeof(uint32_t));
    hipMalloc(&d_isto, sizeof(uint32_t));
    hipMemcpy(dsync, &hsync, sizeof(uint32_t), hipMemcpyHostToDevice);
    hipMemcpy(d_isto, &h_isto, sizeof(uint32_t), hipMemcpyHostToDevice);
    uint64_t *hclks, *dclks;
    hclks = (uint64_t*)malloc(G * sizeof(uint64_t));
    hipMalloc(&dclks, G * sizeof(uint64_t));

    hipLaunchKernelP(kernel, dim3(G), dim3(B), 0, 0,
        dsync, G, timeout, d_isto, dclks);
    hipStreamSynchronize(0);
    hipMemcpy(&h_isto, d_isto, sizeof(uint32_t), hipMemcpyDeviceToHost);
    hipMemcpy(hclks, dclks, G * sizeof(uint64_t), hipMemcpyDeviceToHost);

    uint64_t retval;
    if (h_isto == 1) retval = 0;
    else {
        uint64_t max = 0;
        for (int i=0; i<G; i++) max = max>hclks[i]?max:hclks[i];
        retval = max;
    }
    hipFree(dsync); hipFree(d_isto); hipFree(dclks); free(hclks);
    return retval;
}
int measure_nslot(uint32_t num_mp, uint32_t B,
        void (*kernel)(uint32_t*,uint32_t,uint64_t,uint32_t*,uint64_t*)) {
    const int timeout_threshold = ckpt_nslot_timeout_multiplier;
    // initial timeout
    uint64_t timeout = test_nslot(num_mp, B, 0, kernel) * timeout_threshold;
    int testN = 1;
    while(true) {
        uint32_t testG = testN * num_mp + 1;
        uint64_t ret = test_nslot(testG, B, timeout, kernel) * timeout_threshold;
        if (ret == 0) break;
        else timeout = timeout>ret?timeout:ret;
        testN++;
    }
    return testN;
}

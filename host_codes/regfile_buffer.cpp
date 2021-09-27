// MANUFACTURER=1 (nvidia), =0 (amd)
#define MANUFACTURER
#define REPORT_FILE
#define KERNEL_FILE

#define warp_size
#define limit_threads_per_block
#define LRpB_test_info0
#define LRpB_test_data0
#define LRpB_test_info1
#define LRpB_test_data1
#define num_mp
#define nslot_n1n2

#define ckpt_nslot_timeout_multiplier
#define ckpt_register_test_granularity

#include "gpudiag_runtime.h"
#include "measure_nslot.h"

bool regfile_data_consistent(uint64_t R, uint64_t r_min, int min_R, int reg_unit,
        int Nr, int Nb, int (*nbuf)[limit_threads_per_block(0)/warp_size(0)]) {
    for (int i=0; i<Nr; i++) {
        for (int b=1; b<=Nb; b++) {
            if (nbuf[i][b-1] < 0) continue;
            if (nbuf[i][b-1] >= nslot_n1n2(b-1)) continue;
            uint64_t test_r = min_R + i * reg_unit;
            test_r = (test_r + r_min - 1) / r_min * r_min; // ceil(test_r, r_min)
            if (nbuf[i][b-1] != R / (test_r * b)) return false;
        }
    }
    return true;
}

template <int regtype>
void registerfile_test() {
    const int num_regs = regtype==0?LRpB_test_info0(0):LRpB_test_info1(0);
    const int min_R = regtype==0?LRpB_test_info0(1):LRpB_test_info1(1);
    const int reg_unit = regtype==0?LRpB_test_info0(2):LRpB_test_info1(2);
    if (reg_unit != ckpt_register_test_granularity) {
        printf("register_test_granularity in config_ckpt.py not consistent with "
            "LRpB_test_info in result.txt!\n"
            "Please change ckpt value or run kernel_limits again.\n");
        exit(1);
    }
    const int max_possible_b = limit_threads_per_block(0) / warp_size(0);
    uint32_t regkernel_min = 
#if MANUFACTURER == 0
        regtype==1 ? REGKERNEL_REG1_MIN_R :
#endif
        REGKERNEL_REG0_MIN_R;
    int Nbuf[num_regs][max_possible_b];
    for (int i=0; i<num_regs; i++) for (int j=0; j<max_possible_b; j++) Nbuf[i][j]=-1;

    int num_nonzero_data = 0;
    int regkern_idx = 0;
    for (int i=0; i<num_regs; i++) {
        uint32_t testR = min_R + i * reg_unit;
        if (testR < regkernel_min) continue;
        uint32_t max_b_at_R = regtype==0?LRpB_test_data0(i):LRpB_test_data1(i);
        for (int b=1; b<=max_b_at_R; b++) {
            if (regtype == 0)
                Nbuf[i][b-1] = measure_nslot(num_mp(0), b*warp_size(0), measure_reg0[regkern_idx]);
#if MANUFACTURER == 0
            else
                Nbuf[i][b-1] = measure_nslot(num_mp(0), b*warp_size(0), measure_reg1[regkern_idx]);
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
                num_regs, max_possible_b, Nbuf)) {
            break;
        }
        r_min *= 2;
        if (r_min >= R) {r_min_strange = true; break;}
    }
    // write
    char *buf;
    asprintf(&buf, "# %d. Register file type%d raw data", regtype+1, regtype);
    write_line(buf); free(buf);
    asprintf(&buf, "Register file type%d size", regtype);
    write_graph_data_xs_with_line(buf, num_nonzero_data,
        "1/(rb)", serialInvRB, "Nslot", serialNbuf, R, 0, "max slope");
    free(buf);
    asprintf(&buf, "register_file_type%d_size", regtype);
    write_value(buf, R); free(buf);
    if (r_min_strange) write_line("## cannot measure r_min automatically..");
    else {
        if (r_min == reg_unit) write_line("## r_min is less or equal than below!");
        asprintf(&buf, "register_file_type%d_alloc_unit", regtype);
        write_value(buf, r_min); free(buf);
    }

    free(serialNbuf); free(serialInvRB);
}

int main(int argc, char **argv) {
    GDInit();
    write_init("regfile_buffer");

    registerfile_test<0>();
#if MANUFACTURER == 0
    registerfile_test<1>();
#endif

    return 0;
}
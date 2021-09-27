#define MANUFACTURER
#define REPORT_FILE
#define KERNEL_FILE

#include "gpudiag_runtime.h"
#include "kernel_limits_launchable_host.h"

uint32_t get_max_true(bool (*chkfunc)(uint32_t,uint32_t), 
        uint32_t initval,uint32_t arg) {
    uint32_t testval = initval, max_true_val, min_false_val;
    if (chkfunc(testval, arg)) {
        max_true_val = testval;
        if (!chkfunc(testval+1, arg)) min_false_val = testval+1;
        else {
            while (true) {
                testval *= 2;
                if (!chkfunc(testval, arg)) {min_false_val = testval; break;}
            }
        }
    }
    else {
        min_false_val = testval;
        while(true) {
            testval /= 2;
            if (chkfunc(testval, arg)) {{max_true_val = testval; break;}}
        }
    }
    while (true) {
        if (max_true_val+1 == min_false_val) break;
        uint32_t mid_val = (max_true_val + min_false_val) / 2;
        if (chkfunc(mid_val, arg)) max_true_val = mid_val;
        else min_false_val = mid_val;
    }
    return max_true_val;
}
uint32_t get_min_true_from0(bool (*chkfunc)(uint32_t)) {
    uint32_t testidx = 0;
    while (true) {
        if (chkfunc(testidx)) break;
        testidx++;
    }
    return testidx;
}

// test thread limits
bool chkfunc_LTpB(uint32_t val, uint32_t arg) {return chk_thr(1, val);}
bool chkfunc_LTpG(uint32_t val, uint32_t arg) {return chk_thr(val, 1);}
void do_thr_chk(uint32_t initLTpB, uint32_t initLTpG) {
    uint32_t verifiedLTpB = get_max_true(chkfunc_LTpB, initLTpB, 0);
    write_value("limit_threads_per_block", verifiedLTpB);
    uint32_t verifiedLTpG = initLTpG;
    //uint32_t verifiedLTpG = get_max_true(chkfunc_LTpG, initLTpG, 0);
    write_value("limit_threads_per_grid", verifiedLTpG);
}

// test shmem limits
bool chkfunc_LSpB(uint32_t idx) {return chk_shm(1, 1, idx);}
void do_shm_chk(uint32_t maxLSpB, uint32_t LSpB_unit) {
    uint32_t verifiedLSpBidx = get_min_true_from0(chkfunc_LSpB);
    uint32_t verifiedLSpB = maxLSpB - LSpB_unit * verifiedLSpBidx;
    write_value("limit_sharedmem_per_block", verifiedLSpB);
}

// test reg limits
const char *test_info_str[2] = {"LRpB_test_info0", "LRpB_test_info1"};
const char *test_data_str[2] = {"LRpB_test_data0", "LRpB_test_data1"};
uint32_t do_LRpB_test(uint32_t LRpT, uint32_t Reg_unit, uint32_t initLRpB,
        bool (*chkfunc)(uint32_t,uint32_t), const char* title, const char* xlabel,
        uint32_t regmin, int infoidx) {
    uint32_t verifiedLRpB = 0;
    uint32_t *data = (uint32_t*)malloc((LRpT/Reg_unit)*sizeof(uint32_t));
    // measure for testR = m * Reg_unit && testR > regmin + 16
    uint32_t testR = 0, num_tests = 0, min_testR = 0, regkern_idx = 0;
    while (true) {
        testR += Reg_unit;
        if (testR <= regmin + 16) continue;
        if (testR > LRpT) break;
        num_tests++; if (min_testR == 0) min_testR = testR;
        // measure max_b for testR
        uint32_t max_b = get_max_true(chkfunc, initLRpB/testR, regkern_idx);
        data[regkern_idx] = max_b; regkern_idx++;
        verifiedLRpB = verifiedLRpB>max_b*testR?verifiedLRpB:max_b*testR;
    }
    write_graph_data(title, num_tests, xlabel, min_testR, Reg_unit,
        "max B", data);
    uint32_t test_info[3] = {num_tests, min_testR, Reg_unit};
    write_values(test_info_str[infoidx], test_info, 3);
    write_values(test_data_str[infoidx], data, num_tests);
    free(data);
    return verifiedLRpB;
}
#if MANUFACTURER == 1
bool chkfunc_LRpT(uint32_t idx) {return chk_reg_LRpT(1, 1, idx);}
bool chkfunc_LRpB(uint32_t val, uint32_t arg) {return chk_reg_LRpB(1,val*warp_size,arg);}
void do_reg_chk(uint32_t maxLRpT, uint32_t Reg_unit, uint32_t initLRpB, uint32_t regmin) {
    uint32_t verifiedLRpTidx = get_min_true_from0(chkfunc_LRpT);
    uint32_t verifiedLRpT = maxLRpT - Reg_unit * verifiedLRpTidx;
    if (maxLRpT%Reg_unit!=0 && verifiedLRpTidx>0)
        verifiedLRpT = maxLRpT - (maxLRpT%Reg_unit) - (verifiedLRpTidx-1)*Reg_unit;
    write_value("limit_registers_per_thread", verifiedLRpT);
    uint32_t verifiedLRpB = do_LRpB_test(verifiedLRpT, Reg_unit, initLRpB,
        chkfunc_LRpB, "Regs per Block", "regs/thread", regmin, 0);
    write_value("LRpB_test_info1", 0); write_value("LRpB_test_data1", 0);
    write_value("limit_registers_per_block", verifiedLRpB);
}
#else
bool chkfunc_LsRpT(uint32_t idx) {return chk_sreg_LRpT(1, 1, idx);}
bool chkfunc_LvRpT(uint32_t idx) {return chk_vreg_LRpT(1, 1, idx);}
bool chkfunc_LsRpB(uint32_t val, uint32_t arg) {return chk_sreg_LRpB(1,val*warp_size,arg);}
bool chkfunc_LvRpB(uint32_t val, uint32_t arg) {return chk_vreg_LRpB(1,val*warp_size,arg);}
void do_reg_chk(uint32_t maxLsRpT, uint32_t maxLvRpT, 
        uint32_t Reg_unit, uint32_t initLsRpB, uint32_t initLvRpB, uint32_t regmin) {
    uint32_t verifiedLsRpTidx = get_min_true_from0(chkfunc_LsRpT);
    uint32_t verifiedLvRpTidx = get_min_true_from0(chkfunc_LvRpT);
    uint32_t verifiedLRpT[2] = {
        maxLsRpT - Reg_unit * verifiedLsRpTidx,
        maxLvRpT - Reg_unit * verifiedLvRpTidx
    };
    if (maxLsRpT%Reg_unit!=0 && verifiedLsRpTidx>0)
        verifiedLRpT[0] = maxLsRpT - (maxLsRpT%Reg_unit) - (verifiedLsRpTidx-1)*Reg_unit;
    if (maxLvRpT%Reg_unit!=0 && verifiedLvRpTidx>0)
        verifiedLRpT[1] = maxLvRpT - (maxLvRpT%Reg_unit) - (verifiedLvRpTidx-1)*Reg_unit;
    write_values("limit_registers_per_thread", verifiedLRpT, 2);
    uint32_t verifiedLRpB[2] = {
        do_LRpB_test(verifiedLRpT[0], Reg_unit, initLsRpB, chkfunc_LsRpB,
            "SRegs per Block", "sregs/thread", regmin, 0),
        do_LRpB_test(verifiedLRpT[1], Reg_unit, initLvRpB, chkfunc_LvRpB,
            "VRegs per Block", "vregs/thread", regmin, 1)
    };
    write_values("limit_registers_per_block", verifiedLRpB, 2);
}
#endif

int main(int argc, char **argv) {
    GDInit();
    write_init("kernel_limits");
    do_thr_chk(deliver_propLTpB, deliver_propLTpG);
    do_shm_chk(deliver_compilableLSpB, deliver_LSpB_test_unit);
#if MANUFACTURER == 1
    do_reg_chk(deliver_compilableLRpT0, deliver_Reg_test_unit,
        deliver_propLRpB, deliver_regmin);
#else
    do_reg_chk(deliver_compilableLRpT0, deliver_compilableLRpT1,
        deliver_Reg_test_unit, deliver_propLRpB0, deliver_propLRpB1, deliver_regmin);
#endif
    return 0;
}
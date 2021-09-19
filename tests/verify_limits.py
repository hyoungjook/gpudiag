import subprocess
import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from define import Feature
import config_env as conf
import config_ckpt as ckpt
import kernels.reused_codes as tool

## test kernels: assures provided resource usage.
## on normal execution, saves 1 to *r and exits.
def include():
    return """\
#include "hip/hip_runtime.h"
#include <stdint.h>
"""

def return_if_real():
    return """\
if (real) {
    if (hipThreadIdx_x == 0 && hipBlockIdx_x == 0) *r = 1;
    return;
}\n"""

def normal_kernel(kernel_name):
    return """\
__global__ void {}(uint8_t *r, bool real) {{
    {}
}}
""".format(kernel_name, return_if_real())

def shmem_kernel(kernel_name, shmem_size):
    return """\
__global__ void {}(uint8_t *r, bool real) {{
    {}
    __shared__ uint8_t arr[{}];
    for (int i=0; i<{}; i++) arr[i] = (uint8_t)clock();
    for (int i=0; i<{}; i++) r[i] = arr[i];
}}\n""".format(kernel_name, return_if_real(), shmem_size, shmem_size, shmem_size)

def reg_kernel_nvidia(kernel_name, reg_est):
    # ensures more reg than provided; must be checked with res-usage
    code = """\
__global__ void {}(uint8_t *r, bool real){{
    {}
    asm volatile(".reg .b32 tr<{}>;\\n");
    asm volatile(".reg .b64 addr;\\n");
    asm volatile("mov.u64 addr, %0;\\n"::"l"(r));
    asm volatile(
""".format(kernel_name, return_if_real(), reg_est)
    for i in range(reg_est):
        code += "\"mov.u32 tr{}, %clock;\\n\"\n".format(i)
    code += "\t);\n__syncthreads();\n\tasm volatile(\n"
    for i in range(reg_est):
        code += "\"st.global.u32 [addr+{}], tr{};\\n\"\n"\
            .format(4*i, i)
    code += "\t);\n}\n"
    return code

def find_regmin_nvidia(compilableLRpT, args):
    # (reg_est for reg_kernel_nvidia) = actual_reg_usage + regmin!
    for i in range(compilableLRpT):
        testR = compilableLRpT - i
        tool.write_code(include() + reg_kernel_nvidia("test", testR) +\
                main_check_compilable("test"), args[2])
        regcnts = tool.compile_and_check_resource(args[0], args[2], args[3])
        if regcnts[0] == 0:
            return compilableLRpT - testR

def reg_kernel_nvidia_real_usage(kernel_name, actual_reg, regmin):
    # regmin should be found by find_regmin_nvidia
    if actual_reg <= regmin:
        return normal_kernel(kernel_name) # normal kernel for such small regs!
    return reg_kernel_nvidia(kernel_name, actual_reg - regmin)

def reg_kernel_amd(kernel_name, sregs, vregs):
    code = """\
__global__ void {}(uint8_t *r, bool real){{
    {}\n""".format(kernel_name, return_if_real())
    # ensure 'n_reg'-th reg as operand is compilable
    if sregs > 0:
        code += "\tasm volatile(\"s_mov_b32 s{}, 0\\n\");\n".format(sregs-1)
    if vregs > 0:
        code += "\tasm volatile(\"v_mov_b32 v{}, 0\\n\");\n".format(vregs-1)
    # ensure 'n_reg' registers are actually used
    if sregs > 0:
        code += """\
    uint32_t sdummy[{}];
#pragma unroll {}
    for(int i=0;i<{};i++) asm volatile("s_mov_b32 %0, 0\\n":"=s"(sdummy[i]));
#pragma unroll {}
    for(int i=0;i<{};i++) asm volatile("s_mov_b32 s0, %0\\n"::"s"(sdummy[i]));
""".format(sregs, sregs, sregs, sregs, sregs)
    if vregs > 0:
        code += """\
    uint32_t vdummy[{}];
#pragma unroll {}
    for(int i=0;i<{};i++) asm volatile("v_mov_b32 %0, 0\\n":"=v"(vdummy[i]));
#pragma unroll {}
    for(int i=0;i<{};i++) asm volatile("v_mov_b32 v0, %0\\n"::"v"(vdummy[i]));
""".format(vregs, vregs, vregs, vregs, vregs)
    code += "}\n"
    return code

def make_kernel_array(array_name, N, x0, dx):
    code = "void (*{}[{}])(uint8_t *, bool) = {{\n".format(array_name, N)
    for i in range(N):
        code += "{}_{},\n".format(array_name, x0 - i * dx)
    code += "};\n"
    return code

def main_check_compilable(kernel_name):
    return """\
int main(int argc, char **argv) {{
    uint8_t hi;
    hipLaunchKernelGGL({}, dim3(1), dim3(1), 0, 0, &hi, true);
    return 0;
}}
""".format(kernel_name)

def abort_main_check_launchable(kernel_name, G, B):
    return """\
int main(int argc, char **argv) {{
    uint8_t hv = 0, *dv; hipMalloc(&dv, sizeof(uint8_t));
    hipMemcpy(dv, &hv, sizeof(uint8_t), hipMemcpyHostToDevice);
    hipLaunchKernelGGL({}, dim3({}), dim3({}), 0, 0, dv, true);
    if (hipGetLastError() != hipSuccess) {{hipFree(dv); return 1;}}
    hipStreamSynchronize(0);
    hipMemcpy(&hv, dv, sizeof(uint8_t), hipMemcpyDeviceToHost);
    hipFree(dv);
    if (hv != 1) return 1;
    return 0;
}}\n""".format(kernel_name, G, B)

def no_abort_host_check_launchable(func_name, kernel_name, is_func_array):
    arg_list = "int G, int B"
    kernel_format = kernel_name
    if is_func_array:
        arg_list += ", int idx"
        kernel_format += "[idx]"
    return """\
bool {}({}){{
    uint8_t hv = 0, *dv; hipMalloc(&dv, sizeof(uint8_t));
    hipMemcpy(dv, &hv, sizeof(uint8_t), hipMemcpyHostToDevice);
    hipLaunchKernelP({}, dim3(G), dim3(B), 0, 0, dv, true);
    if (hipGetLastError() != hipSuccess) {{hipFree(dv); return false;}}
    hipStreamSynchronize(0);
    hipMemcpy(&hv, dv, sizeof(uint8_t), hipMemcpyDeviceToHost);
    hipFree(dv);
    if (hv != 1) return false;
    return true;
}}\n""".format(func_name, arg_list, kernel_format)

## check functions
def get_max_true(checkfunc, args, initial_value):
    # checkfunc(val: int, args) = True or False
    # True for val<="max_true", False for val>"max_true"
    # this will find the max_true value
    testval = initial_value
    # find max_true_val, min_false_val
    if checkfunc(testval, args):
        max_true_val = testval
        # speedup: if initial value was right
        if not checkfunc(testval+1, args):
            min_false_val = testval+1
        else: # normal case
            while True:
                testval *= 2
                if not checkfunc(testval, args):
                    min_false_val = testval
                    break
    else:
        min_false_val = testval
        while True:
            testval = testval // 2
            if checkfunc(testval, args):
                max_true_val = testval
                break
    # find the critical point between them
    while True:
        if max_true_val+1 == min_false_val:
            break
        mid_val = (max_true_val + min_false_val) // 2
        if checkfunc(mid_val, args):
            max_true_val = mid_val
        else:
            min_false_val = mid_val
    # found the value
    return max_true_val

def get_max_true_given_maxval(checkfunc, args, max_value, update_unit):
    # similar as above, but max value is given
    testval = max_value
    while True:
        if checkfunc(testval, args):
            break
        testval -= update_unit
    return testval

def get_max_true_code():
    return """\
uint32_t get_max_true(bool (*chkfunc)(uint32_t,uint32_t), uint32_t initval,uint32_t arg) {
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
}\n"""

def get_min_true_from0_code():
    return """\
uint32_t get_min_true_from0(bool (*chkfunc)(uint32_t)) {
    uint32_t testidx = 0;
    while (true) {
        if (chkfunc(testidx)) break;
        testidx++;
    }
    return testidx;
}\n"""

def compilable(code, args):
    tool.write_code(code, args[2])
    return tool.compile_succeed(args[0], args[2], args[3])

def checkfunc_LSpB_compilable(val, args):
    return compilable(
        include() +\
        shmem_kernel("test", val) +\
        main_check_compilable("test"), args)

def checkfunc_LRpT_compilable_nvidia(val, args):
    tool.write_code(
        include() +\
        reg_kernel_nvidia("test", val) +\
        main_check_compilable("test"), args[2])
    regcnts = tool.compile_and_check_resource(args[0], args[2], args[3])
    if regcnts[0] == 0: # no spill regs
        return True
    else: # reg spill occurred
        return val <= regcnts[1]

def checkfunc_LRpT_compilable_amd_sgpr(val, args):
    return compilable(
        include() +\
        reg_kernel_amd("test", val, 0) +\
        main_check_compilable("test"), args)

def checkfunc_LRpT_compilable_amd_vgpr(val, args):
    return compilable(
        include() +\
        reg_kernel_amd("test", 0, val) +\
        main_check_compilable("test"), args)

def runnable(code, args):
    tool.write_code(code, args[2])
    if not tool.compile_succeed(args[0], args[2], args[3]):
        exit(1)
    return tool.run_succeed(args[1], "tmp_obj")

def no_abort_launchable_LRpB_routine():
    return """\
#define INT_TO_LITERAL(i) #i
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
        uint32_t max_b = get_max_true(chkfunc, initLRpB/testR/warp_size, regkern_idx);
        data[regkern_idx] = max_b; regkern_idx++;
        verifiedLRpB = verifiedLRpB>max_b*testR?verifiedLRpB:max_b*testR;
    }
    write_graph_data(title, num_tests, xlabel, min_testR, Reg_unit,
        "max B", data);
    uint32_t test_info[3] = {num_tests, min_testR, Reg_unit};
    write_values("LRpB_test_info" INT_TO_LITERAL(infoidx), test_info, 3);
    if (num_tests == 1) {// dummy data, to circumvent compile error in mp_and_buffers
        data[1] = 0; num_tests++;
    }
    write_values("LRpB_test_data" INT_TO_LITERAL(infoidx), data, num_tests);
    free(data);
    return verifiedLRpB;
}\n"""

def no_abort_launchable_Reg_generate_kernels(compilableLRpT, Reg_unit, regmin, prefix,
        manufacturer, is_sreg):
    # generate kernels for testR = LRpT & m * Reg_unit && >regmin+16
    code = ""
    testR = 0 ; num_tests = 0 ; min_testR = 0 ; num_lrpt_tests = 0
    while (True):
        testR += Reg_unit
        if testR <= regmin + 16:
            continue
        if testR > compilableLRpT:
            break
        num_tests += 1
        if min_testR == 0:
            min_testR = testR
        if manufacturer == "nvidia":
            code += reg_kernel_nvidia_real_usage("{}_{}".format(prefix, testR),
                    testR, regmin)
        else:
            if is_sreg:
                code += reg_kernel_amd("{}_{}".format(prefix, testR),
                        testR, 0)
            else:
                code += reg_kernel_amd("{}_{}".format(prefix, testR),
                        0, testR)
    num_lrpt_tests = num_tests
    if compilableLRpT % Reg_unit != 0:
        if manufacturer == "nvidia":
            code += reg_kernel_nvidia_real_usage("{}_{}".format(prefix, compilableLRpT),
                    compilableLRpT, regmin)
        else:
            if is_sreg:
                code += reg_kernel_amd("{}_{}".format(prefix, compilableLRpT),
                        compilableLRpT, 0)
            else:
                code += reg_kernel_amd("{}_{}".format(prefix, compilableLRpT),
                        0, compilableLRpT)
        num_lrpt_tests += 1
    code += "void (*{}_LRpT[{}])(uint8_t*,bool) = {{\n".format(prefix, num_lrpt_tests)
    if compilableLRpT % Reg_unit != 0:
        code += "{}_{},\n".format(prefix, compilableLRpT)
    for i in range(num_tests):
        code += "{}_{},\n".format(prefix, min_testR + (num_tests-i-1) * Reg_unit)
    code += "};\n"
    code += "void (*{}_LRpB[{}])(uint8_t*,bool) = {{\n".format(prefix, num_tests)
    for i in range(num_tests):
        code += "{}_{},\n".format(prefix, min_testR + i * Reg_unit)
    code += "};\n"
    return code

def no_abort_launchable_test_code(out_dir, warp_size,\
        propLTpB, propLTpG, compilableLSpB, compilableLRpT, propLRpB, regmin):
        # regmin only used for nvidia
    code = include()
    code +=  """\
#include <stdio.h>
#define REPORT_DIR "{}"
#define warp_size {}
#include "tool.h"\n""".format(out_dir + "/", warp_size)
    code += get_max_true_code() + get_min_true_from0_code()
    # test thread limits
    code += normal_kernel("test_thr")
    code += no_abort_host_check_launchable("chk_thr", "test_thr", False)
    code += """\
bool chkfunc_LTpB(uint32_t val, uint32_t arg) {return chk_thr(1, val);}
bool chkfunc_LTpG(uint32_t val, uint32_t arg) {return chk_thr(val, 1);}
void do_thr_chk(uint32_t initLTpB, uint32_t initLTpG) {
    uint32_t verifiedLTpB = get_max_true(chkfunc_LTpB, initLTpB, 0);
    write_value("limit_threads_per_block", verifiedLTpB);
    uint32_t verifiedLTpG = initLTpG;
    //uint32_t verifiedLTpG = get_max_true(chkfunc_LTpG, initLTpG, 0);
    write_value("limit_threads_per_grid", verifiedLTpG);
}\n"""
    # test shmem limits
    LSpB_test_unit = ckpt.values[ckpt.CKPT.shared_memory_test_granularity]
    LSpB_test_num = compilableLSpB // LSpB_test_unit
    for i in range(LSpB_test_num):
        s = compilableLSpB - i * LSpB_test_unit
        code += shmem_kernel("test_shm_{}".format(s), s)
    code += make_kernel_array("test_shm", LSpB_test_num, compilableLSpB, LSpB_test_unit)
    code += no_abort_host_check_launchable("chk_shm", "test_shm", True)
    code += """\
bool chkfunc_LSpB(uint32_t idx) {return chk_shm(1, 1, idx);}
void do_shm_chk(uint32_t maxLSpB, uint32_t LSpB_unit) {
    uint32_t verifiedLSpBidx = get_min_true_from0(chkfunc_LSpB);
    uint32_t verifiedLSpB = maxLSpB - LSpB_unit * verifiedLSpBidx;
    write_value("limit_sharedmem_per_block", verifiedLSpB);
}\n"""
    # test register limits
    Reg_test_unit = ckpt.values[ckpt.CKPT.register_test_granularity]
    code += no_abort_launchable_LRpB_routine()
    if conf.gpu_manufacturer == "nvidia":
        code += no_abort_launchable_Reg_generate_kernels(compilableLRpT,
            Reg_test_unit, regmin, "test_reg", "nvidia", False)
        code += no_abort_host_check_launchable("chk_reg_LRpT", "test_reg_LRpT", True)
        code += no_abort_host_check_launchable("chk_reg_LRpB", "test_reg_LRpB", True)
        code += """\
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
    int zz = {0, 0};
    write_values("LRpB_test_info1", zz, 2); write_values("LRpB_test_data1", zz, 2);
    write_value("limit_registers_per_block", verifiedLRpB);
}\n"""
    else:
        code += no_abort_launchable_Reg_generate_kernels(compilableLRpT[1],
            Reg_test_unit, regmin, "test_vreg", "amd", False)
        code += no_abort_launchable_Reg_generate_kernels(compilableLRpT[0],
            Reg_test_unit, regmin, "test_sreg", "amd", True)
        code += no_abort_host_check_launchable("chk_vreg_LRpT", "test_vreg_LRpT", True)
        code += no_abort_host_check_launchable("chk_vreg_LRpB", "test_vreg_LRpB", True)
        code += no_abort_host_check_launchable("chk_sreg_LRpT", "test_sreg_LRpT", True)
        code += no_abort_host_check_launchable("chk_sreg_LRpB", "test_sreg_LRpB", True)
        code += """\
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
}\n"""
    # main()
    code += """\
int main(int argc, char **argv) {{
    hipSetDevice(0);
    write_init("verify_limits");
    do_thr_chk({}, {});
    do_shm_chk({}, {});\n""".format(propLTpB, propLTpG, compilableLSpB, LSpB_test_unit)
    if conf.gpu_manufacturer == "nvidia":
        code += "\tdo_reg_chk({}, {}, {}, {});\n".format(
            compilableLRpT, Reg_test_unit, propLRpB, regmin)
    else:
        code += "\tdo_reg_chk({}, {}, {}, {}, {}, {});\n".format(
            compilableLRpT[0], compilableLRpT[1], Reg_test_unit,
                propLRpB[0], propLRpB[1], regmin)
    code += """\
    return 0;
}\n"""
    return code

def checkfunc_LTpB_launchable_abort(val, args):
    return runnable(
        include() +\
        normal_kernel("test") +\
        abort_main_check_launchable("test", 1, val), args)

def checkfunc_LTpG_launchable_abort(val, args):
    return runnable(
        include() +\
        normal_kernel("test") +\
        abort_main_check_launchable("test", val, 1), args)

def checkfunc_LSpB_launchable_abort(val, args):
    return runnable(
        include() +\
        shmem_kernel("test", val) +\
        abort_main_check_launchable("test", 1, 1), args)

def checkfunc_LRpT_launchable_abort_nvidia(val, args):
    # should embed args[4] = regmin
    return runnable(
        include() +\
        reg_kernel_nvidia_real_usage("test", val, args[4]) +\
        abort_main_check_launchable("test", 1, 1), args)

def checkfunc_LRpB_launchable_abort_nvidia(val, args):
    # should embed args[4] = regmin, [5] = testR, 
    # [6] = verifiedLTpB, [7] = warp_size
    # val : num of warps to test
    if val * args[7] > args[6] : # cannot launch over LTpB!
        return False
    return runnable(
        include() +\
        reg_kernel_nvidia_real_usage("test", args[5], args[4]) +\
        abort_main_check_launchable("test", 1, val * args[7]),
        args)

def set_default_if_zero(val, default):
    return val if val>0 else default

def verify(proj_path, result_values):
    out_dir = os.path.join(proj_path, "build/", conf.gpu_manufacturer, "verify_limits")
    tool.print_and_exec("mkdir -p " + out_dir)
    tmp_src_file = os.path.join(out_dir, "tmp_src.cpp")
    tmp_obj_file = os.path.join(out_dir, "tmp_obj")
    getmax_args = [proj_path, out_dir, tmp_src_file, tmp_obj_file]

    # load the result from kernel_limits test
    propLTpB = int(result_values[Feature.limit_threads_per_block][0])
    propLTpG = int(result_values[Feature.limit_threads_per_grid][0])
    propLSpB = int(result_values[Feature.limit_sharedmem_per_block][0])
    if conf.gpu_manufacturer == "nvidia":
        propLRpT = int(result_values[Feature.limit_registers_per_thread][0])
        propLRpB = int(result_values[Feature.limit_registers_per_block][0])
    else:
        propLRpT = [int(result_values[Feature.limit_registers_per_thread][0]),\
                    int(result_values[Feature.limit_registers_per_thread][1])]
        propLRpB = [int(result_values[Feature.limit_registers_per_block][0]),\
                    int(result_values[Feature.limit_registers_per_block][1])]
    # set default estimate value if prop is 0
    propLTpB = set_default_if_zero(propLTpB, 1024)
    propLTpG = set_default_if_zero(propLTpG, 1073741824)
    propLSpB = set_default_if_zero(propLSpB, 65536)
    if conf.gpu_manufacturer == "nvidia":
        propLRpT = set_default_if_zero(propLRpT, 255)
        propLRpB = set_default_if_zero(propLRpB, 65536)
    else:
        propLRpT[0] = set_default_if_zero(propLRpT[0], 100)
        propLRpT[1] = set_default_if_zero(propLRpT[1], 256)
        propLRpB[0] = set_default_if_zero(propLRpB[0], 65536)
        propLRpB[1] = set_default_if_zero(propLRpB[1], 65536)

    ## Compilability check
    compilableLSpB = get_max_true(checkfunc_LSpB_compilable, getmax_args, propLSpB)
    if conf.gpu_manufacturer == "nvidia":
        compilableLRpT =\
            get_max_true(checkfunc_LRpT_compilable_nvidia, getmax_args, propLRpT)
    else:
        compilableLRpT = [\
            get_max_true(checkfunc_LRpT_compilable_amd_sgpr, getmax_args, propLRpT[0]),\
            get_max_true(checkfunc_LRpT_compilable_amd_vgpr, getmax_args, propLRpT[1])]

    ## Launchability check
    ### currently, only gpgpusim can abort!
    kernel_can_abort = conf.gpu_manufacturer=="nvidia" and conf.simulator_driven
    regmin = 0
    if conf.gpu_manufacturer == "nvidia":
        regmin = find_regmin_nvidia(compilableLRpT, getmax_args)
    warp_size = int(result_values[Feature.warp_size][0])
    Reg_test_unit = ckpt.values[ckpt.CKPT.register_test_granularity]
    if not kernel_can_abort:
        if not runnable(
            no_abort_launchable_test_code(out_dir, warp_size, propLTpB, propLTpG,
                compilableLSpB, compilableLRpT, propLRpB, regmin), getmax_args):
            exit(1)
    else:
        ## Here, nVidia is assumed for now!
        verifiedLTpB = get_max_true(checkfunc_LTpB_launchable_abort,\
            getmax_args, propLTpB)
        verifiedLTpG = propLTpG
        #verifiedLTpG = get_max_true(checkfunc_LTpG_launchable_abort,\
        #    getmax_args, propLTpG)
        verifiedLSpB = get_max_true_given_maxval(
            checkfunc_LSpB_launchable_abort, getmax_args,
            compilableLSpB, ckpt.values[ckpt.CKPT.shared_memory_test_granularity])
        verifiedLRpT = get_max_true_given_maxval(
            checkfunc_LRpT_launchable_abort_nvidia, getmax_args + [regmin],
            compilableLRpT, Reg_test_unit)

        maxb_for_LRpB = []
        verifiedLRpB = 0
        # measure for testR = m * Reg_test_unit && testR > regmin+16
        testR = 0 ; num_LRpB_tests = 0 ; min_testR = 0
        while (True):
            testR += Reg_test_unit
            if testR <= regmin + 16:
                continue
            if testR > verifiedLRpT:
                break
            num_LRpB_tests += 1
            if min_testR == 0:
                min_testR = testR
            # measure max_b for testR
            max_b = get_max_true(
                checkfunc_LRpB_launchable_abort_nvidia,
                getmax_args + [regmin, testR, verifiedLTpB, warp_size],
                propLRpB // testR // warp_size - 1) # -1: possible speedup
            maxb_for_LRpB += [max_b]
            verifiedLRpB = max(verifiedLRpB, max_b * testR)

        LRpB_graph = "@Reg per Block:{}:reg/thread:{}:{}:max B:".format(
            num_LRpB_tests, min_testR, Reg_test_unit)
        LRpB_data_str = ""
        for i in range(num_LRpB_tests-1):
            LRpB_data_str += "{},".format(maxb_for_LRpB[i])
        LRpB_data_str += "{}".format(maxb_for_LRpB[num_LRpB_tests-1])
        LRpB_graph += LRpB_data_str + "\n"
        
        reportfile = open(os.path.join(out_dir, "report.txt"), 'w')
        reportfile.write("limit_threads_per_block={}\n".format(verifiedLTpB))
        reportfile.write("limit_threads_per_grid={}\n".format(verifiedLTpG))
        reportfile.write("limit_sharedmem_per_block={}\n".format(verifiedLSpB))
        reportfile.write("limit_registers_per_thread={}\n".format(verifiedLRpT))
        
        reportfile.write(LRpB_graph)
        reportfile.write("LRpB_test_info0=[{}, {}, {}]\n".format(
            num_LRpB_tests, min_testR, Reg_test_unit))
        if num_LRpB_tests == 1: # dummy, to circumvent compile err in mp_and_buffers
            reportfile.write("LRpB_test_data0=[{}, 0]\n".format(LRpB_data_str))
        else:
            reportfile.write("LRpB_test_data0=[{}]\n".format(LRpB_data_str))
        reportfile.write("LRpB_test_info1=[0, 0]\n")
        reportfile.write("LRpB_test_data1=[0, 0]\n")
        reportfile.write("limit_registers_per_block={}\n".format(verifiedLRpB))
        reportfile.close()

    return True


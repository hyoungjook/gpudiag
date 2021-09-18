import subprocess
import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from define import Feature
import config_env as conf
import config_ckpt as ckpt

def print_and_exec(cmd):
    print("  (execute) : " + cmd)
    return subprocess.call(cmd, shell=True, executable="/bin/bash")

def write_code(code, file):
    codefile = open(file, 'w')
    codefile.write(code)
    codefile.close()

def compile_succeed(proj_path, in_file, out_file):
    cmd = conf.compile_command
    cmd = cmd.replace("$SRC", in_file)
    cmd = cmd.replace("$BIN", out_file)
    cmd += " -I" + os.path.join(proj_path, "tests/")
    return print_and_exec(cmd) == 0

def compile_and_check_resource(proj_path, in_file, out_file):
    # only called for nvidia
    cmd = conf.compile_command
    cmd = cmd.replace("$SRC", in_file)
    cmd = cmd.replace("$BIN", out_file)
    cmd += " -I" + os.path.join(proj_path, "tests/")
    cmd += " --resource-usage 2> " + out_file + "_resource"
    if print_and_exec(cmd) != 0:
        print_and_exec("cat " + out_file + "_resource")
        exit(1)
    resfile = open(out_file + "_resource", 'r')
    spill_cnt = -1
    reg_cnt = -1
    for line in resfile.readlines():
        tokens = line.split(' ')
        for i in range(len(tokens)):
            if tokens[i].find("spill") >= 0:
                spill_cnt = int(tokens[i-2])
            if tokens[i].find("registers") >= 0:
                reg_cnt = int(tokens[i-1])
    resfile.close()
    if spill_cnt<0 or reg_cnt<0:
        print("Unable to find reg cnt!")
        exit(1)
    return [spill_cnt, reg_cnt]

def run_succeed(obj_dir, obj_name):
    obj_file = os.path.join(obj_dir, obj_name)
    if conf.simulator_driven:
        copied_bin_dir = os.path.join(conf.simulator_path, "gpudiag/", "verify_limits")
        cmd = "mkdir -p " + copied_bin_dir
        cmd += " && cp " + obj_file + " " + copied_bin_dir
        print_and_exec(cmd)
        cmd = conf.run_command
        cmd = cmd.replace("$DIR", copied_bin_dir)
        cmd = cmd.replace("$BIN", obj_name)
        cmd = "cd " + conf.simulator_path + " && " + cmd
        run_status = print_and_exec(cmd)
        cmd = "cp " + os.path.join(copied_bin_dir, "*") + " " + obj_dir
        cmd += " ; rm -rf " + os.path.join(conf.simulator_path, "gpudiag")
        print_and_exec(cmd)
    else:
        cmd = conf.run_command
        cmd = cmd.replace("$DIR", obj_dir)
        cmd = cmd.replace("$BIN", obj_name)
        run_status = print_and_exec(cmd)
    return run_status == 0

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
        write_code(include() + reg_kernel_nvidia("test", testR) +\
                main_check_compilable("test"), args[2])
        regcnts = compile_and_check_resource(args[0], args[2], args[3])
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
    write_code(code, args[2])
    return compile_succeed(args[0], args[2], args[3])

def checkfunc_LSpB_compilable(val, args):
    return compilable(
        include() +\
        shmem_kernel("test", val) +\
        main_check_compilable("test"), args)

def checkfunc_LRpT_compilable_nvidia(val, args):
    write_code(
        include() +\
        reg_kernel_nvidia("test", val) +\
        main_check_compilable("test"), args[2])
    regcnts = compile_and_check_resource(args[0], args[2], args[3])
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
    write_code(code, args[2])
    if not compile_succeed(args[0], args[2], args[3]):
        exit(1)
    return run_succeed(args[1], "tmp_obj")

def no_abort_launchable_LRpB_routine():
    return """\
uint32_t do_LRpB_test(uint32_t LRpT, uint32_t Reg_unit, uint32_t initLRpB,
        bool (*chkfunc)(uint32_t,uint32_t), const char* title, const char* xlabel) {
    uint32_t num_tests = LRpT / Reg_unit;
    if (LRpT % Reg_unit > 0) num_tests++;
    uint32_t verifiedLRpB = 0;
    uint32_t *data = (uint32_t*)malloc(num_tests * sizeof(uint32_t));
    for (int i=0; i<num_tests; i++) {
        uint32_t testR = LRpT - i * Reg_unit;
        uint32_t init_b = initLRpB / testR / warp_size;
        uint32_t max_b = get_max_true(chkfunc, init_b, i);
        data[i] = max_b * warp_size;
        uint32_t LRpB_here = max_b * warp_size * testR;
        verifiedLRpB = verifiedLRpB>LRpB_here?verifiedLRpB:LRpB_here;
    }
    write_graph_data(title, num_tests, xlabel, (int32_t)LRpT, -(int32_t)Reg_unit,
        "max B", data);
    free(data);
    return verifiedLRpB;
}\n"""

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
    Reg_test_unit = ckpt.values[ckpt.CKPT.register_test_granulatiry]
    code += no_abort_launchable_LRpB_routine()
    if conf.gpu_manufacturer == "nvidia":
        LRpT_test_num = compilableLRpT // Reg_test_unit
        if compilableLRpT % Reg_test_unit > 0:
            LRpT_test_num += 1
        for i in range(LRpT_test_num):
            testR = compilableLRpT - i * Reg_test_unit
            code += reg_kernel_nvidia_real_usage("test_reg_{}".format(testR),
                testR, regmin)
        code += make_kernel_array("test_reg", 
            LRpT_test_num, compilableLRpT, Reg_test_unit)
        code += no_abort_host_check_launchable("chk_reg", "test_reg", True)
        code += """\
bool chkfunc_LRpT(uint32_t idx) {return chk_reg(1, 1, idx);}
bool chkfunc_LRpB(uint32_t val, uint32_t arg) {return chk_reg(1,val*warp_size,arg);}
void do_reg_chk(uint32_t maxLRpT, uint32_t Reg_unit, uint32_t initLRpB) {
    uint32_t verifiedLRpTidx = get_min_true_from0(chkfunc_LRpT);
    uint32_t verifiedLRpT = maxLRpT - Reg_unit * verifiedLRpTidx;
    write_value("limit_registers_per_thread", verifiedLRpT);
    uint32_t verifiedLRpB = do_LRpB_test(verifiedLRpT, Reg_unit, initLRpB,
        chkfunc_LRpB, "Regs per Block", "regs/thread");
    write_value("limit_registers_per_block", verifiedLRpB);
}\n"""
    else:
        LsRpT_test_num = compilableLRpT[0] // Reg_test_unit
        LvRpT_test_num = compilableLRpT[1] // Reg_test_unit
        for i in range(LsRpT_test_num):
            testSR = compilableLRpT[0] - i * Reg_test_unit
            code += reg_kernel_amd("test_sreg_{}".format(testSR), testSR, 0)
        for i in range(LvRpT_test_num):
            testVR = compilableLRpT[1] - i * Reg_test_unit
            code += reg_kernel_amd("test_vreg_{}".format(testVR), 0, testVR)
        code += make_kernel_array("test_sreg", 
            LsRpT_test_num, compilableLRpT[0], Reg_test_unit)
        code += make_kernel_array("test_vreg", 
            LvRpT_test_num, compilableLRpT[1], Reg_test_unit)
        code += no_abort_host_check_launchable("chk_sreg", "test_sreg", True)
        code += no_abort_host_check_launchable("chk_vreg", "test_vreg", True)
        code += """\
bool chkfunc_LsRpT(uint32_t idx) {return chk_sreg(1, 1, idx);}
bool chkfunc_LvRpT(uint32_t idx) {return chk_vreg(1, 1, idx);}
bool chkfunc_LsRpB(uint32_t val, uint32_t arg) {return chk_sreg(1,val*warp_size,arg);}
bool chkfunc_LvRpB(uint32_t val, uint32_t arg) {return chk_vreg(1,val*warp_size,arg);}
void do_reg_chk(uint32_t maxLsRpT, uint32_t maxLvRpT, 
        uint32_t Reg_unit, uint32_t initLsRpB, uint32_t initLvRpB) {
    uint32_t verifiedLsRpTidx = get_min_true_from0(chkfunc_LsRpT);
    uint32_t verifiedLvRpTidx = get_min_true_from0(chkfunc_LvRpT);
    uint32_t verifiedLRpT[2] = {
        maxLsRpT - Reg_unit * verifiedLsRpTidx,
        maxLvRpT - Reg_unit * verifiedLvRpTidx
    };
    write_values("limit_registers_per_thread", verifiedLRpT, 2);
    uint32_t verifiedLRpB[2] = {
        do_LRpB_test(verifiedLRpT[0], Reg_unit, initLsRpB,
            chkfunc_LsRpB, "SRegs per Block", "sregs/thread") / warp_size,
        do_LRpB_test(verifiedLRpT[1], Reg_unit, initLvRpB,
            chkfunc_LvRpB, "VRegs per Block", "vregs/thread")
    };
    write_line("# Sregs are counted as one per warp.");
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
        code += "\tdo_reg_chk({}, {}, {});\n".format(
            compilableLRpT, Reg_test_unit, propLRpB)
    else:
        code += "\tdo_reg_chk({}, {}, {}, {}, {});\n".format(
            compilableLRpT[0], compilableLRpT[1], Reg_test_unit, propLRpB[0], propLRpB[1])
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
    print_and_exec("mkdir -p " + out_dir)
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
    Reg_test_unit = ckpt.values[ckpt.CKPT.register_test_granulatiry]
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

        maxB_for_LRpB = []
        num_LRpB_tests = verifiedLRpT // Reg_test_unit
        if verifiedLRpT % Reg_test_unit > 0:
            num_LRpB_tests += 1
        verifiedLRpB = 0
        for i in range(num_LRpB_tests):
            testR = verifiedLRpT - i * Reg_test_unit
            max_b = get_max_true(
                checkfunc_LRpB_launchable_abort_nvidia,
                getmax_args + [regmin, testR, verifiedLTpB, warp_size],
                propLRpB // testR // warp_size - 1) # -1: possible speedup
            maxB_for_LRpB += [max_b * warp_size]
            verifiedLRpB = max(verifiedLRpB, max_b * warp_size * testR)
        LRpB_graph = "@Reg per Block:{}:reg/thread:{}:{}:max B:".format(
            num_LRpB_tests, verifiedLRpT, -Reg_test_unit)
        for i in range(num_LRpB_tests-1):
            LRpB_graph += "{},".format(maxB_for_LRpB[i])
        LRpB_graph += "{}\n".format(maxB_for_LRpB[num_LRpB_tests-1])
        
        reportfile = open(os.path.join(out_dir, "report.txt"), 'w')
        reportfile.write("limit_threads_per_block={}\n".format(verifiedLTpB))
        reportfile.write("limit_threads_per_grid={}\n".format(verifiedLTpG))
        reportfile.write("limit_sharedmem_per_block={}\n".format(verifiedLSpB))
        reportfile.write("limit_registers_per_thread={}\n".format(verifiedLRpT))
        reportfile.write(LRpB_graph)
        reportfile.write("limit_registers_per_block={}\n".format(verifiedLRpB))
        reportfile.close()

    return True


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

def checkfunc_LSpB_compilable(val, args):
    code = """\
#include "hip/hip_runtime.h"
#include <stdint.h>
__global__ void testkernel(uint8_t *r){{
    __shared__ uint8_t arr[{}];
    for (int i=0; i<{}; i++) arr[i] = (uint8_t)clock();
    for (int i=0; i<{}; i++) r[i] = arr[i];
}}
int main() {{
    hipLaunchKernelGGL(testkernel, dim3(1), dim3(1), 0, 0, nullptr);
    return 0;
}}""".format(val, val, val)
    write_code(code, args[2])
    return compile_succeed(args[0], args[2], args[3])

def checkfunc_LRpT_compilable_nvidia(val, args):
    # make the code to use at least val regs
    code = """\
#include "hip/hip_runtime.h"
#include <stdint.h>
__global__ void testkernel(uint8_t *r){{
    asm volatile(".reg .b32 tr<{}>;\\n");
    asm volatile(".reg .b64 addr;\\n");
    asm volatile("mov.u64 addr, %0;\\n"::"l"(r));
""".format(val)
    for i in range(val):
        code += "\tasm volatile(\"mov.u32 tr{}, %clock;\\n\");\n".format(i)
    code += "\tasm volatile(\"bar.sync 0;\");\n"
    for i in range(val):
        code += "\tasm volatile(\"st.global.u32 [addr+{}], tr{};\");\n".format(4*i, i)
    code += """\
}
int main() {
    hipLaunchKernelGGL(testkernel, dim3(1), dim3(1), 0, 0, nullptr);
}"""
    write_code(code, args[2])
    regcnts = compile_and_check_resource(args[0], args[2], args[3])
    if regcnts[0] == 0: # no spill regs
        return True
    else: # reg spill occurred
        return val <= regcnts[1]

def checkfunc_LRpT_compilable_amd_sgpr(val, args):
    code = """\
#include "hip/hip_runtime.h"
#include <stdint.h>
__global__ void testkernel(){{
    asm volatile("s_mov_b32 s{}, 0\\n");
}}
int main() {{
    hipLaunchKernelGGL(testkernel, dim3(1), dim3(1), 0, 0);
    return 0;
}}""".format(val-1)
    write_code(code, args[2])
    return compile_succeed(args[0], args[2], args[3])

def checkfunc_LRpT_compilable_amd_vgpr(val, args):
    code = """\
#include "hip/hip_runtime.h"
#include <stdint.h>
__global__ void testkernel(){{
    asm volatile("v_mov_b32 v{}, 0\\n");
}}
int main() {{
    hipLaunchKernelGGL(testkernel, dim3(1), dim3(1), 0, 0);
    return 0;
}}""".format(val-1)
    write_code(code, args[2])
    return compile_succeed(args[0], args[2], args[3])

def no_abort_launchable_test_code(out_dir,\
        propLTpB, propLTpG, compilableLSpB, compilableLRpT, propLRpB):
    # check thr limits
    code =  """\
#include <stdio.h>
#include <stdint.h>
#include "hip/hip_runtime.h"
#define REPORT_DIR "{}"
#include "tool.h"
__global__ void test_thr(int *out, uint32_t G, uint32_t B) {{
    if (hipThreadIdx_x == B-1 && hipBlockIdx_x == G-1) {{
        *out = 1;
    }}
}}
bool check_launchable_thr(uint32_t G, uint32_t B) {{
    int hout = 0, *dout;
    hipMalloc(&dout, sizeof(int));
    hipMemcpy(dout, &hout, sizeof(int), hipMemcpyHostToDevice);
    hipLaunchKernelP(test_thr, dim3(G), dim3(B), 0, 0, dout, G, B);
    hipStreamSynchronize(0);
    hipMemcpy(&hout, dout, sizeof(int), hipMemcpyDeviceToHost);
    hipFree(dout);
    return hout == 1;
}}\n""".format(out_dir + "/")
    # check shm limits
    LSpB_test_unit = ckpt.values[ckpt.CKPT.shared_memory_test_granularity]
    LSpB_test_num = compilableLSpB // LSpB_test_unit
    for i in range(LSpB_test_num):
        test_LSpB = compilableLSpB - i*LSpB_test_unit
        code += """\
__global__ void test_shm_{}(uint8_t *r, bool real) {{
    __shared__ uint8_t arr[{}];
    if (real) {{
        *r = 1; return;
    }}
    for (int i=0; i<{}; i++) arr[i] = (uint8_t)clock();
    for (int i=0; i<{}; i++) r[i] = arr[i];
}}\n""".format(test_LSpB, test_LSpB, test_LSpB, test_LSpB)
    code += """\
void (*test_shm[{}])(uint8_t *, bool) = {{
""".format(LSpB_test_num)
    for i in range(LSpB_test_num):
        code += "test_shm_{},\n".format(compilableLSpB - i*LSpB_test_unit)
    code += """\
}};
bool check_launchable_shm(uint32_t s) {{
    uint8_t hout = 0, *dout;
    hipMalloc(&dout, sizeof(uint8_t));
    hipMemcpy(dout, &hout, sizeof(uint8_t), hipMemcpyHostToDevice);
    int index_of_function = ({} - s + {}) / {};
    hipLaunchKernelP(test_shm[index_of_function], dim3(1), dim3(1), 0, 0, dout, true);
    hipStreamSynchronize(0);
    hipMemcpy(&hout, dout, sizeof(uint8_t), hipMemcpyDeviceToHost);
    hipFree(dout);
    return hout == 1;
}}\n""".format(compilableLSpB, LSpB_test_unit-1, LSpB_test_unit)
    # check reg limits
    # main()
    code += """\
int main(int argc, char **argv) {{
    hipSetDevice(0);
    uint32_t testLTpB = {}, testLTpG = {}, testLSpB = {};
    printf("Checking LTpB..\\n");
    while(check_launchable_thr(1, testLTpB++));
    while(!check_launchable_thr(1, --testLTpB));
    printf("Checking LTpG..\\n");
    //while(check_launchable_thr(testLTpG++, 1));
    //while(!check_launchable_thr(--testLTpG, 1));
    printf("Checking LSpB..\\n");
    while(!check_launchable_shm(testLSpB)) {{testLSpB-={};}}

    write_init("verify_limits");
    write_value("limit_threads_per_block", testLTpB);
    write_value("limit_threads_per_grid", testLTpG);
    write_value("limit_sharedmem_per_block", testLSpB);

    return 0;
}}""".format(propLTpB, propLTpG, compilableLSpB, LSpB_test_unit)
    return code

def gpgpusim_thread_test_code(testLTpG, testLTpB):
    return """\
#include <stdio.h>
#include "hip/hip_runtime.h"
__global__ void testkernel(){{}}
int main(int argc, char **argv) {{
    hipSetDevice(0);
    hipLaunchKernelGGL(testkernel, dim3({}), dim3({}), 0, 0);
    hipStreamSynchronize(0);
    return 0;
}}
""".format(testLTpG, testLTpB)

def checkfunc_gpgpusim_LTpB(val, args):
    write_code(gpgpusim_thread_test_code(1, val), args[2])
    if not compile_succeed(args[0], args[2], args[3]):
        exit(1)
    return run_succeed(args[1], "tmp_obj")

def checkfunc_gpgpusim_LTpG(val, args):
    write_code(gpgpusim_thread_test_code(val, 1), args[2])
    if not compile_succeed(args[0], args[2], args[3]):
        exit(1)
    return run_succeed(args[1], "tmp_obj")

def checkfunc_gpgpusim_LSpB(val, args):
    code = """\
#include <stdio.h>
#include "hip/hip_runtime.h"
__global__ void testkernel(uint8_t *r, bool real){{
    __shared__ uint8_t arr[{}];
    if (real) return;
    for (int i=0; i<{}; i++) arr[i] = (uint8_t)clock();
    for (int i=0; i<{}; i++) r[i] = arr[i];
}}
int main(int argc, char **argv) {{
    hipSetDevice(0);
    hipLaunchKernelGGL(testkernel, dim3(1), dim3(1), 0, 0, nullptr, true);
    hipStreamSynchronize(0);
    return 0;
}}\n""".format(val, val, val)
    write_code(code, args[2])
    if not compile_succeed(args[0], args[2], args[3]):
        exit(1)
    return run_succeed(args[1], "tmp_obj")

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
        propLRpT = [int(result_values[Feature.limit_registers_per_thread][0])]
        propLRpB = [int(result_values[Feature.limit_registers_per_block][0])]
    else:
        propLRpT = [int(result_values[Feature.limit_registers_per_thread][0]),\
                    int(result_values[Feature.limit_registers_per_thread][1])]
        propLRpB = [int(result_values[Feature.limit_registers_per_block][0]),\
                    int(result_values[Feature.limit_registers_per_block][1])]

    ## Compilability check
    compilableLSpB = get_max_true(checkfunc_LSpB_compilable, getmax_args, propLSpB)
    if conf.gpu_manufacturer == "nvidia":
        compilableLRpT = [\
            get_max_true(checkfunc_LRpT_compilable_nvidia, getmax_args, propLRpT[0])]
    else:
        compilableLRpT = [\
            get_max_true(checkfunc_LRpT_compilable_amd_sgpr, getmax_args, propLRpT[0]),\
            get_max_true(checkfunc_LRpT_compilable_amd_vgpr, getmax_args, propLRpT[1])]

    ## Launchability check
    kernel_can_abort = conf.gpu_manufacturer=="nvidia" and conf.simulator_driven
    if not kernel_can_abort:
        write_code(no_abort_launchable_test_code(out_dir, propLTpB, propLTpG,\
            compilableLSpB, compilableLRpT, propLRpB), tmp_src_file)
        if not compile_succeed(proj_path, tmp_src_file, tmp_obj_file):
            exit(1)
        if not run_succeed(out_dir, "tmp_obj"):
            exit(1)
    else:
        verifiedLTpB = get_max_true(checkfunc_gpgpusim_LTpB,\
            getmax_args, propLTpB)
        verifiedLTpG = propLTpG
        #verifiedLTpG = get_max_true(checkfunc_gpgpusim_LTpG,\
        #    getmax_args, propLTpG)
        verifiedLSpB = compilableLSpB
        while True:
            if checkfunc_gpgpusim_LSpB(verifiedLSpB, getmax_args):
                break
            verifiedLSpB -= ckpt.values[ckpt.CKPT.shared_memory_test_granularity]
        reportfile = open(os.path.join(out_dir, "report.txt"), 'w')
        reportfile.write("limit_threads_per_block={}\n".format(verifiedLTpB))
        reportfile.write("limit_threads_per_grid={}\n".format(verifiedLTpG))
        reportfile.write("limit_sharedmem_per_block={}\n".format(verifiedLSpB))
        reportfile.close()

    # LRpT: only check compilability; add to the report manually
    reportfile = open(os.path.join(out_dir, "report.txt"), 'a')
    if conf.gpu_manufacturer == "nvidia":
        reportfile.write("limit_registers_per_thread={}\n".format(compilableLRpT[0]))
    else:
        reportfile.write("limit_registers_per_thread=[{}, {}]\n".format(\
            compilableLRpT[0], compilableLRpT[1]))
    reportfile.close()

    return True


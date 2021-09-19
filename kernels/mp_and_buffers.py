import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from define import Feature
import config_ckpt as ckpt
import config_env as conf
from . import reused_codes

def measure_num_mp_code(repeat, repeat_for, inicode, repcode, fincode):
    code = """\
__global__ void measure_num_mp(uint32_t *sync, uint32_t G, uint64_t *result,
        uint64_t timeout, uint32_t *is_timeout, uint64_t *totalclk){
    // assumes *sync=0, *is_timeout=0 initially
    // if timeout=0, no timeout.
    uint64_t sclk, eclk, toclk;
    bool thread0 = (hipThreadIdx_x == 0);
    uint32_t G2 = 2 * G, blkid = hipBlockIdx_x;
    // if timeout already, exit immediately
    if (*is_timeout == 1) return;

    // sync1: determine timeout (with timeout)
    bool chktime = (timeout != 0);
    volatile uint32_t *chksync = sync;
    __syncthreads();
    sclk = clock();
    toclk = sclk + timeout;
    if (thread0) atomicAdd(sync, 1);
    while(*chksync < G) {
        if (chktime && clock() > toclk) {
            *is_timeout = 1; return;
        }
    }
    eclk = clock();
    if (thread0) totalclk[blkid] = eclk - sclk;
"""
    code += inicode
    code += """\
#pragma unroll 1
    for (int i=0; i<{}; i++) {{ // warmup icache + repeat_for
        if (i==1) {{
            __syncthreads();
            // sync2: real sync before measurement, no timeout
            if (thread0) atomicAdd(sync, 1);
            while(*chksync < G2);
            // begin measurement
            __syncthreads();
            sclk = clock();
        }}
""".format(repeat_for+1)
    for i in range(repeat):
        code += repcode(i)
    code += fincode
    code += """\
    }
    __syncthreads();
    eclk = clock();
    // save the result
    if (thread0) result[blkid] = eclk - sclk;
}
"""
    return code

def measure_warpstatebuffer_code():
    code = """\
__global__ void measure_warpstatebuffer(uint32_t *sync, uint32_t G,
        uint64_t timeout, uint32_t *is_timeout, uint64_t *totalclk) {
    uint64_t sclk, eclk, toclk;
    bool thread0 = (hipThreadIdx_x == 0);
    bool chktime = (timeout != 0);
    volatile uint32_t *chksync = sync;
    __syncthreads();
    sclk = clock();
    toclk = sclk + timeout;
    if (thread0) atomicAdd(sync, 1);
    while(*chksync < G) {
        if (chktime && clock() > toclk) {
            *is_timeout = 1; return;
        } 
    }
    eclk = clock();
    if (thread0) totalclk[hipBlockIdx_x] = eclk - sclk;
}
"""
    return code

def measure_shmem_code(limit_shmem, shmem_unit):
    code = ""
    for s in range(shmem_unit, limit_shmem+1, shmem_unit):
        code += """\
__global__ void measure_shmem_{}(uint32_t *sync, uint32_t G,
        uint64_t timeout, uint32_t *is_timeout, uint64_t *totalclk) {{
    uint64_t sclk, eclk, toclk;
    bool thread0 = (hipThreadIdx_x == 0);
    bool chktime = (timeout != 0);
    volatile uint32_t *chksync = sync;
    __syncthreads();
    sclk = clock();
    toclk = sclk + timeout;
    if (thread0) atomicAdd(sync, 1);
    while(*chksync < G) {{
        if (chktime && clock() > toclk) {{
            *is_timeout = 1; return;
        }}
    }}
    eclk = clock();
    if (thread0) totalclk[hipBlockIdx_x] = eclk - sclk;
    // ensure shmem usage
    if (G > 0) return; // do not execute, just compile
    __shared__ uint8_t arr[{}];
    uint8_t *fakeptr = (uint8_t*)sync;
    for (int i=0; i<{}; i++) arr[i] = (uint8_t)clock();
    for (int i=0; i<{}; i++) fakeptr[i] = arr[i];
}}
""".format(s, s, s, s)
    code += """\
void (*measure_shmem[{}])(uint32_t*, uint32_t, uint64_t, uint32_t*, uint64_t*) = {{
""".format(limit_shmem // shmem_unit)
    for s in range(shmem_unit, limit_shmem+1, shmem_unit):
        code += "measure_shmem_{},\n".format(s)
    code += "};\n"
    return code

## global variable; calculated in verify_constraint(), used in generate_*()
register_file_kernel_code_regmin = 0

def calc_regcnt_from_regval_nvidia(regval, tmpsrc, tmpobj, proj_path):
    reused_codes.write_code(
        "#include <stdint.h>\n#include \"hip/hip_runtime.h\"\n" +\
        reused_codes.measure_regfile_code(
        "test", regval, conf.gpu_manufacturer, False) +\
        "int main(){hipLaunchKernelGGL(test,dim3(1),dim3(1),0,0," +\
        "nullptr,0,0,nullptr,nullptr);return 0;}\n"
        , tmpsrc)
    return reused_codes.compile_and_check_resource(proj_path, tmpsrc, tmpobj)

def verify_constraint(result_values, proj_path):
    out_dir = os.path.join(proj_path, "build/", conf.gpu_manufacturer, "mp_and_buffers")
    reused_codes.print_and_exec("mkdir -p " + out_dir)
    tmp_src_file = os.path.join(out_dir, "tmp_src.cpp")
    tmp_obj_file = os.path.join(out_dir, "tmp_obj")

    # calculate register_file_kernel_code_regmin
    global register_file_kernel_code_regmin
    if (conf.gpu_manufacturer == "nvidia"):
        LRpT = int(result_values[Feature.limit_registers_per_thread][0])
        testR = LRpT // 2
        regmin = calc_regcnt_from_regval_nvidia(
            testR, tmp_src_file, tmp_obj_file, proj_path)[1] - testR
        register_file_kernel_code_regmin = regmin
    else:
        # manually counted for the measure_regfile_code! It is larger than maximum
        register_file_kernel_code_regmin = 32
    return True

def generate_reg_kernel_set(LRpT, Reg_unit, manufacturer, is_sreg):
    code = ""
    global register_file_kernel_code_regmin
    testR = 0 ; num_tests = 0 ; min_R = 0
    while (True):
        testR += Reg_unit
        if testR <= register_file_kernel_code_regmin + 16:
            continue
        if testR > LRpT:
            break
        num_tests += 1
        if min_R == 0:
            min_R = testR
        # generate code for testR
        if manufacturer == "nvidia":
            code += reused_codes.measure_regfile_code("measure_reg_{}".format(testR),
                testR - register_file_kernel_code_regmin, "nvidia", False)
        else:
            code += reused_codes.measure_regfile_code(
                "measure_{}reg_{}".format('s' if is_sreg else 'v', testR),
                testR, "amd", is_sreg)
    arrname = "reg"
    if manufacturer == "amd":
        arrname = "{}{}".format("s" if is_sreg else "v", arrname)
    arrname = "measure_{}".format(arrname)
    code += "void (*{}[{}])(uint32_t*,uint32_t,uint64_t,uint32_t*,uint64_t*)={{\n"\
        .format(arrname, num_tests)
    for i in range(num_tests):
        code += "{}_{},\n".format(arrname, min_R + i * Reg_unit)
    code += "};\n"
    code += "#define REGKERNEL_{}REG_MIN_R {}\n".format(
        '' if manufacturer=="nvidia" else ('S' if is_sreg else 'V'), min_R)
    return code

def generate_nvidia(result_values):
    # measure_br_time
    code = ""
    code += reused_codes.measure_width_code(100, 5, "br", "",\
        lambda i: "asm volatile(\"MT_BR_{}: bra MT_BR_{};\\n\");\n".format(i, i+1),\
        "asm volatile(\"MT_BR_{}:\\n\");\n".format(100))

    # measure_num_mp
    code += measure_num_mp_code(100, 2, "",\
        lambda i: "asm volatile(\"MNM_BR_{}: bra MNM_BR_{};\\n\");\n".format(i, i+1),\
        "asm volatile(\"MNM_BR_{}:\\n\");\n".format(100))

    # measure warpstatebuffer
    code += measure_warpstatebuffer_code()

    # measure shmem
    limit_shmem = int(result_values[Feature.limit_sharedmem_per_block][0])
    shmem_unit = ckpt.values[ckpt.CKPT.shared_memory_test_granularity]
    code += measure_shmem_code(limit_shmem, shmem_unit)

    # measure registerfile
    global register_file_kernel_code_regmin
    LRpT = int(result_values[Feature.limit_registers_per_thread][0])
    Reg_unit = int(result_values[Feature.LRpB_test_info0][2])
    code += generate_reg_kernel_set(LRpT, Reg_unit, "nvidia", False)    

    return code

def generate_amd(result_values):
    # measure_br_time
    code = ""
    code += reused_codes.measure_width_code(100, 5, "br", "",\
        lambda i: "asm volatile(\"MT_BR_{}: s_branch MT_BR_{}\\n\");\n".format(i, i+1),\
        "asm volatile(\"MT_BR_{}:\\n\");\n".format(100))

    # measure_num_mp
    code += measure_num_mp_code(100, 2, "",\
        lambda i: "asm volatile(\"MNM_BR_{}: s_branch MNM_BR_{}\\n\");\n".format(i, i+1),\
        "asm volatile(\"MNM_BR_{}:\\n\");\n".format(100))

    # measure warpstatebuffer
    code += measure_warpstatebuffer_code()

    # measure shmem
    limit_shmem = int(result_values[Feature.limit_sharedmem_per_block][0])
    shmem_unit = ckpt.values[ckpt.CKPT.shared_memory_test_granularity]
    code += measure_shmem_code(limit_shmem, shmem_unit)

    # measure regfile
    global register_file_kernel_code_regmin
    s_LRpT = int(result_values[Feature.limit_registers_per_thread][0])
    v_LRpT = int(result_values[Feature.limit_registers_per_thread][1])
    Reg_unit = int(result_values[Feature.LRpB_test_info0][2])

    code += generate_reg_kernel_set(v_LRpT, Reg_unit, "amd", False)
    code += generate_reg_kernel_set(s_LRpT, Reg_unit, "amd", True)

    return code
import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from define import Feature
import config_ckpt as ckpt
from . import reused_codes

class EUTest:
    def __init__(self, ckpt, name, width_rep, width_for,\
            inicode, repcode, fincode, repcode2, fincode2):
        # inicode;
        # for (n=0..<repeat_for) {
        #   repcode; repcode; ...; repcode; // repeat times
        #   fincode;
        # }
        # repcode: function of i (i -> string)
        self.run = ckpt # run or not? convenient checkpoint feature
        self.name = name        # string
        self.width_rep = width_rep # number
        self.width_for = width_for # number
        self.dont_measure_width = False
        if width_rep * width_for == 0: # if rep or for is 0, only measure latency
            self.dont_measure_width = True
        self.inicode = inicode  # string code
        self.repcode = repcode  # i=0..<repeat -> string code
        self.fincode = fincode  # n=repeat_for -> string code
        self.repcode2 = repcode2 # used for width test, if provided
        self.fincode2 = fincode2 # used for width test, if provided
        if repcode2 == 0: # if not provided, use same code for width test too
            self.repcode2 = repcode
        if fincode2 == 0:
            self.fincode2 = fincode

### ========== Checkpoint Options for exec_units test! ==========
### Modify of add the target insts to the lists below!

nvidia_insts_to_test = [
    EUTest(True, "br", 100, 5,
        "",
        lambda i: "asm volatile(\"MT_BR_{}: bra MT_BR_{};\\n\");\n".format(i, i+1),
        lambda n: "asm volatile(\"MT_BR_{}:\");\n".format(n), 0, 0),

    EUTest(True, "br_jump", 100, 5,
        "",
        lambda i: "asm volatile(\"MT_BR_{}: bra MT_BR_{};\\n\" \"bar.sync 0;\\n\");\n".format(i, i+1),
        lambda n: "asm volatile(\"MT_BR_{}:\");\n".format(n), 0, 0),

    EUTest(True, "i32_mov", 100, 5,
        "asm volatile(\".reg .s32 mt_sop1;\\n\");\n" +
        "asm volatile(\"mov.s32 mt_sop1, 0;\\n\");\n",
        lambda i: "asm volatile(\"mov.s32 mt_sop1, mt_sop1;\\n\");\n",
        lambda n: "", 0, 0),

    EUTest(True, "f32_add", 100, 5,
        "asm volatile(\".reg .f32 mt_fop1, mt_fop2;\\n\");\n" +
        "asm volatile(\"mov.f32 mt_fop1, 0f00000000;\\n\");\n" +
        "asm volatile(\"mov.f32 mt_fop2, 0f3F800000;\\n\");\n",
        lambda i: "asm volatile(\"add.f32 mt_fop1, mt_fop1, mt_fop2;\\n\");\n",
        lambda n: "", 0, 0),

    EUTest(True, "i32_add", 100, 5,
        "asm volatile(\".reg .s32 mt_sop1, mt_sop2;\\n\");\n" +
        "asm volatile(\"mov.s32 mt_sop1, 0;\\n\");\n" +
        "asm volatile(\"mov.s32 mt_sop2, 3;\\n\");\n",
        lambda i: "asm volatile(\"add.s32 mt_sop1, mt_sop1, mt_sop2;\\n\");\n",
        lambda n: "", 0, 0),

    EUTest(True, "sfu_sin", 100, 1,
        "asm volatile(\".reg .f32 mt_fop1;\\n\");\n" +
        "asm volatile(\"mov.f32 mt_fop1, 0f3F000000;\\n\");\n",
        lambda i: "asm volatile(\"sin.approx.ftz.f32 mt_fop1, mt_fop1;\\n\");\n",
        lambda n: "", 0, 0),

    EUTest(True, "shmem", 100, 1,
        "asm volatile(\".shared .align 4 .b8 mt_shm[4];\\n\");\n" +
        "asm volatile(\".reg .u32 mt_var;\\n\");\n" +
        "asm volatile(\"mov.u32 mt_var, mt_shm;\\n\");\n" +
        "asm volatile(\"st.shared.u32 [mt_shm], mt_var;\\n\");\n",
        lambda i: "asm volatile(\"ld.shared.u32 mt_var, [mt_var];\\n\");\n",
        lambda n: "", 0, 0),

    EUTest(True, "gmem", 100, 1,
        "asm volatile(\".reg .u64 mt_addr;\\n\");\n" +
        "asm volatile(\"mov.b64 mt_addr, %0;\\n\"::\"l\"(result));\n" +
        "asm volatile(\"st.global.u64 [mt_addr], mt_addr;\\n\");\n",
        lambda i: "asm volatile(\"ld.global.u64 mt_addr, [mt_addr];\\n\");\n",
        lambda n: "", 0, 0),
]

amd_insts_to_test = [
    EUTest(True, "nop", 100, 5,
        "",
        lambda i: "asm volatile(\"s_nop 0\\n\");\n",
        lambda n: "", 0, 0),

    EUTest(True, "br", 100, 5,
        "",
        lambda i: "asm volatile(\"MT_BR_{}: s_branch MT_BR_{}\\n\");\n".format(i, i+1),
        lambda n: "asm volatile(\"MT_BR_{}:\");\n".format(n),
        lambda i: "asm volatile(\"MTW_BR_{}: s_branch MTW_BR_{}\\n\");\n".format(i, i+1),
        lambda n: "asm volatile(\"MTW_BR_{}:\");\n".format(n)),

    EUTest(True, "br_jump", 100, 5,
        "",
        lambda i: "asm volatile(\"MT_BRJ_{}: s_branch MT_BRJ_{}\\n\" \"s_nop 0\\n\");\n".format(i, i+1),
        lambda n: "asm volatile(\"MT_BRJ_{}:\");\n".format(n),
        lambda i: "asm volatile(\"MTW_BRJ_{}: s_branch MTW_BRJ_{}\\n\" \"s_nop 0\\n\");\n".format(i, i+1),
        lambda n: "asm volatile(\"MTW_BRJ_{}:\");\n".format(n)),

    EUTest(True, "s_mov", 100, 5,
        "int32_t sop1 = 0;\n",
        lambda i: "asm volatile(\"s_mov_b32 %0, %0\\n\":\"+s\"(sop1));\n",
        lambda n: "", 0, 0),

    EUTest(True, "s_add", 100, 5,
        "int32_t sop1 = 0, sop2 = 3;\n",
        lambda i: "asm volatile(\"s_add_i32 %0, %0, %1\\n\":\"+s\"(sop1):\"s\"(sop2));\n",
        lambda n: "asm volatile(\"s_mov_b32 %0, %0\\n\":\"+s\"(sop1));\n", 0, 0),
        ## s_mov was added as fincode to s_add test, because
        ## the compiler inserts s_cmp at the second last position to compensate the
        ## delay, but the last s_add inst contaminates scc, which makes inf loop.

    EUTest(True, "v_mov", 100, 5,
        "int32_t vop1 = 0;\n",
        lambda i: "asm volatile(\"v_mov_b32 %0, %0\\n\":\"+v\"(vop1));\n",
        lambda n: "", 0, 0),

    EUTest(True, "v_add", 100, 5,
        "uint32_t vop1 = 0, sop1 = 3;\n",
        lambda i: "asm volatile(\"v_add_u32 %0, vcc, %0, %1\\n\":\"+v\"(vop1):\"s\"(sop1));\n",
        lambda n: "", 0, 0),

    EUTest(True, "shmem", 100, 1,
        """\
uint32_t vop1 = 0;
__shared__ uint32_t shm[2];
if (result == nullptr) { // ensure shm usage
#pragma unroll 1
    for (int i=0; i<2; i++) shm[i] = (uint32_t)clock();
#pragma unroll 1
    for (int i=0; i<2; i++) asm volatile("v_mov_b32 v0, %0\\n"::"r"(shm[i]));
}
shm[0] = 0;
""",
        lambda i: "asm volatile(\"ds_read_b32 %0, %0\\n\":\"+v\"(vop1));\n" +
                "asm volatile(\"s_waitcnt lgkmcnt(0)\\n\");\n",
        lambda n: "", 0, 0),
    
    EUTest(False, "gmem", 100, 1,
        "uint64_t vop1 = (uint64_t)result;\n" +
        "*result = vop1;\n",
        lambda i: "asm volatile(\"flat_load_dwordx2 %0, %0\\n\":\"+v\"(vop1));\n" +
                "asm volatile(\"s_waitcnt lgkmcnt(0)\\n\");\n",
        lambda n: "", 0, 0),

]

### ========== End of checkpoint options ==========

def measure_latency_code(repeat, name, inicode, repcode, fincode):
    code = """\
__global__ void measure_latency_{}(uint64_t *result){{
    uint64_t sclk, eclk;
""".format(name)
    code += inicode
    code += """\
#pragma unroll 1
    for (int i=0; i<2; i++) { // icache warmup
        __syncthreads();
        sclk = clock();
"""
    for i in range(repeat):
        code += repcode(i)
    code += fincode
    code += """\
        __syncthreads();
        eclk = clock();
    }
    if (hipThreadIdx_x == 0) *result = eclk - sclk;
}
"""
    return code

def kernel_code_func(test_list):
    lat_rep = ckpt.values[ckpt.CKPT.eu_latency_repeats]
    num_running_tests = 0
    code = ""
    for eut in test_list:
        if not eut.run:
            continue
        code += measure_latency_code(lat_rep, eut.name,\
            eut.inicode, eut.repcode, eut.fincode(lat_rep))
        num_running_tests += 1
        if eut.dont_measure_width:
            continue
        code += reused_codes.measure_width_code(eut.width_rep, eut.width_for,\
            eut.name, eut.inicode, eut.repcode2, eut.fincode2(eut.width_rep))
    code += """\
void (*measure_latency[{}])(uint64_t*) = {{
""".format(num_running_tests)
    for eut in test_list:
        if not eut.run:
            continue
        code += "measure_latency_{},\n".format(eut.name)
    code += """\
}};
void (*measure_width[{}])(uint64_t*) = {{
""".format(num_running_tests)
    for eut in test_list:
        if not eut.run:
            continue
        if eut.dont_measure_width:
            code += "NULL,\n"
        else:
            code += "measure_width_{},\n".format(eut.name)
    code += "};\n"
    code += """\
#define NUM_TEST_OPS {}
const char *op_names[{}] = {{
""".format(num_running_tests, num_running_tests)
    for eut in test_list:
        if not eut.run:
            continue
        code += "\"{}\",\n".format(eut.name)
    code += "};\n"
    return code

def verify_constraint(result_values, proj_path):
    return True

def generate_nvidia(result_values):
    return kernel_code_func(nvidia_insts_to_test)

def generate_amd(result_values):
    return kernel_code_func(amd_insts_to_test)
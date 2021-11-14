from .test_template import test_template
from .define import Test
import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from env.config_template import config_values

class FUTest:
    def __init__(self, run_or_not, name, width_rep, width_for,\
            inicode, repcode, fincode, repcode2, fincode2):
        # inicode;
        # for (n=0..<repeat_for) {
        #   repcode; repcode; ...; repcode; // repeat times
        #   fincode;
        # }
        # repcode: function of i (i -> string)
        self.run = run_or_not
        self.name = name        # string
        self.width_rep = width_rep # number
        self.width_for = width_for # number
        self.inicode = inicode  # string code
        self.repcode = repcode  # i=0..<repeat -> string code
        self.fincode = fincode  # n=repeat_for -> string code
        self.repcode2 = repcode2 # used for width test, if provided
        self.fincode2 = fincode2 # used for width test, if provided
        if repcode2 == 0: # if not provided, use same code for width test too
            self.repcode2 = repcode
        if fincode2 == 0:
            self.fincode2 = fincode

### ========== Options for functional_units test! ==========
### Modify or add the target insts to the lists below!
nvidia_insts_to_test = [
    FUTest(True, "br", 100, 5,
        "",
        lambda i: "asm volatile(\"MT_BR_{}: bra MT_BR_{};\\n\");\n".format(i, i+1),
        lambda n: "asm volatile(\"MT_BR_{}:\");\n".format(n), 0, 0),

    FUTest(True, "br_jump", 100, 5,
        "",
        lambda i: "asm volatile(\"MT_BR_{}: bra MT_BR_{};\\n\" \"bar.sync 0;\\n\");\n".format(i, i+1),
        lambda n: "asm volatile(\"MT_BR_{}:\");\n".format(n), 0, 0),

    FUTest(True, "i32_mov", 100, 5,
        "asm volatile(\".reg .s32 mt_sop1;\\n\");\n" +
        "asm volatile(\"mov.s32 mt_sop1, 0;\\n\");\n",
        lambda i: "asm volatile(\"mov.s32 mt_sop1, mt_sop1;\\n\");\n",
        lambda n: "", 0, 0),

    FUTest(True, "f32_add", 100, 5,
        "asm volatile(\".reg .f32 mt_fop1, mt_fop2;\\n\");\n" +
        "asm volatile(\"mov.f32 mt_fop1, 0f00000000;\\n\");\n" +
        "asm volatile(\"mov.f32 mt_fop2, 0f3F800000;\\n\");\n",
        lambda i: "asm volatile(\"add.f32 mt_fop1, mt_fop1, mt_fop2;\\n\");\n",
        lambda n: "", 0, 0),

    FUTest(True, "i32_add", 100, 5,
        "asm volatile(\".reg .s32 mt_sop1, mt_sop2;\\n\");\n" +
        "asm volatile(\"mov.s32 mt_sop1, 0;\\n\");\n" +
        "asm volatile(\"mov.s32 mt_sop2, 3;\\n\");\n",
        lambda i: "asm volatile(\"add.s32 mt_sop1, mt_sop1, mt_sop2;\\n\");\n",
        lambda n: "", 0, 0),

    FUTest(True, "sfu_sin", 100, 1,
        "asm volatile(\".reg .f32 mt_fop1;\\n\");\n" +
        "asm volatile(\"mov.f32 mt_fop1, 0f3F000000;\\n\");\n",
        lambda i: "asm volatile(\"sin.approx.ftz.f32 mt_fop1, mt_fop1;\\n\");\n",
        lambda n: "", 0, 0),

    FUTest(True, "shmem", 100, 1,
        "asm volatile(\".shared .align 4 .b8 mt_shm[4];\\n\");\n" +
        "asm volatile(\".reg .u32 mt_var;\\n\");\n" +
        "asm volatile(\"mov.u32 mt_var, mt_shm;\\n\");\n" +
        "asm volatile(\"st.shared.u32 [mt_shm], mt_var;\\n\");\n",
        lambda i: "asm volatile(\"ld.shared.u32 mt_var, [mt_var];\\n\");\n",
        lambda n: "", 0, 0),

    FUTest(True, "gmem", 100, 1,
        "asm volatile(\".reg .u64 mt_addr;\\n\");\n" +
        "asm volatile(\"mov.b64 mt_addr, %0;\\n\"::\"l\"(result));\n" +
        "asm volatile(\"st.global.u64 [mt_addr], mt_addr;\\n\");\n",
        lambda i: "asm volatile(\"ld.global.u64 mt_addr, [mt_addr];\\n\");\n",
        lambda n: "", 0, 0),
]

amd_insts_to_test = [
    FUTest(False, "nop", 100, 5,
        "",
        lambda i: "asm volatile(\"s_nop 0\\n\");\n",
        lambda n: "", 0, 0),

    FUTest(False, "br", 100, 5,
        "",
        lambda i: "asm volatile(\"MT_BR_{}: s_branch MT_BR_{}\\n\");\n".format(i, i+1),
        lambda n: "asm volatile(\"MT_BR_{}:\");\n".format(n),
        lambda i: "asm volatile(\"MTW_BR_{}: s_branch MTW_BR_{}\\n\");\n".format(i, i+1),
        lambda n: "asm volatile(\"MTW_BR_{}:\");\n".format(n)),

    FUTest(False, "br_jump", 100, 5,
        "",
        lambda i: "asm volatile(\"MT_BRJ_{}: s_branch MT_BRJ_{}\\n\" \"s_nop 0\\n\");\n".format(i, i+1),
        lambda n: "asm volatile(\"MT_BRJ_{}:\");\n".format(n),
        lambda i: "asm volatile(\"MTW_BRJ_{}: s_branch MTW_BRJ_{}\\n\" \"s_nop 0\\n\");\n".format(i, i+1),
        lambda n: "asm volatile(\"MTW_BRJ_{}:\");\n".format(n)),

    FUTest(True, "s_mov", 100, 5,
        "int32_t sop1 = 0;\n",
        lambda i: "asm volatile(\n" if i==0 else "  \"s_mov_b32 %0, %0\\n\"\n",
        lambda n: "  :\"+s\"(sop1));\n", 0, 0),
        ## all amdgpu instructions that can modify SCC can be grouped into one asm directive.
        ## if not, the kernel can go infinite loop.

    FUTest(True, "s_add", 100, 5,
        "int32_t sop1 = 0, sop2 = 3;\n",
        lambda i: "asm volatile(\n" if i==0 else "  \"s_add_i32 %0, %0, %1\\n\"\n",
        lambda n: "  :\"+s\"(sop1):\"s\"(sop2));\n", 0, 0),
        ## all amdgpu instructions that can modify SCC can be grouped into one asm directive.
        ## if not, the kernel can go infinite loop.

    FUTest(False, "v_mov", 100, 5,
        "int32_t vop1 = 0;\n",
        lambda i: "asm volatile(\"v_mov_b32 %0, %0\\n\":\"+v\"(vop1));\n",
        lambda n: "", 0, 0),

    FUTest(False, "v_add", 100, 5,
        "uint32_t vop1 = 0, sop1 = 3;\n",
        lambda i: "asm volatile(\"v_add_u32 %0, vcc, %0, %1\\n\":\"+v\"(vop1):\"s\"(sop1));\n", # for gfx800 series
        # lambda i: "asm volatile(\"v_add_nc_u32 %0, %1, %0\\n\":\"+v\"(vop1):\"s\"(sop1));\n", # for gfx1000 series
        lambda n: "", 0, 0),

    FUTest(False, "shmem", 100, 1,
        """\
uint32_t vop1 = 0;
__gdshmem uint32_t shm[2];
if (result == nullptr) { // ensure shm usage
#pragma unroll 1
    for (int i=0; i<2; i++) shm[i] = (uint32_t)GDclock();
#pragma unroll 1
    for (int i=0; i<2; i++) asm volatile("v_mov_b32 v0, %0\\n"::"r"(shm[i]));
}
shm[0] = 0;
""",
        lambda i: "asm volatile(\"ds_read_b32 %0, %0\\n\":\"+v\"(vop1));\n" +
                "asm volatile(\"s_waitcnt lgkmcnt(0)\\n\");\n",
        lambda n: "", 0, 0),
    
    FUTest(False, "gmem", 100, 1,
        "uint64_t vop1 = (uint64_t)result;\n" +
        "*result = vop1;\n",
        lambda i: "asm volatile(\"flat_load_dwordx2 %0, %0\\n\":\"+v\"(vop1));\n" +
                "asm volatile(\"s_waitcnt lgkmcnt(0)\\n\");\n",
        lambda n: "", 0, 0),

]

### ========== End of Options ==========

def measure_latency_code(repeat, name, inicode, repcode, fincode):
    code = """\
__gdkernel void measure_latency_{}(__gdbufarg uint64_t *result){{
    uint64_t sclk, eclk;
""".format(name)
    code += inicode
    code += """\
#pragma unroll 1
    for (int i=0; i<2; i++) { // icache warmup
        GDsyncthreads();
        sclk = GDclock();
"""
    for i in range(repeat):
        code += repcode(i)
    code += fincode
    code += """\
        GDsyncthreads();
        eclk = GDclock();
    }
    if (GDThreadIdx == 0) *result = eclk - sclk;
}
"""
    return code

def measure_width_code(repeat, repeat_for, name, inicode, repcode, fincode):
    code = """\
__gdkernel void measure_width_{}(__gdbufarg uint64_t *result) {{
    uint64_t sclk, eclk;
""".format(name, repeat_for)
    code += inicode
    code += """\
    int repeats = 1;
#pragma unroll 1
    for (int i=0; i<2; i++) {{
        if (i==1) {{ // icache warmup done
            repeats = {};
            GDsyncthreads();
            sclk = GDclock();
        }}
#pragma unroll 1
        for (int j=0; j<repeats; j++) {{
""".format(repeat_for)
    for i in range(repeat):
        code += repcode(i)
    code += fincode
    code += """\
        }}
        GDsyncthreads();
        eclk = GDclock();
    }}
    if (GDThreadIdx == 0) *result = eclk - sclk;
}}
""".format(repeat_for-1, repeat_for+1)
    return code

class functional_units (test_template):
    test_enum = Test.functional_units
    def generate_kernel(self):
        if self.conf.manufacturer() == "nvidia":
            test_list = nvidia_insts_to_test
        else:
            test_list = amd_insts_to_test
        lat_rep = self.conf.get_confval(config_values.fu_latency_repeats)
        num_running_tests = 0
        code = ""
        for fut in test_list:
            if not fut.run:
                continue
            code += measure_latency_code(lat_rep, fut.name,
                fut.inicode, fut.repcode, fut.fincode(lat_rep))
            code += measure_width_code(fut.width_rep, fut.width_for,
                fut.name, fut.inicode, fut.repcode2, fut.fincode2(fut.width_rep))
            num_running_tests += 1
        code += f"void (*measure_latency[{num_running_tests}])(uint64_t*)={{"
        for fut in test_list:
            if not fut.run:
                continue
            code += f"measure_latency_{fut.name},\n"
        code += f"}};\nvoid (*measure_width[{num_running_tests}])(uint64_t*)={{"
        for fut in test_list:
            if not fut.run:
                continue
            code += f"measure_width_{fut.name},\n"
        code += "};\n"
        code += f"""\
#define NUM_TEST_OPS {num_running_tests}
const char *op_names[{num_running_tests}] = {{
"""
        for fut in test_list:
            if not fut.run:
                continue
            code += f"\"{fut.name}\",\n"
        code += "};\n"
        return code

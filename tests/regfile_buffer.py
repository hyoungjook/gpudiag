from .test_template import test_template
from .define import Test, Feature
import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from env import script_runner

def measure_regfile_code(kernel_name, reg_val, manufacturer, reg_type):
    code = f"""\
__gdkernel void {kernel_name}(__gdbufarg uint32_t *sync, uint32_t G,
        uint64_t timeout, __gdbufarg uint32_t *is_timeout, __gdbufarg uint64_t *totalclk) {{
    uint64_t sclk, eclk, toclk;
    bool thread0 = (GDThreadIdx == 0);
    bool chktime = (timeout != 0);
    volatile uint32_t *chksync = sync;
    GDsyncthreads();
    sclk = GDclock();
    toclk = sclk + timeout;
    if (thread0) GDatomicAdd(sync, 1);
    while(*chksync < G) {{
        if (chktime && GDclock() > toclk) {{
            *is_timeout = 1; return;
        }}
    }}
    eclk = GDclock();
    if (thread0) totalclk[GDBlockIdx] = eclk - sclk;
    // ensure reg usage
    if (G > 0) return; // do not execute, just compile
"""
    if manufacturer == "nvidia":
        code += f"""\
    asm volatile(".reg .b32 tr<{reg_val}>;\\n");
    asm volatile(".reg .b64 addr;\\n");
    asm volatile("mov.u64 addr, %0;\\n"::"l"(sync));
    asm volatile(
"""
        for i in range(reg_val):
            code += f"\"mov.u32 tr{i}, %clock;\\n\"\n"
        code += "\t);\nGDsyncthreads();\n\tasm volatile(\n"
        for i in range(reg_val):
            code += f"\"st.global.u32 [addr+{4*i}], tr{i};\\n\"\n"
        code += "\t);\n}\n"
    else:
        if reg_type == 0:
            code += f"""\
    uint32_t sdummy[{reg_val}];
#pragma unroll {reg_val}
    for(int i=0;i<{reg_val};i++) asm volatile("s_mov_b32 %0, 0\\n":"=s"(sdummy[i]));
#pragma unroll {reg_val}
    for(int i=0;i<{reg_val};i++) asm volatile("s_mov_b32 s0, %0\\n"::"s"(sdummy[i]));
"""
        else:
            code += f"""\
    uint32_t vdummy[{reg_val}];
#pragma unroll {reg_val}
    for(int i=0;i<{reg_val};i++) asm volatile("v_mov_b32 %0, 0\\n":"=v"(vdummy[i]));
#pragma unroll {reg_val}
    for(int i=0;i<{reg_val};i++) asm volatile("v_mov_b32 v0, %0\\n"::"v"(vdummy[i]));
"""
        code += "};\n"
    return code

def generate_reg_kernel_set(LRpT, Reg_unit, regmin, manufacturer, reg_type):
    code = ""
    testR = 0 ; num_tests = 0 ; min_R = 0
    while (True):
        testR += Reg_unit
        if testR <= regmin + 16:
            continue
        if testR > LRpT:
            break
        num_tests += 1
        if min_R == 0:
            min_R = testR
        # generate code for testR
        if manufacturer == "nvidia":
            code += measure_regfile_code(f"measure_reg{reg_type}_{testR}",
                testR - regmin, "nvidia", reg_type)
        else:
            code += measure_regfile_code(f"measure_reg{reg_type}_{testR}",
                testR, "amd", reg_type)
    arrname = f"measure_reg{reg_type}"
    code += "void (*{}[{}])(uint32_t*,uint32_t,uint64_t,uint32_t*,uint64_t*)={{\n"\
        .format(arrname, num_tests)
    for i in range(num_tests):
        code += f"{arrname}_{min_R + i * Reg_unit},\n"
    code += "};\n"
    code += f"#define REGKERNEL_REG{reg_type}_MIN_R {min_R}\n"
    return code

class regfile_buffer (test_template):
    test_enum = Test.regfile_buffer

    def verify_constraints(self):
        tmpsrc = os.path.join(self.build_dir, "tmp_src.cpp")
        tmpobj = os.path.join(self.build_dir, "tmp_obj")
        if self.conf.manufacturer == "nvidia":
            LRpT = int(self.result_values[Feature.limit_registers_per_thread][0])
            testR = LRpT // 2
            regmin = self.calc_regcnt_from_regval_nvidia(
                testR, tmpsrc, tmpobj)
            self.regfile_kernel_regmin = regmin
        else:
            # manually counted for the code! It is larger than the maximum
            self.regfile_kernel_regmin = 32
        return True

    def calc_regcnt_from_regval_nvidia(self, regval, tmpsrc, tmpobj):
        script_runner.create_file(
            "#include \"gpudiag_runtime.h\"\n" +
            measure_regfile_code(
            "test", regval, self.conf.manufacturer(), False) +
            "int main(){GDLaunchKernel(test,dim3(1),dim3(1),0,0," +
            "nullptr,0,0,nullptr,nullptr);return 0;}\n",
            tmpsrc)
        return script_runner.compile_and_check_resource(
            self.conf, tmpsrc, tmpobj, [self.src_dir, self.build_dir])

    def generate_kernel(self):
        Reg_unit = int(self.result_values[Feature.LRpB_test_info0][2])
        LRpT_str = self.result_values[Feature.limit_registers_per_thread]
        num_regtypes = len(LRpT_str)
        code = ""
        for t in range(num_regtypes):
            code += generate_reg_kernel_set(int(LRpT_str[t]), Reg_unit,
                self.regfile_kernel_regmin, self.conf.manufacturer(), t)
        return code
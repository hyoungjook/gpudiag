from .test_template import test_template
from .define import Test
import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from env.config_template import config_values
from env import script_runner, values_delivery

## ============== Host codes ============== 
def deviceprop_host_code(manufacturer, prop_report):
    manu_num = 1 if manufacturer == "nvidia" else 0
    return f"""\
#define MANUFACTURER {manu_num}
#define REPORT_FILE "{prop_report}"
#include "gpudiag_runtime.h"
int main(int argc, char **argv) {{
    GDInit();
    write_deviceprops();
    return 0;
}}\n"""

def include(minimum):
    code = ""
    if minimum:
        code += "#define MINIMUM_FOR_COMPILE_TESTS\n"
    code += "#include \"gpudiag_runtime.h\"\n"
    return code

def compilable_host_code(kernel_name):
    return f"""\
int main(int argc, char **argv) {{
    GDLaunchKernel({kernel_name}, dim3(1), dim3(1), 0, 0, nullptr, true);
    return 0;
}}\n"""

def launchable_abortable_host_code(kernel_name, G, B):
    return f"""\
int main(int argc, char **argv) {{
    GDInit();
    uint8_t hv = 0, *dv; GDMalloc(&dv, sizeof(uint8_t));
    GDMemcpyHToD(dv, &hv, sizeof(uint8_t));
    GDLaunchKernel({kernel_name}, dim3({G}), dim3({B}), 0, 0, dv, true);
    if (!GDIsLastErrorSuccess()) {{GDFree(dv); return 1;}}
    GDSynchronize();
    GDMemcpyDToH(&hv, dv, sizeof(uint8_t));
    GDFree(dv);
    if (hv != 1) return 1;
    return 0;
}}\n"""

def launchable_noabort_host_code(func_name, kernel_name, is_func_array):
    arg_list = "int G, int B"
    kernel_format = kernel_name
    if is_func_array:
        arg_list += ", int idx"
        kernel_format += "[idx]"
    return f"""\
bool {func_name}({arg_list}){{
    GDInit();
    uint8_t hv = 0, *dv; GDMalloc(&dv, sizeof(uint8_t));
    GDMemcpyHToD(dv, &hv, sizeof(uint8_t));
    GDLaunchKernel({kernel_format}, dim3(G), dim3(B), 0, 0, dv, true);
    if (!GDIsLastErrorSuccess()) {{GDFree(dv); return false;}}
    GDSynchronize();
    GDMemcpyDToH(&hv, dv, sizeof(uint8_t));
    GDFree(dv);
    if (hv != 1) return false;
    return true;
}}\n"""

def deliver_value_code(var_name, value):
    return f"#define {var_name} {value}\n"

## ============== Kernel codes ============== 
def return_if_real():
    return """\
    if (real) {
        if (GDThreadIdx == 0 && GDBlockIdx == 0) *r = 1;
        return;
    }\n"""

def normal_kernel(kernel_name):
    return f"""\
__gdkernel void {kernel_name}(__gdbufarg uint8_t *r, bool real) {{
    {return_if_real()}
}}
"""

def shmem_kernel(kernel_name, shmem_size):
    return f"""\
__gdkernel void {kernel_name}(__gdbufarg uint8_t *r, bool real) {{
    {return_if_real()}
    __gdshmem uint8_t arr[{shmem_size}];
    for (int i=0; i<{shmem_size}; i++) arr[i] = (uint8_t)GDclock();
    for (int i=0; i<{shmem_size}; i++) r[i] = arr[i];
}}\n"""

def reg_kernel(kernel_name, manufacturer, reg_val, reg_type):
    code = f"""\
__gdkernel void {kernel_name}(__gdbufarg uint8_t *r, bool real) {{
    {return_if_real()}
"""
    if manufacturer == "nvidia":
        code += f"""\
    asm volatile(".reg .b32 tr<{reg_val}>;\\n");
    asm volatile(".reg .b64 addr;\\n");
    asm volatile("mov.u64 addr, %0;\\n"::"l"(r));
    asm volatile(
"""
        for i in range(reg_val):
            code += f"\"mov.u32 tr{i}, %clock;\\n\"\n"
        code += "\t);\nGDsyncthreads();\n\tasm volatile(\n"
        for i in range(reg_val):
            code += f"\"st.global.u32 [addr+{4*i}], tr{i};\\n\"\n"
        code += "\t);\n}\n"
        return code
    else: # "amd":
        typech = 's' if reg_type==0 else 'v'
        code += f"""\
    asm volatile("{typech}_mov_b32 {typech}{reg_val-1}, 0\\n");
    uint32_t dummy[{reg_val}];
#pragma unroll {reg_val}
    for (int i=0; i<{reg_val}; i++)
        asm volatile("{typech}_mov_b32 %0, 0\\n":"={typech}"(dummy[i]));
#pragma unroll {reg_val}
    for (int i=0; i<{reg_val}; i++)
        asm volatile("{typech}_mov_b32 {typech}0, %0\\n"::"{typech}"(dummy[i]));
}}\n"""
        return code

def reg_kernel_nvidia_real_usage(kernel_name, actual_reg, regmin):
    if actual_reg <= regmin:
        return normal_kernel(kernel_name)
    return reg_kernel(kernel_name, "nvidia", actual_reg-regmin, 0)

def make_kernel_array(array_name, N, x0, dx):
    code = f"void (*{array_name}[{N}])(uint8_t *, bool) = {{\n"
    for i in range(N):
        code += f"{array_name}_{x0 - i * dx},\n"
    code += "};\n"
    return code

def launchable_noabort_reg_kernel_set(compilableLRpT, Reg_unit, regmin, prefix,
        manufacturer, reg_type):
    # generate kernels, testR = Reg_unit * m and compilableLRpT if not included.
    # testR should also be > regmin+16 for safety..
    code = ""
    min_testR = ((regmin+16)//Reg_unit + 1) * Reg_unit
    num_tests = (compilableLRpT - min_testR) // Reg_unit + 1
    num_lrpt_tests = num_tests if (compilableLRpT%Reg_unit==0) else num_tests+1
    for i in range(num_lrpt_tests):
        testR = min_testR + i * Reg_unit
        if i == num_tests:
            testR = compilableLRpT
        if manufacturer == "nvidia":
            code += reg_kernel_nvidia_real_usage(f"{prefix}_{testR}", testR, regmin)
        else: # "amd":
            code += reg_kernel(f"{prefix}_{testR}", "amd", testR, reg_type)
    # generate kernel array
    code += f"void (*{prefix}_LRpT[{num_lrpt_tests}])(uint8_t*,bool) = {{\n"
    if num_lrpt_tests > num_tests:
        code += f"{prefix}_{compilableLRpT},\n"
    for i in range(num_tests):
        code += f"{prefix}_{min_testR + (num_tests-i-1) * Reg_unit},\n"
    code += "};\n"
    code += f"void (*{prefix}_LRpB[{num_tests}])(uint8_t*,bool) = {{\n"
    for i in range(num_tests):
        code += f"{prefix}_{min_testR + i * Reg_unit},\n"
    code += "};\n"
    return code

## ============== Checkfuncs for getmax ============== 
def checkfunc_LSpB_compilable(val, instance):
    return instance.compilable(
        include(minimum=True) + shmem_kernel("test", val) +
        compilable_host_code("test"))
def checkfunc_LRpT_compilable_nvidia(val, instance):
    script_runner.create_file(
        include(minimum=True) + reg_kernel("test", "nvidia", val, 0) +
        compilable_host_code("test"), instance.tmpsrc)
    regcnts = script_runner.compile_and_check_resource(
        instance.conf, instance.tmpsrc, instance.tmpobj, [instance.src_dir])
    if regcnts[0] == 0: # no spill regs
        return True
    return val <= regcnts[1] # reg spill occurred
def checkfunc_LRpT_compilable_amd_sreg(val, instance):
    return instance.compilable(
        include(minimum=True) + reg_kernel("test", "amd", val, 0) +
        compilable_host_code("test"))
def checkfunc_LRpT_compilable_amd_vreg(val, instance):
    return instance.compilable(
        include(minimum=True) + reg_kernel("test", "amd", val, 1) +
        compilable_host_code("test"))

def checkfunc_LTpB_launchable_abort(val, instance):
    return instance.runnable(
        include(minimum=False) + normal_kernel("test") +
        launchable_abortable_host_code("test", 1, val))
def checkfunc_LTpG_launchable_abort(val, instance):
    return instance.runnable(
        include(minimum=False) + normal_kernel("test") +
        launchable_abortable_host_code("test", val, 1))
def checkfunc_LSpB_launchable_abort(val, instance):
    return instance.runnable(
        include(minimum=False) + shmem_kernel("test", val) +
        launchable_abortable_host_code("test", 1, 1))
def checkfunc_LRpT_launchable_abort_nvidia(val, instance):
    return instance.runnable(
        include(minimum=False) +
        reg_kernel_nvidia_real_usage("test", val, instance.regmin) +
        launchable_abortable_host_code("test", 1, 1))
def checkfunc_LRpB_launchable_abort_nvidia(val, args):
    (instance, testR) = args
    if val * instance.warpSize > instance.verifiedLTpB:
        return False # cannot launch over LTpB
    return instance.runnable(
        include(minimum=False) +
        reg_kernel_nvidia_real_usage("test", testR, instance.regmin) +
        launchable_abortable_host_code("test", 1, val * instance.warpSize))

## ============== Tool functions ============== 
def parse_prop_report(prop_report_file):
    report = open(prop_report_file, 'r')
    ret = ["", "", "", "", "", ""]
    for l in report.readlines():
        if l[0] != '#':
            ret[int(l[0])] = l[2:].replace('\n','').replace('[','').replace(']','')
    report.close()
    if ret[4].find(',') >= 0:
        ret4 = [int(ret[4].split(',')[0]), int(ret[4].split(',')[1])]
        ret5 = [int(ret[5].split(',')[0]), int(ret[5].split(',')[1])]
    else:
        ret4 = [int(ret[4])]
        ret5 = [int(ret[5])]
    return (int(ret[0]), int(ret[1]), int(ret[2]), int(ret[3]), ret4, ret5)

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

class kernel_limits (test_template):
    test_enum = Test.kernel_limits

    def compilable(self, code):
        script_runner.create_file(code, self.tmpsrc)
        return script_runner.compile_succeed(self.conf, self.tmpsrc, self.tmpobj,
            [self.src_dir, self.build_dir])
    
    def runnable(self, code):
        if not self.compilable(code):
            exit(1)
        return script_runner.run_succeed(self.conf, self.build_dir, "tmp_obj")

    def get_from_deviceprop(self):
        prop_report_path = os.path.join(self.build_dir, f"prop_report.txt")
        if not self.runnable(deviceprop_host_code(
            self.conf.manufacturer(), prop_report_path)):
            return False
        (self.warpSize, self.propLTpB, self.propLTpG, self.propLSpB,
            self.propLRpT, self.propLRpB) = parse_prop_report(prop_report_path)
        return True

    @staticmethod
    def set_default_if_zero(val, default):
        return val if val>0 else default
    def set_default_if_invalid_prop(self):
        self.propLTpB = kernel_limits.set_default_if_zero(self.propLTpB, 1024)
        self.propLTpG = kernel_limits.set_default_if_zero(self.propLTpG, 1073741824)
        self.propLSpB = kernel_limits.set_default_if_zero(self.propLSpB, 65536)
        for i in range(len(self.propLRpT)):
            self.propLRpT[i] = kernel_limits.set_default_if_zero(self.propLRpT[i], 200)
            self.propLRpB[i] = kernel_limits.set_default_if_zero(self.propLRpB[i], 2048)

    def compilability_check(self):
        self.compilableLSpB = get_max_true(
            checkfunc_LSpB_compilable, self, self.propLSpB)
        if self.conf.manufacturer() == "nvidia":
            self.compilableLRpT = [get_max_true(
                checkfunc_LRpT_compilable_nvidia, self, self.propLRpT[0])]
        else: # "amd":
            self.compilableLRpT = [
                get_max_true(checkfunc_LRpT_compilable_amd_sreg,
                    self, self.propLRpT[0]),
                get_max_true(checkfunc_LRpT_compilable_amd_vreg,
                    self, self.propLRpT[1])]
        return True

    def calc_regmin(self):
        self.regmin = 0
        if self.conf.manufacturer() == "nvidia":
            for i in range(self.compilableLRpT[0]):
                testR = self.compilableLRpT[0] - i
                script_runner.create_file(
                    include(minimum=True) +
                    reg_kernel("test", "nvidia", testR, 0) +
                    compilable_host_code("test"), self.tmpsrc)
                regcnts = script_runner.compile_and_check_resource(
                    self.conf, self.tmpsrc, self.tmpobj, [self.src_dir])
                if regcnts[0] == 0:
                    self.regmin = i
                    return

    def launchability_check_abortable(self):
        # nvidia is assumed here!
        ## TEST LTpB, LTpG, LSpB, LRpT
        Reg_unit = self.conf.get_confval(config_values.register_test_granularity)
        self.verifiedLTpB = get_max_true(checkfunc_LTpB_launchable_abort,
            self, self.propLTpB)
        self.verifiedLTpG = self.propLTpG # too long time if actually verified..
        # self.verifiedLTpG = get_max_true(checkfunc_LTpG_launchable_abort,
        #   self, self.propLTpG)
        self.verifiedLSpB = get_max_true_given_maxval(
            checkfunc_LSpB_launchable_abort, self, self.compilableLSpB,
            self.conf.get_confval(config_values.shared_memory_test_granularity))
        self.verifiedLRpT = [get_max_true_given_maxval(
            checkfunc_LRpT_launchable_abort_nvidia, self, self.compilableLRpT[0],
            Reg_unit)]
        ## TEST LRpB
        self.verifiedLRpB = [0] ; maxb_for_LRpB = []
            # measure for testR = m * Reg_test_unit && testR > regmin+16
        min_testR = ((self.regmin+16)//Reg_unit + 1) * Reg_unit
        num_LRpB_tests = (self.verifiedLRpT[0] - min_testR) // Reg_unit + 1
        for i in range(num_LRpB_tests):
            testR = min_testR + i * Reg_unit
            max_b = get_max_true(
                checkfunc_LRpB_launchable_abort_nvidia,
                (self, testR), self.propLRpB[0] // testR)
            maxb_for_LRpB += [max_b]
            self.verifiedLRpB[0] = max(self.verifiedLRpB[0], max_b * testR)

        LRpB_graph = f"@Reg per Block:{num_LRpB_tests}:" +\
            f"reg/thread:{min_testR}:{Reg_unit}:max B:"
        LRpB_data_str = ""
        for i in range(num_LRpB_tests-1):
            LRpB_data_str += f"{maxb_for_LRpB[i]},"
        LRpB_data_str += f"{maxb_for_LRpB[num_LRpB_tests-1]}"
        LRpB_graph += LRpB_data_str + "\n"
        ## WRITE to report
        reportfile = open(self.report_path, 'w')
        reportfile.write("## ========== kernel_limits report ==========\n")
        reportfile.write(f"limit_threads_per_block={self.verifiedLTpB}\n")
        reportfile.write(f"limit_threads_per_grid={self.verifiedLTpG}\n")
        reportfile.write(f"limit_sharedmem_per_block={self.verifiedLSpB}\n")
        reportfile.write(f"limit_registers_per_thread={self.verifiedLRpT[0]}\n")
        reportfile.write(LRpB_graph)
        reportfile.write(f"LRpB_test_info0=[{num_LRpB_tests}, {min_testR}, {Reg_unit}]\n")
        reportfile.write(f"LRpB_test_data0=[{LRpB_data_str}]\n")
        reportfile.write("LRpB_test_info1=0\n")
        reportfile.write("LRpB_test_data1=0\n")
        reportfile.write(f"limit_registers_per_block={self.verifiedLRpB[0]}\n")
        reportfile.close()

        return True

    def launchability_noabort_kernel(self):
        code = normal_kernel("test_thr")
        shmem_unit = self.conf.get_confval(config_values.shared_memory_test_granularity)
        shmem_test_num = self.compilableLSpB // shmem_unit
        for i in range(shmem_test_num):
            s = self.compilableLSpB - i * shmem_unit
            code += shmem_kernel(f"test_shm_{s}", s)
        code += make_kernel_array(
            "test_shm", shmem_test_num, self.compilableLSpB, shmem_unit)
        reg_unit = self.conf.get_confval(config_values.register_test_granularity)
        if self.conf.manufacturer() == "nvidia":
            code += launchable_noabort_reg_kernel_set(self.compilableLRpT[0],
                reg_unit, self.regmin, "test_reg", "nvidia", 0)
        else: # "amd":
            code += launchable_noabort_reg_kernel_set(self.compilableLRpT[1],
                reg_unit, self.regmin, "test_vreg", "amd", 1)
            code += launchable_noabort_reg_kernel_set(self.compilableLRpT[0],
                reg_unit, self.regmin, "test_sreg", "amd", 0)
        return code
    def launchability_noabort_host(self):
        code = launchable_noabort_host_code("chk_thr", "test_thr", False)
        code += launchable_noabort_host_code("chk_shm", "test_shm", True)
        if self.conf.manufacturer() == "nvidia":
            code += launchable_noabort_host_code("chk_reg_LRpT", "test_reg_LRpT", True)
            code += launchable_noabort_host_code("chk_reg_LRpB", "test_reg_LRpB", True)
        else: # "amd":
            code += launchable_noabort_host_code("chk_vreg_LRpT", "test_vreg_LRpT", True)
            code += launchable_noabort_host_code("chk_vreg_LRpB", "test_vreg_LRpB", True)
            code += launchable_noabort_host_code("chk_sreg_LRpT", "test_sreg_LRpT", True)
            code += launchable_noabort_host_code("chk_sreg_LRpB", "test_sreg_LRpB", True)
        code += deliver_value_code("deliver_propLTpB", self.propLTpB)
        code += deliver_value_code("deliver_propLTpG", self.propLTpG)
        code += deliver_value_code("deliver_compilableLSpB", self.compilableLSpB)
        code += deliver_value_code("deliver_LSpB_test_unit",
            self.conf.get_confval(config_values.shared_memory_test_granularity))
        for i in range(len(self.compilableLRpT)):
            code += deliver_value_code(f"deliver_compilableLRpT{i}", self.compilableLRpT[i])
            code += deliver_value_code(f"deliver_propLRpB{i}", self.propLRpB[i])
        code += deliver_value_code("deliver_Reg_test_unit",
            self.conf.get_confval(config_values.register_test_granularity))
        code += deliver_value_code("deliver_regmin", self.regmin)
        code += deliver_value_code("warp_size", self.warpSize)
        return code
    def launchability_check_not_abortable(self):
        src_path = os.path.join(self.src_dir, f"{self.tname}_launchable.cpp")
        out_bin_path = os.path.join(self.build_dir, self.tname)
        out_host_path = os.path.join(self.build_dir, f"{self.tname}_launchable_host.h")
        out_kern_path = os.path.join(self.build_dir, f"{self.tname}_launchable.h")
        # generate kernels and host codes
        script_runner.create_file(self.launchability_noabort_kernel(), out_kern_path)
        script_runner.create_file(self.launchability_noabort_host(), out_host_path)
        # insert inputs into tmp src file
        values_delivery.embed_values_to_cpp(
            self.conf.manufacturer(), self.report_path, f"{self.tname}_launchable",
            self.result_values, self.conf.get_cvalues(), src_path, self.tmpsrc)
        # compile
        if not script_runner.compile_succeed(
            self.conf, self.tmpsrc, self.tmpobj, [self.src_dir, self.build_dir]):
            return False
        # run
        return script_runner.run_succeed(self.conf, self.build_dir, "tmp_obj")

    def append_warpsize_to_report(self):
        rf = open(self.report_path, 'a')
        rf.write(f"warp_size={self.warpSize}\n")
        rf.close()

    def run(self):
        self.tmpsrc = os.path.join(self.build_dir, "tmp_src.cpp")
        self.tmpobj = os.path.join(self.build_dir, "tmp_obj")

        # Deviceprop
        if not self.get_from_deviceprop():
            return False
        self.set_default_if_invalid_prop()

        # Compilable?
        if not self.compilability_check():
            return False
        self.calc_regmin()

        # Launchable?
        ## currently, only gpgpusim can abort
        kernel_can_abort = self.conf.manufacturer() == "nvidia" and\
            self.conf.is_simulator()
        if kernel_can_abort:
            if not self.launchability_check_abortable():
                return False
        else:
            if not self.launchability_check_not_abortable():
                return False

        self.append_warpsize_to_report()

        return True
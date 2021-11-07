from .test_template import test_template
from .define import Test
import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from env.config_template import config_values
from env import script_runner

def icache_code(repeats):
    return  f"""\
__gdkernel void measure_icache_{repeats}(__gdbufarg uint64_t *result) {{
    uint64_t sclk, eclk;
#pragma unroll 1
    for (int i=0; i<2; i++){{
        sclk = GDclock();
#pragma unroll {repeats}
        for (int j=0; j<{repeats}; j++) {{
            GDsyncthreads();
        }}
        eclk = GDclock();
    }}
    *result = eclk - sclk;
}}\n"""

def icache_kernel_array(max_repeats, interval):
    code = f"""\
void (*measure_icache[{max_repeats // interval}])(uint64_t*) = {{
"""
    for i in range(max_repeats // interval):
        code += f"measure_icache_{(i+1)*interval},\n"
    code += "};\n"
    return code

def embed_instsize_info(instsize, max_repeats, interval_repeats):
    ## inst_size = arr[0] + arr[1] * repeat
    return f"""\
#define ICACHE_INSTSIZE_A {instsize[0]}
#define ICACHE_INSTSIZE_B {instsize[1]}
#define ICACHE_MAX_REPEATS {max_repeats}
#define ICACHE_INTERVAL {interval_repeats}
"""

def parse_dumpfile_ptx(dumpfile):
    first_clock_found = False
    inst_cnt_from_first_clock = 0
    for l in dumpfile.readlines():
        if l.find(';') >= 0:
            if l.find('clock') >= 0:
                if not first_clock_found:
                    first_clock_found = True
                else:
                    break
            if first_clock_found:
                inst_cnt_from_first_clock += 1
    return inst_cnt_from_first_clock * 8 # 8 bytes per 1 inst in gpgpusim

def parse_dumpfile_line_sass(line):
    if line.find(';') < 0:
        return ('', 0) # no instruction
    tokens = line.split(' ')
    for i in len(tokens):
        if tokens[i].find('/*'):
            addr = int(tokens[i].replace('/*','').replace('*/',''), 16)
            break
    inst = ''
    if line.find('CLOCK'):
        inst = 'CLOCK'
    return (inst, addr)

def parse_dumpfile_sass(dumpfile):
    first_clock_addr = -1
    second_clock_addr = -1
    for l in dumpfile.readlines():
        (inst, addr) = parse_dumpfile_line_sass(l)
        if inst == 'CLOCK':
            if first_clock_addr < 0:
                first_clock_addr = addr
            elif second_clock_addr < 0:
                second_clock_addr = addr
                break
    return second_clock_addr - first_clock_addr

def parse_dumpfile_line_hsaco(line):
    if line.find('//') < 0:
        return ('', 0) # no instruction
    tokens = line.split(' ')
    inst = -1 ; addr = -1
    for i in range(len(tokens)):
        if tokens[i] != '':
            if inst == -1:
                inst = tokens[i]
            elif tokens[i] == '//':
                addr = int(tokens[i+1].replace(':',''), 16)
    return (inst, addr)

def parse_dumpfile_hsaco(dumpfile):
    first_memtime_addr = -1
    second_memtime_addr = -1
    for l in dumpfile.readlines():
        (inst, addr) = parse_dumpfile_line_hsaco(l)
        if inst == '':
            continue
        if inst.find('memtime') >= 0:
            if first_memtime_addr < 0:
                first_memtime_addr = addr
            elif second_memtime_addr < 0:
                second_memtime_addr = addr
                break
    return second_memtime_addr - first_memtime_addr

class icache_hierarchy (test_template):
    test_enum = Test.icache_hierarchy

    def verify_constraints(self):
        tmpsrc = os.path.join(self.build_dir, "tmp_src.cpp")
        tmpobj = os.path.join(self.build_dir, "tmp_obj")
        interval = self.conf.get_confval(
            config_values.icache_investigate_interval_B)
        instsize1 = self.calc_instsize(tmpsrc, tmpobj, interval)
        instsize2 = self.calc_instsize(tmpsrc, tmpobj, interval * 2)
        self.instsize_info = [
            2 * instsize1 - instsize2,
            (instsize2 - instsize1) // interval
        ]
        return True

    def generate_kernel(self):
        max_repeats = self.conf.get_confval(config_values.max_icache_investigate_KiB)
        interval = self.conf.get_confval(config_values.icache_investigate_interval_B)
        # convert from KiB and Byte to repeats
        max_repeats = int((max_repeats*1000-self.instsize_info[0])/self.instsize_info[1])
        interval = int(interval / self.instsize_info[1])
        code = ""
        for i in range(max_repeats // interval):
            code += icache_code((i+1) * interval)
        code += icache_kernel_array(max_repeats, interval)
        code += embed_instsize_info(self.instsize_info, max_repeats, interval)
        return code

    def calc_instsize(self, tmpsrc, tmpobj, repeats):
        script_runner.create_file(
            "#include \"gpudiag_runtime.h\"\n" +
            icache_code(repeats) +
            f"int main(){{GDLaunchKernel(measure_icache_{repeats}," +
            "dim3(1),dim3(1),0,0,nullptr); return 0;}\n",
            tmpsrc)
        if not script_runner.compile_succeed(
                self.conf, tmpsrc, tmpobj, [self.src_dir, self.build_dir]):
            exit(1)
        tmpout = tmpobj + "_disasm"
        if not script_runner.objdump_succeed(
                self.conf, tmpobj, tmpout):
            exit(1)
        dumpfile = open(tmpout, 'r')
        if self.conf.manufacturer() == "nvidia":
            if self.conf.is_simulator():
                inst_size = parse_dumpfile_ptx(dumpfile)
            else:
                inst_size = parse_dumpfile_sass(dumpfile)
        else:
            inst_size = parse_dumpfile_hsaco(dumpfile)
        dumpfile.close()
        return inst_size



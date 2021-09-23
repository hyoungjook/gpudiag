import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from define import Feature
import config_ckpt as ckpt
import config_env as conf
from . import reused_codes

def icache_code(repeats):
    return  """\
__global__ void measure_icache_{}(uint64_t *result) {{
    uint64_t sclk, eclk;
#pragma unroll 1
    for (int i=0; i<2; i++){{
        sclk = clock();
#pragma unroll {}
        for (int j=0; j<{}; j++) {{
            __syncthreads();
        }}
        eclk = clock();
    }}
    *result = eclk - sclk;
}}
""".format(repeats, repeats, repeats)

def icache_kernel_array(max_repeats, interval):
    code = """\
void (*measure_icache[{}])(uint64_t*) = {{
""".format(max_repeats // interval)
    for i in range(max_repeats // interval):
        code += "measure_icache_{},\n".format((i+1)*interval)
    code += "};\n"
    return code

## global variable
# inst_size = arr[0] + arr[1] * repeat
icache_code_inst_size_per_repeat = [0, 0]

def embed_instsize_info():
    global icache_code_inst_size_per_repeat
    return """\
#define ICACHE_INSTSIZE_A {}
#define ICACHE_INSTSIZE_B {}
""".format(icache_code_inst_size_per_repeat[0], icache_code_inst_size_per_repeat[1])

def objdump(tmpobj, tmpout):
    cmd = conf.objdump_command
    cmd = cmd.replace("$BIN", tmpobj)
    cmd = cmd.replace("$OUT", tmpout)
    return reused_codes.print_and_exec(cmd) == 0

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

def calc_instsize(proj_path, tmpsrc, tmpobj, repeats):
    reused_codes.write_code(
        "#include <stdint.h>\n#include \"hip/hip_runtime.h\"\n" +\
        icache_code(repeats) +\
        "int main(){{hipLaunchKernelGGL(measure_icache_{},".format(repeats) +\
        "dim3(1),dim3(1),0,0,nullptr); return 0;}\n"
        , tmpsrc)
    if not reused_codes.compile_succeed(proj_path, tmpsrc, tmpobj):
        exit(1)
    tmpout = tmpobj + "_disasm"
    if not objdump(tmpobj, tmpout):
        exit(1)
    dumpfile = open(tmpout, 'r')
    inst_size = 0
    if conf.gpu_manufacturer == "nvidia":
        if conf.simulator_driven:
            inst_size = parse_dumpfile_ptx(dumpfile)
        else:
            inst_size = parse_dumpfile_sass(dumpfile)
    else:
        inst_size = parse_dumpfile_hsaco(dumpfile)
    dumpfile.close()
    return inst_size

def verify_constraint(result_values, proj_path):
    out_dir = os.path.join(proj_path, "build/", conf.gpu_manufacturer, "icache_hierarchy")
    reused_codes.print_and_exec("mkdir -p " + out_dir)
    tmp_src_file = os.path.join(out_dir, "tmp_src.cpp")
    tmp_obj_file = os.path.join(out_dir, "tmp_obj")
    interval = ckpt.values[ckpt.CKPT.icache_investigate_interval]
    instsize1 = calc_instsize(proj_path, tmp_src_file, tmp_obj_file, interval)
    instsize2 = calc_instsize(proj_path, tmp_src_file, tmp_obj_file, interval * 2)
    global icache_code_inst_size_per_repeat
    icache_code_inst_size_per_repeat = [
        2 * instsize1 - instsize2,
        (instsize2 - instsize1) // interval
    ]
    return True

def generate_nvidia(result_values):
    max_repeats = ckpt.values[ckpt.CKPT.max_icache_investigate_repeats]
    interval = ckpt.values[ckpt.CKPT.icache_investigate_interval]
    code = ""
    for i in range(max_repeats // interval):
        code += icache_code((i+1) * interval)
    code += icache_kernel_array(max_repeats, interval)
    code += embed_instsize_info()
    return code

def generate_amd(result_values):
    max_repeats = ckpt.values[ckpt.CKPT.max_icache_investigate_repeats]
    interval = ckpt.values[ckpt.CKPT.icache_investigate_interval]
    code = ""
    for i in range(max_repeats // interval):
        code += icache_code((i+1) * interval)
    code += icache_kernel_array(max_repeats, interval)
    code += embed_instsize_info()
    return code
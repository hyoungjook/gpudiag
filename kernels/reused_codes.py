import subprocess, sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import config_env as conf

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
        copied_bin_dir = os.path.join(conf.simulator_path, "gpudiag/", "kernel_limits")
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

def measure_width_code(repeat, repeat_for, name, inicode, repcode, fincode):
    code = """\
__global__ void measure_width_{}(uint64_t *result) {{
    uint64_t sclk, eclk;
""".format(name, repeat_for)
    code += inicode
    code += """\
    int repeats = 1;
#pragma unroll 1
    for (int i=0; i<2; i++) {{
        if (i==1) {{ // icache warmup done
            repeats = {};
            __syncthreads();
            sclk = clock();
        }}
#pragma unroll 1
        for (int j=0; j<repeats; j++) {{
""".format(repeat_for)
    for i in range(repeat):
        code += repcode(i)
    code += fincode
    code += """\
        }}
        __syncthreads();
        eclk = clock();
    }}
    if (hipThreadIdx_x == 0) *result = eclk - sclk;
}}
""".format(repeat_for-1, repeat_for+1)
    return code

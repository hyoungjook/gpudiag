import os, subprocess

## Shell scripts
def print_and_exec(cmd):
    print("  (execute) : " + cmd)
    return subprocess.call(cmd, shell=True, executable="/bin/bash")

def mkdir(dir):
    return print_and_exec("mkdir -p " + dir)

def cat(path):
    return print_and_exec("cat " + path)

def rm(path):
    return print_and_exec("rm -r " + path)

def cp(src, dest):
    return print_and_exec(f"cp {src} {dest}")

def is_dir_exists(dir):
    return print_and_exec(f"[ -d {dir} ]") == 0

def compile_succeed(conf, src, bin, includes):
    return print_and_exec(conf.compile_cmd(src, bin, includes)) == 0

def compile_and_check_resource(conf, src, bin, includes):
    res = bin + "_resource"
    if print_and_exec(conf.compile_cmd(src, bin, includes) +
            " --resource-usage 2> " + res) != 0:
        cat(res)
        exit(1)
    resfile = open(res, 'r')
    spill_cnt = -1 ; reg_cnt = -1
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

def run_succeed(conf, bin_dir, bin_name):
    bin_path = os.path.join(bin_dir, bin_name)
    if conf.is_simulator():
        main_path = conf.simulator_path()
        if not is_dir_exists(main_path):
            print(f"The simulator path {main_path} doesn't exist.")
            return False
        # copy binary to the simulator path
        copied_bin_dir = os.path.join(main_path, "gpudiag")
        mkdir(copied_bin_dir)
        cp(bin_path, copied_bin_dir)
        # change to main_path and run
        run_status = print_and_exec(
            f"cd {main_path} && " +
            f"{conf.run_cmd(copied_bin_dir, bin_name)}")
        # copy back files and remove tmp dirs
        cp(os.path.join(copied_bin_dir, "*"), bin_dir)
        rm(os.path.join(main_path, "gpudiag"))
    else:
        run_status = print_and_exec(
            conf.run_cmd(bin_dir, bin_name))
    return run_status == 0

def objdump_succeed(conf, tmpobj, tmpout):
    return print_and_exec(conf.objdump_cmd(tmpobj, tmpout)) == 0

## Files
def create_file(content, path):
    f = open(path, 'w')
    f.write(content)
    f.close()

def append_to_file(content, path):
    f = open(path, 'a')
    f.write(content)
    f.close()


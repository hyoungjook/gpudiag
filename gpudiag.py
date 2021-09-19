import os
import subprocess
from datetime import datetime
import importlib
import matplotlib.pyplot as plt
from define import Test, Feature
import config_env as conf
import config_ckpt as ckpt

def verify_config():
    if conf.gpu_manufacturer != "nvidia" and\
        conf.gpu_manufacturer != "amd":
        print("gpu_manufacturer value is not allowed.")
        print("Please re-check config.py")
        exit(1)

def print_and_exec(cmd):
    print("  (execute) : " + cmd)
    return subprocess.call(cmd, shell=True, executable="/bin/bash")

def get_values_from_result_txt(result_file):
    values = {}
    for feat in Feature:
        fname = feat.name
        val = "not found"
        report = open(result_file, 'r')
        for line in report.readlines():
            if line[0] == '#' or line[0] == '@':
                continue
            idx = line.find('=')
            if idx < 0:
                continue
            key = line[:idx]
            if key == fname:
                # use the latest value
                val = line[idx+1:].replace('\n', '')
        report.close()
        if val != "not found":
            val = val.replace('[', '')
            val = val.replace(']', '')
            val = val.replace(' ', '')
            values[feat] = val.split(',')
    return values

def verify_constraints(test, proj_path):
    testname = test.name
    result_txt = os.path.join(proj_path, "result.txt")
    result_values = get_values_from_result_txt(result_txt)
    kgen = importlib.import_module("kernels." + testname)
    return kgen.verify_constraint(result_values, proj_path)

def inline_c_array_str(i, values):
    if i == 0:
        return values[0]
    if i == 1:
        return "(i==1?{}:{})".format(values[1], values[0])
    return "(i=={}?{}:{})".format(i, values[i], inline_c_array_str(i-1, values))

def embed_input_line(line, manufacturer, out_dir, result_values):
    defword = "#define "
    if line.find(defword + "MANUFACTURER") >= 0:
        return defword + "MANUFACTURER {}\n".format(1 if manufacturer=="nvidia" else 0)
    if line.find(defword + "REPORT_DIR") >= 0:
        return defword + "REPORT_DIR \"{}\"\n".format(out_dir + "/")
    for feat in Feature:
        fname = feat.name
        if line.find(defword + fname) >= 0:
            modline = defword + fname
            values = result_values[feat] # Keyerror if result.txt don't have the data
            if len(values) > 1:
                modline += "(i)"
            modline += " {}\n".format(inline_c_array_str(len(values)-1, values))
            return modline
    for ck in ckpt.CKPT:
        cname = "ckpt_" + ck.name
        if line.find(defword + cname) >= 0:
            return defword + cname + " {}\n".format(ckpt.values[ck])
    return line

def embed_input(manufacturer, out_dir, result_file, in_file, out_file):
    original = open(in_file, 'r')
    modified = open(out_file, 'w')
    modify_done = False
    result_values = get_values_from_result_txt(result_file)
    for line in original.readlines():
        if line.find("#include") >= 0:
            modify_done = True
        if not modify_done:
            modline = embed_input_line(line, manufacturer, out_dir, result_values)
        else:
            modline = line
        modified.write(modline)
    original.close()
    modified.close()

def compile(test, proj_path):
    testname = test.name
    src_dir = os.path.join(proj_path, "tests/")
    src_path = os.path.join(src_dir, testname + ".cpp")
    out_dir = os.path.join(proj_path, "build/", conf.gpu_manufacturer, testname)
    out_path = os.path.join(out_dir, testname)
    
    # make build dir
    print_and_exec("mkdir -p " + out_dir)

    # generate kernel if needed
    result_txt = os.path.join(proj_path, "result.txt")
    result_values = get_values_from_result_txt(result_txt)
    kgen = importlib.import_module("kernels." + testname)
    kernel_code = "// Generated from kernels/{}.py\n".format(testname)
    if conf.gpu_manufacturer == "nvidia":
        kernel_code += kgen.generate_nvidia(result_values)
    else:
        kernel_code += kgen.generate_amd(result_values)
    kernel_file = open(os.path.join(out_dir, testname + ".h"), 'w')
    kernel_file.write(kernel_code)
    kernel_file.close()

    # insert inputs into tmp source file
    copied_src_path = os.path.join(src_dir, testname + "_tmp.cpp")
    embed_input(conf.gpu_manufacturer, out_dir, result_txt,\
        src_path, copied_src_path)

    # compile
    cmd = conf.compile_command
    cmd = cmd.replace("$SRC", copied_src_path)
    cmd = cmd.replace("$BIN", out_path)
    cmd += " -I" + out_dir                              # kernel.h
    cmd += " -I" + os.path.join(proj_path, "tests/")    # tool.h
    compile_status = print_and_exec(cmd)

    # remove tmp source file
    cmd = "rm " + copied_src_path
    print_and_exec(cmd)

    if compile_status != 0:
        print("Error in compiling " + testname + ".")
        return False
    return True

def run(test, proj_path):
    testname = test.name
    bin_dir = os.path.join(proj_path, "build/", conf.gpu_manufacturer, testname)
    bin_name = testname
    bin_path = os.path.join(bin_dir, bin_name)
    if conf.simulator_driven:
        # check simulator main_path
        main_path = conf.simulator_path
        if main_path[0] != "/":
            print("The simulator path " + main_path + "should be absolute.")
            print("Please re-check the config.py")
            return False
        cmd = "[ -d " + main_path + " ]"
        if print_and_exec(cmd) != 0:
            print("The simulator path " + main_path + "doesn't exist.")
            print("Please re-check the config.py")
            return False

        # copy binary to the simulator path
        copied_bin_dir = os.path.join(main_path, "gpudiag/", testname)
        cmd = "mkdir -p " + copied_bin_dir
        cmd += " && cp " + bin_path + " " + copied_bin_dir
        if print_and_exec(cmd) != 0:
            print("Copying into simulator path failed.")
            return False
        
        # cd, run
        cmd = "cd " + main_path + " && "
        run_cmd = conf.run_command
        run_cmd = run_cmd.replace("$DIR", copied_bin_dir)
        run_cmd = run_cmd.replace("$BIN", bin_name)
        cmd += run_cmd
        run_status = print_and_exec(cmd)

        # copy back files and remove simpath/gpudiag
        cmd = "cp " + os.path.join(copied_bin_dir, "*") + " " + bin_dir
        cmd += " ; rm -rf " + os.path.join(main_path, "gpudiag")
        print_and_exec(cmd)
        if run_status != 0:
            print("Error in running " + testname + ".")
            return False
    else:
        cmd = conf.run_command
        cmd = cmd.replace("$DIR", bin_dir)
        cmd = cmd.replace("$BIN", bin_name)
        if print_and_exec(cmd) != 0:
            print("Error in running " + testname + ".")
            return False

    return True

def add_datetime_to_result_txt(proj_path):
    result_path = os.path.join(proj_path, "result.txt")
    result = open(result_path, 'a')
    result.write("### Executed at {} ###\n".format(datetime.now()))
    result.close()

def create_graph_directory(proj_path):
    graph_dir = os.path.join(proj_path, "result_graphs")
    print_and_exec("mkdir -p " + graph_dir)

def draw_graph(line, test, proj_path):
    # format: @title:dataN:xlabel:x0:dx:ylabel:y0,y1,...,yn\n
    # or, instead of :x0:dx:, xdata manually with :p:x0,x1,...,xn:
    # optional line drawings: ..., yn:a,b,label\n where y=ax+b
    plt.clf()
    if line[0] != '@':
        exit(1)
    tokens = line[1:].replace('\n', '').split(':')
    title = tokens[0] ; dataN = int(tokens[1])
    # xdata
    xlabel = tokens[2]
    max_x = 0 ; max_y = 0 ; xdata = []
    if tokens[3] != 'p':
        x0 = float(tokens[3]) ; dx = float(tokens[4])
        for i in range(dataN):
            xdata += [x0 + i * dx]
        max_x = x0 + (dataN-1) * dx
    else:
        xstrs = tokens[4].split(',')
        for i in range(dataN):
            xdata += [float(xstrs[i])]
            max_x = max(max_x, xdata[i])
    # ydata
    ylabel = tokens[5]
    ydata = [] ; ystrs = tokens[6].split(',')
    for i in range(dataN):
        ydata += [float(ystrs[i])]
        max_y = max(max_y, ydata[i])
    # figure name
    figure_name = title.replace(' ', '_').replace('(', '_').replace(')', '_')
    figure_path = os.path.join(proj_path, "result_graphs/",\
        test.name + "_" + figure_name + ".png")
    # additional line
    if len(tokens) > 7:
        slope = float(tokens[7].split(',')[0])
        y_itcpt = float(tokens[7].split(',')[1])
        linelabel = tokens[7].split(',')[2]
        addline_x = []; addline_y = []
        if slope == 0:
            addline_x = [0, max_x] ; addline_y = [y_itcpt, y_itcpt]
        else:
            x_at_maxy = (max_y - y_itcpt) / slope
            if x_at_maxy < max_x:
                addline_x = [0, x_at_maxy] ; addline_y = [y_itcpt, max_y]
            else:
                addline_x = [0, max_x] ; addline_y = [y_itcpt, y_itcpt+slope*max_x]
        plt.plot(addline_x, addline_y, 'k--', label=linelabel)
    # plot
    plt.plot(xdata, ydata, 'ko')
    if len(tokens) > 7:
        plt.legend()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(figure_path)
    plt.clf()

def update(test, proj_path, time_consumed):
    testname = test.name
    bin_dir = os.path.join(proj_path, "build/", conf.gpu_manufacturer, testname)
    report_path = os.path.join(bin_dir, "report.txt")
    result_path = os.path.join(proj_path, "result.txt")
    # cat report.txt
    report_path = os.path.join(bin_dir, "report.txt")
    cmd = "cat " + report_path
    print_and_exec(cmd)
    # copy to result.txt
    try:
        report = open(report_path, 'r')
    except FileNotFoundError:
        print("Report file " + report_path + " doesn't exist.")
        return False
    result = open(result_path, 'a')
    for line in report.readlines():
        result.write(line)
        if line[0] == '@':
            draw_graph(line, test, proj_path)
    result.write("Time consumed: {}\n".format(time_consumed))
    result.write("\n")
    result.close()
    report.close()
    return True

if __name__ == "__main__":
    verify_config()
    project_path = os.path.dirname(os.path.abspath(__file__))
    add_datetime_to_result_txt(project_path)
    create_graph_directory(project_path)
    for test in Test:
        if not ckpt.run_test[test]:
            continue
        start_time = datetime.now()
        if test == Test.verify_limits:
            verify_limit_test = importlib.import_module("tests." + test.name)
            if not verify_limit_test.verify(project_path,\
                get_values_from_result_txt(os.path.join(project_path, "result.txt"))):
                exit(1)
        else:
            if not verify_constraints(test, project_path):
                exit(1)
            if not compile(test, project_path):
                exit(1)
            if not run(test, project_path):
                exit(1)
        end_time = datetime.now()
        if not update(test, project_path, end_time - start_time):
            exit(1)
## value passings between python, host and device codes
import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from tests.define import Feature
from .config_template import config_values

## result.txt -> python
def extract_values_from_txt(txt_path):
    # return: dict: feature -> array of strings
    values = {}
    for feat in Feature:
        fname = feat.name
        found = False
        report = open(txt_path, 'r')
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
                found = True
        report.close()
        if found:
            val = val.replace('[', '').replace(']', '')
            val = val.replace(' ', '')
            values[feat] = val.split(',')
    return values

## python -> host_code.cpp
def inline_c_array_str(i, vals):
    if i == 1:
        return f"({vals[0]})"
    return f"(i=={i-1}?{vals[i-1]}:{inline_c_array_str(i-1, vals)})"

def embed_input_line(line, manufacturer, report_path, tname, fvalues, cvalues):
    defword = "#define "
    if line.find(defword + "MANUFACTURER") >= 0:
        manufacturer_num = 1 if manufacturer=="nvidia" else 0
        return f"{defword}MANUFACTURER {manufacturer_num}\n"
    if line.find(defword + "REPORT_FILE") >= 0:
        return f"{defword}REPORT_FILE \"{report_path}\"\n"
    if line.find(defword + "KERNEL_FILE") >= 0:
        return f"{defword}KERNEL_FILE \"{tname}.h\"\n"
    for feat in Feature:
        fname = feat.name
        if line.find(defword + fname) >= 0:
            modline = defword + fname + "(i)"
            try:
                vals = fvalues[feat]
            except KeyError:
                print(f"The required {fname} doesn't exist in result.txt")
                exit(1)
            modline += f" {inline_c_array_str(len(vals), vals)}\n"
            return modline
    for confval in config_values:
        cname = f"ckpt_{confval.name}"
        if line.find(defword + cname) >= 0:
            return f"{defword}{cname} {cvalues[confval]}\n"
    return line

def embed_values_to_cpp(manufacturer, report_path, tname, fvalues, cvalues,
        in_src, out_src):
    original = open(in_src, 'r')
    modified = open(out_src, 'w')
    modify_done = False
    for line in original.readlines():
        if line.find('#include') >= 0:
            modify_done = True
        if not modify_done:
            modline = embed_input_line(line,
                manufacturer, report_path, tname, fvalues, cvalues)
        else:
            modline = line
        modified.write(modline)
    original.close()
    modified.close()
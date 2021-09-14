import os, subprocess

cleanup_list = [
    'build/',
    'result.txt',
    'result_graphs/',
]

def print_and_exec(cmd):
    print("  (execute) : " + cmd)
    return subprocess.call(cmd, shell=True, executable="/bin/bash")

if __name__ == "__main__":
    project_path = os.path.dirname(os.path.abspath(__file__))
    for target in cleanup_list:
        cmd = "rm -rf " + os.path.join(project_path, target)
        print_and_exec(cmd)
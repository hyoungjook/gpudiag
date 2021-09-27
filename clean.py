import os
from env import script_runner

cleanup_list = [
    'build/',
    'result.txt',
    'result_graphs/',
]

if __name__ == "__main__":
    project_path = os.path.dirname(os.path.abspath(__file__))
    for target in cleanup_list:
        script_runner.rm(os.path.join(project_path, target))
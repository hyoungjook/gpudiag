import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from define import Feature
import config_ckpt as ckpt
from . import reused_codes

def generate_nvidia(result_values):
    # measure_br_time
    code = ""
    code += reused_codes.measure_width_code(100, 5, "br", "",\
        lambda i: "asm volatile(\"MT_BR_{}: bra MT_BR_{};\\n\");\n".format(i, i+1),\
        "asm volatile(\"MT_BR_{}:\\n\");\n".format(100))

    # measure_num_mp
    code += reused_codes.measure_num_mp_code(100, 2, "",\
        lambda i: "asm volatile(\"MNM_BR_{}: bra MNM_BR_{};\\n\");\n".format(i, i+1),\
        "asm volatile(\"MNM_BR_{}:\\n\");\n".format(100))

    # measure warpstatebuffer
    code += reused_codes.measure_warpstatebuffer_code()

    # measure shmem
    limit_shmem = int(result_values[Feature.limit_sharedmem_per_block][0])
    shmem_unit = ckpt.values[ckpt.CKPT.shared_memory_test_granularity]
    code += reused_codes.measure_shmem_code(limit_shmem, shmem_unit)
    return code

def generate_amd(result_values):
    # measure_br_time
    code = ""
    code += reused_codes.measure_width_code(100, 5, "br", "",\
        lambda i: "asm volatile(\"MT_BR_{}: s_branch MT_BR_{}\\n\");\n".format(i, i+1),\
        "asm volatile(\"MT_BR_{}:\\n\");\n".format(100))

    # measure_num_mp
    code += reused_codes.measure_num_mp_code(100, 2, "",\
        lambda i: "asm volatile(\"MNM_BR_{}: s_branch MNM_BR_{}\\n\");\n".format(i, i+1),\
        "asm volatile(\"MNM_BR_{}:\\n\");\n".format(100))

    # measure warpstatebuffer
    code += reused_codes.measure_warpstatebuffer_code()

    # measure shmem
    limit_shmem = int(result_values[Feature.limit_sharedmem_per_block][0])
    shmem_unit = ckpt.values[ckpt.CKPT.shared_memory_test_granularity]
    code += reused_codes.measure_shmem_code(limit_shmem, shmem_unit)
    return code
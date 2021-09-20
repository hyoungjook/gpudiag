import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from define import Feature
import config_ckpt as ckpt

def verify_constraint(result_values, proj_path):
    return True

def dcache_linesize_code(num_repeat):
    return """\
__global__ void measure_l1d_linesize(uint32_t *arr, uint64_t *result) {{
    // arr should be ready of sizeof(uint32_t) * num_repeat
    // first value should be ignored; icache can affect the first value
    uint64_t sclk, eclk;
    uint32_t value, dummy = 0;
    uint32_t *ptr = arr;
    __shared__ uint64_t result_buf[{}];
#pragma unroll 1
    for (int i=0; i<{}; i++) {{
        ptr = arr + i;
        value += (uint32_t)((uint64_t)ptr);
        __syncthreads();
        sclk = clock();
        value = *ptr;
        dummy += value;
        __syncthreads();
        eclk = clock();
        result_buf[i] = eclk - sclk;
    }}
    // first value is useless anyway
    result[0] = (uint64_t)dummy + result_buf[0];
    for (int i=1; i<{}; i++) {{
        result[i] = result_buf[i];
    }}
}}
""".format(num_repeat, num_repeat, num_repeat)

def dcache_hierarchy_code(max_lines):
    return """\
__global__ void measure_dcache(uint32_t *arr, int wordperline, uint64_t *result) {{
    // arr should be ready of sizeof(uint32_t) * wordperline * max_lines
    // first value should be ignored; icache can affect the first value
    uint64_t sclk, eclk;
    uint32_t value, dummy = 0;
    uint32_t *ptr = arr;
    __shared__ uint64_t result_buf[{}];
    for (int i={}-1; i>=0; i--) {{ // warmup
        int test_idx = i * wordperline;
        value = arr[test_idx];
        dummy += value;
    }}
#pragma unroll 1
    for (int i=0; i<{}; i++) {{
        ptr = arr + (i * wordperline);
        value += (uint32_t)((uint64_t)ptr);
        __syncthreads();
        sclk = clock();
        value = *ptr;
        dummy += value;
        __syncthreads();
        eclk = clock();
        result_buf[i] = eclk - sclk;
    }}
    // first value is useless anyway
    result[0] = (uint64_t)dummy + result_buf[0];
    for (int i=1; i<{}; i++) {{
        result[i] = result_buf[i];
    }}
}}
""".format(max_lines, max_lines, max_lines, max_lines)

def generate_nvidia(result_values):
    code = dcache_linesize_code(100)
    limit_shm = int(result_values[Feature.limit_sharedmem_per_block][0])
    code += dcache_hierarchy_code(limit_shm // 8)
    return code

def generate_amd(result_values):
    code = dcache_linesize_code(100)
    limit_shm = int(result_values[Feature.limit_sharedmem_per_block][0])
    code += dcache_hierarchy_code(limit_shm // 8)
    return code

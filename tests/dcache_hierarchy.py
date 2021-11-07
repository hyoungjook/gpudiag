from .test_template import test_template
from .define import Test, Feature
import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from env.config_template import config_values

def dcache_linesize_code(num_repeat):
    return f"""\
__gdkernel void measure_l1d_linesize(__gdbufarg uint32_t *arr, __gdbufarg uint64_t *result) {{
    // arr should be ready of sizeof(uint32_t) * num_repeat
    // first value should be ignored; icache can affect the first value
    uint64_t sclk, eclk;
    uint32_t value, dummy = 0;
    uint32_t *ptr = arr;
    __gdshmem uint64_t result_buf[{num_repeat}];
#pragma unroll 1
    for (int i=0; i<{num_repeat}; i++) {{
        ptr = arr + i;
        value += (uint32_t)((uint64_t)ptr);
        GDsyncthreads();
        sclk = GDclock();
        value = *ptr;
        dummy += value;
        GDsyncthreads();
        eclk = GDclock();
        result_buf[i] = eclk - sclk;
    }}
    // first value is useless anyway
    result[0] = (uint64_t)dummy + result_buf[0];
    for (int i=1; i<{num_repeat}; i++) {{
        result[i] = result_buf[i];
    }}
}}
"""

def dcache_hierarchy_code(max_lines):
    return f"""\
__gdkernel void measure_dcache(__gdbufarg uint32_t *arr, int wordperline, __gdbufarg uint64_t *result) {{
    // arr should be ready of sizeof(uint32_t) * wordperline * max_lines
    // first value should be ignored; icache can affect the first value
    uint64_t sclk, eclk;
    uint32_t value, dummy = 0;
    uint32_t *ptr = arr;
    __gdshmem uint64_t result_buf[{max_lines}];
    for (int i={max_lines}-1; i>=0; i--) {{ // warmup
        int test_idx = i * wordperline;
        value = arr[test_idx];
        dummy += value;
    }}
#pragma unroll 1
    for (int i=0; i<{max_lines}; i++) {{
        ptr = arr + (i * wordperline);
        value += (uint32_t)((uint64_t)ptr);
        GDsyncthreads();
        sclk = GDclock();
        value = *ptr;
        dummy += value;
        GDsyncthreads();
        eclk = GDclock();
        result_buf[i] = eclk - sclk;
    }}
    // first value is useless anyway
    result[0] = (uint64_t)dummy + result_buf[0];
    for (int i=1; i<{max_lines}; i++) {{
        result[i] = result_buf[i];
    }}
}}
#define NUM_DCACHE_REPEAT {max_lines}
"""

class dcache_hierarchy(test_template):
    test_enum = Test.dcache_hierarchy
    
    def generate_kernel(self):
        code = dcache_linesize_code(100)
        limit_shm = int(self.result_values[Feature.limit_sharedmem_per_block][0])
        limit_ckpt = self.conf.get_confval(config_values.max_dcache_investigate_repeats)
        code += dcache_hierarchy_code(min(limit_shm//8, limit_ckpt))
        return code
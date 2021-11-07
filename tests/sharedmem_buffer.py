from .test_template import test_template
from .define import Test, Feature
import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from env.config_template import config_values

def measure_shmem_code(limit_shmem, shmem_unit):
    code = ""
    for s in range(shmem_unit, limit_shmem+1, shmem_unit):
        code += f"""\
__gdkernel void measure_shmem_{s}(__gdbufarg uint32_t *sync, uint32_t G,
        uint64_t timeout, __gdbufarg uint32_t *is_timeout, __gdbufarg uint64_t *totalclk) {{
    uint64_t sclk, eclk, toclk;
    bool thread0 = (GDThreadIdx == 0);
    bool chktime = (timeout != 0);
    volatile uint32_t *chksync = sync;
    GDsyncthreads();
    sclk = GDclock();
    toclk = sclk + timeout;
    if (thread0) GDatomicAdd(sync, 1);
    while(*chksync < G) {{
        if (chktime && GDclock() > toclk) {{
            *is_timeout = 1; return;
        }}
    }}
    eclk = GDclock();
    if (thread0) totalclk[GDBlockIdx] = eclk - sclk;
    // ensure shmem usage
    if (G > 0) return; // do not execute, just compile
    __gdshmem uint8_t arr[{s}];
    uint8_t *fakeptr = (uint8_t*)sync;
    for (int i=0; i<{s}; i++) arr[i] = (uint8_t)GDclock();
    for (int i=0; i<{s}; i++) fakeptr[i] = arr[i];
}}
"""
    code += f"""\
void (*measure_shmem[{limit_shmem//shmem_unit}])(uint32_t*,uint32_t,uint64_t,uint32_t*,uint64_t*)={{
"""
    for s in range(shmem_unit, limit_shmem+1, shmem_unit):
        code += f"measure_shmem_{s},\n"
    code += "};\n"
    return code

class sharedmem_buffer (test_template):
    test_enum = Test.sharedmem_buffer
    
    def generate_kernel(self):
        limit_shmem = int(self.result_values[Feature.limit_sharedmem_per_block][0])
        shmem_unit = self.conf.get_confval(config_values.shared_memory_test_granularity)
        return measure_shmem_code(limit_shmem, shmem_unit)
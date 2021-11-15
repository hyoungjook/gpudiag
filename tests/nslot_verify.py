from .test_template import test_template
from .define import Test
from . import functional_units
import sys, os

### ========== Options for nslot_verify test! ==========
### Pick the FUTest to investigate, by its name
nvidia_inst_to_test = "shmem"
amd_inst_to_test = "shmem"
### How many for repetitions?
for_repetitions = 1000

### ========== End of Options ==========

def measure_width_multiple_G_code(repeat, repeat_for, inicode, repcode, fincode):
    code = f"""\
__gdkernel void measure_width_verify(__gdbufarg uint64_t *result) {{
    uint64_t sclk, eclk;
    {inicode}
    int repeats = 1;
#pragma unroll 1
    for (int i=0; i<2; i++) {{
        if (i==1) {{ // icache warmup done
            repeats = {repeat_for};
            GDsyncthreads();
            sclk = GDclock();
        }}
#pragma unroll 1
        for (int j=0; j<repeats; j++) {{
            asm volatile(
"""
    for i in range(repeat):
        code += repcode(i)
    code += fincode
    code += f"""\
            );
        }}
        GDsyncthreads();
        eclk = GDclock();
    }}
    if (GDThreadIdx == 0) {{
        result[2*GDBlockIdx] = sclk;
        result[2*GDBlockIdx+1] = eclk;
    }}
}}
"""
    return code

class nslot_verify (test_template):
    test_enum = Test.nslot_verify
    def generate_kernel(self):
        if self.conf.manufacturer() == "nvidia":
            target = nvidia_inst_to_test
            test_list = functional_units.nvidia_insts_to_test
        else:
            target = amd_inst_to_test
            test_list = functional_units.amd_insts_to_test
        code = ""
        for fut in test_list:
            if fut.name == target:
                target_fut = fut
                break
        code += measure_width_multiple_G_code(
            target_fut.width_rep, for_repetitions,
            target_fut.inicode, target_fut.repcode2,
            target_fut.fincode2(target_fut.width_rep))
        code += f"""\
#define TEST_INST_NAME "{target_fut.name}"
"""
        return code

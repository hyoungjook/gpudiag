from .test_template import test_template
from .define import Test

def test_warpsched_code(repeat, clock_t, clock_asm_code):
    code = f"""
__gdkernel void test_warpsched(__gdbufarg uint64_t *result) {{
    {clock_t} buf[{repeat}];
    #pragma unroll 1
    for (int i=0; i<2; i++) {{
        GDsyncthreads();
        #pragma unroll {repeat}
        for (int j=0; j<{repeat}; j++) {{
            {clock_asm_code}
        }}
        GDsyncthreads();
    }}
    const int warp_id = GDThreadIdx / warp_size(0);
    if (GDThreadIdx % warp_size(0) == 0) {{
        #pragma unroll {repeat}
        for (int i=0; i<{repeat}; i++)
            result[warp_id * {repeat} + i] = (uint64_t)buf[i];
    }}
}}
"""
    return code

class warpsched_policy (test_template):
    test_enum = Test.warpsched_policy

    def generate_kernel(self):
        if self.conf.manufacturer() == "nvidia":
            return test_warpsched_code(10, "uint32_t",
                "asm volatile(\"mov.u32 %0, %%clock;\\n\":\"=r\"(buf[j]));"
            )
        else:
            return test_warpsched_code(10, "uint64_t",
                "asm volatile(\"s_memtime %0\\n\":\"=s\"(buf[j]));"
            )
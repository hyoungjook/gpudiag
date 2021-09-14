import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from define import Feature

def generate_nvidia(result_values):
    num_repeat = int(result_values[Feature.limit_registers_per_thread][0]) - 10
    code = """\
__global__ void measure_l1i_linesize(uint64_t *result){{
    asm volatile(".reg .u32 mll_rbuf<{}>, dummy;\\n");
""".format(num_repeat)
    for i in range(num_repeat):
        code += """\
    asm volatile("mov.u32 mll_rbuf{}, %clock;\\n");
    asm volatile("mov.u32 dummy, mll_rbuf{};\\n");
""".format(i, i)
    code += """\
    asm volatile(".reg .u64 mll_addr;\\n");
    asm volatile("mov.u64 mll_addr, %0;\\n"::"l"(result));
"""
    for i in range(num_repeat):
        code += """\
    asm volatile("st.global.u32 [mll_addr+{}], mll_rbuf{};\\n");
""".format(8*i, i)
    code += "}\n"
    return code


def generate_amd(result_values):
    num_repeat = int(int(result_values[Feature.limit_registers_per_thread][0])/2)-3
    code = """\
__global__ void measure_l1i_linesize(uint64_t *result){
    asm volatile("s_load_dwordx2 s[0:1], s[4:5], 0x0\\n");
"""
    for i in range(num_repeat):
        code += """\
    asm volatile("s_memtime s[{}:{}]\\n");
    asm volatile("s_mov_b32 s{}, s{}\\n");
    asm volatile("s_nop 0\\n");
""".format(2*(i+1), 2*(i+1)+1, 2*(num_repeat+1), 2*(i+1))

    for i in range(num_repeat):
        code += """\
    asm volatile("v_mov_b32 v0, s0\\n");
    asm volatile("v_mov_b32 v1, s1\\n");
    asm volatile("v_mov_b32 v2, s{}\\n");
    asm volatile("v_mov_b32 v3, s{}\\n");
    asm volatile("flat_store_dwordx2 v[0:1], v[2:3]\\n");
    asm volatile("s_add_u32 s0, s0, 8\\n");
    asm volatile("s_addc_u32 s1, s1, 0\\n");
""".format(2*(i+1), 2*(i+1)+1)

    # 2*num_repeat+2 sregs, 4 vregs used
    code += """\
    // to ensure register usage
    uint32_t sdummy[{}];
    for(int i=0;i<{};i++) asm volatile("s_mov_b32 %0, 0\\n":"=s"(sdummy[i]));
    for(int i=0;i<{};i++) asm volatile("s_mov_b32 s0, %0\\n"::"s"(sdummy[i]));
    uint32_t vdummy[{}];
    for(int i=0;i<{};i++) asm volatile("v_mov_b32 %0, 0\\n":"=v"(vdummy[i]));
    for(int i=0;i<{};i++) asm volatile("v_mov_b32 v0, %0\\n"::"v"(vdummy[i]));
}}
""".format(2*num_repeat+2,2*num_repeat+2,2*num_repeat+2, 4,4,4)
    code += "}\n"
    return code

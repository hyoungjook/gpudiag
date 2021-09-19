import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from define import Feature
import config_ckpt as ckpt

def verify_constraint(result_values, proj_path):
    return True

def generate_nvidia(result_values):
    limit_shmem = int(result_values[Feature.limit_sharedmem_per_block][0])
    l1i_line_size = int(result_values[Feature.l1i_linesize][0])
    max_hierarchy_size = ckpt.values[ckpt.CKPT.max_icache_investigate_size_KiB]
    num_blocks = max_hierarchy_size * 1024 // l1i_line_size
    if num_blocks * 4 > limit_shmem:
        print("Shared memory is too small to run the icache_hierarchy test.")
        exit(1)
    num_inst_per_block = l1i_line_size // 8
    if num_inst_per_block < 4:
        print("L1I line size is too small to run the icache_hierarchy test.")
        exit(1)
    code = """\
__global__ void measure_icache(uint64_t *result) {{
    asm volatile(".reg .pred rev, psave, pnop;\\n");
    asm volatile(".reg .u32 shm_addr, idx, data;\\n");
    asm volatile(".reg .u64 out_addr;\\n");
    asm volatile(".shared .align 4 .b8 shm[{}];\\n");

    asm volatile("setp.ne.u32 pnop, 0, 0;\\n");
    asm volatile("setp.eq.u32 rev, 0, 0;\\n");
    asm volatile("mov.u32 shm_addr, shm;\\n");
    asm volatile("bra.uni MIC_BLK_{};\\n");

    asm volatile("MIC_BLK_0:\\n");
    asm volatile("mov.u32 data, %clock;\\n");
    asm volatile("setp.eq.u32 rev, 0, 1;\\n");
    asm volatile("st.shared.u32 [shm_addr], data;\\n");
    asm volatile("@rev bra MIC_BLK_0;\\n");
""".format(4*num_blocks, num_blocks-1)
    for i in range(num_inst_per_block-4):
        code += "\tasm volatile(\"@pnop mov.u32 data, 0;\\n\");\n"
    code += "\n"

    code += "\tasm volatile(\n";
    for n in range(num_blocks-1):
        code += """\
        "MIC_BLK_{}:\\n"
        "@!rev mov.u32 data, %clock;\\n"
        "@!rev setp.eq.u32 pnop, 0, 1;\\n"
        "@!rev st.shared.u32 [shm_addr+{}], data;\\n"
        "@rev bra MIC_BLK_{};\\n"
""".format(n+1, 4*(n+1), n)
        for i in range(num_inst_per_block-4):
            code += "\t\t\"@pnop mov.u32 data, 0;\\n\"\n"
    code += "\t);\n"

    code += """\
    asm volatile("mov.u32 idx, 0;\\n");
    asm volatile("mov.u64 out_addr, %0;\\n"::"l"(result));
    asm volatile("MIC_WRITE_BEGIN:\\n");
    asm volatile("ld.shared.u32 data, [shm_addr];\\n");
    asm volatile("st.global.u32 [out_addr], data;\\n");
    asm volatile("add.u32 shm_addr, shm_addr, 4;\\n");
    asm volatile("add.u64 out_addr, out_addr, 8;\\n");
    asm volatile("add.u32 idx, idx, 1;\\n");
    asm volatile("setp.lt.u32 psave, idx, {};\\n");
    asm volatile("@psave bra MIC_WRITE_BEGIN;\\n");
}}
""".format(num_blocks)

    return code


def generate_amd(result_values):
    limit_shmem = int(result_values[Feature.limit_sharedmem_per_block][0])
    l1i_line_size = int(result_values[Feature.l1i_linesize][0])
    max_hierarchy_size = ckpt.values[ckpt.CKPT.max_icache_investigate_size_KiB]
    num_blocks = max_hierarchy_size * 1024 // l1i_line_size
    if num_blocks * 8 > limit_shmem:
        print("Shared memory is too small to run the icache_hierarchy test.")
        exit(1)
    num_inst_per_block = l1i_line_size // 4
    if num_inst_per_block < 8:
        print("L1I line size is too small to run the icache_hierarchy test.")
        exit(1)
    code = """\
__global__ void measure_icache(uint64_t *result) {{
    asm volatile("s_mov_b32 m0, -1\\n");
    // s[0:1]: addr of result
    asm volatile("s_load_dwordx2 s[0:1], s[4:5], 0x0\\n");
    asm volatile("s_waitcnt lgkmcnt(0)\\n");
    // v0=0, used for ds_write
    asm volatile("v_mov_b32 v0, 0\\n");
    // s[2:3], s[4:5] : used for saveexec
    asm volatile("s_mov_b64 s[2:3], 0\\n");
    asm volatile("s_and_saveexec_b64 s[4:5], s[2:3]\\n");
    // now, s[4:5]=0xFF... s[2:3]=0, exec=0
    asm volatile("s_branch MIC_BLK_{}\\n");

    // s[6:7], v[1:2] : clock value buffer
    asm volatile("MIC_BLK_0:\\n");
    asm volatile("s_memtime s[6:7]\\n");                    // 8B
    asm volatile("s_or_saveexec_b64 s[2:3], s[4:5]\\n");    // 4B
    // now, s[2:3]=0, s[4:5]=0xFF.., exec=0xFF..
    asm volatile("v_mov_b32 v1, s6\\n");                    // 4B
    asm volatile("v_mov_b32 v2, s7\\n");                    // 4B
    asm volatile("s_waitcnt lgkmcnt(0)\\n");                // 4B
    asm volatile("ds_write_b64 v0, v[1:2], offset:0\\n");   // 8B
""".format(num_blocks-1)
    for i in range(num_inst_per_block-8):
        code += "\tasm volatile(\"s_nop 0\\n\");\n"
    code += "\n"

    code += "\tasm volatile(\n"
    for n in range(num_blocks-1):
        code += """\
        "MIC_BLK_{}:\\n"
        "s_memtime s[6:7]\\n"                   // 8B
        "s_cbranch_execz MIC_BLK_{}\\n"         // 4B
        "v_mov_b32 v1, s6\\n"                   // 4B
        "v_mov_b32 v2, s7\\n"                   // 4B
        "s_waitcnt lgkmcnt(0)\\n"               // 4B
        "ds_write_b64 v0, v[1:2] offset:{}\\n"  // 8B
""".format(n+1, n, 8*(n+1))
        for i in range(num_inst_per_block-8):
            code += "\t\t\"s_nop 0\\n\"\n"
    code += "\t);\n"

    code += """\
    // v[0:1], s[0:1] : addr of result, v4, s4: addr of shmem
    // v[2:3] : data buffer
    asm volatile("s_mov_b32 s4, 0\\n");
    asm volatile("MIC_WRITE_BEGIN:\\n");
    asm volatile("s_waitcnt lgkmcnt(0)\\n");
    asm volatile("v_mov_b32 v0, s0\\n");
    asm volatile("v_mov_b32 v1, s1\\n");
    asm volatile("v_mov_b32 v4, s4\\n");
    asm volatile("ds_read_b64 v[2:3], v4\\n");
    asm volatile("s_waitcnt lgkmcnt(0)\\n");
    asm volatile("s_waitcnt vmcnt(0)\\n");
    asm volatile("flat_store_dwordx2 v[0:1], v[2:3]\\n");
    asm volatile("s_add_u32 s0, s0, 8\\n");
    asm volatile("s_addc_u32 s1, s1, 0\\n");
    asm volatile("s_add_u32 s4, s4, 8\\n");
    asm volatile("v_cmp_gt_u32 vcc, {}, v4\\n");
    asm volatile("s_cbranch_vccnz MIC_WRITE_BEGIN\\n");
    asm volatile("s_waitcnt vmcnt(0)\\n");
""".format(8*(num_blocks-1))
    # 8 sregs, 5 vregs, 8*num_blocks B shm used
    code += """\
    // to ensure resource usage
    uint32_t sdummy[{}];
    for(int i=0;i<{};i++) asm volatile("s_mov_b32 %0, 0\\n":"=s"(sdummy[i]));
    for(int i=0;i<{};i++) asm volatile("s_mov_b32 s0, %0\\n"::"s"(sdummy[i]));
    uint32_t vdummy[{}];
    for(int i=0;i<{};i++) asm volatile("v_mov_b32 %0, 0\\n":"=v"(vdummy[i]));
    for(int i=0;i<{};i++) asm volatile("v_mov_b32 v0, %0\\n"::"v"(vdummy[i]));
    __shared__ uint32_t shmdummy[{}];
    if (sdummy[0] != 0) {{
        for(int i=0;i<{};i++) shmdummy[i]=(uint32_t)clock();
        for(int i=0;i<{};i++) asm volatile("v_mov_b32 v0, %0\\n"::"r"(shmdummy[i]));
    }}
}}
""".format(8,8,8, 5,5,5, 2*num_blocks,2*num_blocks,2*num_blocks)

    return code
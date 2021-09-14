import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from define import Feature
import config_ckpt as ckpt
import reused_codes

def generate_nvidia(result_values):
    # measure_br_time
    code = ""
    code += reused_codes.measure_time_code(100, 5, "br", "",\
        lambda i: "asm volatile(\"MT_BR_{}: bra MT_BR_{};\\n\" \"bar.sync 0;\\n\");\n".format(i, i+1),\
        "asm volatile(\"MT_BR_{}:\\n\");\n".format(100))

    # measure_num_mp
    num_mp_repeat = 100
    num_mp_repeat_for = 2
    code += """\
__global__ void measure_num_mp(uint32_t *sync, uint32_t G, uint64_t *result,
        uint64_t timeout, uint32_t *is_timeout, uint64_t *totalclk){{
    // assumes *sync=0, *is_timeout=0 initially
    // if timeout=0, no timeout.
    asm volatile(".reg .pred mnm_thread0, mnm_pwhile, mnm_prep, mnm_timed, mnm_chktime;\\n");
    asm volatile(".reg .u32 mnm_tid, mnm_bid, mnm_sclk, mnm_eclk;\\n");
    asm volatile(".reg .u32 mnm_G, mnm_val, mnm_rep, mnm_timeout;\\n");
    asm volatile(".reg .u64 mnm_addr, mnm_bid64;\\n");

    // set variables
    asm volatile("mov.u32 mnm_tid, %tid.x;\\n");
    asm volatile("setp.eq.u32 mnm_thread0, mnm_tid, 0;\\n");
    asm volatile("@mnm_thread0 mov.u32 mnm_bid, %ctaid.x;\\n");
    asm volatile("@mnm_thread0 shl.b32 mnm_bid, mnm_bid, 3;\\n");
    asm volatile("@mnm_thread0 cvt.u64.u32 mnm_bid64, mnm_bid;\\n");
    asm volatile("mov.u32 mnm_rep, 0;\\n");
    asm volatile("mov.u32 mnm_G, %0;\\n"::"r"(G));
    asm volatile("cvt.u32.u64 mnm_timeout, %0;\\n"::"l"(timeout));
    asm volatile("setp.ne.u32 mnm_chktime, mnm_timeout, 0;\\n");

    // if timeout already, exit immediately
    asm volatile("ld.global.u32 mnm_val, [%0];\\n"::"l"(is_timeout));
    asm volatile("setp.eq.u32 mnm_timed, mnm_val, 1;\\n");
    asm volatile("@mnm_timed ret;\\n");

    // sync1: determine timeout (with timeout)
    asm volatile("bar.sync 0;\\n");
    asm volatile("mov.u32 mnm_sclk, %clock;\\n");
    asm volatile("add.u32 mnm_timeout, mnm_sclk, mnm_timeout;\\n");
    asm volatile("@!mnm_thread0 bra MNM_WAIT1;\\n");
    asm volatile("atom.global.add.u32 mnm_eclk, [%0], 1;\\n"::"l"(sync));

    asm volatile("MNM_WAIT1:\\n");
    asm volatile("mov.u32 mnm_eclk, %clock;\\n");
    asm volatile("@!mnm_chktime mov.u32 mnm_eclk, 0;\\n");
    asm volatile("setp.lt.u32 mnm_timed, mnm_timeout, mnm_eclk;\\n");
    asm volatile("@mnm_timed st.global.u32 [%0], 1;\\n"::"l"(is_timeout));
    asm volatile("@mnm_timed ret;\\n"); // timeout end
    asm volatile("ld.global.u32 mnm_val, [%0];\\n"::"l"(sync));
    asm volatile("setp.lt.u32 mnm_pwhile, mnm_val, mnm_G;\\n");
    asm volatile("@mnm_pwhile bra MNM_WAIT1;\\n");

    // sync1 passed: save timeout value & set mnm_G = 2*G (for sync2 use)
    asm volatile("mov.u32 mnm_eclk, %clock;\\n");
    asm volatile("@mnm_thread0 sub.u32 mnm_val, mnm_eclk, mnm_sclk;\\n");
    asm volatile("@mnm_thread0 mov.u64 mnm_addr, %0;\\n"::"l"(totalclk));
    asm volatile("@mnm_thread0 add.u64 mnm_addr, mnm_addr, mnm_bid64;\\n");
    asm volatile("@mnm_thread0 st.global.u32 [mnm_addr], mnm_val;\\n");
    asm volatile("shl.b32 mnm_G, mnm_G, 1;\\n");
    
    // warmup icache
    asm volatile("mov.u32 mnm_rep, {};\\n"); 
    // mnm_rep=repeat_for: warmup, mnm_rep<=repeat_for-1: repeating
    asm volatile("bra MNM_SKIP_ON_WARMUP;\\n");
    asm volatile("MNM_WARMUP_ICACHE_DONE:\\n");
    asm volatile("mov.u32 mnm_rep, 0;\\n");

        // sync2: real sync before measurement, no timeout
        asm volatile("bar.sync 0;\\n");
        asm volatile("@!mnm_thread0 bra MNM_WAIT2;\\n");
        asm volatile("atom.global.add.u32 mnm_eclk, [%0], 1;\\n"::"l"(sync));
        asm volatile("MNM_WAIT2:\\n");
        asm volatile("ld.global.u32 mnm_val, [%0];\\n"::"l"(sync));
        asm volatile("setp.lt.u32 mnm_pwhile, mnm_val, mnm_G;\\n");
        asm volatile("@mnm_pwhile bra MNM_WAIT2;\\n");

        // begin measurement
        asm volatile("MNM_SKIP_ON_WARMUP:\\n");
        asm volatile("@mnm_thread0 mov.u32 mnm_sclk, %clock;\\n");
        asm volatile("MNM_REPEAT_BEGIN:\\n");
""".format(num_mp_repeat_for)
    
    for i in range(num_mp_repeat):
        code += "\t\tasm volatile(\"MNM_BR_{}: bra MNM_BR_{};\\n\" \"bar.sync 0;\\n\");\n".format(i, i+1)
    code += "\t\tasm volatile(\"MNM_BR_{}:\\n\");\n".format(num_mp_repeat)

    code += """\
        asm volatile("setp.lt.u32 mnm_prep, mnm_rep, {};\\n");
        asm volatile("add.u32 mnm_rep, mnm_rep, 1;\\n");
        asm volatile("@mnm_prep bra MNM_REPEAT_BEGIN;\\n");

        asm volatile("bar.sync 0;\\n");
        asm volatile("@mnm_thread0 mov.u32 mnm_eclk, %clock;\\n");

    asm volatile("setp.eq.u32 mnm_prep, mnm_rep, {};\\n");
    asm volatile("@mnm_prep bra MNM_WARMUP_ICACHE_DONE;\\n");

    // save the result
    asm volatile("@mnm_thread0 sub.u32 mnm_val, mnm_eclk, mnm_sclk;\\n");
    asm volatile("@mnm_thread0 mov.u64 mnm_addr, %0;\\n"::"l"(result));
    asm volatile("@mnm_thread0 add.u64 mnm_addr, mnm_addr, mnm_bid64;\\n");
    asm volatile("@mnm_thread0 st.global.u32 [mnm_addr], mnm_val;\\n");
""".format(num_mp_repeat_for-1, num_mp_repeat_for+1)
    code += "}\n"

    # measure warpstatebuffer
    code += """\
__global__ void measure_warpstatebuffer(uint32_t *sync, uint32_t G, uint64_t timeout,
        uint32_t *is_timeout, uint64_t *totalclk) {
    
    asm volatile(".reg .pred mb_thread0, mb_pwhile, mb_timed, mb_chktime;\\n");
    asm volatile(".reg .u32 mb_val, mb_sclk, mb_eclk, mb_to32;\\n");
    asm volatile(".reg .u64 mb_addr, mb_bid64;\\n");

    // init values
    asm volatile("mov.u32 mb_val, %tid.x;\\n");
    asm volatile("setp.eq.u32 mb_thread0, mb_val, 0;\\n");
    asm volatile("@mb_thread0 mov.u32 mb_val, %ctaid.x;\\n");
    asm volatile("@mb_thread0 shl.b32 mb_val, mb_val, 3;\\n");
    asm volatile("@mb_thread0 cvt.u64.u32 mb_bid64, mb_val;\\n");
    asm volatile("cvt.u32.u64 mb_to32, %0;\\n"::"l"(timeout));
    asm volatile("setp.ne.u32 mb_chktime, mb_to32, 0;\\n");

    // synchronize with timeout
    asm volatile("bar.sync 0;\\n");
    asm volatile("mov.u32 mb_sclk, %clock;\\n");
    asm volatile("add.u32 mb_to32, mb_sclk, mb_to32;\\n");
    asm volatile("@!mb_thread0 bra MB_WAIT;\\n");
    asm volatile("atom.global.add.u32 mb_eclk, [%0], 1;\\n"::"l"(sync));
    asm volatile("MB_WAIT:\\n");

    asm volatile("mov.u32 mb_eclk, %clock;\\n");
    asm volatile("@!mb_chktime mov.u32 mb_eclk, 0;\\n");
    asm volatile("setp.lt.u32 mb_timed, mb_to32, mb_eclk;\\n");
    asm volatile("@mb_timed st.global.u32 [%0], 1;\\n"::"l"(is_timeout));
    asm volatile("@mb_timed ret;\\n");
    asm volatile("ld.global.u32 mb_val, [%0];\\n"::"l"(sync));
    asm volatile("setp.lt.u32 mb_pwhile, mb_val, %0;\\n"::"r"(G));
    asm volatile("@mb_pwhile bra MB_WAIT;\\n");

    // sync succeed, store timeout reference value
    asm volatile("mov.u32 mb_eclk, %clock;\\n");
    asm volatile("@mb_thread0 mov.u64 mb_addr, %0;\\n"::"l"(totalclk));
    asm volatile("@mb_thread0 add.u64 mb_addr, mb_addr, mb_bid64;\\n");
    asm volatile("@mb_thread0 sub.u32 mb_val, mb_eclk, mb_sclk;\\n");
    asm volatile("@mb_thread0 st.global.u32 [mb_addr], mb_val;\\n");
}
"""

    # measure shmem
    limit_shmem = int(result_values[Feature.limit_sharedmem_per_block][0])
    shmem_unit = ckpt.values[ckpt.CKPT.shared_memory_test_granularity]
    code += ""
    for s in range(shmem_unit, limit_shmem+1, shmem_unit):
        code += """\
__global__ void measure_shmem_{}(uint32_t *sync, uint32_t G, uint64_t timeout,
        uint32_t *is_timeout, uint64_t *totalclk) {{
    asm volatile(".reg .pred mb_thread0, mb_pwhile, mb_timed, mb_chktime;\\n");
    asm volatile(".reg .u32 mb_val, mb_sclk, mb_eclk, mb_to32;\\n");
    asm volatile(".reg .u64 mb_addr, mb_bid64;\\n");
    asm volatile(".shared .align 1 .b8 ms{}_shm[{}];\\n");

    asm volatile("mov.u32 mb_val, %tid.x;\\n");
    asm volatile("setp.eq.u32 mb_thread0, mb_val, 0;\\n");
    asm volatile("@mb_thread0 mov.u32 mb_val, %ctaid.x;\\n");
    asm volatile("@mb_thread0 shl.b32 mb_val, mb_val, 3;\\n");
    asm volatile("@mb_thread0 cvt.u64.u32 mb_bid64, mb_val;\\n");
    asm volatile("cvt.u32.u64 mb_to32, %0;\\n"::"l"(timeout));
    asm volatile("setp.ne.u32 mb_chktime, mb_to32, 0;\\n");
    
    asm volatile("bar.sync 0;\\n");
    asm volatile("mov.u32 mb_sclk, %clock;\\n");
    asm volatile("add.u32 mb_to32, mb_sclk, mb_to32;\\n");
    asm volatile("@!mb_thread0 bra MB_WAIT;\\n");
    asm volatile("atom.global.add.u32 mb_eclk, [%0], 1;\\n"::"l"(sync));
    asm volatile("MB_WAIT:\\n");

    asm volatile("mov.u32 mb_eclk, %clock;\\n");
    asm volatile("@!mb_chktime mov.u32 mb_eclk, 0;\\n");
    asm volatile("setp.lt.u32 mb_timed, mb_to32, mb_eclk;\\n");
    asm volatile("@mb_timed st.global.u32 [%0], 1;\\n"::"l"(is_timeout));
    asm volatile("@mb_timed ret;\\n");
    asm volatile("ld.global.u32 mb_val, [%0];\\n"::"l"(sync));
    asm volatile("setp.lt.u32 mb_pwhile, mb_val, %0;\\n"::"r"(G));
    asm volatile("@mb_pwhile bra MB_WAIT;\\n");
    
    asm volatile("mov.u32 mb_eclk, %clock;\\n");
    asm volatile("@mb_thread0 mov.u64 mb_addr, %0;\\n"::"l"(totalclk));
    asm volatile("@mb_thread0 add.u64 mb_addr, mb_addr, mb_bid64;\\n");
    asm volatile("@mb_thread0 sub.u32 mb_val, mb_eclk, mb_sclk;\\n");
    asm volatile("@mb_thread0 st.global.u32 [mb_addr], mb_val;\\n");
}}
""".format(s, s, s)

    code += """\
void (*measure_shmem[{}])(uint32_t*, uint32_t, uint64_t, uint32_t*, uint64_t*) = {{
""".format(limit_shmem // shmem_unit)
    for s in range(shmem_unit, limit_shmem+1, shmem_unit):
        code += "measure_shmem_{},\n".format(s)
    code += "};\n"

    return code

def generate_amd(result_values):
    return ""
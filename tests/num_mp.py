from .test_template import test_template
from .define import Test
import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

def measure_num_mp_code(repeat, repeat_for, inicode, repcode, fincode):
    code = """\
__gdkernel void measure_num_mp(__gdbufarg uint32_t *sync, uint32_t G, __gdbufarg uint64_t *result,
        uint64_t timeout, __gdbufarg uint32_t *is_timeout, __gdbufarg uint64_t *totalclk){
    // assumes *sync=0, *is_timeout=0 initially
    // if timeout=0, no timeout.
    uint64_t sclk, eclk, toclk;
    bool thread0 = (GDThreadIdx == 0);
    uint32_t G2 = 2 * G, blkid = GDBlockIdx;
    // if timeout already, exit immediately
    if (*is_timeout == 1) return;

    // sync1: determine timeout (with timeout)
    bool chktime = (timeout != 0);
    volatile uint32_t *chksync = sync;
    GDsyncthreads();
    sclk = GDclock();
    toclk = sclk + timeout;
    if (thread0) GDatomicAdd(sync, 1);
    while(*chksync < G) {
        if (chktime && GDclock() > toclk) {
            *is_timeout = 1; return;
        }
    }
    eclk = GDclock();
    if (thread0) totalclk[blkid] = eclk - sclk;
"""
    code += inicode
    code += f"""\
#pragma unroll 1
    for (int i=0; i<{repeat_for+1}; i++) {{ // warmup icache + repeat_for
        if (i==1) {{
            GDsyncthreads();
            // sync2: real sync before measurement, no timeout
            if (thread0) GDatomicAdd(sync, 1);
            while(*chksync < G2);
            // begin measurement
            GDsyncthreads();
            sclk = GDclock();
        }}\n"""
    for i in range(repeat):
        code += repcode(i)
    code += fincode
    code += """\
    }
    GDsyncthreads();
    eclk = GDclock();
    // save the result
    if (thread0) result[blkid] = eclk - sclk;
}\n"""
    return code

class num_mp (test_template):
    test_enum = Test.num_mp
    
    def generate_kernel(self):
        if self.conf.manufacturer() == "nvidia":
            repcode = lambda i: "asm volatile(\"MT_BR_{}: bra MT_BR_{};\\n\");\n"\
                .format(i, i+1)
            fincode = "asm volatile(\"MT_BR_{}:\\n\");\n".format(100)
        else:
            repcode = lambda i: "asm volatile(\"MNM_BR_{}: s_branch MNM_BR_{}\\n\");\n"\
                .format(i, i+1)
            fincode = "asm volatile(\"MNM_BR_{}:\\n\");\n".format(100)
        return measure_num_mp_code(100, 2, "", repcode, fincode)
from .test_template import test_template
from .define import Test

def measure_warpstatebuffer_code():
    code = """\
__gdkernel void measure_warpstatebuffer(__gdbufarg uint32_t *sync, uint32_t G,
        uint64_t timeout, __gdbufarg uint32_t *is_timeout, __gdbufarg uint64_t *totalclk) {
    uint64_t sclk, eclk, toclk;
    bool thread0 = (GDThreadIdx == 0);
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
    if (thread0) totalclk[GDBlockIdx] = eclk - sclk;
}\n"""
    return code

class warpstate_buffer (test_template):
    test_enum = Test.warpstate_buffer
    
    def generate_kernel(self):
        return measure_warpstatebuffer_code()
from .test_template import test_template
from .define import Test

def measure_warpstatebuffer_code():
    code = """\
__global__ void measure_warpstatebuffer(uint32_t *sync, uint32_t G,
        uint64_t timeout, uint32_t *is_timeout, uint64_t *totalclk) {
    uint64_t sclk, eclk, toclk;
    bool thread0 = (GDThreadIdx == 0);
    bool chktime = (timeout != 0);
    volatile uint32_t *chksync = sync;
    __syncthreads();
    sclk = clock();
    toclk = sclk + timeout;
    if (thread0) atomicAdd(sync, 1);
    while(*chksync < G) {
        if (chktime && clock() > toclk) {
            *is_timeout = 1; return;
        } 
    }
    eclk = clock();
    if (thread0) totalclk[GDBlockIdx] = eclk - sclk;
}\n"""
    return code

class warpstate_buffer (test_template):
    test_enum = Test.warpstate_buffer
    
    def generate_kernel(self):
        return measure_warpstatebuffer_code()
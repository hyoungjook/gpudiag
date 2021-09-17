def measure_latency_code(repeat, name, inicode, repcode, fincode):
    code = """\
__global__ void measure_latency_{}(uint64_t *result){{
    uint64_t sclk, eclk;
""".format(name)
    code += inicode
    code += """\
#pragma unroll 1
    for (int i=0; i<2; i++) { // icache warmup
        __syncthreads();
        sclk = clock();
"""
    for i in range(repeat):
        code += repcode(i)
    code += fincode
    code += """\
        __syncthreads();
        eclk = clock();
    }
    if (hipThreadIdx_x == 0) *result = eclk - sclk;
}
"""
    return code

def measure_width_code(repeat, repeat_for, name, inicode, repcode, fincode):
    code = """\
__global__ void measure_width_{}(uint64_t *result) {{
    uint64_t sclk, eclk;
""".format(name, repeat_for)
    code += inicode
    code += """\
    int repeats = 1;
#pragma unroll 1
    for (int i=0; i<2; i++) {{
        if (i==1) {{ // icache warmup done
            repeats = {};
            __syncthreads();
            sclk = clock();
        }}
#pragma unroll 1
        for (int j=0; j<repeats; j++) {{
""".format(repeat_for)
    for i in range(repeat):
        code += repcode(i)
    code += fincode
    code += """\
        }}
        __syncthreads();
        eclk = clock();
    }}
    if (hipThreadIdx_x == 0) *result = eclk - sclk;
}}
""".format(repeat_for-1, repeat_for+1)
    return code

def measure_num_mp_code(repeat, repeat_for, inicode, repcode, fincode):
    code = """\
__global__ void measure_num_mp(uint32_t *sync, uint32_t G, uint64_t *result,
        uint64_t timeout, uint32_t *is_timeout, uint64_t *totalclk){
    // assumes *sync=0, *is_timeout=0 initially
    // if timeout=0, no timeout.
    uint64_t sclk, eclk, toclk;
    bool thread0 = (hipThreadIdx_x == 0);
    uint32_t G2 = 2 * G, blkid = hipBlockIdx_x;
    // if timeout already, exit immediately
    if (*is_timeout == 1) return;

    // sync1: determine timeout (with timeout)
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
    if (thread0) totalclk[blkid] = eclk - sclk;
"""
    code += inicode
    code += """\
#pragma unroll 1
    for (int i=0; i<{}; i++) {{ // warmup icache + repeat_for
        if (i==1) {{
            __syncthreads();
            // sync2: real sync before measurement, no timeout
            if (thread0) atomicAdd(sync, 1);
            while(*chksync < G2);
            // begin measurement
            __syncthreads();
            sclk = clock();
        }}
""".format(repeat_for+1)
    for i in range(repeat):
        code += repcode(i)
    code += fincode
    code += """\
    }
    __syncthreads();
    eclk = clock();
    // save the result
    if (thread0) result[blkid] = eclk - sclk;
}
"""
    return code

def measure_warpstatebuffer_code():
    code = """\
__global__ void measure_warpstatebuffer(uint32_t *sync, uint32_t G,
        uint64_t timeout, uint32_t *is_timeout, uint64_t *totalclk) {
    uint64_t sclk, eclk, toclk;
    bool thread0 = (hipThreadIdx_x == 0);
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
    if (thread0) totalclk[hipBlockIdx_x] = eclk - sclk;
}
"""
    return code

def measure_shmem_code(limit_shmem, shmem_unit):
    code = ""
    for s in range(shmem_unit, limit_shmem+1, shmem_unit):
        code += """\
__global__ void measure_shmem_{}(uint32_t *sync, uint32_t G,
        uint64_t timeout, uint32_t *is_timeout, uint64_t *totalclk) {{
    uint64_t sclk, eclk, toclk;
    bool thread0 = (hipThreadIdx_x == 0);
    bool chktime = (timeout != 0);
    volatile uint32_t *chksync = sync;
    __syncthreads();
    sclk = clock();
    toclk = sclk + timeout;
    if (thread0) atomicAdd(sync, 1);
    while(*chksync < G) {{
        if (chktime && clock() > toclk) {{
            *is_timeout = 1; return;
        }}
    }}
    eclk = clock();
    if (thread0) totalclk[hipBlockIdx_x] = eclk - sclk;
    // ensure shmem usage
    if (G > 0) return; // do not execute, just compile
    __shared__ uint8_t arr[{}];
    uint8_t *fakeptr = (uint8_t*)sync;
    for (int i=0; i<{}; i++) arr[i] = (uint8_t)clock();
    for (int i=0; i<{}; i++) fakeptr[i] = arr[i];
}}
""".format(s, s, s, s)
    code += """\
void (*measure_shmem[{}])(uint32_t*, uint32_t, uint64_t, uint32_t*, uint64_t*) = {{
""".format(limit_shmem // shmem_unit)
    for s in range(shmem_unit, limit_shmem+1, shmem_unit):
        code += "measure_shmem_{},\n".format(s)
    code += "};\n"

    return code
#ifndef MINIMUM_FOR_COMPILE_TESTS
#include <iostream>
#include <fstream>
#include <string>
#include <assert.h>
#endif
#include <stdint.h>
#include <stdio.h>

#ifndef MINIMUM_FOR_COMPILE_TESTS
// ========== Common tools ==========
#define MAX(a, b) ((a)>(b)?(a):(b))
#define MIN(a, b) ((a)<(b)?(a):(b))

#ifndef REPORT_FILE
#define REPORT_FILE ""
#endif

void write_init(const char *testname) {
    std::ofstream out(REPORT_FILE);
    out << "## ========== " << testname << " report ==========" << std::endl;
    out.close();
}

void write_line(const char *line) {
    std::ofstream out(REPORT_FILE, std::ios_base::app);
    out << line << std::endl;
    out.close();
}

template <typename T>
void write_value(const char *fname, T val) {
    std::ofstream out(REPORT_FILE, std::ios_base::app);
    out << fname << "=" << val << std::endl;
    out.close();
}

template <typename T>
void write_values(const char *fname, T *vals, int size) {
    assert(size > 0);
    if (size == 1) {
        write_value(fname, vals[0]);
        return;
    }
    std::ofstream out(REPORT_FILE, std::ios_base::app);
    out << fname << "=[";
    for (int i=0; i<size-1; i++) {
        out << vals[i] << ", ";
    }
    out << vals[size-1] << "]" << std::endl;
    out.close();
}

// no ':' allowed in names, but ' ' is allowed.
// if ylabel starts with '--', points will be connected with line.
template <typename Tx, typename Ty>
void write_graph_data(const char *title, int n, const char *xlabel, Tx x0, Tx dx,
        const char *ylabel, Ty *ys) {
    assert(n > 0);
    std::ofstream out(REPORT_FILE, std::ios_base::app);
    out << "@" << title << ":" << n << ":";
    out << xlabel << ":" << x0 << ":" << dx << ":";
    out << ylabel << ":";
    for (int i=0; i<n-1; i++) out << ys[i] << ",";
    out << ys[n-1] << std::endl;
    out.close();
}

template <typename Tx, typename Ty>
void write_graph_data_xs(const char *title, int n, const char *xlabel, Tx *xs,
        const char *ylabel, Ty *ys) {
    assert(n > 0);
    std::ofstream out(REPORT_FILE, std::ios_base::app);
    out << "@" << title << ":" << n << ":";
    out << xlabel << ":p:";
    for (int i=0; i<n-1; i++) out << xs[i] << ",";
    out << xs[n-1] << ":";
    out << ylabel << ":";
    for (int i=0; i<n-1; i++) out << ys[i] << ",";
    out << ys[n-1] << std::endl;
    out.close();
}

template <typename Tx, typename Ty, typename Tls, typename Tly>
void write_graph_data_with_line(const char *title, int n,
        const char *xlabel, Tx x0, Tx dx, const char *ylabel, Ty *ys,
        Tls slope, Tly y_intercept, const char *linelabel) {
    assert(n > 0);
    std::ofstream out(REPORT_FILE, std::ios_base::app);
    out << "@" << title << ":" << n << ":";
    out << xlabel << ":" << x0 << ":" << dx << ":";
    out << ylabel << ":";
    for (int i=0; i<n-1; i++) out << ys[i] << ",";
    out << ys[n-1] << ":";
    out << slope << "," << y_intercept << ",";
    out << linelabel << std::endl;
    out.close();
}

template <typename Tx, typename Ty, typename Tls, typename Tly>
void write_graph_data_xs_with_line(const char *title, int n,
        const char *xlabel, Tx *xs, const char *ylabel, Ty *ys,
        Tls slope, Tly y_intercept, const char *linelabel) {
    assert(n > 0);
    std::ofstream out(REPORT_FILE, std::ios_base::app);
    out << "@" << title << ":" << n << ":";
    out << xlabel << ":p:";
    for (int i=0; i<n-1; i++) out << xs[i] << ",";
    out << xs[n-1] << ":";
    out << ylabel << ":";
    for (int i=0; i<n-1; i++) out << ys[i] << ",";
    out << ys[n-1] << ":";
    out << slope << "," << y_intercept << ",";
    out << linelabel << std::endl;
    out.close();
}
#endif

// ========== Environment-specific interfaces ==========
#if defined(__NVCC__)       // NVCC

#include <cuda_runtime.h>

#ifndef MINIMUM_FOR_COMPILE_TESTS

#define GDInit()\
{cudaError_t err = cudaSetDevice(0);\
if (err != cudaSuccess) printf("SetDevice: %s\n", cudaGetErrorName(err));}

#define GDMalloc(p, size)   cudaMalloc(p, size)

#define GDFree(p)           cudaFree(p)

#define GDSynchronize()     cudaDeviceSynchronize()

#define GDMemcpyHToD(destp, srcp, size) \
cudaMemcpy(destp, srcp, size, cudaMemcpyHostToDevice)

#define GDMemcpyDToH(destp, srcp, size) \
cudaMemcpy(destp, srcp, size, cudaMemcpyDeviceToHost)

#define GDIsLastErrorSuccess() (cudaGetLastError() == cudaSuccess)

#endif

#define GDLaunchKernel(kern, gDim, bDim, dynshm, stream, ...) \
{printf("  Launched " #kern ": (G,B)=(%d, %d)\n",(int)gDim.x,(int)bDim.x);\
fflush(stdout); kern<<<gDim, bDim, dynshm>>>(__VA_ARGS__);}

#define GDThreadIdx (threadIdx.x)
#define GDBlockIdx (blockIdx.x)
#define __gdkernel __global__
#define __gdbufarg
#define __gdshmem __shared__
#define GDsyncthreads() __syncthreads()
#define GDatomicAdd(p, val) atomicAdd(p, val)
#define GDclock() clock()
#ifdef KERNEL_FILE
#include KERNEL_FILE
#endif

#elif defined(__HIPCC__)    // HIPCC

#include "hip/hip_runtime.h"

#ifndef MINIMUM_FOR_COMPILE_TESTS

#define GDInit()\
{hipError_t err = hipSetDevice(0);\
if (err != hipSuccess) printf("SetDevice: %s\n", hipGetErrorName(err));}

#define GDMalloc(p, size)   hipMalloc(p, size)

#define GDFree(p)           hipFree(p)

#define GDSynchronize()     hipStreamSynchronize(0)

#define GDMemcpyHToD(destp, srcp, size) \
hipMemcpy(destp, srcp, size, hipMemcpyHostToDevice)

#define GDMemcpyDToH(destp, srcp, size) \
hipMemcpy(destp, srcp, size, hipMemcpyDeviceToHost)

#define GDIsLastErrorSuccess() (hipGetLastError() == hipSuccess)

#endif

#define GDLaunchKernel(kern, gDim, bDim, dynshm, stream, ...) \
{printf("  Launched " #kern ": (G,B)=(%d, %d)\n",(int)gDim.x,(int)bDim.x);\
fflush(stdout); hipLaunchKernelGGL(kern, gDim, bDim, dynshm, stream, __VA_ARGS__);}

#define GDThreadIdx (hipThreadIdx_x)
#define GDBlockIdx (hipBlockIdx_x)
#define __gdkernel __global__
#define __gdbufarg
#define __gdshmem __shared__
#define GDsyncthreads() __syncthreads()
#define GDatomicAdd(p, val) atomicAdd(p, val)
#define GDclock() clock()
#ifdef KERNEL_FILE
#include KERNEL_FILE
#endif

#else                       // TRY OPENCL
#if defined(__APPLE__)
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#ifndef MINIMUM_FOR_COMPILE_TESTS

#define GDchk(err, func) \
if (err != CL_SUCCESS) {printf(#func ": error %d\n", err); exit(1);}

static cl_int GDlasterr;
static cl_device_id GDdevid;
static cl_context GDctx;
static cl_command_queue GDcmdq;
static cl_program GDprog;

#include <sstream>

void GDInit() {
    cl_platform_id pf;
    GDchk(clGetPlatformIDs(1, &pf, NULL), clGetPlatformIDs);
    GDchk(clGetDeviceIDs(pf, CL_DEVICE_TYPE_GPU, 1, &GDdevid, NULL), clGetDeviceIDs);
    GDctx = clCreateContext(NULL, 1, &GDdevid, NULL, NULL, &GDlasterr); GDchk(GDlasterr, clCreateContext);
    GDcmdq = clCreateCommandQueue(GDctx, GDdevid, 0, &GDlasterr); GDchk(GDlasterr, clCreateCommandQueue);
#ifdef KERNEL_FILE
    std::ifstream srcf(KERNEL_FILE);
    if(!srcf.is_open()) {printf("ifstream: failed to open\n"); exit(1);};
    std::stringstream srcbuf;
#if MANUFACTURER == 1
    srcbuf << "inline uint32_t GDclock() {\n"
        "uint32_t val;\n"
        "asm volatile(\"mov.u32 %0, %%clock;\\n\":\"=r\"(val));\n"
        "return val;\n"
    "}\n";
#else
    srcbuf << "inline uint64_t GDclock() {\n"
        "uint64_t val;\n"
        "asm volatile(\"s_memtime %0\\n\":\"=l\"(val));\n"
        "return val;\n"
    "}\n";
#endif
    srcbuf << srcf.rdbuf(); srcf.close();
    size_t srcl = srcbuf.str().length();
    char *srcp = (char *)malloc(srcl+1);
    strncpy(srcp, srcbuf.str().c_str(), srcl+1);
    GDprog = clCreateProgramWithSource(GDctx, 1, (const char**)&srcp, &srcl, &GDlasterr); GDchk(GDlasterr, clCreateProgramWithSource);
    const char *buildoptions =
    "-DGDThreadIdx=get_local_id(0) -DGDBlockIdx=get_group_id(0) "
    "-D__gdkernel=__kernel -D__gdbufarg=__global -D__gdshmem=__local "
    "-DGDsyncthreads()=barrier(CLK_LOCAL_MEM_FENCE|CLK_GLOBAL_MEM_FENCE) "
    "-DGDatomicAdd(p,val)=atomic_add(p,val) "
    "-Dint8_t=char -Duint8_t=unsigned\\ char -Dint32_t=int -Duint32_t=unsigned\\ int "
    "-Dint64_t=long\\ long -Duint64_t=unsigned\\ long\\ long "
    "-cl-opt-disable";
    GDlasterr = clBuildProgram(GDprog, 1, &GDdevid, buildoptions, NULL, NULL);
    if (GDlasterr != CL_SUCCESS) {
        size_t logl; clGetProgramBuildInfo(GDprog, GDdevid, CL_PROGRAM_BUILD_LOG, 0, NULL, &logl);
        char *logp = (char*)malloc(logl+1); clGetProgramBuildInfo(GDprog, GDdevid, CL_PROGRAM_BUILD_LOG, logl, logp, NULL);
        fprintf(stderr, "Build failed:\n%s\n", logp); free(logp); GDchk(GDlasterr, clBuildProgram);
    }
    free(srcp);
#endif
}

struct dim3 {
    unsigned long x;
    dim3(): x(1) {}
    dim3(unsigned long xx): x(xx) {}
};

template <typename T>
void GDMalloc(T **p, size_t size) {
    *p = (T *)clCreateBuffer(GDctx, CL_MEM_READ_WRITE, size, NULL, &GDlasterr);
}

#define GDFree(p) clReleaseMemObject((cl_mem)(p));

#define GDSynchronize() /*the subsequent GDMemcpyDToH will synchronize the kernel.*/

#define GDMemcpyHToD(destp, srcp, size) \
GDlasterr = clEnqueueWriteBuffer(GDcmdq, (cl_mem)(destp), CL_TRUE, 0, (size), (srcp), 0, NULL, NULL);

#define GDMemcpyDToH(destp, srcp, size) \
GDlasterr = clEnqueueReadBuffer(GDcmdq, (cl_mem)(srcp), CL_TRUE, 0, (size), (destp), 0, NULL, NULL);

#define GDIsLastErrorSuccess() (GDlasterr == CL_SUCCESS)

#define GDLaunchKernel(kern, gDim, bDim, dynshm, stream, ...) {\
cl_kernel kn;\
kn = clCreateKernel(GDprog, #kern, &GDlasterr); GDchk(GDlasterr, clCreateKernel);\
size_t lws[1] = {(size_t)bDim.x}; size_t gws[1] = {(size_t)(bDim.x * gDim.x)};\
GDsetKernelArg(kn, 0, __VA_ARGS__);\
GDlasterr = clEnqueueNDRangeKernel(GDcmdq, kn, 1, NULL, gws, lws, 0, NULL, NULL);\
}

#endif // MINIMUM_FOR_COMPILE_TESTS

#endif // OpenCL

// ========== Environment-specific kernel_limit properties ==========
#ifndef MINIMUM_FOR_COMPILE_TESTS
#if defined(__NVCC__)
void write_deviceprops() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    write_init("prop_values");
    write_value("0", prop.warpSize);
    write_value("1", prop.maxThreadsPerBlock);
    write_value("2", prop.maxGridSize[0]);
    write_value("3", prop.sharedMemPerBlock);
    write_value("4", 255);
    write_value("5", prop.regsPerBlock / prop.warpSize);
}
#elif defined(__HIPCC__)
void write_deviceprops() {
    hipDeviceProp_t prop;
    hipGetDeviceProperties(&prop, 0);
    write_init("prop_values");
    write_value("0", prop.warpSize);
    write_value("1", prop.maxThreadsPerBlock);
    write_value("2", prop.maxGridSize[0]);
    write_value("3", prop.sharedMemPerBlock);
#if MANUFACTURER == 1
    write_value("4", 255);
    write_value("5", prop.regsPerBlock / prop.warpSize);
#else
    uint32_t buf[2]; buf[1] = 256;
    if (prop.gcnArch < 1000) buf[0] = 102;
    else buf[0] = 106;
    write_values("4", buf, 2);
    buf[0] = buf[1] = prop.regsPerBlock / prop.warpSize;
    write_values("5", buf, 2);
#endif
}
#else // opencl
#define GDcldevinfo(infotype, valtype, ptr) \
GDchk(clGetDeviceInfo(GDdevid, infotype, sizeof(valtype), ptr, NULL), clGetDeviceInfo)
void write_deviceprops() {
    // Retrievable data
    size_t LTpB; cl_ulong LSpB;
    GDcldevinfo(CL_DEVICE_MAX_WORK_GROUP_SIZE, size_t, &LTpB);
    GDcldevinfo(CL_DEVICE_LOCAL_MEM_SIZE, cl_ulong, &LSpB);
    // Manually specify others
    size_t warpSize, LTpG, LRpB_expected;
#if MANUFACTURER == 1
    warpSize = 32; LTpG = 1073742824; // 2^30
    LRpB_expected = 65536;
#else
    warpSize = 64; LTpG = 2147483647; // 2^31-1
    LRpB_expected = 65536;
#endif
    // write
    write_init("prop_values");
    write_value("0", warpSize);
    write_value("1", LTpB);
    write_value("2", LTpG);
    write_value("3", LSpB);
#if MANUFACTURER == 1
    write_value("4", 255);
    write_value("5", LRpB_expected / warpSize);
#else
    uint32_t buf[2] = {102, 256};
    write_values("4", buf, 2);
    buf[0] = buf[1] = LRpB_expected / warpSize;
    write_values("5", buf, 2);
#endif
}
#endif // end opencl
#endif // end MINIMUM_FOR_COMPILE_TESTS
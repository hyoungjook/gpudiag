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

#ifdef KERNEL_FILE
#include KERNEL_FILE
#endif

#else                       // TRY OPENCL
#include <CL/cl.h>

#endif

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

#endif // end opencl
#endif // end MINIMUM_FOR_COMPILE_TESTS
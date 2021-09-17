#include <stdio.h>
#include <stdint.h>
#include <fstream>
#include <string>
#include "hip/hip_runtime.h"
#include <assert.h>

#define CHECK(code)\
{\
    hipError_t err = code;\
    if (err != hipSuccess) {\
        printf("%s: %s\n", #code, hipGetErrorName(err));\
        printf("    %s\n", hipGetErrorString(err));\
    }\
}

#define hipLaunchKernelP(name, g, b, shm, str, ...) \
{printf("  Launched " #name ": (G,B)=(%d,%d)\n", (int)g.x, (int)b.x);\
hipLaunchKernelGGL(name, g, b, shm, str, __VA_ARGS__);}

#define OUT_NAME "report.txt"

void write_init(const char *testname) {
    std::ofstream out(REPORT_DIR OUT_NAME);
    out << "## " << testname << " report" << std::endl;
    out.close();
}

void write_line(const char *line) {
    std::ofstream out(REPORT_DIR OUT_NAME, std::ios_base::app);
    out << line << std::endl;
    out.close();
}

template <typename T>
void write_value(const char *fname, T val) {
    std::ofstream out(REPORT_DIR OUT_NAME, std::ios_base::app);
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
    std::ofstream out(REPORT_DIR OUT_NAME, std::ios_base::app);
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
    std::ofstream out(REPORT_DIR OUT_NAME, std::ios_base::app);
    out << "@" << title << ":" << n << ":";
    out << xlabel << ":" << x0 << ":" << dx << ":";
    out << ylabel << ":";
    for (int i=0; i<n-1; i++) out << ys[i] << ",";
    out << ys[n-1] << std::endl;
    out.close();
}

template <typename Tx, typename Ty, typename Tls, typename Tly>
void write_graph_data_with_line(const char *title, int n, const char *xlabel, Tx x0, Tx dx,
        const char *ylabel, Ty *ys, Tls slope, Tly y_intercept, const char *linelabel) {
    assert(n > 0);
    std::ofstream out(REPORT_DIR OUT_NAME, std::ios_base::app);
    out << "@" << title << ":" << n << ":";
    out << xlabel << ":" << x0 << ":" << dx << ":";
    out << ylabel << ":";
    for (int i=0; i<n-1; i++) out << ys[i] << ",";
    out << ys[n-1] << ":";
    out << slope << "," << y_intercept << ",";
    out << linelabel << std::endl;
    out.close();
}
// MANUFACTURER=1 (nvidia), =0 (amd)
#define MANUFACTURER
#define REPORT_DIR

#define warp_size
#define limit_threads_per_block

#include "tool.h"
#include "exec_units.h"

int main(int argc, char **argv) {
    CHECK(hipSetDevice(0));


    return 0;
}
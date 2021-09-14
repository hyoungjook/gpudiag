// MANUFACTURER=1 (nvidia), =0 (amd)
#define MANUFACTURER
#define REPORT_DIR
#define l1i_linesize

#define ckpt_max_icache_investigate_size_KiB

#include "tool.h"
#include "icache_hierarchy.h"

const int max_KiB = ckpt_max_icache_investigate_size_KiB;

int main(int argc, char **argv) {
    CHECK(hipSetDevice(0));

    const uint64_t num_blocks = max_KiB * 1024 / l1i_linesize;
    const int max_level = 10;

    uint64_t hres[num_blocks];
    uint64_t *dres;
    const size_t arrsize = num_blocks * sizeof(uint64_t);
    hipMalloc(&dres, arrsize);
    for (int i=0; i<num_blocks; i++) hres[i] = 0;
    hipMemcpy(dres, hres, arrsize, hipMemcpyHostToDevice);

    hipLaunchKernelGGL(measure_icache, dim3(1), dim3(1), 0, 0, dres);
    hipDeviceSynchronize();
    hipMemcpy(hres, dres, arrsize, hipMemcpyDeviceToHost);

    for (int i=0; i<num_blocks-1; i++) {
        hres[i] = hres[i+1] - hres[i];
    }

    // use the result to extract the icache data
    int level_n_hit[num_blocks-1];
    uint64_t level_n_hit_latency[max_level];
    for (int i=0; i<max_level; i++) level_n_hit_latency[i] = 0;
    level_n_hit_latency[0] = (hres[0] + hres[1]) / 2;
    int current_level = 0;
    uint64_t level_n_first_occurrence[max_level];

    // label each data
    for (uint64_t i=0; i<num_blocks-1; i++) {
        if (hres[i] > 1.5*level_n_hit_latency[current_level]) {
            // new hierarchy discovered!
            current_level++;
            level_n_hit[i] = current_level;
            level_n_hit_latency[current_level] = hres[i];
            level_n_first_occurrence[current_level] = i;
            continue;
        }

        // determine which level hit the data is
        float lat = (float)hres[i];
        float min_err = lat / (float)level_n_hit_latency[0] + 2;
        int level_of_min_error = 0;
        for (int l=0; l<=current_level; l++) {
            float lat_n = (float)level_n_hit_latency[l];
            float err = lat / lat_n;
            err = err<1 ? 1-err : err-1;
            if (err < min_err) {
                min_err = err; level_of_min_error = l;
            }
        }
        level_n_hit[i] = level_of_min_error;
    }
    int max_observed_level = current_level;

    // calc capacity
    uint64_t level_n_capacity[max_level];
    for (int l=0; l<=max_observed_level-1; l++) {
        level_n_capacity[l] = (level_n_first_occurrence[l+1]+1) * l1i_linesize;
    }

    // calc linesize
    uint32_t level_n_linesize[max_level];
    level_n_linesize[0] = l1i_linesize;
    bool found_max_observed_levels_linesize = true;
    for (int l=1; l<=max_observed_level; l++) {
        uint64_t idx = level_n_first_occurrence[l] + 1;
        while(level_n_hit[idx] != l) {
            idx++;
            if (idx > num_blocks-2) {
                found_max_observed_levels_linesize = false;
                break;
            }
        }
        level_n_linesize[l] = (idx - level_n_first_occurrence[l]) * l1i_linesize;
    }

    write_init("icache_hierarchy");
    write_line("# 1. Raw data");
    write_graph_data("Icache hierarchy", num_blocks-1, 
        "inst. size(B)", l1i_linesize, l1i_linesize,
        "latency", hres);

    write_line("# 2. Which level hit is each data?");
    char buf[num_blocks * 10 + 50];
    int bufptr = sprintf(buf, "# ");
    for (int i=0; i<num_blocks-1; i++) {
        bufptr += sprintf(buf+bufptr, "%d, ", level_n_hit[i]);
    }
    write_line(buf);

    sprintf(buf, "# 3. Observed hierarchy up to %dKiB", max_KiB);
    write_line(buf);
    if (max_observed_level > 0) {
        write_values("icache_capacities", level_n_capacity, max_observed_level);
    }
    else {
        write_line("# only 1 level is observed.");
        write_line("# L1I capacity is at least following.");
        write_value("icache_capacities", max_KiB * 1024);
    }
    if (found_max_observed_levels_linesize) {
        write_values("icache_linesizes", level_n_linesize, max_observed_level+1);
    }
    else {
        write_values("icache_linesizes", level_n_linesize, max_observed_level);
    }
    

    hipFree(dres);

    return 0;
}
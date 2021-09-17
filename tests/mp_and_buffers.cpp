// MANUFACTURER=1 (nvidia), =0 (amd)
#define MANUFACTURER
#define REPORT_DIR

#define warp_size
#define limit_threads_per_block
#define limit_sharedmem_per_block

#define ckpt_nslot_timeout_multiplier
#define ckpt_shared_memory_test_granularity
#define ckpt_mab_skip_num_mp_and_use
#define ckpt_mab_skip_bbsize_and_use
#define ckpt_mab_skip_shmem

#include "tool.h"
#include "mp_and_buffers.h"
#include <vector>

// returns num_mp
uint32_t num_mp_test();
// returns barrier_buffer_size
uint32_t warpstatebuffer_test(uint32_t num_mp);
void sharedmemory_test(uint32_t num_mp, uint32_t barrier_buffer_size);

uint64_t test_nslot_warpstatebuffer(uint32_t G, int b, uint64_t timeout);
int measure_nslot_warpstatebuffer(int b, uint32_t num_mp);
uint64_t test_nslot_sharedmem(uint32_t G, int s_idx, uint64_t timeout);
int measure_nslot_sharedmem(int s_idx, uint32_t num_mp);

int main(int argc, char **argv) {
    CHECK(hipSetDevice(0));
    write_init("mp_and_buffers");
    uint32_t num_mp = 0, barrier_buffer_size = 0;

    if (ckpt_mab_skip_num_mp_and_use <= 0) {
        printf("Running num_mp test..\n");
        num_mp = num_mp_test();
    }
    else num_mp = ckpt_mab_skip_num_mp_and_use;

    if (ckpt_mab_skip_bbsize_and_use <= 0) {
        printf("Running warpstatebuffer test..\n");
        barrier_buffer_size = warpstatebuffer_test(num_mp);
    }
    else barrier_buffer_size = ckpt_mab_skip_bbsize_and_use;

    if (ckpt_mab_skip_shmem <= 0) {
        printf("Running sharedmemory test..\n");
        sharedmemory_test(num_mp, barrier_buffer_size);
    }
    
    return 0;
}

uint32_t num_mp_test() {
    const int iter_num = limit_threads_per_block / warp_size;
    // Measure n_eff of BR
    uint64_t htime, *dtime;
    hipMalloc(&dtime, sizeof(uint64_t));
    uint64_t br_times_raw[iter_num];
    float T_br_at_c[iter_num];
    for (int b=1; b<=iter_num; b++) {
        htime = 0;
        hipMemcpy(dtime, &htime, sizeof(uint64_t), hipMemcpyHostToDevice);
        hipLaunchKernelP(
            measure_width_br, dim3(1), dim3(b * warp_size), 0, 0, dtime);
        hipStreamSynchronize(0);
        hipMemcpy(&htime, dtime, sizeof(uint64_t), hipMemcpyDeviceToHost);
        br_times_raw[b-1] = htime;
    }
    for (int c=0; c<iter_num; c++) {
        T_br_at_c[c] = (float)br_times_raw[c] / (float)br_times_raw[0];
    }
    write_line("# 1. Branch op. n_eff raw data");
    write_graph_data("f(c) of branch op.", iter_num, "c", 1, 1,
        "measured f", T_br_at_c);

    // Calculate c_br
    int c_br = -1;
    for (int c=0; c<iter_num; c++) {
        int index_of_2cbr = (c+1)*2-1;
        if (index_of_2cbr >= iter_num) break;
        if (T_br_at_c[index_of_2cbr] > 1.8 * T_br_at_c[c]) {
            c_br = c+1; break;
        }
    }
    if (c_br == -1) {
        printf("Failed to find c_br!\n");
        exit(1);
    }
    write_line("# 2. c_br result");
    char buf[30];
    sprintf(buf, "# c_br: %d", c_br);
    write_line(buf);
    hipFree(dtime);

    // Measure n_mp
    uint32_t hsync = 0, *dsync;
    hipMalloc(&dsync, sizeof(uint32_t));
    uint64_t *hmptime, *dmptime;
    uint32_t h_isto, *d_isto; // is_timeout
    hipMalloc(&d_isto, sizeof(uint32_t));
    uint64_t *htotclk, *dtotclk;

    std::vector<uint64_t> mp_times;
    uint32_t testG = 1;
    uint64_t mp_avg_until_G = 0;
    uint64_t mp_timeout = 0; // initially, no timeout (=0)
    while (true) {
        hipMemcpy(dsync, &hsync, sizeof(uint32_t), hipMemcpyHostToDevice);
        hmptime = (uint64_t*)malloc(testG * sizeof(uint64_t));
        hipMalloc(&dmptime, testG * sizeof(uint64_t));
        h_isto = 0;
        hipMemcpy(d_isto, &h_isto, sizeof(uint32_t), hipMemcpyHostToDevice);
        htotclk = (uint64_t*)malloc(testG * sizeof(uint64_t));
        hipMalloc(&dtotclk, testG * sizeof(uint64_t));

        hipLaunchKernelP(measure_num_mp, dim3(testG), dim3(c_br * warp_size),
            0, 0, dsync, testG, dmptime, mp_timeout, d_isto, dtotclk);
        hipStreamSynchronize(0);

        // decide timeout
        hipMemcpy(htotclk, dtotclk, testG * sizeof(uint64_t), hipMemcpyDeviceToHost);
        uint64_t max = 0;
        for (int i=0; i<testG; i++) max = max > htotclk[i] ? max : htotclk[i];
        max *= ckpt_nslot_timeout_multiplier;
        hipMemcpy(&h_isto, d_isto, sizeof(uint32_t), hipMemcpyDeviceToHost);
        if (testG > 1 && h_isto == 1) {
            hipFree(dmptime); free(hmptime);
            break; // if Nslot @ b=c_br is 1, deadlock will occur @ G=n_mp+1
        }
        mp_timeout = mp_timeout > max ? mp_timeout : max;

        // measure max time, detect sudden increase
        hipMemcpy(hmptime, dmptime, testG * sizeof(uint64_t), hipMemcpyDeviceToHost);
        max = 0;
        for (int i=0; i<testG; i++) max = max > hmptime[i] ? max : hmptime[i];
        mp_times.push_back(max);
        hipFree(dmptime); free(hmptime);
        
        if (testG == 1) mp_avg_until_G = max;
        else {
            if ((float)max > 1.5 * (float)mp_avg_until_G) {
                break; // if Nslot @ b=c_br is >=2, time will be x2 @ G=n_mp+1
            }
            mp_avg_until_G = (mp_avg_until_G * (testG-1) + max) / testG;
        }
        testG++;
    }
    uint32_t num_mp = testG - 1;
    write_line("# 3. num_mp raw data");
    uint64_t *tmp_data = (uint64_t*)malloc(mp_times.size() * sizeof(uint64_t));
    for (int i=0; i<mp_times.size(); i++) tmp_data[i] = mp_times[i];
    write_graph_data("N_mp measurements", mp_times.size(),
        "GridDim", 1, 1, "latency", tmp_data);
    free(tmp_data);
    write_value("num_mp", num_mp);
    hipFree(dsync);
    hipFree(d_isto);
    return num_mp;
}

uint32_t warpstatebuffer_test(uint32_t num_mp) {
    const int iter_num = limit_threads_per_block / warp_size;
    int Nbuf[iter_num];
    for (int b=1; b<=iter_num; b++) {
        Nbuf[b-1] = measure_nslot_warpstatebuffer(b, num_mp);
    }
    uint32_t barrier_buffer_size;
    uint32_t wsb_size_min = 0, wsb_size_max = 0xFFFFFFFF;
    int non_N1_data_cnt = 0;
    bool N2_conflict_occurred = false;
    for (int b=iter_num; b>=1; b--) {
        int n_at_b = Nbuf[b-1];
        non_N1_data_cnt++;
        // possible values for N2
        uint32_t min, max;
        min = n_at_b * b; max = (n_at_b + 1) * b - 1;
        // intersection
        uint32_t tmp_wsb_size_min = wsb_size_min>min?wsb_size_min:min;
        uint32_t tmp_wsb_size_max = wsb_size_max<max?wsb_size_max:max;
        if (tmp_wsb_size_min > tmp_wsb_size_max) {
            N2_conflict_occurred = true;
            barrier_buffer_size = (uint32_t)n_at_b;
            break;
        }
        wsb_size_min = tmp_wsb_size_min;
        wsb_size_max = tmp_wsb_size_max;
    }
    if (!N2_conflict_occurred) barrier_buffer_size = (uint32_t)Nbuf[0];
    write_line("# 4. Warp state buffer Nslot raw data");
    write_graph_data("Warp state buffer size", iter_num,
        "BlockDim", 1, 1, "Nslot", Nbuf);
    char buf[100];
    write_line("# 5. Warp state buffer analysis");
    int bufptr = sprintf(buf, "# Found %d non-N1 data", non_N1_data_cnt);
    if (non_N1_data_cnt == 0) bufptr += sprintf(buf+bufptr, ": unable to analyze N2");
    write_line(buf);
    write_value("barrier_buffer_size", barrier_buffer_size);
    if (non_N1_data_cnt > 0) {
        write_line("# [min possible value, max possible value]");
        int wsb_size[2]; wsb_size[0] = wsb_size_min; wsb_size[1] = wsb_size_max;
        write_values("warp_state_buffer_size", wsb_size, 2);
    }
    return barrier_buffer_size;
}

uint64_t test_nslot_warpstatebuffer(uint32_t G, int b, uint64_t timeout) {
    uint32_t hsync = 0, *dsync, h_isto = 0, *d_isto;
    hipMalloc(&dsync, sizeof(uint32_t));
    hipMalloc(&d_isto, sizeof(uint32_t));
    hipMemcpy(dsync, &hsync, sizeof(uint32_t), hipMemcpyHostToDevice);
    hipMemcpy(d_isto, &h_isto, sizeof(uint32_t), hipMemcpyHostToDevice);
    uint64_t *hclks, *dclks;
    hclks = (uint64_t*)malloc(G * sizeof(uint64_t));
    hipMalloc(&dclks, G * sizeof(uint64_t));

    hipLaunchKernelP(measure_warpstatebuffer,
        dim3(G), dim3(b * warp_size), 0, 0,
        dsync, G, timeout, d_isto, dclks);
    hipStreamSynchronize(0);
    hipMemcpy(&h_isto, d_isto, sizeof(uint32_t), hipMemcpyDeviceToHost);
    hipMemcpy(hclks, dclks, G * sizeof(uint64_t), hipMemcpyDeviceToHost);

    uint64_t retval;
    if (h_isto == 1) retval = 0;
    else {
        uint64_t max = 0;
        for (int i=0; i<G; i++) max = max>hclks[i]?max:hclks[i];
        retval = max;
    }
    hipFree(dsync); hipFree(d_isto); hipFree(dclks); free(hclks);
    return retval;
}
int measure_nslot_warpstatebuffer(int b, uint32_t num_mp) {
    const int timeout_threshold = ckpt_nslot_timeout_multiplier;
    // initial timeout
    uint64_t timeout = test_nslot_warpstatebuffer(num_mp, b, 0) * timeout_threshold;
    int testN = 1;
    while(true) {
        uint32_t testG = testN * num_mp + 1;
        uint64_t ret = test_nslot_warpstatebuffer(testG, b, timeout) * timeout_threshold;
        if (ret == 0) break;
        else timeout = timeout>ret?timeout:ret;
        testN++;
    }
    return testN;
}

void sharedmemory_test(uint32_t num_mp, uint32_t barrier_buffer_size) {
    const int shmem_unit = ckpt_shared_memory_test_granularity;
    const int iter_num = limit_sharedmem_per_block / shmem_unit;
    int Nbuf[iter_num];
    for (int i=0; i<iter_num; i++) {
        Nbuf[i] = measure_nslot_sharedmem(i, num_mp);
    }
    uint64_t shm_size_min = barrier_buffer_size, shm_size_max = 0xFFFFFFFFFFFFFFFF;
    int non_N1_data_cnt = 0;
    bool S_conflict_occurred = false;
    for (int i=0; i<iter_num; i++) {
        uint64_t test_s = shmem_unit * (i+1);
        int n_at_s = Nbuf[i];
        if (n_at_s >= barrier_buffer_size) continue; // truncated data
        non_N1_data_cnt++;
        // possible values for S
        uint32_t min, max;
        min = n_at_s * test_s; max = (n_at_s + 1) * test_s - 1;
        // intersection
        shm_size_min = shm_size_min>min?shm_size_min:min;
        shm_size_max = shm_size_max<max?shm_size_max:max;
        if (shm_size_min > shm_size_max) {
            S_conflict_occurred = true;
        }
    }
    write_line("# 6. Shared memory Nslot raw data");
    write_graph_data("Shared memory size", iter_num,
        "kernel sharedmem size (B)", shmem_unit, shmem_unit,
        "Nslot", Nbuf);
    write_line("# 7. Shared memory analysis");
    char buf[100];
    int bufptr = sprintf(buf, "# Found %d non-N1 data", non_N1_data_cnt);
    if (non_N1_data_cnt == 0) bufptr += sprintf(buf+bufptr, ": unable to analyze N2");
    if (S_conflict_occurred) bufptr += sprintf(buf+bufptr, ", but conflict occurred");
    write_line(buf);
    if (non_N1_data_cnt > 0 && !S_conflict_occurred) {
        write_line("# [min possible value, max possible value]");
        int shm_size[2]; shm_size[0] = shm_size_min; shm_size[1] = shm_size_max;
        write_values("shared_memory_size", shm_size, 2);
    }
}

uint64_t test_nslot_sharedmem(uint32_t G, int s_idx, uint64_t timeout) {
    uint32_t hsync, *dsync, h_isto, *d_isto;
    hipMalloc(&dsync, sizeof(uint32_t));
    hipMalloc(&d_isto, sizeof(uint32_t));
    hsync = 0; h_isto = 0;
    hipMemcpy(dsync, &hsync, sizeof(uint32_t), hipMemcpyHostToDevice);
    hipMemcpy(d_isto, &h_isto, sizeof(uint32_t), hipMemcpyHostToDevice);
    uint64_t *hclks, *dclks;
    hclks = (uint64_t*)malloc(G * sizeof(uint64_t));
    hipMalloc(&dclks, G * sizeof(uint64_t));

    hipLaunchKernelP(measure_shmem[s_idx], dim3(G), dim3(1), 0, 0,
        dsync, G, timeout, d_isto, dclks);
    hipStreamSynchronize(0);
    hipMemcpy(&h_isto, d_isto, sizeof(uint32_t), hipMemcpyDeviceToHost);
    hipMemcpy(hclks, dclks, G * sizeof(uint64_t), hipMemcpyDeviceToHost);

    uint64_t retval;
    if (h_isto == 1) retval = 0;
    else {
        uint64_t max = 0;
        for (int i=0; i<G; i++) max = max>hclks[i]?max:hclks[i];
        retval = max;
    }
    hipFree(dsync); hipFree(d_isto); hipFree(dclks); free(hclks);
    return retval;
}
int measure_nslot_sharedmem(int s_idx, uint32_t num_mp) {
    const int timeout_threshold = ckpt_nslot_timeout_multiplier;
    // initial timeout
    uint64_t timeout = test_nslot_sharedmem(num_mp, s_idx, 0) * timeout_threshold;
    int testN = 1;
    while(true) {
        uint32_t testG = testN * num_mp + 1;
        uint64_t ret = test_nslot_sharedmem(testG, s_idx, timeout) * timeout_threshold;
        if (ret == 0) break;
        else timeout = timeout>ret?timeout:ret;
        testN++;
    }
    return testN;
}
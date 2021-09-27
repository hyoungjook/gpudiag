uint64_t test_nslot(uint32_t G, uint32_t B, uint64_t timeout,
        void (*kernel)(uint32_t*,uint32_t,uint64_t,uint32_t*,uint64_t*)) {
    uint32_t hsync = 0, *dsync, h_isto = 0, *d_isto;
    GDMalloc(&dsync, sizeof(uint32_t));
    GDMalloc(&d_isto, sizeof(uint32_t));
    GDMemcpyHToD(dsync, &hsync, sizeof(uint32_t));
    GDMemcpyHToD(d_isto, &h_isto, sizeof(uint32_t));
    uint64_t *hclks, *dclks;
    hclks = (uint64_t*)malloc(G * sizeof(uint64_t));
    GDMalloc(&dclks, G * sizeof(uint64_t));

    GDLaunchKernel(kernel, dim3(G), dim3(B), 0, 0,
        dsync, G, timeout, d_isto, dclks);
    GDSynchronize();
    GDMemcpyDToH(&h_isto, d_isto, sizeof(uint32_t));
    GDMemcpyDToH(hclks, dclks, G * sizeof(uint64_t));

    uint64_t retval;
    if (h_isto == 1) retval = 0;
    else {
        uint64_t max = 0;
        for (int i=0; i<G; i++) max = max>hclks[i]?max:hclks[i];
        retval = max;
    }
    GDFree(dsync); GDFree(d_isto); GDFree(dclks); free(hclks);
    return retval;
}
int measure_nslot(uint32_t num_mp, uint32_t B,
        void (*kernel)(uint32_t*,uint32_t,uint64_t,uint32_t*,uint64_t*)) {
    const int timeout_threshold = ckpt_nslot_timeout_multiplier;
    // initial timeout
    uint64_t timeout = test_nslot(num_mp, B, 0, kernel) * timeout_threshold;
    int testN = 1;
    while(true) {
        uint32_t testG = testN * num_mp + 1;
        uint64_t ret = test_nslot(testG, B, timeout, kernel) * timeout_threshold;
        if (ret == 0) break;
        else timeout = timeout>ret?timeout:ret;
        testN++;
    }
    return testN;
}

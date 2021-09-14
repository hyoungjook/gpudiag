
def measure_time_code(repeat, repeat_for, name, inicode, repcode, fincode):
    # inicode;
    # for (n=0..<repeat_for) {
    #   repcode; repcode; ...; repcode; // repeat times
    #   fincode;
    # }
    # repcode: function of i (i -> string)
    code = """\
__global__ void measure_time_{}(uint64_t *result){{
    asm volatile(".reg .pred mt_p, mt_thread0;\\n");
    asm volatile(".reg .u32 mt_sclk, mt_eclk, mt_rep, mt_tid;\\n");
    asm volatile("mov.u32 mt_tid, %tid.x;\\n");
    asm volatile("setp.eq.u32 mt_thread0, mt_tid, 0;\\n");
""".format(name)
    code += inicode
    code += """\
    asm volatile("mov.u32 mt_rep, {};\\n");
    asm volatile("MT_WARMUP_ICACHE_DONE_{}:\\n");
    asm volatile("bar.sync 0;\\n");
    asm volatile("@mt_thread0 mov.u32 mt_sclk, %clock;\\n");
    asm volatile("MT_REPEAT_BEGIN_{}:\\n");
""".format(repeat_for, name, name)
    for i in range(repeat):
        code += repcode(i)
    code += fincode
    code += """\
    asm volatile("setp.lt.u32 mt_p, mt_rep, {};\\n");
    asm volatile("add.u32 mt_rep, mt_rep, 1;\\n");
    asm volatile("@mt_p bra MT_REPEAT_BEGIN_{};\\n");

    asm volatile("bar.sync 0;\\n");
    asm volatile("@mt_thread0 mov.u32 mt_eclk, %clock;\\n");

    asm volatile("setp.eq.u32 mt_p, mt_rep, {};\\n");
    asm volatile("@mt_p mov.u32 mt_rep, 0;\\n");
    asm volatile("@mt_p bra MT_WARMUP_ICACHE_DONE_{};\\n");

    asm volatile("@mt_thread0 sub.u32 mt_eclk, mt_eclk, mt_sclk;\\n");
    asm volatile("@mt_thread0 st.global.u32 [%0], mt_eclk;\\n"::"l"(result));
}}
""".format(repeat_for-1, name, repeat_for+1, name)
    return code
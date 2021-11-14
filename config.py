from tests.define import Test
from env.config_template import config_attribute as conf
from env.config_template import config_values as confval

##### ========================================= #####
##### Select the config preset you want to use! #####
select_config_preset = "gem5"

##### ========================================= #####
#####         Select which test to run!         #####
select_tests_to_run = {
    Test.kernel_limits      : False,
    Test.icache_hierarchy   : False,
    Test.dcache_hierarchy   : False,
    Test.functional_units   : True,
    Test.num_mp             : False,
    Test.warpstate_buffer   : False,
    Test.sharedmem_buffer   : False,
    Test.regfile_buffer     : False,
    Test.warpsched_policy   : False,
}

## You can define the config preset that matches your env. ##
define_config_presets = [
    # gpgpusim config
    {   conf.name: "gpgpusim",
        conf.manufacturer: "nvidia",
        conf.is_simulator: True,
        conf.simulator_path: "/gpgpusim",
        conf.compile_cmd: "nvcc -cudart shared -x cu " +\
            "-gencode arch=compute_75,code=compute_75 " +\
            "-gencode arch=compute_75,code=sm_75 $SRC -o $BIN -Xptxas=-O0",
        conf.run_cmd: "export CUDA_INSTALL_PATH=/usr/local/cuda && " +\
            ". ./setup_environment debug && " +\
            "cp configs/tested-cfgs/SM75_RTX2060_notperfect/gpgpusim.config $DIR && " +\
            "cd $DIR && ./$BIN",
        conf.objdump_cmd: "cuobjdump -ptx $BIN > $OUT",
        conf.use_values: {
            confval.shared_memory_test_granularity: 4096,
            confval.register_test_granularity: 64,
            confval.max_icache_investigate_KiB: 16,
            confval.icache_investigate_interval_B: 512,
            confval.max_dcache_investigate_repeats: 1024,
            confval.fu_latency_repeats: 32,
            confval.nslot_timeout_multiplier: 5,
        }
    },

    # gem5-GCN config
    {   conf.name: "gem5",
        conf.manufacturer: "amd",
        conf.is_simulator: True,
        conf.simulator_path: "/gem5",
        conf.compile_cmd: "/opt/rocm/hip/bin/hipcc --amdgpu-target=gfx803 " +\
            "$SRC -o $BIN",
        conf.run_cmd: "build/GCN3_X86/gem5.opt configs/multigpu/multigpu_se.py " +\
            "-c $DIR/$BIN -n4 --dgpu --gfx-version=gfx803 --reg-alloc-policy=dynamic",
        conf.objdump_cmd: "/opt/rocm/hip/bin/extractkernel -i $BIN && " +\
            "mv $BIN-*.isa $OUT",
        conf.use_values: {
            confval.shared_memory_test_granularity: 4096,
            confval.register_test_granularity: 64,
            confval.max_icache_investigate_KiB: 64,
            confval.icache_investigate_interval_B: 2048,
            confval.max_dcache_investigate_repeats: 1024,
            confval.fu_latency_repeats: 32,
            confval.nslot_timeout_multiplier: 5,
        }
    },

    # AWS g4ad config
    {   conf.name: "g4ad",
        conf.manufacturer: "amd",
        conf.is_simulator: False,
        conf.simulator_path: "",
        conf.compile_cmd: "hipcc --amdgpu-target=gfx1011 " +\
            "$SRC -o $BIN",
        conf.run_cmd: "$DIR/$BIN",
        conf.objdump_cmd: "", # not supported!
        conf.use_values: {
            confval.shared_memory_test_granularity: 128,
            confval.register_test_granularity: 16,
            confval.max_icache_investigate_KiB: 64,
            confval.icache_investigate_interval_B: 2048,
            confval.max_dcache_investigate_repeats: 16384,
            confval.fu_latency_repeats: 32,
            confval.nslot_timeout_multiplier: 10,
        }
    },
]

## ========== The meaning of each config preset attributes ========== ##
conf.name
#       : name of the config preset
conf.manufacturer
#       : manufacturer of the gpu. nvidia or amd
conf.is_simulator
#       : is your environment a simulator?
conf.simulator_path
#       : only valid if is_simulator is True.
#       Should be specified in absolute path.
conf.compile_cmd
#       : command to compile the gpu code.
#       Use source file $SRC and output binary $BIN
conf.run_cmd
#       : command to run the gpu binary.
#       Use binary $BIN in directory $DIR (relative to simulator_path)
#       If is_simulator, assume it is executed on the simulator_path.
conf.objdump_cmd
#       : command to disassemble the gpu binary.
#       Use binary $BIN and disassembled output $OUT
conf.use_values
#       : values to use when running the tests.

## ========== The meaning of each values in the use_values ========== ##

confval.shared_memory_test_granularity
# The test result related to the shared memory will have the accuracy of
# following granularity. Smaller value = more accurate = more time-consuming.
## USED IN: kernel_limits, sharedmem_buffer

confval.register_test_granularity
# The test result related to the register file will have the accuracy of
# following granularity. Smaller value = more accurate = more time-consuming.
## USED IN: kernel_limits, regfile_buffer

confval.max_icache_investigate_KiB
confval.icache_investigate_interval_B
# The icache hierarchy will be investigated up to the following. (KiB unit)
# interval: only investigate every (interval) repeats. (Byte unit)
## USED IN: icache_hierarchy

confval.max_dcache_investigate_repeats
# The dcache hierarchy will be investigated up to the following.
# Max size investigated: (x(L1D$ linesize) Bytes)
# If limit_sharedmem_per_block/8 is less then below, that value will be used.
## USED IN: dcache_hierarchy

confval.fu_latency_repeats
# num of instruction repeats when measuring the latency
## USED_IN: functional_units

confval.nslot_timeout_multiplier
# Tests will determine deadlock if the sync consumes more time than
# (previously detected max time consumption) * (this multiplier).
# For simulations, 5 is enough.
## USED IN: num_mp, *_buffer

from enum import Enum
from enum import auto
from define import Test

# True: Run the test / False: Skip the test
run_test = {
    Test.kernel_limits:     False,
    Test.verify_limits:     False,
    Test.l1i_linesize:      True,
    Test.icache_hierarchy:  False,
    Test.mp_and_buffers:    False,
}

# Test configuration value definition
class CKPT(Enum):
    shared_memory_test_granularity = auto()
    max_icache_investigate_size_KiB = auto()
    nslot_timeout_multiplier = auto()
    mab_start_with_num_mp = auto()
    mab_start_with_barrier_buffer_size = auto()

# Test configuration value assignment
values = {
    ## The test result related to the shared memory will have the accuracy of
    ## following granularity. Smaller value = more accurate = more time-consuming.
    ### USED IN: verify_limits, sharedmem_size
    CKPT.shared_memory_test_granularity: 1024,

    ## The icache hierarchy will be investigated up to the following size (KiB).
    ## For gpgpusim with lengauer-tarjan algorithm, 128 is recommended.
    ## Larger value = more hierarchy revealed = more time-consuming.
    ### USED IN: icache_hierarchy
    CKPT.max_icache_investigate_size_KiB: 128,

    ## Tests will determine deadlock if the sync consumes more time than
    ## (previously detected max time consumption) * (this multiplier).
    ## For simulations, 10 is enough.
    ### USED IN: mp_and_buffers
    CKPT.nslot_timeout_multiplier: 10,

    ## mp_and_buffers runs multiple tests:
    ## (1) num_mp / (2) barrier_buffer_size & warp_state_buffer_size /
    ## (3) shared_memory_size
    ## If you want to skip the former and start from the later,
    ## specify following values.
    ## Test is run if the value <= 0.
    ### USED IN: mp_and_buffers
    CKPT.mab_start_with_num_mp: 0, # if >0, start from (2)
    CKPT.mab_start_with_barrier_buffer_size: 0, # if >0, start from (3)

}


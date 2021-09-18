from enum import Enum
from enum import auto

class Test(Enum):
    kernel_limits = auto()
    verify_limits = auto()
    l1i_linesize = auto()
    icache_hierarchy = auto()
    mp_and_buffers = auto()
    functional_units = auto()

class Feature(Enum):
    warp_size = auto()
    limit_threads_per_block = auto()
    limit_threads_per_grid = auto()
    limit_sharedmem_per_block = auto()
    limit_registers_per_thread = auto()
    limit_registers_per_block = auto()

    l1i_linesize = auto()
    icache_capacities = auto()
    icache_linesizes = auto()

    num_mp = auto()
    barrier_buffer_size = auto()
    warp_state_buffer_size = auto()
    shared_memory_size = auto()

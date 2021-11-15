from .test_template import test_template
from .define import Test
import functional_units
import sys, os

### ========== Options for nslot_verify test! ==========
### Pick the FUTest to investigate, by its name
nvidia_inst_to_test = "shmem"
amd_inst_to_test = "shmem"
### How many for repetitions?
for_repetitions = 100

class nslot_verify (test_template):
    test_enum = Test.nslot_verify
    def generate_kernel(self):
        if self.conf.manufacturer() == "nvidia":
            target = nvidia_inst_to_test
            test_list = functional_units.nvidia_insts_to_test
        else:
            target = amd_inst_to_test
            test_list = functional_units.amd_insts_to_test
        code = ""
        for fut in test_list:
            if fut.name == target:
                target_fut = fut
                break
        code += functional_units.measure_width_code(
            target_fut.width_rep, for_repetitions,
            "verify", target_fut.inicode, target_fut.repcode2,
            target_fut.fincode2(target_fut.width_rep))
        code += f"""\
#define TEST_INST_NAME "{target_fut.name}"
"""
        return code

from enum import Enum, auto
from tests.define import Test

class config_attribute (Enum):
    name = auto()
    manufacturer = auto()
    is_simulator = auto()
    simulator_path = auto()
    compile_cmd = auto()
    run_cmd = auto()
    objdump_cmd = auto()
    use_values = auto()

class config_values (Enum):
    shared_memory_test_granularity = auto()
    register_test_granularity = auto()
    max_icache_investigate_KiB = auto()
    icache_investigate_interval_B = auto()
    max_dcache_investigate_repeats = auto()
    fu_latency_repeats = auto()
    nslot_timeout_multiplier = auto()

class config_preset:
    @staticmethod
    def select_dict_from_list(dict_list, preset_name):
        for p in dict_list:
            try:
                pname = p[config_attribute.name]
            except KeyError:
                continue
            if pname == preset_name:
                return p
        # cannot find such preset
        print("The config preset specified in config.py does not exist.")
        print("Please define the preset or change the selected preset.")
        exit(1)

    @staticmethod
    def verify_checkpoint(ckpt_dict):
        for t in Test:
            try:
                val = ckpt_dict[t]
            except KeyError:
                print("The select_tests_to_run in config.py doesn't contain enough data.")
                print(f"Please specify whether to run the test '{t.name}'.")
                exit(1)
            if type(val) != bool:
                print("The select_tests_to_run in config.py contains invalid data.")
                print(f"Please check the value of '{t.name}' again.")
                exit(1)

    def __init__(self, conf_dict):
        self._name = config_preset.retrieve_from_dict(
            conf_dict, config_attribute.name, str)
        self._manufacturer = config_preset.retrieve_from_dict(
            conf_dict, config_attribute.manufacturer, str)
        self._is_simulator = config_preset.retrieve_from_dict(
            conf_dict, config_attribute.is_simulator, bool)
        if self._is_simulator:
            self._simulator_path = config_preset.retrieve_from_dict(
                conf_dict, config_attribute.simulator_path, str)
            if self._simulator_path[0] != '/':
                print("Please specify simulator_path in absolute path.")
                exit(1)
        self._compile_cmd = config_preset.retrieve_from_dict(
            conf_dict, config_attribute.compile_cmd, str)
        self._run_cmd = config_preset.retrieve_from_dict(
            conf_dict, config_attribute.run_cmd, str)
        self._objdump_cmd = config_preset.retrieve_from_dict(
            conf_dict, config_attribute.objdump_cmd, str)
        self._use_values = config_preset.retrieve_from_dict(
            conf_dict, config_attribute.use_values, dict)
        config_preset.verify_confvals(self._use_values)

    @staticmethod
    def retrieve_from_dict(conf_dict, attr, expected_type):
        try:
            val = conf_dict[attr]
        except KeyError:
            print("The selcted config preset doesn't contain enough information.")
            print(f"Please specify the '{attr.name}' attribute.")
            exit(1)
        if type(val) != expected_type:
            print("The selected config preset contains invalid data.")
            print(f"Please check the '{attr.name}' attribute again.")
            exit(1)
        return val
    
    @staticmethod
    def verify_confvals(use_values):
        for confvaltype in config_values:
            try:
                val = use_values[confvaltype]
            except KeyError:
                print("The selected config preset doesn't contain enough information.")
                print(f"Please specify the '{confvaltype}' value in the use_values attribute.")
                exit(1)
            if type(val) != int and type(val) != float:
                print("The selected config preset contains invalid data.")
                print(f"Please check the '{confvaltype}' value in the use_values attribute.")
                exit(1)
    
    def name(self):
        return self._name
    def manufacturer(self):
        return self._manufacturer
    def is_simulator(self):
        return self._is_simulator
    def simulator_path(self):
        if self._is_simulator:
            return self._simulator_path
        return ""
    def compile_cmd(self, src_path, bin_path, include_dirs):
        # include_dirs : array of strings
        cmd = self._compile_cmd
        cmd = cmd.replace("$SRC", src_path)
        cmd = cmd.replace("$BIN", bin_path)
        for incdir in include_dirs:
            cmd += " -I" + incdir
        return cmd
    def run_cmd(self, bin_dir, bin_name):
        cmd = self._run_cmd
        cmd = cmd.replace("$DIR", bin_dir)
        cmd = cmd.replace("$BIN", bin_name)
        return cmd
    def objdump_cmd(self, bin_path, out_path):
        cmd = self._objdump_cmd
        cmd = cmd.replace("$BIN", bin_path)
        cmd = cmd.replace("$OUT", out_path)
        return cmd
    def get_cvalues(self):
        return self._use_values
    def get_confval(self, confval_type):
        return self._use_values[confval_type]
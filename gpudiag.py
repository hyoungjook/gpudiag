import os, importlib
from datetime import datetime
import config
from env.config_template import config_preset
from env.result_manager import result_manager
from tests.define import Test

if __name__ == "__main__":
    proj_path = os.path.dirname(os.path.abspath(__file__))

    # retrieve preset from the config.py
    preset = config_preset(config_preset.select_dict_from_list(
        config.define_config_presets, config.select_config_preset
    ))
    config_preset.verify_checkpoint(config.select_tests_to_run)
    
    # initialize result manager
    result = result_manager(proj_path, preset)
    result.add_timestamp()
    result.create_graph_directory()
    for test in Test:
        if not config.select_tests_to_run[test]:
            continue
        start_time = datetime.now()
        test_module = importlib.import_module(f"tests.{test.name}")
        test_class = getattr(test_module, test.name)
        test_instance = test_class(proj_path, preset, result)
        if not test_instance.run():
            exit(1)
        end_time = datetime.now()
        result.update(test, end_time - start_time)

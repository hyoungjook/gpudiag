import os, sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from env import script_runner, values_delivery

class test_template:
    test_enum = 0 # it is enum for the real test class

    def __init__(self, proj_path, conf, result):
        self.tname = self.test_enum.name
        self.conf = conf
        self.result_values = result.get_fvalues()
        self.proj_path = proj_path
        self.src_dir = os.path.join(self.proj_path, "host_codes")
        self.build_dir = os.path.join(self.proj_path, "build",
            self.conf.manufacturer(), self.tname)
        self.report_path = os.path.join(self.build_dir, "report.txt")
        script_runner.mkdir(self.build_dir)

    def verify_constraints(self):
        return True

    def generate_kernel(self):
        return ""

    def compile(self):
        # define necessary paths
        src_path = os.path.join(self.src_dir, f"{self.tname}.cpp")
        out_bin_path = os.path.join(self.build_dir, self.tname)
        out_kern_path = os.path.join(self.build_dir, f"{self.tname}.h")
        # generate kernel
        script_runner.create_file(self.generate_kernel(), out_kern_path)
        # insert inputs into tmp source file
        tmp_src_path = os.path.join(self.src_dir, f"{self.tname}_tmp.cpp")
        values_delivery.embed_values_to_cpp(
            self.conf.manufacturer(), self.report_path, self.tname,
            self.result_values, self.conf.get_cvalues(),
            src_path, tmp_src_path)
        # compile
        compile_status = script_runner.compile_succeed(
            self.conf, tmp_src_path, out_bin_path,
            [self.src_dir, self.build_dir]
        )
        # remove tmp source file
        script_runner.rm(tmp_src_path)
        return compile_status

    def run(self):
        if not self.verify_constraints():
            return False
        if not self.compile():
            return False
        return script_runner.run_succeed(
            self.conf, self.build_dir, self.tname
        )
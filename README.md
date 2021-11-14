# GPUDiag
Systematic reverse-engineering diagnosis tool for GPUs.
made by Hyoungjoo Kim, Seoul Nat'l University, 2021

## Requirements
The target GPU HW and driver, or a GPU simulator should be installed to the system.
Also, a supported GPU runtime should be installed.
GPUDiag supports CUDA, ROCm HIP and OpenCL(partially supported, not recommended).
In addition, python3 and matplotlib is required.
~~~shell
sudo apt install -y python3 python3-pip
python3 -m pip install --upgrade pip
pip3 install matplotlib
~~~

## How to use
### Modify config.py
- Add your environment info to `define_config_presets` in `config.py`.
- Set `select_config_preset` in `config.py` to your preset name.
- Select the tests you want to run in `select_tests_to_run` in `config.py`
    - For `functional_units` test, you should also modify `nvidia_insts_to_test` or `amd_insts_to_test` in `tests/functional_units.py`.

### Execute
After setting `config.py`, to run the test set,
~~~shell
python3 gpudiag.py
~~~
If you want to clean build & result files,
~~~shell
sudo python3 clean.py
~~~

### Using with gpgpusim
You should prepare gpgpusim-compatible environment first.
The Dockerfile at dockerfiles/gpgpusim is tested.

The following modifications should be applied to the gpgpusim.
- NOTE: the default config uses modified gpgpusim.config at 
configs/tested-cfgs/SM75_RTX2060_notperfect, which was 
modified as non-perfect icache, 4 SMs, 8KiB L1I$, 16KiB L1D$, etc..

Download and build gpgpu-sim.
Download gpudiag.
~~~shell
cd workdir
ls # >> gpgpusim/ gpudiag/
sudo docker run --rm -it -v $PWD/gpgpusim:/gpgpusim \
    -v $PWD/gpudiag:/gpudiag -w /gpudiag gpgpusim_image
## edit config_env.py to match the environment
python3 gpudiag
~~~

### Using with gem5-apu
You should prepare gem5-apu-compatible environment first.
The Dockerfile at dockerfiles/gem5 is tested.

The following modifications should be applied to the gem5.
- Because gem5 21.1.0 don't have s_memtime instruction implemented, 
you may have to implement it manually.
- My repo at https://github.com/kimhj1473/zolp_gem5.git is tested.
- If you modified it yourself, you may have to change config_env.py
run_command.

Download and build gem5.
Download gpudiag.
~~~shell
cd workdir
ls # >> gem5/ gpudiag/
sudo docker run --rm -it -v $PWD/gem5:/gem5 \
    -v $PWD/gpudiag:/gpudiag -w /gpudiag gem5gcn_image
## edit config_env.py to match the environment
python3 gpudiag
~~~
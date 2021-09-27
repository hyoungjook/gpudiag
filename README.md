# GPUDiag
Systematic reverse-engineering diagnosis tool for GPUs.
made by Hyoungjoo Kim, Seoul Nat'l University, 2021

## Requirements
~~~shell
sudo apt install -y python3 pip
pip3 install matplotlib
~~~

## How to use
~~~shell
# set config.py
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
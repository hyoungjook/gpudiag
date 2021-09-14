config_preset = "gpgpusim"
##config_preset = "gem5"
##config_preset = "custom"

#============================================================================
# gpu_manufacturer: manufacturer of the gpu.
if config_preset == "gpgpusim":
    gpu_manufacturer = "nvidia"
elif config_preset == "gem5":
    gpu_manufacturer = "amd"
else:
    gpu_manufacturer = ""

#============================================================================
# simulator_driven: simulator or real HW?
if config_preset == "gpgpusim" or config_preset == "gem5":
    simulator_driven = True
else:
    simulator_driven = False

#============================================================================
# simulator_path: main path of the simulator.
#   Only valid if simulator_driven==True.
#   Please specify in absolute path.
if config_preset == "gpgpusim":
    simulator_path = "/gpgpusim"
elif config_preset == "gem5":
    simulator_path = "/gem5"
else:
    simulator_path = ""

#============================================================================
# compile_command: command to compile the gpu code.
#   Use source file $SRC and output $BIN
if config_preset == "gpgpusim":
    # should be also compiled as sass as -O0, to verify shmem & reg
    compile_command = "/opt/rocm/hip/bin/hipcc -cudart shared " +\
        "-gencode arch=compute_75,code=compute_75 " +\
        "-gencode arch=compute_75,code=sm_75 $SRC -o $BIN -Xptxas=-O0"
elif config_preset == "gem5":
    compile_command = "/opt/rocm/hip/bin/hipcc --amdgpu-target=gfx803 " +\
        "$SRC -o $BIN"
else:
    compile_command = ""

#============================================================================
# run_command: command to run the gpu binary.
#   If simulator_driven, assume the command is executed 
#   on the simulator_path.
#   Use binary $BIN in directory $DIR (relative to the simulator_path)

if config_preset == "gpgpusim":
    run_command = "export CUDA_INSTALL_PATH=/usr/local/cuda && " +\
        ". ./setup_environment debug && " +\
        "cp configs/tested-cfgs/SM75_RTX2060_notperfect/gpgpusim.config $DIR && " +\
        "cd $DIR && ./$BIN"
elif config_preset == "gem5":
    run_command = "build/GCN3_X86/gem5.opt configs/multigpu/multigpu_se.py " +\
        "-c $DIR/$BIN -n4 --dgpu --gfx-version=gfx803"
else:
    run_command = "$DIR/$BIN"

#============================================================================
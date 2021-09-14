
FROM nvidia/cuda:11.0.3-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt -y update
RUN apt -y upgrade

# GPGPUSim dependencies
RUN apt-get -y install build-essential xutils-dev bison zlib1g-dev flex libglu1-mesa-dev\
    git cmake vim

#### GPGPUSim documentation dependencies
###RUN apt-get -y install doxygen graphviz

#### Aerialvision dependencies
###RUN apt-get -y install python-pmw python-ply python-numpy libpng-dev python3-matplotlib

# install hip
RUN git clone -b rocm-4.3.x https://github.com/ROCm-Developer-Tools/HIP.git
RUN cd HIP ; mkdir build
WORKDIR /HIP/build
RUN cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="/opt/rocm/" -DCMAKE_INSTALL_PREFIX=/opt/rocm/hip ..
RUN make && make install
WORKDIR /

# python3 needed for gpudiag
RUN apt-get -y install python3 pip && pip3 install matplotlib

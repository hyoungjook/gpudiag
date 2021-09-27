
FROM nvidia/cuda:11.0.3-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt -y update
RUN apt -y upgrade

# GPGPUSim dependencies
RUN apt-get -y install build-essential xutils-dev bison zlib1g-dev flex libglu1-mesa-dev\
    git cmake vim

# python3 needed for gpudiag
RUN apt-get -y install python3 pip && pip3 install matplotlib

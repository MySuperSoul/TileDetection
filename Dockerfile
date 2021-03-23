FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu16.04

RUN rm -rf /var/lib/apt/lists/* \
    /etc/apt/sources.list.d/cuda.list \
    /etc/apt/sources.list.d/nvidia-ml.list && \
    sed -i s@/archive.ubuntu.com/@/mirrors.ustc.edu.cn/@g /etc/apt/sources.list && \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git \
        curl \
        vim \
        zsh \
        sudo \
        ctags \
        htop \
        tmux \
        ssh \
        xterm \
        zip \
        unzip \
        wget \
        gdb \
        bc \
        man \
        less \
        debconf-utils \
        locate \
        silversearcher-ag \
        ca-certificates \
        libboost-all-dev \
        libjpeg-dev \
        libxau6 \
        libxdmcp6 \
        libxcb1 \
        libxext6 \
        libx11-6 \
        ca-certificates \
        automake \
        autoconf \
        libtool \
        pkg-config \
        lsof \
        libxext-dev \
        libx11-dev \
        xauth \
        x11-utils \
        x11proto-gl-dev \
        libpng-dev \
        php7.0 \
        php7.0-curl \
        libsparsehash-dev \
        gosu \
        rsync

RUN apt-get update && apt-get install -y ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        python3.7 \
        python3.7-dev \
        python3.7-tk && \
    wget -O ~/get-pip.py \
        https://bootstrap.pypa.io/get-pip.py && \
    python3.7 ~/get-pip.py && \
    ln -s /usr/bin/python3.7 /usr/local/bin/python3 && \
    ln -s /usr/bin/python3.7 /usr/local/bin/python

# Install python lib
RUN echo "[global] timeout = 6000 \
index-url = https://mirrors.aliyun.com/pypi/simple \
" >/etc/pip.conf && \
    python3 -m pip --no-cache-dir install -i https://mirrors.aliyun.com/pypi/simple --upgrade \
        setuptools && \
    python3 -m pip --no-cache-dir install -i https://mirrors.aliyun.com/pypi/simple --upgrade \
        numpy \
        numba \
        scipy \
        pandas \
        cloudpickle \
        scikit-learn \
        matplotlib \
        Cython \
        shapely \
        plotly \
        pyyaml \
        loguru \
        opencv-python \
        tqdm \
        easydict \
        glob2 \
        future \
        protobuf \
        enum34 \
        typing \
        ipython \
        jupyter \
        fire \
        tensorboardX \
        ninja \
        jupyterlab \
        yacs \
        scikit-image \
        pybind11 \
        spyder-kernels \
        flake8 \
        yapf \
        addict \
        onnx \
        onnxruntime \
        pytest \
        tb-nightly \
        gpustat

ADD . /work
WORKDIR /work
ENV FORCE_CUDA="1"

# install torch and torchvision
RUN pip install torch-1.6.0+cu101-cp37-cp37m-linux_x86_64.whl
RUN pip install torchvision-0.7.0+cu101-cp37-cp37m-linux_x86_64.whl && \
    rm *.whl

# Install MMCV
RUN pip install https://download.openmmlab.com/mmcv/dist/1.2.5/torch1.6.0/cu101/mmcv_full-1.2.5%2Btorch1.6.0%2Bcu101-cp37-cp37m-manylinux1_x86_64.whl -i https://mirrors.aliyun.com/pypi/simple

# Install MMDetection
# RUN conda clean --all
RUN pip install -r requirements/build.txt -i https://mirrors.aliyun.com/pypi/simple
RUN pip install -v -e . -i https://mirrors.aliyun.com/pypi/simple

# install mish cuda
RUN cd /tmp && git clone https://github.com/JunnYu/mish-cuda && cd mish-cuda && python setup.py build install

RUN pip install ai-hub flask -i https://mirrors.aliyun.com/pypi/simple

RUN ldconfig && \
    updatedb && \
    apt-get clean && \
    apt-get autoremove && \
    rm -rf /var/lib/apt/lists/* /tmp/* ~/*
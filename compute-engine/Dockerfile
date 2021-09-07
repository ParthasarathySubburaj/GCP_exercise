FROM ubuntu:20.04

RUN apt-get update && DEBIAN_FRONTEND=noninteractive
RUN apt-get install -y tzdata
RUN apt-get install -y \
        libsm6 \
        libxext6 \
        git \
        ninja-build \
        libglib2.0-0 \
        libsm6 \
        libxrender-dev \
        libxext6 \
        libboost-dev

RUN apt-get install -y --no-install-recommends \
        git \
        python3-dev  \
        python3-pip \
        neovim \
        unzip \
        wget \
                && apt-get clean \
                && rm -rf /var/lib/apt/lists/*

RUN pip3 install --upgrade pip

RUN pip3 install --no-cache \
        wheel \
        argparse \
        requests \
        setuptools \
        pyYAML \
        Cython

RUN pip3 install --no-cache \
        mxnet \
        gluoncv \
        jupyterlab \
        Flask==2.0.1 \
        gunicorn==20.1.0

ENV APP_HOME /app
WORKDIR $APP_HOME
COPY . ./

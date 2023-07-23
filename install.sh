#!/bin/bash

conda create --prefix env python=3.7
conda activate ./env

# DONT INSTALL TORCH 2
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia

# NEED AT LEAST 32 GB OF RAM, USE SWAP OR SOMETHING
swapoff -a
dd if=/dev/zero of=/swapfile bs=1M count=16384
mkwap /swapfile
swapon /swapfile

export MAX_JOBS=8;
export PATH=/usr/local/cuda-11.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda-11.8/

cd ./MinkowskiEngine

python setup.py install

cd -

pip install -r requirements.txt
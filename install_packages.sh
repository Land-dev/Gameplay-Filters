#!/bin/bash

# Create a new conda environment with Python 3.8
conda create -n mujoco python=3.8 -y

# Activate the newly created environment
conda activate mujoco

# Install CUDA from the specific NVIDIA label
conda install cuda -c nvidia/label/cuda-11.8.0 -y

# Install PyTorch with CUDA 11.8 support
conda install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia/label/cuda-11.8.0 -y

# Install additional packages from conda-forge
conda install -c conda-forge suitesparse jupyter notebook omegaconf numpy tqdm gym dill plotly shapely wandb matplotlib pybullet pandas -y

# Install JAX and JAXlib with CUDA support via pip
pip install --upgrade jax==0.4.8 jaxlib==0.4.7+cuda11.cudnn86 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Install mujoco
pip install mujoco

# Install ISAACS
pip install -e .

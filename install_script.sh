#!/bin/bash

# Ensure a path is provided
if [ -z "$1" ]; then
  echo "Usage: bash install_script.sh <path_to_clone_LightGBM>"
  exit 1
fi

LIGHTGBM_PATH=$1

# Update system and install dependencies
sudo add-apt-repository ppa:graphics-drivers/ppa -y
sudo apt update && sudo apt-get update

# Install NVIDIA drivers and OpenCL dependencies
sudo apt-get install --no-install-recommends -y nvidia-410
sudo apt-get install --no-install-recommends -y nvidia-opencl-icd-410 nvidia-opencl-dev opencl-headers

# Install development tools and Boost libraries
sudo apt-get install --no-install-recommends -y git cmake build-essential libboost-dev libboost-system-dev libboost-filesystem-dev
sudo apt-get install -y libboost-all-dev

# Clone LightGBM repository
if [ ! -d "$LIGHTGBM_PATH" ]; then
  git clone --recursive https://github.com/microsoft/LightGBM "$LIGHTGBM_PATH"
else
  echo "LightGBM directory already exists. Skipping clone."
fi

# Change directory to LightGBM
cd "$LIGHTGBM_PATH" || exit

# Build LightGBM with CUDA support
cmake -B build -S . -DUSE_CUDA=1 -DOpenCL_LIBRARY=/usr/local/cuda/lib64/libOpenCL.so -DOpenCL_INCLUDE_DIR=/usr/local/cuda/include/
cmake --build build -j$(nproc)

# Install LightGBM with CUDA support
pip install setuptools
sh ./build-python.sh install --precompile
sh ./build-python.sh install --cuda

# Install cuML and RAPIDS AI libraries
pip install \
    --extra-index-url=https://pypi.nvidia.com \
    "cudf-cu12==25.2.*" "dask-cudf-cu12==25.2.*" "cuml-cu12==25.2.*" \
    "cugraph-cu12==25.2.*" "nx-cugraph-cu12==25.2.*" "cuspatial-cu12==25.2.*" \
    "cuproj-cu12==25.2.*" "cuxfilter-cu12==25.2.*" "cucim-cu12==25.2.*" \
    "pylibraft-cu12==25.2.*" "raft-dask-cu12==25.2.*" "cuvs-cu12==25.2.*" \
    "nx-cugraph-cu12==25.2.*"

echo "Installation complete! LightGBM and cuML are now set up."


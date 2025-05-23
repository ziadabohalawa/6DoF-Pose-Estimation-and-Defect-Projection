#!/bin/bash
# Pre-reboot setup - everything except CUDA extensions build

set -e  # Exit on any error
echo "--------------------------------------------------------------------------------------"
echo "------------------------------------Starting setup------------------------------------"
cd ~/6DoF_PE_DP || { echo "6DoF_PE_DP directory not found. Did you clone the repo?"; exit 1; }

# System & Python Setup
echo "-----------------------------Installing system packages.------------------------------"
echo "--------------------------------------------------------------------------------------"
sudo apt update && sudo apt upgrade -y
sudo apt install -y git unzip ffmpeg libgl1 software-properties-common \
  build-essential cmake gcc-11 g++-11 libboost-all-dev libeigen3-dev \
  xauth xorg wget curl python3-pip

sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt update
sudo apt install -y python3.9 python3.9-venv python3.9-dev

echo "--------------------------------------------------------------------------------------"
echo "-------------------------Setting up Python virtual environment------------------------"
echo "--------------------------------------------------------------------------------------"
python3.9 -m venv ~/6dof_pe_dp
source ~/6dof_pe_dp/bin/activate

# Python packages
echo "--------------------------------------------------------------------------------------"
echo "------------------------------Installing Python packages------------------------------"
echo "--------------------------------------------------------------------------------------"
pip install --upgrade pip
pip install awscli pybind11[global]
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
export CMAKE_PREFIX_PATH=$(pybind11-config --cmakedir)

# CUDA Installation
echo "--------------------------------------------------------------------------------------"
echo "-------------------------------Installing CUDA toolkit--------------------------------"
echo "--------------------------------------------------------------------------------------"
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install -y cuda-toolkit-11-8

# NVIDIA Driver
echo "--------------------------------------------------------------------------------------"
echo "---------Installing NVIDIA driver---------"
echo "--------------------------------------------------------------------------------------"
sudo add-apt-repository -y ppa:graphics-drivers/ppa
sudo apt update
sudo apt install -y nvidia-driver-535

# Environment Variables
echo "--------------------------------------------------------------------------------------"
echo "----------------------------Setting up environment variables--------------------------"
echo "--------------------------------------------------------------------------------------"
cat >> ~/.bashrc << 'EOF'
export CUDA_HOME=/usr/local/cuda-11.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export TORCH_CUDA_ARCH_LIST="7.5"
source ~/6dof_pe_dp/bin/activate
EOF

source ~/.bashrc

# Project Dependencies (pre-build)
echo "--------------------------------------------------------------------------------------"
echo "----------------------------Installing project dependencies---------------------------"
echo "--------------------------------------------------------------------------------------"
cd ~/6DoF_PE_DP || { echo "6DoF_PE_DP directory not found. Did you clone the repo?"; exit 1; }

pip install torch==2.0.0+cu118 torchvision==0.15.1+cu118 torchaudio==2.0.1+cu118 \
  --extra-index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
pip install git+https://github.com/NVlabs/nvdiffrast.git
pip install kaolin==0.15.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.0.0_cu118.html
pip install pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py39_cu118_pyt200/download.html

# Download Data and Weights
echo "--------------------------------------------------------------------------------------"
echo "---------------------------Downloading demo data and weights--------------------------"
echo "--------------------------------------------------------------------------------------"
mkdir -p demo_data weights
aws s3 cp s3://6dof-pe-dp-bucket/demo_data/ demo_data/ --recursive
aws s3 cp s3://6dof-pe-dp-bucket/weights/ weights/ --recursive

# Azure Kinect SDK
echo "--------------------------------------------------------------------------------------"
echo "-----------------------------Installing Azure Kinect SDK------------------------------"
echo "--------------------------------------------------------------------------------------"
mkdir -p ~/6DoF_PE_DP/k4a_tools && cd ~/6DoF_PE_DP/k4a_tools
wget http://ftp.de.debian.org/debian/pool/main/libs/libsoundio/libsoundio1_1.1.0-1_amd64.deb
wget https://packages.microsoft.com/ubuntu/18.04/prod/pool/main/libk/libk4a1.4/libk4a1.4_1.4.1_amd64.deb
wget https://packages.microsoft.com/ubuntu/18.04/prod/pool/main/libk/libk4a1.4-dev/libk4a1.4-dev_1.4.1_amd64.deb
wget https://packages.microsoft.com/ubuntu/18.04/prod/pool/main/k/k4a-tools/k4a-tools_1.4.1_amd64.deb
sudo dpkg -i libsoundio1_1.1.0-1_amd64.deb
sudo dpkg -i libk4a1.4_1.4.1_amd64.deb
sudo dpkg -i libk4a1.4-dev_1.4.1_amd64.deb
sudo dpkg -i k4a-tools_1.4.1_amd64.deb
sudo apt-get install -f -y

echo ""
echo "Pre-reboot setup complete!"
echo ""
echo " Next steps:"
echo "1. Run: sudo reboot"
echo "2. After reboot, run: source ~/6dof_pe_dp/bin/activate"
echo "3. Verify GPU: nvidia-smi"
echo "4. Verify CUDA: nvcc --version" 
echo "5. Build extensions: cd ~/6DoF_PE_DP && bash build_all_env.sh"
echo ""

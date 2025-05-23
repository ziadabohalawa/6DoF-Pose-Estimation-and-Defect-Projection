# 6DoF Pose Estimation and Defect Projection System
This repository contains the implementation of a system for accurate 6-DoF pose estimation and 3D defect projection of industrial objects. The solution combines deep learning-based pose detection with classical refinement and visualization techniques to enable precise, real-time analysis of object surfaces.
# Core Components
- **FoundationPose:** Initial 6DoF object pose estimation from RGB-D data.

- **ICP (Iterative Closest Point):** Geometric refinement of the pose using 3D point cloud alignment.

- **Ray Tracing:** Accurate projection of detected 2D image defects onto a 3D mesh surface.

- **Plotly Dash:** Interactive 3D web visualization for real-time defect inspection.

# Pipeline Overview
**1. Data Acquisition:**
RGB, depth image, and point cloud data are captured using Azure Kinect DK.

**2. Initial Pose Estimation:**
The FoundationPose model estimates the object’s 6DoF pose from RGB-D input.

**3. Pose Refinement:**
ICP aligns the 3D model to the scene point cloud for improved accuracy.

**4. Defect Detection:**
Detected defect heatmaps are processed and filtered by intensity thresholds.

**5. Ray Tracing Projection:**
2D heatmap pixels are back-projected onto the 3D mesh using ray-mesh intersection.

**6. Web-Based Visualization:**
Defect projections are shown via a Plotly Dash app with rotation, zoom, and overlay features.

![pipleline_overview](https://github.com/ziadabohalawa/6DoF-Pose-Estimation-and-Defect-Projection/blob/32474e4ba11b792ab8ca935a9ce0b3423d89c78e/Pipeline_overview.png)

# Data prepare
1. Download all network weights from [here](https://drive.google.com/drive/folders/1DFezOAD0oD1BblsXVxqDsl8fj0qzB82i) and put them under the folder weights/. For the refiner, you will need 2023-10-28-18-33-37. For scorer, you will need 2024-01-11-20-02-45.

2. [Download demo data](https://drive.google.com/drive/folders/1pRyFmxYXmAnpku7nGRioZaKrVJtIsroP) and extract them under the folder demo_data/

# Installation

### Conda environment Setup

```bash
cd 6DoF_PE_DP/

# Create a new Conda environment with Python 3.9
conda create -n pedp python=3.9
conda activate pedp

# Install Eigen3 3.4.0 under conda environment
conda install conda-forge::eigen=3.4.0
export CMAKE_PREFIX_PATH="$CMAKE_PREFIX_PATH:/eigen/path/under/conda"

# Install PyTorch 2.0.0 + CUDA 11.8
pip install torch==2.0.0+cu118 torchvision==0.15.1+cu118 torchaudio==2.0.1+cu118 --extra-index-url https://download.pytorch.org/whl/cu118

# install dependencies
python -m pip install -r requirements.txt

# Install NVDiffRast
python -m pip install --quiet --no-cache-dir git+https://github.com/NVlabs/nvdiffrast.git

# Kaolin (Optional, needed if running model-free setup)
python -m pip install --quiet --no-cache-dir kaolin==0.15.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.0.0_cu118.html

# PyTorch3D
python -m pip install --quiet --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py39_cu118_pyt200/download.html

# Set CMAKE_PREFIX_PATH
export CMAKE_PREFIX_PATH=$CONDA_PREFIX/lib/python3.9/site-packages/pybind11/share/cmake/pybind11

# Build the C++/CUDA extensions
bash build_all_conda.sh
```

### docker setup
```
cd docker/
docker pull wenbowen123/foundationpose && docker tag wenbowen123/foundationpose foundationpose  # Or to build from scratch: docker build --network host -t foundationpose .
bash docker/run_container.sh
```


If it's the first time you launch the container, you need to build extensions.
```
bash build_all.sh
```

Later you can execute into the container without re-build.
```
docker exec -it foundationpose bash
```
- You need to add the necessary commands to the Dockerfile to use the Azure Kinect SDK. [Azure-Kinect-SDK Docker Guide](https://github.com/microsoft/Azure-Kinect-Sensor-SDK/blob/develop/scripts/docker)

- Then, when running the container, you need to include the necessary flags to access the camera feed and send data to the localhost network.
# Usage

```bash
# navigate to the root directory of the project
cd 2d_3d_pe_dp/

# for a demo run:
python run.py --test_scene_dir path/to/data_folder --demo

# for example: 
python run.py --test_scene_dir demo_data/tless_07 --demo

# for a live run with Azure Kinect DK:
python run.py --test_scene_dir demo_data/data_folder --capture_background True
```

# AWS Deployment

## 1️⃣ Prepare S3 Storage

### 1.1 Create an S3 Bucket

Create a bucket named:

```text
6dof-pe-dp-bucket
```

> Make sure your EC2 IAM role has `AmazonS3ReadOnlyAccess`.

### 1.2 Upload Project Assets

From your local machine:

```bash
aws s3 cp demo_data/ s3://6dof-pe-dp-bucket/demo_data/ --recursive
aws s3 cp weights/ s3://6dof-pe-dp-bucket/weights/ --recursive
```

---

## 2️⃣ Launch and Connect to an EC2 Instance

### 2.1 Instance Configuration

* **Instance Type:** `g4dn.xlarge`
* **AMI:** Ubuntu 22.04
* **Storage:** 75 GB (gp2 or gp3)
* **IAM Role:** With S3 read access
* **Security Group:** Allow TCP port `22` (SSH) and `8050` (web access)

### 2.2 Connect via SSH

```bash
ssh -X -i your-key.pem ubuntu@<EC2_PUBLIC_IP>
```

---

## 3️⃣ Clone the Repository

```bash
git clone https://github.com/ziadabohalawa/6DoF_PE_DP.git
cd 6DoF_PE_DP
```

---

## 4️⃣ Set Up the Environment

Choose one of the following setup methods:

---

### 🔹 Option A: Automated Setup (Recommended)

1. Run the provided script:

   ```bash
   bash setup_env.sh
   ```

2. Once it completes, proceed to **Step 8: Reboot & Build CUDA Extensions**.

---

### 🔹 Option B: Manual Setup

Follow the steps below **in order** to manually configure the system.



## 5️⃣ Manual Setup Steps

### 5.1 Install System & Python Dependencies

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y git unzip ffmpeg libgl1 software-properties-common \
  build-essential cmake gcc-11 g++-11 libboost-all-dev libeigen3-dev \
  xauth xorg wget curl python3-pip

sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt update
sudo apt install -y python3.9 python3.9-venv python3.9-dev

python3.9 -m venv ~/6dof_pe_dp
source ~/6dof_pe_dp/bin/activate
```

### 5.2 Install AWS CLI, Pybind11 & Update PATH

```bash
pip install --upgrade pip
pip install awscli pybind11[global]
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
export CMAKE_PREFIX_PATH=$(pybind11-config --cmakedir)
```

---

## 6️⃣ Install GPU, CUDA & Project Dependencies

### 6.1 Install CUDA 11.8

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install -y cuda-toolkit-11-8
```

### 6.2 Install NVIDIA Driver 535

```bash
sudo add-apt-repository -y ppa:graphics-drivers/ppa
sudo apt update
sudo apt install -y nvidia-driver-535
```

### 6.3 Set Environment Variables

```bash
echo 'export CUDA_HOME=/usr/local/cuda-11.8' >> ~/.bashrc
echo 'export PATH=$CUDA_HOME/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
echo 'export TORCH_CUDA_ARCH_LIST="7.5"' >> ~/.bashrc
echo 'source ~/6dof_pe_dp/bin/activate' >> ~/.bashrc
source ~/.bashrc
```

---

## 7️⃣ Set Up the Project

### 7.1 Install Python Dependencies

```bash
cd ~/6DoF_PE_DP

pip install torch==2.0.0+cu118 torchvision==0.15.1+cu118 torchaudio==2.0.1+cu118 \
  --extra-index-url https://download.pytorch.org/whl/cu118

pip install -r requirements.txt
pip install git+https://github.com/NVlabs/nvdiffrast.git
pip install kaolin==0.15.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.0.0_cu118.html
pip install pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py39_cu118_pyt200/download.html
```

### 7.2 Download Demo Data and Model Weights

```bash
mkdir -p demo_data weights
aws s3 cp s3://6dof-pe-dp-bucket/demo_data/ demo_data/ --recursive
aws s3 cp s3://6dof-pe-dp-bucket/weights/ weights/ --recursive
```

### 7.3 Install Azure Kinect SDK

```bash
mkdir -p ~/6DoF_PE_DP/k4a_tools && cd ~/6DoF_PE_DP/k4a_tools

# Install dependency
wget http://ftp.de.debian.org/debian/pool/main/libs/libsoundio/libsoundio1_1.1.0-1_amd64.deb
sudo dpkg -i libsoundio1_1.1.0-1_amd64.deb

# Install Kinect SDK
wget https://packages.microsoft.com/ubuntu/18.04/prod/pool/main/libk/libk4a1.4/libk4a1.4_1.4.1_amd64.deb
wget https://packages.microsoft.com/ubuntu/18.04/prod/pool/main/libk/libk4a1.4-dev/libk4a1.4-dev_1.4.1_amd64.deb
wget https://packages.microsoft.com/ubuntu/18.04/prod/pool/main/k/k4a-tools/k4a-tools_1.4.1_amd64.deb

sudo dpkg -i libk4a1.4_1.4.1_amd64.deb
sudo dpkg -i libk4a1.4-dev_1.4.1_amd64.deb
sudo dpkg -i k4a-tools_1.4.1_amd64.deb
sudo apt-get install -f -y
```

---

## 8️⃣ Reboot & Build CUDA Extensions

### 8.1 Reboot the Instance

```bash
sudo reboot
```

### 8.2 Reconnect & Activate Environment

```bash
ssh -X -i your-key.pem ubuntu@<EC2_PUBLIC_IP>
source ~/6dof_pe_dp/bin/activate
cd ~/6DoF_PE_DP
```

### 8.3 Build CUDA Extensions

```bash
bash build_all_env.sh
```

---

## ✅ Run the Application

```bash
python3 run.py --test_scene_dir demo_data/tless_07 --demo
```

Then open your browser:

```
http://<EC2_PUBLIC_IP>:8050
```

---


## data_folder Structure

```
data_folder/
│
├── configs/
│   ├── camera_extrinsics.json        # Extrinsics camera parameters
│   ├── camera_intrinsics.json        # Intrinsics camera parameters 
│   └── icp_parameters.json           # Parameters for ICP algorithm
│
├── depth/                            # Contains depth images of the scene
│
├── masks/                            # Object masks used for segmentation
│
├── mesh/                             # 3D model files of the object
│   ├── model.obj                     # Original 3D model in OBJ format
│   ├── model.ply                     # 3D model in PLY format
│   └── model_scaled_down.obj         # Downscaled model 
│
├── pcd/                              # Point Cloud Data captured from the scene
│
├── rgb/                              # Captured RGB images
│
└── background/                       # Subfolder containing background.ply
    └── background.ply     

in case of a live run there is no need for these folders: depth, pcd, rgb. 
```
## Features


**capture_background**

To capture the background by recording an empty scene, use the flag `--capture_background True`. Ensure the scene is clear of any objects before running with this flag. You will be prompted to insert the object after capturing the background.


**shorter_side**

The `shorter_side` parameter is automatically set to match the smallest dimension of the depth and color images, helping to reduce resolution. For instance, if the color image resolution is 1280x720 and the depth image resolution is 640x480, `shorter_side` will be set to 480.

This setting can also help manage resource usage. If you encounter low memory availability or CUDA "out of memory" errors, consider manually setting a lower value for `shorter_side` to optimize performance.


**est_refine_iter & track_refine_iter**

These flags define the number of iterations for two key processes:
- **`est_refine_iter`**: Sets the iteration count for initial pose estimation.
- **`track_refine_iter`**: Sets the iteration count for pose tracking.

Increasing these values can improve Pose Estimation for objects that are challenging to detect, such as small items (e.g., `tless_02`).


**debug**

Use the `--debug` flag for in-depth troubleshooting:
- Setting `--debug 2` enables debugging for FoundationPose steps.
- Setting `--debug 3` enables debugging for ICP steps.

## Notes
- This repo assumes objects are pre-scanned and available as 3D meshes.

- The file model_scaled_down.obj has been uniformly scaled down by a factor of 1000.

- In the live run a Kinect Azure is needed, the data_foler should contain the meshes (mesh folder), icp_parameters.json and a mask of the Object.

- Heatmaps can be generated using any external defect detection method, Just Update the get_heatmap function in datareader.py to implement the defect detection method
- AWS Deployment:
   * Use `ssh -X` to enable GUI-based apps like `cv2.imshow`, or fallback to `cv2.imwrite`.
   * Kinect SDK is installed using `.deb` files due to lack of native support on Ubuntu 22.04.
## Acknowledgements

 - [FoundationPose](https://github.com/NVlabs/FoundationPose)
 - [ICP](https://www.open3d.org/docs/release/tutorial/pipelines/icp_registration.html)


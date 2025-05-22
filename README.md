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
This repo assumes objects are pre-scanned and available as 3D meshes.

The file model_scaled_down.obj has been uniformly scaled down by a factor of 1000.

In the live run a Kinect Azure is needed, the data_foler should contain the meshes (mesh folder), icp_parameters.json and a mask of the Object.

Heatmaps can be generated using any external defect detection method, Just Update the get_heatmap function in datareader.py to implement the defect detection method

## Acknowledgements

 - [FoundationPose](https://github.com/NVlabs/FoundationPose)
 - [ICP](https://www.open3d.org/docs/release/tutorial/pipelines/icp_registration.html)


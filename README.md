
# Installation

### Conda environment Setup

```bash
# navigate to the root direcory of the project
cd 2d_3d_pe_dp/

# create conda environment
conda create -n foundationpose python=3.9

# activate conda environment
conda activate foundationpose

# Install Eigen3 3.4.0 under conda environment
conda install conda-forge::eigen=3.4.0
export CMAKE_PREFIX_PATH="$CMAKE_PREFIX_PATH:/eigen/path/under/conda"

# install dependencies
python -m pip install -r requirements.txt

# Install NVDiffRast
python -m pip install --quiet --no-cache-dir git+https://github.com/NVlabs/nvdiffrast.git

# Kaolin (Optional, needed if running model-free setup)
python -m pip install --quiet --no-cache-dir kaolin==0.15.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.0.0_cu118.html

# PyTorch3D
python -m pip install --quiet --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py39_cu118_pyt200/download.html

# Build extensions
CMAKE_PREFIX_PATH=$CONDA_PREFIX/lib/python3.9/site-packages/pybind11/share/cmake/pybind11 bash build_all_conda.sh
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


## Acknowledgements

 - [FoundationPose](https://github.com/NVlabs/FoundationPose)
 - [ICP](https://www.open3d.org/docs/release/tutorial/pipelines/icp_registration.html)


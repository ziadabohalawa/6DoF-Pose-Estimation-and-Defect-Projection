import os
import cv2
import numpy as np
import open3d as o3d
import pykinect_azure as pykinect
from pathlib import Path
import logging
import json
import time


def initialize():
    """
    Initializes the Kinect device with default configuration settings.

    Parameters:
    None

    Returns:
    device (pykinect.PyKinectAzure): The initialized Kinect device.
    device_config (pykinect.K4AConfiguration): The configuration settings for the Kinect device.
    """
    pykinect.initialize_libraries()

    # Modify camera configuration
    device_config = pykinect.default_configuration
    device_config.color_format = pykinect.K4A_IMAGE_FORMAT_COLOR_BGRA32
    device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_720P
    device_config.depth_mode = pykinect.K4A_DEPTH_MODE_NFOV_2X2BINNED
    device = pykinect.start_device(config=device_config)
    time.sleep(1)
    return device, device_config

def get_extrinsics(device, device_config):
    """
    Retrieves the extrinsic parameters between the depth camera and the color camera.

    Returns:
        rotation_cd (numpy.ndarray): Rotation matrix from color to depth camera.
        translation_cd (numpy.ndarray): Translation vector from color to depth camera.
        rotation_dc (numpy.ndarray): Rotation matrix from depth to color camera.
        translation_dc (numpy.ndarray): Translation vector from depth to color camera.
    """
    # Get the calibration
    calibration = device.get_calibration(device_config.depth_mode, device_config.color_resolution)

    # Get the extrinsic parameters
    extrinsics_cd = calibration.extrinsics[pykinect.K4A_CALIBRATION_TYPE_COLOR][pykinect.K4A_CALIBRATION_TYPE_DEPTH]
    rotation_cd = np.array(extrinsics_cd.rotation).reshape(3, 3)
    translation_cd = np.array(extrinsics_cd.translation)

    extrinsics_dc = calibration.extrinsics[pykinect.K4A_CALIBRATION_TYPE_DEPTH][pykinect.K4A_CALIBRATION_TYPE_COLOR]
    rotation_dc = np.array(extrinsics_dc.rotation).reshape(3, 3)
    translation_dc = np.array(extrinsics_dc.translation)

    print(f"Color to Depth Rotation:\n{rotation_cd}")
    print(f"Color to Depth Translation:\n{translation_cd}")
    print(f"Depth to Color Rotation:\n{rotation_dc}")
    print(f"Depth to Color Translation:\n{translation_dc}")

    return rotation_cd, translation_cd, rotation_dc, translation_dc

def save_extrinsics(save_dir, rotation_cd, translation_cd, rotation_dc, translation_dc):
    """Save the extrinsic parameters (rotation and translation matrices) between the color and depth cameras."""
    # Save to JSON file
    extrinsics_data = {
        "color_to_depth": {
            "rotation_matrix": rotation_cd.tolist(),  
            "translation_vector_meters": translation_cd.tolist()
        },
        "depth_to_color": {
            "rotation_matrix": rotation_dc.tolist(),
            "translation_vector_meters": translation_dc.tolist()
        }
    }

    extrinsic_filename = os.path.join(save_dir, 'camera_extrinsics.json')
    with open(extrinsic_filename, "w") as json_file:
        json.dump(extrinsics_data, json_file, indent=4)

    logging.info("Extrinsic parameters saved to extrinsic_parameters.json")

def get_intrinsics(device, depth_mode, color_resolution):
    """
    Retrieves camera calibration and intrinsic parameters for both depth and color cameras.

    Parameters:
    device (pykinect_azure.Device): The Kinect device object.
    depth_mode (pykinect_azure.K4A_DEPTH_MODE): The depth mode for the device.
    color_resolution (pykinect_azure.K4A_COLOR_RESOLUTION): The color resolution for the device.

    Returns:
    tuple: A tuple containing the depth camera intrinsic matrix (depth_K) and the color camera intrinsic matrix (color_K).
    """
    calibration = device.get_calibration(depth_mode=depth_mode, color_resolution=color_resolution)

    depth_intrinsics = calibration.depth_params
    color_intrinsics = calibration.color_params

    depth_K = [
        [depth_intrinsics.fx, 0, depth_intrinsics.cx],
        [0, depth_intrinsics.fy, depth_intrinsics.cy],
        [0, 0, 1]
    ]

    color_K = [
        [color_intrinsics.fx, 0, color_intrinsics.cx],
        [0, color_intrinsics.fy, color_intrinsics.cy],
        [0, 0, 1]
    ]

    return depth_K, color_K

def save_intrinsics(save_dir, depth_K, color_K):
    """Saves the intrinsic parameters of the depth and color cameras to a JSON file."""
    intrinsics = {
        'depth': depth_K,
        'color': color_K
    }

    intrinsic_filename = os.path.join(save_dir, 'camera_intrinsics.json')
    with open(intrinsic_filename, 'w') as f:
        json.dump(intrinsics, f, indent=4)
    logging.info(f"Saved {intrinsic_filename}")
    
def build_pinhole_intrinsics(device, depth_K, color_K):
    """
    Build Open3D PinholeCameraIntrinsic objects for depth and color cameras.

    Parameters:
    - width (int): Image width.
    - height (int): Image height.
    - depth_K (list): Intrinsic matrix for the depth camera (3x3 nested list).
    - color_K (list): Intrinsic matrix for the color camera (3x3 nested list).

    Returns:
    - depth_intrinsics (o3d.camera.PinholeCameraIntrinsic): Intrinsics for the depth camera.
    - color_intrinsics (o3d.camera.PinholeCameraIntrinsic): Intrinsics for the color camera.
    """
    try:
        if device_config.color_resolution == pykinect.K4A_COLOR_RESOLUTION_720P:
            width, height = 1280, 720
        elif device_config.color_resolution == pykinect.K4A_COLOR_RESOLUTION_1080P:
            width, height = 1920, 1080
        else:
            width, height = 1280, 720
    except:
        width, height = 1280, 720
    fx_d = depth_K[0][0]
    fy_d = depth_K[1][1]
    cx_d = depth_K[0][2]
    cy_d = depth_K[1][2]
    depth_intrinsics = o3d.camera.PinholeCameraIntrinsic(
        width, height, fx_d, fy_d, cx_d, cy_d
    )

    fx_c = color_K[0][0]
    fy_c = color_K[1][1]
    cx_c = color_K[0][2]
    cy_c = color_K[1][2]
    color_intrinsics = o3d.camera.PinholeCameraIntrinsic(
        width, height, fx_c, fy_c, cx_c, cy_c
    )

    return depth_intrinsics, color_intrinsics

def get_last_frame_id(save_dir):
    """
    Retrieves the ID of the last frame saved in the specified directory.

    Parameters:
    save_dir (str): The directory where the RGB images are saved.

    Returns:
    int: The ID of the last frame saved in the directory. If no RGB images are found, returns -1.
    """
    rgb_files = list(Path(save_dir).glob('rgb_*.png'))
    if not rgb_files:
        return -1

    last_file = max(rgb_files, key=lambda x: x.name)

    return int(last_file.stem.split('_')[-1])

def save_info_json(save_dir, color_k_matrix):
    """
    Saves the intrinsic matrix (K) of the color camera to info.json file.

    Parameters:
    save_dir (str): The directory where the JSON file will be saved.
    color_k_matrix (numpy.ndarray): The intrinsic matrix of the color camera.

    Returns:
    None
    """
    rgb_files = list(Path(save_dir).glob('rgb_*.png'))

    info = {}
    for file in rgb_files:
        filename = os.path.basename(file)

        info[filename] = {
            'K': color_k_matrix.tolist()  
        }

    info_filename = os.path.join(str(save_dir), 'info.json')
    with open(info_filename, 'w') as f:
        json.dump(info, f, indent=4)
    logging.info(f"Saved {info_filename}")

def capture_frame(device):
    """
    Captures a single frame from the device and returns color_image, depth_image, points.

    Parameters:
    device (pykinect_azure.Device): The Kinect device object used for capturing images and point clouds.

    Returns:
    color_image (numpy.ndarray): The color image captured from the device.
    depth_image (numpy.ndarray): The depth image captured from the device.
    points (numpy.ndarray): The 3D points captured from the device.

    Raises:
    logging.error: If the color image, depth image, or point cloud could not be captured.
    """
    capture = device.update()

    ret_depth, depth_image = capture.get_depth_image()
    ret_color, color_image = capture.get_color_image()
    ret_points, points = capture.get_pointcloud()

    while not ret_color or not ret_depth or not ret_points:
        logging.error("Failed to get image or point cloud.")
        ret_depth, depth_image = capture.get_depth_image()
        ret_color, color_image = capture.get_color_image()
        ret_points, points = capture.get_pointcloud()

    return color_image, depth_image, points

def save_frame(color_image, depth_image, point_cloud, save_dir, frame_id):
    """Saves the color image, depth image, and point cloud to disk."""
    color_image_filename = os.path.join(save_dir, f'rgb_{frame_id:03d}.png')
    cv2.imwrite(color_image_filename, cv2.cvtColor(color_image, cv2.COLOR_BGRA2BGR))
    logging.info(f"Saved {color_image_filename}")

    depth_image_filename = os.path.join(save_dir, f'depth_{frame_id:03d}.png')
    cv2.imwrite(depth_image_filename, depth_image)
    logging.info(f"Saved {depth_image_filename}")

    o3d_point_cloud = o3d.geometry.PointCloud()
    o3d_point_cloud.points = o3d.utility.Vector3dVector(point_cloud)

    point_cloud_filename = os.path.join(save_dir, f'cloud_{frame_id:03d}.ply')
    o3d.io.write_point_cloud(point_cloud_filename, o3d_point_cloud)
    logging.info(f"Saved {point_cloud_filename}")

def capture_new_background(device):
    """
    Captures the background (empty box), performs a countdown, and saves the point cloud to a file.
    
    Args:
        device: The capture device (e.g., Kinect or another sensor).
    
    Returns:
        The background point cloud object.
    """
    logging.info("Please make sure the box is empty.")
    countdown(5, message="Capturing empty box in")
    color_image, depth_image, points = capture_frame(device)
    
    background_pcd = create_point_cloud(points)
    
    save_path = "data/tmp/background/box.ply"
    os.makedirs(os.path.dirname(save_path), exist_ok=True) 
    o3d.io.write_point_cloud(save_path, background_pcd)
    logging.info(f"Background point cloud captured and saved to {save_path}")
    
    return background_pcd
 
def process_depth_image(depth_image):
    """Converts the depth image to 16-bit format if not already."""
    if depth_image.dtype != np.uint16:
        depth_image.astype(np.uint16) 
    return depth_image

def create_point_cloud(points):
    """Creates an Open3D point cloud object from 3D points."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd
 
def display_color_images(color_image):
    """Displays color images."""
    cv2.imshow("Live RGB Feed", cv2.cvtColor(color_image, cv2.COLOR_BGRA2BGR))
    cv2.waitKey(1)
    
def display_depth_images(depth_image):
    """Displays depth images."""
    cv2.imshow('Depth Image', depth_image)
    cv2.waitKey(1)

def initialize_visualizer(pcd):
    """Initializes and returns an Open3D visualizer along with an empty point cloud."""
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.get_render_option().background_color = [0.1, 0.1, 0.1] 
    vis.get_render_option().point_size = 1
    return vis, pcd

def display_pcd(points, vis, pcd):
    """Updates and displays point cloud using Open3D."""
    if len(points) == 0:
        return  
    valid_points = points[np.all(points != 0, axis=1)]
    if len(valid_points) > 0:
        pcd.points = o3d.utility.Vector3dVector(valid_points)
        vis.update_geometry(pcd)
        vis.poll_events()
        vis.update_renderer()
   
def countdown(seconds, message="Resuming in"):
    """Displays a countdown timer."""
    for i in range(seconds, 0, -1):
        logging.info(f"{message} {i} seconds...")
        time.sleep(1)
    logging.info("\n")

def handle_pause(frame_count, start_frame, interval, dim_frame, dim_interval):
    """Handles the pause logic during data capture."""
    if (frame_count - start_frame + 1) % dim_frame == 0:
        logging.info("\nDIM LIGHT - Pausing for 20 seconds...")
        countdown(dim_interval, message="Resuming in")
        logging.info("Resuming capture...                ")
    else:
        # Wait for the specified interval
        countdown(interval, message="Next capture in")
        logging.info("Continuing capture...                ")

def capture_save(device, base_dir, frame_count=1):
    """
    Captures a frame and saves it to disk.

    Parameters:
    device (pykinect_azure.Device): The Kinect device object used for capturing images and point clouds.
    base_dir (str): The base directory where the captured data will be saved.
    frame_count (int, optional): The frame count for the captured data. Default is 1.

    Returns:
    bool: True if the capture and save are successful, False otherwise.
    """
    color_image, depth_image, points = capture_frame(device)
    if color_image is None or depth_image is None or points is None:
        logging.error("Failed to capture image or point cloud.")
        return False  
    display_color_images(color_image)
    save_frame(color_image, depth_image, points, base_dir, frame_count)
    return True  

def pvnet_data_capture(device, save_dir, total_captures, interval, dim_light_frame, dim_interval):
    """
    Captures and saves RGB, depth, and point cloud data for generating a PVNet dataset.

    This function captures and saves essential data (RGB images, depth images, and point clouds) using a Kinect Azure device.
    The captured data is formatted for later conversion into the PVNet dataset format using the 'linemod-transform-main' tool,
    which is commonly used for object pose estimation. The extrinsic and intrinsic parameters of the camera are saved, 
    along with the captured images and point clouds, in the specified directory.

    Parameters:
    device (pykinect_azure.Device): The Kinect Azure device object used for capturing RGB, depth, and point cloud data.
    dir (str): The directory where the captured data will be saved for conversion into the PVNet dataset format.
    total_captures (int): The total number of frames to capture during the data collection process.
    interval (int): The interval in seconds between consecutive captures to allow for movement or adjustment of the object.
    dim_light_frame (int): The frame number after which the light will be dimmed to simulate different lighting conditions.
    dim_interval (int): The interval in seconds at which the light will be dimmed for specific frames.

    Returns:
    None
    """
    rotation_cd, translation_cd, rotation_dc, translation_dc = get_extrinsics()
    save_extrinsics(save_dir, rotation_cd, translation_cd, rotation_dc, translation_dc)

    device, device_config = initialize()

    depth_intrinsics, color_intrinsics = get_intrinsics(device, device_config.depth_mode, device_config.color_resolution)
    save_intrinsics(save_dir, depth_intrinsics, color_intrinsics)

    logging.info("Starting data capture...")

    last_frame_id = get_last_frame_id(save_dir)
    start_frame = last_frame_id + 1

    cv2.namedWindow("Live RGB Feed", cv2.WINDOW_NORMAL)

    for frame_count in range(start_frame, start_frame + total_captures):
        color_image, depth_image, points = capture_frame(device)
        save_frame(color_image, depth_image, points, save_dir, frame_count)
        logging.info(f"Captured and saved frame {frame_count}/{start_frame + total_captures - 1}")
        display_color_images(color_image)

        handle_pause(frame_count, start_frame, interval, dim_light_frame, dim_interval)

    save_info_json(save_dir, color_intrinsics)

    logging.info("Data capture complete.")
    
def continuous_capture(device):
    """
    Continuously captures and displays images and point clouds.

    Parameters:
    device (pykinect_azure.Device): The Kinect device object used for capturing images and point clouds.
    
    The function continuously captures color and depth images, as well as 3D points, from the Kinect device.
    It then displays the point cloud, color image, and depth image using the Open3D visualizer, OpenCV, and
    waits for a key press. If the 'q' key is pressed, the function breaks the loop and stops capturing.
    """
    points = [[0, 0, 0]]
    pcd = create_point_cloud(points)
    vis, pcd = initialize_visualizer(pcd)  
    while True:
        color_image, depth_image, points = capture_frame(device)
        if color_image is None or depth_image is None or points is None:
            continue
        display_pcd(points, vis, pcd)
        display_color_images(color_image)  
        if cv2.waitKey(1) == ord('q'):
            break

if __name__ == "__main__":
    base_dir = '/home/z/BA/ICP/dataset'
    os.makedirs(base_dir, exist_ok=True)

    total_captures = 25  # Adjust the number of frames you want to capture
    interval = 7  # Interval in seconds between captures
    dim_light_frame = 10 # dim light each dim_light_frame frames for dim_interval
    dim_interval = 15 # seconds
    
    device, device_config = initialize()
    time.sleep(2)
    try:
        # get_extrinsics()
        # pvnet_data_capture(device, base_dir, total_captures, interval,dim_light_frame, dim_interval)
        # capture_save(device, BASE_DIR)
        # continuous_capture(device)
        rot, tran, rot1, tran1 = get_extrinsics(device, device_config)
        print(f"{rot},\n{tran},\n, {rot1},\n{tran1},\n")
    finally:
        device.stop_cameras()
        device.close()
        cv2.destroyAllWindows()
        logging.info("Device closed and windows destroyed.")

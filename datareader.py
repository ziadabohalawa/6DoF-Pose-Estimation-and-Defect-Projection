from Utils import *
import json,os,sys
from pathlib import Path
import pykinect_azure as pykinect
pykinect.initialize_libraries()

BOP_LIST = ['lmo','tless','ycbv','hb','tudl','icbin','itodd']
BOP_DIR = os.getenv('BOP_DIR')

class KinectReader:
    def __init__(self, base_dir, capture_background=False, downscale=1, shorter_side=None, zfar=np.inf, arguments=None):
        self.base_dir = base_dir
        self.downscale = downscale
        self.zfar = zfar
        self.file_id = 0
        self.color_files = []  # Placeholder for compatibility
        self.id_strs = []      # Placeholder for compatibility
        self.parameters = self.update_config(arguments)    
        # Initialize the Kinect device
        self.device, self.device_config = self.initialize()

        # Retrieve camera intrinsics
        self.get_intrinsics()
        self.get_extrinsics()

        if shorter_side is None:
          shorter_side = min(self.color_H, self.color_W, self.depth_H, self.depth_W)
        elif shorter_side is not None:
          shorter_side = arguments.shorter_side
            
        self.downscale = shorter_side / min(self.color_H, self.color_W)

        self.color_H = int(self.color_H * self.downscale)
        self.color_W = int(self.color_W * self.downscale)
        self.color_K = np.array(self.color_K)
        self.depth_K = np.array(self.depth_K)
        self.color_K[:2] *= self.downscale
        self.depth_K[:2] *= self.downscale # Maybe we dont need it anymore

        # Placeholder for last captured data
        self.last_color = None
        self.last_depth = None
        self.last_points = None

        # Initialize background point cloud
        self.capture_background = capture_background
        self.get_background()
        
        self.get_target()
        
    def initialize(self):
        """
        Initializes the Kinect device with default configuration settings.
        """
        # Modify camera configuration
        device_config = pykinect.default_configuration
        device_config.color_format = pykinect.K4A_IMAGE_FORMAT_COLOR_BGRA32
        device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_720P
        device_config.depth_mode = pykinect.K4A_DEPTH_MODE_NFOV_2X2BINNED
        device = pykinect.start_device(config=device_config)
        time.sleep(1)
        return device, device_config

    def stop_camera(self):
      self.device.stop_cameras()
      self.device.close()
  
    def update_config(self, args):
      """Update the configuration based on command-line arguments."""
      config = self.get_parameters()
      if args.debug >= 3:
          config['debug_vis'] = True
      if args.box is not None:
          config['box'] = args.box
      if args.mesh is not None:
          config['mesh'] = args.mesh
      if args.voxel_size is not None:
          config['voxel_size'] = args.voxel_size
      return config
    
    def get_video_name(self):
        return "KinectLiveStream"

    def __len__(self):
        return float('inf')

    def get_gt_pose(self, i):
        logging.info("GT pose not available for live data")
        return None

    def update(self):
        """
        Captures a new frame from the device and updates the internal state.
        """
        color_image, depth_image, points = self.capture_frame()
        self.last_color = color_image
        self.last_depth = depth_image
        self.last_points = points
        self.file_id += 1

    def get_parameters(self):
        """
        Retrieves ICP parameters from a configuration file.
        """
        with open(f'{self.base_dir}/configs/icp_parameters.json', 'r') as file:
            param = json.load(file)
        return param

    def get_color(self, i=None):
        if self.last_color is not None:
            color = self.last_color[..., :3]  
            color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
            color = cv2.resize(color, (self.color_W, self.color_H), \
              interpolation=cv2.INTER_NEAREST)
            return color
        else:
            logging.warning("No color image captured yet.")
            return None

    def get_mask(self, color_image, i=None):
      try:
          mask_path = f"{self.base_dir}/masks/0000.png"
          
          if not os.path.exists(mask_path):
              raise FileNotFoundError("Mask file not found")
          
          mask = cv2.imread(mask_path, -1)
          
          if len(mask.shape) == 3:
              for c in range(3):
                  if mask[...,c].sum() > 0:
                      mask = mask[...,c]
                      break
          
          mask = cv2.resize(mask, (self.color_W, self.color_H), 
                            interpolation=cv2.INTER_NEAREST).astype(bool).astype(np.uint8)
          
          return mask
      
      except (FileNotFoundError, AttributeError, TypeError):
          try:
              gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
              
              _, binary_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
              inverted_mask = cv2.bitwise_not(binary_mask)
              
              kernel = np.ones((3,3), np.uint8)
              refined_mask = inverted_mask
              
              refined_mask = cv2.morphologyEx(refined_mask, cv2.MORPH_OPEN, kernel, iterations=2)
              refined_mask = cv2.morphologyEx(refined_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
              
              cv2.imwrite(f"{self.base_dir}/masks/0000.png", refined_mask.astype(np.uint8) * 255)
              mask = cv2.resize(refined_mask, (self.color_W, self.color_H), 
                                interpolation=cv2.INTER_NEAREST).astype(bool).astype(np.uint8)
              return mask
          
          except Exception as e:
              print(f"Error generating mask: {e}")
              return np.zeros((self.color_H, self.color_W), dtype=np.uint8)

    def get_heatmap(self, color_image):
        npy_file = f"{self.base_dir}/heatmap/0002.npy"
        heatmap_data = np.load(npy_file)
        heatmap_size = heatmap_data.shape[0]
        
        scale = heatmap_size / min(color_image.shape[:2])
        new_height = int(color_image.shape[0] * scale)
        new_width = int(color_image.shape[1] * scale)
        
        color_resized = cv2.resize(color_image, \
          (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        start_y = (new_height - heatmap_size) // 2
        start_x = (new_width - heatmap_size) // 2
        color_cropped = color_resized[start_y:start_y+heatmap_size, \
          start_x:start_x+heatmap_size]
        
        heatmap = heatmap_data - np.min(heatmap_data)
        heatmap = heatmap / np.max(heatmap)
        output_size = min(int(self.color_H/self.downscale), \
          int(self.color_W/self.downscale))
        
        heatmap_vis = cv2.resize(heatmap, (output_size, output_size), \
          interpolation=cv2.INTER_LINEAR)
        color_original = cv2.resize(color_cropped, (output_size, output_size),\
          interpolation=cv2.INTER_NEAREST)
        
        heatmap_full = np.zeros((int(self.color_H/self.downscale), \
          int(self.color_W/self.downscale)))
        y_start = (int(self.color_H/self.downscale) - output_size) // 2
        x_start = (int(self.color_W/self.downscale) - output_size) // 2
        heatmap_full[y_start:y_start+output_size, x_start:x_start+output_size] \
          = heatmap_vis
        return heatmap_full, color_original, heatmap_vis, color_original

    def load_point_cloud(self, filepath):
        """
        Loads a point cloud from a file.
        """
        return o3d.io.read_point_cloud(filepath)

    def load_depth_image(self, filepath):
        """
        Loads a depth image from a file.
        """
        return cv2.imread(filepath, cv2.IMREAD_UNCHANGED)

    def load_color_image(self, filepath):
        """
        Loads a color image from a file.
        """
        return cv2.imread(filepath, cv2.IMREAD_COLOR)

    def get_extrinsics(self):
        """
        Retrieves the extrinsic parameters between the depth camera and the color camera.
        """
        with open(f"{self.base_dir}/configs/camera_extrinsics.json", "r") as file:
            transformation_data = json.load(file)

        # Extract the rotation matrix and translation vector
        rotation_matrix = np.array(transformation_data["color_to_depth"]["rotation_matrix"])
        translation_vector = np.array(transformation_data["color_to_depth"]["translation_vector"]).reshape(3, 1)

        # Construct the 4x4 transformation matrix
        self.color_to_depth = np.eye(4)  # Initialize a 4x4 identity matrix
        self.color_to_depth[:3, :3] = rotation_matrix  # Top-left 3x3 block is the rotation matrix
        self.color_to_depth[:3, 3] = translation_vector.flatten()  # Top-right 3x1 block is the translation vector
        self.inverse_color_to_depth = np.linalg.inv(self.color_to_depth)
        
        # Extract the rotation matrix and translation vector
        rotation_matrix = np.array(transformation_data["depth_to_color"]["rotation_matrix"])
        translation_vector = np.array(transformation_data["depth_to_color"]["translation_vector"]).reshape(3, 1)

        # Construct the 4x4 transformation matrix
        self.depth_to_color = np.eye(4)  # Initialize a 4x4 identity matrix
        self.depth_to_color[:3, :3] = rotation_matrix  # Top-left 3x3 block is the rotation matrix
        self.depth_to_color[:3, 3] = translation_vector.flatten()  # Top-right 3x1 block is the translation vector
        self.inverse_depth_to_color = np.linalg.inv(self.depth_to_color) 

    def get_intrinsics(self):
        """
        Retrieves camera calibration and intrinsic parameters for both depth and color cameras.
        """
        calibration = self.device.get_calibration(
            self.device_config.depth_mode, self.device_config.color_resolution
        )

        depth_intrinsics = calibration.depth_params
        color_intrinsics = calibration.color_params

        self.depth_K = [
            [depth_intrinsics.fx, 0, depth_intrinsics.cx],
            [0, depth_intrinsics.fy, depth_intrinsics.cy],
            [0, 0, 1]
        ]

        self.color_K = [
            [color_intrinsics.fx, 0, color_intrinsics.cx],
            [0, color_intrinsics.fy, color_intrinsics.cy],
            [0, 0, 1]
        ]

        color_resolution_dict = {
            0: None,            # K4A_COLOR_RESOLUTION_OFF
            1: (1280, 720),     # K4A_COLOR_RESOLUTION_720P
            2: (1920, 1080),    # K4A_COLOR_RESOLUTION_1080P
            3: (2560, 1440),    # K4A_COLOR_RESOLUTION_1440P
            4: (2048, 1536),    # K4A_COLOR_RESOLUTION_1536P
            5: (3840, 2160),    # K4A_COLOR_RESOLUTION_2160P
            6: (4096, 3072)     # K4A_COLOR_RESOLUTION_3072P
        }

        depth_mode_dict = {
            0: None,            # K4A_DEPTH_MODE_OFF
            1: (320, 288),      # K4A_DEPTH_MODE_NFOV_2X2BINNED
            2: (640, 576),      # K4A_DEPTH_MODE_NFOV_UNBINNED
            3: (512, 512),      # K4A_DEPTH_MODE_WFOV_2X2BINNED
            4: (1024, 1024),    # K4A_DEPTH_MODE_WFOV_UNBINNED
            5: (1024, 1024)     # K4A_DEPTH_MODE_PASSIVE_IR
        }

        color_dimensions = color_resolution_dict.get(self.device_config.color_resolution)
        depth_dimensions = depth_mode_dict.get(self.device_config.depth_mode)

        self.depth_H = depth_dimensions[1]
        self.depth_W = depth_dimensions[0]
        self.color_H = color_dimensions[1]
        self.color_W = color_dimensions[0]

        self.depth_pinhole = self.build_pinhole_intrinsics(
            self.depth_W, self.depth_H, self.depth_K)
        self.color_pinhole = self.build_pinhole_intrinsics(
            self.color_W, self.color_H, self.color_K)

    def build_pinhole_intrinsics(self, width, height, K):
        return o3d.camera.PinholeCameraIntrinsic(
            width, height, K[0][0], K[1][1], K[0][2], K[1][2])

    def get_source(self, i=None):
        """
        Returns the point cloud from the last captured frame.
        """
        if self.last_points is not None:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(self.last_points)
            return pcd
        else:
            logging.warning("No point cloud captured yet.")
            return None

    def get_background(self):
        """
        Captures the background point cloud if not already captured.
        """
        if self.capture_background:
            self.background = self.capture_new_background()
        else:
          self.background = self.load_point_cloud(f'{self.base_dir}/background/box.ply')

    def get_target(self):
      """
      Loads the target model (object mesh and point cloud).
      """
      self.target_mesh = o3d.io.read_triangle_mesh(f"{self.base_dir}/mesh/model.obj")
      self.target_mesh.compute_vertex_normals()
      self.target = self.load_point_cloud(f'{self.base_dir}/mesh/model.ply')

    def get_initial_pose(self):
            return np.eye(4)  

    def scale_translation_to_millimeters(self, pose):
        transformation_scaled = pose.copy()
        transformation_scaled[:3, -1] *= 1000  
        return transformation_scaled  

    def get_depth(self, i=None):
        if self.last_depth is not None:
            depth = self.last_depth.astype(np.float32) / 1e3  # Convert to meters
            depth = cv2.resize(depth, \
              (self.color_W, self.color_H), interpolation=cv2.INTER_NEAREST)
            depth[(depth < 0.001) | (depth >= self.zfar)] = 0
            return depth
        else:
            logging.warning("No depth image captured yet.")
            return None

    def capture_frame(self):
        """
        Captures a single frame from the device and returns color_image, depth_image, points.
        """
        capture = self.device.update()

        ret_depth, depth_image = capture.get_depth_image()
        ret_color, color_image = capture.get_color_image()
        ret_points, points = capture.get_pointcloud()

        while not ret_color or not ret_depth or not ret_points:
            logging.error("Failed to get image or point cloud.")
            capture = self.device.update()
            ret_depth, depth_image = capture.get_depth_image()
            ret_color, color_image = capture.get_color_image()
            ret_points, points = capture.get_pointcloud()

        return color_image, depth_image, points

    def capture_new_background(self):
        """
        Captures the background (empty scene), performs a countdown, and returns the point cloud.
        """
        logging.info("Please make sure the scene is empty.")
        self.countdown(5, message="Capturing background in")
        color_image, depth_image, points = self.capture_frame()
        background = self.create_point_cloud(points)

        save_path = f"{self.base_dir}/background/box.ply"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        o3d.io.write_point_cloud(save_path, background)
        logging.info(f"Background point cloud captured and saved to {save_path}")
        logging.info("Please put the object in the Box.")
        self.countdown(5, message="Capturing object in")
        return background

    def create_point_cloud(self, points):
        """
        Creates an Open3D point cloud object from 3D points.
        """
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        return pcd

    def countdown(self, seconds, message=""):
        """
        Displays a countdown in the console.
        """
        for i in range(seconds, 0, -1):
            print(f"{message} {i} seconds...")
            time.sleep(1)
        print("Capturing now...")

    def process_depth_image(self, depth_image):
        """
        Converts the depth image to 16-bit format if not already.
        """
        if depth_image.dtype != np.uint16:
            depth_image = depth_image.astype(np.uint16)
        return depth_image

    def save_extrinsics(self):
        """
        Saves the extrinsic parameters (rotation and translation matrices) between the color and depth cameras.
        """
        extrinsics_data = {
            "color_to_depth": {
                "rotation_matrix": self.rotation_cd.tolist(),
                "translation_vector_meters": self.translation_cd.tolist()
            },
            "depth_to_color": {
                "rotation_matrix": self.rotation_dc.tolist(),
                "translation_vector_meters": self.translation_dc.tolist()
            }
        }

        extrinsic_filename = os.path.join(self.base_dir, '/configs/camera_extrinsics.json')
        with open(extrinsic_filename, "w") as json_file:
            json.dump(extrinsics_data, json_file, indent=4)

        logging.info(f"Extrinsic parameters saved to {extrinsic_filename}")

    def save_intrinsics(self, save_dir):
        """
        Saves the intrinsic parameters of the depth and color cameras to a JSON file.
        """
        intrinsics = {
            'depth': {
                'fx': self.depth_K[0][0],
                'fy': self.depth_K[1][1],
                'cx': self.depth_K[0][2],
                'cy': self.depth_K[1][2],
                'width': self.depth_W,
                'height': self.depth_H
            },
            'color': {
                'fx': self.color_K[0][0],
                'fy': self.color_K[1][1],
                'cx': self.color_K[0][2],
                'cy': self.color_K[1][2],
                'width': self.color_W,
                'height': self.color_H
            }
        }

        intrinsic_filename = os.path.join(save_dir, 'camera_intrinsics.json')
        with open(intrinsic_filename, 'w') as f:
            json.dump(intrinsics, f, indent=4)
        logging.info(f"Intrinsic parameters saved to {intrinsic_filename}")

    def save_frame(self, color_image, depth_image, point_cloud, save_dir, frame_id):
        """
        Saves the color image, depth image, and point cloud to disk.
        """
        color_image_filename = os.path.join(save_dir, f'rgb_{frame_id:03d}.png')
        cv2.imwrite(color_image_filename, cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB))
        logging.info(f"Saved {color_image_filename}")

        depth_image_filename = os.path.join(save_dir, f'depth_{frame_id:03d}.png')
        cv2.imwrite(depth_image_filename, depth_image)
        logging.info(f"Saved {depth_image_filename}")

        o3d_point_cloud = self.create_point_cloud(point_cloud)

        point_cloud_filename = os.path.join(save_dir, f'cloud_{frame_id:03d}.ply')
        o3d.io.write_point_cloud(point_cloud_filename, o3d_point_cloud)
        logging.info(f"Saved {point_cloud_filename}")

    def get_last_frame_id(self, save_dir):
        """
        Retrieves the ID of the last frame saved in the specified directory.
        """
        rgb_files = list(Path(save_dir).glob('rgb_*.png'))
        if not rgb_files:
            return -1
        last_file = max(rgb_files, key=lambda x: x.name)

        return int(last_file.stem.split('_')[-1])

    def save_info_json(self, save_dir):
        """
        Saves the intrinsic matrix (K) of the color camera to info.json file.
        """
        rgb_files = list(Path(save_dir).glob('rgb_*.png'))

        info = {}
        for file in rgb_files:
            filename = os.path.basename(file)

            info[filename] = {
                'K': self.color_K.tolist()  
            }

        info_filename = os.path.join(str(save_dir), 'info.json')
        with open(info_filename, 'w') as f:
            json.dump(info, f, indent=4)
        logging.info(f"Saved {info_filename}")


class DataReader:
  def __init__(self,base_dir, downscale=1, shorter_side=None, zfar=np.inf, arguments=None):
    self.base_dir = base_dir
    self.downscale = downscale
    self.zfar = zfar
    self.color_files = sorted(glob.glob(f"{self.base_dir}/rgb/*.png"))
    self.file_id = 0
    self.parameters = self.update_config(arguments)    
    self.get_intrinsics()
    self.get_extrinsics()

    self.color_K = np.array(self.color_K)
    self.id_strs = []
    for color_file in self.color_files:
      id_str = os.path.basename(color_file).replace('.png','')
      self.id_strs.append(id_str)
    self.color_H, self.color_W = cv2.imread(self.color_files[0]).shape[:2]
    self.depth_H, self.depth_W = cv2.imread(self.color_files[0].replace('rgb','depth'),-1).shape[:2]
    
    if shorter_side is None:
      shorter_side = min(self.color_H, self.color_W, self.depth_H, self.depth_W)
    elif shorter_side is not None:
      shorter_side = arguments.shorter_side
        
    self.downscale = shorter_side / min(self.color_H, self.color_W)
    logging.info(f"shorter_side: {shorter_side}")
    logging.info(f"Downscaling factor: {self.downscale}")
    
    self.color_H = int(self.color_H*self.downscale)
    self.color_W = int(self.color_W*self.downscale)
    self.color_K[:2] *= self.downscale
    self.get_background()
    self.get_target()
    
    self.gt_pose_files = sorted(glob.glob(f'{self.base_dir}/annotated_poses/*'))

    self.videoname_to_object = {
      'bleach0': "021_bleach_cleanser",
      'bleach_hard_00_03_chaitanya': "021_bleach_cleanser",
      'cracker_box_reorient': '003_cracker_box',
      'cracker_box_yalehand0': '003_cracker_box',
      'mustard0': '006_mustard_bottle',
      'mustard_easy_00_02': '006_mustard_bottle',
      'sugar_box1': '004_sugar_box',
      'sugar_box_yalehand0': '004_sugar_box',
      'tomato_soup_can_yalehand0': '005_tomato_soup_can',
    }

  def update_config(self, args):
    """Update the configuration based on command-line arguments."""
    config = self.get_parameters()
    if args.debug >= 3:
        config['debug_vis'] = True
    if args.box is not None:
        config['box'] = args.box
    if args.mesh is not None:
        config['mesh'] = args.mesh
    if args.voxel_size is not None:
        config['voxel_size'] = args.voxel_size
    return config

  def get_video_name(self):
    return self.base_dir.split('/')[-1]

  def __len__(self):
    return len(self.color_files)

  def get_gt_pose(self,i=0):
    try:
      pose = np.loadtxt(self.gt_pose_files[i]).reshape(4,4)
      return pose
    except:
      logging.info("GT pose not found, return None")
      return None

  def update(self):
    return
    
  def get_parameters(self):    
    with open(f'{self.base_dir}/configs/icp_parameters.json', 'r') as file:
      param = json.load(file)
    return param
      
  def get_color(self,i=0):
    color = imageio.imread(self.color_files[i])[...,:3]
    color = cv2.resize(color, (self.color_W,self.color_H), \
      interpolation=cv2.INTER_NEAREST)
    return color

  def get_mask(self, color_image, i=None):
      try:
          mask_path = f"{self.base_dir}/masks/0000.png"
          
          if not os.path.exists(mask_path):
              raise FileNotFoundError("Mask file not found")
          
          mask = cv2.imread(mask_path, -1)
          
          if len(mask.shape) == 3:
              for c in range(3):
                  if mask[...,c].sum() > 0:
                      mask = mask[...,c]
                      break
          
          mask = cv2.resize(mask, (self.color_W, self.color_H), 
                            interpolation=cv2.INTER_NEAREST).astype(bool).astype(np.uint8)
          
          return mask
      
      except (FileNotFoundError, AttributeError, TypeError):
          try:
              gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
              
              _, binary_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
              inverted_mask = cv2.bitwise_not(binary_mask)
              
              kernel = np.ones((3,3), np.uint8)
              refined_mask = inverted_mask
              
              refined_mask = cv2.morphologyEx(refined_mask, cv2.MORPH_OPEN, kernel, iterations=2)
              refined_mask = cv2.morphologyEx(refined_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
              
              cv2.imwrite(f"{self.base_dir}/masks/0000.png", refined_mask.astype(np.uint8) * 255)
              mask = cv2.resize(refined_mask, (self.color_W, self.color_H), 
                                interpolation=cv2.INTER_NEAREST).astype(bool).astype(np.uint8)
              return mask
          
          except Exception as e:
              print(f"Error generating mask: {e}")
              return np.zeros((self.color_H, self.color_W), dtype=np.uint8)

  def get_heatmap(self, color_image):
      npy_file = f"{self.base_dir}/heatmap/0002.npy"
      heatmap_data = np.load(npy_file)
      
      heatmap_size = heatmap_data.shape[0]
      
      scale = heatmap_size / min(color_image.shape[:2])
      
      new_height = int(color_image.shape[0] * scale)
      new_width = int(color_image.shape[1] * scale)
      
      color_resized = cv2.resize(color_image, (new_width, new_height), \
        interpolation=cv2.INTER_AREA)
      
      start_y = (new_height - heatmap_size) // 2
      start_x = (new_width - heatmap_size) // 2
      color_cropped = color_resized[start_y:start_y+heatmap_size, \
        start_x:start_x+heatmap_size]
      
      heatmap = heatmap_data - np.min(heatmap_data)
      heatmap = heatmap / np.max(heatmap)
      
      output_size = min(int(self.color_H/self.downscale), \
        int(self.color_W/self.downscale))
      
      heatmap_vis = cv2.resize(heatmap, (output_size, output_size), \
        interpolation=cv2.INTER_LINEAR)
      color_original = cv2.resize(color_cropped, \
        (output_size, output_size), interpolation=cv2.INTER_NEAREST)
      
      heatmap_full = np.zeros((int(self.color_H/self.downscale), \
        int(self.color_W/self.downscale)))
      y_start = (int(self.color_H/self.downscale) - output_size) // 2
      x_start = (int(self.color_W/self.downscale) - output_size) // 2
      heatmap_full[y_start:y_start+output_size, x_start:x_start+output_size] = \
        heatmap_vis
      return heatmap_full, color_original, heatmap_vis, color_original
    
  def load_point_cloud(self, filepath):
      return o3d.io.read_point_cloud(filepath)

  def load_depth_image(self, filepath):
      return cv2.imread(filepath, cv2.IMREAD_UNCHANGED)

  def load_color_image(self, filepath):
      return cv2.imread(filepath, cv2.IMREAD_COLOR)

  def get_intrinsics(self):
      with open(f'{self.base_dir}/configs/camera_intrinsics.json', 'r') as file:
          intrinsics_data = json.load(file)
      self.depth_K = [
        [intrinsics_data["depth"]["fx"], 0, intrinsics_data["depth"]["cx"]],
        [0, intrinsics_data["depth"]["fy"], intrinsics_data["depth"]["cy"]],
        [0, 0, 1]
        ]
      self.color_K = [
        [intrinsics_data["color"]["fx"], 0, intrinsics_data["color"]["cx"]],
        [0, intrinsics_data["color"]["fy"], intrinsics_data["color"]["cy"]],
        [0, 0, 1]
        ]
      self.depth_H = intrinsics_data["depth"]["height"]
      self.depth_W = intrinsics_data["depth"]["width"]
      self.color_H = intrinsics_data["color"]["height"]
      self.color_W = intrinsics_data["color"]["width"]
      self.depth_pinhole = \
        self.build_pinhole_intrinsics(self.depth_W, self.depth_H, self.depth_K)
      self.color_pinhole = \
        self.build_pinhole_intrinsics(self.color_W , self.color_H , self.color_K)
      
  def build_pinhole_intrinsics(self, width, height, K):
      return o3d.camera.PinholeCameraIntrinsic(width, \
        height, K[0][0], K[1][1], K[0][2], K[1][2])

  def get_source(self, i=0):
    pcd_path = self.color_files[i].replace('/rgb/', '/pcd/').replace('.png', '.ply').replace('/rgb_', '/cloud_')
    return self.load_point_cloud(pcd_path)

  def get_background(self):
      self.background = \
        self.load_point_cloud(f'{self.base_dir}/background/box.ply')

  def get_target(self):
      self.target_mesh =\
        o3d.io.read_triangle_mesh(f"{self.base_dir}/mesh/model.obj")
      self.target_mesh.compute_vertex_normals()
      self.target = self.load_point_cloud(f'{self.base_dir}/mesh/model.ply')

  def get_initial_pose(self):
      return np.eye(4)  
  def scale_translation_to_millimeters(self, pose):
      transformation_scaled = pose.copy()
      transformation_scaled[:3, -1] *= 1000  
      return transformation_scaled
    
  def get_extrinsics(self):
      
      with open(f"{self.base_dir}/configs/camera_extrinsics.json", "r") as file:
          transformation_data = json.load(file)

      rotation_matrix = \
        np.array(transformation_data["color_to_depth"]["rotation_matrix"])
      translation_vector = \
        np.array(transformation_data["color_to_depth"]["translation_vector"]).reshape(3, 1)

      self.color_to_depth = np.eye(4)  
      self.color_to_depth[:3, :3] = rotation_matrix  
      self.color_to_depth[:3, 3] = translation_vector.flatten()  
      self.inverse_color_to_depth = np.linalg.inv(self.color_to_depth)
      
      rotation_matrix = \
        np.array(transformation_data["depth_to_color"]["rotation_matrix"])
      translation_vector = \
        np.array(transformation_data["depth_to_color"]["translation_vector"]).reshape(3, 1)

      self.depth_to_color = np.eye(4) 
      self.depth_to_color[:3, :3] = rotation_matrix  
      self.depth_to_color[:3, 3] = translation_vector.flatten()  
      self.inverse_depth_to_color = np.linalg.inv(self.depth_to_color)     

  def get_depth(self,i=0):
    depth = cv2.imread(self.color_files[i].replace('rgb','depth'),-1)/1e3
    depth = cv2.resize(depth, (self.color_W,self.color_H), \
      interpolation=cv2.INTER_NEAREST)
    depth[(depth<0.001) | (depth>=self.zfar)] = 0
    return depth
  
  def stop_camera(self):
    return 
      
  def get_xyz_map(self,i=0):
    depth = self.get_depth(i)
    xyz_map = depth2xyzmap(depth, self.K)
    return xyz_map

  def get_occ_mask(self,i=0):
    hand_mask_file = self.color_files[i].replace('rgb','masks_hand')
    occ_mask = np.zeros((self.color_H,self.color_W), dtype=bool)
    if os.path.exists(hand_mask_file):
      occ_mask = occ_mask | (cv2.imread(hand_mask_file,-1)>0)

    right_hand_mask_file = self.color_files[i].replace('rgb','masks_hand_right')
    if os.path.exists(right_hand_mask_file):
      occ_mask = occ_mask | (cv2.imread(right_hand_mask_file,-1)>0)

    occ_mask = cv2.resize(occ_mask, (self.color_W,self.color_H), \
      interpolation=cv2.INTER_NEAREST)

    return occ_mask.astype(np.uint8)

  def get_gt_mesh(self):
    ob_name = self.videoname_to_object[self.get_video_name()]
    YCB_base_dir = os.getenv('YCB_base_dir')
    mesh = trimesh.load(f'{YCB_base_dir}/models/{ob_name}/textured_simple.obj')
    return mesh


class YcbineoatReader:
  def __init__(self, base_dir, downscale=1, shorter_side=None, zfar=np.inf):
      self.base_dir = base_dir
      self.downscale = downscale
      self.zfar = zfar
      self.mask_file = sorted(glob.glob(f"{self.base_dir}/masks/*.png"))
      # Initialize Kinect device
      pykinect.initialize_libraries()
      device_config = pykinect.default_configuration
      device_config.color_format = pykinect.K4A_IMAGE_FORMAT_COLOR_BGRA32
      device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_720P
      device_config.depth_mode = pykinect.K4A_DEPTH_MODE_NFOV_2X2BINNED
      self.device_config = device_config
      self.device = pykinect.start_device(config=device_config)
      time.sleep(1)
      # Get camera calibration to obtain intrinsic parameters
      self.calibration = self.device.get_calibration(
          self.device_config.depth_mode,
          self.device_config.color_resolution
      )

      # Extract the intrinsic parameters
      color_calib = self.calibration.color_camera_calibration
      intrinsics = color_calib.intrinsics.parameters.param
      fx = intrinsics.fx
      fy = intrinsics.fy
      cx = intrinsics.cx
      cy = intrinsics.cy
      self.K = np.array([[fx, 0, cx],
                          [0, fy, cy],
                          [0,  0,  1]])

      # Set image dimensions from the camera calibration
      self.color_W = color_calib.resolution_width
      self.color_H = color_calib.resolution_height

      if shorter_side is not None:
          self.downscale = shorter_side / min(self.color_H, self.color_W)

      self.color_H = int(self.color_H * self.downscale)
      self.color_W = int(self.color_W * self.downscale)
      self.K[:2] *= self.downscale

  def get_mask(self,i):
    mask = cv2.imread(self.mask_file,-1) 
    if len(self.mask_file) == 0:
      logging.error("No mask file found in the specified directory.")
      self.mask = None
    else:
      if len(mask.shape)==3:
        for c in range(3):
          if mask[...,c].sum()>0:
            mask = mask[...,c]
            break
      mask = cv2.resize(mask, (self.color_W,self.color_H), \
        interpolation=cv2.INTER_NEAREST).astype(bool).astype(np.uint8)
    return mask

  def update(self):
    self.capture = self.device.update()

  def get_color(self, i):
    ret_color, color_image = self.capture.get_color_image()
    if ret_color:
      color_image = cv2.cvtColor(color_image, cv2.COLOR_BGRA2BGR)
      color_image = cv2.resize(color_image, \
        (self.color_W, self.color_H), interpolation=cv2.INTER_NEAREST)
      return color_image
    else:
      logging.error("Failed to get color image.")
      return None

  def get_depth(self, i):
    ret_depth, depth_image = self.capture.get_depth_image()
    if ret_depth:
      depth_image = depth_image.astype(np.float32) / 1000.0  # Convert mm to meters
      depth_image = cv2.resize(depth_image, \
        (self.color_W, self.color_H), interpolation=cv2.INTER_NEAREST)
      depth_image[(depth_image < 0.001) | (depth_image >= self.zfar)] = 0
      return depth_image
    else:
      logging.error("Failed to get depth image.")
      return None
    
  def get_heatmap(self, color, max_intensity=1.0, sigma=50):
    # # TODO:needs to get the actuals heatmap of the defect
    image_shape = color.shape[:2]
    heatmap = np.zeros(image_shape)
    center_y, center_x = image_shape[0] // 2, image_shape[1] // 2
    heatmap[center_y, center_x] = max_intensity
    heatmap = cv2.GaussianBlur(heatmap, (0, 0), sigma)
    heatmap = heatmap / np.max(heatmap)
    return heatmap

  def get_video_name(self):
    return self.base_dir.split('/')[-1]

  def __len__(self): # TODO: this has no need for me 
    return len(self.mask_file)

  def get_gt_pose(self,i):
    try:
      pose = np.loadtxt(self.gt_pose_files[i]).reshape(4,4)
      return pose
    except:
      logging.info("GT pose not found, return None")
      return None
  def get_xyz_map(self,i):
    depth = self.get_depth(i)
    xyz_map = depth2xyzmap(depth, self.K)
    return xyz_map

  def get_occ_mask(self,i):
    hand_mask_file = self.color_files[i].replace('rgb','masks_hand')
    occ_mask = np.zeros((self.color_H,self.color_W), dtype=bool)
    if os.path.exists(hand_mask_file):
      occ_mask = occ_mask | (cv2.imread(hand_mask_file,-1)>0)

    right_hand_mask_file = self.color_files[i].replace('rgb','masks_hand_right')
    if os.path.exists(right_hand_mask_file):
      occ_mask = occ_mask | (cv2.imread(right_hand_mask_file,-1)>0)

    occ_mask = cv2.resize(occ_mask, \
      (self.color_W,self.color_H), interpolation=cv2.INTER_NEAREST)

    return occ_mask.astype(np.uint8)

  def get_gt_mesh(self):
    ob_name = self.videoname_to_object[self.get_video_name()]
    YCB_base_dir = os.getenv('YCB_base_dir')
    mesh = trimesh.load(f'{YCB_base_dir}/models/{ob_name}/textured_simple.obj')
    return mesh


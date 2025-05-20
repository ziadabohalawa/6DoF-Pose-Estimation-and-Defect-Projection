import numpy as np
import open3d as o3d
import cv2
import json
import matplotlib.pyplot as plt
from matplotlib import cm
import threading
import copy
import logging
import os

'''''''''''''''''''''''''''''''''Ray Tracing'''''''''''''''''''''''''''''''''
def load_color_image(image_path):
    """ Load an RGB image from the specified path. """
    img = cv2.imread(image_path)
    return img

def load_mesh(mesh_path):
    """ Load a mesh from a given path and compute its vertex normals. """
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    mesh.compute_vertex_normals()
    return mesh

def load_point_cloud(file_path):
    """ Load a point cloud file. """
    pcd = o3d.io.read_point_cloud(file_path)
    return pcd

def load_intrinsics(json_file_path):
    """
    Load intrinsic parameters from a JSON file.

    Parameters:
    json_file_path (str): The path to the JSON file containing intrinsic parameters.

    Returns:
    color_intrinsic (o3d.camera.PinholeCameraIntrinsic): The intrinsic parameters for the color camera.
    depth_intrinsic (o3d.camera.PinholeCameraIntrinsic): The intrinsic parameters for the depth camera.
    """
    with open(json_file_path, 'r') as f:
        intrinsics = json.load(f)

    depth_intrinsic = o3d.camera.PinholeCameraIntrinsic()
    depth_intrinsic.set_intrinsics(
        width=intrinsics['depth']['width'],
        height=intrinsics['depth']['height'],
        fx=intrinsics['depth']['fx'],
        fy=intrinsics['depth']['fy'],
        cx=intrinsics['depth']['cx'],
        cy=intrinsics['depth']['cy']
    )

    color_intrinsic = o3d.camera.PinholeCameraIntrinsic()
    color_intrinsic.set_intrinsics(
        width=intrinsics['color']['width'],
        height=intrinsics['color']['height'],
        fx=intrinsics['color']['fx'],
        fy=intrinsics['color']['fy'],
        cx=intrinsics['color']['cx'],
        cy=intrinsics['color']['cy']
    )

    return color_intrinsic, depth_intrinsic

def load_extrinsics(file_path):
    """
    Load extrinsic parameters (rotation matrix and translation vector) from a JSON file.

    Parameters:
    json_file_path (str): The path to the JSON file containing the extrinsic parameters.

    Returns:
    color_to_depth_trans (numpy.ndarray): The 4x4 transformation matrix from color to depth.
    depth_to_color_trans (numpy.ndarray): The 4x4 transformation matrix from depth to color.
    """
    json_file_path = f'{file_path}/configs/camera_extrinsics.json'
    with open(str(json_file_path), 'r') as file:
        extrinsics_data = json.load(file)

    rotation_matrix = extrinsics_data["color_to_depth"]["rotation_matrix"]
    translation_vector = extrinsics_data["color_to_depth"]["translation_vector"][0]  

    color_to_depth_trans = np.eye(4) 
    color_to_depth_trans[:3, :3] = np.array(rotation_matrix)
    color_to_depth_trans[:3, 3] = np.array(translation_vector)

    rotation_matrix = extrinsics_data["depth_to_color"]["rotation_matrix"]
    translation_vector = extrinsics_data["depth_to_color"]["translation_vector"][0]  
    depth_to_color_trans = np.eye(4)  
    depth_to_color_trans[:3, :3] = np.array(rotation_matrix)
    depth_to_color_trans[:3, 3] = np.array(translation_vector)
    return color_to_depth_trans, depth_to_color_trans

def choose_points(image):
    """
    Selects points on the depth image for defect detection.

    Parameters:
    depth_image: The depth image from which to select points.

    Returns:
    points (list): A list of tuples representing the selected points on the depth image.
    """
    points = []
    fig, ax = plt.subplots()
    if len(image.shape) == 2 or (len(image.shape) == 3 and image.shape[2] == 1):
        ax.imshow(image, cmap='gray')
        ax.set_title('Click on the image to select defects. Press ESC to finish.')
    elif len(image.shape) == 3 and image.shape[2] == 3:
        ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        ax.set_title('Click on the image to select points. Press ESC to finish.')
    else:
        raise ValueError("Unsupported image format")
    def onclick(event):
        """Handles mouse click events to select points on the depth image."""
        if event.button == \
            1 and event.xdata is not None and event.ydata is not None:
            x, y = int(event.xdata), int(event.ydata)
            points.append((x, y))
            logging.info(f"Point selected: ({x}, {y})")
            ax.plot(x, y, 'ro') 
            fig.canvas.draw()  

    def onkey(event):
        """Handles keyboard events to finish selecting points on the depth image."""
        if event.key == 'escape':
            plt.close(fig)  

    click_cid = fig.canvas.mpl_connect('button_press_event', onclick)
    key_cid = fig.canvas.mpl_connect('key_press_event', onkey)

    plt.show()
    fig.canvas.mpl_disconnect(click_cid)
    fig.canvas.mpl_disconnect(key_cid)
    return points

def generate_centered_heatmap(image_shape, max_intensity=1.0, sigma=50):
    """
    Generate a centered Gaussian heatmap of specified size.

    Parameters:
    image_shape (tuple): The shape of the image (height, width).
    max_intensity (float, optional): The maximum intensity value for the center pixel. Default is 1.0.
    sigma (int, optional): The standard deviation for the Gaussian blur. Default is 50.

    Returns:
    heatmap (numpy.ndarray): A 2D numpy array representing the Gaussian heatmap.
    """
    heatmap = np.zeros(image_shape)
    center_y, center_x = image_shape[0] // 2, image_shape[1] // 2
    heatmap[center_y, center_x] = max_intensity
    heatmap = cv2.GaussianBlur(heatmap, (0, 0), sigma)
    heatmap = heatmap / np.max(heatmap)

    return heatmap

def visualize_heatmap(color_image, heatmap):
    """Visualize the heat map on top of the color image."""
    plt.imshow(color_image)
    plt.imshow(heatmap, alpha=0.5, cmap='hot')
    plt.colorbar()
    plt.title('Image with Heat Map Overlay')
    plt.show()

def heatmap_to_points(heatmap, threshold=0.5):
    """
    Converts a heatmap into a list of points with their corresponding intensities.

    Parameters:
    heatmap (numpy.ndarray): A 2D array representing the heatmap.
    threshold (float, optional): The intensity threshold for selecting points from the heatmap. Defaults to 0.5.

    Returns:
    list: A list of tuples representing the coordinates and intensities of the points in the heatmap.
    """
    y_coords, x_coords = np.where(heatmap > threshold)
    intensities = heatmap[y_coords, x_coords]
    points_with_intensity = list(zip(x_coords, y_coords, intensities))
    return points_with_intensity

def estimate_normals(pcd):
    """ Estimate normals for the point cloud. """
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=10, max_nn=30))
    return pcd

def create_mesh(pcd):
    """ Create a mesh using Poisson reconstruction. """
    mesh, densities = \
        o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=8)

    o3d.visualization.draw_geometries([mesh])
    return mesh

def compute_rays(points, intrinsic):
    """
    Compute the normalized 3D rays corresponding to the given 2D image points.

    Parameters:
    points (list of tuples): A list of 2D image points (x, y, intensity) representing the coordinates and intensities of the points.
    intrinsic (Open3D CameraIntrinsicsParameters): The intrinsic parameters of the camera.

    Returns:
    tuple: A tuple containing:
        - np.ndarray: A 2D NumPy array containing the normalized 3D rays corresponding to the input points.
        - np.ndarray: A 1D NumPy array containing the intensities of the input points.
    """
    fx = intrinsic.intrinsic_matrix[0, 0]
    fy = intrinsic.intrinsic_matrix[1, 1]
    cx = intrinsic.intrinsic_matrix[0, 2]
    cy = intrinsic.intrinsic_matrix[1, 2]

    rays = []
    intensities = []
    for (x, y, intensity) in points:
        x_normalized = (x - cx) / fx
        y_normalized = (y - cy) / fy
        ray = np.array([x_normalized, y_normalized, 1.0])
        ray /= np.linalg.norm(ray)
        rays.append(ray)
        intensities.append(intensity)
    return np.array(rays), np.array(intensities)

def intersect_rays_with_mesh(mesh, rays, origin, intensities):
    """
    Intersect rays with a triangle mesh and return the intersection points with their intensities.

    Parameters:
    mesh (o3d.geometry.TriangleMesh): The triangle mesh to intersect with.
    rays (numpy.ndarray): An array of 3D ray directions. Each row represents a ray direction.
    origin (numpy.ndarray): An array of 3D ray origins. Each row represents a ray origin.
    intensities (numpy.ndarray): An array of intensities corresponding to each ray.

    Returns:
    tuple: A tuple containing:
        - numpy.ndarray: An array of 3D intersection points. Each row represents an intersection point.
        - numpy.ndarray: An array of intensities corresponding to the intersection points.
    """
    if not mesh.has_triangle_normals():
        mesh.compute_triangle_normals()
    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()
        
    t_mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)

    num_rays = rays.shape[0]
    origins = np.tile(origin, (num_rays, 1))
    ray_tensor = np.hstack((origins, rays))

    o3d_rays = o3d.core.Tensor(ray_tensor, dtype=o3d.core.Dtype.Float32)

    ray_mesh_intersection = o3d.t.geometry.RaycastingScene()
    _ = ray_mesh_intersection.add_triangles(t_mesh)

    intersections = ray_mesh_intersection.cast_rays(o3d_rays)

    hit_points = intersections['t_hit'].numpy()
    valid_hits = hit_points != np.inf

    intersection_points = origins[valid_hits] + \
        rays[valid_hits] * \
            hit_points[valid_hits, np.newaxis]
    intersection_intensities = intensities[valid_hits]

    return intersection_points, intersection_intensities

def create_intersection_pcd(intersections, intensities):
    """
    Creates a colored point cloud from the given intersection points and intensities.
    High intensity points are colored red, low intensity points are colored blue, transitioning
    through yellow and green.

    Parameters:
    intersections (numpy.ndarray): A 2D array containing the 3D coordinates of the intersection points.
        Each row represents a point with the format [x, y, z].
    intensities (numpy.ndarray): A 1D array containing the intensities of the intersection points.

    Returns:
    o3d.geometry.PointCloud: A point cloud object containing the colored intersection points.
    """
    intersection_pcd = o3d.geometry.PointCloud()
    intersection_pcd.points = o3d.utility.Vector3dVector(intersections)

    # Normalize intensities to range [0, 1]
    normalized_intensities = (intensities - np.min(intensities)) / \
                              (np.max(intensities) - np.min(intensities))

    # Use matplotlib's 'jet' colormap to map normalized intensities to colors
    cmap = cm.get_cmap('jet')
    colors = cmap(normalized_intensities)[:, :3]  # Get RGB values (exclude alpha)

    intersection_pcd.colors = o3d.utility.Vector3dVector(colors)
    return intersection_pcd

def project_debug_rays(rays, origin):
    """
    Function to create a LineSet geometry representing debug rays when no intersections are found.

    Parameters:
    rays (numpy.ndarray): A 2D array representing the 3D coordinates of the rays. Each row contains the ray's direction vector.
    origin (numpy.ndarray): A 1D array representing the origin point of the rays.

    Returns:
    debug_rays (o3d.geometry.LineSet): A LineSet geometry representing the debug rays. Each ray is colored red.
    """
    logging.info("No intersections found.")
    debug_rays = o3d.geometry.LineSet()
    # Extend the rays by 1000 units to visualize them in the 3D space.
    points = np.vstack((np.tile(origin, (len(rays), 1)), 
                        origin + rays * 1000))
    lines = [[i, i + len(rays)] for i in range(len(rays))]
    debug_rays.points = o3d.utility.Vector3dVector(points)
    debug_rays.lines = o3d.utility.Vector2iVector(lines)
    debug_rays.paint_uniform_color([1, 0, 0])

    return debug_rays

def create_heatmap_overlay(color_image, heatmap, min_intensity=0.1, max_intensity=0.9):
    """Create an overlay of the heatmap on the color image, automatically normalizing and 
       applying a specific intensity range from min_intensity to max_intensity."""
    
    heatmap_min = np.min(heatmap)
    heatmap_max = np.max(heatmap)
    heatmap_normalized = (heatmap - heatmap_min) / (heatmap_max - heatmap_min)
    
    heatmap_clipped = \
        np.clip(heatmap_normalized, min_intensity, max_intensity)
    clipped_normalized = \
        (heatmap_clipped - min_intensity) / (max_intensity - min_intensity)
    
    heatmap_rgb = \
        cv2.applyColorMap((clipped_normalized * 255).astype(np.uint8), cv2.COLORMAP_JET)

    if len(color_image.shape) == 2:
        color_image = cv2.cvtColor(color_image, cv2.COLOR_GRAY2RGB)
    elif color_image.shape[2] == 4:
        color_image = cv2.cvtColor(color_image, cv2.COLOR_RGBA2RGB)
    save_overlay(heatmap_rgb, f"debug/overlay/overlay___.png")
    overlay = cv2.addWeighted(color_image, 0.8, heatmap_rgb, 0.2, 0)
    return overlay

def save_overlay(overlay, save_path="overlay_image.png"):
    """Save the overlay image to the specified path."""
    # Ensure the directory exists
    directory = os.path.dirname(save_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
    
    # Save the image
    saved = cv2.imwrite(save_path, overlay)

'''''''''''''''''''''''''''''depth projection'''''''''''''''''''''''''''''''''
def load_depth_image(depth_image_path):
    """Load a depth image from the specified file path."""
    depth_image = cv2.imread(str(depth_image_path), cv2.IMREAD_UNCHANGED)
    return depth_image

def heatmap_to_point3d(heatmap, depth_image, intrinsic, threshold=0.1):
    """
    Converts a heatmap and corresponding depth image into a 3D point cloud.

    Parameters:
    heatmap (numpy.ndarray): A 2D array representing the heatmap.
    depth_image (numpy.ndarray): A 2D array representing the depth image.
    intrinsic (o3d.camera.PinholeCameraIntrinsic): The intrinsic parameters of the camera.
    threshold (float, optional): The intensity threshold for including points in the 3D point cloud. Default is 0.1.

    Returns:
    numpy.ndarray: A 2D array representing the 3D point cloud. Each row contains the x, y, z coordinates and intensity of a point.
    """
    points_3D = []
    height, width = heatmap.shape
    max_value = np.max(heatmap)
    depth_height, depth_width = depth_image.shape

    for y in range(height):
        for x in range(width):
            # Ensure x and y are within bounds of depth image
            if y >= depth_height or x >= depth_width:
                continue 


            intensity = heatmap[y, x] / max_value
            if intensity > threshold:
                depth = depth_image[y, x]
                if depth > 0:
                    x3d = (x - intrinsic.intrinsic_matrix[0, 2]) \
                        * depth / intrinsic.intrinsic_matrix[0, 0]
                    y3d = (y - intrinsic.intrinsic_matrix[1, 2]) \
                        * depth / intrinsic.intrinsic_matrix[1, 1]
                    z3d = depth * 0.98
                    points_3D.append([x3d, y3d, z3d, intensity])

    return np.array(points_3D)

def pcd_from_point3d(points_3D):
    """
    Convert a list of 3D points into an Open3D PointCloud object.

    Parameters:
    points_3D (list or numpy.ndarray): A list or 2D numpy array of 3D points, where each point is represented as [x, y, z].

    Returns:
    point_cloud (open3d.geometry.PointCloud): An Open3D PointCloud object containing the input 3D points.

    Raises:
    ValueError: If the input list or numpy array is empty.
    """
    if len(points_3D) == 0:
        raise ValueError("No valid 3D points found.")
    points_3D = np.array(points_3D)
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points_3D[:, :3])
    return point_cloud

def align_to_surface(defect_points, target_pcd, offset=0.1):
    """
    Aligns defect points to the surface of a target point cloud by finding the 
    nearest point and offsetting it along the normal.

    Parameters:
    defect_points (numpy.ndarray): An array of defect points in the format [x, y, z, intensity].
    target_pcd (open3d.geometry.PointCloud): The target point cloud to align the defect points to.
    offset (float, optional): The distance to offset the defect points along the normal. Default is 0.1.

    Returns:
    offset_points (numpy.ndarray): An array of offset points in the format [x, y, z].
    aligned_points (numpy.ndarray): An array of aligned points in the format [x, y, z].
    """
    if not target_pcd.has_normals():
        target_pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=0.1, max_nn=30
                )
            )
    # Create a KD-Tree for efficient nearest neighbor search using the target point cloud
    tree = o3d.geometry.KDTreeFlann(target_pcd)
    aligned_points = []
    offset_points = []

    for point in defect_points:
        spatial_point = point[:3]

        # Perform a k-NN (k-Nearest Neighbors) search to find the nearest point in the target point cloud
        [_, idx, _] = tree.search_knn_vector_3d(spatial_point, 1)  # 1 means searching for the nearest neighbor

        # Extract the nearest point's spatial coordinates from the target point cloud
        nearest_point = np.asarray(target_pcd.points)[idx[0]]

        # Extract the normal vector of the nearest point from the target point cloud
        normal = np.asarray(target_pcd.normals)[idx[0]]
        aligned_points.append(nearest_point)

        # Calculate the offset point by shifting the nearest point along its normal direction
        offset_point = nearest_point + normal * offset

        # Append the offset point to the offset points list
        offset_points.append(offset_point)
    return np.array(offset_points), np.array(aligned_points)

def calc_coordinates(depth_image, points, intrinsic):
    """
    This function calculates the 3D coordinates of given 2D points in an image using the depth information.

    Parameters:
    depth_image (numpy.ndarray): A 2D array representing the depth information of the image.
    points (list): A list of 2D points (x, y) in the image.
    intrinsic (object): An object containing the intrinsic parameters of the camera.

    Returns:
    points_3D (numpy.ndarray): A 2D numpy array representing the 3D coordinates of the given points.
    """
    points_3D = []
    fx = intrinsic.intrinsic_matrix[0, 0]
    fy = intrinsic.intrinsic_matrix[1, 1]
    cx = intrinsic.intrinsic_matrix[0, 2]
    cy = intrinsic.intrinsic_matrix[1, 2]

    for i in points:
        x, y = i
        depth = depth_image[y, x]
        if depth == 0:
            logging.info(f"Depth is zero at coordinates x = {x}, y = {y}. Skipping this point.")
            continue
        x3d = (x - cx) * depth / fx
        y3d = (y - cy) * depth / fy
        point = [x3d, y3d, depth] 
        points_3D.append(point)

    points_3D = np.array(points_3D, dtype=np.float64)
    logging.info(f"Points 3D array: {points_3D}")
    return points_3D

def visualize(list_of_objects):
    """
    Visualize the target point cloud and the offset points in a separate thread.

    Parameters:
    list_of_objects (list): List of geometries to visualize (point clouds or meshes).

    Returns:
    None
    """
    def run_visualization():
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="Defect Visualization")
        for obj in list_of_objects:
            vis.add_geometry(obj)

        ctr = vis.get_view_control()
        ctr.set_front([0, 0, -1])  
        ctr.set_up([0, -1, 0])    
        ctr.set_lookat([0, 0, 0])  
        ctr.set_zoom(0.5)  
        
        opt = vis.get_render_option()
        opt.background_color = np.asarray([1, 1, 1]) 
        opt.point_size = 5

        vis.run()
        vis.destroy_window()

    vis_thread = threading.Thread(target=run_visualization)
    vis_thread.start()
'''''''''''''''''''''''''''Main Functions'''''''''''''''''''''''''''''''''''''''
def ray_tracing(data_dir, target_mesh, heatmap, color_intrinsics, heatmap_threshold = 0.5):
    """
    Perform ray tracing on a 3D mesh using a 2D heatmap to detect intersections.

    This function takes a 3D point cloud (target), a heatmap representing areas of defects ,
    and camera intrinsic parameters. The function uses the heatmap to generate 
    points of defects, traces rays from the camera through these points, 
    and checks for intersections with the 3D mesh.

    Parameters:
    target (o3d.geometry.PointCloud): The 3D point cloud representing the CAD model.
    heatmap (numpy.ndarray): The 2D heatmap used for ray tracing.
    color_intrinsics (dict): The intrinsic parameters of the color camera.

    Returns:
    intersection_pcd (o3d.geometry.PointCloud): The point cloud of intersection points.
    mesh (o3d.geometry.TriangleMesh): The 3D mesh representing the CAD model.
    """
    origin = np.array([0,0,0])

    color_to_depth_trans, _ = load_extrinsics(data_dir)

    target_mesh_copy = copy.deepcopy(target_mesh)
    target_mesh_copy.transform(np.linalg.inv(color_to_depth_trans))
    points_with_intensity = heatmap_to_points(heatmap, heatmap_threshold)

    rays, intensities = compute_rays(points_with_intensity, color_intrinsics)
    intersections, intersection_intensities = \
        intersect_rays_with_mesh(target_mesh_copy, rays, origin, intensities)

    if len(intersections) > 0:
        intersection_pcd = \
            create_intersection_pcd(intersections, intersection_intensities)
        return intersection_pcd, target_mesh_copy
    else:
        debug_rays = project_debug_rays(rays, origin)
        return debug_rays, target_mesh_copy

def ray_tracing_points(data_dir, target, intrinsic_parameters, image):
    """    
    Perform ray tracing on a 3D mesh using the input image and camera intrinsic parameters.

    This function takes a 3D point cloud (target) and an image, then uses the 
    intrinsic camera parameters to compute rays through selected points in the image. 
    These rays are traced through the mesh to detect intersection points,
    which are returned as a point cloud.

    Parameters:
    target (o3d.geometry.PointCloud): The point cloud representing the mesh.
    intrinsic_parameters (o3d.camera.PinholeCameraIntrinsic): The intrinsic parameters of the camera.
    image (numpy.ndarray): The image used for ray tracing.

    Returns:
    intersection_pcd (o3d.geometry.PointCloud): The point cloud representing the intersection points.
    mesh (o3d.geometry.TriangleMesh): The mesh used for ray tracing.
    """
    origin = np.array([0,0,0])
    color_to_depth_trans, _ = load_extrinsics(data_dir)

    pcd = estimate_normals(target)
    mesh = create_mesh(pcd)
    mesh.paint_uniform_color([0, 0, 1])
    mesh.transform(np.linalg.inv(color_to_depth_trans))

    points = choose_points(image)
    rays = compute_rays(points, intrinsic_parameters)
    intersections = intersect_rays_with_mesh(mesh, rays, origin)

    if len(intersections) > 0:
        # Create a point cloud from intersection points
        intersection_pcd = o3d.geometry.PointCloud()
        intersection_pcd.points = o3d.utility.Vector3dVector(intersections)
        intersection_pcd.paint_uniform_color([1, 0, 0])  # Red color
        return intersection_pcd, mesh
    else:
        logging.info("No intersections found.")
        debug_rays = o3d.geometry.LineSet()
        points = np.vstack((np.tile(origin, (len(rays), 1)), 
                            origin + rays * 1000))  # Extend rays by 1000 units
        lines = [[i, i + len(rays)] for i in range(len(rays))]
        debug_rays.points = o3d.utility.Vector3dVector(points)
        debug_rays.lines = o3d.utility.Vector2iVector(lines)
        debug_rays.paint_uniform_color([1, 0, 0])
        mesh.paint_uniform_color([0, 0, 1])
        return debug_rays, mesh

def depth_projection_heatmap(depth_image, intrinsic, target, defects):
    """
    Project depth image defects (heatmap) onto the 3D surface of the target point cloud.

    Parameters:
    depth_image (numpy.ndarray): The depth image from which defects are to be projected.
    intrinsic (o3d.camera.PinholeCameraIntrinsic): The intrinsic parameters of the camera.
    target (o3d.geometry.PointCloud): The 3D point cloud representing the target surface.
    defects (numpy.ndarray): The coordinates of the defects in the depth image.

    Returns:
    offset_points (numpy.ndarray): The 3D coordinates of the defects after being offset from the target surface.
    aligned_points (numpy.ndarray): The 3D coordinates of the defects after being aligned with the target surface.
    point3d (numpy.ndarray): The 3D coordinates of the defects in the target point cloud.
    """
    point3d = heatmap_to_point3d(defects, depth_image, intrinsic)
    offset_points, aligned_points = align_to_surface(point3d, target, offset=0.5)
    return offset_points, aligned_points, point3d

def depth_projection_points(depth_image, intrinsic, target):
    """
    Project depth image defects onto the 3D surface of the target point cloud.

    Parameters:
    depth_image (numpy.ndarray): The depth image from which defects are to be projected.
    intrinsic (o3d.camera.PinholeCameraIntrinsic): The intrinsic parameters of the camera.
    target (o3d.geometry.PointCloud): The 3D point cloud representing the target surface.

    Returns:
    offset_points (numpy.ndarray): The 3D coordinates of the defects after being offset from the target surface.
    aligned_points (numpy.ndarray): The 3D coordinates of the defects after being aligned with the target surface.
    point3d (numpy.ndarray): The 3D coordinates of the defects in the target point cloud.
    """
    defect_points = choose_points(depth_image)
    point3d = calc_coordinates(depth_image, defect_points, intrinsic)
    offset_points, aligned_points = align_to_surface(point3d, target, offset=0.5)
    return offset_points, aligned_points, point3d

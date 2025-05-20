import open3d as o3d
import numpy as np
import time
import copy
import logging
import json
import os

def timeit(func):
    """Decorator function to measure the execution time of a given function."""
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        logging.info(f":: {func.__name__} executed in {end - start:.6f} seconds")
        return result
    return wrapper

def demo_data(base_dir='demo_data/turbine'):
    file = '000'
    source = o3d.io.read_point_cloud(f'{base_dir}/pcd/cloud_{file}.ply')
    background = o3d.io.read_point_cloud(f'{base_dir}/background/box.ply')
    target = o3d.io.read_point_cloud(f'{base_dir}/mesh/model.ply')
    
    fp_transformation = read_pose(f'debug/ob_in_cam/0{file}.txt') #the run.py should be run with --debug 2 flag so that we get the transformation file
    scaled_transformation = scale_up_transformation(fp_transformation)
    
    color_to_depth, depth_to_color = get_extrinsic_calibration()
    initial_fp_transformation = np.dot(color_to_depth, scaled_transformation)
    with open(f'{base_dir}/configs/icp_parameters.json', 'r') as file:
        icp_param = json.load(file)
    return target, source, background, initial_fp_transformation, icp_param

def draw_registration_result(source, target, transformation, Name, param, debug_path="debug/icp"):
    """Draws the ICP registration result using Open3D visualization and saves it to a file."""
    if not param['debug_vis']:
        return
    
    if not os.path.exists(debug_path):
        os.makedirs(debug_path)

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=Name, visible=True)

    if target == 0:
        source_temp = copy.deepcopy(source)
        vis.add_geometry(source_temp)
    elif source == 0:
        target_temp = copy.deepcopy(target)
        vis.add_geometry(target_temp)
    else:
        source_temp = copy.deepcopy(source)
        target_temp = copy.deepcopy(target)
        target_temp.transform(np.linalg.inv(transformation))
        vis.add_geometry(source_temp)
        vis.add_geometry(target_temp)
    
    ctr = vis.get_view_control()
    ctr.set_front([0, 0, -1])
    ctr.set_up([0, -1, 0])
    ctr.set_lookat([0, 0, 0])
    ctr.set_zoom(0.5)
    
    vis.poll_events()
    vis.update_renderer()
    
    file_name = os.path.join(debug_path, f"{Name}.jpg")
    time.sleep(0.5)
    vis.capture_screen_image(file_name)
    vis.destroy_window()

def draw_reslut(source, target, transformation, Name):
    """Draws the ICP registration result using Open3D visualization and saves it to a file."""

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=Name, visible=True)
    transformation
    inv_trans = np.linalg.inv(transformation)
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    target_temp.transform(inv_trans)
    
    center = np.mean(np.asarray(source_temp.points), axis=0)
    source_temp.scale(1000, center=center)

    vis.add_geometry(source_temp)
    vis.add_geometry(target_temp)
    
    ctr = vis.get_view_control()
    ctr.set_front([0, 0, -1])
    ctr.set_up([0, -1, 0])
    ctr.set_lookat([0, 0, 0])
    ctr.set_zoom(0.5)
    
    vis.poll_events()
    vis.update_renderer()

    time.sleep(0.5)
    vis.destroy_window()
''''''''''''''''''''''''''''' Preprocess Target '''''''''''''''''''''''''''''
'''
def preprocess_target(pcd, param):
    """
    Preprocess the target point cloud for registration.

    Parameters:
    pcd (o3d.geometry.PointCloud): The input target point cloud.
    param (dict): The configuration parameters.

    Returns:
    target_processed (o3d.geometry.PointCloud): The preprocessed target point cloud.
    target_fpfh (o3d.pipelines.registration.Feature): The Fast Point Feature Histograms (FPFH) of the preprocessed target point cloud.
    """
    params = param['preprocess_target']
    target_processed = o3d.geometry.PointCloud()

    draw_registration_result(0, pcd, np.identity(4), "1_Target", param)

    bounding_box = pcd.get_axis_aligned_bounding_box()
    bbox_size = bounding_box.get_extent()

    num_points = len(pcd.points)
    volume = bbox_size[0] * bbox_size[1] * bbox_size[2]
    avg_density = num_points / volume

    target_density = 2.5e-3  # Add to icp_parameters.json instead of max_pcd
    downsample_rate = min(1.0, target_density / avg_density)
    
    target_processed = pcd.random_down_sample(downsample_rate)

    estimate_normals(target_processed, params)
    target_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        target_processed,
        o3d.geometry.KDTreeSearchParamHybrid(
            radius=params['fpfh_radius'], max_nn=params['fpfh_max_nn']
        )
    )
    draw_registration_result(0, target_processed, np.identity(4), "2_preprocessed_Target", param)
    return target_processed, target_fpfh
'''
@timeit
def preprocess_target(pcd, param):
    """
    Preprocess the target point cloud for registration.

    Parameters:
    pcd (o3d.geometry.PointCloud): The input target point cloud.
    param (dict): The configuration parameters.

    Returns:
    target_processed (o3d.geometry.PointCloud): The preprocessed target point cloud.
    target_fpfh (o3d.pipelines.registration.Feature): The Fast Point Feature Histograms (FPFH) of the preprocessed target point cloud.
    """
    params = param['preprocess_target']
    target_processed = o3d.geometry.PointCloud()
    
    draw_registration_result(0, pcd, np.identity(4), 
                             "1_Target", param)
    if len(pcd.points) > params['max_pcd']:
        indices = np.random.choice(
            len(pcd.points), params['max_pcd'], replace=False)
        target_processed.points = o3d.utility.Vector3dVector(
            np.asarray(pcd.points)[indices])
        if pcd.has_normals():
            target_processed.normals = o3d.utility.Vector3dVector(
                np.asarray(pcd.normals)[indices])
        if pcd.has_colors():
            target_processed.colors = o3d.utility.Vector3dVector(
                np.asarray(pcd.colors)[indices])
    else:
        logging.info(f":: Point cloud already has less than or exactly {params['max_pcd']} points.")
        target_processed = pcd  

    estimate_normals(target_processed, params)
    target_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        target_processed,
        o3d.geometry.KDTreeSearchParamHybrid(
            radius=params['fpfh_radius'], max_nn=params['fpfh_max_nn']
            )
        )
    draw_registration_result(0, target_processed, np.identity(4), 
                             "2_preprocessed_Target", param)
    return target_processed, target_fpfh

''''''''''''''''''''''''''''' Preprocess Source '''''''''''''''''''''''''''''
@timeit
def preprocess_source(pcd, background, param, i=0):
    """
    Preprocesses the source point cloud by down-sampling, estimating normals,
    optionally removing planes or background, and visualizing at each step.

    Parameters:
    - pcd: Point cloud to preprocess.
    - param: Dictionary containing all parameters and configurations.

    Returns:
    source_processed (o3d.geometry.PointCloud): The preprocessed source point cloud.
    source_fpfh (o3d.pipelines.registration.Feature): The Fast Point Feature Histograms (FPFH) of the preprocessed source point cloud.
    
    """
    params = param['preprocess_source']
    if i > 0:
        params['down_sample'] = 5
    background = background.voxel_down_sample(voxel_size=params['down_sample']*2)
    pcd_down = pcd.voxel_down_sample(voxel_size=params['down_sample'])
    draw_registration_result(
        background, 0, np.identity(4), "3_Background", param
    )
    draw_registration_result(
        pcd_down, 0, np.identity(4), "4_Source_before_removal", param
    )

    # Step 1: Perform plane segmentation on the scene point cloud
    plane_model, inliers = \
        perform_plane_segmentation(pcd_down, params['plane_removal'])
    average_normal = np.array([1, 1, 1], dtype=float)
    if i==0:
        estimate_normals(pcd_down, params)
        average_normal = compute_average_normal(pcd_down)
        logging.info(f":: Average Normal for Source = {average_normal}")
    
    # Get the normal of the plane and flip if necessary
    plane_model, plane_normal = \
        flip_plane_normal_if_needed(plane_model, average_normal)

    # Remove the points below plane from the scene
    source_processed = remove_points_below_plane(pcd_down, plane_model)    

    draw_registration_result(
        source_processed, 0, np.identity(4), "5_Source_after_remove_points_below_plane", param
    )

    if param['box']:
        source_processed = background_removal(source_processed, background)
        draw_registration_result(
            source_processed, 0, np.identity(4), \
                "6_Source_after_background_removal", param
        )
    else:
        source_processed = remove_plane(pcd_down, inliers)
    if param['mesh']:
        mesh = create_and_smooth_mesh(source_processed, params)
        source_processed = mesh_to_pcd(mesh, params)
        draw_registration_result(
            source_processed, 0, np.identity(4), "5_mesh", param
        )
    source_processed = filter_largest_cluster(source_processed)
    source_processed = remove_statistical_outliers(source_processed,\
        nb_neighbors=75, std_ratio=0.01)

    if i ==0:
        estimate_normals(background, params)
        estimate_normals(source_processed, params)

        source_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            source_processed,
            o3d.geometry.KDTreeSearchParamHybrid(
                radius=params['fpfh_radius'], max_nn=params['fpfh_max_nn']
            )
        )
    else:
        source_fpfh = 0    

    draw_registration_result(
        source_processed, 0, np.identity(4), "7_Source_Preprocessed", param
    )
    source_filtered = source_processed
    return source_processed, source_filtered, source_fpfh

def filter_largest_cluster(pcd, eps=10, min_points=10):
    """
    Filters and retains only the largest cluster in a point cloud.
    
    Parameters:
    - pcd (open3d.geometry.PointCloud): The input point cloud.
    - eps (float): The maximum distance between two points for them to be considered in the same neighborhood (DBSCAN parameter).
    - min_points (int): The minimum number of points to form a dense region (DBSCAN parameter).
    
    Returns:
    - open3d.geometry.PointCloud: A point cloud containing only the largest cluster.
    """
    # Cluster the point cloud using DBSCAN
    labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=True))

    # Identify the unique clusters and their sizes
    unique_labels, counts = np.unique(labels, return_counts=True)

    # Filter out noise points (label = -1) and find the largest cluster
    valid_labels = unique_labels[unique_labels != -1]
    if len(valid_labels) == 0:
        print("No valid clusters found.")
        return None
    
    largest_cluster_label = valid_labels[np.argmax(counts[unique_labels != -1])]

    # Select points that belong to the largest cluster
    largest_cluster = pcd.select_by_index(np.where(labels == largest_cluster_label)[0])

    return largest_cluster

def estimate_normals(pcd, params):
    """ Estimate normals for the point cloud. """
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=2, max_nn=5))
    return pcd

def remove_statistical_outliers(pcd, nb_neighbors=20, std_ratio=1.0):
    """Remove statistical outliers from the point cloud."""
    clean_pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors,\
        std_ratio=std_ratio)
    return clean_pcd

def compute_average_normal(pcd):
    """Compute the average normal vector of the point cloud."""
    pcd_copy = copy.deepcopy(pcd)
    pcd_copy = pcd_copy.voxel_down_sample(voxel_size=10)
    normals = np.asarray(pcd_copy.normals)
    average_normal = np.mean(normals, axis=0)
    average_normal /= np.linalg.norm(average_normal)
    return average_normal

def perform_plane_segmentation(pcd, param):
    """Perform plane segmentation on the point cloud."""
    plane_model, inliers = pcd.segment_plane(
        distance_threshold=param['distance_threshold'],
        ransac_n=3, num_iterations=param['num_iterations']
    )
    return plane_model, inliers

def check_normals(plane_model, average_normal):
    flip = False
    [a, b, c, d] = plane_model
    plane_normal = np.array([a, b, c])
    plane_normal /= np.linalg.norm(plane_normal) 

    dot_product = np.dot(plane_normal, average_normal)
    if dot_product < 0:
        return flip == True

def flip_plane_normal_if_needed(plane_model, average_normal):
    """
    Flip the plane normal if it is pointing in the opposite direction of the average normal.
    This helps to avoid wrongly oriented planes. and make sure we remove 
    the point below the planes when remove_points_below_plane is called
    """
    [a, b, c, d] = plane_model
    plane_normal = np.array([a, b, c])
    plane_normal /= np.linalg.norm(plane_normal) 

    dot_product = np.dot(plane_normal, average_normal)
    if dot_product < 0:
        [a, b, c, d] = plane_model
        plane_normal = -plane_normal
        plane_model = [-a, -b, -c, -d]
        logging.info(":: Plane normal was flipped to match the majority of normals.")
    return plane_model, plane_normal

def remove_plane(pcd, inliers):
    """Remove the plane from the point cloud."""
    remaining_pcd = pcd.select_by_index(inliers, invert=True)
    return remaining_pcd

def remove_points_below_plane(pcd, plane_model):
    """Remove points below the detected plane."""
    [a, b, c, d] = plane_model
    points = np.asarray(pcd.points)
    distances = (a * points[:, 0] +\
        b * points[:, 1] +\
        c * points[:, 2] + d) / np.sqrt(a**2 + b**2 + c**2)
    
    above_plane_points = points[distances <= 0]
    above_plane_pcd = o3d.geometry.PointCloud()
    above_plane_pcd.points = o3d.utility.Vector3dVector(above_plane_points)
    return above_plane_pcd

def background_removal(pcd, background_pcd, threshold=10):
    """Remove background points (empty box) from the point cloud."""
    background_tree = o3d.geometry.KDTreeFlann(background_pcd)
    points = np.asarray(pcd.points)
    final_object_points = []
    for point in points:
        [k, idx, _] = background_tree.search_radius_vector_3d(point, threshold)
        if k == 0:
            final_object_points.append(point)
    final_object_pcd = o3d.geometry.PointCloud()
    if len(final_object_pcd.points) == 0:
        return pcd
    else:
        final_object_pcd.points = \
            o3d.utility.Vector3dVector(np.array(final_object_points))
        return final_object_pcd

def read_pose(file):
    with open(file, 'r') as file:
        lines = file.readlines()
        trans = np.array([[float(value) for value in line.split()] for line in lines])
    return trans

def scale_up_transformation(fp_transformation):
    transformation_scaled = fp_transformation.copy()
    transformation_scaled[:3, -1] *= 1000  # Scale the translation component
    logging.info(f":: fp_transformation: {transformation_scaled}")
    return transformation_scaled

def transform_object(pcd, transformation):
  object_copy = copy.deepcopy(pcd)
  object_copy.transform(transformation)
  return object_copy

def get_extrinsic_calibration(base_dir='demo_data/turbine'):
    with open(f"{base_dir}/configs/camera_extrinsics.json", "r") as file:
        transformation_data = json.load(file)

    rotation_matrix = np.array(transformation_data["color_to_depth"]["rotation_matrix"])
    translation_vector = np.array(transformation_data["color_to_depth"]["translation_vector"]).reshape(3, 1)

    color_to_depth = np.eye(4)  
    color_to_depth[:3, :3] = rotation_matrix 
    color_to_depth[:3, 3] = translation_vector.flatten()  
    inverse_color_to_depth = np.linalg.inv(color_to_depth)
    
    # Extract the rotation matrix and  vector
    rotation_matrix = np.array(transformation_data["depth_to_color"]["rotation_matrix"])
    translation_vector = np.array(transformation_data["depth_to_color"]["translation_vector"]).reshape(3, 1)

    depth_to_color = np.eye(4)  
    depth_to_color[:3, :3] = rotation_matrix  
    depth_to_color[:3, 3] = translation_vector.flatten() 
    inverse_depth_to_color = np.linalg.inv(depth_to_color)     
    return color_to_depth, depth_to_color

def create_and_smooth_mesh(filtered_pcd, param):

    """
    Creates a mesh from the filtered point cloud if the 'mesh' parameter is
    enabled in the configuration. It applies ball pivoting and Poisson disk
    sampling to create and refine the mesh.

    Returns:
    - The processed mesh as a point cloud.
    """
    params = param['mesh']
    radii = [params['radius'], params['radius'] * 2, params['radius'] * 4]
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        filtered_pcd, o3d.utility.DoubleVector(radii)
    )
    mesh_out = mesh.filter_smooth_simple(params['number_of_iterations'])
    mesh_out.compute_vertex_normals()
    
    return mesh_out

def mesh_to_pcd(mesh_out, param):
    """
    Converts a mesh to a point cloud using Poisson disk sampling.
    Estimates normals for the resulting point cloud.
    Draws the resulting point cloud for visualization.
    """
    params = param['mesh']
    pcd = mesh_out.sample_points_poisson_disk(
        params['number_of_points']
    )
    estimate_normals(pcd, params)
    return pcd

''''''''''''''''''''''''''''' ICP '''''''''''''''''''''''''''''
def execute_global_registration(source_processed, target_processed, source_fpfh, target_fpfh, param):
    """
    Executes global registration using RANSAC-based on feature matching.

    Parameters:
    source_processed (o3d.geometry.PointCloud): The preprocessed source point cloud.
    target_processed (o3d.geometry.PointCloud): The preprocessed target point cloud.
    source_fpfh (o3d.geometry.PointCloud): The Fast Point Feature Histograms (FPFH) of the source point cloud.
    target_fpfh (o3d.geometry.PointCloud): The Fast Point Feature Histograms (FPFH) of the target point cloud.
    param (dict): The parameters for the registration process.

    Returns:
    result (o3d.pipelines.registration.RegistrationResult): The result of the global registration.
    """
    params = param['execute_global_registration']
    result = \
        o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_processed, target_processed, source_fpfh, target_fpfh, False,
        params['distance_threshold'],
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, 
        [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
            params['correspondence_checkers'][0]['value']
            ),
         o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
             params['distance_threshold']
             ),
         o3d.pipelines.registration.CorrespondenceCheckerBasedOnNormal(
             params['angle_threshold']
             )
         ],
        o3d.pipelines.registration.RANSACConvergenceCriteria(
            params['ransac_criteria']['iterations'], 
            params['ransac_criteria']['confidence']
            )
        )
    return result

def refine_registration(source, target, transformation, param):
    """
    Performs refinement of the registration result using Iterative Closest Point (ICP) algorithm.

    Parameters:
    source (o3d.geometry.PointCloud): The source point cloud.
    target (o3d.geometry.PointCloud): The target point cloud.
    transformation (numpy.ndarray): The initial transformation matrix.
    param (dict): The parameters for the registration process.

    Returns:
    result (o3d.pipelines.registration.RegistrationResult): The refined registration result.
    """
    params = param['refine_registration']
    result = o3d.pipelines.registration.registration_icp(
        source, target, params['distance_threshold'], transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    return result

def run_icp(source_processed, target_processed, source_fpfh, target_fpfh, param):
    """
    Executes the Iterative Closest Point (ICP) algorithm for point cloud registration.

    Parameters:
    source_processed (o3d.geometry.PointCloud): The preprocessed source point cloud.
    target_processed (o3d.geometry.PointCloud): The preprocessed target point cloud.
    source_fpfh (o3d.geometry.PointCloud): The Fast Point Feature Histograms (FPFH) of the source point cloud.
    target_fpfh (o3d.geometry.PointCloud): The Fast Point Feature Histograms (FPFH) of the target point cloud.
    param (dict): The parameters for the registration process.

    Returns:
    result_icp (o3d.pipelines.registration.RegistrationResult): The result of the refined registration.
    result_ransac (o3d.pipelines.registration.RegistrationResult): The result of the global registration.
    """
    result_ransac = execute_global_registration(
        source_processed, target_processed, 
        source_fpfh, target_fpfh, param)
    result_icp = refine_registration(
        source_processed, target_processed, 
        result_ransac.transformation, param)
    return result_icp, result_ransac

def improve_result(source_processed, original_target_processed, current_result, parameter):
    """
    This function refines the registration result by performing additional iterations of registration.
    The function continues this process until the fitness and RMSE criteria are met or the maximum number of iterations is reached.

    Parameters:
    source_processed (o3d.geometry.PointCloud): The processed source point cloud.
    original_target_processed (o3d.geometry.PointCloud): The original target point cloud.
    current_result (o3d.pipelines.registration.RegistrationResult): The current registration result.
    parameter (dict): A dictionary of parameters for processing.

    Returns:
    improved_result (o3d.pipelines.registration.RegistrationResult): The improved registration result.
    """
    parameters = copy.deepcopy(parameter)
    target_processed = copy.deepcopy(original_target_processed)

    if not hasattr(current_result, 'fitness') or current_result.fitness is None:
        initial_fp_transformation = current_result
        current_result = o3d.pipelines.registration.RegistrationResult()
        current_result.fitness = 0.8
        current_result.inlier_rmse = 3.0
        current_result.transformation = initial_fp_transformation
    best_fitness = current_result.fitness
    best_rmse = current_result.inlier_rmse
    best_transformation = np.linalg.inv(current_result.transformation)
    iteration = 0
    max_iterations = 50
    logging.info(":: Additional refinements")
    x = 0.1
    while iteration < max_iterations and \
        (best_fitness < parameters['run_icp']['fitness_threshold'] or \
            best_rmse > parameters['run_icp']['rmse_threshold']):
        current_param = parameters.copy()
        current_param['refine_registration']['distance_threshold'] *= \
            np.random.uniform(0.8, 1.2)

        noise_rotation = o3d.geometry.get_rotation_matrix_from_xyz(
            [np.random.uniform(-0.01, 0.01) for _ in range(3)])
        noise_translation = np.random.uniform(-x, x, 3)
        noise_transform = np.eye(4)
        noise_transform[:3, :3] = noise_rotation
        noise_transform[:3, 3] = noise_translation
        current_transform = np.dot(noise_transform, best_transformation)

        try:
            refined_result = refine_registration(
                source_processed, target_processed,
                current_transform, current_param)

            if refined_result.fitness > 0 and refined_result.inlier_rmse > 0:
                if refined_result.fitness > best_fitness or \
                    (refined_result.fitness == best_fitness and \
                        refined_result.inlier_rmse < best_rmse):
                    best_fitness = refined_result.fitness
                    best_rmse = refined_result.inlier_rmse
                    best_transformation = refined_result.transformation
                    logging.info(f":: Improved result: Fitness = \
                        {best_fitness:.4f}, RMSE = {best_rmse:.4f}")

            else:
                logging.info(f":: Iteration {iteration + 1} produced an invalid result. Skipping.")
                x+=.25
        except Exception as e:
            logging.info(f":: Error in refinement iteration {iteration + 1}: {str(e)}. Skipping this iteration.")

        iteration += 1

    logging.info(f":: Total iterations: {iteration}")

    improved_result = o3d.pipelines.registration.RegistrationResult()
    improved_result.fitness = best_fitness
    improved_result.inlier_rmse = best_rmse
    improved_result.transformation = best_transformation

    return improved_result

def predict_z_axis_adjustment(source, target, initial_fp_transformation, param, max_adjustment=50, initial_step=10):
    """
    Predicts the optimal z-axis adjustment for point cloud registration using an adaptive search strategy.

    Parameters:
    source (o3d.geometry.PointCloud): The source point cloud.
    target (o3d.geometry.PointCloud): The target point cloud.
    initial_fp_transformation (np.array): The initial transformation matrix.
    param (dict): The parameters for the registration process.
    max_adjustment (float): The maximum absolute z-axis adjustment to try (in mm).
    initial_step (float): The initial step size for z-axis adjustment (in mm).

    Returns:
    best_adjustment (float): The best z-axis adjustment found.
    best_fitness (float): The fitness score of the best adjustment.
    best_rmse (float): The RMSE of the best adjustment.
    """
    best_adjustment = 0
    best_fitness = 0
    best_rmse = float('inf')
    
    current_adjustment = 0
    step = initial_step
    direction = 1 
    
    while abs(step) >= 0.1:  
        current_transformation = np.copy(initial_fp_transformation)
        current_transformation[2, 3] -= current_adjustment
        target_adjusted = copy.deepcopy(target)
        
        result_icp = o3d.pipelines.registration.registration_icp(
            source, target_adjusted, 
            param['refine_registration']['distance_threshold'],
            np.linalg.inv(current_transformation),
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=1)
        )
        
        if result_icp.fitness > best_fitness or \
            (result_icp.fitness == best_fitness and \
                result_icp.inlier_rmse < best_rmse):
            best_adjustment = current_adjustment
            best_fitness = result_icp.fitness
            best_rmse = result_icp.inlier_rmse
            current_adjustment += step * direction
        else:
            direction *= -1
            step /= 2
            current_adjustment += step * direction
        
        if abs(current_adjustment) > max_adjustment:
            current_adjustment = max_adjustment * np.sign(current_adjustment)
            step /= 1.25
            direction *= -1
        
        if best_fitness > 0.95:
            break

    logging.info(f":: Best z-axis adjustment: {best_adjustment:.2f}mm, Fitness: {best_fitness:.4f}, RMSE: {best_rmse:.4f}")
    return best_adjustment, best_fitness, best_rmse

''''''''''''''''''''''''''''' Main '''''''''''''''''''''''''''''
def determine_pose(source, target, background, initial_fp_transformation, parameters, icp=False):
    param = copy.deepcopy(parameters)
    source.paint_uniform_color([1, 0, 0])
    target.paint_uniform_color([0, 0, 1])
    start_time_total = time.perf_counter()
    target_processed, target_fpfh = preprocess_target(target, param)
    source_processed, source_filtered, source_fpfh = preprocess_source(source, background, param)
    draw_registration_result(source, target, np.eye(4), '8_Before_Pose_estimation', param)
        
    if icp:
        result_icp, result_ransac = run_icp(source_processed, target_processed,
                                            source_fpfh, target_fpfh, param)
        logging.info(f"-- Initial Attempt"
                        f"\n:: Global registeration results: Inlier_rmse: {result_ransac.inlier_rmse:.4f}, Fitness: {result_ransac.fitness:.4f}"
                        f"\n:: Refine registeration results: Inlier_rmse: {result_icp.inlier_rmse:.4f}, Fitness: {result_icp.fitness:.4f}")
        refinement_attempts = 1 
        while result_icp.fitness < param['run_icp']['fitness_threshold'] \
                        or result_icp.inlier_rmse > param['run_icp']['rmse_threshold']:
            result_icp, result_ransac = run_icp(source_processed, target_processed,
                                            source_fpfh, target_fpfh, param)
            logging.info(f"-- Attempt {refinement_attempts}"
                        f"\n:: Global registeration results: Inlier_rmse: {result_ransac.inlier_rmse:.4f}, Fitness: {result_ransac.fitness:.4f}"
                        f"\n:: Refine registeration results: Inlier_rmse: {result_icp.inlier_rmse:.4f}, Fitness: {result_icp.fitness:.4f}")
            refinement_attempts += 1 
        result_icp.transformation = np.linalg.inv(result_icp.transformation)

        z_adjustment = 0
    else:
        draw_registration_result(source, target, \
            np.linalg.inv(initial_fp_transformation), \
                '9_FoundationPose_Transformation', param)
        
        z_adjustment, best_fitness, best_rmse = predict_z_axis_adjustment(
            source_processed, target_processed, initial_fp_transformation, param)
        
        initial_fp_transformation[2, 3] += z_adjustment
        logging.info(f":: Predicted Z-axis adjustment: {z_adjustment:.2f}mm")

        draw_registration_result(source_processed, target, \
            np.linalg.inv(initial_fp_transformation), '10_After_Z_Axis_adjustment', param)
        
        
        result_icp = o3d.pipelines.registration.RegistrationResult()
        result_icp.fitness = best_fitness
        result_icp.inlier_rmse = best_rmse
        result_icp.transformation = initial_fp_transformation
        
    # Perform final refinement
    best_result_icp = \
        improve_result(source_processed, target_processed, result_icp, param)

    end_time_total = time.perf_counter()
    logging.info(f"-- Final Results"
                 f"\n:: Refine registration results: Inlier_rmse: {best_result_icp.inlier_rmse:.4f}, Fitness: {best_result_icp.fitness:.4f}"
                 f"\n:: Pose Estimation Execution Time: {end_time_total - start_time_total:.2f} seconds"
                 f"\n:: Final Transformation Matrix : \n{np.linalg.inv(best_result_icp.transformation)}")
    # Combine the adjusted transformation with the refinement result
    target_transformed = copy.deepcopy(target)
    target_transformed.transform(np.linalg.inv(best_result_icp.transformation))
    draw_registration_result(source, target, best_result_icp.transformation, '11_Result', param)
    
    return target_transformed, best_result_icp, z_adjustment, target_processed

def refine_pose_with_icp(source, target, background, initial_fp_transformation, parameters):
    """
    Determines the pose using FoundationPose Initial Transformation, 
    predicts Z-axis adjustment and refine the result using ICP.

    Parameters:
        source (o3d.geometry.PointCloud): The source point cloud.
        target (o3d.geometry.PointCloud): The target point cloud.
        background (o3d.geometry.PointCloud): The background point cloud.
        initial_fp_transformation (np.ndarray): The initial foundation pose transformation matrix.
        parameters (dict): A dictionary of parameters for processing.

    Returns:
        target_transformed (o3d.geometry.PointCloud): The transformed target point cloud.
        best_result_icp (o3d.pipelines.registration.RegistrationResult): The final ICP registration result.
        z_adjustment (float): The Z-axis adjustment applied.
        target_processed (o3d.geometry.PointCloud): The processed target point cloud.
    """
    param = copy.deepcopy(parameters)

    source.paint_uniform_color([1, 0, 0])  # Red color for source
    target.paint_uniform_color([0, 0, 1])  # Blue color for target

    target_processed, target_fpfh = preprocess_target(target, param)
    source_processed, source_filtered, source_fpfh = preprocess_source(source, background, param)

    draw_registration_result(
        source, target, np.eye(4), 
        '8_Before_Pose_estimation', param)
    draw_registration_result(
        source, target, np.linalg.inv(initial_fp_transformation),
        '9_FoundationPose_Transformation', param
    )

    # Predict the Z-axis adjustment based on the initial transformation
    z_adjustment, best_fitness, best_rmse = predict_z_axis_adjustment(
        source_processed, target_processed, initial_fp_transformation, param
    )

    # Adjust the initial transformation along the Z-axis
    initial_fp_transformation[2, 3] += z_adjustment
    logging.info(f":: Predicted Z-axis adjustment: {z_adjustment:.2f}mm")

    draw_registration_result(
        source_processed, target, np.linalg.inv(initial_fp_transformation),
        '10_After_Z_Axis_adjustment', param
    )

    # Create a registration result
    result_icp = o3d.pipelines.registration.RegistrationResult()
    result_icp.fitness = best_fitness
    result_icp.inlier_rmse = best_rmse
    result_icp.transformation = initial_fp_transformation

    # Perform final refinement steps
    best_result_icp = improve_result(
        source_processed, target_processed, result_icp, param
    )
    logging.info(
        f"-- Final Results"
        f"\n:: Refine registration results: Inlier_rmse: {best_result_icp.inlier_rmse:.4f}, "
        f"Fitness: {best_result_icp.fitness:.4f}"
        f"\n:: Final Transformation Matrix:\n{np.linalg.inv(best_result_icp.transformation)}"
    )

    # Apply the final transformation
    target_transformed = copy.deepcopy(target)
    target_transformed.transform(np.linalg.inv(best_result_icp.transformation))

    draw_registration_result(
        source, target, best_result_icp.transformation, '11_Result', param
    )

    return target_transformed, best_result_icp, z_adjustment, target_processed

def demo_icp(base_dir='demo_data/turbine'):
    debug = True
    info = True
    i=0
    tries = 1
    icp = False
    
    if debug:
        level = logging.DEBUG
    elif info:
        level = logging.INFO
    else:
        level = logging.WARNING

    logging.basicConfig(
        filename=None,  
        filemode='a', 
        format='%(message)s',  
        level=level 
    )

    target, source, background, initial_fp_transformation, icp_param = demo_data(base_dir)

    t0 = time.perf_counter()
    for i in range(tries):
        _, _, _, _ = determine_pose(source, 
                target, background, initial_fp_transformation,
                icp_param, icp=icp)
        logging.info(f"Try number {i}")
    logging.info(f"Average time for {tries} \
        iterations {(time.perf_counter()-t0)/tries}\n \
            Total time {(time.perf_counter()-t0)}")

if __name__ == "__main__":
    base_dir='demo_data/turbine'
    demo_icp(base_dir=base_dir)

from estimater import *
from datareader import *
from src import *
import argparse
import open3d as o3d
import cv2 
from multiprocessing import Queue

def main(args):
  
  # Create a queue to pass data between the Dash application and the estimator
  data_queue = Queue()
  capture_queue = Queue()
  # Start the Dash application in a separate thread
  dash_thread = threading.Thread(target=run_dash_app, 
                                 args=(data_queue, capture_queue,), daemon=True)
  dash_thread.start()
  
  # mesh = trimesh.load(args.mesh_file)
  mesh = trimesh.load(f"{args.test_scene_dir}/mesh/model_scaled_down.obj")
  
  # Set up debugging
  debug = args.debug
  debug_dir = args.debug_dir
  os.system(f'rm -rf {debug_dir}/* && mkdir -p {debug_dir}/track_vis {debug_dir}/ob_in_cam')

  # Compute the oriented bounding box of the mesh
  to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
  bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)

  # Initialize pose estimation models
  scorer = ScorePredictor()
  refiner = PoseRefinePredictor()
  glctx = dr.RasterizeCudaContext()
  
  # Initialize the estimator
  est = FoundationPose(model_pts=mesh.vertices, 
                       model_normals=mesh.vertex_normals, mesh=mesh,
                        scorer=scorer, refiner=refiner, debug_dir=debug_dir, 
                        debug=debug, glctx=glctx
                        )
  logging.info("Estimator initialization done")
  input("Press Enter to continue...")
  
  # Initialize the data reader
  if args.demo is True:
    # demo-data reader
    reader = DataReader(base_dir=args.test_scene_dir, 
                                shorter_side=args.shorter_side,
                                zfar=np.inf, arguments=args)
  else:
    # live-data reader for Kinect camera
    logging.info("live demo")
    reader = KinectReader(base_dir=args.test_scene_dir, 
                          capture_background=args.capture_background, 
                          shorter_side=args.shorter_side,
                          zfar=np.inf, arguments=args)
  
  #
  i = 0
  intersection_pcds = []
  detect_defect = False
  
  # Defect Detection logic (this is just a dummy heatmap, TODO: defect detection logic needs to be implemented)
  reader.update()
  heatmap, color_original, heatmap_vis, _= reader.get_heatmap(reader.get_color(i))
  overlay = create_heatmap_overlay(color_original, heatmap_vis)
  cv2.imwrite('src/assets/overlay.png', overlay)

  while True:
    logging.info(f'i: {i}')
    # Update the reader to get the latest data
    reader.update()
    color = reader.get_color(i)
    depth = reader.get_depth(i)
    source = reader.get_source(i)
    if color is None or depth is None:
      continue
    if i == 0: 
      # Initial pose registration
      mask = reader.get_mask(color, i).astype(bool)
      pose = est.register(K=reader.color_K, rgb=color, depth=depth,
                          ob_mask=mask, iteration=args.est_refine_iter)
      
      if debug >= 3:
        m = mesh.copy()
        m.apply_transform(pose)
        m.export(f'{debug_dir}/model_tf.obj')
        xyz_map = depth2xyzmap(depth, reader.color_K)
        valid = depth >= 0.001
        pcd = toOpen3dCloud(xyz_map[valid], color[valid])
        o3d.io.write_point_cloud(f'{debug_dir}/scene_complete.ply', pcd)
      
      # Convert pose to millimeters and compute initial transformation
      pose_in_mm = reader.scale_translation_to_millimeters(pose)
      initial_transformation = np.dot(reader.color_to_depth, pose_in_mm)
      
      # Perform Z-axis Allignment and ICP-refinement to refine the initial pose
      _, initial_icp_result, _, target_processed = refine_pose_with_icp(source, 
                            reader.target, reader.background, 
                            initial_transformation, reader.parameters)
      
      # Compute the relative transformation between the initial pose from FP and the final pose after refinement, so that we keep adjsuting for the error in the FP transformation when tracking
      delta_pose = np.linalg.inv(initial_transformation) @\
        np.linalg.inv(initial_icp_result.transformation)
        
      current_transformation = initial_icp_result.transformation
      
      target_mesh_copy = transform_object(reader.target_mesh, 
                        np.linalg.inv(initial_icp_result.transformation))
      
      # Perform ray tracing to find the intersection point between the target mesh and the heatmap
      defect_mesh_intersection_pcd, target_transformed = ray_tracing(
        reader.base_dir, target_mesh_copy, heatmap, 
        reader.color_pinhole, heatmap_threshold = 0.75
          )
      
      defect_mesh_intersection_pcd.transform(reader.color_to_depth)
      intersection_pcds.append(defect_mesh_intersection_pcd)

      if debug >= 2:
        save_overlay(overlay, save_path=f"debug/overlay/overlay_{i}.png")
        draw_registration_result(target_mesh_copy, 
                        defect_mesh_intersection_pcd, np.eye(4),
                        f"{i}_Mesh_Defect_Intersection", reader.parameters, 
                        debug_path="debug/results"
                        )
      # Store the transformation for future reference
      previous_transformation = initial_icp_result.transformation
      # Update the Dash application with new data
      update_dash_data(intersection_pcds, target_mesh_copy)
    else:
      # For subsequent frames, track the object pose
      pose = est.track_one(rgb=color, depth=depth, K=reader.color_K, 
                           iteration=args.track_refine_iter
                           )
      # Convert pose to millimeters and compute initial transformation
      pose_in_mm = reader.scale_translation_to_millimeters(pose)
      initial_transformation = np.dot(reader.color_to_depth, pose_in_mm)

      # Check for a new defect detectin command from the queue
      if not capture_queue.empty():
          capture_queue.get()
          detect_defect = True
          logging.info("New Defect Detectin initiated!")

      if detect_defect: 
          # Generate heatmap and overlay for the new frame
          heatmap, color_original, heatmap_vis, _ = \
            reader.get_heatmap(reader.get_color(i))
          overlay = create_heatmap_overlay(color_original, heatmap_vis)
          cv2.imwrite('src/assets/overlay.png', overlay)
          # Preprocess the source point cloud
          source_processed, _, _ = \
            preprocess_source(source, reader.background, reader.parameters, i=i
                              )
            
          if debug >= 2:
            save_overlay(overlay, save_path=f"debug/overlay/overlay_{i}.png")
            target_copy = copy.deepcopy(reader.target)
            target_copy.paint_uniform_color([1, 0.706, 0])  # Gold color
            source.paint_uniform_color([1, 0, 0])  
            draw_registration_result(source, target_copy, 
                np.linalg.inv(initial_transformation), 
                f'{i}_Target_Source_transformed', 
                reader.parameters, debug_path="debug/results")
          # Improve the result using the current source and target point clouds
          current_result = improve_result(source_processed, 
                            target_processed, initial_transformation, 
                            reader.parameters
                            )
          
          # Get the current transformation
          current_transformation = current_result.transformation
          # Compute the relative transformation between the initial pose and the refined pose, simply said: transformation error correction
          delta_pose = np.linalg.inv(initial_transformation) @\
            np.linalg.inv(current_transformation
                          )
          target_mesh_copy = transform_object(reader.target_mesh, 
                                    np.linalg.inv(current_transformation)
                                    )
          # Compute the relative transformation from the previous to the current frame (previous defect detection frame to current one)
          relative_transformation = \
            np.linalg.inv(current_transformation) @ previous_transformation

          # Perform ray tracing to generate the new intersection point cloud
          new_intersection_pcd, transformed_target_mesh = ray_tracing(
              reader.base_dir,
              target_mesh_copy,
              heatmap,
              reader.color_pinhole,
              heatmap_threshold=0.75
          )

          # Update existing intersection point clouds to align with the current frame
          for intersection_point_cloud in intersection_pcds:
              intersection_point_cloud.transform(relative_transformation)

          # Transform the new intersection point cloud to the depth camera's coordinate system
          new_intersection_pcd.transform(reader.color_to_depth)
          intersection_pcds.append(new_intersection_pcd)

          #Update the previous transformation for future reference
          previous_transformation = current_transformation
          # Update the Dash application with new data
          update_dash_data(intersection_pcds, target_mesh_copy)
          detect_defect = False
      else:
        # Update the current transformation with the delta_pose (transformation error correction)
        current_transformation = np.linalg.inv(initial_transformation @ delta_pose)
        # draw_reslut(source, reader.target, current_transformation, f"{i}")

    os.makedirs(f'{debug_dir}/ob_in_cam', exist_ok=True)
    np.savetxt(f'{debug_dir}/ob_in_cam/{i:04d}.txt', pose.reshape(4, 4))  
    
    if debug >= 1:
      center_pose = pose @ np.linalg.inv(to_origin)
      vis = draw_posed_3d_box(reader.color_K, img=color, 
                              ob_in_cam=center_pose, bbox=bbox)
      vis = draw_xyz_axis(color, ob_in_cam=center_pose, 
                          scale=0.1, K=reader.color_K,
                          thickness=3, transparency=0, is_input_rgb=True)
      cv2.namedWindow('Tracking', cv2.WINDOW_NORMAL)
      cv2.imshow('Tracking', vis[..., ::-1])
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
      print("Quitting...")
      break

    if debug >= 2:
      os.makedirs(f'{debug_dir}/track_vis', exist_ok=True)
      imageio.imwrite(f'{debug_dir}/track_vis/{i:04d}.png', vis)
        
    i += 1 
  reader.stop_camera()  # Stop the camera
  cv2.destroyAllWindows()  
  dash_thread.join()
   
if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  code_dir = os.path.dirname(os.path.realpath(__file__))
  parser.add_argument('--mesh_file', type=str, default=f'{code_dir}/demo_data/mustard0/mesh/textured_simple.obj')
  parser.add_argument('--test_scene_dir', type=str, default=f'{code_dir}/demo_data/mustard0')
  parser.add_argument('--est_refine_iter', type=int, default=5)
  parser.add_argument('--track_refine_iter', type=int, default=2)
  parser.add_argument('--debug', type=int, default=1)
  parser.add_argument('--debug_dir', type=str, default=f'{code_dir}/debug')
  parser.add_argument('--shorter_side', type=int, default=None)
  parser.add_argument('--demo', action='store_true', help="Run in demo mode (default is True).")
  parser.add_argument('--no-demo', dest='demo', action='store_false', help="Disable demo mode.")
  parser.add_argument('--icp', default=False, type=bool, help='Method for Pose Estimation')
  parser.add_argument('--info', default=True, type=bool, help = "set the logging level to INFO")
  parser.add_argument('--box', type=bool, help='Enable or disable background removal.')
  parser.add_argument('--mesh', type=bool, help='Enable or disable mesh.')
  parser.add_argument('--capture_background', type=bool, default=False, help='Capture a new background in case of a new camera position.')
  parser.add_argument('--voxel_size', type=float, help='Set voxel size.')

  args = parser.parse_args()

  set_logging_format()
  set_seed(0)
  main(args)

#python run.py --test_scene_dir /home/z/BA/2d_3d_pe_dp1/demo_data/000020 --mesh_file /home/z/BA/2d_3d_pe_dp1/demo_data/000020/mesh/model_scaled_down.obj --shorter_side 288 --demo
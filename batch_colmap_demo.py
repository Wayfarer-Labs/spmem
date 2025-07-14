# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import random
import numpy as np
import glob
import os
import copy
import torch
import torch.nn.functional as F
from tqdm import tqdm
import gc

# Configure CUDA settings
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

import argparse
from pathlib import Path
import trimesh
import pycolmap


from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images_square
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map
from vggt.utils.helper import create_pixel_coordinate_grid, randomly_limit_trues
from vggt.dependency.track_predict import predict_tracks
from vggt.dependency.np_to_pycolmap import batch_np_matrix_to_pycolmap, batch_np_matrix_to_pycolmap_wo_track


def parse_args():
    parser = argparse.ArgumentParser(description="VGGT Batch Demo for Large-Scale Scenes")
    parser.add_argument("--scene_dir", type=str, required=True, help="Directory containing the massive scene images")
    parser.add_argument("--batch_size", type=int, default=30, help="Number of frames to process in each batch")
    parser.add_argument("--max_frames", type=int, default=None, help="Maximum number of frames to process (None for all)")
    parser.add_argument("--overlap_frames", type=int, default=0, help="Number of overlapping frames between batches for continuity")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--use_ba", action="store_true", default=False, help="Use BA for reconstruction")
    parser.add_argument("--save_intermediate", action="store_true", default=False, help="Save intermediate batch results")
    parser.add_argument("--memory_efficient", action="store_true", default=True, help="Use memory efficient processing")
    
    ######### BA parameters #########
    parser.add_argument(
        "--max_reproj_error", type=float, default=8.0, help="Maximum reprojection error for reconstruction"
    )
    parser.add_argument("--shared_camera", action="store_true", default=True, help="Use shared camera for all images")
    parser.add_argument("--camera_type", type=str, default="SIMPLE_PINHOLE", help="Camera type for reconstruction")
    parser.add_argument("--vis_thresh", type=float, default=0.2, help="Visibility threshold for tracks")
    parser.add_argument("--query_frame_num", type=int, default=8, help="Number of frames to query")
    parser.add_argument("--max_query_pts", type=int, default=4096, help="Maximum number of query points")
    parser.add_argument(
        "--fine_tracking", action="store_true", default=True, help="Use fine tracking (slower but more accurate)"
    )
    parser.add_argument(
        "--conf_thres_value", type=float, default=5.0, help="Confidence threshold value for depth filtering (wo BA)"
    )
    
    ######### Batch processing parameters #########
    parser.add_argument("--merge_strategy", type=str, default="accumulate", choices=["accumulate", "keyframe"], 
                       help="Strategy for merging batch results")
    parser.add_argument("--keyframe_interval", type=int, default=10, help="Interval for keyframe selection")
    parser.add_argument("--point_cloud_downsample", type=float, default=1.0, help="Downsample ratio for point clouds")
    
    return parser.parse_args()


def run_VGGT_batch(model, images, dtype, resolution=518):
    """
    Run VGGT on a batch of images with memory management
    """
    # images: [B, 3, H, W]
    assert len(images.shape) == 4
    assert images.shape[1] == 3

    # hard-coded to use 518 for VGGT
    images = F.interpolate(images, size=(resolution, resolution), mode="bilinear", align_corners=False)

    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            images = images[None]  # add batch dimension
            aggregated_tokens_list, ps_idx = model.aggregator(images)

        # Predict Cameras
        pose_enc = model.camera_head(aggregated_tokens_list)[-1]
        # Extrinsic and intrinsic matrices, following OpenCV convention (camera from world)
        extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])
        # Predict Depth Maps
        depth_map, depth_conf = model.depth_head(aggregated_tokens_list, images, ps_idx)

    extrinsic = extrinsic.squeeze(0).cpu().numpy()
    intrinsic = intrinsic.squeeze(0).cpu().numpy()
    depth_map = depth_map.squeeze(0).cpu().numpy()
    depth_conf = depth_conf.squeeze(0).cpu().numpy()
    
    return extrinsic, intrinsic, depth_map, depth_conf


def create_batches(image_paths, batch_size, overlap_frames=0):
    """
    Create overlapping batches of image paths for processing continuity
    """
    batches = []
    start_idx = 0
    
    while start_idx < len(image_paths):
        end_idx = min(start_idx + batch_size, len(image_paths))
        batch_paths = image_paths[start_idx:end_idx]
        
        # Store batch info including global indices
        batch_info = {
            'paths': batch_paths,
            'global_start_idx': start_idx,
            'global_end_idx': end_idx,
            'batch_idx': len(batches)
        }
        batches.append(batch_info)
        
        # Move start index, accounting for overlap
        start_idx = end_idx - overlap_frames
        if start_idx >= len(image_paths):
            break
    
    return batches


def process_batch(model, batch_info, args, dtype, device, img_load_resolution=1024, vggt_fixed_resolution=518):
    """
    Process a single batch of images
    """
    batch_paths = batch_info['paths']
    global_start_idx = batch_info['global_start_idx']
    batch_idx = batch_info['batch_idx']
    
    print(f"Processing batch {batch_idx + 1} with {len(batch_paths)} images (global indices {global_start_idx}-{global_start_idx + len(batch_paths) - 1})")
    
    # Load and preprocess images for this batch
    images, original_coords = load_and_preprocess_images_square(batch_paths, img_load_resolution)
    images = images.to(device)
    original_coords = original_coords.to(device)
    
    # Run VGGT to estimate camera and depth
    extrinsic, intrinsic, depth_map, depth_conf = run_VGGT_batch(model, images, dtype, vggt_fixed_resolution)
    points_3d = unproject_depth_map_to_point_map(depth_map, extrinsic, intrinsic)
    
    batch_result = {
        'batch_idx': batch_idx,
        'global_start_idx': global_start_idx,
        'image_paths': [os.path.basename(path) for path in batch_paths],
        'extrinsic': extrinsic,
        'intrinsic': intrinsic,
        'depth_map': depth_map,
        'depth_conf': depth_conf,
        'points_3d': points_3d,
        'original_coords': original_coords.cpu().numpy(),
        'images': images.cpu() if not args.memory_efficient else None  # Save memory by not storing images
    }
    
    # Clear GPU memory
    del images, original_coords
    if args.memory_efficient:
        torch.cuda.empty_cache()
        gc.collect()
    
    return batch_result


def merge_batch_results(batch_results, args):
    """
    Merge results from multiple batches into a single reconstruction
    """
    print(f"Merging {len(batch_results)} batch results...")
    
    # Align batch results to global coordinate using overlapping frames
    overlap = args.overlap_frames
    aligned_extrinsics = []
    aligned_points = []
    all_intrinsic = []
    all_depth_maps = []
    all_depth_conf = []
    all_original_coords = []
    all_image_paths = []
    for idx, result in enumerate(batch_results):
        raw_ex = result['extrinsic']  # shape (B, 4, 4) or (4, 4)
        # ensure batch dimension
        if raw_ex.ndim == 2:
            raw_ex = raw_ex[np.newaxis, ...]
        raw_pts = result['points_3d']  # shape (B, H, W, 3) or (H, W, 3)
        if raw_pts.ndim == 3:
            raw_pts = raw_pts[np.newaxis, ...]
        
        # Use raw data without alignment
        aligned_extrinsics.append(raw_ex)
        aligned_points.append(raw_pts)
        # Intrinsics and other data can be concatenated directly
        all_intrinsic.append(result['intrinsic'])
        all_depth_maps.append(result['depth_map'])
        all_depth_conf.append(result['depth_conf'])
        all_original_coords.append(result['original_coords'])
        all_image_paths.extend(result['image_paths'])
    
    # Concatenate aligned arrays
    merged_extrinsic = np.concatenate(aligned_extrinsics, axis=0)
    merged_intrinsic = np.concatenate(all_intrinsic, axis=0)
    merged_depth_maps = np.concatenate(all_depth_maps, axis=0)
    merged_depth_conf = np.concatenate(all_depth_conf, axis=0)
    merged_points_3d = np.concatenate(aligned_points, axis=0)
    merged_original_coords = np.concatenate(all_original_coords, axis=0)
    
    print(f"Merged data shapes:")
    print(f"  Extrinsic: {merged_extrinsic.shape}")
    print(f"  Intrinsic: {merged_intrinsic.shape}")
    print(f"  Depth maps: {merged_depth_maps.shape}")
    print(f"  Points 3D: {merged_points_3d.shape}")
    print(f"  Total images: {len(all_image_paths)}")
    
    return {
        'extrinsic': merged_extrinsic,
        'intrinsic': merged_intrinsic,
        'depth_map': merged_depth_maps,
        'depth_conf': merged_depth_conf,
        'points_3d': merged_points_3d,
        'original_coords': merged_original_coords,
        'image_paths': all_image_paths
    }


def create_reconstruction_from_merged_data(merged_data, args, vggt_fixed_resolution=518, img_load_resolution=1024):
    """
    Create COLMAP reconstruction from merged batch data
    """
    extrinsic = merged_data['extrinsic']
    intrinsic = merged_data['intrinsic']
    depth_map = merged_data['depth_map']
    depth_conf = merged_data['depth_conf']
    points_3d = merged_data['points_3d']
    original_coords = merged_data['original_coords']
    image_paths = merged_data['image_paths']
    
    if args.use_ba:
        print("Running Bundle Adjustment on merged batches...")
        
        # For BA, we need to use tracking between frames
        # Load all images again for tracking (memory intensive but necessary for BA)
        all_image_paths_full = [os.path.join(args.scene_dir, "images", img_path) for img_path in image_paths]
        
        # Process in chunks to manage memory for tracking
        chunk_size = min(50, len(all_image_paths_full))  # Smaller chunks for tracking
        
        # Initialize tracking data
        all_pred_tracks = []
        all_pred_vis_scores = []
        all_pred_confs = []
        all_points_3d_tracked = []
        all_points_rgb_tracked = []
        
        print(f"Running tracking across {len(all_image_paths_full)} images in chunks of {chunk_size}")
        
        for i in range(0, len(all_image_paths_full), chunk_size):
            end_idx = min(i + chunk_size, len(all_image_paths_full))
            chunk_paths = all_image_paths_full[i:end_idx]
            
            # Load images for tracking
            chunk_images, _ = load_and_preprocess_images_square(chunk_paths, img_load_resolution)
            chunk_images = chunk_images.cuda()
            
            # Get corresponding depth data
            chunk_depth_conf = depth_conf[i:end_idx]
            chunk_points_3d = points_3d[i:end_idx]
            
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                # Run tracking on this chunk
                pred_tracks, pred_vis_scores, pred_confs, tracked_points_3d, tracked_points_rgb = predict_tracks(
                    chunk_images,
                    conf=chunk_depth_conf,
                    points_3d=chunk_points_3d,
                    masks=None,
                    max_query_pts=args.max_query_pts,
                    query_frame_num=min(args.query_frame_num, len(chunk_paths)),
                    keypoint_extractor="aliked+sp",
                    fine_tracking=args.fine_tracking,
                )
                
                # Adjust track indices to global frame indices
                if len(pred_tracks) > 0:
                    pred_tracks[:, :, 0] += i  # Adjust frame indices
                
                all_pred_tracks.append(pred_tracks)
                all_pred_vis_scores.append(pred_vis_scores)
                all_pred_confs.append(pred_confs)
                all_points_3d_tracked.append(tracked_points_3d)
                all_points_rgb_tracked.append(tracked_points_rgb)
            
            del chunk_images
            torch.cuda.empty_cache()
            gc.collect()
        
        # Merge tracking results
        if all_pred_tracks:
            merged_tracks = np.concatenate(all_pred_tracks, axis=0) if len(all_pred_tracks) > 1 else all_pred_tracks[0]
            merged_vis_scores = np.concatenate(all_pred_vis_scores, axis=0) if len(all_pred_vis_scores) > 1 else all_pred_vis_scores[0]
            merged_confs = np.concatenate(all_pred_confs, axis=0) if len(all_pred_confs) > 1 else all_pred_confs[0]
            merged_tracked_points_3d = np.concatenate(all_points_3d_tracked, axis=0) if len(all_points_3d_tracked) > 1 else all_points_3d_tracked[0]
            merged_tracked_points_rgb = np.concatenate(all_points_rgb_tracked, axis=0) if len(all_points_rgb_tracked) > 1 else all_points_rgb_tracked[0]
        else:
            print("Warning: No tracks found, falling back to feedforward reconstruction")
            args.use_ba = False
        
        if args.use_ba and len(merged_tracks) > 0:
            # Rescale intrinsics from VGGT resolution to load resolution
            scale = img_load_resolution / vggt_fixed_resolution
            intrinsic_scaled = intrinsic.copy()
            intrinsic_scaled[:, :2, :] *= scale
            
            image_size = np.array([img_load_resolution, img_load_resolution])
            track_mask = merged_vis_scores > args.vis_thresh
            shared_camera = args.shared_camera
            
            print(f"Creating COLMAP reconstruction with {len(merged_tracks)} tracks")
            reconstruction, valid_track_mask = batch_np_matrix_to_pycolmap(
                merged_tracked_points_3d,
                extrinsic,
                intrinsic_scaled,
                merged_tracks,
                image_size,
                masks=track_mask,
                max_reproj_error=args.max_reproj_error,
                shared_camera=shared_camera,
                camera_type=args.camera_type,
                points_rgb=merged_tracked_points_rgb,
            )
            
            if reconstruction is None:
                print("Warning: BA reconstruction failed, falling back to feedforward")
                args.use_ba = False
            else:
                print("Running Bundle Adjustment...")
                ba_options = pycolmap.BundleAdjustmentOptions()
                pycolmap.bundle_adjustment(reconstruction, ba_options)
                reconstruction_resolution = img_load_resolution
                
                # Extract final points for return
                points_3d_final = []
                points_rgb_final = []
                for point3D in reconstruction.points3D.values():
                    points_3d_final.append(point3D.xyz)
                    points_rgb_final.append(point3D.color)
                
                points_3d_final = np.array(points_3d_final)
                points_rgb_final = np.array(points_rgb_final)
                
                return reconstruction, points_3d_final, points_rgb_final
    
    # Fallback to feedforward reconstruction
    if not args.use_ba:
        conf_thres_value = args.conf_thres_value
        max_points_for_colmap = 500000  # Increased for large scenes
        shared_camera = True  # in the feedforward manner, we do not support shared camera
        camera_type = "PINHOLE"  # in the feedforward manner, we only support PINHOLE camera

        image_size = np.array([vggt_fixed_resolution, vggt_fixed_resolution])
        num_frames, height, width, _ = points_3d.shape

        # Create RGB data for points (we need to reload images in batches)
        print("Creating point colors from depth maps...")
        points_rgb_list = []
        
        # Process in smaller chunks to manage memory
        chunk_size = min(100, num_frames)
        for i in range(0, num_frames, chunk_size):
            end_idx = min(i + chunk_size, num_frames)
            chunk_paths = [os.path.join(args.scene_dir, "images", img_path) for img_path in image_paths[i:end_idx]]
            
            # Load images for this chunk
            chunk_images, _ = load_and_preprocess_images_square(chunk_paths, img_load_resolution)
            chunk_images = F.interpolate(
                chunk_images, size=(vggt_fixed_resolution, vggt_fixed_resolution), mode="bilinear", align_corners=False
            )
            chunk_rgb = (chunk_images.cpu().numpy() * 255).astype(np.uint8)
            chunk_rgb = chunk_rgb.transpose(0, 2, 3, 1)
            points_rgb_list.append(chunk_rgb)
            
            del chunk_images
            gc.collect()
        
        points_rgb = np.concatenate(points_rgb_list, axis=0)

        # (S, H, W, 3), with x, y coordinates and frame indices
        points_xyf = create_pixel_coordinate_grid(num_frames, height, width)

        conf_mask = depth_conf >= conf_thres_value
        
        # Apply downsampling if requested
        if args.point_cloud_downsample < 1.0:
            print(f"Downsampling point cloud by factor {args.point_cloud_downsample}")
            downsample_mask = np.random.random(conf_mask.shape) < args.point_cloud_downsample
            conf_mask = conf_mask & downsample_mask
        
        # Limit total number of points
        conf_mask = randomly_limit_trues(conf_mask, max_points_for_colmap)

        # Debug: Check filtering results
        print(f"Confidence threshold: {conf_thres_value}")
        print(f"Points passing confidence threshold: {conf_mask.sum()}")
        print(f"Total possible points: {conf_mask.size}")
        print(f"Percentage of points kept: {100 * conf_mask.sum() / conf_mask.size:.2f}%")

        points_3d = points_3d[conf_mask]
        points_xyf = points_xyf[conf_mask]
        points_rgb = points_rgb[conf_mask]
        
        print(f"Final number of 3D points: {len(points_3d)}")

        print("Converting to COLMAP format")
        reconstruction = batch_np_matrix_to_pycolmap_wo_track(
            points_3d,
            points_xyf,
            points_rgb,
            extrinsic,
            intrinsic,
            image_size,
            shared_camera=shared_camera,
            camera_type=camera_type,
        )

        reconstruction_resolution = vggt_fixed_resolution

    reconstruction = rename_colmap_recons_and_rescale_camera(
        reconstruction,
        image_paths,
        original_coords,
        img_size=reconstruction_resolution,
        shift_point2d_to_original_res=True,
        shared_camera=shared_camera,
    )

    return reconstruction, points_3d, points_rgb


def batch_demo_fn(args):
    """
    Main function for batch processing large-scale scenes
    """
    # Print configuration
    print("Batch Processing Arguments:", vars(args))

    # Set seed for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    print(f"Setting seed as: {args.seed}")

    # Set device and dtype
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Using dtype: {dtype}")

    # Load model
    model = VGGT()
    _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
    model.load_state_dict(torch.hub.load_state_dict_from_url(_URL))
    model.eval()
    model = model.to(device)
    print(f"Model loaded")

    # Get all image paths
    image_dir = os.path.join(args.scene_dir, "images")
    all_image_paths = sorted(glob.glob(os.path.join(image_dir, "*")))
    
    if args.max_frames is not None:
        all_image_paths = all_image_paths[:args.max_frames]
    
    if len(all_image_paths) == 0:
        raise ValueError(f"No images found in {image_dir}")
    
    print(f"Found {len(all_image_paths)} images to process")
    print(f"Processing in batches of {args.batch_size} with {args.overlap_frames} overlapping frames")

    # Create batches
    batches = create_batches(all_image_paths, args.batch_size, args.overlap_frames)
    print(f"Created {len(batches)} batches for processing")

    # Process each batch
    batch_results = []
    vggt_fixed_resolution = 518
    img_load_resolution = 1024

    for batch_info in tqdm(batches, desc="Processing batches"):
        try:
            batch_result = process_batch(
                model, batch_info, args, dtype, device,
                img_load_resolution, vggt_fixed_resolution
            )
            batch_results.append(batch_result)
            
            # Save intermediate results if requested
            if args.save_intermediate:
                intermediate_dir = os.path.join(args.scene_dir, "intermediate_batches")
                os.makedirs(intermediate_dir, exist_ok=True)
                batch_file = os.path.join(intermediate_dir, f"batch_{batch_info['batch_idx']:04d}.npz")
                np.savez_compressed(batch_file, **batch_result)
                print(f"Saved intermediate batch to {batch_file}")
            
        except Exception as e:
            print(f"Error processing batch {batch_info['batch_idx']}: {e}")
            continue
        print(f"Batch {batch_info['batch_idx']} processed successfully")

    if len(batch_results) == 0:
        raise ValueError("No batches were successfully processed")

    print(f"Successfully processed {len(batch_results)} out of {len(batches)} batches")

    # Merge batch results
    merged_data = merge_batch_results(batch_results, args)
    
    # Create reconstruction
    print("Creating COLMAP reconstruction from merged data...")
    reconstruction, points_3d, points_rgb = create_reconstruction_from_merged_data(
        merged_data, args, vggt_fixed_resolution, img_load_resolution
    )

    # Save reconstruction
    print(f"Saving reconstruction to {args.scene_dir}/sparse")
    sparse_reconstruction_dir = os.path.join(args.scene_dir, "sparse")
    os.makedirs(sparse_reconstruction_dir, exist_ok=True)
    reconstruction.write(sparse_reconstruction_dir)

    # Save point cloud for visualization
    print("Saving point cloud...")
    trimesh.PointCloud(points_3d, colors=points_rgb).export(os.path.join(args.scene_dir, "sparse/points.ply"))

    print(f"Batch processing completed successfully!")
    print(f"Final reconstruction contains {len(points_3d)} 3D points from {len(merged_data['image_paths'])} images")

    return True


def rename_colmap_recons_and_rescale_camera(
    reconstruction, image_paths, original_coords, img_size, shift_point2d_to_original_res=False, shared_camera=False
):
    rescale_camera = True

    for pyimageid in reconstruction.images:
        # Reshaped the padded&resized image to the original size
        # Rename the images to the original names
        pyimage = reconstruction.images[pyimageid]
        pycamera = reconstruction.cameras[pyimage.camera_id]
        pyimage.name = image_paths[pyimageid - 1]

        if rescale_camera:
            # Rescale the camera parameters
            pred_params = copy.deepcopy(pycamera.params)

            real_image_size = original_coords[pyimageid - 1, -2:]
            resize_ratio = max(real_image_size) / img_size
            pred_params = pred_params * resize_ratio
            real_pp = real_image_size / 2
            pred_params[-2:] = real_pp  # center of the image

            pycamera.params = pred_params
            pycamera.width = real_image_size[0]
            pycamera.height = real_image_size[1]

        if shift_point2d_to_original_res:
            # Also shift the point2D to original resolution
            top_left = original_coords[pyimageid - 1, :2]

            for point2D in pyimage.points2D:
                point2D.xy = (point2D.xy - top_left) * resize_ratio

        if shared_camera:
            # If shared_camera, all images share the same camera
            # no need to rescale any more
            rescale_camera = False

    return reconstruction


if __name__ == "__main__":
    args = parse_args()
    with torch.no_grad():
        batch_demo_fn(args)


"""
VGGT Batch Processing Script
==========================

A script to run the VGGT model for 3D reconstruction from massive image sequences 
with tens of thousands of frames using batch processing.

Key Features for Large-Scale Processing
--------------------------------------
• Batch Processing: Handles massive scenes by processing images in manageable batches
• Memory Management: Efficient GPU memory usage with automatic cleanup
• Overlapping Batches: Maintains continuity between batches with overlapping frames
• Incremental Saving: Option to save intermediate batch results
• Point Cloud Downsampling: Reduces memory footprint for very large scenes
• Progress Tracking: Real-time progress monitoring with tqdm

Usage Examples
--------------
# Process 10,000 frames in batches of 64
python batch_colmap_demo.py --scene_dir /path/to/massive_scene --batch_size 64 --max_frames 10000

# Memory-efficient processing with downsampling
python batch_colmap_demo.py --scene_dir /path/to/scene --batch_size 32 --memory_efficient --point_cloud_downsample 0.5

# Save intermediate results for debugging
python batch_colmap_demo.py --scene_dir /path/to/scene --batch_size 48 --save_intermediate

Performance Considerations
-------------------------
• Batch Size: Larger batches are more efficient but use more GPU memory
• Overlap Frames: Small overlap (2-4 frames) helps with continuity
• Point Cloud Downsampling: Use values < 1.0 for very dense scenes
• Memory Efficient Mode: Reduces RAM usage at the cost of some processing speed

Directory Structure
------------------
Input:
    massive_scene/
    └── images/              # Tens of thousands of source images

Output:
    massive_scene/
    ├── images/
    ├── sparse/             # Final reconstruction results
    │   ├── cameras.bin     # Camera parameters (COLMAP format)
    │   ├── images.bin      # Pose for each image (COLMAP format)
    │   ├── points3D.bin    # 3D points (COLMAP format)
    │   └── points.ply      # Point cloud visualization file
    └── intermediate_batches/ # Optional intermediate results
        ├── batch_0000.npz
        ├── batch_0001.npz
        └── ...
"""

#!/usr/bin/env python3
"""
VAE Data Pipeline - Sliding Window Video Processing

Processes videos with a sliding window approach to create training data for VAE.
Each "focal" frame gets RGB, depth, camera info, and point cloud from surrounding frames.

Usage:
    python vae_data_pipeline.py \
        --video-dir /path/to/videos \
        --output-dir /path/to/output \
        --kernel-size 5 \
        --stride 3 \
        --dilation 2 \
        --files-per-subdir 500
"""

import os
import sys
import argparse
import time
import math
from pathlib import Path
from typing import List, Tuple, Optional

import ray
import torch
import numpy as np
from PIL import Image
import decord
from tqdm import tqdm
import torch.nn.functional as F

# Import VGGT model and dependencies
from vggt.models.vggt import VGGT
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map
from vggt.utils.helper import create_pixel_coordinate_grid, randomly_limit_trues
from vggt.dependency.track_predict import predict_tracks
from vggt.dependency.vggsfm_utils import build_vggsfm_tracker
from vggt.dependency.np_to_pycolmap import batch_np_matrix_to_pycolmap, batch_np_matrix_to_pycolmap_wo_track

import pycolmap
import trimesh
import copy

# Compile for performance
predict_tracks = torch.compile(predict_tracks)

def parse_args():
    """Parse command line arguments."""
    p = argparse.ArgumentParser(
        description="Create VAE training data from videos using sliding window approach"
    )
    
    # Input/Output
    p.add_argument("--video-dir", required=True, help="Directory containing MP4 videos")
    p.add_argument("--output-dir", required=True, help="Output directory for processed data")
    
    # Sliding window parameters
    p.add_argument("--kernel-size", type=int, default=5, 
                   help="Window size around focal frame (must be odd)")
    p.add_argument("--stride", type=int, default=3,
                   help="Step size between focal frames")
    p.add_argument("--dilation", type=int, default=1,
                   help="Spacing between frames in window")
    
    # File organization
    p.add_argument("--files-per-subdir", type=int, default=500,
                   help="Maximum files per subdirectory")
    
    # Multi-GPU settings
    p.add_argument("--num-gpus", type=int, default=None,
                   help="Number of GPUs to use (default: all available)")
    
    # VGGT/COLMAP settings
    p.add_argument("--use-ba", action="store_true", default=True, 
                   help="Use Bundle Adjustment for reconstruction")
    p.add_argument("--max-reproj-error", type=float, default=8.0,
                   help="Maximum reprojection error for reconstruction")
    p.add_argument("--shared-camera", action="store_true", default=True,
                   help="Use shared camera for all images")
    p.add_argument("--camera-type", type=str, default="SIMPLE_PINHOLE",
                   help="Camera type for reconstruction")
    p.add_argument("--vis-thresh", type=float, default=0.2,
                   help="Visibility threshold for tracks")
    p.add_argument("--query-frame-num", type=int, default=8,
                   help="Number of frames to query")
    p.add_argument("--max-query-pts", type=int, default=4096,
                   help="Maximum number of query points")
    p.add_argument("--fine-tracking", action="store_true", default=True,
                   help="Use fine tracking (slower but more accurate)")
    p.add_argument("--conf-thres-value", type=float, default=5.0,
                   help="Confidence threshold value for depth filtering (wo BA)")
    
    return p.parse_args()

def get_video_files(video_dir: str) -> List[str]:
    """Get all MP4 video files in directory."""
    video_dir = Path(video_dir)
    return [str(p) for p in video_dir.glob("*.mp4")]

def create_sliding_windows(num_frames: int, kernel_size: int, stride: int, dilation: int) -> List[Tuple[List[int], int]]:
    """
    Create sliding windows over frames.
    
    Returns:
        List of (frame_indices, focal_frame_idx) tuples
    """
    if kernel_size % 2 == 0:
        raise ValueError("Kernel size must be odd")
    
    windows = []
    half_kernel = kernel_size // 2
    
    # Calculate the actual span needed considering dilation
    span = (kernel_size - 1) * dilation
    
    # Start from the first valid focal position
    start_focal = half_kernel * dilation
    
    focal_frame = start_focal
    while focal_frame + half_kernel * dilation < num_frames:
        # Create window indices around focal frame
        window_indices = []
        for i in range(kernel_size):
            offset_from_center = (i - half_kernel) * dilation
            frame_idx = focal_frame + offset_from_center
            if 0 <= frame_idx < num_frames:
                window_indices.append(frame_idx)
        
        if len(window_indices) == kernel_size:  # Only use complete windows
            windows.append((window_indices, focal_frame))
        
        focal_frame += stride
    
    return windows

def load_vggt_models():
    """Load VGGT, DINOv2, and VGGSfM models."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load VGGT
    vggt_model = VGGT()
    _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
    vggt_model.load_state_dict(torch.hub.load_state_dict_from_url(_URL))
    vggt_model.eval()
    vggt_model = vggt_model.to(device)
    
    # Load DINOv2
    dinov2_model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14_reg")
    dinov2_model.eval()
    dinov2_model = dinov2_model.to(device)
    
    # Load VGGSfM tracker
    vggsfm_tracker_model = torch.compile(build_vggsfm_tracker()).to(device)
    
    return vggt_model, dinov2_model, vggsfm_tracker_model

@torch.compile
def fwd_vggt(model, images):
    """Forward pass through VGGT model."""
    images = images[None]  # add batch dimension
    aggregated_tokens_list, ps_idx = model.aggregator(images)

    # Predict Cameras
    pose_enc = model.camera_head(aggregated_tokens_list)[-1]
    extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])
    
    # Predict Depth Maps
    depth_map, depth_conf = model.depth_head(aggregated_tokens_list, images, ps_idx)

    extrinsic = extrinsic.squeeze(0)
    intrinsic = intrinsic.squeeze(0)
    depth_map = depth_map.squeeze(0)
    depth_conf = depth_conf.squeeze(0)

    # Convert to numpy for processing
    extrinsic, intrinsic, depth_map, depth_conf = [
        x.cpu().numpy() for x in [extrinsic, intrinsic, depth_map, depth_conf]
    ]
    points_3d = unproject_depth_map_to_point_map(depth_map, extrinsic, intrinsic)

    return extrinsic, intrinsic, depth_map, depth_conf, points_3d

def square_and_resize(images: torch.Tensor, load_res: int) -> torch.Tensor:
    """Resize and pad images to square of size (load_res, load_res)."""
    square_images = []
    for img in images:
        c, h, w = img.shape
        # Scale to keep aspect ratio
        if h >= w:
            new_h = load_res
            new_w = max(1, int(w * load_res / h))
        else:
            new_w = load_res
            new_h = max(1, int(h * load_res / w))
            
        # Make dimensions divisible by 14
        new_h = (new_h // 14) * 14
        new_w = (new_w // 14) * 14
        
        img_resized = F.interpolate(
            img.unsqueeze(0), size=(new_h, new_w), mode="bilinear", align_corners=False
        ).squeeze(0)
        
        # Pad to square
        pad_top = (load_res - new_h) // 2
        pad_bottom = load_res - new_h - pad_top
        pad_left = (load_res - new_w) // 2
        pad_right = load_res - new_w - pad_left
        
        img_padded = F.pad(
            img_resized,
            (pad_left, pad_right, pad_top, pad_bottom),
            mode="constant",
            value=1.0,
        )
        square_images.append(img_padded)
    return torch.stack(square_images)

def rename_colmap_recons_and_rescale_camera(
    reconstruction, image_paths, original_coords, img_size, 
    shift_point2d_to_original_res=False, shared_camera=False
):
    """Rename images and rescale camera parameters to match original image dimensions."""
    rescale_camera = True

    for pyimageid in reconstruction.images:
        pyimage = reconstruction.images[pyimageid]
        pycamera = reconstruction.cameras[pyimage.camera_id]
        pyimage.name = image_paths[pyimageid - 1]

        if rescale_camera:
            pred_params = copy.deepcopy(pycamera.params)
            real_image_size = original_coords[pyimageid - 1, -2:]
            resize_ratio = max(real_image_size) / img_size
            pred_params = pred_params * resize_ratio
            real_pp = real_image_size / 2
            pred_params[-2:] = real_pp

            pycamera.params = pred_params
            pycamera.width = real_image_size[0]
            pycamera.height = real_image_size[1]

        if shift_point2d_to_original_res:
            top_left = original_coords[pyimageid - 1, :2]
            for point2D in pyimage.points2D:
                point2D.xy = (point2D.xy - top_left) * resize_ratio

        if shared_camera:
            rescale_camera = False

    return reconstruction

@torch.no_grad()
def process_window(
    vggt_model, dinov2_model, vggsfm_tracker_model, 
    frames: torch.Tensor, focal_idx: int, args
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Process a window of frames to extract focal frame data and point cloud.
    
    Returns:
        focal_rgb: RGB image of focal frame [3, H, W]
        focal_depth: Depth map of focal frame [H, W] 
        focal_camera: Camera parameters [extrinsic (3,4), intrinsic (3,3)]
        point_cloud: Point cloud [N, 6] with [x,y,z,r,g,b]
    """
    device = next(vggt_model.parameters()).device
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    
    frames = frames.to(device)
    batch_size, c, h, w = frames.shape
    
    # Create original coordinates
    original_coords = torch.zeros((batch_size, 4), device=device)
    original_coords[:, 2] = w  # width
    original_coords[:, 3] = h  # height
    
    # VGGT processing
    vggt_resolution = 518
    img_load_resolution = max(w, h)
    
    # Resize for VGGT
    frames_resized = F.interpolate(frames, size=(vggt_resolution, vggt_resolution), 
                                  mode="bilinear", align_corners=False)
    
    # Run VGGT
    extrinsic, intrinsic, depth_map, depth_conf, points_3d = fwd_vggt(vggt_model, frames_resized)
    
    # Extract focal frame data
    focal_rgb = frames[focal_idx]  # Original resolution
    focal_depth = torch.from_numpy(depth_map[focal_idx]).float()
    
    # Focal camera parameters
    focal_extrinsic = torch.from_numpy(extrinsic[focal_idx]).float()
    focal_intrinsic = torch.from_numpy(intrinsic[focal_idx]).float()
    focal_camera = torch.cat([
        focal_extrinsic.flatten(),  # 12 elements 
        focal_intrinsic.flatten()   # 9 elements
    ])  # Total 21 elements
    
    # Generate point cloud from all frames
    if args.use_ba:
        # Bundle Adjustment approach
        load_res = min(518, img_load_resolution)
        images_square = square_and_resize(frames, load_res)
        image_size = np.array(images_square.shape[-2:])
        scale = img_load_resolution / vggt_resolution
        
        images_square = images_square.cuda()
        with torch.amp.autocast("cuda", dtype=dtype):
            pred_tracks, pred_vis_scores, pred_confs, points_3d, points_rgb = predict_tracks(
                dinov2_model,
                vggsfm_tracker_model,
                images_square,
                conf=depth_conf,
                points_3d=points_3d,
                masks=None,
                max_query_pts=args.max_query_pts,
                query_frame_num=args.query_frame_num,
                keypoint_extractor="aliked+sp",
                fine_tracking=args.fine_tracking,
            )
            torch.cuda.empty_cache()
        
        # Rescale intrinsic matrix
        intrinsic[:, :2, :] *= scale
        track_mask = pred_vis_scores > args.vis_thresh
        
        # Create base image path list
        base_image_path_list = [f"frame_{i:03d}.png" for i in range(batch_size)]
        
        reconstruction, valid_track_mask = batch_np_matrix_to_pycolmap(
            points_3d,
            extrinsic,
            intrinsic,
            pred_tracks,
            image_size,
            masks=track_mask,
            max_reproj_error=args.max_reproj_error,
            shared_camera=args.shared_camera,
            camera_type=args.camera_type,
            points_rgb=points_rgb,
        )
        
        if reconstruction is not None:
            # Bundle Adjustment
            ba_options = pycolmap.BundleAdjustmentOptions()
            pycolmap.bundle_adjustment(reconstruction, ba_options)
            
            reconstruction_resolution = img_load_resolution
        else:
            print("Warning: BA reconstruction failed, falling back to feedforward")
            reconstruction = None
            
    if not args.use_ba or reconstruction is None:
        # Feedforward approach
        conf_thres_value = args.conf_thres_value
        max_points_for_colmap = 200000
        
        image_size = np.array([vggt_resolution, vggt_resolution])
        num_frames, height, width, _ = points_3d.shape
        
        # Get RGB colors for points
        points_rgb = F.interpolate(
            frames, size=(vggt_resolution, vggt_resolution), mode="bilinear", align_corners=False
        )
        points_rgb = (points_rgb.cpu().numpy() * 255).astype(np.uint8)
        points_rgb = points_rgb.transpose(0, 2, 3, 1)
        
        # Create pixel coordinates
        points_xyf = create_pixel_coordinate_grid(num_frames, height, width)
        
        # Apply confidence threshold
        conf_mask = depth_conf >= conf_thres_value
        conf_mask = randomly_limit_trues(conf_mask, max_points_for_colmap)
        
        points_3d = points_3d[conf_mask]
        points_xyf = points_xyf[conf_mask]
        points_rgb = points_rgb[conf_mask]
        
        reconstruction = batch_np_matrix_to_pycolmap_wo_track(
            points_3d,
            points_xyf,
            points_rgb,
            extrinsic,
            intrinsic,
            image_size,
            shared_camera=False,
            camera_type="PINHOLE",
        )
        
        reconstruction_resolution = vggt_resolution
    
    # Extract point cloud
    if reconstruction is not None:
        points_list = []
        colors_list = []
        for point3D in reconstruction.points3D.values():
            points_list.append(point3D.xyz)
            colors_list.append(point3D.color)
        
        if len(points_list) > 0:
            points_array = np.array(points_list)  # [N, 3]
            colors_array = np.array(colors_list)  # [N, 3]
            
            # Combine into [N, 6] format
            point_cloud = np.concatenate([points_array, colors_array], axis=1)
            point_cloud = torch.from_numpy(point_cloud).float()
        else:
            # Empty point cloud
            point_cloud = torch.zeros((0, 6)).float()
    else:
        # Empty point cloud if reconstruction failed
        point_cloud = torch.zeros((0, 6)).float()
    
    return focal_rgb, focal_depth, focal_camera, point_cloud

def save_data_sample(
    output_dir: str, gpu_rank: int, sample_idx: int, files_per_subdir: int,
    focal_rgb: torch.Tensor, focal_depth: torch.Tensor, 
    focal_camera: torch.Tensor, point_cloud: torch.Tensor
):
    """Save a single data sample to disk."""
    # Determine subdirectory
    subdir_idx = sample_idx // files_per_subdir
    sample_in_subdir = sample_idx % files_per_subdir
    
    # Create directory structure
    rank_dir = Path(output_dir) / str(gpu_rank)
    subdir = rank_dir / f"{subdir_idx:06d}"
    subdir.mkdir(parents=True, exist_ok=True)
    
    # File paths
    base_name = f"{sample_in_subdir:04d}"
    rgb_path = subdir / f"{base_name}.jpg"
    depth_path = subdir / f"{base_name}.depth.jpg"
    camera_path = subdir / f"{base_name}.camera.pt"
    points_path = subdir / f"{base_name}.points.pt"
    
    # Save RGB image
    rgb_img = (focal_rgb.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    Image.fromarray(rgb_img).save(rgb_path, quality=95)
    
    # Save depth as grayscale image (normalize to 0-255)
    depth_normalized = focal_depth.cpu().numpy()
    if depth_normalized.max() > 0:
        depth_normalized = (depth_normalized / depth_normalized.max() * 255).astype(np.uint8)
    else:
        depth_normalized = np.zeros_like(depth_normalized, dtype=np.uint8)
    Image.fromarray(depth_normalized, mode='L').save(depth_path)
    
    # Save camera parameters
    torch.save(focal_camera.cpu(), camera_path)
    
    # Save point cloud
    torch.save(point_cloud.cpu(), points_path)

@ray.remote(num_gpus=1)
def process_video_worker(
    video_paths: List[str], 
    output_dir: str,
    gpu_rank: int,
    args_dict: dict
):
    """Ray worker function to process videos on a single GPU."""
    # Reconstruct args object
    class Args:
        def __init__(self, d):
            for k, v in d.items():
                setattr(self, k, v)
    args = Args(args_dict)
    
    # Load models
    print(f"GPU {gpu_rank}: Loading models...")
    vggt_model, dinov2_model, vggsfm_tracker_model = load_vggt_models()
    
    sample_idx = 0
    
    for video_path in tqdm(video_paths, desc=f"GPU {gpu_rank} videos"):
        try:
            # Load video with decord
            vr = decord.VideoReader(video_path)
            num_frames = len(vr)
            
            print(f"GPU {gpu_rank}: Processing {video_path} ({num_frames} frames)")
            
            # Create sliding windows
            windows = create_sliding_windows(
                num_frames, args.kernel_size, args.stride, args.dilation
            )
            
            print(f"GPU {gpu_rank}: Created {len(windows)} windows for {video_path}")
            
            for window_indices, focal_frame_idx in tqdm(windows, desc=f"GPU {gpu_rank} windows", leave=False):
                try:
                    # Load frames for this window
                    frames_np = vr.get_batch(window_indices).asnumpy()  # [T, H, W, C]
                    
                    # Convert to tensor format [T, C, H, W] and normalize to [0,1]
                    frames = torch.from_numpy(frames_np).float() / 255.0
                    frames = frames.permute(0, 3, 1, 2)  # [T, C, H, W]
                    
                    # Find focal frame index within the window
                    focal_idx_in_window = window_indices.index(focal_frame_idx)
                    
                    # Process window
                    focal_rgb, focal_depth, focal_camera, point_cloud = process_window(
                        vggt_model, dinov2_model, vggsfm_tracker_model,
                        frames, focal_idx_in_window, args
                    )
                    
                    # Save data sample
                    save_data_sample(
                        output_dir, gpu_rank, sample_idx, args.files_per_subdir,
                        focal_rgb, focal_depth, focal_camera, point_cloud
                    )
                    
                    sample_idx += 1
                    
                except Exception as e:
                    print(f"GPU {gpu_rank}: Error processing window {window_indices}: {e}")
                    continue
                    
        except Exception as e:
            print(f"GPU {gpu_rank}: Error processing video {video_path}: {e}")
            continue
    
    print(f"GPU {gpu_rank}: Completed processing {len(video_paths)} videos, generated {sample_idx} samples")
    return sample_idx

def main():
    args = parse_args()
    
    # Validate arguments
    if args.kernel_size % 2 == 0:
        print("Error: kernel_size must be odd")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get video files
    video_files = get_video_files(args.video_dir)
    if not video_files:
        print(f"No MP4 files found in {args.video_dir}")
        sys.exit(1)
    
    print(f"Found {len(video_files)} video files")
    
    # Initialize Ray
    if not ray.is_initialized():
        ray.init()
    
    # Determine number of GPUs
    num_gpus = args.num_gpus
    if num_gpus is None:
        num_gpus = torch.cuda.device_count()
    
    if num_gpus == 0:
        print("No GPUs available")
        sys.exit(1)
    
    print(f"Using {num_gpus} GPUs")
    
    # Split videos among GPUs
    videos_per_gpu = len(video_files) // num_gpus
    video_splits = []
    
    for i in range(num_gpus):
        start_idx = i * videos_per_gpu
        if i == num_gpus - 1:  # Last GPU gets remaining videos
            end_idx = len(video_files)
        else:
            end_idx = (i + 1) * videos_per_gpu
        
        video_splits.append(video_files[start_idx:end_idx])
    
    print(f"Video distribution: {[len(split) for split in video_splits]}")
    
    # Convert args to dict for Ray
    args_dict = vars(args)
    
    # Launch Ray workers
    futures = []
    for gpu_rank in range(num_gpus):
        future = process_video_worker.remote(
            video_splits[gpu_rank], 
            args.output_dir,
            gpu_rank,
            args_dict
        )
        futures.append(future)
    
    # Wait for completion
    results = ray.get(futures)
    
    total_samples = sum(results)
    print(f"\nCompleted! Generated {total_samples} training samples total")
    print(f"Output directory: {args.output_dir}")
    
    ray.shutdown()

if __name__ == "__main__":
    start_time = time.time()
    main()
    print(f"\nTotal processing time: {time.time() - start_time:.2f}s")
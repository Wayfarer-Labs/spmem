#!/usr/bin/env python3
"""
VAE Data Pipeline - Sliding Window Video Processing

Processes videos with a sliding window approach to create training data for VAE.
Each "focal" frame gets RGB, depth, camera info, and point cloud from surrounding frames.

Usage:
    python vae_data_pipeline_single.py \
        --video-dir /path/to/videos \
        --output-dir /path/to/output \
        --kernel-size 5 \
        --stride 3 \
        --dilation 2 \
        --files-per-subdir 500 \
        --num-gpus 4 \
        --rank 0
"""

import os
import sys
import argparse
import time
import math
from pathlib import Path
from typing import List, Tuple, Optional

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

import traceback
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
    p.add_argument("--batch-size", required=True, help="number of parallel windows to process")

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
    p.add_argument("--rank", type=int, default=0,
                   help="Rank of this process (which GPU index to use)")
    p.add_argument("--world-size", type=int, default=8,
                   help="Number of GPUs to use")
                   
    # VGGT/COLMAP settings
    p.add_argument("--use-ba", action="store_true", default=False, 
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
    p.add_argument("--conf-thres-value", type=float, default=1.01,
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

def run_VGGT(model, images, dtype, resolution=518):
    """
    Run VGGT model to get extrinsic, intrinsic matrices and depth maps.
    From colmap_demo.py
    """
    # images: [B, N, 3, H, W]
    assert len(images.shape) == 5
    assert images.shape[2] == 3

    b, n, c, h, w = images.shape
    # flatten B and N to get a 4-D tensor
    images = images.view(b * n, c, h, w)
    # hard-coded to use 518 for VGGT
    images = F.interpolate(images, size=(resolution, resolution), mode="bilinear", align_corners=False)
    # restore [B, N, 3, res, res]
    images = images.view(b, n, c, resolution, resolution)

    with torch.no_grad():
        with torch.amp.autocast("cuda", dtype=dtype):
            extrinsic, intrinsic, depth_map, depth_conf, points_3d = fwd_vggt(model, images)

    return extrinsic, intrinsic, depth_map, depth_conf, points_3d

@torch.compile
def fwd_vggt(model, images):
    """Forward pass through VGGT model."""
    # images = images[None]  # add batch dimension
    aggregated_tokens_list, ps_idx = model.aggregator(images)

    # Predict Cameras
    pose_enc = model.camera_head(aggregated_tokens_list)[-1]
    # Extrinsic and intrinsic matrices, following OpenCV convention (camera from world)
    extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])
    # Predict Depth Maps
    depth_map, depth_conf = model.depth_head(aggregated_tokens_list, images, ps_idx)

    extrinsic = extrinsic.squeeze(0)
    intrinsic = intrinsic.squeeze(0)
    depth_map = depth_map.squeeze(0)
    depth_conf = depth_conf.squeeze(0)

    # numpy based, reference implementation
    extrinsic, intrinsic, depth_map, depth_conf = [x.cpu().numpy() for x in [extrinsic, intrinsic, depth_map, depth_conf]]
    points_3d = unproject_depth_map_to_point_map(depth_map, extrinsic, intrinsic)

    # experimental gpu implementation
    # points_3d = torch_experimental_unproject_depth_map_to_point_map(depth_map, extrinsic, intrinsic)

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
    frames_batch: torch.Tensor, focal_idxs: List[int], args
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
    """
    Process a batch of windows of frames to extract focal-frame data and point clouds.

    Inputs:
      frames_batch: [B, T, C, H, W] tensor
      focal_idxs:   list of B focal-frame indices within each window
    Returns:
      4 lists of length B: focal_rgb, focal_depth, focal_camera, point_cloud
    """
    device = next(vggt_model.parameters()).device
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    B, T, C, H, W = frames_batch.shape
    out_rgbs, out_depths, out_cams, out_pcls = [], [], [], []

    for i, focal_idx in enumerate(focal_idxs):
        # Select the i-th window
        # frames = frames_batch[i].to(device)        # [T, C, H, W]
        frames = frames_batch.to(device)  # [T, C, H, W]
        # 1) Original image coords
        original_coords = torch.zeros((T, 4), device=device)
        original_coords[:, 2] = W
        original_coords[:, 3] = H

        # 2) VGGT inference (add batch dim)
        extrinsic, intrinsic, depth_map, depth_conf, points_3d = run_VGGT(
            vggt_model,
            frames,               # [1, T, 3, H, W]
            dtype,
            518
        )
        # extrinsic: np array [T, 3, 4], intrinsic: [T, 3, 3],
        # depth_map, depth_conf: [T, res, res], points_3d: [T, res, res, 3]

        # 3) Extract focal-frame RGB and depth
        focal_rgb   = frames[focal_idx]                                # [3, H, W]
        focal_depth = torch.from_numpy(depth_map[focal_idx]).float()   # [res, res]

        # 4) Camera parameters for focal frame
        fe = torch.from_numpy(extrinsic[focal_idx]).float()            # [3,4]
        fi = torch.from_numpy(intrinsic[focal_idx]).float()            # [3,3]
        focal_camera = torch.cat([fe.flatten(), fi.flatten()])         # [3*4 + 3*3 = 21]

        # 5) Point-cloud reconstruction
        if args.use_ba:
            load_res = min(518, max(H, W))
            images_square = square_and_resize(frames, load_res)
            image_size = np.array(images_square.shape[-2:])
            scale = max(H, W) / 518

            images_square = images_square.to(device)
            with torch.amp.autocast("cuda", dtype=dtype):
                pred_tracks, pred_vis_scores, pred_confs, ba_points, points_rgb = predict_tracks(
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

            intrinsic[:] = intrinsic * scale if isinstance(intrinsic, np.ndarray) else intrinsic.mul_(scale)
            track_mask = pred_vis_scores > args.vis_thresh
            base_paths = [f"frame_{j:03d}.png" for j in range(T)]
            reconstruction, _ = batch_np_matrix_to_pycolmap(
                ba_points,
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
            recon_res = max(H, W)
            if reconstruction is not None:
                ba_opts = pycolmap.BundleAdjustmentOptions()
                pycolmap.bundle_adjustment(reconstruction, ba_opts)
        else:
            # Simple non-BA point-cloud: sample 3D points + their RGB colors
            max_points_for_colmap = 200000

            res = depth_conf.shape[-1]
            conf_mask = depth_conf >= args.conf_thres_value
            conf_mask = randomly_limit_trues(conf_mask, max_points_for_colmap)

            # 3D points at VGGT resolution
            pts3d = points_3d[conf_mask]                   # [N, 3]
            # corresponding pixel coordinates
            pix_grid = create_pixel_coordinate_grid(T, res, res)[conf_mask]  # [N, 3]

            # Resize all frames to VGGT resolution for color lookup
            with torch.no_grad():
                inp = frames if frames.dim() == 4 else frames.unsqueeze(0)
                frames_res = F.interpolate(
                    inp,                      # [T, C, H, W] or [1, C, H, W]
                    size=(res, res),
                    mode="bilinear",
                    align_corners=False
                )                         # [T, C, res, res] or [1, C, res, res]
                # if we added a batch dim, remove it
                if frames.dim() == 3:
                    frames_res = frames_res.squeeze(0)
             # convert to HWC
            arr = frames_res.cpu().numpy().transpose(0, 2, 3, 1)  # [T, res, res, 3]

            # clamp pixel indices and gather colors
            pix = pix_grid.astype(int)
            # First column is frame index, should be in range [0, T-1]
            pix[:, 0] = np.clip(pix[:, 0], 0, T - 1)
            # Second and third columns are y,x coordinates
            pix[:, 1] = np.clip(pix[:, 1], 0, res - 1)
            pix[:, 2] = np.clip(pix[:, 2], 0, res - 1)

            # Extract colors at each pixel location and convert to uint8
            colors = np.zeros((len(pts3d), 3), dtype=np.uint8)
            for i, (f, y, x) in enumerate(pix):
                colors[i] = (arr[f, y, x] * 255).astype(np.uint8)

            reconstruction = batch_np_matrix_to_pycolmap_wo_track(
                pts3d,
                pix_grid,
                colors,  # RGB data in [N, 3] format as uint8
                extrinsic,
                intrinsic,
                np.array([res, res]),
                shared_camera=args.shared_camera,
                camera_type=args.camera_type,
            )
            recon_res = res

        # 6) Rename/rescale camera & extract point cloud
        if reconstruction is not None:
            base_paths = [f"frame_{j:03d}.png" for j in range(T)]
            reconstruction = rename_colmap_recons_and_rescale_camera(
                reconstruction,
                base_paths,
                original_coords.cpu().numpy(),
                img_size=recon_res,
                shift_point2d_to_original_res=False,
                shared_camera=args.shared_camera if args.use_ba else False,
            )

            pts_list, cols_list = [], []
            for p3d in reconstruction.points3D.values():
                pts_list.append(p3d.xyz)
                cols_list.append(p3d.color)
            if pts_list:
                points_array = np.array(pts_list)    # [N, 3]
                colors_array = np.array(cols_list)    # [N, 3]
                pc = np.concatenate([points_array, colors_array], axis=1)
                point_cloud = torch.from_numpy(pc).float()
                ply_path = os.path.join("./", "points.ply")
                print(f"Points passing confidence threshold: {conf_mask.sum()}")
                print(f"Percentage of points kept: {100 * conf_mask.sum() / conf_mask.size:.2f}%")

                print(f"Saving point cloud with {len(point_cloud)} points to {ply_path}")
                trimesh.PointCloud(points_array, colors=colors_array).export(ply_path)

            else:
                point_cloud = torch.zeros((0, 6))
        else:
            point_cloud = torch.zeros((0, 6))

        # 7) Collect outputs
        out_rgbs.append(focal_rgb)
        out_depths.append(focal_depth)
        out_cams.append(focal_camera)
        out_pcls.append(point_cloud)

    return out_rgbs, out_depths, out_cams, out_pcls

def process_window_batch(vggt_model, dinov2_model, vggsfm_tracker_model,
    frames_batch: torch.Tensor, focal_idxs: List[int], args
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
    """
    A thin wrapper around process_window that accepts:
      - frames_batch: [B, T, C, H, W]
      - focal_idxs:  list of B ints (focal index per window)
    Returns 4 lists, each of length B.
    """
    # outs = ([], [], [], [])
    # for i, fi in enumerate(focal_idxs):
    #     frgb, fdepth, fcam, pcl = process_window(
    #         vggt_model, dinov2_model, vggsfm_tracker_model,
    #         frames_batch[i], fi, args
    #     )
    #     outs[0].append(frgb)
    #     outs[1].append(fdepth)
    #     outs[2].append(fcam)
    #     outs[3].append(pcl)
    # return outs
    # feed the whole batch in one go and let process_window handle the per-window split
    return process_window(
        vggt_model, dinov2_model, vggsfm_tracker_model,
        frames_batch, focal_idxs, args
    )


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

    Image.fromarray(depth_normalized[...,0], mode='L').save(depth_path)
    
    # Save camera parameters
    torch.save(focal_camera.cpu(), camera_path)
    
    # Save point cloud
    torch.save(point_cloud.cpu(), points_path)

def process_video_worker(
    video_paths: List[str], 
    output_dir: str,
    gpu_rank: int,
    args
):
    """Process videos on a single GPU (for a given rank)."""
    # Set CUDA device for this process
    torch.cuda.set_device(gpu_rank)
    
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
            
            window_batch_size = getattr(args, "batch_size", 4)
            # split windows into batches of N
            window_batches = [
                windows[i : i + window_batch_size]
                for i in range(0, len(windows), window_batch_size)
            ]

            for batch in tqdm(window_batches, desc=f"GPU {gpu_rank} window batches", leave=False):
                frames_list = []
                focal_idxs  = []
                try:

                    # load all frames for this batch
                    for window_indices, focal_frame_idx in batch:
                        frames_np = vr.get_batch(window_indices).asnumpy()    # [T, H, W, C]
                        frames = torch.from_numpy(frames_np).float().div_(255.0)  # [T, H, W, C]
                        frames = frames.permute(0, 3, 1, 2)                       # [T, C, H, W]
                        frames_list.append(frames)
                        focal_idxs.append(window_indices.index(focal_frame_idx))
                    # stack into [B, T, C, H, W]
                    frames_batch = torch.stack(frames_list, dim=0)

                    print(f"shapes: {frames_batch.shape}, focal_idxs: {focal_idxs}")
                    # process the whole batch
                    frgbs, fdepths, fcams, pcls = process_window_batch(
                        vggt_model, dinov2_model, vggsfm_tracker_model,
                        frames_batch, focal_idxs, args
                    )

                    # save each result
                    for frgb, fdepth, fcam, pcl in zip(frgbs, fdepths, fcams, pcls):
                        save_data_sample(
                            output_dir, gpu_rank, sample_idx, args.files_per_subdir,
                            frgb, fdepth, fcam, pcl
                        )
                        sample_idx += 1

                except Exception as e:
                    error_trace = traceback.format_exc()
                    print(f"GPU {gpu_rank}: Error processing window {window_indices}: {e}")
                    print(f"Traceback:\n{error_trace}")
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
    
    # Determine number of GPUs (per node)
    num_gpus = args.num_gpus
    if num_gpus is None:
        num_gpus = torch.cuda.device_count()
    
    if num_gpus == 0:
        print("No GPUs available")
        sys.exit(1)
    
    # Use world_size for splitting work across all processes
    world_size = args.world_size
    if world_size is None:
        world_size = num_gpus  # fallback
    
    if args.rank < 0 or args.rank >= world_size:
        print(f"Invalid rank {args.rank}. Must be in [0, {world_size-1}]")
        sys.exit(1)
    
    print(f"Using {num_gpus} GPUs per node. World size: {world_size}. This process is rank {args.rank}.")
    
    # Split videos among world_size (all processes)
    video_splits = []
    videos_per_rank = len(video_files) // world_size
    remainder = len(video_files) % world_size
    start_idx = 0
    for i in range(world_size):
        extra = 1 if i < remainder else 0
        end_idx = start_idx + videos_per_rank + extra
        video_splits.append(video_files[start_idx:end_idx])
        start_idx = end_idx

    print(f"Video distribution: {[len(split) for split in video_splits]}")
    
    # Get the video list for this rank
    my_videos = video_splits[args.rank]
    
    # Process videos for this rank
    total_samples = process_video_worker(
        my_videos,
        args.output_dir,
        args.rank,
        args
    )
    
    print(f"\nCompleted! Generated {total_samples} training samples for rank {args.rank}")
    print(f"Output directory: {args.output_dir}")

if __name__ == "__main__":
    start_time = time.time()
    main()
    print(f"\nTotal processing time: {time.time() - start_time:.2f}s")

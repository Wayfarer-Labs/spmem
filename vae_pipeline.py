import os
import argparse
import sys

from typing import List, Tuple

# from streaming_vggt import load_vggt_model, get_dinov2_model, run_VGGT, list_video_keys
from streaming_vggt import (
    load_vggt_model,
    get_dinov2_model,
    run_VGGT,
    square_and_resize,
    unproject_depth_map_to_point_map,
    create_pixel_coordinate_grid,
    randomly_limit_trues,
    predict_tracks,
    batch_np_matrix_to_pycolmap,
    batch_np_matrix_to_pycolmap_wo_track,
    rename_colmap_recons_and_rescale_camera,
    process_batch,
)

from vggt.dependency.vggsfm_utils import build_vggsfm_tracker
from tqdm import tqdm

import boto3
from botocore.exceptions import NoCredentialsError, PartialCredentialsError
import torch

def get_presigned_url(s3, bucket, key, expires=3600):
    return s3.generate_presigned_url(
        "get_object",
        Params={"Bucket": bucket, "Key": key},
        ExpiresIn=expires,
    )

def parse_args():
    p = argparse.ArgumentParser(
        description="Stream-download & batch-process videos from S3"
    )
    
    p.add_argument("--bucket",             required=True, help="S3 bucket name")
    p.add_argument("--prefix",             default="",     help="S3 prefix/folder for videos")
    p.add_argument("--ext",                default="mp4",  help="Video file extension filter")
    p.add_argument("--frame-batch-size",  type=int, default=50, help="Number of frames per batch to process")
    p.add_argument("--target-bucket", required=True, help="S3 bucket for uploads")
    
    # Input/Output
    p.add_argument("--video-dir", required=True, help="Directory containing MP4 videos")
    p.add_argument("--output-dir", required=True, help="Output directory for processed data")
    
    # Sliding window parameters
    p.add_argument("--window-size", type=int, default=5, 
                   help="sliding window size around focal frame (must be odd)")
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
    
    # COLMAP reconstruction options
    p.add_argument("--use-ba", action="store_true", default=False, help="Use BA for reconstruction")
    p.add_argument("--max-reproj-error", type=float, default=8.0, help="Maximum reprojection error for reconstruction")
    p.add_argument("--shared-camera", action="store_true", default=True, help="Use shared camera for all images")
    p.add_argument("--camera-type", type=str, default="SIMPLE_PINHOLE", help="Camera type for reconstruction")
    p.add_argument("--vis-thresh", type=float, default=0.2, help="Visibility threshold for tracks")
    p.add_argument("--query-frame-num", type=int, default=8, help="Number of frames to query")
    p.add_argument("--max-query-pts", type=int, default=4096, help="Maximum number of query points")
    p.add_argument("--fine-tracking", action="store_true", default=True, help="Use fine tracking (slower but more accurate)")
    p.add_argument("--conf-thres-value", type=float, default=5.0, help="Confidence threshold value for depth filtering (wo BA)")
    
    return p.parse_args()

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

def main():
    args = parse_args()
    
    # AWS/S3 setup
    aws_cfg = {
        "aws_access_key_id":     os.getenv("AWS_ACCESS_KEY_ID"),
        "aws_secret_access_key": os.getenv("AWS_SECRET_ACCESS_KEY"),
        "region_name":           os.getenv("AWS_REGION", "us-east-1"),
    }
    if os.getenv("AWS_ENDPOINT_URL"):
        aws_cfg["endpoint_url"] = os.getenv("AWS_ENDPOINT_URL")

    try:
        s3 = boto3.client("s3", **aws_cfg)
        s3.head_bucket(Bucket=args.bucket)
    except (NoCredentialsError, PartialCredentialsError):
        print("Error: AWS credentials missing or invalid.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error accessing bucket: {e}", file=sys.stderr)
        sys.exit(1)

    # Validate arguments
    if args.window_size % 2 == 0:
        print("Error: window_size must be odd")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get video files
    keys = list_video_keys(s3, args.bucket, args.prefix, args.ext)

    if not keys:
        print(f"No MP4 files found in {args.video_dir}")
        sys.exit(1)

    print(f"Found {len(keys)} video files")
    
    # Determine number of GPUs
    num_gpus = args.num_gpus
    if num_gpus is None:
        num_gpus = torch.cuda.device_count()
    
    if num_gpus == 0:
        print("No GPUs available")
        sys.exit(1)
    
    print(f"Using {num_gpus} GPUs")
    
    # Split videos among GPUs
    videos_per_gpu = len(keys) // num_gpus
    video_splits = []
    
    for i in range(num_gpus):
        start_idx = i * videos_per_gpu
        if i == num_gpus - 1:  # Last GPU gets remaining videos
            end_idx = len(keys)
        else:
            end_idx = (i + 1) * videos_per_gpu

        video_splits.append(keys[start_idx:end_idx])

    print(f"Video distribution: {[len(split) for split in video_splits]}")

    model = load_vggt_model()
    dinov2_model = get_dinov2_model()
    vggsfm_tracker_model = torch.compile(build_vggsfm_tracker()).cuda()

    if model is None:
        print("Failed to load VGGT model. Exiting.")
        return

    # Print configuration
    print("COLMAP Reconstruction Configuration:")
    print(f"  Use Bundle Adjustment: {args.use_ba}")
    print(f"  Max Reprojection Error: {args.max_reproj_error}")
    print(f"  Shared Camera: {args.shared_camera}")
    print(f"  Camera Type: {args.camera_type}")
    print(f"  Visibility Threshold: {args.vis_thresh}")
    print(f"  Query Frame Number: {args.query_frame_num}")
    print(f"  Max Query Points: {args.max_query_pts}")
    print(f"  Fine Tracking: {args.fine_tracking}")
    print(f"  Confidence Threshold: {args.conf_thres_value}")

    # Stream & batch-process each
    for video_path in tqdm(keys, desc="Videos", unit="video"):
        url = get_presigned_url(s3, args.bucket, video_path)

        print(f"GPU {gpu_rank}: Processing {video_path} ({num_frames} frames)")
        
        # Create sliding windows
        windows = create_sliding_windows(
            num_frames, args.kernel_size, args.stride, args.dilation
        )
        
        print(f"GPU {gpu_rank}: Created {len(windows)} windows for {video_path}")

        process_streaming_video(model, dinov2_model, vggsfm_tracker_model, url, batch_size=args.frame_batch_size,
                                s3_client=s3, target_bucket=args.target_bucket, args=args)

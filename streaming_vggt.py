#!/usr/bin/env python3
"""
stream_process_s3_videos_batches.py

Stream-download raw videos from S3 and process them in batches of frames.

Usage:
    python stream_process_s3_videos_batches.py \
        --bucket cod-yt-playlist-spmem-tensors \
        --prefix raw_videos/ \
        --ext mp4 \
        --frame-batch-size 32

Required Environment Variables:
    AWS_ACCESS_KEY_ID       - Your AWS access key ID
    AWS_SECRET_ACCE        if len(batch) >= batch_size:
            process_batch(model, batch, batch_idx, video_name, s3_client, target_bucket, start_idx)
            batch = []
            batch_idx += 1

    # final partial batch
    if batch:
        process_batch(model, batch, batch_idx, video_name, s3_client, target_bucket, start_idx)

    process.wait()

    # Generate static colored point cloud from all processed batches
    # This would require accumulating all points from all batches
    # For now, we'll skip this and just process per-batch
    print(f"   done processing {video_name} in {batch_idx + 1} batches")AWS secret access key
    AWS_REGION             - AWS region (optional, defaults to us-east-1)
    AWS_ENDPOINT_URL       - Custom endpoint URL (optional)
"""

import os
import sys
import argparse
import time

import boto3
from botocore.exceptions import NoCredentialsError, PartialCredentialsError
from tqdm import tqdm

import ffmpeg
import numpy as np
import torch
import io
from PIL import Image
from botocore.exceptions import ClientError
import math
import subprocess
import json
from torchvision import transforms as TF
import torch.nn.functional as F
import copy
import trimesh
import pycolmap

# Import VGGT model and preprocessing functions
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images_square
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map
from vggt.utils.helper import create_pixel_coordinate_grid, randomly_limit_trues
from vggt.dependency.track_predict import predict_tracks
from vggt.dependency.vggsfm_utils import build_vggsfm_tracker
from vggt.dependency.np_to_pycolmap import batch_np_matrix_to_pycolmap, batch_np_matrix_to_pycolmap_wo_track


predict_tracks = torch.compile(predict_tracks)


# Create output directory from environment or default to 'outputs'
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_vggt_model():
    """Load the VGGT model using the same approach as colmap_demo.py."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = VGGT()
    _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
    model.load_state_dict(torch.hub.load_state_dict_from_url(_URL))
    model.eval()
    model = model.to(device)
    return model

# helper to upload per-frame data to S3
def upload_frame_data(s3_client, bucket, video_name, idx, frame_np, depth, pose, trajectory):
    idx_str = f"{idx:06d}"
    prefix = f"{video_name}/"
    keys = {
        'frame': prefix + f"frame_{idx_str}.png",
        'depth': prefix + f"depth_{idx_str}.pt",
        'pose': prefix + f"pose_3x3_{idx_str}.pt",
        'trajectory': prefix + f"trajectory_{idx_str}.pt"
    }
    # prepare tensors
    for kind, key in keys.items():
        # skip existing
        try:
            s3_client.head_object(Bucket=bucket, Key=key)
            print(f"Skipping existing {key}")
            continue
        except ClientError as e:
            if e.response['Error']['Code'] != '404': raise
        # upload content
        buf = io.BytesIO()
        if kind == 'frame':
            Image.fromarray(frame_np).save(buf, format='PNG')
            body = buf.getvalue()
            s3_client.put_object(Bucket=bucket, Key=key, Body=body, ContentType='image/png')
        else:
            tensor = {'depth': depth, 'pose': pose, 'trajectory': trajectory}[kind] if kind in ['depth','pose','trajectory'] else None
            torch.save(tensor, buf)
            s3_client.put_object(Bucket=bucket, Key=key, Body=buf.getvalue())
        # print(f"Uploaded {key}")


def parse_args():
    p = argparse.ArgumentParser(
        description="Stream-download & batch-process videos from S3"
    )
    p.add_argument("--bucket",             required=True, help="S3 bucket name")
    p.add_argument("--prefix",             default="",     help="S3 prefix/folder for videos")
    p.add_argument("--ext",                default="mp4",  help="Video file extension filter")
    p.add_argument("--frame-batch-size",  type=int, default=50,
                   help="Number of frames per batch to process")
    p.add_argument("--target-bucket", required=True, help="S3 bucket for uploads")
    
    # COLMAP reconstruction options
    p.add_argument("--use-ba", action="store_true", default=True, help="Use BA for reconstruction")
    p.add_argument("--max-reproj-error", type=float, default=8.0, help="Maximum reprojection error for reconstruction")
    p.add_argument("--shared-camera", action="store_true", default=True, help="Use shared camera for all images")
    p.add_argument("--camera-type", type=str, default="SIMPLE_PINHOLE", help="Camera type for reconstruction")
    p.add_argument("--vis-thresh", type=float, default=0.2, help="Visibility threshold for tracks")
    p.add_argument("--query-frame-num", type=int, default=8, help="Number of frames to query")
    p.add_argument("--max-query-pts", type=int, default=4096, help="Maximum number of query points")
    p.add_argument("--fine-tracking", action="store_true", default=True, help="Use fine tracking (slower but more accurate)")
    p.add_argument("--conf-thres-value", type=float, default=5.0, help="Confidence threshold value for depth filtering (wo BA)")
    
    return p.parse_args()

def list_video_keys(s3, bucket, prefix, ext):
    paginator = s3.get_paginator("list_objects_v2")
    keys = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.lower().endswith(f".{ext.lower()}"):
                keys.append(key)
    return keys

def get_presigned_url(s3, bucket, key, expires=3600):
    return s3.generate_presigned_url(
        "get_object",
        Params={"Bucket": bucket, "Key": key},
        ExpiresIn=expires,
    )

def safe_probe(url):
    """
    Run ffprobe with reconnect flags via subprocess and retry on failure.
    """
    base_cmd = [
        "ffprobe",
        "-v", "error",
        "-print_format", "json",
        "-show_streams",
        "-reconnect", "1",
        "-reconnect_streamed", "1",
        "-reconnect_delay_max", "2",
        url
    ]
    try:
                
        proc = subprocess.run(base_cmd, capture_output=True)
        if proc.returncode == 0:
    
            return json.loads(proc.stdout)
    except subprocess.CalledProcessError as e:
        print(f"ffprobe failed: {e.stderr.decode('utf8', errors='ignore')}")
        return None

def get_fps(url):
    # Run ffprobe and get JSON metadata
    reconnect_args = ['-reconnect', '1',
                      '-reconnect_streamed', '1',
                      '-reconnect_delay_max', '2']
    try:
        probe = safe_probe(url)
    except ffmpeg.Error as e:
        raise RuntimeError(f"ffprobe failed: {e.stderr.decode('utf8', errors='ignore')}")

    # Find the first video stream
    video_stream = next((s for s in probe['streams'] if s['codec_type'] == 'video'), None)
    if video_stream is None:
        raise RuntimeError('No video stream found')

    # r_frame_rate is a string like "30000/1001" or "25/1"
    num, den = video_stream['r_frame_rate'].split('/')
    fps = float(num) / float(den)
    return fps


def get_dinov2_model(model_name="dinov2_vitb14_reg", device="cuda"):
    dino_v2_model = torch.hub.load("facebookresearch/dinov2", model_name)
    dino_v2_model.eval()
    dino_v2_model = dino_v2_model.to(device)
    return dino_v2_model


@torch.no_grad()
def process_batch(model, dinov2_model, vggsfm_tracker_model, frames, batch_index, video_name, s3_client, target_bucket, args):
    """
    Process a batch of frames using VGGT model for COLMAP reconstruction.
    `frames` is a list/array of shape [batch_size, H, W, C]
    """
    print(f"    → processing batch {batch_index} ({len(frames)} frames) of {video_name}")
    
    device = next(model.parameters()).device
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    # dtype = torch.float16  # Use float32 for better precision in reconstruction
    
    images = frames.to(device)  # Move frames to the same device as the model
    # Load images and get original coordinates (similar to colmap_demo.py)
    # For streaming, we'll use the current frame dimensions
    original_height, original_width = frames[0].shape[:2]
    
    # Create original_coords tensor to match colmap_demo format
    # This represents the crop/padding info, but for streaming we use the full frame
    original_coords = torch.zeros((len(frames), 4), device=device)
    original_coords[:, 2] = original_width   # width
    original_coords[:, 3] = original_height  # height
    
    # VGGT fixed resolution and image load resolution
    vggt_fixed_resolution = 518
    img_load_resolution = max(original_width, original_height)
    
    try:
        # Run VGGT to estimate camera and depth
        extrinsic, intrinsic, depth_map, depth_conf, points_3d = run_VGGT(model, images, dtype, vggt_fixed_resolution)

        print(f"Generated depth maps for batch {batch_index}")
        print(f"Depth map shape: {depth_map.shape}")
        print(f"Points 3D shape: {points_3d.shape}")
        
        # Create base image path list for this batch
        base_image_path_list = [f"frame_{batch_index:03d}_{i:03d}.png" for i in range(len(frames))]
        
        if args.use_ba:
            # Use Bundle Adjustment approach
            # Prepare square images for predict_tracks using colmap preprocessing
            load_res = min(518, max(original_width, original_height))
            images_square = square_and_resize(images, load_res)
                
            image_size = np.array(images_square.shape[-2:])
            print(f"Image size for BA: {image_size}")
            scale = img_load_resolution / vggt_fixed_resolution
            shared_camera = args.shared_camera

            images_square = images_square.cuda()
            with torch.amp.autocast("cuda", dtype=dtype):
                # Predicting Tracks
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

            # rescale the intrinsic matrix from 518 to img_load_resolution
            intrinsic[:, :2, :] *= scale
            track_mask = pred_vis_scores > args.vis_thresh

            reconstruction, valid_track_mask = batch_np_matrix_to_pycolmap(
                points_3d,
                extrinsic,
                intrinsic,
                pred_tracks,
                image_size,
                masks=track_mask,
                max_reproj_error=args.max_reproj_error,
                shared_camera=shared_camera,
                camera_type=args.camera_type,
                points_rgb=points_rgb,
            )

            if reconstruction is None:
                print(f"Warning: No reconstruction could be built with BA for batch {batch_index}")
                return

            # Bundle Adjustment
            ba_options = pycolmap.BundleAdjustmentOptions()
            pycolmap.bundle_adjustment(reconstruction, ba_options)

            reconstruction_resolution = img_load_resolution

        else:
            # Use feedforward approach without BA
            conf_thres_value = args.conf_thres_value
            max_points_for_colmap = 200000
            shared_camera = False
            camera_type = "PINHOLE"

            image_size = np.array([vggt_fixed_resolution, vggt_fixed_resolution])
            num_frames, height, width, _ = points_3d.shape

            points_rgb = F.interpolate(
                images, size=(vggt_fixed_resolution, vggt_fixed_resolution), mode="bilinear", align_corners=False
            )
            points_rgb = (points_rgb.cpu().numpy() * 255).astype(np.uint8)
            points_rgb = points_rgb.transpose(0, 2, 3, 1)

            # (S, H, W, 3), with x, y coordinates and frame indices
            points_xyf = create_pixel_coordinate_grid(num_frames, height, width)

            conf_mask = depth_conf >= conf_thres_value
            # at most writing max_points_for_colmap 3d points to colmap reconstruction object
            conf_mask = randomly_limit_trues(conf_mask, max_points_for_colmap)

            print(f"Points passing confidence threshold: {conf_mask.sum()}")
            print(f"Percentage of points kept: {100 * conf_mask.sum() / conf_mask.size:.2f}%")

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
                shared_camera=shared_camera,
                camera_type=camera_type,
            )

            reconstruction_resolution = vggt_fixed_resolution

        # Rename and rescale camera parameters
        reconstruction = rename_colmap_recons_and_rescale_camera(
            reconstruction,
            base_image_path_list,
            original_coords.cpu().numpy(),
            img_size=reconstruction_resolution,
            shift_point2d_to_original_res=False,
            shared_camera=shared_camera if args.use_ba else False,
        )

        # Save reconstruction files for this batch
        sparse_dir = os.path.join(OUTPUT_DIR, f"{video_name}_batch_{batch_index}_sparse")
        os.makedirs(sparse_dir, exist_ok=True)
        reconstruction.write(sparse_dir)
        
        # Extract points for PLY export
        points_list = []
        colors_list = []
        for point3D in reconstruction.points3D.values():
            points_list.append(point3D.xyz)
            colors_list.append(point3D.color)
        
        if len(points_list) > 0:
            points_array = np.array(points_list)
            colors_array = np.array(colors_list)
            
            # Save point cloud
            ply_path = os.path.join(sparse_dir, "points.ply")
            trimesh.PointCloud(points_array, colors=colors_array).export(ply_path)
            
            print(f"Saved COLMAP reconstruction for batch {batch_index} to {sparse_dir}")
            print(f"Generated {len(points_list)} 3D points")
        else:
            print(f"Warning: No valid 3D points generated for batch {batch_index}")
            
    except Exception as e:
        print(f"Error processing batch {batch_index}: {e}")
        import traceback
        traceback.print_exc()

def process_streaming_video(model, dinov2_model, vggsfm_tracker_model, url, batch_size, s3_client, target_bucket, args):
    """
    Stream from `url` via ffmpeg, accumulate `batch_size` frames, then process each batch.
    """
    # Derive a nice name for logging
    video_name = url.split("/")[-1].split("?")[0]
    print(f"\n→ streaming {video_name}")

    # 1) Probe the stream to get its width/height
    # probe size with reconnect & retry
    probe = safe_probe(url)
    video_stream = next(s for s in probe['streams'] if s['codec_type'] == 'video')
    width, height = int(video_stream['width']), int(video_stream['height'])
    # determine starting frame index from existing uploads
    prefix = f"{video_name}/frame_"
    resp = s3_client.list_objects_v2(Bucket=target_bucket, Prefix=prefix)
    existing = resp.get('Contents', [])
    if existing:
        idxs = [int(obj['Key'].split('_')[-1].split('.')[0]) for obj in existing]
        start_idx = max(idxs) + 1
        print(f"Resuming from frame index {start_idx}")
    else:
        start_idx = 0
    start_idx = 0
    # (fps-based batch counting removed)
    # compute batch count if nb_frames metadata is available
    if 'nb_frames' in video_stream and video_stream['nb_frames'].isdigit():
        total_frames = int(video_stream['nb_frames'])
        remaining = max(0, total_frames - start_idx)
        total_batches = math.ceil(remaining / batch_size)
        print(f"Remaining frames: {remaining}, batches: {total_batches} (batch size={batch_size})")
    else:
        print("Batch count unavailable (nb_frames missing)")

    fps = get_fps(url)
    # start_seconds = start_idx / fps
    start_seconds = 50

    # 2) Launch ffmpeg as a subprocess, outputting rawvideo RGB24 to stdout

    # launch ffmpeg process, skipping to start_idx by frame number if resuming
    # reconnect flags
    recon_args = ['-reconnect', '1',
                  '-reconnect_streamed', '1',
                  '-reconnect_delay_max', '2']

    process = (
        ffmpeg
        .input(url, ss=start_seconds)
        .filter('scale', -1, 1080)  # ensure correct resolution
        .output('pipe:', format='rawvideo', pix_fmt='rgb24')
        .global_args(*recon_args)
        .run_async(pipe_stdout=True, pipe_stderr=True)
    )

    frame_size = width * height * 3  # bytes per frame
    # initialize batch and batch counter
    batch = []
    batch_idx = 0
    frame_count = 5
    idx = 0

    # read frames
    while True:
        # read exactly one frame
        in_bytes = process.stdout.read(frame_size)
        if not in_bytes or len(in_bytes) < frame_size:
            break

        buffer_copy = in_bytes[:]
        # turn bytes into H×W×3 uint8 numpy array
        # Convert frames to proper format for VGGT
        # frames are numpy arrays of shape (H, W, 3) with values 0-255
        # Convert to tensors and normalize to [0, 1]
        frame = (
            torch
            .frombuffer(buffer_copy, dtype=torch.uint8)
            .reshape((height, width, 3))
            .permute(2, 0, 1) # Convert to (3, H, W)
            .float() / 255.0
        )

        if idx < frame_count:
            idx += 1
            print(".", end="", flush=True)
            continue

        batch.append(frame)
        if len(batch) >= batch_size:
            batch_tensor = torch.stack(batch) # Shape: (N, 3, H, W)
            process_batch(model, dinov2_model, vggsfm_tracker_model, batch_tensor, batch_idx, video_name, s3_client, target_bucket, args)
            batch = []
            batch_idx += 1
            idx = 0
            

    # final partial batch
    if batch:
        process_batch(model, dinov2_model, batch, batch_idx, video_name, s3_client, target_bucket, args)

    process.wait()

    # Generate static colored point cloud from all processed batches
    # This would require accumulating all points from all batches
    # For now, we'll skip this and just process per-batch
    print(f"   done processing {video_name} in {batch_idx + 1} batches")

    if process.returncode != 0:
        err = process.stderr.read().decode('utf8', errors='ignore')
        print(f"ffmpeg exited {process.returncode}:\n{err}")
    else:
        print(f"   done {video_name}")


def run_VGGT(model, images, dtype, resolution=518):
    """
    Run VGGT model to get extrinsic, intrinsic matrices and depth maps.
    From colmap_demo.py
    """
    # images: [B, 3, H, W]
    assert len(images.shape) == 4
    assert images.shape[1] == 3

    # hard-coded to use 518 for VGGT
    images = F.interpolate(images, size=(resolution, resolution), mode="bilinear", align_corners=False)

    with torch.no_grad():
        with torch.amp.autocast("cuda", dtype=dtype):
            extrinsic, intrinsic, depth_map, depth_conf, points_3d = fwd(model, images)

    return extrinsic, intrinsic, depth_map, depth_conf, points_3d


def torch_experimental_unproject_depth_map_to_point_map(
    depth_map: torch.Tensor,          # (S,H,W[,1])
    extrinsics_cam: torch.Tensor,     # (S,3,4) : camera ← world   [R | t]
    intrinsics_cam: torch.Tensor,     # (S,3,3)
) -> torch.Tensor:                    # ⟼ (S,H,W,3)   world coords
    assert torch.is_tensor(depth_map)  and depth_map.is_cuda
    assert torch.is_tensor(extrinsics_cam) and extrinsics_cam.is_cuda
    assert torch.is_tensor(intrinsics_cam) and intrinsics_cam.is_cuda
    assert extrinsics_cam.shape[-2:] == (3, 4)
    assert intrinsics_cam.shape[-2:] == (3, 3)

    # ------------ reshape / constants --------------------------------------
    if depth_map.ndim == 4 and depth_map.shape[-1] == 1:          # (S,H,W,1) → (S,H,W)
        depth_map = depth_map.squeeze(-1)

    S, H, W = depth_map.shape
    device, dtype = depth_map.device, depth_map.dtype

    # ------------ pixel grid (homogeneous) ---------------------------------
    ys, xs = torch.meshgrid(
        torch.arange(H, device=device, dtype=dtype),
        torch.arange(W, device=device, dtype=dtype),
        indexing="ij",
    )                                        # each (H,W)
    pix_h = torch.stack((xs, ys, torch.ones_like(xs)))   # (3,H,W)
    pix_h = pix_h.unsqueeze(0).expand(S, -1, -1, -1)     # (S,3,H,W)

    # ------------ camera ray directions ------------------------------------
    K_inv   = torch.linalg.inv(intrinsics_cam)           # (S,3,3)
    rays    = torch.einsum("bij,bjhw->bihw", K_inv, pix_h)   # (S,3,H,W)

    pts_cam = rays * depth_map.unsqueeze(1)              # (S,3,H,W)

    # ------------ camera → world -------------------------------------------
    R = extrinsics_cam[..., :3]                          # (S,3,3)
    t = extrinsics_cam[..., 3:].unsqueeze(-1)            # (S,3,1,1)

    pts_world = torch.einsum("bij,bjhw->bihw", R.transpose(1, 2), pts_cam - t)

    return pts_world.permute(0, 2, 3, 1)                 # (S,H,W,3)


@torch.compile
def fwd(model, images):
    images = images[None]  # add batch dimension
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


def rename_colmap_recons_and_rescale_camera(
    reconstruction, image_paths, original_coords, img_size, shift_point2d_to_original_res=False, shared_camera=False
):
    """
    Rename images and rescale camera parameters to match original image dimensions.
    From colmap_demo.py
    """
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

def square_and_resize(images: torch.Tensor, load_res: int) -> torch.Tensor:
    """
    Resize and pad images to square of size (load_res, load_res).
    Images are tensors of shape (N, 3, H, W) with values in [0,1].
    """
    square_images = []
    for img in images:
        c, h, w = img.shape
        # scale to keep aspect ratio
        if h >= w:
            new_h = load_res
            new_w = max(1, int(w * load_res / h))
        else:
            new_w = load_res
            new_h = max(1, int(h * load_res / w))
        # make dimensions divisible by 14
        new_h = (new_h // 14) * 14
        new_w = (new_w // 14) * 14
        img_resized = F.interpolate(
            img.unsqueeze(0), size=(new_h, new_w), mode="bilinear", align_corners=False
        ).squeeze(0)
        # pad to square
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

    # List keys
    print(f"Listing .{args.ext} files in s3://{args.bucket}/{args.prefix}")
    keys = list_video_keys(s3, args.bucket, args.prefix, args.ext)
    if not keys:
        print("No videos found. Exiting.")
        return

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
    for key in tqdm(keys[2:], desc="Videos", unit="video"):
        url = get_presigned_url(s3, args.bucket, key)
        process_streaming_video(model, dinov2_model, vggsfm_tracker_model, url, batch_size=args.frame_batch_size,
                                s3_client=s3, target_bucket=args.target_bucket, args=args)

if __name__ == "__main__":
    start = time.time()
    main()
    print(f"\nAll done! Total time: {time.time() - start:.2f}s")

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

# Import VGGT model and preprocessing functions
from vggt.models.vggt import VGGT

# Create output directory from environment or default to 'outputs'
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def preprocess_frame_for_vggt(frame_np, target_size=518):
    """
    Preprocess a single numpy frame for VGGT model input.
    Based on load_and_preprocess_images from VGGT codebase.
    
    Args:
        frame_np: numpy array of shape (H, W, 3) with values 0-255
        target_size: target size for the longest dimension
        
    Returns:
        torch.Tensor: preprocessed frame tensor of shape (3, H, W) normalized to [0,1]
    """
    # Convert numpy to PIL Image
    img = Image.fromarray(frame_np.astype(np.uint8))
    
    # Convert to RGB if needed
    img = img.convert("RGB")
    
    width, height = img.size
    to_tensor = TF.ToTensor()
    
    # Make the largest dimension target_size while maintaining aspect ratio
    if width >= height:
        new_width = target_size
        new_height = round(height * (new_width / width) / 14) * 14  # Make divisible by 14
    else:
        new_height = target_size
        new_width = round(width * (new_height / height) / 14) * 14  # Make divisible by 14
    
    # Resize with new dimensions (width, height)
    img = img.resize((new_width, new_height), Image.Resampling.BICUBIC)
    img_tensor = to_tensor(img)  # Convert to tensor (0, 1)
    
    # Pad to make a square of target_size x target_size
    h_padding = target_size - img_tensor.shape[1]
    w_padding = target_size - img_tensor.shape[2]
    
    if h_padding > 0 or w_padding > 0:
        pad_top = h_padding // 2
        pad_bottom = h_padding - pad_top
        pad_left = w_padding // 2
        pad_right = w_padding - pad_left
        
        # Pad with white (value=1.0)
        img_tensor = torch.nn.functional.pad(
            img_tensor, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=1.0
        )
    
    return img_tensor

def load_vggt_model():
    """Load the VGGT model."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)
    # model.eval()
    return model

def extract_colors_from_vggt_predictions(world_points, images, world_points_conf, conf_threshold=0.0001):
    """
    Extract RGB colors from input images corresponding to world points.
    Based on the implementation in main.py.
    
    Args:
        world_points: Tensor of shape (batch, frames, height, width, 3) - 3D world coordinates
        images: Tensor of shape (batch, frames, 3, height, width) - input images  
        world_points_conf: Tensor of shape (batch, frames, height, width) - confidence scores
        conf_threshold: Float - minimum confidence threshold for valid points
        
    Returns:
        valid_points: numpy array of valid 3D points
        valid_colors: numpy array of corresponding RGB colors (0-255)
        valid_conf: numpy array of corresponding confidence scores
    """
    # Convert to numpy and move to CPU
    world_points_np = world_points.cpu().numpy()[0]  # Remove batch dimension: (frames, height, width, 3)
    world_points_conf_np = world_points_conf.cpu().numpy()[0]  # (frames, height, width)
    
    # Convert images from (batch, frames, 3, H, W) to (frames, H, W, 3)
    images_np = images.cpu().numpy()[0].transpose(0, 2, 3, 1)  # (frames, height, width, 3)
    image_colors = (images_np * 255).astype(np.uint8)  # Convert to 0-255 range
    
    # Flatten all arrays
    world_points_flat = world_points_np.reshape(-1, 3)
    colors_flat = image_colors.reshape(-1, 3)
    conf_flat = world_points_conf_np.reshape(-1) - 1
    
    # Apply confidence threshold
    valid_mask = conf_flat > conf_threshold
    
    valid_points = world_points_flat[valid_mask]
    valid_colors = colors_flat[valid_mask]
    valid_conf = conf_flat[valid_mask]
    
    return valid_points, valid_colors, valid_conf

def export_ply_with_colors(points, colors, filename=None, confidence=None):
    """Export points with colors to a PLY file or return as bytes."""
    ply_lines = []
    ply_lines.append("ply\n")
    ply_lines.append("format ascii 1.0\n")
    ply_lines.append(f"element vertex {len(points)}\n")
    ply_lines.append("property float x\n")
    ply_lines.append("property float y\n")
    ply_lines.append("property float z\n")
    ply_lines.append("property uchar red\n")
    ply_lines.append("property uchar green\n")
    ply_lines.append("property uchar blue\n")
    if confidence is not None:
        ply_lines.append("property float confidence\n")
    ply_lines.append("end_header\n")
    
    if confidence is not None:
        for point, color, conf in zip(points, colors, confidence):
            ply_lines.append(f"{point[0]} {point[1]} {point[2]} {color[0]} {color[1]} {color[2]} {conf}\n")
    else:
        for point, color in zip(points, colors):
            ply_lines.append(f"{point[0]} {point[1]} {point[2]} {color[0]} {color[1]} {color[2]}\n")
    
    ply_content = ''.join(ply_lines)
    
    if filename:
        with open(filename, 'w') as f:
            f.write(ply_content)
        return None
    else:
        return ply_content.encode('utf-8')

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

def process_batch(model, frames, batch_index, video_name, s3_client, target_bucket, start_idx=0, pipeline=None):
    """
    Process a batch of frames using VGGT model.
    `frames` is a list/array of shape [batch_size, H, W, C]
    """
    print(f"    → processing batch {batch_index} ({len(frames)} frames) of {video_name}")
    
    device = next(model.parameters()).device
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    
    # Preprocess frames for VGGT
    preprocessed_frames = []
    for frame in frames:
        preprocessed_frame = preprocess_frame_for_vggt(frame)
        preprocessed_frames.append(preprocessed_frame)
    
    # Stack into batch tensor
    images = torch.stack(preprocessed_frames).to(device)  # Shape: (N, 3, H, W)
    images = images.unsqueeze(0)  # Add batch dimension: (1, N, 3, H, W)
    
    # Run VGGT inference
    ply_bytes = None
    try:
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=dtype):
                predictions = model(images)
                print("Prediction keys:", list(predictions.keys()))
                # Extract world points and colors
                world_points = predictions["world_points"]
                world_points_conf = predictions["world_points_conf"] 
                
                # Extract colored point cloud
                valid_points, valid_colors, valid_conf = extract_colors_from_vggt_predictions(
                    world_points, images, world_points_conf, conf_threshold=0.0001
                )
                
                if len(valid_points) > 0:
                    ply_bytes = export_ply_with_colors(valid_points, valid_colors, confidence=valid_conf)
                    print(f"Generated {len(valid_points)} colored points for batch {batch_index}")
                else:
                    print(f"Warning: No valid points generated for batch {batch_index}")
                    
    except Exception as e:
        print(f"Error processing batch {batch_index}: {e}")
    
    # Only upload if we successfully generated PLY data
    if ply_bytes:
        # upload colored point cloud as tensor to S3
        s3_key = f"{video_name}/batch_{batch_index}_pointcloud.ply"
        s3_client.put_object(Bucket=target_bucket, Key=s3_key, Body=ply_bytes)
        print(f"Uploaded batch {batch_index} pointcloud to s3://{target_bucket}/{s3_key}")
    else:
        print(f"Warning: No PLY data generated for batch {batch_index}")

    # upload each frame's data
    # compute global frame index offset by start_idx
    # base_idx = start_idx + batch_index * len(frames)
    # for i, frame in tqdm(enumerate(frames), desc=f"Batch {batch_index+1}/{total_batches}", unit="frame"):
    # # for i, frame in enumerate(frames):
    #     global_idx = base_idx + i
    #     depth = results.get('depths')[i]
    #     proj = results.get('projection_matrix')
    #     # extract 3x3 pose
    #     pose = torch.from_numpy(proj)[:3, :3] if isinstance(proj, np.ndarray) else proj[:3, :3]
    #     traj = results.get('trajectory')[i]
    #     upload_frame_data(s3_client, target_bucket, video_name, global_idx, frame, depth, pose, traj)

def process_streaming_video(model, url, batch_size, s3_client, target_bucket):
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
    start_seconds = start_idx / fps

    # 2) Launch ffmpeg as a subprocess, outputting rawvideo RGB24 to stdout

    # launch ffmpeg process, skipping to start_idx by frame number if resuming
    # reconnect flags
    recon_args = ['-reconnect', '1',
                  '-reconnect_streamed', '1',
                  '-reconnect_delay_max', '2']
    if start_idx > 0:
        process = (
            ffmpeg
            .input(url, ss=start_seconds)
            .filter('setpts', 'PTS-STARTPTS')
            .filter('scale', -1, 336)  # ensure correct resolution
            .output('pipe:', format='rawvideo', pix_fmt='rgb24')
            .global_args(*recon_args)
            .run_async(pipe_stdout=True, pipe_stderr=True)
        )
    else:
        process = (
            ffmpeg
            .input(url)
            .filter('scale', -1, 336)  # ensure correct resolution
            .output('pipe:', format='rawvideo', pix_fmt='rgb24')
            .global_args(*recon_args)
            .run_async(pipe_stdout=True, pipe_stderr=True)
        )

    frame_size = width * height * 3  # bytes per frame
    # initialize batch and batch counter
    batch = []
    batch_idx = 0
    # read frames
    while True:
        # read exactly one frame
        in_bytes = process.stdout.read(frame_size)
        if not in_bytes or len(in_bytes) < frame_size:
            break

        # turn bytes into H×W×3 uint8 numpy array
        frame = (
            np
            .frombuffer(in_bytes, np.uint8)
            .reshape((height, width, 3))
        )

        batch.append(frame)
        if len(batch) >= batch_size:
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
    print(f"   done processing {video_name} in {batch_idx + 1} batches")

    if process.returncode != 0:
        err = process.stderr.read().decode('utf8', errors='ignore')
        print(f"ffmpeg exited {process.returncode}:\n{err}")
    else:
        print(f"   done {video_name}")


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

    if model is None:
        print("Failed to load AnyCam model. Exiting.")
        return

    # Stream & batch-process each
    for key in tqdm(keys, desc="Videos", unit="video"):
        url = get_presigned_url(s3, args.bucket, key)
        process_streaming_video(model, url, batch_size=args.frame_batch_size,
                                s3_client=s3, target_bucket=args.target_bucket)

if __name__ == "__main__":
    start = time.time()
    main()
    print(f"\nAll done! Total time: {time.time() - start:.2f}s")

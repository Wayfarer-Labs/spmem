#!/usr/bin/env python3
"""
Test script for the streaming VGGT implementation.
"""

import torch
import numpy as np
from streaming_vggt import preprocess_frame_for_vggt, load_vggt_model, extract_colors_from_vggt_predictions, export_ply_with_colors
import glob
from PIL import Image

def test_preprocessing():
    """Test the frame preprocessing function."""
    print("Testing frame preprocessing...")
    
    # Load a test image
    test_images = glob.glob("testdata/*.png")[:30]
    if not test_images:
        print("No test images found in testdata/")
        return
    
    print(f"Testing with {len(test_images)} images...")
    
    # Test preprocessing individual frames
    preprocessed_frames = []
    for img_path in test_images:
        # Load image as numpy array (simulating what we get from ffmpeg)
        img = Image.open(img_path)
        frame_np = np.array(img)
        
        # Preprocess for VGGT
        preprocessed = preprocess_frame_for_vggt(frame_np)
        preprocessed_frames.append(preprocessed)
        print(f"  {img_path}: {frame_np.shape} -> {preprocessed.shape}")
    
    # Stack into batch
    batch = torch.stack(preprocessed_frames)
    print(f"Batch shape: {batch.shape}")
    
    return batch

def test_vggt_model():
    """Test the VGGT model with preprocessed frames."""
    print("\nTesting VGGT model...")
    
    # Get preprocessed frames
    batch = test_preprocessing()
    if batch is None:
        return
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load model
    print("Loading VGGT model...")
    model = load_vggt_model()
    
    # Prepare input (add batch dimension)
    images = batch.to(device).unsqueeze(0)  # (1, N, 3, H, W)
    print(f"Input shape: {images.shape}")
    
    # Run inference
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    print(f"Using dtype: {dtype}")
    
    try:
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=dtype):
                predictions = model(images)
                
                print("Prediction keys:", list(predictions.keys()))
                for key, value in predictions.items():
                    if isinstance(value, torch.Tensor):
                        print(f"  {key}: {value.shape}")
                    else:
                        print(f"  {key}: {type(value)}")
                
                # Extract colored point cloud
                world_points = predictions["world_points"]
                world_points_conf = predictions["world_points_conf"] 
                
                valid_points, valid_colors, valid_conf = extract_colors_from_vggt_predictions(
                    world_points, images, world_points_conf, conf_threshold=0.1
                )
                
                print(f"Generated {len(valid_points)} valid colored points")
                
                if len(valid_points) > 0:
                    # Export to PLY
                    ply_bytes = export_ply_with_colors(valid_points, valid_colors, confidence=valid_conf)
                    
                    # Save test PLY file
                    with open("testdata/streaming_test_output.ply", "wb") as f:
                        f.write(ply_bytes)
                    print("Saved colored point cloud to testdata/streaming_test_output.ply")
                
    except Exception as e:
        print(f"Error during model inference: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_vggt_model()

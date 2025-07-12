import torch
import numpy as np
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
import glob

def extract_colors_from_images(world_points, images, world_points_conf, conf_threshold=0.1):
    """
    Extract RGB colors from input images corresponding to world points.
    
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
    conf_flat = world_points_conf_np.reshape(-1)
    
    # Apply confidence threshold
    valid_mask = conf_flat > conf_threshold
    
    valid_points = world_points_flat[valid_mask]
    valid_colors = colors_flat[valid_mask]
    valid_conf = conf_flat[valid_mask]
    
    return valid_points, valid_colors, valid_conf

# add color to the points
def export_ply_with_colors(points, colors, filename, confidence=None):
    """Export points with colors to a PLY file."""
    print(f"Points shape: {points.shape}, Colors shape: {colors.shape}")
    
    with open(filename, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        if confidence is not None:
            f.write("property float confidence\n")
        f.write("end_header\n")
        
        if confidence is not None:
            for point, color, conf in zip(points, colors, confidence):
                f.write(f"{point[0]} {point[1]} {point[2]} {color[0]} {color[1]} {color[2]} {conf}\n")
        else:
            for point, color in zip(points, colors):
                f.write(f"{point[0]} {point[1]} {point[2]} {color[0]} {color[1]} {color[2]}\n")

def export_ply(points, filename):
    print(points.shape)
    """Export points to a PLY file."""
    with open(filename, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("end_header\n")
        for point in points:
            f.write(f"{point[0]} {point[1]} {point[2]}\n")

if __name__ == "__main__":
    image_names = glob.glob("testdata/*.png")  # Load all PNG images from testdata directory
    print(f"Found {len(image_names)} images in testdata directory.")
    print(image_names)
    image_names = image_names[:50]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    # Determine dtype based on device and GPU capability
    if device == "cuda":
        # bfloat16 is supported on Ampere GPUs (Compute Capability 8.0+)
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    else:
        # Default to float32 on CPU
        dtype = torch.float32

    # Initialize the model and load the pretrained weights.
    # This will automatically download the model weights the first time it's run, which may take a while.
    model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)

    # Load and preprocess example images (replace with your own image paths)
    # image_names = ["testdata/frame_00062.png", "testdata/frame_00097.png", "testdata/frame_00176.png", "testdata/frame_00268.png"]  
    images = load_and_preprocess_images(image_names).to(device)

    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            # Predict attributes including cameras, depth maps, and point maps.
            predictions = model(images)
            
            # Debug: print prediction keys and shapes
            print("Prediction keys and shapes:")
            for key, value in predictions.items():
                if isinstance(value, torch.Tensor):
                    print(f"  {key}: {value.shape}")
                else:
                    print(f"  {key}: {type(value)}")

            # Extract world points and colors
            world_points = predictions["world_points"].cpu().numpy()[0]  # Shape: (4, 294, 518, 3)
            world_points_conf = predictions["world_points_conf"].cpu().numpy()[0]  # Shape: (4, 294, 518)
            
            # Convert images to numpy and get colors at corresponding pixels
            # images is of shape (4, 3, 294, 518), we need to transpose to (4, 294, 518, 3)
            image_colors = images.cpu().numpy().transpose(0, 2, 3, 1)  # (4, 294, 518, 3)
            image_colors = (image_colors * 255).astype(np.uint8)  # Convert to 0-255 range
            
            # Reshape everything to 1D
            world_points_flat = world_points.reshape(-1, 3)
            colors_flat = image_colors.reshape(-1, 3)
            conf_flat = world_points_conf.reshape(-1)
            
            # Filter by confidence (keep points with confidence > threshold)
            conf_threshold = 0.1  # Adjust this threshold as needed
            valid_mask = conf_flat > conf_threshold
            
            valid_points = world_points_flat[valid_mask]
            valid_colors = colors_flat[valid_mask]
            
            print(f"Total points: {len(world_points_flat)}")
            print(f"Valid points after filtering: {len(valid_points)}")
            
            # Export both versions
            export_ply(predictions["world_points"].cpu().numpy()[0].reshape(-1, 3), "testdata/output.ply")
            export_ply_with_colors(valid_points, valid_colors, "testdata/output_with_colors.ply")

#!/usr/bin/env python3
"""
Batch Processing Script for VGGT 3D Reconstruction
==================================================

This script processes image sequences in overlapping batches and aligns
the resulting point clouds using ICP registration.

Directory Structure:
    input_folder/
    └── images/           # Source images for reconstruction
    
    output_folder/
    ├── batch_0/
    │   ├── images/       # Batch 0 images
    │   └── sparse/       # Batch 0 reconstruction
    ├── batch_1/
    │   ├── images/       # Batch 1 images  
    │   └── sparse/       # Batch 1 reconstruction
    └── aligned/
        ├── combined.ply  # Final aligned point cloud
        └── transforms/   # Transformation matrices
"""

import os
import argparse
import numpy as np
import glob
import shutil
from pathlib import Path
import trimesh
import torch

# Import VGGT demo functionality
from colmap_demo import demo_fn, parse_args as demo_parse_args
from registration import icp_umeyama, icp_color_trimmed


def parse_args():
    parser = argparse.ArgumentParser(description="Batch Processing with Point Cloud Alignment")
    
    # Input/Output
    parser.add_argument("--input_dir", type=str, required=True, 
                       help="Directory containing input images")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Directory to save batch results and aligned output")
    
    # Batch parameters
    parser.add_argument("--batch_size", type=int, default=20,
                       help="Number of images per batch")
    parser.add_argument("--overlap", type=int, default=5,
                       help="Number of overlapping images between batches")
    parser.add_argument("--max_batches", type=int, default=None,
                       help="Maximum number of batches to process (None for all)")
    
    # VGGT parameters (inherit from demo)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_ba", action="store_true", default=False)
    parser.add_argument("--max_reproj_error", type=float, default=8.0)
    parser.add_argument("--shared_camera", action="store_true", default=False)
    parser.add_argument("--camera_type", type=str, default="SIMPLE_PINHOLE")
    parser.add_argument("--vis_thresh", type=float, default=0.2)
    parser.add_argument("--query_frame_num", type=int, default=8)
    parser.add_argument("--max_query_pts", type=int, default=512)
    parser.add_argument("--fine_tracking", action="store_true", default=True)
    parser.add_argument("--conf_thres_value", type=float, default=1.01)
    
    # Alignment parameters
    parser.add_argument("--alignment_method", type=str, choices=["icp", "umeyama"], 
                       default="icp", help="Point cloud alignment method")
    parser.add_argument("--normalize_scale", action="store_true", default=True,
                       help="Normalize point clouds to unit mean distance before alignment (recommended for VGGT)")
    parser.add_argument("--no_normalize_scale", dest="normalize_scale", action="store_false",
                       help="Disable scale normalization before alignment")
    parser.add_argument("--icp_max_iter", type=int, default=250,
                       help="Maximum ICP iterations")
    parser.add_argument("--icp_tolerance", type=float, default=1e-6,
                       help="ICP convergence tolerance")
    parser.add_argument("--min_overlap_points", type=int, default=10000,
                       help="Minimum overlapping points required for alignment")
    
    return parser.parse_args()


def create_batches(image_paths, batch_size, overlap):
    """
    Create overlapping batches from image paths.
    
    Args:
        image_paths: List of image file paths
        batch_size: Number of images per batch
        overlap: Number of overlapping images between consecutive batches
        
    Returns:
        List of batches, where each batch is a list of image paths
    """
    if len(image_paths) <= batch_size:
        return [image_paths]
    
    batches = []
    start_idx = 0
    
    while start_idx < len(image_paths):
        end_idx = min(start_idx + batch_size, len(image_paths))
        batch = image_paths[start_idx:end_idx]
        batches.append(batch)
        
        # If this is the last batch (reached end), break
        if end_idx >= len(image_paths):
            break
            
        # Move start index forward by (batch_size - overlap)
        start_idx += batch_size - overlap
        
        # Ensure we don't create a batch that's too small
        remaining = len(image_paths) - start_idx
        if remaining < batch_size and remaining > 0:
            # Merge remaining images with the last batch
            start_idx = len(image_paths) - batch_size
    
    return batches


def setup_batch_directory(batch_dir, batch_images):
    """
    Set up directory structure for a batch and copy images.
    """
    os.makedirs(batch_dir, exist_ok=True)
    batch_images_dir = os.path.join(batch_dir, "images")
    os.makedirs(batch_images_dir, exist_ok=True)
    
    # Copy images to batch directory
    for img_path in batch_images:
        img_name = os.path.basename(img_path)
        dst_path = os.path.join(batch_images_dir, img_name)
        shutil.copy2(img_path, dst_path)
    
    return batch_images_dir


def load_point_cloud_from_ply(ply_path):
    """
    Load point cloud from PLY file.
    """
    if not os.path.exists(ply_path):
        return None, None
        
    mesh = trimesh.load(ply_path)
    if hasattr(mesh, 'vertices'):
        points = np.array(mesh.vertices)
        colors = np.array(mesh.colors) if hasattr(mesh, 'colors') else None
        return points, colors
    return None, None


def find_overlapping_points(points1, points2, distance_threshold=0.1):
    """
    Find overlapping points between two point clouds using distance threshold.
    
    Returns:
        indices1, indices2: indices of overlapping points in each cloud
    """
    from sklearn.neighbors import NearestNeighbors
    
    if len(points1) == 0 or len(points2) == 0:
        return np.array([]), np.array([])
    
    # Find nearest neighbors
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(points2)
    distances, indices = nbrs.kneighbors(points1)
    
    # Filter by distance threshold
    valid_mask = distances.flatten() < distance_threshold
    indices1 = np.where(valid_mask)[0]
    indices2 = indices[valid_mask].flatten()
    
    return indices1, indices2


def normalize_point_cloud_scale(points):
    """
    Normalize point cloud to unit mean distance from origin.
    
    As noted by chenguolin: VGGT is trained on normalized scenes with unit mean distance,
    so outputs from different runs should be scaled by mean distance before alignment.
    
    Returns:
        normalized_points: points scaled to unit mean distance
        scale_factor: factor used for normalization (to reverse if needed)
    """
    if len(points) == 0:
        return points, 1.0
    
    # Calculate mean distance from origin
    distances = np.linalg.norm(points, axis=1)
    mean_distance = np.mean(distances)
    
    if mean_distance == 0:
        return points, 1.0
    
    # Normalize to unit mean distance
    scale_factor = 1.0 / mean_distance
    normalized_points = points * scale_factor
    
    return normalized_points, scale_factor


def align_point_clouds(source_points, source_color, target_points, target_color, normalize_scale=True, **kwargs):
    """
    Align source point cloud to target point cloud.
    
    Args:
        source_points: source point cloud to align
        target_points: target point cloud (reference)
        normalize_scale: whether to normalize point clouds to unit mean distance before alignment
        **kwargs: additional parameters for alignment methods
    
    Returns:
        aligned_points: transformed source points
        transform_info: dictionary with transformation details including normalization info
    """
    original_source = source_points.copy()
    original_target = target_points.copy()
    
    # Normalize point clouds to unit mean distance (VGGT requirement)
    if normalize_scale:
        print("  Normalizing point clouds to unit mean distance...")
        source_norm, source_scale = normalize_point_cloud_scale(source_points)
        target_norm, target_scale = normalize_point_cloud_scale(target_points)
        
        print(f"    Source scale factor: {source_scale:.6f}")
        print(f"    Target scale factor: {target_scale:.6f}")
        
        # Use normalized points for alignment
        align_source = source_norm
        align_target = target_norm
    else:
        align_source = source_points
        align_target = target_points
        source_scale = 1.0
        target_scale = 1.0
    
    # Perform alignment on normalized points
    # aligned_norm, scale, R, t, rmse_history = icp_umeyama(
    #     align_source, align_target,
    #     max_iterations=kwargs.get('max_iter', 50),
    #     tol=kwargs.get('tolerance', 1e-6)
    # )
    aligned_norm, scale, R, t, rmse_history = icp_color_trimmed(align_source, source_color, align_target, target_color,
                                                   max_iterations=kwargs.get('max_iter', 50))

    final_rmse = rmse_history[-1] if rmse_history else float('inf')
    
    # Transform back to original scale space
    if normalize_scale:
        # The transformation was computed in normalized space, now apply to original space
        # First, normalize original source
        source_normalized = original_source * source_scale
        # Apply transformation
        aligned_transformed = (scale * (R @ source_normalized.T)).T + t
        # Scale back to target's original scale
        aligned_points = aligned_transformed / target_scale
    else:
        aligned_points = aligned_norm
    
    transform_info = {
        'scale': scale,
        'rotation': R,
        'translation': t,
        'rmse_history': rmse_history,
        'final_rmse': final_rmse,
        'source_normalization_scale': source_scale,
        'target_normalization_scale': target_scale,
        'normalized_alignment': normalize_scale
    }
    
    return aligned_points, transform_info


def process_batch(batch_idx, batch_images, output_dir, args):
    """
    Process a single batch through VGGT.
    """
    print(f"\n{'='*50}")
    print(f"Processing Batch {batch_idx}")
    print(f"{'='*50}")
    
    # Setup batch directory
    batch_dir = os.path.join(output_dir, f"batch_{batch_idx}")
    batch_images_dir = setup_batch_directory(batch_dir, batch_images)
    
    print(f"Batch directory: {batch_dir}")
    print(f"Number of images: {len(batch_images)}")
    
    # Create arguments for VGGT demo
    demo_args = argparse.Namespace()
    demo_args.scene_dir = batch_dir
    demo_args.seed = args.seed
    demo_args.use_ba = args.use_ba
    demo_args.max_reproj_error = args.max_reproj_error
    demo_args.shared_camera = args.shared_camera
    demo_args.camera_type = args.camera_type
    demo_args.vis_thresh = args.vis_thresh
    demo_args.query_frame_num = args.query_frame_num
    demo_args.max_query_pts = args.max_query_pts
    demo_args.fine_tracking = args.fine_tracking
    demo_args.conf_thres_value = args.conf_thres_value
    
    # Run VGGT on batch
    try:
        with torch.no_grad():
            success = demo_fn(demo_args)
        
        if success:
            print(f"✓ Batch {batch_idx} processed successfully")
            return os.path.join(batch_dir, "sparse", "points.ply")
        else:
            print(f"✗ Batch {batch_idx} processing failed")
            return None
            
    except Exception as e:
        print(f"✗ Error processing batch {batch_idx}: {str(e)}")
        return None


def main():
    args = parse_args()
    
    print("Batch Processing with Point Cloud Alignment")
    print("=" * 50)
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Batch size: {args.batch_size}")
    print(f"Overlap: {args.overlap}")
    print(f"Alignment method: {args.alignment_method}")
    print(f"Scale normalization: {'enabled' if args.normalize_scale else 'disabled'}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get all image paths
    image_dir = os.path.join(args.input_dir, "images")
    if not os.path.exists(image_dir):
        image_dir = args.input_dir
    
    image_paths = sorted(glob.glob(os.path.join(image_dir, "*.[jp][pn]g")))
    if not image_paths:
        raise ValueError(f"No images found in {image_dir}")
    
    print(f"Found {len(image_paths)} images")
    
    # Create batches
    batches = create_batches(image_paths, args.batch_size, args.overlap)
    if args.max_batches:
        batches = batches[:args.max_batches]
    
    print(f"Created {len(batches)} batches")
    
    # Process each batch
    batch_point_clouds = []
    batch_transforms = []
    
    for batch_idx, batch_images in enumerate(batches):
        ply_path = process_batch(batch_idx, batch_images, args.output_dir, args)
        
        if ply_path and os.path.exists(ply_path):
            points, colors = load_point_cloud_from_ply(ply_path)
            if points is not None:
                batch_point_clouds.append({
                    'points': points,
                    'colors': colors,
                    'ply_path': ply_path,
                    'batch_idx': batch_idx
                })
                print(f"✓ Loaded {len(points)} points from batch {batch_idx}")
            else:
                print(f"✗ Failed to load points from {ply_path}")
        else:
            print(f"✗ No point cloud generated for batch {batch_idx}")
    
    if len(batch_point_clouds) < 2:
        print("Warning: Need at least 2 successful batches for alignment")
        return
    
    # Align point clouds
    print(f"\n{'='*50}")
    print("Aligning Point Clouds")
    print(f"{'='*50}")
    
    aligned_point_clouds = [batch_point_clouds[0]]  # First batch is reference
    cumulative_transforms = []
    
    for i in range(1, len(batch_point_clouds)):
        print(f"\nAligning batch {i} to batch {i-1}")
        
        source_points = batch_point_clouds[i]['points']
        source_colors = batch_point_clouds[i]['colors']
        target_points = aligned_point_clouds[-1]['points']
        target_colors = aligned_point_clouds[-1]['colors']

        # Find overlapping regions (simplified - you might want more sophisticated overlap detection)
        print(f"Source points: {len(source_points)}")
        print(f"Target points: {len(target_points)}")
        
        try:
            aligned_points, transform_info = align_point_clouds(
                source_points, source_colors, target_points, target_colors,
                normalize_scale=args.normalize_scale,
                max_iter=args.icp_max_iter,
                tolerance=args.icp_tolerance
            )
            
            print(f"✓ Alignment successful")
            if transform_info['final_rmse'] is not None:
                print(f"  Final RMSE: {transform_info['final_rmse']:.6f}")
            if transform_info.get('normalized_alignment', False):
                print(f"  Source normalization scale: {transform_info['source_normalization_scale']:.6f}")
                print(f"  Target normalization scale: {transform_info['target_normalization_scale']:.6f}")
                print(f"  Alignment scale (normalized space): {transform_info['scale']:.6f}")
            
            # Store aligned point cloud
            aligned_batch = batch_point_clouds[i].copy()
            aligned_batch['points'] = aligned_points
            aligned_point_clouds.append(aligned_batch)
            cumulative_transforms.append(transform_info)
            
        except Exception as e:
            print(f"✗ Alignment failed: {str(e)}")
            # Add unaligned points (fallback)
            aligned_point_clouds.append(batch_point_clouds[i])
            cumulative_transforms.append(None)
    
    # Combine all aligned point clouds
    print(f"\n{'='*50}")
    print("Combining Point Clouds")
    print(f"{'='*50}")
    
    all_points = []
    all_colors = []
    
    for batch_data in aligned_point_clouds:
        all_points.append(batch_data['points'])
        if batch_data['colors'] is not None:
            all_colors.append(batch_data['colors'])
    
    combined_points = np.vstack(all_points)
    combined_colors = np.vstack(all_colors) if all_colors else None
    
    print(f"Combined point cloud: {len(combined_points)} points")
    
    # Save results
    aligned_dir = os.path.join(args.output_dir, "aligned")
    os.makedirs(aligned_dir, exist_ok=True)
    
    # Save combined point cloud
    combined_ply_path = os.path.join(aligned_dir, "combined.ply")
    if combined_colors is not None:
        combined_cloud = trimesh.PointCloud(combined_points, colors=combined_colors)
    else:
        combined_cloud = trimesh.PointCloud(combined_points)
    combined_cloud.export(combined_ply_path)
    print(f"✓ Saved combined point cloud: {combined_ply_path}")
    
    # Save transformation info
    transforms_dir = os.path.join(aligned_dir, "transforms")
    os.makedirs(transforms_dir, exist_ok=True)
    
    for i, transform_info in enumerate(cumulative_transforms):
        if transform_info is not None:
            transform_file = os.path.join(transforms_dir, f"batch_{i+1}_to_{i}.npz")
            np.savez(transform_file, **transform_info)
            print(f"✓ Saved transform {i+1}->{i}: {transform_file}")
    
    print(f"\n{'='*50}")
    print("Batch Processing Complete!")
    print(f"{'='*50}")
    print(f"Output directory: {args.output_dir}")
    print(f"Combined point cloud: {combined_ply_path}")
    print(f"Individual batches: {len(batch_point_clouds)}")
    print(f"Total points: {len(combined_points)}")


if __name__ == "__main__":
    main()

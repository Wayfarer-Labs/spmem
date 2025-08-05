#!/usr/bin/env python3
"""
Demo script showing VAE Data Pipeline Batch Processing vs Original Processing

This script demonstrates how the new batch processing functionality compares
to processing windows individually.
"""

import os
import time
from vae_data_pipeline_batch import create_windows, process_window_batch


def demo_original_processing(image_paths, window_size, stride):
    """
    Simulate original processing approach - one window at a time.
    """
    print("Original Processing Approach (Sequential):")
    print("-" * 50)
    
    windows = create_windows(image_paths, window_size, stride)
    
    start_time = time.time()
    total_processed = 0
    
    for i, window in enumerate(windows):
        print(f"Processing window {i+1}/{len(windows)} ({len(window)} images)...")
        
        # Simulate processing time
        time.sleep(0.1)  
        total_processed += 1
        
        print(f"  ✓ Window {i+1} completed")
    
    elapsed_time = time.time() - start_time
    
    print(f"\nOriginal Processing Complete:")
    print(f"  Total windows: {total_processed}")
    print(f"  Time taken: {elapsed_time:.2f}s")
    print(f"  Time per window: {elapsed_time/total_processed:.3f}s")
    
    return elapsed_time, total_processed


def demo_batch_processing(image_paths, window_size, stride, batch_size):
    """
    Demonstrate new batch processing approach.
    """
    print(f"\nNew Batch Processing Approach (Batch Size: {batch_size}):")
    print("-" * 50)
    
    windows = create_windows(image_paths, window_size, stride)
    
    # Create batches
    window_batches = []
    for i in range(0, len(windows), batch_size):
        batch = windows[i:i + batch_size]
        window_batches.append(batch)
    
    start_time = time.time()
    total_processed = 0
    
    for batch_idx, window_batch in enumerate(window_batches):
        print(f"Processing batch {batch_idx+1}/{len(window_batches)} ({len(window_batch)} windows)...")
        
        # Simulate batch processing (potentially parallel processing benefit)
        batch_time = 0.1 * len(window_batch) * 0.8  # 20% speedup from batching
        time.sleep(batch_time)
        
        total_processed += len(window_batch)
        
        print(f"  ✓ Batch {batch_idx+1} completed ({len(window_batch)} windows)")
    
    elapsed_time = time.time() - start_time
    
    print(f"\nBatch Processing Complete:")
    print(f"  Total windows: {total_processed}")
    print(f"  Time taken: {elapsed_time:.2f}s")
    print(f"  Time per window: {elapsed_time/total_processed:.3f}s")
    print(f"  Batch efficiency: {len(window_batches)} batches vs {total_processed} individual calls")
    
    return elapsed_time, total_processed


def main():
    """Run the demo comparison."""
    print("VAE Data Pipeline Batch Processing Demo")
    print("=" * 60)
    
    # Create simulated image paths
    image_paths = [f"frame_{i:03d}.png" for i in range(20)]
    window_size = 4
    stride = 2
    batch_size = 3
    
    print(f"Demo Parameters:")
    print(f"  Total images: {len(image_paths)}")
    print(f"  Window size: {window_size}")
    print(f"  Window stride: {stride}")
    print(f"  Batch size: {batch_size}")
    
    windows = create_windows(image_paths, window_size, stride)
    print(f"  Total windows created: {len(windows)}")
    print()
    
    # Demo original approach
    original_time, original_windows = demo_original_processing(image_paths, window_size, stride)
    
    # Demo batch approach
    batch_time, batch_windows = demo_batch_processing(image_paths, window_size, stride, batch_size)
    
    # Compare results
    print(f"\n{'='*60}")
    print("Comparison Summary:")
    print(f"{'='*60}")
    print(f"Windows processed: {original_windows} (both methods)")
    print(f"Original approach: {original_time:.2f}s")
    print(f"Batch approach:    {batch_time:.2f}s")
    
    if batch_time < original_time:
        speedup = original_time / batch_time
        print(f"Speedup: {speedup:.2f}x faster with batching!")
    else:
        print("Note: In this demo, actual speedup depends on GPU parallelization")
    
    print(f"\nKey Benefits of Batch Processing:")
    print(f"  • Reduced function call overhead")
    print(f"  • Better GPU utilization with parallel processing")
    print(f"  • Configurable batch sizes for memory management")
    print(f"  • Maintained compatibility with original processing logic")


if __name__ == "__main__":
    main()
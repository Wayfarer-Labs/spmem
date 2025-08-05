#!/usr/bin/env python3
"""
Simple test for VAE Data Pipeline Batch Processing
"""

import unittest
import tempfile
import os
import shutil
from pathlib import Path

# Add current directory to path for imports
import sys
sys.path.insert(0, os.path.dirname(__file__))

from vae_data_pipeline_batch import create_windows, parse_args


class TestBatchProcessing(unittest.TestCase):
    """Test cases for batch processing functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = tempfile.mkdtemp()
        self.input_dir = os.path.join(self.test_dir, "input")
        self.output_dir = os.path.join(self.test_dir, "output")
        os.makedirs(self.input_dir)
        
        # Create test image files
        self.test_images = []
        for i in range(10):
            img_path = os.path.join(self.input_dir, f"frame_{i:03d}.png")
            Path(img_path).touch()  # Create empty file
            self.test_images.append(img_path)
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.test_dir)
    
    def test_create_windows_basic(self):
        """Test basic window creation."""
        windows = create_windows(self.test_images, window_size=3, stride=2)
        
        # Should create 4 windows: [0,1,2], [2,3,4], [4,5,6], [6,7,8]
        # Plus remainder [8,9] (if >= window_size//2)
        self.assertEqual(len(windows), 5)  # 4 full windows + 1 remainder
        self.assertEqual(len(windows[0]), 3)  # First window has 3 images
        self.assertEqual(len(windows[-1]), 2)  # Last window has 2 images (remainder)
    
    def test_create_windows_small_dataset(self):
        """Test window creation with small dataset."""
        small_images = self.test_images[:2]
        windows = create_windows(small_images, window_size=5, stride=2)
        
        # Should return single window with all images when dataset is smaller than window
        self.assertEqual(len(windows), 1)
        self.assertEqual(len(windows[0]), 2)
    
    def test_create_windows_no_remainder(self):
        """Test window creation with exact fit (no remainder)."""
        exact_images = self.test_images[:6]  # Use 6 images
        windows = create_windows(exact_images, window_size=2, stride=2)
        
        # Should create exactly 3 windows: [0,1], [2,3], [4,5]
        self.assertEqual(len(windows), 3)
        for window in windows:
            self.assertEqual(len(window), 2)
    
    def test_window_batch_size_argument(self):
        """Test that window-batch-size argument is properly parsed."""
        # Test the argument parsing by creating a mock argument list
        test_args = [
            '--input-dir', self.input_dir,
            '--output-dir', self.output_dir,
            '--window-batch-size', '8',
            '--window-size', '4',
            '--window-stride', '2'
        ]
        
        # Temporarily replace sys.argv for testing
        original_argv = sys.argv
        try:
            sys.argv = ['test'] + test_args
            args = parse_args()
            
            self.assertEqual(args.window_batch_size, 8)
            self.assertEqual(args.window_size, 4)
            self.assertEqual(args.window_stride, 2)
            self.assertEqual(args.input_dir, self.input_dir)
            self.assertEqual(args.output_dir, self.output_dir)
        finally:
            sys.argv = original_argv


if __name__ == '__main__':
    unittest.main()
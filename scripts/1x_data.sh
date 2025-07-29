#!/bin/bash
python vae_data_pipeline.py \
    --video-dir /path/to/mp4s \
    --output-dir /path/to/output \
    --kernel-size 50 \
    --stride 6 \
    --dilation 6 \
    --files-per-subdir 500 \
    --num-gpus 8
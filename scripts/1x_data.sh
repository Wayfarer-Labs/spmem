#!/bin/bash
uv run vae_data_pipeline_single.py \
    --video-dir /mnt/data/shahbuland/video-proc-2/datasets/cod-yt \
    --output-dir . \
    --kernel-size 15 \
    --stride 6 \
    --dilation 6 \
    --files-per-subdir 500 \
    --num-gpus 1

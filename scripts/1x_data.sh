#!/bin/bash
uv run /workspace/vae_data_pipeline_single.py \
    --video-dir /mnt/data/shahbuland/video-proc-2/datasets/cod-yt \
    --output-dir . \
    --kernel-size 101 \
    --stride 6 \
    --dilation 6 \
    --files-per-subdir 500 \
    --num-gpus 4 \
    --batch-size 1
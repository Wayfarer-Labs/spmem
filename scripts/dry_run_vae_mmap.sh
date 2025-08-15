#!/usr/bin/env bash
# Dry run helper for vae_data_pipeline_mmap.py
# Usage:
#   ./scripts/dry_run_vae_mmap.sh /path/to/video_root /tmp/vae_out \
#       --kernel-size 5 --stride 3 --dilation 1 --batch-size 4 --world-size 4
# Additional args after the first two positional parameters are forwarded.
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <video_root> <output_dir> [extra args...]" >&2
  exit 1
fi

VIDEO_ROOT=$1; shift
OUTPUT_DIR=$1; shift

# Default params (can be overridden by providing them after the two required ones)
python vae_data_pipeline_mmap.py \
  --video-dir "${VIDEO_ROOT}" \
  --output-dir "${OUTPUT_DIR}" \
  --batch-size 4 \
  --kernel-size 5 \
  --stride 3 \
  --dilation 1 \
  --files-per-subdir 500 \
  --world-size 1 \
  --rank 0 \
  --dry-run \
  "$@"

#!/usr/bin/env bash
# Driver script for vae_data_pipeline_single.py (tensorification pipeline)
# Wraps argument parsing + sensible defaults + safety checks.
#
# Usage examples:
#   scripts/run_tensorification_pipeline.sh --root /data/videos
#   scripts/run_tensorification_pipeline.sh -r ./videos -c 1500 -s 360 640 -p 48
#   scripts/run_tensorification_pipeline.sh -r ./videos --force
#   scripts/run_tensorification_pipeline.sh -r ./videos --dry-run
#
# Options:
#   -r | --root DIR          Root directory containing (possibly nested) .mp4 files (required)
#   -c | --chunk-size N      Frames per tensor chunk (default: 2000)
#   -s | --size H W          Output frame size (height width) (default: 360 640)
#   -p | --cpus N            Number of CPUs to allocate to Ray (default: system detected or 8)
#   -f | --force             Force overwrite existing split rgb tensors
#   -n | --dry-run           List videos that would be processed, then exit
#   (Python interpreter selection removed; always uses `uv run`)
#   -v | --verbose           Echo the final command before running
#   -h | --help              Show this help
#
# Exit codes:
#   0 success / dry-run success
#   1 usage error
#   2 prerequisite missing
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PY_FILE="${REPO_ROOT}/tensors_from_mp4s_pyav.py"

if [[ ! -f "$PY_FILE" ]]; then
  echo "[ERR] Cannot locate tensors_from_mp4s_pyav.py at $PY_FILE" >&2
  exit 2
fi

# Defaults
ROOT_DIR=/mnt/data/shahbuland/video-proc-2/datasets/cod-yt
CHUNK_SIZE=2000
SIZE_H=518
SIZE_W=921
# Interpreter handled by `uv run`
FORCE_OVERWRITE=1
DRY_RUN=0
VERBOSE=0

CPUS=2

print_help() {
  sed -n '1,65p' "$0" | grep -v '^#!/' | sed '/^set -e/d'
}

# Argument parsing
while [[ $# -gt 0 ]]; do
  case "$1" in
    -r|--root)
      ROOT_DIR="$2"; shift 2;;
    -c|--chunk-size)
      CHUNK_SIZE="$2"; shift 2;;
    -s|--size)
      if [[ $# -lt 3 ]]; then echo "[ERR] --size requires two ints" >&2; exit 1; fi
      SIZE_H="$2"; SIZE_W="$3"; shift 3;;
    -p|--cpus)
      CPUS="$2"; shift 2;;
    -f|--force)
      FORCE_OVERWRITE=1; shift;;
    -n|--dry-run)
      DRY_RUN=1; shift;;
    -v|--verbose)
      VERBOSE=1; shift;;
    -h|--help)
      print_help; exit 0;;
    --) shift; break;;
    -*) echo "[ERR] Unknown option: $1" >&2; print_help; exit 1;;
    *) echo "[ERR] Unexpected argument: $1" >&2; print_help; exit 1;;
  esac
done

if [[ -z "$ROOT_DIR" ]]; then
  echo "[ERR] --root is required" >&2
  print_help
  exit 1
fi
if [[ ! -d "$ROOT_DIR" ]]; then
  echo "[ERR] Root dir not found: $ROOT_DIR" >&2
  exit 2
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "[ERR] 'uv' is not installed or not on PATH. See: https://github.com/astral-sh/uv" >&2
  exit 2
fi

# Build python args
ARGS=("$PY_FILE" --root_dir "$ROOT_DIR" --chunk_size "$CHUNK_SIZE" --output_size "$SIZE_H" "$SIZE_W" --num_cpus "$CPUS")
if [[ $FORCE_OVERWRITE -eq 1 ]]; then
  ARGS+=(--force_overwrite)
fi

# Dry-run listing
if [[ $DRY_RUN -eq 1 ]]; then
  echo "[DRY-RUN] Would execute: $PYTHON_BIN ${ARGS[*]}"
  echo "[DRY-RUN] Discovering .mp4 files under $ROOT_DIR ..." >&2
  mapfile -t VIDEOS < <(find "$ROOT_DIR" -type f -name '*.mp4' | sort)
  if [[ ${#VIDEOS[@]} -eq 0 ]]; then
    echo "[DRY-RUN] No videos found." >&2
    exit 0
  fi
  echo "[DRY-RUN] Found ${#VIDEOS[@]} video(s):" >&2
  printf '  %s\n' "${VIDEOS[@]}" >&2
  exit 0
fi

if [[ $VERBOSE -eq 1 ]]; then
  echo "[INFO] Running: uv run ${ARGS[*]}"
fi

exec uv run "${ARGS[@]}"

#!/usr/bin/env python3
"""
VAE Data Pipeline (MMAP / Chunked Tensor Version)

This script is a near-copy of `vae_data_pipeline_single.py`, adapted to read
pre-extracted RGB frame tensors produced by `tensors_from_mp4s_pyav.py`.

Instead of decoding videos on-the-fly (e.g. with decord), we:
  1. Discover all `splits/` directories under a provided `--video-dir` root.
  2. Treat each `splits/` directory as a logical video composed of chunk files
     named like `XXXXXXXX_rgb.pt` (uint8 tensors saved by the extractor).
  3. Provide a lightweight virtual indexing layer that maps global frame
     indices to (chunk_file, in_chunk_index) so we can assemble sliding windows
     without loading the entire video into RAM.

Core model / window / saving functionality is intentionally left equivalent
(aside from minor refactors needed for chunked loading). Point-cloud / BA logic
is copied structurally; if the original file had incomplete sections those are
kept as placeholders here as well, to avoid changing core behavior.

Usage (example):
    python vae_data_pipeline_mmap.py \
        --video-dir /data/my_videos_root \
        --output-dir /data/vae_samples \
        --kernel-size 5 --stride 3 --dilation 1 \
        --batch-size 4 --files-per-subdir 500 \
        --num-gpus 4 --rank 0 --world-size 4

Notes:
  * A "video" here is defined as one `splits/` directory. If you placed multiple
    original .mp4 files in the SAME parent folder before running the extractor,
    their chunks may have been co-mingled; this script assumes each `splits/`
    folder corresponds to exactly one logical video.
  * Tensors are uint8 in shape [N, C, H, W] with C=3. We normalize to [0,1]
    float32 only right before feeding the model.
  * To avoid repeated disk IO, a small LRU cache (default size 2) keeps recent
    chunk tensors in memory.

"""
from __future__ import annotations
import argparse
from glob import glob
import os
import sys
import time
import copy
import json
import traceback
from pathlib import Path
from collections import OrderedDict
from typing import List, Tuple, Optional, Dict

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

# --- Imports from original pipeline (assumed available in project) ---
from vggt.models.vggt import VGGT
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map
from vggt.utils.helper import create_pixel_coordinate_grid, randomly_limit_trues
from vggt.dependency.track_predict import predict_tracks
from vggt.dependency.vggsfm_utils import build_vggsfm_tracker
from vggt.dependency.np_to_pycolmap import (
    batch_np_matrix_to_pycolmap,
    batch_np_matrix_to_pycolmap_wo_track,
)

# Optional (BA)
try:
    import pycolmap  # type: ignore
except ImportError:  # pragma: no cover
    pycolmap = None

# (Compile heavy funcs for perf if supported)
try:  # guard for environments without torch.compile
    predict_tracks = torch.compile(predict_tracks)  # type: ignore
except Exception:
    pass


def parse_args():
    p = argparse.ArgumentParser(
        description="Create VAE training data from pre-chunked video tensors using sliding window approach"
    )
    # IO
    p.add_argument("--video-dir", required=True, help="Root directory containing one or more 'splits/' directories")
    p.add_argument("--output-dir", required=True, help="Output directory for processed data")
    p.add_argument("--batch-size", type=int, required=True, help="Number of windows processed in parallel")
    # Sliding window
    p.add_argument("--kernel-size", type=int, default=5, help="Window size (must be odd)")
    p.add_argument("--stride", type=int, default=3, help="Stride between focal frames")
    p.add_argument("--dilation", type=int, default=1, help="Frame dilation within window")
    # File org
    p.add_argument("--files-per-subdir", type=int, default=500, help="Max samples per leaf directory")
    # Multi-GPU / rank
    p.add_argument("--num-gpus", type=int, default=None, help="GPUs available on node (default: all)")
    p.add_argument("--rank", type=int, default=0, help="This process rank")
    p.add_argument("--world-size", type=int, default=1, help="Total number of ranks (data split at video granularity)")
    # VGGT / COLMAP related
    p.add_argument("--use-ba", action="store_true", default=False, help="Use bundle adjustment path")
    p.add_argument("--max-reproj-error", type=float, default=8.0, help="Max reprojection error")
    p.add_argument("--shared-camera", action="store_true", default=True, help="Shared camera flag")
    p.add_argument("--camera-type", type=str, default="SIMPLE_PINHOLE", help="Camera model type")
    p.add_argument("--vis-thresh", type=float, default=0.2, help="Track visibility threshold")
    p.add_argument("--query-frame-num", type=int, default=8, help="Number of query frames for tracking")
    p.add_argument("--max-query-pts", type=int, default=4096, help="Max query points")
    p.add_argument("--fine-tracking", action="store_true", default=True, help="Fine tracking toggle")
    p.add_argument("--conf-thres-value", type=float, default=1.01, help="Depth confidence threshold (no BA)")
    # Chunk caching
    p.add_argument("--chunk-cache-size", type=int, default=2, help="LRU cache size for chunk tensors")
    p.add_argument("--index-meta", action="store_true", help="Persist per-video chunk size metadata JSON for faster restarts")
    p.add_argument("--dry-run", action="store_true", help="List discovered videos, frame counts, window counts and exit without loading models")
    p.add_argument("--allow-skip-corrupt", action="store_true", help="Skip corrupted *_rgb.pt chunk files instead of aborting (logs names)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Chunked video abstraction
# ---------------------------------------------------------------------------
class ChunkedVideo:
    """Represents a logical video backed by multiple chunk tensor files.

    Each chunk file: torch.save of uint8 tensor [n, C=3, H, W]. We lazily load
    only the needed chunks when assembling windows.
    """
    def __init__(self, splits_dir: Path, cache_size: int = 2, persist_index: bool = False, allow_skip_corrupt: bool = False):
        self.splits_dir = splits_dir
        self.cache_size = cache_size
        self.persist_index = persist_index
        self.allow_skip_corrupt = allow_skip_corrupt
        self.chunk_files = sorted(
            [p for p in splits_dir.glob("*_rgb.pt") if p.is_file()]
        )
        if not self.chunk_files:
            raise FileNotFoundError(f"No *_rgb.pt chunk files in {splits_dir}")
        self._cache: OrderedDict[str, torch.Tensor] = OrderedDict()
        self._chunk_sizes: List[int] = []
        self._shape: Optional[Tuple[int, int, int]] = None  # (C,H,W)
        self._cum_sizes: List[int] = []  # exclusive cum sum
        self._meta_path = splits_dir / "index_meta.json"
        self._build_index()

    def _build_index(self):
        if self.persist_index and self._meta_path.exists():
            try:
                meta = json.loads(self._meta_path.read_text())
                if meta.get("files") == [str(p.name) for p in self.chunk_files]:
                    self._chunk_sizes = meta["sizes"]
                    self._shape = tuple(meta["shape"])  # type: ignore
                    total = 0
                    self._cum_sizes = []
                    for s in self._chunk_sizes:
                        self._cum_sizes.append(total)
                        total += s
                    return
            except Exception:
                pass  # fall back to recompute
        # Need to scan files
        self._chunk_sizes.clear()
        self._cum_sizes.clear()
        total = 0
        corrupt_files = []
        valid_files = []
        for cf in self.chunk_files:
            try:
                t = torch.load(cf, map_location="cpu")  # [n,3,H,W]
            except Exception as e:
                corrupt_files.append((cf, str(e)))
                if not self.allow_skip_corrupt:
                    raise RuntimeError(f"Corrupted chunk file: {cf} error={e}") from e
                continue
            # Basic structural validation
            if not isinstance(t, torch.Tensor) or t.ndim != 4 or t.shape[1] != 3:
                msg = f"Invalid tensor shape in {cf}: expected [N,3,H,W] got {getattr(t,'shape',None)}"
                corrupt_files.append((cf, msg))
                if not self.allow_skip_corrupt:
                    raise RuntimeError(msg)
                continue
            n, c, h, w = t.shape
            if self._shape is None:
                self._shape = (c, h, w)
            else:
                if self._shape != (c, h, w):
                    msg = f"Inconsistent chunk frame shape in {cf}: {c,h,w} vs {self._shape}"
                    if not self.allow_skip_corrupt:
                        raise ValueError(msg)
                    corrupt_files.append((cf, msg))
                    continue
            self._chunk_sizes.append(n)
            self._cum_sizes.append(total)
            total += n
            valid_files.append(cf)
        # Replace chunk_files with only valid ones if skipping corrupt
        if self.allow_skip_corrupt:
            self.chunk_files = valid_files
            if corrupt_files:
                print(f"[ChunkedVideo] {self.splits_dir}: skipped {len(corrupt_files)} corrupt chunks; kept {len(valid_files)}")
                for cf, err in corrupt_files[:5]:
                    print(f"  corrupt: {cf.name} -> {err}")
                if len(corrupt_files) > 5:
                    print(f"  ... {len(corrupt_files)-5} more corrupt files")
        if not self._chunk_sizes:
            raise RuntimeError(f"No valid chunk files remain in {self.splits_dir} (corrupt or incompatible)")
        if self.persist_index:
            try:
                meta = {
                    "files": [p.name for p in self.chunk_files],
                    "sizes": self._chunk_sizes,
                    "shape": list(self._shape),
                }
                self._meta_path.write_text(json.dumps(meta))
            except Exception:
                pass

    @property
    def num_frames(self) -> int:
        return sum(self._chunk_sizes)

    @property
    def frame_shape(self) -> Tuple[int, int, int]:  # (C,H,W)
        if self._shape is None:
            raise RuntimeError("Frame shape not initialized")
        return self._shape

    def _load_chunk(self, idx: int) -> torch.Tensor:
        cf = self.chunk_files[idx]
        key = str(cf)
        if key in self._cache:
            val = self._cache.pop(key)
            self._cache[key] = val  # move to end (recent)
            return val
        t = torch.load(cf, map_location="cpu")  # uint8
        self._cache[key] = t
        if len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)  # evict LRU
        return t

    def _global_to_chunk(self, frame_idx: int) -> Tuple[int, int]:
        # binary search over cum sizes
        # cum[i] = start index of chunk i
        lo, hi = 0, len(self._chunk_sizes) - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            start = self._cum_sizes[mid]
            end = start + self._chunk_sizes[mid]
            if frame_idx < start:
                hi = mid - 1
            elif frame_idx >= end:
                lo = mid + 1
            else:
                return mid, frame_idx - start
        raise IndexError(frame_idx)

    def get_frames(self, indices: List[int]) -> torch.Tensor:
        """Return frames stacked [T,3,H,W] as float in [0,1]."""
        # Group by chunk
        by_chunk: Dict[int, List[Tuple[int, int]]] = {}
        for pos, gidx in enumerate(indices):
            c_idx, off = self._global_to_chunk(gidx)
            by_chunk.setdefault(c_idx, []).append((pos, off))
        C, H, W = self._shape
        out = torch.empty(len(indices), C, H, W, dtype=torch.float32)
        for c_idx, lst in by_chunk.items():
            ck = self._load_chunk(c_idx)  # [n,3,H,W] uint8
            for pos, off in lst:
                out[pos] = ck[off].float() / 255.0
        return out


# ---------------------------------------------------------------------------
# Sliding windows (unchanged logic)
# ---------------------------------------------------------------------------

def create_sliding_windows(num_frames: int, kernel_size: int, stride: int, dilation: int) -> List[Tuple[List[int], int]]:
    if kernel_size % 2 == 0:
        raise ValueError("Kernel size must be odd")
    windows = []
    half = kernel_size // 2
    start_focal = half * dilation
    focal = start_focal
    while focal + half * dilation < num_frames:
        idxs = []
        for i in range(kernel_size):
            offset = (i - half) * dilation
            fi = focal + offset
            if 0 <= fi < num_frames:
                idxs.append(fi)
        if len(idxs) == kernel_size:
            windows.append((idxs, focal))
        focal += stride
    return windows


# ---------------------------------------------------------------------------
# Model loading & forward (copied / lightly adapted)
# ---------------------------------------------------------------------------

def load_vggt_models(device: torch.device):
    """Load VGGT, DINOv2 and tracker models onto a specific device.

    Passing the device explicitly ensures multi-process multi-GPU jobs pin
    allocations to the intended GPU instead of defaulting to cuda:0.
    """
    if torch.cuda.is_available():
        torch.cuda.set_device(device)  # hard pin this process to the device
    vggt_model = VGGT()
    _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
    vggt_model.load_state_dict(
        torch.hub.load_state_dict_from_url(_URL, map_location=device)
    )
    vggt_model.eval().to(device)
    dinov2_model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14_reg")
    dinov2_model.eval().to(device)
    vggsfm_tracker_model = build_vggsfm_tracker().to(device)
    try:  # optional compile
        vggsfm_tracker_model = torch.compile(vggsfm_tracker_model)  # type: ignore
    except Exception:
        pass
    return vggt_model, dinov2_model, vggsfm_tracker_model


def _fwd_vggt(model, images):
    aggregated_tokens_list, ps_idx = model.aggregator(images)
    pose_enc = model.camera_head(aggregated_tokens_list)[-1]
    extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])
    depth_map, depth_conf = model.depth_head(aggregated_tokens_list, images, ps_idx)
    extrinsic, intrinsic, depth_map, depth_conf = [x.detach().cpu().numpy() for x in (extrinsic, intrinsic, depth_map, depth_conf)]
    points_3d = []
    for i in range(depth_map.shape[0]):
        points_3d.append(
            unproject_depth_map_to_point_map(depth_map[i], extrinsic[i] if len(extrinsic.shape) > 2 else extrinsic, intrinsic[i] if len(intrinsic.shape) > 2 else intrinsic)
        )
    points_3d = np.stack(points_3d)
    return extrinsic, intrinsic, depth_map, depth_conf, points_3d


def run_VGGT(model, images, dtype, resolution=518):
    assert images.ndim == 5 and images.shape[2] == 3
    B, N, C, H, W = images.shape
    flat = images.view(B * N, C, H, W)
    flat = F.interpolate(flat, size=(resolution, resolution), mode="bilinear", align_corners=False)
    images_res = flat.view(B, N, C, resolution, resolution)
    with torch.no_grad():
        with torch.amp.autocast("cuda", dtype=dtype):
            extrinsic, intrinsic, depth_map, depth_conf, points_3d = _fwd_vggt(model, images_res)
    return extrinsic, intrinsic, depth_map, depth_conf, points_3d


def square_and_resize(images: torch.Tensor, load_res: int) -> torch.Tensor:
    sq = []
    for img in images:  # [C,H,W]
        c, h, w = img.shape
        if h >= w:
            new_h, new_w = load_res, max(1, int(w * load_res / h))
        else:
            new_w, new_h = load_res, max(1, int(h * load_res / w))
        new_h = (new_h // 14) * 14
        new_w = (new_w // 14) * 14
        img_r = F.interpolate(img.unsqueeze(0), size=(new_h, new_w), mode="bilinear", align_corners=False).squeeze(0)
        pad_t = (load_res - new_h) // 2
        pad_b = load_res - new_h - pad_t
        pad_l = (load_res - new_w) // 2
        pad_r = load_res - new_w - pad_l
        img_p = F.pad(img_r, (pad_l, pad_r, pad_t, pad_b), value=1.0)
        sq.append(img_p)
    return torch.stack(sq)


# ---------------------------------------------------------------------------
# Processing windows (simplified to fit batch mode cleanly)
# ---------------------------------------------------------------------------
@torch.no_grad()
def process_window_batch(vggt_model, dinov2_model, vggsfm_tracker_model, frames_batch: torch.Tensor, focal_global_idxs: List[int], window_indices: List[List[int]], args):
    """Process a batch of sliding windows.

    frames_batch: [B,T,3,H,W] float32 in [0,1]
    window_indices: list of per-window global frame indices (same length B)
    focal_global_idxs: list of global focal frame indices (length B)
    Returns lists of tensors (length B).
    """
    device = next(vggt_model.parameters()).device
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    B, T, C, H, W = frames_batch.shape
    images = frames_batch.to(device)
    extrinsic, intrinsic, depth_map, depth_conf, points_3d = run_VGGT(vggt_model, images, dtype, 518)
    # Convert arrays to torch (batch-first: [B,N,...])
    extrinsic_t = torch.from_numpy(extrinsic).float()  # [B,N,3,4]
    intrinsic_t = torch.from_numpy(intrinsic).float()  # [B,N,3,3]
    depth_t = torch.from_numpy(depth_map).float()      # [B,N,R,R]

    out_rgbs: List[torch.Tensor] = []      # each [3,H,W]
    out_depths: List[torch.Tensor] = []    # each [R,R]
    out_cameras: List[torch.Tensor] = []   # each [21]
    out_pcls: List[torch.Tensor] = []      # each [N_pts,6]

    for b in range(B):
        # focal index within window
        focal_idx_local = window_indices[b].index(focal_global_idxs[b])
        focal_rgb = images[b, focal_idx_local]  # [3,H,W]
        focal_depth = depth_t[b, focal_idx_local]  # [R,R]
        fe = extrinsic_t[b, focal_idx_local].reshape(-1)  # 12
        fi = intrinsic_t[b, focal_idx_local].reshape(-1)  # 9
        focal_cam = torch.cat([fe, fi], dim=0)            # 21
        point_cloud = torch.zeros(0, 6)  # Placeholder (retain original structure)
        out_rgbs.append(focal_rgb.cpu())
        out_depths.append(focal_depth.cpu())
        out_cameras.append(focal_cam.cpu())
        out_pcls.append(point_cloud)
    return out_rgbs, out_depths, out_cameras, out_pcls


# ---------------------------------------------------------------------------
# Saving samples (same interface)
# ---------------------------------------------------------------------------

def save_data_sample(output_dir: str, gpu_rank: int, sample_idx: int, files_per_subdir: int, focal_rgb: torch.Tensor, focal_depth: torch.Tensor, focal_camera: torch.Tensor, point_cloud: torch.Tensor):
    subdir_idx = sample_idx // files_per_subdir
    sample_in_subdir = sample_idx % files_per_subdir
    rank_dir = Path(output_dir) / str(gpu_rank)
    out_dir = rank_dir / f"{subdir_idx:06d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    base = f"{sample_in_subdir:04d}"
    rgb_path = out_dir / f"{base}.jpg"
    depth_path = out_dir / f"{base}.depth.jpg"
    camera_path = out_dir / f"{base}.camera.pt"
    points_path = out_dir / f"{base}.points.pt"
    # RGB save
    rgb_img = (focal_rgb.permute(1, 2, 0).clamp(0, 1).numpy() * 255).astype(np.uint8)
    Image.fromarray(rgb_img).save(rgb_path, quality=95)
    # Depth save (normalize)
    depth_np = focal_depth.numpy()
    if depth_np.max() > 0:
        dn = (depth_np / depth_np.max() * 255).astype(np.uint8)
    else:
        dn = np.zeros_like(depth_np, dtype=np.uint8)
    Image.fromarray(dn, mode="L").save(depth_path)
    torch.save(focal_camera, camera_path)
    torch.save(point_cloud, points_path)


# ---------------------------------------------------------------------------
# Video discovery & worker
# ---------------------------------------------------------------------------

def discover_chunked_videos(root: str, cache_size: int, persist_index: bool, allow_skip_corrupt: bool = False) -> List[ChunkedVideo]:
    """Discover all directories that contain VGGT chunk tensors (*_rgb.pt).

    Handles several common layouts produced by the extractor:
      1. <video_dir>/splits/*.pt
      2. <root>/<any_depth>/splits/*.pt
      3. Root itself is a splits directory containing *.pt
      4. Directories that contain *_rgb.pt even if not named 'splits'

    We purposefully avoid relying solely on Path.rglob('splits') because
    users reported missing directories; instead we scan for the pattern
    *_rgb.pt and register the parent directories.
    """
    root_path = Path(root).expanduser().resolve()
    if not root_path.exists():
        print(f"discover_chunked_videos: root does not exist: {root_path}")
        return []

    candidate_dirs: set[Path] = set()

    # Fast pass: any directory named 'splits'
    try:
        for p in root_path.rglob('splits'):
            if p.is_dir():
                candidate_dirs.add(p)
    except Exception:
        pass

    # Pass scanning for *_rgb.pt files (robust, may be slower on huge trees)
    pt_pattern = '*_rgb.pt'
    for pt_file in root_path.rglob(pt_pattern):
        if pt_file.is_file():
            candidate_dirs.add(pt_file.parent)

    # If root itself directly holds chunk files
    if any(root_path.glob(pt_pattern)):
        candidate_dirs.add(root_path)

    # Filter: keep only dirs that actually have >=1 *_rgb.pt
    filtered_dirs: List[Path] = []
    for d in candidate_dirs:
        if any(d.glob(pt_pattern)):
            filtered_dirs.append(d)

    if not filtered_dirs:
        print(f"discover_chunked_videos: No chunk dirs with pattern {pt_pattern} under {root_path}")
        return []

    # Sort for determinism
    filtered_dirs.sort()
    print(f"Discovered {len(filtered_dirs)} chunk directories. Initializing ChunkedVideo objects...")

    vids: List[ChunkedVideo] = []
    for d in filtered_dirs:
        try:
            vids.append(ChunkedVideo(d, cache_size=cache_size, persist_index=persist_index, allow_skip_corrupt=allow_skip_corrupt))
        except Exception as e:
            print(f"Skipping {d}: {e}")

    print(f"Validated {len(vids)} chunked video directories (kept). Skipped {len(filtered_dirs) - len(vids)}.")
    # Show a brief sample for debugging
    for show in vids[:5]:
        print(f"  sample dir: {show.splits_dir} (frames={show.num_frames})")
    if len(vids) > 5:
        print(f"  ... {len(vids)-5} more")
    return vids


def process_videos_worker(videos: List[ChunkedVideo], args, gpu_rank: int, device: torch.device):
    if torch.cuda.is_available():
        torch.cuda.set_device(device)
    print(f"GPU {gpu_rank} (device={device}): Loading models...")
    vggt_model, dinov2_model, vggsfm_tracker_model = load_vggt_models(device)
    sample_idx = 0
    for vid in tqdm(videos, desc=f"Rank {gpu_rank} videos"):
        try:
            num_frames = vid.num_frames
            windows = create_sliding_windows(num_frames, args.kernel_size, args.stride, args.dilation)
            if not windows:
                continue
            batches = [windows[i:i + args.batch_size] for i in range(0, len(windows), args.batch_size)]
            for batch in tqdm(batches, leave=False, desc="window batches"):
                window_indices = [w[0] for w in batch]
                focal_globals = [w[1] for w in batch]
                frames_list = [vid.get_frames(idxs) for idxs in window_indices]  # each [T,3,H,W]
                frames_batch = torch.stack(frames_list, dim=0)  # [B,T,3,H,W]
                frgbs, fdepths, fcams, pcls = process_window_batch(
                    vggt_model, dinov2_model, vggsfm_tracker_model,
                    frames_batch, focal_globals, window_indices, args
                )
                for i in range(len(frgbs)):
                    save_data_sample(
                        args.output_dir, gpu_rank, sample_idx, args.files_per_subdir,
                        frgbs[i], fdepths[i], fcams[i], pcls[i]
                    )
                    sample_idx += 1
        except Exception as e:
            print(f"Rank {gpu_rank}: Error processing video at {vid.splits_dir}: {e}")
            print(traceback.format_exc())
    print(f"Rank {gpu_rank}: Completed {sample_idx} samples")
    return sample_idx


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    if args.kernel_size % 2 == 0:
        print("Error: kernel_size must be odd"); sys.exit(1)
    os.makedirs(args.output_dir, exist_ok=True)
    videos_all = discover_chunked_videos(
        args.video_dir,
        args.chunk_cache_size,
        args.index_meta,
        allow_skip_corrupt=args.allow_skip_corrupt,
    )
    if not videos_all:
        print(f"No chunked videos (splits/) found under {args.video_dir}")
        sys.exit(1)
    print(f"Discovered {len(videos_all)} chunked videos")
    num_gpus = args.num_gpus if args.num_gpus is not None else torch.cuda.device_count()
    world_size = args.world_size
    if args.rank < 0 or args.rank >= world_size:
        print(f"Invalid rank {args.rank}"); sys.exit(1)
    # Validate consistent frame shapes across all videos
    shapes = [v.frame_shape for v in videos_all]
    ref_shape = shapes[0]
    bad = [ (i,str(v.splits_dir),sh) for i,(v,sh) in enumerate(zip(videos_all,shapes)) if sh != ref_shape ]
    if bad:
        print("ERROR: Inconsistent frame shapes detected across chunked videos:")
        for i,path,sh in bad[:20]:
            print(f"  idx={i} path={path} shape={sh} (expected {ref_shape})")
        if len(bad) > 20:
            print(f"  ... {len(bad)-20} more mismatches")
        print("Aborting. Ensure all extracted chunks use identical resolution.")
        sys.exit(1)
    else:
        if args.dry_run:
            print(f"Frame shape validation OK: all {len(videos_all)} videos share shape {ref_shape}")
    # Dry run: summarize and exit before any model loading
    if args.dry_run:
        # Build stats per video
        per_video_stats = []  # (path, frames, windows)
        for vid in videos_all:
            nf = vid.num_frames
            win = len(create_sliding_windows(nf, args.kernel_size, args.stride, args.dilation))
            per_video_stats.append((str(vid.splits_dir), nf, win))
        # Assign videos to ranks with same modulo strategy as training phase
        rank_assignments: List[List[Tuple[str,int,int]]] = [[] for _ in range(world_size)]
        for i, stat in enumerate(per_video_stats):
            rank_assignments[i % world_size].append(stat)
        # Global summary only on rank 0
        if args.rank == 0:
            total_frames = sum(v[1] for v in per_video_stats)
            total_windows = sum(v[2] for v in per_video_stats)
            print("\n=== DRY RUN GLOBAL SUMMARY (rank 0) ===")
            print(f"Videos: {len(per_video_stats)} | Total frames: {total_frames} | Total windows: {total_windows}")
            rank_window_counts = [sum(v[2] for v in lst) for lst in rank_assignments]
            print("Windows per rank: " + ", ".join(str(x) for x in rank_window_counts))
        # Per-rank detailed listing (only this rank prints its allocation)
        my_stats = rank_assignments[args.rank]
        print(f"\n=== DRY RUN RANK {args.rank} SUMMARY ===")
        print(f"Assigned videos: {len(my_stats)}")
        print(f"Frames (sum): {sum(v[1] for v in my_stats)} | Windows (sum): {sum(v[2] for v in my_stats)}")
        for path, nf, win in my_stats:
            print(f"video: {path} | frames: {nf} | windows: {win}")
        print("Dry run complete (rank view). Exiting without model initialization.")
        return
    # Split videos deterministically across world_size
    per_rank: List[List[ChunkedVideo]] = [[] for _ in range(world_size)]
    for i, v in enumerate(videos_all):
        per_rank[i % world_size].append(v)
    my_videos = per_rank[args.rank]
    print(f"Rank {args.rank}: {len(my_videos)} videos assigned")
    # Determine this process's GPU (local index) and device
    if torch.cuda.is_available():
        num_visible = num_gpus if num_gpus else torch.cuda.device_count()
        local_gpu = args.rank % num_visible
        device = torch.device(f"cuda:{local_gpu}")
    else:
        print("cuda failed")
        device = torch.device("cpu")
        local_gpu = -1
    print(f"Rank {args.rank}: assigned local GPU index {local_gpu} / device {device}")
    total_samples = process_videos_worker(my_videos, args, args.rank, device)
    print(f"Rank {args.rank}: DONE. Samples={total_samples}")
    print(f"Output dir: {args.output_dir}")

if __name__ == "__main__":
    start = time.time()
    main()
    print(f"Total time: {time.time()-start:.2f}s")

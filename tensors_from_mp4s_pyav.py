import argparse
import gc
import os
import tempfile
import json
import hashlib
import shutil
from typing import Tuple, Optional

import ray
import av
import torch
from tqdm import tqdm

# PyAV-based version of tensors_from_mp4s.py that avoids decord dependency.
# Functionality: Traverse root directory (or list of roots), find all .mp4 files,
# decode frames, chunk them, resize, and save RGB tensors per chunk.

def get_video_paths(root_dir):
    video_paths = []
    root_dirs = [root_dir] if isinstance(root_dir, str) else root_dir
    for dir in root_dirs:
        for root, _, files in os.walk(dir):
            for file in files:
                if file.endswith('.mp4'):
                    video_paths.append(os.path.join(root, file))
    return video_paths


def to_tensor(frames):
    # frames: list of np arrays [H,W,C] uint8
    frames = [torch.from_numpy(f) for f in frames]
    frames = torch.stack(frames, dim=0)  # [N,H,W,C]
    frames = frames.permute(0, 3, 1, 2)  # [N,C,H,W]
    # frames = torch.nn.functional.interpolate(frames, size=output_size, mode='bilinear', align_corners=False)
    frames = frames.to(torch.uint8).cpu()
    return frames


def _iter_video_frames_pyav(path, output_size):
    # Yields numpy arrays in RGB order
    # Using PyAV's native decoding. PyAV gives frames in planar format; convert to RGB ndarray.
    with av.open(path) as container:
        stream = container.streams.video[0]
        for frame in container.decode(stream):
            frame = frame.reformat(width=output_size[1], height=output_size[0])
            # Convert to RGB24 packed format then to ndarray
            img = frame.to_ndarray(format='rgb24')  # shape [H,W,3], uint8
            # print(img.shape, img.dtype)  # Debugging output
            yield img


def _tensor_sha256(t: torch.Tensor) -> str:
    # compute hash on CPU contiguous bytes
    arr = t.contiguous().numpy().tobytes()
    return hashlib.sha256(arr).hexdigest()


def _atomic_save_tensor(tensor: torch.Tensor, out_path: str, verify: bool, max_retries: int = 3):
    """Atomically save a tensor with optional verification & metadata.

    Writes to a temp file then moves into place to avoid partial writes.
    If verify=True, re-loads the tensor, compares shape/dtype/hash.
    A sidecar JSON (.meta) file stores basic metadata & hash for later quick checks.
    """
    directory = os.path.dirname(out_path)
    base = os.path.basename(out_path)
    meta_path = out_path + '.meta'
    for attempt in range(1, max_retries + 1):
        tmp_fd, tmp_path = tempfile.mkstemp(dir=directory, prefix=base + '.tmp.')
        os.close(tmp_fd)
        try:
            # Save tensor
            torch.save(tensor, tmp_path)
            # fsync to reduce risk of corruption on crash
            with open(tmp_path, 'rb') as f:
                os.fsync(f.fileno())

            if verify:
                # Reload and verify
                reloaded = torch.load(tmp_path, map_location='cpu', weights_only=True)
                if not isinstance(reloaded, torch.Tensor):
                    raise ValueError('Reloaded object is not a tensor')
                if reloaded.shape != tensor.shape or reloaded.dtype != tensor.dtype:
                    raise ValueError(f'Shape/dtype mismatch after save: {reloaded.shape}/{reloaded.dtype} vs {tensor.shape}/{tensor.dtype}')
                # Hash compare
                h1 = _tensor_sha256(tensor.cpu())
                h2 = _tensor_sha256(reloaded.cpu())
                if h1 != h2:
                    raise ValueError('Hash mismatch after save (possible corruption)')
                meta = {
                    'shape': list(tensor.shape),
                    'dtype': str(tensor.dtype),
                    'sha256': h1,
                    'bytes': int(tensor.numel() * tensor.element_size()),
                }
                with open(meta_path + '.tmp', 'w') as mf:
                    json.dump(meta, mf)
                os.replace(meta_path + '.tmp', meta_path)

            # Atomic move to final path
            os.replace(tmp_path, out_path)
            return
        except Exception as e:
            # Cleanup temp
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass
            if attempt == max_retries:
                raise RuntimeError(f'Failed to save {out_path} after {max_retries} attempts: {e}') from e
        finally:
            # Ensure tmp_fd closed (already closed) and partial meta removed
            pass


@ray.remote
def decode_video(path: str, split_dir: Optional[str], chunk_size: int, output_size, verify_save: bool = False):
    """Decode one video file into chunked RGB tensors.

    Parameters
    ----------
    path : str
        Source .mp4 path.
    split_dir : Optional[str]
        Destination directory to place chunk files ("splits" directory). If None,
        defaults to sibling 'splits' directory next to the video (legacy behavior).
    chunk_size : int
        Number of frames per saved tensor chunk.
    output_size : (int,int)
        (H,W) resize for decoded frames prior to saving.
    verify_save : bool
        If True, perform hash verification after each save.
    """
    if split_dir is None:
        split_dir = os.path.join(os.path.dirname(path), "splits")
    os.makedirs(split_dir, exist_ok=True)

    frames = []
    n_frames = 0
    split_ind = 0

    for img in _iter_video_frames_pyav(path, output_size):
        frames.append(img)
        n_frames += 1
        if n_frames >= chunk_size:
            chunk = to_tensor(frames)
            frames.clear()
            out_file = os.path.join(split_dir, f"{split_ind:08d}_rgb.pt")
            _atomic_save_tensor(chunk, out_file, verify=verify_save)
            n_frames = 0
            split_ind += 1
            del chunk
            gc.collect()
            print(".", end='')

    if frames:
        chunk = to_tensor(frames)
        out_file = os.path.join(split_dir, f"{split_ind:08d}_rgb.pt")
        _atomic_save_tensor(chunk, out_file, verify=verify_save)
        del chunk
        gc.collect()

    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, required=True, help='Root directory containing videos')
    parser.add_argument('--chunk_size', type=int, default=2000, help='Number of frames per chunk')
    parser.add_argument('--output_size', type=int, nargs=2, default=[360, 640], help='Output size as two integers: height width')
    parser.add_argument('--force_overwrite', action='store_true', help='Force overwrite existing rgb tensors')
    parser.add_argument('--num_cpus', type=int, default=80, help='Number of CPUs to use for Ray')
    parser.add_argument('--output-root', type=str, default=None, help='Optional root directory to place all splits/ subdirectories (mirrors relative paths under --root_dir)')
    # parser.add_argument('--verify-save', action='store_true', help='After writing each chunk, reload & hash-verify (slower, ensures integrity)')
    args = parser.parse_args()

    video_paths = get_video_paths(args.root_dir)

    ray.init(num_cpus=args.num_cpus)

    paths_to_process = []  # list of (video_path, split_dir)
    root_dir_abs = os.path.abspath(args.root_dir)
    for path in video_paths:
        # Determine destination splits directory (supports optional relocation)
        if args.output_root:
            try:
                rel_parent = os.path.relpath(os.path.dirname(path), root_dir_abs)
            except ValueError:
                # If path not under root_dir (should not happen), fallback to basename
                rel_parent = os.path.basename(os.path.dirname(path))
            split_dir = os.path.join(os.path.abspath(args.output_root), rel_parent, 'splits')
        else:
            split_dir = os.path.join(os.path.dirname(path), 'splits')

        if os.path.exists(split_dir) and not args.force_overwrite:
            # Heuristic: consider processed if at least one *_rgb.pt exists
            if any(f.endswith('_rgb.pt') for f in os.listdir(split_dir)):
                print(f"Skipping {path} - already processed at {split_dir}")
                continue
        elif os.path.exists(split_dir) and args.force_overwrite:
            for file in os.listdir(split_dir):
                if file.endswith('_rgb.pt'):
                    try:
                        os.remove(os.path.join(split_dir, file))
                    except OSError:
                        pass
        paths_to_process.append((path, split_dir))

    if not paths_to_process:
        print("No videos to process")
        return

    # Launch decoding tasks
    futures = [decode_video.remote(path, split_dir, args.chunk_size, tuple(args.output_size), True)
               for (path, split_dir) in paths_to_process]

    with tqdm(total=len(futures), desc="Processing videos") as pbar:
        while futures:
            done, futures = ray.wait(futures)
            completed_paths = ray.get(done)
            for p in completed_paths:
                print(f"Completed processing {p}")
            pbar.update(len(done))

    print("All videos processed!")


if __name__ == '__main__':
    main()

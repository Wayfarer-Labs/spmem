import argparse
import gc
import os

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
            yield img


@ray.remote
def decode_video(path, chunk_size, output_size):
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
            torch.save(chunk, os.path.join(split_dir, f"{split_ind:08d}_rgb.pt"))
            n_frames = 0
            split_ind += 1
            del chunk
            gc.collect()

    if frames:
        chunk = to_tensor(frames)
        torch.save(chunk, os.path.join(split_dir, f"{split_ind:08d}_rgb.pt"))
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
    args = parser.parse_args()

    video_paths = get_video_paths(args.root_dir)

    ray.init(num_cpus=args.num_cpus)

    paths_to_process = []
    for path in video_paths:
        split_dir = os.path.join(os.path.dirname(path), "splits")
        if os.path.exists(split_dir) and not args.force_overwrite:
            print(f"Skipping {path} - already processed")
            continue
        elif os.path.exists(split_dir) and args.force_overwrite:
            for file in os.listdir(split_dir):
                if file.endswith('_rgb.pt'):
                    os.remove(os.path.join(split_dir, file))
        paths_to_process.append(path)

    if not paths_to_process:
        print("No videos to process")
        return

    futures = [decode_video.remote(path, args.chunk_size, tuple(args.output_size)) for path in paths_to_process]

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

uv run streaming_vggt.py --bucket cod-yt-playlist --ext mp4 --frame-batch-size 50 --target-bucket cod-yt-playlist-spmem-tensors

uv run batch_colmap_demo.py --conf_thres_value 1.2 --scene_dir cod-output/ --max_frames 1000

(base) sky@sky-b860-root-1f34085b-head:~/sky_workdir/spmem$ scp -r ^Cky-b860-root:/home/sky/sky_workdir/spmem/testdata/images
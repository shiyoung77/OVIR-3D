import argparse
import subprocess
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Queue
from pathlib import Path
from itertools import repeat
from time import perf_counter

import torch


global available_devices


def init_process(_available_devices):
    global available_devices
    available_devices = _available_devices


def proposed_fusion(dataset, video, idx, args):
    device = available_devices.get()
    print(f"Processing {video = }, {idx = }, {device = }")

    output_file = f"proposed_fusion_detic_iou-{args.iou_thresh:.2f}_recall-{args.recall_thresh:.2f}_" \
                  f"feature-{args.feature_similarity_thresh:.2f}_interval-{args.interval}.pkl"
    # output_file = f"proposed_fusion_detic_iou-{args.iou_thresh:.2f}_recall-{args.recall_thresh:.2f}_" \
    #               f"feature-{args.feature_similarity_thresh:.2f}_interval-{args.interval}_with-kmeans.pkl"
    command = f"""
        OMP_NUM_THREADS=6 python proposed_fusion.py \
            --dataset "{dataset}" \
            --video "{video}" \
            --detic_exp "{args.detic_exp}" \
            --iou_thresh "{args.iou_thresh}" \
            --recall_thresh "{args.recall_thresh}" \
            --feature_similarity_thresh "{args.feature_similarity_thresh}" \
            --depth_thresh "{args.depth_thresh}" \
            --interval "{args.interval}" \
            --visibility_thresh "{args.visibility_thresh}" \
            --size_thresh "{args.size_thresh}" \
            --device "cuda:{device}" \
            --output_file "{output_file}" \
            --stride "{args.stride}" \
            --vocab_feature_file "src/scannet200.npy" \
            --no-sam \
            --tqdm \
            --no-vis \
    """
    subprocess.run(command, shell=True)
    available_devices.put(device)
    print(f"Finish processing {video = }, {idx = }, {device = }")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="~/t7/ScanNet/aligned_scans")
    # parser.add_argument("--dataset", type=str, default="~/t7/ycb_video")
    parser.add_argument("--detic_exp", default="imagenet21k-0.3")
    parser.add_argument("--video", type=str, default="")
    parser.add_argument("--iou_thresh", type=float, default=0.25)
    parser.add_argument("--recall_thresh", type=float, default=0.5)
    parser.add_argument("--depth_thresh", type=float, default=0.1)
    parser.add_argument("--interval", type=int, default=300)
    parser.add_argument("--visibility_thresh", type=float, default=0.2)
    parser.add_argument("--feature_similarity_thresh", type=float, default=0.75)
    parser.add_argument("--size_thresh", type=int, default=50)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--num_gpus", type=int, default=-1, help="Number of GPUs to use, -1 for all")
    args = parser.parse_args()

    if args.num_gpus == -1:
        max_workers = torch.cuda.device_count()
    else:
        max_workers = min(torch.cuda.device_count(), args.num_gpus)

    dataset = Path(args.dataset).expanduser()
    if "ycb_video" in args.dataset:
        videos = [f"{i:04d}" for i in range(48, 60)]
    else:
        videos = [i.name for i in sorted(dataset.iterdir())]
    if args.video:
        videos = [args.video]

    print(f"{dataset = }")
    print(f"{len(videos) = }")
    print(f"{max_workers = }")

    _available_devices = Queue()
    for i in range(max_workers):
        _available_devices.put(i)
    init_args = (_available_devices,)

    tic = perf_counter()
    with ProcessPoolExecutor(max_workers=max_workers, initializer=init_process, initargs=init_args) as executor:
        executor.map(proposed_fusion, repeat(dataset), videos, range(len(videos)), repeat(args))
    print(f"Process {len(videos)} takes {perf_counter() - tic}s")


if __name__ == "__main__":
    main()

# This script is used to preprocess the ScanNet dataset.

# You need to first download ScanNet dataset from "https://kaldir.vc.in.tum.de/scannet_benchmark/documentation"
# and then extract raw RGB-D frames to "scans" folder using the provided script at
# https://github.com/ScanNet/ScanNet/tree/master/SensReader/python:
# This will generate the following structure:
# {ScanNet_path}/
#     scans/
#         {video_name}/
#             ...

# Put this script at {ScanNet_path}/scannet_preprocess.py, change the path of 'scannet_git_repo' and run it.

# This script will generate "aligned_scans" folder, which contains the following structure:
# {ScanNet_path}/
#     aligned_scans/
#         {video_name}/
#             color/
#                 0000-color.jpg
#                 0001-color.jpg
#                 ...
#             depth/
#                 0000-depth.png
#                 0001-depth.png
#                 ...
#             poses/
#                 0000-pose.txt
#                 0001-pose.txt
#                 ...
#             config.json
#             {video_name}.txt
#             {video_name}_clean_2.ply

import os
import shutil
import json

import numpy as np
import cv2
from tqdm import tqdm


def main():
    scannet_git_repo = os.path.expanduser("~/software/ScanNet/")  # change this path

    input_folder = "scans"
    output_folder = "aligned_scans"
    mode = "val"

    scans = []
    with open(os.path.join(scannet_git_repo, f"/Tasks/Benchmark/scannetv2_{mode}.txt"), 'r') as fp:
        line = fp.readline().strip()
        while line:
            if not line.endswith("00"):
                scans.append(line)
            line = fp.readline().strip()

    for i, scan in enumerate(scans):
        print(f"[{i + 1}/{len(scans)}]: processing {scan}")

        # =============================== copy pose files ===============================
        print("Copying pose files...")
        pose_folder = os.path.join(input_folder, scan, 'pose')
        output_pose_folder = os.path.join(output_folder, scan, 'poses')
        os.makedirs(output_pose_folder, exist_ok=True)

        pose_files = sorted(os.listdir(pose_folder), key=lambda x: int(x[:-4]))
        for i, pose_file in enumerate(tqdm(pose_files)):
            pose_path = os.path.join(pose_folder, pose_file)
            output_pose_path = os.path.join(output_pose_folder, f"{i:04d}-pose.txt")
            shutil.copy(pose_path, output_pose_path)

        # =============================== copy color images ===============================
        print("Copying resized color images...")
        color_folder = os.path.join(input_folder, scan, 'color')
        output_im_folder = os.path.join(output_folder, scan, 'color')
        os.makedirs(output_im_folder, exist_ok=True)

        color_im_files = sorted(os.listdir(color_folder), key=lambda x: int(x[:-4]))
        for i, color_im_file in enumerate(tqdm(color_im_files)):
            color_im_path = os.path.join(color_folder, color_im_file)
            img = cv2.imread(color_im_path)
            resized_img = cv2.resize(img, (640, 480), interpolation=cv2.INTER_LINEAR)

            output_im_path = os.path.join(output_im_folder, f"{i:04d}-color.jpg")
            cv2.imwrite(output_im_path, resized_img)

        # =============================== copy depth images ===============================
        print("Copying depth images...")
        depth_folder = os.path.join(input_folder, scan, 'depth')

        output_im_folder = os.path.join(output_folder, scan, 'depth')
        os.makedirs(output_im_folder, exist_ok=True)

        depth_im_files = sorted(os.listdir(depth_folder), key=lambda x: int(x[:-4]))
        for i, depth_im_file in enumerate(tqdm(depth_im_files)):
            depth_im_path = os.path.join(depth_folder, depth_im_file)
            depth_output_path = os.path.join(output_im_folder, f"{i:04d}-depth.png")
            shutil.copy(depth_im_path, depth_output_path)

        # =============================== convert config file ===============================
        print("Generating config files...")
        cam_info_path = os.path.join(input_folder, scan, f"{scan}.txt")
        lines = []
        with open(cam_info_path, 'r') as fp:
            line = fp.readline().strip()
            while line:
                lines.append(line)
                line = fp.readline().strip()
        cam_intr = np.eye(3)
        indices = [(0, 0), (1, 1), (0, 2), (1, 2)]
        prefixes = ["fx_depth", "fy_depth", "mx_depth", "my_depth"]
        for index, prefix in zip(indices, prefixes):
            found = False
            for line in lines:
                if line.startswith(prefix):
                    cam_intr[index] = line.split('=')[1][1:]
                    found = True
                    break
            assert found, f"{scan}-{prefix}"
        print(cam_intr)

        config = dict()
        config['id'] = scan
        config['cam_intr'] = cam_intr.tolist()
        config['depth_scale'] = 1000

        config_path = os.path.join(output_folder, scan, 'config.json')
        with open(config_path, 'w') as fp:
            json.dump(config, fp, indent=4)

        # =============================== copy reconstruction and label file ===============================
        files = [f"{scan}_vh_clean_2.labels.ply", f"{scan}_vh_clean_2.ply",
                 f"{scan}_vh_clean_2.0.010000.segs.json", f"{scan}.txt"]
        for file in files:
            src_path = os.path.join(input_folder, scan, file)
            tgt_path = os.path.join(output_folder, scan, file)
            shutil.copy(src_path, tgt_path)


if __name__ == "__main__":
    main()

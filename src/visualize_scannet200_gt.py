import copy
import sys
import importlib
import argparse
from pathlib import Path
from functools import partial

import git
import numpy as np
import open3d as o3d

from utils import vis_pcd


def vis_gt_instances(scene_pcd, instance_ids, cat_id_to_label):
    print("Press J to show next instance, K to show previous instance")
    pcd_vis = copy.deepcopy(scene_pcd)
    ptr = [0]
    unique_instance_ids = np.unique(instance_ids)

    def show_next_instance_callback(visualizer, action, mod, pressed_key):
        if action == 0:  # key is pressed
            if pressed_key == 'J':
                ptr[0] += 1
            elif pressed_key == 'K':
                ptr[0] -= 1
            instance_id = unique_instance_ids[ptr[0]]
            cat_id = instance_id // 1000
            label = cat_id_to_label[cat_id]
            print(f"{cat_id = }, {label = }")
            color_vis = np.asarray(pcd_vis.colors)
            color_vis[:] = np.asarray(scene_pcd.colors)
            color_vis[instance_ids == instance_id] = (1, 1, 0)
        return True

    vis = o3d.visualization.VisualizerWithKeyCallback()
    for key in ['J', 'K']:
        vis.register_key_action_callback(ord(key), partial(show_next_instance_callback, pressed_key=key))
    vis.create_window()
    vis.add_geometry(pcd_vis)
    vis.run()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, default="~/t7/ScanNet")
    parser.add_argument("-v", "--video", type=str, default="scene0208_00")  # 0011, 0591, 0645
    parser.add_argument("--mode", type=str, default="validation")
    args = parser.parse_args()

    git_repo = Path(git.Repo(search_parent_directories=True).working_tree_dir)
    sys.path.append(str(git_repo))

    scannet200 = importlib.import_module("scannet_related_scripts.scannet200_constants")
    SCANNET_COLOR_MAP_200 = scannet200.SCANNET_COLOR_MAP_200
    CLASS_LABELS = scannet200.CLASS_LABELS_200
    VALID_CLASS_IDS = scannet200.VALID_CLASS_IDS_200
    ID_TO_LABEL = {}
    LABEL_TO_ID = {}
    for i in range(len(VALID_CLASS_IDS)):
        LABEL_TO_ID[CLASS_LABELS[i]] = VALID_CLASS_IDS[i]
        ID_TO_LABEL[VALID_CLASS_IDS[i]] = CLASS_LABELS[i]

    dataset = Path(args.dataset).expanduser()
    video_path = dataset / "aligned_scans" / args.video
    scene_pcd_path = video_path / f"{args.video}_vh_clean_2.ply"
    scene_pcd = o3d.io.read_point_cloud(str(scene_pcd_path))
    print(f"{len(scene_pcd.points) = }")
    vis_pcd(scene_pcd)

    gt_path = git_repo / "scannet_related_scripts" / "scannet200_instance_gt" / args.mode / f"{args.video}.txt"
    instance_ids = np.array(gt_path.read_text().splitlines()).astype(int)
    num_pts = len(instance_ids)
    instance_labels = instance_ids // 1000
    unique_instance_ids = np.unique(instance_ids)
    unique_categories = np.unique(instance_labels)
    unique_labels = [ID_TO_LABEL[i] for i in unique_categories if i in ID_TO_LABEL]

    print(f"{unique_labels = }")
    print(f"{len(unique_instance_ids) = }")
    print(f"{len(unique_categories) = }")

    instance_to_color = dict()
    for instance_id in unique_instance_ids:
        label = instance_id // 1000
        if label not in SCANNET_COLOR_MAP_200 or label in [0, 1, 3]:
            label_rgb = np.array([0, 0, 0])
        else:
            label_rgb = np.random.random((3,))
        instance_to_color[instance_id] = label_rgb

    pcd_vis = copy.deepcopy(scene_pcd)
    colors = np.zeros((num_pts, 3))
    for instance_id in unique_instance_ids:
        colors[instance_ids == instance_id] = instance_to_color[instance_id]
    pcd_vis.colors = o3d.utility.Vector3dVector(colors)
    vis_pcd(pcd_vis)

    vis_gt_instances(scene_pcd, instance_ids, ID_TO_LABEL)


if __name__ == "__main__":
    main()

import sys
import copy
import json
import argparse
from pathlib import Path
from functools import partial

import git
import numpy as np
import open3d as o3d

from utils import vis_pcd
from vocabs import vocabs


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
            cat_id = instance_id  # for ycb video dataset, each scene can only have one instance of each category
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
    parser.add_argument("-d", "--dataset", type=str, default="~/t7/ycb_video")
    parser.add_argument("-v", "--video", type=str, default="0048")
    args = parser.parse_args()

    git_repo = Path(git.Repo(search_parent_directories=True).working_tree_dir)
    sys.path.append(str(git_repo))

    dataset = Path(args.dataset).expanduser()
    video_path = dataset / args.video
    scene_pcd_path = video_path / "scan-0.005.pcd"
    scene_pcd = o3d.io.read_point_cloud(str(scene_pcd_path))
    num_pts = len(scene_pcd.points)
    print(f"{num_pts = }")
    vis_pcd(scene_pcd)

    gt_path = video_path / "annotations" / "instance_annotation.json"
    with open(gt_path, 'r') as f:
        gt = json.load(f)

    print(f"{len(gt) = }")
    instance_ids = np.zeros(num_pts, dtype=np.int32)
    for instance in gt:
        pt_indices = instance['indices']
        instance_ids[pt_indices] = instance['label'] + 1
    unique_instance_ids = np.unique(instance_ids)

    instance_to_color = dict()
    for instance_id in unique_instance_ids:
        label_rgb = np.random.random((3,))
        instance_to_color[instance_id] = label_rgb

    pcd_vis = copy.deepcopy(scene_pcd)
    colors = np.zeros((num_pts, 3))
    for instance_id in unique_instance_ids:
        if instance_id == 0:
            continue
        colors[instance_ids == instance_id] = instance_to_color[instance_id]
    pcd_vis.colors = o3d.utility.Vector3dVector(colors)
    vis_pcd(pcd_vis)

    ID_TO_LABEL = {i: vocab for i, vocab in enumerate(vocabs['ycb_video'], start=1)}
    vis_gt_instances(scene_pcd, instance_ids, ID_TO_LABEL)


if __name__ == "__main__":
    main()

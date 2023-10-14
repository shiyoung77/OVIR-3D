import sys
import glob
import importlib
import copy
import pickle
import argparse
from functools import partial
from pathlib import Path

import git
import numpy as np
import open3d as o3d
import clip
import torch
import torch.nn.functional as F
from scipy.spatial import distance

from utils import vis_pcd
from proposed_fusion import visualize_scene_graph


def get_clip_feature(clip_model, text_label, normalize=True, prompt_fn=lambda x: f"a {x}", device="cpu"):
    print("computing text features...")
    if isinstance(text_label, str):
        text_inputs = clip.tokenize(prompt_fn(text_label)).to(device)
    else:
        text_inputs = torch.cat([clip.tokenize(prompt_fn(c)) for c in text_label]).to(device)

    # in case the vocab is too large
    chunk_size = 100
    chunks = torch.split(text_inputs, chunk_size, dim=0)
    text_features = []
    for i, chunk in enumerate(chunks):
        chunk_feature = clip_model.encode_text(chunk).detach()
        text_features.append(chunk_feature)

    text_features = torch.cat(text_features, dim=0)
    if normalize:
        text_features = F.normalize(text_features, dim=-1).detach()
    return text_features


def visualize_instances(scene_pcd, scene_graph, vocabs=None, vocab_features=None, device="cuda:0"):
    print("Press J to show next instance, K to show previous instance")
    if vocab_features is not None:
        if not torch.is_tensor(vocab_features):
            vocab_features = torch.from_numpy(vocab_features).to(device)

    pcd_vis = copy.deepcopy(scene_pcd)
    ptr = [0]
    prev_feature = torch.from_numpy(scene_graph.nodes[0]['feature']).to(device)

    def show_next_instance_callback(visualizer, action, mod, pressed_key):
        if action == 0:  # key is pressed
            if pressed_key == 'J':
                ptr[0] += 1
            elif pressed_key == 'K':
                ptr[0] -= 1
            instance_id = ptr[0] % len(scene_graph.nodes)
            instance_feature = torch.from_numpy(scene_graph.nodes[instance_id]['feature']).to(device)

            # compute feature distance from last instance
            feature_distance = instance_feature @ prev_feature
            # print(f"feature cosine distance from last instance: {feature_distance:.3f}")
            prev_feature[:] = instance_feature

            closest_category_ids = (vocab_features @ instance_feature).topk(5).indices
            top5_closest_categories = [vocabs[i.item()] for i in closest_category_ids]
            if vocabs is None or vocab_features is None:
                try:
                    top5_closest_categories = scene_graph.nodes[instance_id]['top5_vocabs']
                except KeyError:
                    top5_closest_categories = None
            print(f"{top5_closest_categories = }")

            colors = np.asarray(pcd_vis.colors)
            colors[:] = np.asarray(scene_pcd.colors)
            pt_indices = scene_graph.nodes[instance_id]['pt_indices']
            colors[pt_indices] = (1, 1, 0)
        return True

    vis = o3d.visualization.VisualizerWithKeyCallback()
    for key in ['J', 'K']:
        vis.register_key_action_callback(ord(key), partial(show_next_instance_callback, pressed_key=key))
    vis.create_window()
    vis.add_geometry(pcd_vis)
    vis.run()


def text_query(clip_model, scene_pcd, scene_graph, device="cuda:0"):
    query = input("Please query an object: (input 'q' to quit)\n")
    while query != 'q':
        text_feature = get_clip_feature(clip_model, query, normalize=True, device=device)
        text_feature = text_feature.cpu().detach().numpy().flatten()

        # compute feature distances
        distances = []
        for node_id in scene_graph.nodes:
            instance_feature = scene_graph.nodes[node_id]['feature']
            feature_distance = distance.cosine(text_feature, instance_feature)
            distances.append(feature_distance)
        distances = np.array(distances)
        ascending_indices = distances.argsort()
        ptr = [0]

        pcd_vis = copy.deepcopy(scene_pcd)
        matched_id = ascending_indices[0]
        indices_3d = scene_graph.nodes[matched_id]['pt_indices']
        pcd_colors = np.asarray(pcd_vis.colors)
        pcd_colors[indices_3d] = (0, 1, 1)
        print(f"{matched_id = }, feature cosine distance: {distances[matched_id]}")

        def show_next_instance_callback(visualizer, action, mod, pressed_key):
            if action == 0:  # key is pressed
                if pressed_key == 'J':
                    ptr[0] += 1
                elif pressed_key == 'K':
                    ptr[0] -= 1
                idx = ptr[0] % len(scene_graph.nodes)
                instance_id = ascending_indices[idx]
                print(f"{idx = }, feature cosine distance: {distances[instance_id]}")
                pt_indices = scene_graph.nodes[instance_id]['pt_indices']
                colors = np.asarray(pcd_vis.colors)
                colors[:] = np.asarray(scene_pcd.colors)
                colors[pt_indices] = (0, 1, 1)
            return True

        vis = o3d.visualization.VisualizerWithKeyCallback()
        for key in ['J', 'K']:
            vis.register_key_action_callback(ord(key), partial(show_next_instance_callback, pressed_key=key))
        vis.create_window()
        vis.add_geometry(pcd_vis)
        vis.run()

        query = input("Please query an object: (input 'q' to quit)\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, default="~/t7/ScanNet/")
    parser.add_argument("-v", "--video", type=str, default="scene0645_00")
    # parser.add_argument("-d", "--dataset", type=str, default="~/t7/ycb_video")
    # parser.add_argument("-v", "--video", type=str, default="0048")
    parser.add_argument("--detic_exp", type=str, default="imagenet21k-0.3")
    parser.add_argument("--prediction_file",
                        default="proposed_fusion_detic_iou-0.25_recall-0.50_feature-0.75_interval-300.pkl")
    parser.add_argument("--vocabs", default="lvis")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    git_repo = Path(git.Repo(search_parent_directories=True).working_tree_dir)
    sys.path.append(str(git_repo))

    print(f"Using {args.vocabs} vocabulary.")

    dataset = Path(args.dataset).expanduser()
    if "ScanNet" in args.dataset:
        video_path = dataset / "aligned_scans" / args.video
        scene_pcd_path = video_path / f"{args.video}_vh_clean_2.ply"
    elif "ycb_video" in args.dataset:
        video_path = dataset / args.video
        scene_pcd_path = video_path / f"scan-0.005.pcd"
    else:  # custom data
        video_path = dataset / args.video
        scene_pcd_path = glob.glob(str(video_path / "scan-*.pcd"))[0]

    clip_model, _ = clip.load('ViT-B/32', args.device)
    vocabs = importlib.import_module("src.vocabs").vocabs[args.vocabs]
    vocab_features = get_clip_feature(clip_model=clip_model, text_label=vocabs, normalize=True, device=args.device)

    print(f"{str(scene_pcd_path) = }")
    scene_pcd = o3d.io.read_point_cloud(str(scene_pcd_path))
    prediction_path = video_path / 'detic_output' / args.detic_exp / 'predictions' / args.prediction_file
    with open(prediction_path, 'rb') as fp:
        scene_graph = pickle.load(fp)

    print("Visualizing scene point cloud")
    vis_pcd(scene_pcd)

    print(f'Visualizing all instances in {args.video}')
    visualize_scene_graph(scene_pcd, scene_graph, show_center=True)

    print(f"Visualizing instances in {args.video} one by one")
    visualize_instances(scene_pcd, scene_graph, vocabs, vocab_features, device=args.device)

    print("Visualizing instances by query")
    text_query(clip_model, scene_pcd, scene_graph, device=args.device)


if __name__ == "__main__":
    main()

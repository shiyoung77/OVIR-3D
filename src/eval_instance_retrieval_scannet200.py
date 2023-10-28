import os
import time
import sys
import pickle
import argparse
import importlib
from pathlib import Path

import git
import numpy as np
import open3d as o3d
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

from concurrent.futures import ProcessPoolExecutor
from itertools import repeat


def get_gt_instances(gt_instance_ids, valid_ids=None):
    unique_instance_ids = np.unique(gt_instance_ids)
    N = unique_instance_ids.shape[0]
    masks = []
    cat_ids = []
    for i in range(N):
        unique_id = unique_instance_ids[i]
        cat_id = unique_id // 1000
        if valid_ids and cat_id not in valid_ids:
            continue
        mask = gt_instance_ids == unique_id
        cat_ids.append(cat_id)
        masks.append(mask)
    cat_ids = np.array(cat_ids)
    masks = np.stack(masks)
    return cat_ids, masks


def get_predicted_instances(scene_graph, feature_name="feature"):
    node_features = []
    node_masks = []
    n_pts = scene_graph.graph['n_pts']
    for node in scene_graph.nodes:
        node_features.append(scene_graph.nodes[node][feature_name])
        node_mask = np.zeros(n_pts, dtype=np.bool_)
        node_mask[scene_graph.nodes[node]["pt_indices"]] = True
        node_masks.append(node_mask)
    node_features = np.vstack(node_features)  # (N, n_dims)
    node_masks = np.stack(node_masks)  # (N, n_pts)
    return node_features, node_masks


def CalculateAveragePrecision(rec, prec):
    """
    https://github.com/rafaelpadilla/Object-Detection-Metrics/blob/master/lib/Evaluator.py#L295
    """
    mrec = [0]
    for e in rec:
        mrec.append(e)
    mrec.append(1)

    mpre = [0]
    for e in prec:
        mpre.append(e)
    mpre.append(0)

    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    ii = []
    for i in range(len(mrec) - 1):
        if mrec[1+i] != mrec[i]:
            ii.append(i + 1)
    ap = 0
    for i in ii:
        ap = ap + np.sum((mrec[i] - mrec[i - 1]) * mpre[i])
    return [ap, mpre[0:len(mpre) - 1], mrec[0:len(mpre) - 1], ii]


@ignore_warnings(category=ConvergenceWarning)
def process_single_scan(args, dataset, gt_path: Path, cat_id_to_feature, valid_ids):
    scan = os.path.splitext(gt_path.name)[0]
    print(f"Start processing {scan = }")
    scan_path = dataset / scan
    scene_pcd = o3d.io.read_point_cloud(str(scan_path / f"{scan}_vh_clean_2.ply"))
    scene_graph_path = scan_path / args.detic_output_folder / args.detic_exp / "predictions" / args.prediction_file
    if not scene_graph_path.is_file():
        print("========================================")
        print(f"{scene_graph_path = } doesn't exist!")
        print("========================================")
        return

    output_path = scan_path / args.detic_output_folder / args.detic_exp / "results" / f"{args.feature_name}_{args.prediction_file}"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(scene_graph_path, 'rb') as fp:
        scene_graph = pickle.load(fp)
        scene_graph.graph['scan'] = scan
        scene_graph.graph['n_pts'] = len(scene_pcd.points)

    # read all predictions
    if args.feature_name == "representative_feature":
        detection_folder = scan_path / args.detic_output_folder / args.detic_exp / 'instances'
        frame_to_features = dict()
        for instance_file in detection_folder.iterdir():
            frame_id = instance_file.stem.split('-')[0]
            with open(instance_file, 'rb') as fp:
                instance = pickle.load(fp)
                features = instance.pred_box_features.numpy()
            frame_to_features[frame_id] = features

        for i in scene_graph.nodes:
            node = scene_graph.nodes[i]
            node['features'] = []
            for detection in node['detections']:
                frame, mask_idx, _ = detection
                node['features'].append(frame_to_features[frame][mask_idx])
            node['features'] = np.stack(node['features'], axis=0)

            # compute K representative features
            if node['features'].shape[0] <= args.K:
                rand_indices = np.random.choice(node['features'].shape[0], args.K)
                node['representative_features'] = node['features'][rand_indices]
            else:
                k_means = KMeans(n_clusters=args.K, n_init='auto', random_state=0)
                k_means.fit(node['features'])
                node['representative_features'] = k_means.cluster_centers_

    gt_instance_ids = np.loadtxt(gt_path, dtype=np.int64)
    gt_cat_ids, gt_masks = get_gt_instances(gt_instance_ids, valid_ids=valid_ids)
    pred_features, pred_masks = get_predicted_instances(scene_graph, feature_name=args.feature_name)
    if args.feature_name == 'representative_features':
        assert pred_features.shape[0] // args.K == pred_masks.shape[0], f"{pred_features.shape[0] = }, {pred_masks.shape[0] = }"

    ap_results = compute_ap_for_each_scan(pred_features, pred_masks, gt_cat_ids, gt_masks, cat_id_to_feature)
    with open(output_path, 'wb') as fp:
        pickle.dump(ap_results, fp)
    print(f"Processed {scan = }. Results saved to: {output_path}")
    return ap_results


def compute_ap_for_each_scan(pred_features, pred_masks, gt_cat_ids, gt_masks, cat_id_to_feature):
    """
    ref: https://github.com/rafaelpadilla/Object-Detection-Metrics/blob/master/lib/Evaluator.py
    """
    iou_thresholds = np.zeros(11)
    iou_thresholds[0] = 0.25
    iou_thresholds[1:] = np.arange(0.5, 1, 0.05)

    ap_dict = dict()
    unique_cat_ids = np.unique(gt_cat_ids)
    for cat_id in unique_cat_ids:
        if cat_id not in cat_id_to_feature:
            continue

        ap_dict[cat_id] = dict()
        cat_feature = cat_id_to_feature[cat_id]
        similarities = pred_features @ cat_feature

        # if using K representative features, take the max similarity
        if pred_features.shape[0] != pred_masks.shape[0]:
            K = pred_features.shape[0] // pred_masks.shape[0]
            similarities = similarities.reshape(-1, K).max(1)
            assert similarities.shape[0] == pred_masks.shape[0]

        # sort pred mask based on similarity (descending order)
        sorted_indices = np.argsort(-similarities)
        sorted_node_masks = pred_masks[sorted_indices]  # (N, n_pts)
        cat_masks = gt_masks[gt_cat_ids == cat_id]  # (M, n_pts)

        # compute IoU matrix between all predictions and gt masks
        intersection = sorted_node_masks.astype(np.float32) @ cat_masks.astype(np.float32).T  # (N, M)
        union = sorted_node_masks.sum(1, keepdims=True) + cat_masks.sum(1, keepdims=True).T - intersection
        iou_matrix = intersection / (union + 1e-6)
        N, M = iou_matrix.shape  # (num_pred, num_gt)

        # ref: https://github.com/rafaelpadilla/Object-Detection-Metrics/blob/master/lib/Evaluator.py#L102
        for iou_threshold in iou_thresholds:
            visited = np.zeros((M,), dtype=np.bool_)
            TP = np.zeros((N,), dtype=np.bool_)
            FP = np.zeros((N,), dtype=np.bool_)
            for n in range(N):  # go through all sorted predictions based on similarity
                iou_max = -np.inf
                m_match = -1
                for m in range(M):  # find the best gt match for this prediction
                    iou = iou_matrix[n, m]
                    if iou > iou_max:
                        iou_max = iou
                        m_match = m
                if iou_max >= iou_threshold:
                    if not visited[m_match]:
                        TP[n] = True
                        visited[m_match] = True
                    else:  # gt mask has been matched with other predictions
                        FP[n] = True
                else:
                    FP[n] = True
            acc_TP = np.cumsum(TP)
            acc_FP = np.cumsum(FP)
            precisions = acc_TP / (acc_FP + acc_TP)
            recalls = acc_TP / M
            ap, _, _, _ = CalculateAveragePrecision(recalls, precisions)
            ap_name = f"ap_{int(iou_threshold * 100)}"
            ap_dict[cat_id][ap_name] = ap
    return ap_dict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, default="~/t7/ScanNet/aligned_scans")
    parser.add_argument("--detic_output_folder", type=str, default="detic_output")
    parser.add_argument("--detic_exp", type=str, default="imagenet21k-0.3")
    parser.add_argument("--prediction_file", type=str,
                        default="proposed_fusion_detic_iou-0.25_recall-0.50_feature-0.75_interval-300.pkl")
    parser.add_argument("--feature_name", type=str, default="feature", choices=['feature', 'representative_features'])
    parser.add_argument("--K", type=int, default=64)
    parser.add_argument("--max_workers", type=int, default=8)
    args = parser.parse_args()

    if args.feature_name == "feature":  # i.e. mean feature
        print("========================================")
        print("Evaluate using mean feature for each instance")
        print("========================================")
    else:
        print("========================================")
        print(f"Evaluate using {args.K} representative features for each instance.")
        print("This requires the Detic output files to be available.")
        print("========================================")

    git_repo = Path(git.Repo(search_parent_directories=True).working_tree_dir)
    sys.path.append(str(git_repo))

    dataset = Path(args.dataset).expanduser()
    scannet200_constants = importlib.import_module("scannet_related_scripts.scannet200_constants")
    scannet200_splits = importlib.import_module("scannet_related_scripts.scannet200_splits")

    gt_folder = git_repo / "scannet_related_scripts" / "scannet200_instance_gt" / "validation"
    gt_paths = sorted(gt_folder.iterdir())

    id_mapping_path = git_repo / "scannet_related_scripts" / "scannetv2-labels.combined.tsv"
    df = pd.read_csv(id_mapping_path, sep="\t")

    CLASS_LABELS = scannet200_constants.CLASS_LABELS_200
    VALID_CLASS_IDS = scannet200_constants.VALID_CLASS_IDS_200

    ID_TO_LABEL = {}
    LABEL_TO_ID = {}
    for i in range(len(VALID_CLASS_IDS)):
        LABEL_TO_ID[CLASS_LABELS[i]] = VALID_CLASS_IDS[i]
        ID_TO_LABEL[VALID_CLASS_IDS[i]] = CLASS_LABELS[i]

    # uncountable categories: i.e. "wall", "floor" and their subcategories + "ceiling"
    INVALID_IDS = [df['id'][i] for i in range(df.shape[0])
                   if df['nyuClass'][i] in ("wall", "floor") or df['nyuClass'][i] == "ceiling"]

    id_mapping = {cat_id: i for i, cat_id in enumerate(scannet200_constants.VALID_CLASS_IDS_200)}

    HEAD_CATS = set(LABEL_TO_ID[i] for i in scannet200_splits.HEAD_CATS_SCANNET_200)
    COMMON_CATS = set(LABEL_TO_ID[i] for i in scannet200_splits.COMMON_CATS_SCANNET_200)
    TAIL_CATS = set(LABEL_TO_ID[i] for i in scannet200_splits.TAIL_CATS_SCANNET_200)
    VAL_IDS = set(i for i in scannet200_splits.CLASS_LABELS_200_VALIDATION if i not in INVALID_IDS)
    print(f"{len(scannet200_splits.CLASS_LABELS_200_VALIDATION) = }")
    print(f"After filtering uncountable categories, {len(VAL_IDS) = }")

    vocab_features = np.load(git_repo / "src" / "scannet200.npy")
    cat_id_to_feature = {cat_id: vocab_features[id_mapping[cat_id]] for cat_id in VALID_CLASS_IDS}

    # ================================================================================
    tic = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        executor.map(process_single_scan, repeat(args), repeat(dataset), gt_paths, repeat(cat_id_to_feature), repeat(VAL_IDS))
    print(f"Processed {len(gt_paths)} scans in: {time.perf_counter() - tic:.2f}s")
    # ================================================================================

    iou_thresholds = np.zeros(11)
    iou_thresholds[0] = 0.25
    iou_thresholds[1:] = np.arange(0.5, 1, 0.05)

    total_ap_results = dict()
    for cat_id in VAL_IDS:
        total_ap_results[cat_id] = dict()
        for iou_threshold in iou_thresholds:
            total_ap_results[cat_id][f"ap_{int(iou_threshold * 100)}"] = list()

    for gt_path in gt_paths:
        scan = os.path.splitext(gt_path.name)[0]
        scan_path = dataset / scan
        output_path = scan_path / args.detic_output_folder / args.detic_exp / "results" / f"{args.feature_name}_{args.prediction_file}"
        if not output_path.is_file():
            continue
        with open(output_path, 'rb') as fp:
            ap_results = pickle.load(fp)
        for cat_id in ap_results:
            if cat_id not in VAL_IDS:
                continue
            for iou_threshold in iou_thresholds:
                ap_name = f"ap_{int(iou_threshold * 100)}"
                total_ap_results[cat_id][ap_name].append(ap_results[cat_id][ap_name])

    # write retrival mAP results
    if args.feature_name == "representative_features":
        output_path = git_repo / "results" / f"{args.detic_exp}-{args.feature_name}-{args.K}-mAP.txt"
    else:
        output_path = git_repo / "results" / f"{args.detic_exp}-{args.feature_name}-mAP.txt"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as fp:
        fp.write("id,label,ap,ap50,ap25\n")

        cat_id_to_ap = dict()
        for cat_id in VAL_IDS:
            mAP = []
            for iou_threshold in iou_thresholds:  # [0.25, 0.5, 0.55, ..., 0.95]
                ap_name = f"ap_{int(iou_threshold * 100)}"
                mAP.append(np.mean(total_ap_results[cat_id][ap_name]).item())
            mAP.append(np.mean(mAP[1:]).item())  # overall mAP = mean(ap50, ap55, ..., ap95)
            cat_id_to_ap[cat_id] = mAP

        # AP for each category set
        for name, cat_ids in {"all": VAL_IDS, "head": HEAD_CATS, "common": COMMON_CATS, "tail": TAIL_CATS}.items():
            mAP_set = []
            for cat_id in cat_ids:
                if cat_id in VAL_IDS:
                    mAP_set.append(cat_id_to_ap[cat_id])
            mAP_set = np.stack(mAP_set)
            mAP_set = np.mean(mAP_set, axis=0)
            fp.write(f"_,{name}")
            for i in [11, 1, 0]:  # [mAP, ap50, ap25]
                fp.write(f',{mAP_set[i]:.3f}')
            fp.write('\n')

        # AP for each category
        for cat_id in VAL_IDS:
            fp.write(f"{cat_id},{ID_TO_LABEL[cat_id]}")
            for i in [11, 1, 0]:  # [mAP, ap50, ap25]
                fp.write(f',{cat_id_to_ap[cat_id][i]:.3f}')
            fp.write('\n')
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()

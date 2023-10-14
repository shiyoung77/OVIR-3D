import os
import importlib
import pickle
import sys
import copy
import json
import glob
import argparse
import operator
from time import perf_counter
from functools import reduce
from collections import deque
from pathlib import Path

import git
import numpy as np
import numpy.linalg as la
import torch
import torch.nn.functional as F
import cv2
import open3d as o3d
import networkx as nx
import pycocotools.mask
from tqdm import tqdm
from numba import njit
from scipy.optimize import linear_sum_assignment
from torchmetrics.functional import pairwise_cosine_similarity
from sklearn.cluster import DBSCAN
# from pytorch3d.ops import box3d_overlap

from utils import read_detectron_instances, compute_projected_pts, compute_visibility_mask, vis_pcd, plane_detection_o3d
from visualize_predictions import visualize_scene_graph


@njit
def iou_matching_greedy(iou_matrix, area_counter, iou_thresh=0.5, unmatched_indicator=-1):
    M, N = iou_matrix.shape
    correspondences = np.full((M,), fill_value=unmatched_indicator, dtype=np.int32)
    matched_keys = np.zeros((N,), dtype=np.bool_)
    sorted_query = np.argsort(-area_counter)  # sort query segments by area (descending order)
    for i in sorted_query:
        for j in np.argsort(-iou_matrix[i]):  # sort key segments by iou (descending order)
            if matched_keys[j]:
                continue
            iou = iou_matrix[i, j]
            if iou >= iou_thresh:
                correspondences[i] = j
                matched_keys[j] = True
            break
    return correspondences


@njit
def iou_matching_greedy_2(iou_matrix, iou_thresh=0.5, unmatched_indicator=-1):
    M, N = iou_matrix.shape
    correspondences = np.full((M,), fill_value=unmatched_indicator, dtype=np.int32)
    for i in range(M):
        for j in np.argsort(-iou_matrix[i]):  # sort key segments by iou (descending order)
            iou = iou_matrix[i, j]
            if iou >= iou_thresh:
                correspondences[i] = j
            break
    return correspondences


@njit
def matching_with_feature_rejection(iou_matrix, precision_matrix, recall_matrix, feature_similarity_matrix,
                                    iou_thresh=0.3, precision_recall_thresh=0.8, feature_similarity_thresh=0.8,
                                    unmatched_indicator=-1):
    M, N = iou_matrix.shape
    correspondences = np.full((M,), fill_value=unmatched_indicator, dtype=np.int32)
    for i in range(M):
        for j in np.argsort(-iou_matrix[i]):  # sort key segments by iou (descending order)
            iou = iou_matrix[i, j]
            feature_similarity = feature_similarity_matrix[i, j]
            if (iou >= iou_thresh) and (feature_similarity >= feature_similarity_thresh):
                correspondences[i] = j
                break
    return correspondences


def iou_matching_hungarian(iou_matrix, iou_thresh=0.5, unmatched_indicator=-1):
    M, N = iou_matrix.shape
    correspondences = np.full((M,), fill_value=unmatched_indicator, dtype=np.int32)
    row_indices, col_indices = linear_sum_assignment(-iou_matrix)
    mask = iou_matrix[row_indices, col_indices] > iou_thresh
    row_indices = row_indices[mask]
    col_indices = col_indices[mask]
    correspondences[row_indices] = col_indices
    return correspondences


@njit
def compute_visible_masked_pts(scene_pts, projected_pts, visibility_mask, pred_masks):
    N = scene_pts.shape[0]
    M, _, _ = pred_masks.shape  # (M, H, W)
    masked_pts = np.zeros((M, N), dtype=np.bool_)
    visible_indices = np.nonzero(visibility_mask)[0]
    for m in range(M):
        for i in visible_indices:
            x, y = projected_pts[i]
            if pred_masks[m, y, x]:
                masked_pts[m, i] = True
    return masked_pts


def integrate_instances(instance_pt_count, num_instances, masked_pts, correspondences, instance_features,
                        pred_features, instance_detections, frame_id, valid_mask_indices, unmatched_indicator=-1):
    assert valid_mask_indices.shape[0] == masked_pts.shape[0]
    for m in range(masked_pts.shape[0]):
        c = correspondences[m]
        if c == unmatched_indicator:
            c = num_instances
            instance_detections[c] = []
            num_instances += 1
        instance_pt_count[c] += masked_pts[m]
        instance_features[c] += pred_features[m]
        # pred_feature = pred_features[m].detach().cpu().numpy()
        mask_size = torch.sum(masked_pts[m]).detach().cpu().numpy()
        instance_detections[c].append((frame_id, valid_mask_indices[m], mask_size))
    return num_instances


def compute_relation_matrix(masked_pts, instance_pt_count, visibility_mask):
    visibility_mask = torch.from_numpy(visibility_mask).to(instance_pt_count.device)
    instance_pt_mask = instance_pt_count.to(torch.bool)
    instance_pt_mask[:, ~visibility_mask] = False
    instance_pt_mask = instance_pt_mask.to(torch.float32)
    masked_pts = masked_pts.to(torch.float32)

    intersection = masked_pts @ instance_pt_mask.T  # (M, num_instances)
    masked_pts_sum = masked_pts.sum(1, keepdims=True)  # (M, 1)
    instance_pt_mask_sum = instance_pt_mask.sum(1, keepdims=True)  # (num_instances, 1)

    union = masked_pts_sum + instance_pt_mask_sum.T - intersection
    iou_matrix = intersection / (union + 1e-6)
    recall_matrix = intersection / (instance_pt_mask_sum.T + 1e-6)
    precision_matrix = intersection / (masked_pts_sum + 1e-6)
    return iou_matrix, precision_matrix, recall_matrix


def compute_relation_matrix_self(instance_pt_count):
    if not torch.is_tensor(instance_pt_count):
        instance_pt_count = torch.from_numpy(instance_pt_count)
    instance_pt_mask = instance_pt_count.to(torch.bool).to(torch.float32)
    intersection = instance_pt_mask @ instance_pt_mask.T  # (M, num_instances)
    inliers = instance_pt_mask.sum(1, keepdims=True)
    union = inliers + inliers.T - intersection
    iou_matrix = intersection / (union + 1e-6)
    precision_matrix = intersection / (inliers.T + 1e-6)
    recall_matrix = intersection / (inliers + 1e-6)
    return iou_matrix, precision_matrix, recall_matrix


def compute_recall_3d_matrix(boxes_o3d, intersection_3d_matrix):
    N = len(boxes_o3d)
    assert intersection_3d_matrix.shape[0] == N
    volume = torch.tensor([box.volume() for box in boxes_o3d], device=intersection_3d_matrix.device)  # (N,)
    scale = 100  # magic number, intersection_3d_matrix is scaled to prevent numerical issues
    recall_3d_matrix = intersection_3d_matrix / (volume.unsqueeze(0) * scale ** 3)
    return recall_3d_matrix


def filter_pt_by_visibility_count(instance_pt_count, visibility_count, visibility_thresh=0.2, inplace=False):
    if isinstance(visibility_count, np.ndarray):
        visibility_count = torch.from_numpy(visibility_count).to(instance_pt_count.device)
    instance_pt_visibility = instance_pt_count / visibility_count.clip(min=1e-6).unsqueeze(0)
    if not inplace:
        instance_pt_count = torch.clone(instance_pt_count)
    instance_pt_count[instance_pt_visibility < visibility_thresh] = 0
    return instance_pt_count


def filter_by_instance_size(instance_pt_count, size_thresh=50):
    mask = instance_pt_count.to(torch.bool).sum(1) > size_thresh
    keep_indices = torch.nonzero(mask)[:, 0]
    return keep_indices


def find_connected_components(adj_matrix):
    if torch.is_tensor(adj_matrix):
        adj_matrix = adj_matrix.detach().cpu().numpy()
    assert adj_matrix.shape[0] == adj_matrix.shape[1], "adjacency matrix should be a square matrix"

    N = adj_matrix.shape[0]
    clusters = []
    visited = np.zeros(N, dtype=np.bool_)
    for i in range(N):
        if visited[i]:
            continue
        cluster = []
        queue = deque([i])
        visited[i] = True
        while queue:
            j = queue.popleft()
            cluster.append(j)
            for k in np.nonzero(adj_matrix[j])[0]:
                if not visited[k]:
                    queue.append(k)
                    visited[k] = True
        clusters.append(cluster)
    return clusters


def merge_instances(instance_pt_count, instance_features, instance_detections, clusters, inplace=True):
    _, N = instance_pt_count.shape
    _, C = instance_features.shape
    M = len(clusters)
    device = instance_pt_count.device
    merged_instance_pt_count = torch.zeros((M, N), dtype=instance_pt_count.dtype, device=device)
    merged_features = torch.zeros((M, C), dtype=instance_features.dtype, device=device)
    merged_detections = dict()
    for i, cluster in enumerate(clusters):
        merged_instance_pt_count[i] = instance_pt_count[cluster].sum(0)
        merged_features[i] = instance_features[cluster].sum(0)
        merged_detections[i] = reduce(operator.add, (instance_detections[j] for j in cluster))
    if inplace:
        instance_pt_count[:M] = merged_instance_pt_count
        instance_features[:M] = merged_features
        instance_pt_count[M:] = 0
        instance_features[M:] = 0
    return merged_instance_pt_count, merged_features, merged_detections


def resolve_overlapping_masks(pred_masks, pred_scores, score_thresh=0.5, device="cuda:0"):
    M, H, W = pred_masks.shape
    pred_masks = torch.from_numpy(pred_masks).to(device)
    panoptic_masks = torch.clone(pred_masks)
    scores = torch.from_numpy(pred_scores)[:, None, None].repeat(1, H, W).to(device)
    scores[~pred_masks] = 0
    indices = ((scores == torch.max(scores, dim=0, keepdim=True).values) & pred_masks).nonzero()
    panoptic_masks = torch.zeros((M, H, W), dtype=torch.bool, device=device)
    panoptic_masks[indices[:, 0], indices[:, 1], indices[:, 2]] = True
    panoptic_masks[scores > score_thresh] = True  # if prediction score is high enough, keep the mask anyway
    return panoptic_masks.detach().cpu().numpy()


def instance_fusion(video_path, detic_exp, scene_pcd, iou_thresh=0.3, recall_thresh=0.5, feature_similarity_thresh=0.75,
                    depth_thresh=0.04, size_thresh=50, interval=300, visibility_thresh=0.2, stride=1,
                    ground_indices=None, use_sam=False, disable_tqdm=False, device="cuda:0"):
    # load camera configs
    with open(video_path / "config.json", 'r') as fp:
        cam_config = json.load(fp)
    cam_intr = np.asarray(cam_config['cam_intr'])
    depth_scale = cam_config['depth_scale']

    max_instances = 3000  # subject to change if more instances exist in huge scenes
    num_instances = 0  # number of instances integrated to the 3D scene
    N = len(scene_pcd.points)
    instance_pt_count = torch.zeros((max_instances, N), dtype=torch.int32, device=device)
    instance_features = torch.zeros((max_instances, 512), dtype=torch.half, device=device)
    visibility_count = np.zeros((N,), dtype=np.int32)
    instance_detections = dict()

    frame_ids = [f.split('-')[0] for f in sorted(os.listdir(video_path / 'color'))]
    frame_ids = [frame_ids[i] for i in range(0, len(frame_ids), stride)]
    for i, frame_id in enumerate(tqdm(frame_ids, disable=disable_tqdm)):
        if use_sam:
            sam_folder = video_path / 'sam_output' / detic_exp
            sam_result_path = sam_folder / 'instances' / f'{frame_id}-color.pkl'
            try:
                detic_output = read_detectron_instances(sam_result_path, rle_to_mask=False)
            except FileNotFoundError:
                continue
            if not detic_output.sam_masks_rle:
                continue
            pred_masks = np.stack([pycocotools.mask.decode(rle) for rle in detic_output.sam_masks_rle])
            detic_output.pred_masks = torch.from_numpy(pred_masks).to(torch.bool)  # (M, H, W)
        else:
            detic_folder = video_path / 'detic_output' / detic_exp
            detic_result_path = detic_folder / 'instances' / f'{frame_id}-color.pkl'
            detic_output = read_detectron_instances(detic_result_path)

        pred_scores = detic_output.scores.numpy()  # (M,)
        pred_masks = detic_output.pred_masks.numpy()  # (M, H, W)
        pred_features = detic_output.pred_box_features.to(device)  # (M, 512)
        pred_features = F.normalize(pred_features, dim=1, p=2)
        # pred_classes = detic_output.pred_classes.numpy()  # (M,)
        # sam_qualities = detic_output.sam_qualities

        pred_masks = resolve_overlapping_masks(pred_masks, pred_scores, device=device)

        cam_pose = np.loadtxt(video_path / 'poses' / f"{frame_id}-pose.txt")
        pcd = copy.deepcopy(scene_pcd).transform(np.linalg.inv(cam_pose))
        scene_pts = np.asarray(pcd.points)
        projected_pts = compute_projected_pts(scene_pts, cam_intr)

        depth_im_path = str(video_path / 'depth' / f"{frame_id}-depth.png")
        depth_im = cv2.imread(depth_im_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / depth_scale
        visibility_mask = compute_visibility_mask(scene_pts, projected_pts, depth_im, depth_thresh=depth_thresh)
        visibility_count[visibility_mask] += 1

        masked_pts = compute_visible_masked_pts(scene_pts, projected_pts, visibility_mask, pred_masks)  # (M, N)
        masked_pts = torch.from_numpy(masked_pts).to(device)
        if ground_indices is not None:
            masked_pts[:, ground_indices] = 0
        mask_area = torch.sum(masked_pts, dim=1).detach().cpu().numpy()  # (M,)

        # filter small/bad predictions
        valid_mask = mask_area >= 25
        # valid_mask = valid_mask & (sam_qualities >= 0.5)
        masked_pts = masked_pts[valid_mask]
        pred_features = pred_features[valid_mask]
        pred_scores = pred_scores[valid_mask]
        valid_mask_indices = np.where(valid_mask)[0]

        iou_matrix, precision_matrix, recall_matrix = compute_relation_matrix(
            masked_pts, instance_pt_count[:num_instances], visibility_mask
        )
        feature_similarity_matrix = pairwise_cosine_similarity(pred_features, instance_features)
        iou_matrix_np = iou_matrix.cpu().numpy().astype(np.float32)  # (M, num_instances)
        precision_matrix_np = precision_matrix.cpu().numpy().astype(np.float32)  # (M, num_instances)
        recall_matrix_np = recall_matrix.cpu().numpy().astype(np.float32)  # (M, num_instances)
        feature_similarity_matrix_np = feature_similarity_matrix.detach().cpu().numpy().astype(np.float32)

        unmatched_indicator = -1
        # filter_indicator = -2
        # correspondences = iou_matching_greedy(iou_matrix_np, mask_area, iou_thresh=iou_thresh,
        #                                       unmatched_indicator=unmatched_indicator)
        # correspondences = iou_matching_greedy_2(iou_matrix_np, iou_thresh=iou_thresh,
        #                                         unmatched_indicator=unmatched_indicator)
        # correspondences = iou_matching_hungarian(iou_matrix_np, iou_thresh=iou_thresh,
        #                                          unmatched_indicator=unmatched_indicator)
        correspondences = matching_with_feature_rejection(
            iou_matrix_np, precision_matrix_np, recall_matrix_np, feature_similarity_matrix_np,
            iou_thresh=iou_thresh,
            precision_recall_thresh=recall_thresh,
            feature_similarity_thresh=feature_similarity_thresh,
            unmatched_indicator=unmatched_indicator
        )

        num_instances = integrate_instances(instance_pt_count, num_instances, masked_pts, correspondences,
                                            instance_features, pred_features, instance_detections, frame_id,
                                            valid_mask_indices, unmatched_indicator=unmatched_indicator)

        # if False:
        # if (i == len(frame_ids) - 1) or (i != 0 and i % interval == 0):
        if (i == len(frame_ids) - 1) or (i % interval == 0):
            # print(f"# instances before filtering: {num_instances}")
            # ===================== filter instances based on visibility and size =====================
            tic = perf_counter()
            filter_pt_by_visibility_count(instance_pt_count[:num_instances], visibility_count,
                                          visibility_thresh=visibility_thresh, inplace=True)
            keep_indices = filter_by_instance_size(instance_pt_count[:num_instances], size_thresh=size_thresh)
            instance_pt_count[:len(keep_indices)] = instance_pt_count[:num_instances][keep_indices]
            instance_pt_count[len(keep_indices):] = 0
            instance_features[:len(keep_indices)] = instance_features[:num_instances][keep_indices]
            instance_features[len(keep_indices):] = 0
            num_instances = len(keep_indices)
            instance_detections = {new_idx: instance_detections[idx.item()] for new_idx, idx in enumerate(keep_indices)}
            # print(f"# instances after filtering: {num_instances}")
            # print(f"filtering instances takes: {perf_counter() - tic}s")

            # ===================== merge instances based on IoU, recall, and feature similarity =====================
            tic = perf_counter()
            iou_matrix, _, recall_matrix = compute_relation_matrix_self(instance_pt_count[:num_instances])
            semantic_similarity_matrix = pairwise_cosine_similarity(instance_features[:num_instances],
                                                                    instance_features[:num_instances])
            # adjacency_matrix = (iou_matrix >= iou_thresh) & (semantic_similarity_matrix >= feature_similarity_thresh)
            adjacency_matrix = ((iou_matrix >= iou_thresh) | (recall_matrix >= recall_thresh)) \
                & (semantic_similarity_matrix >= feature_similarity_thresh)
            adjacency_matrix = adjacency_matrix | adjacency_matrix.T

            # merge instances based on the adjacency matrix
            connected_components = find_connected_components(adjacency_matrix)
            num_instances = len(connected_components)
            _, _, instance_detections = merge_instances(
                instance_pt_count, instance_features, instance_detections, connected_components, inplace=True
            )
            # print(f"# instances after merging: {num_instances}")
            # print(f"merging instances takes: {perf_counter() - tic}s")

    instance_pt_count = instance_pt_count[:num_instances]
    instance_features = instance_features[:num_instances]
    instance_features = F.normalize(instance_features, dim=1, p=2)
    # return instance_pt_count, instance_features, instance_detections
    return post_processing(scene_pcd, instance_pt_count, instance_features, instance_detections,
                           size_thresh=size_thresh, iou_thresh=iou_thresh,
                           recall_thresh=recall_thresh,
                           feature_similarity_thresh=feature_similarity_thresh)


def post_processing(scene_pcd, instance_pt_count, instance_features, instance_detections,
                    size_thresh=50, iou_thresh=0.3, recall_thresh=0.5, feature_similarity_thresh=0.7):
    num_instances = instance_pt_count.shape[0]
    # print(f"# instances before post-processing: {num_instances}")

    # ===================== filter by segment connectivity =====================
    tic = perf_counter()
    mean_stats = np.empty((num_instances, 3), dtype=np.float32)
    cov_stats = np.empty((num_instances, 3, 3), dtype=np.float32)
    pcd_pts = []
    keep_indices = []
    for i in range(num_instances):
        pt_indices = instance_pt_count[i].nonzero()[:, 0].detach().cpu().numpy()
        segment_pcd = scene_pcd.select_by_index(pt_indices)
        mean, cov = segment_pcd.compute_mean_and_covariance()
        mean_stats[i] = mean
        cov_stats[i] = cov
        pts = np.array(segment_pcd.points)
        pcd_pts.append(pts)

        dbscan = DBSCAN(eps=0.1, min_samples=1)
        dbscan.fit(pts)
        unique_labels, counts = np.unique(dbscan.labels_, return_counts=True)
        largest_cluster_label = unique_labels[np.argmax(counts)]
        inlier_indices = pt_indices[dbscan.labels_ == largest_cluster_label]
        outlier_mask = np.ones(instance_pt_count.shape[1], dtype=bool)
        outlier_mask[inlier_indices] = False
        instance_pt_count[i, outlier_mask] = 0
        # if len(inlier_indices) > 0.5 * pts.shape[0] and len(inlier_indices) > size_thresh:
        if len(inlier_indices) > size_thresh:
            keep_indices.append(i)
    keep_indices = torch.LongTensor(keep_indices)
    instance_pt_count = instance_pt_count[keep_indices]
    instance_features = instance_features[keep_indices]
    instance_detections = {new_idx: instance_detections[idx.item()] for new_idx, idx in enumerate(keep_indices)}
    # num_instances = len(keep_indices)
    # print(f"filtering instances takes: {perf_counter() - tic}s")
    # print(f"# instances after filtering: {num_instances}")

    # ===================== merge by 3D IoU of minimum bounding boxes =====================
    # tic = perf_counter()
    # boxes, boxes_o3d = compute_3d_bboxes(scene_pcd, instance_pt_count)
    # intersection_3d_matrix, iou_3d_matrix = box3d_overlap(boxes, boxes, eps=1e-3)
    # recall_3d_matrix = compute_recall_3d_matrix(boxes_o3d, intersection_3d_matrix)
    # semantic_similarity_matrix = pairwise_cosine_similarity(instance_features, instance_features)
    # # adjacency_matrix = (iou_3d_matrix >= iou_thresh) & (semantic_similarity_matrix >= feature_similarity_thresh)
    # adjacency_matrix = ((iou_3d_matrix >= iou_thresh) | (recall_3d_matrix >= recall_thresh)) \
    #     & (semantic_similarity_matrix >= feature_similarity_thresh)
    # connected_components = find_connected_components(adjacency_matrix)
    # num_instances = len(connected_components)
    # print(f"merging instances takes: {perf_counter() - tic}s")
    # print(f"# instances after merging: {num_instances}")

    # visualize bboxes that will be merged
    # pcd_vis = copy.deepcopy(scene_pcd)
    # colors = np.random.random((num_instances, 3))
    # for cc, color in zip(connected_components, colors):
    #     if len(cc) < 2:
    #         continue
    #     for idx in cc:
    #         boxes_o3d[idx].color = color
    # geometries = [pcd_vis, *boxes_o3d]
    # vis_pcd(geometries)

    # instance_pt_count, instance_features, instance_detections = merge_instances(
    #     instance_pt_count, instance_features, instance_detections, connected_components, inplace=False
    # )
    # instance_features = F.normalize(instance_features, dim=1, p=2)
    return instance_pt_count, instance_features, instance_detections


def compute_3d_bboxes(scene_pcd, instance_pt_count):
    boxes_o3d = []
    boxes = []
    for i in range(instance_pt_count.shape[0]):
        pt_indices = torch.nonzero(instance_pt_count[i])[:, 0].cpu().numpy()
        instance_pcd = scene_pcd.select_by_index(pt_indices)
        box = instance_pcd.get_minimal_oriented_bounding_box(robust=True)
        boxes_o3d.append(box)

        # Do NOT use box.get_box_points() as points are not in the required order for pytorch3d.ops.box3d_overlap
        # https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/ops/iou_box3d.py#L130
        scale = 100  # magic number, prevent numerical issues when computing the iou of 3d boxes
        margin_ratio = 1.05  # make the 3D bounding box slightly larger (5% margin)
        dx, dy, dz = box.extent / 2
        dx, dy, dz = dx * scale * margin_ratio, dy * scale * margin_ratio, dz * scale * margin_ratio
        base_pts = np.array([
            [-dx, -dy, -dz],
            [-dx, dy, -dz],
            [dx, dy, -dz],
            [dx, -dy, -dz],
            [-dx, -dy, dz],
            [-dx, dy, dz],
            [dx, dy, dz],
            [dx, -dy, dz],
        ])
        box_pts = (box.R @ base_pts.T).T + box.center * scale
        boxes.append(box_pts)

    boxes = torch.from_numpy(np.stack(boxes)).to(torch.float32).to(instance_pt_count.device)
    return boxes, boxes_o3d


def build_scene_graph(scene_pcd, instance_pt_count, instance_features, instance_detections,
                      vocabs=None, vocab_features=None):
    N = instance_pt_count.shape[0]
    scene_pcd = copy.deepcopy(scene_pcd)
    G = nx.DiGraph()
    G.add_nodes_from(range(N))
    for i in range(N):
        pt_indices = torch.nonzero(instance_pt_count[i])[:, 0].cpu().numpy().astype(np.int32)
        pcd = scene_pcd.select_by_index(pt_indices)
        pts = np.asarray(pcd.points)
        G.nodes[i]['center'] = np.mean(pts, axis=0)
        G.nodes[i]['pt_indices'] = pt_indices
        G.nodes[i]['feature'] = instance_features[i].detach().cpu().numpy()
        G.nodes[i]['detections'] = instance_detections[i]
        if vocabs is not None and vocab_features is not None:
            closest_category_ids = (vocab_features @ instance_features[i]).topk(5).indices
            top5_closest_categories = [vocabs[i.item()] for i in closest_category_ids]
            G.nodes[i]['top5_vocabs'] = top5_closest_categories
    return G


def get_scannet_ground_indices(metadata_path, scene_pcd, thresh=0.025):
    transformation = np.eye(4)
    with open(metadata_path, 'r') as fp:
        for line in fp.readlines():
            line = line.strip()
            if line.startswith("axisAlignment"):
                transformation = np.array(line.split(' = ')[1].split()).astype(float).reshape((4, 4))
                break
    scene_pcd_copy = copy.deepcopy(scene_pcd)
    scene_pcd_copy.transform(transformation)
    ground_mask = np.asarray(scene_pcd_copy.points)[:, 2] < thresh
    ground_indices = np.nonzero(ground_mask)[0]
    return ground_indices


def get_ground_indices(scene_pcd, thresh=0.025):
    plane_frame, inlier_ratio = plane_detection_o3d(scene_pcd,
                                                    max_iterations=1000,
                                                    inlier_thresh=0.005,
                                                    visualize=False)
    cam_pose = la.inv(plane_frame)
    scene_pcd_copy = copy.deepcopy(scene_pcd)
    scene_pcd_copy.transform(cam_pose)
    ground_mask = np.asarray(scene_pcd_copy.points)[:, 2] < thresh
    ground_indices = np.nonzero(ground_mask)[0]
    return ground_indices


def vis_selected_indices(scene_pcd, indices):
    scene_pcd_copy = copy.deepcopy(scene_pcd)
    color = np.asarray(scene_pcd_copy.colors)
    color[indices] = (1, 0, 0)
    vis_pcd(scene_pcd_copy)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, default="~/t7/ScanNet/aligned_scans")
    parser.add_argument("-v", "--video", type=str, default="scene0645_00")
    parser.add_argument("--detic_exp", default="imagenet21k-0.3")
    # parser.add_argument("-d", "--dataset", type=str, default="~/t7/custom_tabletop")
    # parser.add_argument("-v", "--video", type=str, default="recording")
    # parser.add_argument("--detic_exp", default="lvis-0.3")
    parser.add_argument("--iou_thresh", type=float, default=0.25)
    parser.add_argument("--recall_thresh", type=float, default=0.5)
    parser.add_argument("--depth_thresh", type=float, default=0.04)
    parser.add_argument("--visibility_thresh", type=float, default=0.2)
    parser.add_argument("--feature_similarity_thresh", type=float, default=0.75)
    parser.add_argument("--size_thresh", type=int, default=50)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--vocab_feature_file", default="src/scannet200.npy")
    parser.add_argument("--output_file", default="proposed_fusion_detic.pkl")
    parser.add_argument("--interval", type=int, default=300)
    parser.add_argument("--sam", default=False, action=argparse.BooleanOptionalAction)  # python 3.9+
    parser.add_argument("--tqdm", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--vis", default=True, action=argparse.BooleanOptionalAction)
    args = parser.parse_args()
    print(args)

    git_repo = Path(git.Repo(search_parent_directories=True).working_tree_dir)
    sys.path.append(str(git_repo))

    dataset = Path(args.dataset).expanduser()
    video_path = dataset / args.video
    if "ScanNet" in args.dataset:
        scene_pcd_path = video_path / f"{args.video}_vh_clean_2.ply"
        scene_pcd = o3d.io.read_point_cloud(str(scene_pcd_path))
        vocabs = importlib.import_module("src.vocabs").vocabs['scannet200']
        vocab_features = torch.from_numpy(np.load(git_repo / args.vocab_feature_file)).to(args.device)
        metadata_path = os.path.join(video_path, f"{args.video}.txt")
        ground_indices = get_scannet_ground_indices(metadata_path, scene_pcd, thresh=0.025)
        if args.vis:
            vis_selected_indices(scene_pcd, ground_indices)
    elif "ycb_video" in args.dataset:
        scene_pcd_path = video_path / "scan-0.005.pcd"
        scene_pcd = o3d.io.read_point_cloud(str(scene_pcd_path))
        vocabs = importlib.import_module("src.vocabs").vocabs['lvis']
        vocab_features = torch.from_numpy(np.load(git_repo / args.vocab_feature_file)).to(args.device)
        ground_indices = None
    else:  # custom data
        scene_pcd_path = glob.glob(str(video_path / "scan-*.pcd"))[0]
        scene_pcd = o3d.io.read_point_cloud(str(scene_pcd_path))
        ground_indices = None
        vocabs = importlib.import_module("src.vocabs").vocabs['lvis']
        vocab_features = torch.from_numpy(np.load(git_repo / args.vocab_feature_file)).to(args.device)
        if args.vis:
            vis_selected_indices(scene_pcd, ground_indices)
    print(f"{scene_pcd_path = }")

    instance_pt_count, instance_features, instance_detections = instance_fusion(
        video_path, args.detic_exp, scene_pcd,
        iou_thresh=args.iou_thresh,
        recall_thresh=args.recall_thresh,
        feature_similarity_thresh=args.feature_similarity_thresh,
        depth_thresh=args.depth_thresh,
        size_thresh=args.size_thresh,
        interval=args.interval,
        visibility_thresh=args.visibility_thresh,
        stride=args.stride,
        ground_indices=ground_indices,
        use_sam=args.sam,
        disable_tqdm=not args.tqdm,
        device=args.device
    )

    scene_graph = build_scene_graph(scene_pcd, instance_pt_count, instance_features, instance_detections,
                                    vocabs=vocabs, vocab_features=vocab_features)

    output_folder = video_path / 'detic_output' / args.detic_exp / "predictions"
    os.makedirs(output_folder, exist_ok=True)
    if args.sam:
        output_path = output_folder / args.output_file.replace("detic", "sam")
    else:
        output_path = output_folder / args.output_file
    with open(output_path, 'wb') as fp:
        pickle.dump(scene_graph, fp)
        print(f"Saved to {output_path}")

    if args.vis:
        visualize_scene_graph(scene_pcd, scene_graph, show_center=True)


if __name__ == "__main__":
    main()

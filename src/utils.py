import os
import copy
from typing import Union, Dict
import pickle

import numpy as np
import numpy.linalg as la
import cv2
import open3d as o3d
import pandas as pd
import torch
import pycocotools.mask
from numba import njit
from detectron2.structures import Instances
from detectron2.utils.visualizer import Visualizer, ColorMode, BoxMode, _create_text_labels, GenericMask


class CustomVisualizer(Visualizer):

    def __init__(self, img_rgb, metadata=None, scale=1.0, instance_mode=ColorMode.IMAGE):
        super().__init__(img_rgb, metadata, scale, instance_mode)

    def draw_instance_predictions(self, predictions):
        """
        Draw instance-level prediction results on an image.

        Args:
            predictions (Instances): the output of an instance detection/segmentation
                model. Following fields will be used to draw:
                "pred_boxes", "pred_classes", "scores", "pred_masks" (or "pred_masks_rle").

        Returns:
            output (VisImage): image object with visualizations.
        """
        boxes = predictions.pred_boxes if predictions.has("pred_boxes") else None
        scores = predictions.scores if predictions.has("scores") else None
        classes = predictions.pred_classes.tolist() if predictions.has("pred_classes") else None
        labels = _create_text_labels(classes, scores, self.metadata.get("thing_classes", None))
        keypoints = predictions.pred_keypoints if predictions.has("pred_keypoints") else None

        if predictions.has("pred_masks"):
            masks = np.asarray(predictions.pred_masks)
            masks = [GenericMask(x, self.output.height, self.output.width) for x in masks]
        else:
            masks = None

        if self._instance_mode == ColorMode.SEGMENTATION and self.metadata.get("thing_colors"):
            colors = [
                [x / 255 for x in self.metadata.thing_colors[c]] for c in classes
            ]
            alpha = 0.8
        else:
            colors = None
            alpha = 0.5

        if self._instance_mode == ColorMode.IMAGE_BW:
            self.output.reset_image(
                self._create_grayscale_image(
                    (predictions.pred_masks.any(dim=0) > 0).numpy()
                    if predictions.has("pred_masks")
                    else None
                )
            )
            alpha = 0.3

        self.overlay_instances(
            masks=masks,
            boxes=boxes,
            labels=labels,
            keypoints=keypoints,
            assigned_colors=colors,
            alpha=alpha,
        )
        return self.output

    def draw_dataset_dict(self, dic):
        """
        Draw annotations/segmentaions in Detectron2 Dataset format.

        Args:
            dic (dict): annotation/segmentation data of one image, in Detectron2 Dataset format.

        Returns:
            output (VisImage): image object with visualizations.
        """
        annos = dic.get("annotations", None)
        if annos:
            if "segmentation" in annos[0]:
                masks = [x["segmentation"] for x in annos]
            else:
                masks = None
            if "keypoints" in annos[0]:
                keypts = [x["keypoints"] for x in annos]
                keypts = np.array(keypts).reshape(len(annos), -1, 3)
            else:
                keypts = None

            boxes = [
                BoxMode.convert(x["bbox"], x["bbox_mode"], BoxMode.XYXY_ABS)
                if len(x["bbox"]) == 4
                else x["bbox"]
                for x in annos
            ]

            colors = None
            category_ids = [x["category_id"] for x in annos]
            if self._instance_mode == ColorMode.SEGMENTATION and self.metadata.get("thing_colors"):
                colors = [
                    [x / 255 for x in self.metadata.thing_colors[c]]  # DO NOT JITTER !!!
                    for c in category_ids
                ]
            names = self.metadata.get("thing_classes", None)
            if names is not None:
                labels = [names[i] for i in category_ids]
            else:
                labels = [str(i) for i in category_ids]
            self.overlay_instances(
                labels=labels, boxes=boxes, masks=masks, keypoints=keypts, assigned_colors=colors
            )
        return self.output

    def _change_color_brightness(self, color, brightness_factor):
        modified_color = super()._change_color_brightness(color, brightness_factor)
        modified_color = tuple(color if color < 1 else 1 for color in modified_color)
        return modified_color


def imread(filepath: str, mode='rgb'):
    assert mode in ['rgb', 'bgr', 'unchanged']
    if mode == 'rgb':
        return cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2RGB)
    if mode == 'bgr':
        return cv2.imread(filepath)
    if mode == 'unchanged':
        return cv2.imread(filepath, cv2.IMREAD_UNCHANGED)


def mask_to_rle(mask) -> Dict:
    if torch.is_tensor(mask):
        mask = mask.detach().cpu().numpy()
    assert isinstance(mask, np.ndarray)
    rle = pycocotools.mask.encode(np.asfortranarray(mask))
    rle['counts'] = rle['counts'].decode('utf-8')
    return rle


def read_detectron_instances(filepath: Union[str, os.PathLike], rle_to_mask=True) -> Instances:
    with open(filepath, 'rb') as fp:
        instances = pickle.load(fp)
        if rle_to_mask:
            if instances.pred_masks_rle:
                pred_masks = np.stack([pycocotools.mask.decode(rle) for rle in instances.pred_masks_rle])
                instances.pred_masks = torch.from_numpy(pred_masks).to(torch.bool)  # (M, H, W)
            else:
                instances.pred_masks = torch.empty((0, 0, 0), dtype=torch.bool)
    return instances


@njit
def compute_projected_pts(pts, cam_intr):
    N = pts.shape[0]
    projected_pts = np.empty((N, 2), dtype=np.int64)
    fx, fy = cam_intr[0, 0], cam_intr[1, 1]
    cx, cy = cam_intr[0, 2], cam_intr[1, 2]
    for i in range(pts.shape[0]):
        z = pts[i, 2]
        x = int(np.round(fx * pts[i, 0] / z + cx))
        y = int(np.round(fy * pts[i, 1] / z + cy))
        projected_pts[i, 0] = x
        projected_pts[i, 1] = y
    return projected_pts


@njit
def compute_intersection_of_index_arrays(arr1, arr2):
    """
    arr1 and arr2 must be sorted in advance!
    """
    N, M = arr1.shape[0], arr2.shape[0]
    if N == 0 or M == 0:
        return 0
    ptr1, ptr2 = 0, 0
    inlier = 0
    while ptr1 < N and ptr2 < M:
        if arr1[ptr1] < arr2[ptr2]:
            ptr1 += 1
        elif arr1[ptr1] > arr2[ptr2]:
            ptr2 += 1
        else:
            inlier += 1
            ptr1 += 1
            ptr2 += 1
    return inlier


@njit
def compute_visibility_mask(pts, projected_pts, depth_im, depth_thresh=0.005):
    im_h, im_w = depth_im.shape
    visibility_mask = np.zeros(projected_pts.shape[0]).astype(np.bool8)
    for i in range(projected_pts.shape[0]):
        x, y = projected_pts[i]
        z = pts[i, 2]
        if x < 0 or x >= im_w or y < 0 or y >= im_h:
            continue
        if depth_im[y, x] == 0:
            continue
        if np.abs(z - depth_im[y, x]) < depth_thresh:
            visibility_mask[i] = True
    return visibility_mask


@njit
def compute_visible_indices(pts, projected_pts, depth_im, depth_thresh=0.005):
    im_h, im_w = depth_im.shape
    visibility_mask = np.zeros(projected_pts.shape[0]).astype(np.bool8)
    for i in range(projected_pts.shape[0]):
        x, y = projected_pts[i]
        z = pts[i, 2]
        if x < 0 or x >= im_w or y < 0 or y >= im_h:
            continue
        if depth_im[y, x] == 0:
            continue
        if np.abs(z - depth_im[y, x]) < depth_thresh:
            visibility_mask[i] = True
    visible_indices = visibility_mask.nonzero()[0]
    return visible_indices


def create_pcd(depth_im: np.ndarray,
               cam_intr: np.ndarray,
               color_im: np.ndarray = None,
               cam_extr: np.ndarray = np.eye(4)):
    intrinsic_o3d = o3d.camera.PinholeCameraIntrinsic()
    intrinsic_o3d.intrinsic_matrix = cam_intr
    depth_im_o3d = o3d.geometry.Image(depth_im)
    if color_im is not None:
        color_im_o3d = o3d.geometry.Image(color_im)
        rgbd = o3d.geometry.RGBDImage().create_from_color_and_depth(color_im_o3d, depth_im_o3d, depth_scale=1,
                                                                    convert_rgb_to_intensity=False)
        pcd = o3d.geometry.PointCloud().create_from_rgbd_image(rgbd, intrinsic_o3d, extrinsic=cam_extr)
    else:
        pcd = o3d.geometry.PointCloud().create_from_depth_image(depth_im_o3d, intrinsic_o3d, extrinsic=cam_extr,
                                                                depth_scale=1)
    return pcd


def vis_pcd(pcd, cam_pose=None, coord_frame_size=0.2):
    if not isinstance(pcd, list):
        pcd = [pcd]
    pcd_frame = o3d.geometry.TriangleMesh().create_coordinate_frame(size=coord_frame_size)
    if cam_pose is not None:
        cam_frame = o3d.geometry.TriangleMesh().create_coordinate_frame(size=coord_frame_size)
        cam_frame.transform(cam_pose)
        o3d.visualization.draw_geometries([*pcd, pcd_frame, cam_frame])
    else:
        o3d.visualization.draw_geometries([*pcd, pcd_frame])


@njit
def project_xyz_and_labels_to_mask(im_h, im_w, xyz, labels, cam_intr, bg_label=0):
    seg_mask = np.zeros((im_h, im_w), dtype=np.uint16)
    fx, fy = cam_intr[0, 0], cam_intr[1, 1]
    cx, cy = cam_intr[0, 2], cam_intr[1, 2]
    for i in range(xyz.shape[0]):
        x = int(np.round((xyz[i, 0] * fx / xyz[i, 2]) + cx))
        y = int(np.round((xyz[i, 1] * fy / xyz[i, 2]) + cy))
        if 0 <= x < im_w and 0 <= y < im_h:
            if labels[i] != bg_label:
                seg_mask[y, x] = labels[i]
    return seg_mask


@njit
def generate_colors_for_pts(pt_classes, color_map, unknown_class=-1):
    N = pt_classes.shape[0]
    result = np.zeros((N, 3), dtype=np.float64)
    for i in range(N):
        pt_class = pt_classes[i]
        if pt_class == unknown_class:
            result[i] = 0
        else:
            result[i] = color_map[pt_class]
    return result


def visualize_instance_pcd(pcd, instance_ids, color_map=None, unknown_class=-1):
    if color_map is None:
        unique_ids = np.unique(instance_ids)
        max_id = unique_ids[-1]
        color_map = np.random.uniform(0, 1, size=(max_id + 1, 3))
    colors = generate_colors_for_pts(instance_ids, color_map, unknown_class=unknown_class)
    pcd_vis = copy.deepcopy(pcd)
    pcd_vis.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([pcd_vis])
    return pcd_vis, color_map


def write_3d_output(pcd, labels, output_path):
    df = pd.DataFrame()
    df[['x', 'y', 'z']] = np.asarray(pcd.points, dtype=np.float32)
    df[['r', 'g', 'b']] = np.round(np.asarray(pcd.colors) * 255).astype(np.uint8)
    df[['nx', 'ny', 'nz']] = np.asarray(pcd.normals, dtype=np.float32)
    df['labels'] = labels
    df.to_csv(output_path, index=False)


def read_pcd_from_csv(path):
    df = pd.read_csv(path)
    labels = df['labels'].to_numpy()
    pts = df[['x', 'y', 'z']].to_numpy()
    colors = df[['r', 'g', 'b']].to_numpy()
    normals = df[['nx', 'ny', 'nz']].to_numpy()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(colors / 255)
    pcd.normals = o3d.utility.Vector3dVector(normals)
    return pcd, labels


def multiscale_icp(src: o3d.geometry.PointCloud,
                   tgt: o3d.geometry.PointCloud,
                   voxel_size_list: list,
                   max_iter_list: list,
                   init: np.ndarray = np.eye(4),
                   inverse: bool = False):

    if len(src.points) > len(tgt.points):
        return multiscale_icp(tgt, src, voxel_size_list, max_iter_list, init=la.inv(init), inverse=True)

    reg = o3d.pipelines.registration
    transformation = init.astype(np.float32)
    result_icp = None
    for i, (voxel_size, max_iter) in enumerate(zip(voxel_size_list, max_iter_list)):
        src_down = src.voxel_down_sample(voxel_size)
        tgt_down = tgt.voxel_down_sample(voxel_size)

        src_down.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(knn=30))
        tgt_down.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(knn=30))

        result_icp = reg.registration_icp(
            src_down, tgt_down, max_correspondence_distance=voxel_size*3,
            init=transformation,
            estimation_method=reg.TransformationEstimationPointToPlane(),
            criteria=reg.ICPConvergenceCriteria(max_iteration=max_iter)
        )
        transformation = result_icp.transformation

    if inverse and result_icp is not None:
        result_icp.transformation = la.inv(result_icp.transformation)
    return result_icp


def generate_coordinate_frame(T, scale=0.05):
    mesh = o3d.geometry.TriangleMesh().create_coordinate_frame()
    mesh.scale(scale, center=np.array([0, 0, 0]))
    return mesh.transform(T)


def plane_detection_o3d(pcd: o3d.geometry.PointCloud,
                        inlier_thresh: float,
                        max_iterations: int = 1000,
                        visualize: bool = False,
                        in_cam_frame: bool = True):
    # http://www.open3d.org/docs/release/tutorial/geometry/pointcloud.html#Plane-segmentation
    plane_model, inliers = pcd.segment_plane(distance_threshold=inlier_thresh,
                                             ransac_n=3,
                                             num_iterations=max_iterations)
    [a, b, c, d] = plane_model  # ax + by + cz + d = 0
    inlier_cloud = pcd.select_by_index(inliers)
    inlier_cloud.paint_uniform_color([1, 0, 0])
    outlier_cloud = pcd.select_by_index(inliers, invert=True)
    max_inlier_ratio = len(inliers) / len(np.asarray(pcd.points))

    # sample the inlier point that is closest to the camera origin as the world origin
    inlier_pts = np.asarray(inlier_cloud.points)
    squared_distances = np.sum(inlier_pts ** 2, axis=1)
    closest_index = np.argmin(squared_distances)
    x, y, z = inlier_pts[closest_index]
    origin = np.array([x, y, (-d - a * x - b * y) / (c + 1e-12)])
    plane_normal = np.array([a, b, c])
    plane_normal /= np.linalg.norm(plane_normal)

    if in_cam_frame:
        if plane_normal @ origin > 0:
            plane_normal *= -1
    elif plane_normal[2] < 0:
        plane_normal *= -1

    # randomly sample x_dir and y_dir given plane normal as z_dir
    x_dir = np.array([-plane_normal[2], 0, plane_normal[0]])
    x_dir /= la.norm(x_dir)
    y_dir = np.cross(plane_normal, x_dir)
    plane_frame = np.eye(4)
    plane_frame[:3, 0] = x_dir
    plane_frame[:3, 1] = y_dir
    plane_frame[:3, 2] = plane_normal
    plane_frame[:3, 3] = origin

    if visualize:
        plane_frame_vis = generate_coordinate_frame(plane_frame, scale=0.05)
        cam_frame_vis = generate_coordinate_frame(np.eye(4), scale=0.05)
        o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud, plane_frame_vis, cam_frame_vis])

    return plane_frame, max_inlier_ratio

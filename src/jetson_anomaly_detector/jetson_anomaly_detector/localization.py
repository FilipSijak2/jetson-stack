from __future__ import annotations

import math
from typing import Optional, Sequence

import cv2
import numpy as np

from .models import (
    BoundingBox3D,
    CameraIntrinsics,
    DistanceEstimate,
    LaserScan,
    ObjectPoseMap,
    RobotPoseMap,
)


def normalize_angle(angle_rad: float) -> float:
    return math.atan2(math.sin(angle_rad), math.cos(angle_rad))


def bbox_bearing_offset_rad(
    bbox_xyxy: Sequence[int],
    image_width: int,
    camera_horizontal_fov_deg: float,
    camera_yaw_offset_deg: float = 0.0,
) -> float:
    if image_width <= 0:
        return math.radians(camera_yaw_offset_deg)
    center_x = (float(bbox_xyxy[0]) + float(bbox_xyxy[2])) * 0.5
    normalized = (center_x / float(image_width)) - 0.5
    # Image x grows to the right, while ROS positive yaw grows to the left.
    return math.radians(camera_yaw_offset_deg) - normalized * math.radians(camera_horizontal_fov_deg)


def bbox_bearing_intrinsics_rad(
    bbox_xyxy: Sequence[int],
    image_width: int,
    intrinsics: CameraIntrinsics,
    camera_yaw_offset_deg: float = 0.0,
) -> float:
    if image_width <= 0 or intrinsics.width <= 0 or intrinsics.fx <= 0.0:
        return math.radians(camera_yaw_offset_deg)
    scale_x = float(image_width) / float(intrinsics.width)
    fx = intrinsics.fx * scale_x
    cx = intrinsics.cx * scale_x
    center_x = (float(bbox_xyxy[0]) + float(bbox_xyxy[2])) * 0.5
    # Camera x grows right; ROS positive yaw grows left.
    return math.radians(camera_yaw_offset_deg) - math.atan2(center_x - cx, fx)


def estimate_laser_distance_m(
    scan: LaserScan,
    bbox_xyxy: Sequence[int],
    image_width: int,
    camera_horizontal_fov_deg: float,
    camera_yaw_offset_deg: float,
    half_window_deg: float,
    min_distance_m: float,
    max_distance_m: float,
    camera_intrinsics: Optional[CameraIntrinsics] = None,
) -> Optional[float]:
    if scan.ranges.size == 0 or scan.angle_increment == 0.0:
        return None

    if camera_intrinsics is not None:
        target_bearing = bbox_bearing_intrinsics_rad(
            bbox_xyxy, image_width, camera_intrinsics, camera_yaw_offset_deg
        )
    else:
        target_bearing = bbox_bearing_offset_rad(
            bbox_xyxy,
            image_width,
            camera_horizontal_fov_deg,
            camera_yaw_offset_deg,
        )
    half_window = math.radians(max(0.1, half_window_deg))
    scan_max = scan.range_max if math.isfinite(scan.range_max) and scan.range_max > 0.0 else max_distance_m
    lower = max(min_distance_m, scan.range_min)
    upper = min(max_distance_m, scan_max)

    valid_ranges = []
    for index, distance in enumerate(scan.ranges):
        distance_f = float(distance)
        if not np.isfinite(distance_f) or distance_f < lower or distance_f > upper:
            continue
        angle = scan.angle_min + index * scan.angle_increment
        if abs(normalize_angle(angle - target_bearing)) <= half_window:
            valid_ranges.append(distance_f)

    if not valid_ranges:
        return None
    return min(valid_ranges)


def estimate_depth_distance_m(
    depth_image: np.ndarray,
    bbox_xyxy: Sequence[int],
    image_width: int,
    image_height: int,
    min_distance_m: float,
    max_distance_m: float,
    roi_scale: float,
    min_valid_pixels: int,
    percentile: float = 50.0,
) -> Optional[float]:
    if depth_image.size == 0 or image_width <= 0 or image_height <= 0:
        return None
    depth_height, depth_width = depth_image.shape[:2]
    if depth_width <= 0 or depth_height <= 0:
        return None

    x1, y1, x2, y2 = [float(value) for value in bbox_xyxy]
    if x2 <= x1 or y2 <= y1:
        return None

    scale = max(0.1, min(1.0, float(roi_scale)))
    center_x = (x1 + x2) * 0.5
    center_y = (y1 + y2) * 0.5
    half_width = (x2 - x1) * scale * 0.5
    half_height = (y2 - y1) * scale * 0.5

    sx = float(depth_width) / float(image_width)
    sy = float(depth_height) / float(image_height)
    dx1 = max(0, int(math.floor((center_x - half_width) * sx)))
    dy1 = max(0, int(math.floor((center_y - half_height) * sy)))
    dx2 = min(depth_width, int(math.ceil((center_x + half_width) * sx)))
    dy2 = min(depth_height, int(math.ceil((center_y + half_height) * sy)))
    if dx2 <= dx1 or dy2 <= dy1:
        return None

    roi = depth_image[dy1:dy2, dx1:dx2]
    valid = roi[np.isfinite(roi)]
    valid = valid[(valid >= min_distance_m) & (valid <= max_distance_m)]
    if valid.size < max(1, int(min_valid_pixels)):
        return None

    return float(np.percentile(valid, max(0.0, min(100.0, float(percentile)))))


def estimate_depth_measurement(
    depth_image: np.ndarray,
    bbox_xyxy: Sequence[int],
    image_width: int,
    image_height: int,
    min_distance_m: float,
    max_distance_m: float,
    roi_scale: float,
    min_valid_pixels: int,
    percentile: float = 50.0,
    age_s: Optional[float] = None,
    object_mask: Optional[np.ndarray] = None,
    mask_erode_px: int = 0,
) -> Optional[DistanceEstimate]:
    if depth_image.size == 0 or image_width <= 0 or image_height <= 0:
        return None
    depth_height, depth_width = depth_image.shape[:2]
    x1, y1, x2, y2 = [float(value) for value in bbox_xyxy[:4]]
    center_x = (x1 + x2) * 0.5
    center_y = (y1 + y2) * 0.5
    half_width = max(0.5, (x2 - x1) * max(0.1, min(1.0, roi_scale)) * 0.5)
    half_height = max(0.5, (y2 - y1) * max(0.1, min(1.0, roi_scale)) * 0.5)
    sx = float(depth_width) / float(image_width)
    sy = float(depth_height) / float(image_height)
    dx1 = max(0, int(math.floor((center_x - half_width) * sx)))
    dy1 = max(0, int(math.floor((center_y - half_height) * sy)))
    dx2 = min(depth_width, int(math.ceil((center_x + half_width) * sx)))
    dy2 = min(depth_height, int(math.ceil((center_y + half_height) * sy)))
    if dx2 <= dx1 or dy2 <= dy1:
        return None

    roi = depth_image[dy1:dy2, dx1:dx2]
    selection = np.isfinite(roi)
    if object_mask is not None and object_mask.size > 0:
        mask = np.asarray(object_mask, dtype=np.uint8)
        if mask.shape != (depth_height, depth_width):
            mask = cv2.resize(
                mask,
                (depth_width, depth_height),
                interpolation=cv2.INTER_NEAREST,
            )
        if mask_erode_px > 0:
            mask = cv2.erode(
                mask,
                np.ones((3, 3), dtype=np.uint8),
                iterations=int(mask_erode_px),
            )
        selection &= mask[dy1:dy2, dx1:dx2].astype(bool)
    valid = roi[selection]
    valid = valid[(valid >= min_distance_m) & (valid <= max_distance_m)]
    if valid.size < max(1, int(min_valid_pixels)):
        return None

    distance = float(np.percentile(valid, max(0.0, min(100.0, float(percentile)))))
    median = float(np.median(valid))
    robust_sigma = 1.4826 * float(np.median(np.abs(valid - median)))
    return DistanceEstimate(
        distance_m=distance,
        source="depth",
        uncertainty_m=max(0.005, robust_sigma),
        valid_sample_count=int(valid.size),
        age_s=age_s,
    )


def estimate_object_pose_map(
    robot_pose: RobotPoseMap,
    bbox_xyxy: Sequence[int],
    image_width: int,
    default_distance_m: float,
    camera_horizontal_fov_deg: float,
    camera_yaw_offset_deg: float = 0.0,
    measured_distance_m: Optional[float] = None,
    camera_intrinsics: Optional[CameraIntrinsics] = None,
    measured_distance_is_axial: bool = False,
) -> ObjectPoseMap:
    if camera_intrinsics is not None:
        bearing_offset = bbox_bearing_intrinsics_rad(
            bbox_xyxy, image_width, camera_intrinsics, camera_yaw_offset_deg
        )
    else:
        bearing_offset = bbox_bearing_offset_rad(
            bbox_xyxy,
            image_width,
            camera_horizontal_fov_deg,
            camera_yaw_offset_deg,
        )
    bearing = robot_pose.yaw + bearing_offset
    distance = measured_distance_m if measured_distance_m is not None else default_distance_m
    if measured_distance_is_axial:
        cosine = max(0.1, abs(math.cos(bearing_offset - math.radians(camera_yaw_offset_deg))))
        distance = float(distance) / cosine
    distance = max(0.05, float(distance))
    return ObjectPoseMap(
        x=robot_pose.x + distance * math.cos(bearing),
        y=robot_pose.y + distance * math.sin(bearing),
        z=0.0,
    )


def estimate_3d_bounds_camera(
    depth_image: np.ndarray,
    bbox_xyxy: Sequence[int],
    image_width: int,
    image_height: int,
    intrinsics: CameraIntrinsics,
    object_mask: Optional[np.ndarray],
    min_distance_m: float,
    max_distance_m: float,
    min_valid_points: int,
    lower_percentile: float = 5.0,
    upper_percentile: float = 95.0,
    mask_erode_px: int = 1,
    sample_stride: int = 2,
    minimum_thickness_m: float = 0.05,
) -> Optional[BoundingBox3D]:
    if (
        depth_image.size == 0
        or image_width <= 0
        or image_height <= 0
        or intrinsics.fx <= 0.0
        or intrinsics.fy <= 0.0
        or intrinsics.width <= 0
        or intrinsics.height <= 0
    ):
        return None

    depth_height, depth_width = depth_image.shape[:2]
    sx = float(depth_width) / float(image_width)
    sy = float(depth_height) / float(image_height)
    x1, y1, x2, y2 = [float(value) for value in bbox_xyxy[:4]]
    dx1 = max(0, int(math.floor(x1 * sx)))
    dy1 = max(0, int(math.floor(y1 * sy)))
    dx2 = min(depth_width, int(math.ceil(x2 * sx)))
    dy2 = min(depth_height, int(math.ceil(y2 * sy)))
    if dx2 <= dx1 or dy2 <= dy1:
        return None

    selection = np.zeros((depth_height, depth_width), dtype=np.uint8)
    if object_mask is not None and object_mask.size > 0:
        mask = np.asarray(object_mask, dtype=np.uint8)
        if mask.shape != (depth_height, depth_width):
            mask = cv2.resize(
                mask,
                (depth_width, depth_height),
                interpolation=cv2.INTER_NEAREST,
            )
        selection = mask
        if mask_erode_px > 0:
            selection = cv2.erode(
                selection,
                np.ones((3, 3), dtype=np.uint8),
                iterations=int(mask_erode_px),
            )
    else:
        selection[dy1:dy2, dx1:dx2] = 1

    roi_gate = np.zeros_like(selection)
    roi_gate[dy1:dy2, dx1:dx2] = 1
    selection &= roi_gate
    stride = max(1, int(sample_stride))
    if stride > 1:
        sampled = np.zeros_like(selection)
        sampled[::stride, ::stride] = selection[::stride, ::stride]
        selection = sampled

    valid = (
        selection.astype(bool)
        & np.isfinite(depth_image)
        & (depth_image >= min_distance_m)
        & (depth_image <= max_distance_m)
    )
    if int(np.count_nonzero(valid)) < max(1, int(min_valid_points)):
        return None

    initial_depths = depth_image[valid].astype(np.float32)
    median_depth = float(np.median(initial_depths))
    robust_sigma = 1.4826 * float(
        np.median(np.abs(initial_depths - median_depth))
    )
    depth_band = max(float(minimum_thickness_m), 3.0 * robust_sigma)
    valid &= np.abs(depth_image - median_depth) <= depth_band
    point_count = int(np.count_nonzero(valid))
    if point_count < max(1, int(min_valid_points)):
        return None

    vs, us = np.nonzero(valid)
    zs = depth_image[valid].astype(np.float32)
    intrinsics_sx = float(depth_width) / float(intrinsics.width)
    intrinsics_sy = float(depth_height) / float(intrinsics.height)
    fx = intrinsics.fx * intrinsics_sx
    fy = intrinsics.fy * intrinsics_sy
    cx = intrinsics.cx * intrinsics_sx
    cy = intrinsics.cy * intrinsics_sy
    xs = (us.astype(np.float32) - cx) * zs / fx
    ys = (vs.astype(np.float32) - cy) * zs / fy

    low = max(0.0, min(49.0, float(lower_percentile)))
    high = max(51.0, min(100.0, float(upper_percentile)))
    x_low, x_high = np.percentile(xs, [low, high])
    y_low, y_high = np.percentile(ys, [low, high])
    z_low, z_high = np.percentile(zs, [low, high])
    size_x = max(0.01, float(x_high - x_low))
    size_y = max(0.01, float(y_high - y_low))
    size_z = max(float(minimum_thickness_m), float(z_high - z_low))
    return BoundingBox3D(
        center_x=float((x_low + x_high) * 0.5),
        center_y=float((y_low + y_high) * 0.5),
        center_z=float((z_low + z_high) * 0.5),
        size_x=size_x,
        size_y=size_y,
        size_z=size_z,
        valid_point_count=point_count,
    )


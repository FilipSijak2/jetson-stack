from __future__ import annotations

import math
from typing import Optional, Sequence

import numpy as np

from .models import LaserScan, ObjectPoseMap, RobotPoseMap


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


def estimate_laser_distance_m(
    scan: LaserScan,
    bbox_xyxy: Sequence[int],
    image_width: int,
    camera_horizontal_fov_deg: float,
    camera_yaw_offset_deg: float,
    half_window_deg: float,
    min_distance_m: float,
    max_distance_m: float,
) -> Optional[float]:
    if scan.ranges.size == 0 or scan.angle_increment == 0.0:
        return None

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


def estimate_object_pose_map(
    robot_pose: RobotPoseMap,
    bbox_xyxy: Sequence[int],
    image_width: int,
    default_distance_m: float,
    camera_horizontal_fov_deg: float,
    camera_yaw_offset_deg: float = 0.0,
    measured_distance_m: Optional[float] = None,
) -> ObjectPoseMap:
    bearing_offset = bbox_bearing_offset_rad(
        bbox_xyxy,
        image_width,
        camera_horizontal_fov_deg,
        camera_yaw_offset_deg,
    )
    bearing = robot_pose.yaw + bearing_offset
    distance = measured_distance_m if measured_distance_m is not None else default_distance_m
    distance = max(0.05, float(distance))
    return ObjectPoseMap(
        x=robot_pose.x + distance * math.cos(bearing),
        y=robot_pose.y + distance * math.sin(bearing),
        z=0.0,
    )


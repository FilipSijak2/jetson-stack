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


from __future__ import annotations

import math
from typing import Sequence

from .models import ObjectPoseMap, RobotPoseMap


def estimate_object_pose_map(
    robot_pose: RobotPoseMap,
    bbox_xyxy: Sequence[int],
    image_width: int,
    default_distance_m: float,
    camera_horizontal_fov_deg: float,
) -> ObjectPoseMap:
    if image_width <= 0:
        bearing_offset = 0.0
    else:
        center_x = (float(bbox_xyxy[0]) + float(bbox_xyxy[2])) * 0.5
        normalized = (center_x / float(image_width)) - 0.5
        bearing_offset = normalized * math.radians(camera_horizontal_fov_deg)

    bearing = robot_pose.yaw + bearing_offset
    distance = max(0.05, float(default_distance_m))
    return ObjectPoseMap(
        x=robot_pose.x + distance * math.cos(bearing),
        y=robot_pose.y + distance * math.sin(bearing),
        z=0.0,
    )


from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


@dataclass(frozen=True)
class Detection:
    label: str
    confidence: float
    bbox_xyxy: List[int]
    track_id: Optional[int] = None
    mask: Optional[np.ndarray] = field(default=None, compare=False, repr=False)


@dataclass(frozen=True)
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int


@dataclass(frozen=True)
class DistanceEstimate:
    distance_m: float
    source: str
    uncertainty_m: Optional[float] = None
    valid_sample_count: int = 0
    age_s: Optional[float] = None
    axial_depth_m: Optional[float] = None


@dataclass(frozen=True)
class RobotPoseMap:
    x: float
    y: float
    yaw: float


@dataclass(frozen=True)
class ObjectPoseMap:
    x: float
    y: float
    z: float = 0.0


@dataclass(frozen=True)
class AnomalyClusterSummary:
    cluster_id: str
    label: str
    object_pose: ObjectPoseMap
    count: int


@dataclass(frozen=True)
class OccupancyGridMap:
    width: int
    height: int
    resolution: float
    origin_x: float
    origin_y: float
    frame_id: str
    data: np.ndarray


@dataclass(frozen=True)
class LaserScan:
    angle_min: float
    angle_max: float
    angle_increment: float
    range_min: float
    range_max: float
    ranges: np.ndarray


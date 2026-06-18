from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass(frozen=True)
class Detection:
    label: str
    confidence: float
    bbox_xyxy: List[int]


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
class OccupancyGridMap:
    width: int
    height: int
    resolution: float
    origin_x: float
    origin_y: float
    frame_id: str
    data: np.ndarray


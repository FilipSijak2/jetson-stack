from __future__ import annotations

from pathlib import Path
from typing import Tuple

import cv2
import numpy as np

from .models import ObjectPoseMap, OccupancyGridMap


def occupancy_grid_to_image(grid: OccupancyGridMap) -> np.ndarray:
    image = np.full((grid.height, grid.width), 127, dtype=np.uint8)
    image[grid.data == 0] = 255
    image[grid.data >= 50] = 0
    image[(grid.data > 0) & (grid.data < 50)] = 200
    return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)


def map_to_pixel(grid: OccupancyGridMap, x: float, y: float) -> Tuple[int, int]:
    px = int((x - grid.origin_x) / grid.resolution)
    py_map = int((y - grid.origin_y) / grid.resolution)
    py = grid.height - 1 - py_map
    return px, py


def draw_anomaly_snapshot(
    grid: OccupancyGridMap,
    object_pose: ObjectPoseMap,
    label: str,
) -> np.ndarray:
    image = occupancy_grid_to_image(grid)
    px, py = map_to_pixel(grid, object_pose.x, object_pose.y)
    px = max(0, min(grid.width - 1, px))
    py = max(0, min(grid.height - 1, py))

    color = (0, 0, 255)
    cv2.rectangle(image, (max(0, px - 8), max(0, py - 8)), (min(grid.width - 1, px + 8), min(grid.height - 1, py + 8)), color, 2)
    cv2.circle(image, (px, py), 5, color, -1)
    cv2.putText(
        image,
        f"ANOMALY: {label}",
        (min(max(px + 10, 5), max(5, grid.width - 220)), max(py - 10, 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
        cv2.LINE_AA,
    )
    return image


def save_map_snapshot(
    grid: OccupancyGridMap,
    object_pose: ObjectPoseMap,
    label: str,
    output_path: Path,
) -> np.ndarray:
    image = draw_anomaly_snapshot(grid, object_pose, label)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), image):
        raise ValueError(f"Failed to write map snapshot: {output_path}")
    return image


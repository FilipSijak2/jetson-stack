from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from .models import Detection, ObjectPoseMap, RobotPoseMap
from .ros_messages import iso_timestamp


def pose_to_dict(pose: RobotPoseMap) -> Dict[str, float]:
    return {"x": pose.x, "y": pose.y, "yaw": pose.yaw}


def object_pose_to_dict(pose: ObjectPoseMap) -> Dict[str, float]:
    return {"x": pose.x, "y": pose.y, "z": pose.z}


def build_event(
    event_id: str,
    detection: Detection,
    robot_pose: RobotPoseMap,
    object_pose: ObjectPoseMap,
    ttl_sec: float,
    original_image: Path,
    annotated_image: Path,
    map_snapshot: Optional[Path],
    event_log: Path,
) -> Dict[str, Any]:
    return {
        "id": event_id,
        "timestamp": iso_timestamp(),
        "label": detection.label,
        "type": "semantic_object_anomaly",
        "confidence": round(float(detection.confidence), 4),
        "status": "active",
        "ttl_sec": int(ttl_sec),
        "bbox_xyxy": [int(value) for value in detection.bbox_xyxy],
        "robot_pose_map": pose_to_dict(robot_pose),
        "object_pose_map": object_pose_to_dict(object_pose),
        "jetson_files": {
            "original_image": str(original_image),
            "annotated_image": str(annotated_image),
            "map_snapshot": str(map_snapshot) if map_snapshot else None,
            "event_log": str(event_log),
        },
    }


class EventJsonlWriter:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, event: Dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, sort_keys=True) + "\n")


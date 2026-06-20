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
    cluster_id: str,
    cluster_count: int,
    cluster_merge_radius_m: float,
    ttl_sec: float,
    original_image: Optional[Path],
    annotated_image: Optional[Path],
    map_snapshot: Optional[Path],
    daily_map_summary: Optional[Path],
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
        "cluster": {
            "id": cluster_id,
            "count": int(cluster_count),
            "merge_radius_m": round(float(cluster_merge_radius_m), 3),
        },
        "jetson_files": {
            "original_image": str(original_image) if original_image else None,
            "annotated_image": str(annotated_image) if annotated_image else None,
            "map_snapshot": str(map_snapshot) if map_snapshot else None,
            "daily_map_summary": str(daily_map_summary) if daily_map_summary else None,
            "event_log": str(event_log),
        },
    }


def build_readable_event(event: Dict[str, Any]) -> str:
    robot_pose = event.get("robot_pose_map") or {}
    object_pose = event.get("object_pose_map") or {}
    cluster = event.get("cluster") or {}
    files = event.get("jetson_files") or {}
    bbox = event.get("bbox_xyxy") or []

    lines = [
        f"Anomaly {event.get('id', '-')}",
        f"time: {event.get('timestamp', '-')}",
        f"label: {event.get('label', '-')}  confidence: {float(event.get('confidence', 0.0)):.2f}",
        f"cluster: {cluster.get('id', '-')}  count: {cluster.get('count', 1)}  radius_m: {cluster.get('merge_radius_m', '-')}",
        (
            "object_map: "
            f"x={float(object_pose.get('x', 0.0)):.2f} "
            f"y={float(object_pose.get('y', 0.0)):.2f} "
            f"z={float(object_pose.get('z', 0.0)):.2f}"
        ),
        (
            "robot_map: "
            f"x={float(robot_pose.get('x', 0.0)):.2f} "
            f"y={float(robot_pose.get('y', 0.0)):.2f} "
            f"yaw={float(robot_pose.get('yaw', 0.0)):.2f}"
        ),
        f"bbox_xyxy: {bbox}",
        f"daily_map: {files.get('daily_map_summary') or files.get('map_snapshot') or '-'}",
        f"event_log: {files.get('event_log', '-')}",
    ]
    return "\n".join(lines)


class EventJsonlWriter:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, event: Dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, sort_keys=True) + "\n")


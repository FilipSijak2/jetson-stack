from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List

from .models import ObjectPoseMap
from .ros_messages import ros_time_now


MARKER_ADD = 0
MARKER_DELETE = 2
MARKER_SPHERE = 2
MARKER_TEXT_VIEW_FACING = 9


@dataclass
class ActiveMarker:
    event_id: str
    marker_base_id: int
    label: str
    object_pose: ObjectPoseMap
    expires_at: float
    ttl_s: float


class MarkerManager:
    def __init__(self, frame_id: str = "map") -> None:
        self.frame_id = frame_id
        self.active: Dict[str, ActiveMarker] = {}
        self.delete_queue: List[int] = []

    def add(self, event_id: str, marker_base_id: int, label: str, object_pose: ObjectPoseMap, ttl_s: float) -> None:
        self.active[event_id] = ActiveMarker(
            event_id=event_id,
            marker_base_id=marker_base_id,
            label=label,
            object_pose=object_pose,
            expires_at=time.monotonic() + ttl_s,
            ttl_s=ttl_s,
        )

    def build_marker_array(self) -> Dict[str, Any]:
        self._expire_old_markers()
        markers = []
        stamp = ros_time_now()
        for marker in self.active.values():
            markers.append(self._object_marker(marker, stamp))
            markers.append(self._text_marker(marker, stamp))
        while self.delete_queue:
            marker_id = self.delete_queue.pop(0)
            markers.append(self._delete_marker(marker_id, stamp))
        return {"markers": markers}

    def _expire_old_markers(self) -> None:
        now = time.monotonic()
        expired = [event_id for event_id, marker in self.active.items() if marker.expires_at <= now]
        for event_id in expired:
            marker = self.active.pop(event_id)
            self.delete_queue.extend([marker.marker_base_id, marker.marker_base_id + 1])

    def _base(self, marker: ActiveMarker, marker_id: int, marker_type: int, stamp: Dict[str, int]) -> Dict[str, Any]:
        return {
            "header": {"stamp": stamp, "frame_id": self.frame_id},
            "ns": "jetson_anomalies",
            "id": marker_id,
            "type": marker_type,
            "action": MARKER_ADD,
            "pose": {
                "position": {
                    "x": marker.object_pose.x,
                    "y": marker.object_pose.y,
                    "z": marker.object_pose.z,
                },
                "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
            },
            "lifetime": {"sec": int(marker.ttl_s), "nanosec": int((marker.ttl_s % 1.0) * 1_000_000_000)},
        }

    def _object_marker(self, marker: ActiveMarker, stamp: Dict[str, int]) -> Dict[str, Any]:
        msg = self._base(marker, marker.marker_base_id, MARKER_SPHERE, stamp)
        msg.update(
            {
                "scale": {"x": 0.22, "y": 0.22, "z": 0.22},
                "color": {"r": 1.0, "g": 0.05, "b": 0.0, "a": 1.0},
            }
        )
        return msg

    def _text_marker(self, marker: ActiveMarker, stamp: Dict[str, int]) -> Dict[str, Any]:
        msg = self._base(marker, marker.marker_base_id + 1, MARKER_TEXT_VIEW_FACING, stamp)
        msg["pose"]["position"]["z"] = marker.object_pose.z + 0.45
        msg.update(
            {
                "scale": {"x": 0.0, "y": 0.0, "z": 0.25},
                "color": {"r": 1.0, "g": 0.1, "b": 0.0, "a": 1.0},
                "text": f"ANOMALY: {marker.label}",
            }
        )
        return msg

    def _delete_marker(self, marker_id: int, stamp: Dict[str, int]) -> Dict[str, Any]:
        return {
            "header": {"stamp": stamp, "frame_id": self.frame_id},
            "ns": "jetson_anomalies",
            "id": marker_id,
            "type": MARKER_SPHERE,
            "action": MARKER_DELETE,
            "pose": {
                "position": {"x": 0.0, "y": 0.0, "z": 0.0},
                "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
            },
            "scale": {"x": 0.0, "y": 0.0, "z": 0.0},
            "color": {"r": 0.0, "g": 0.0, "b": 0.0, "a": 0.0},
            "lifetime": {"sec": 0, "nanosec": 0},
        }


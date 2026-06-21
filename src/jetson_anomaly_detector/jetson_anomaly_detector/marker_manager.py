from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .models import AnomalyClusterSummary, ObjectPoseMap
from .ros_messages import ros_time_now


MARKER_ADD = 0
MARKER_DELETE = 2
MARKER_CUBE = 1
MARKER_TEXT_VIEW_FACING = 9
TEXT_MARKER_HEIGHT_M = 0.08
TEXT_MARKER_Z_OFFSET_M = 0.18
OBJECT_MARKER_SIZE_M = 0.20


@dataclass
class ActiveCluster:
    cluster_id: str
    marker_base_id: int
    label: str
    object_pose: ObjectPoseMap
    count: int
    expires_at: float
    ttl_s: float


class MarkerManager:
    def __init__(
        self,
        frame_id: str = "map",
        merge_radius_m: float = 0.20,
        object_marker_size_m: float = OBJECT_MARKER_SIZE_M,
        text_height_m: float = TEXT_MARKER_HEIGHT_M,
        text_z_offset_m: float = TEXT_MARKER_Z_OFFSET_M,
        text_show_count: bool = False,
    ) -> None:
        self.frame_id = frame_id
        self.merge_radius_m = merge_radius_m
        self.object_marker_size_m = max(0.05, float(object_marker_size_m))
        self.text_height_m = max(0.01, float(text_height_m))
        self.text_z_offset_m = max(0.0, float(text_z_offset_m))
        self.text_show_count = bool(text_show_count)
        self.active: Dict[str, ActiveCluster] = {}
        self.delete_queue: List[int] = []
        self.next_marker_base_id = 2
        self.next_cluster_index = 1

    def add_or_update(
        self,
        label: str,
        object_pose: ObjectPoseMap,
        observed_count: int,
        ttl_s: float,
    ) -> AnomalyClusterSummary:
        cluster = self._nearest_cluster(label, object_pose)
        if cluster is None:
            cluster = ActiveCluster(
                cluster_id=f"cluster_{self.next_cluster_index:05d}",
                marker_base_id=self.next_marker_base_id,
                label=label,
                object_pose=object_pose,
                count=max(1, observed_count),
                expires_at=time.monotonic() + ttl_s,
                ttl_s=ttl_s,
            )
            self.next_cluster_index += 1
            self.next_marker_base_id += 2
            self.active[cluster.cluster_id] = cluster
        else:
            cluster.object_pose = self._blend_pose(cluster.object_pose, object_pose)
            cluster.count = max(cluster.count, observed_count)
            cluster.expires_at = time.monotonic() + ttl_s
            cluster.ttl_s = ttl_s

        return self._summary(cluster)

    def build_marker_array(self) -> Dict[str, Any]:
        self._expire_old_markers()
        markers = []
        stamp = ros_time_now()
        for cluster in self.active.values():
            markers.append(self._object_marker(cluster, stamp))
            markers.append(self._text_marker(cluster, stamp))
        while self.delete_queue:
            marker_id = self.delete_queue.pop(0)
            markers.append(self._delete_marker(marker_id, stamp))
        return {"markers": markers}

    def summaries(self) -> List[AnomalyClusterSummary]:
        return [self._summary(cluster) for cluster in self.active.values()]

    def _nearest_cluster(self, label: str, object_pose: ObjectPoseMap) -> Optional[ActiveCluster]:
        nearest = None
        nearest_distance = float("inf")
        for cluster in self.active.values():
            if cluster.label != label:
                continue
            distance = math.hypot(cluster.object_pose.x - object_pose.x, cluster.object_pose.y - object_pose.y)
            if distance <= self.merge_radius_m and distance < nearest_distance:
                nearest = cluster
                nearest_distance = distance
        return nearest

    def _expire_old_markers(self) -> None:
        now = time.monotonic()
        expired = [cluster_id for cluster_id, cluster in self.active.items() if cluster.expires_at <= now]
        for cluster_id in expired:
            cluster = self.active.pop(cluster_id)
            self.delete_queue.extend([cluster.marker_base_id, cluster.marker_base_id + 1])

    def _base(self, cluster: ActiveCluster, marker_id: int, marker_type: int, stamp: Dict[str, int]) -> Dict[str, Any]:
        return {
            "header": {"stamp": stamp, "frame_id": self.frame_id},
            "ns": "jetson_anomaly_clusters",
            "id": marker_id,
            "type": marker_type,
            "action": MARKER_ADD,
            "pose": {
                "position": {
                    "x": cluster.object_pose.x,
                    "y": cluster.object_pose.y,
                    "z": cluster.object_pose.z,
                },
                "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
            },
            "lifetime": {"sec": int(cluster.ttl_s), "nanosec": int((cluster.ttl_s % 1.0) * 1_000_000_000)},
        }

    def _object_marker(self, cluster: ActiveCluster, stamp: Dict[str, int]) -> Dict[str, Any]:
        msg = self._base(cluster, cluster.marker_base_id, MARKER_CUBE, stamp)
        msg.update(
            {
                "scale": {"x": self.object_marker_size_m, "y": self.object_marker_size_m, "z": 0.08},
                "color": {"r": 1.0, "g": 0.05, "b": 0.0, "a": 0.75},
            }
        )
        return msg

    def _text_marker(self, cluster: ActiveCluster, stamp: Dict[str, int]) -> Dict[str, Any]:
        msg = self._base(cluster, cluster.marker_base_id + 1, MARKER_TEXT_VIEW_FACING, stamp)
        msg["pose"]["position"]["z"] = cluster.object_pose.z + self.text_z_offset_m
        msg.update(
            {
                "scale": {"x": 0.0, "y": 0.0, "z": self.text_height_m},
                "color": {"r": 1.0, "g": 0.1, "b": 0.0, "a": 1.0},
                "text": self._label_text(cluster),
            }
        )
        return msg

    def _delete_marker(self, marker_id: int, stamp: Dict[str, int]) -> Dict[str, Any]:
        return {
            "header": {"stamp": stamp, "frame_id": self.frame_id},
            "ns": "jetson_anomaly_clusters",
            "id": marker_id,
            "type": MARKER_CUBE,
            "action": MARKER_DELETE,
            "pose": {
                "position": {"x": 0.0, "y": 0.0, "z": 0.0},
                "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
            },
            "scale": {"x": 0.0, "y": 0.0, "z": 0.0},
            "color": {"r": 0.0, "g": 0.0, "b": 0.0, "a": 0.0},
            "lifetime": {"sec": 0, "nanosec": 0},
        }

    def _summary(self, cluster: ActiveCluster) -> AnomalyClusterSummary:
        return AnomalyClusterSummary(
            cluster_id=cluster.cluster_id,
            label=cluster.label,
            object_pose=cluster.object_pose,
            count=cluster.count,
        )

    def _blend_pose(self, current: ObjectPoseMap, new_pose: ObjectPoseMap) -> ObjectPoseMap:
        return ObjectPoseMap(
            x=(current.x + new_pose.x) * 0.5,
            y=(current.y + new_pose.y) * 0.5,
            z=(current.z + new_pose.z) * 0.5,
        )

    def _label_text(self, cluster: ActiveCluster) -> str:
        if not self.text_show_count or cluster.count <= 1:
            return cluster.label
        return f"{cluster.label} x{cluster.count}"

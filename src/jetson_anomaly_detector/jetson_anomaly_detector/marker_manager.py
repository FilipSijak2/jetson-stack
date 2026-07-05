from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .models import (
    AnomalyClusterSummary,
    BoundingBox3D,
    Detection,
    ObjectPoseMap,
    RobotPoseMap,
)
from .ros_messages import ros_time_now


MARKER_ADD = 0
MARKER_DELETE = 2
MARKER_DELETE_ALL = 3
MARKER_CUBE = 1
MARKER_LINE_STRIP = 4
MARKER_LINE_LIST = 5
MARKER_TEXT_VIEW_FACING = 9
TEXT_MARKER_HEIGHT_M = 0.08
TEXT_MARKER_Z_OFFSET_M = 0.18
OBJECT_MARKER_SIZE_M = 0.20


def build_detection_3d_marker_array(
    detections: Sequence[Tuple[Detection, BoundingBox3D]],
    frame_id: str,
    stamp: Optional[Dict[str, int]],
    ttl_s: float,
    line_width_m: float,
    text_enabled: bool = True,
    text_height_m: float = 0.06,
    text_show_label: bool = True,
    text_show_confidence: bool = True,
    text_show_distance: bool = True,
) -> Dict[str, Any]:
    markers: List[Dict[str, Any]] = []
    marker_stamp = stamp or ros_time_now()
    lifetime = {
        "sec": int(ttl_s),
        "nanosec": int((ttl_s % 1.0) * 1_000_000_000),
    }
    for index, (detection, bounds) in enumerate(detections):
        base_id = (
            max(0, int(detection.track_id)) * 2
            if detection.track_id is not None
            else 1_000_000 + index * 2
        )
        corners = _bounding_box_corners(bounds)
        edge_indices = (
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7),
        )
        points = [
            {"x": corners[point][0], "y": corners[point][1], "z": corners[point][2]}
            for edge in edge_indices
            for point in edge
        ]
        common = {
            "header": {"stamp": marker_stamp, "frame_id": frame_id},
            "ns": "jetson_anomaly_detections_3d",
            "action": MARKER_ADD,
            "lifetime": lifetime,
        }
        color = _track_color(detection.track_id)
        markers.append(
            {
                **common,
                "id": base_id,
                "type": MARKER_LINE_LIST,
                "pose": {
                    "position": {"x": 0.0, "y": 0.0, "z": 0.0},
                    "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
                },
                "scale": {"x": max(0.002, float(line_width_m)), "y": 0.0, "z": 0.0},
                "color": {**color, "a": 0.95},
                "points": points,
            }
        )
        if not text_enabled:
            continue
        track_text = (
            f" #{detection.track_id}" if detection.track_id is not None else ""
        )
        text_parts = []
        if text_show_label:
            text_parts.append(f"{detection.label}{track_text}")
        elif detection.track_id is not None:
            text_parts.append(f"#{detection.track_id}")
        if text_show_confidence:
            text_parts.append(f"{detection.confidence:.2f}")
        if text_show_distance:
            text_parts.append(f"{bounds.center_z:.2f} m")
        markers.append(
            {
                **common,
                "id": base_id + 1,
                "type": MARKER_TEXT_VIEW_FACING,
                "pose": {
                    "position": {
                        "x": bounds.center_x,
                        "y": bounds.center_y - bounds.size_y * 0.6,
                        "z": bounds.center_z,
                    },
                    "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
                },
                "scale": {"x": 0.0, "y": 0.0, "z": max(0.01, float(text_height_m))},
                "color": {**color, "a": 1.0},
                "text": " · ".join(text_parts),
            }
        )
    return {"markers": markers}


def _track_color(track_id: Optional[int]) -> Dict[str, float]:
    """Return a stable bright color so adjacent tracked bottles stay readable."""
    palette = (
        (0.10, 1.00, 0.20),
        (0.10, 0.75, 1.00),
        (1.00, 0.35, 0.10),
        (0.95, 0.20, 0.85),
        (0.95, 0.85, 0.10),
        (0.45, 0.35, 1.00),
    )
    index = 0 if track_id is None else abs(int(track_id)) % len(palette)
    red, green, blue = palette[index]
    return {"r": red, "g": green, "b": blue}


def _bounding_box_corners(bounds: BoundingBox3D) -> List[Tuple[float, float, float]]:
    hx, hy, hz = bounds.size_x * 0.5, bounds.size_y * 0.5, bounds.size_z * 0.5
    return [
        (bounds.center_x - hx, bounds.center_y - hy, bounds.center_z - hz),
        (bounds.center_x + hx, bounds.center_y - hy, bounds.center_z - hz),
        (bounds.center_x + hx, bounds.center_y + hy, bounds.center_z - hz),
        (bounds.center_x - hx, bounds.center_y + hy, bounds.center_z - hz),
        (bounds.center_x - hx, bounds.center_y - hy, bounds.center_z + hz),
        (bounds.center_x + hx, bounds.center_y - hy, bounds.center_z + hz),
        (bounds.center_x + hx, bounds.center_y + hy, bounds.center_z + hz),
        (bounds.center_x - hx, bounds.center_y + hy, bounds.center_z + hz),
    ]


@dataclass
class ActiveCluster:
    cluster_id: str
    marker_base_id: int
    label: str
    object_pose: ObjectPoseMap
    count: int
    expires_at: float
    ttl_s: float
    robot_pose: Optional[RobotPoseMap] = None
    uncertainty_m: Optional[float] = None
    track_id: Optional[int] = None
    ray_expires_at: float = 0.0


class MarkerManager:
    def __init__(
        self,
        frame_id: str = "map",
        merge_radius_m: float = 0.20,
        association_radius_m: Optional[float] = None,
        object_marker_size_m: float = OBJECT_MARKER_SIZE_M,
        text_height_m: float = TEXT_MARKER_HEIGHT_M,
        text_z_offset_m: float = TEXT_MARKER_Z_OFFSET_M,
        text_show_count: bool = False,
        text_compact: bool = True,
        tracked_object_min_separation_m: float = 0.01,
        track_reassociation_radius_m: Optional[float] = None,
        track_reassociation_ray_tolerance_m: float = 0.35,
        max_far_jump_m: float = 0.60,
        ray_enabled: bool = True,
        ray_ttl_s: float = 2.0,
        uncertainty_enabled: bool = True,
        uncertainty_sigma_scale: float = 2.0,
        uncertainty_min_radius_m: float = 0.05,
        uncertainty_max_radius_m: float = 1.0,
        auxiliary_line_width_m: float = 0.025,
    ) -> None:
        self.frame_id = frame_id
        self.merge_radius_m = max(0.01, float(merge_radius_m))
        self.association_radius_m = max(
            self.merge_radius_m,
            float(association_radius_m) if association_radius_m is not None else self.merge_radius_m,
        )
        self.object_marker_size_m = max(0.05, float(object_marker_size_m))
        self.text_height_m = max(0.01, float(text_height_m))
        self.text_z_offset_m = max(0.0, float(text_z_offset_m))
        self.text_show_count = bool(text_show_count)
        self.text_compact = bool(text_compact)
        self.tracked_object_min_separation_m = max(
            0.01, float(tracked_object_min_separation_m)
        )
        self.track_reassociation_radius_m = max(
            self.association_radius_m,
            float(track_reassociation_radius_m)
            if track_reassociation_radius_m is not None
            else self.association_radius_m,
        )
        self.track_reassociation_ray_tolerance_m = max(
            0.05, float(track_reassociation_ray_tolerance_m)
        )
        self.max_far_jump_m = max(0.05, float(max_far_jump_m))
        self.ray_enabled = bool(ray_enabled)
        self.ray_ttl_s = max(0.1, float(ray_ttl_s))
        self.uncertainty_enabled = bool(uncertainty_enabled)
        self.uncertainty_sigma_scale = max(0.0, float(uncertainty_sigma_scale))
        self.uncertainty_min_radius_m = max(0.0, float(uncertainty_min_radius_m))
        self.uncertainty_max_radius_m = max(
            self.uncertainty_min_radius_m, float(uncertainty_max_radius_m)
        )
        self.auxiliary_line_width_m = max(0.005, float(auxiliary_line_width_m))
        self.active: Dict[str, ActiveCluster] = {}
        self.delete_queue: List[int] = []
        self.marker_stride = 4
        self.next_marker_base_id = self.marker_stride
        self.next_cluster_index = 1

    def add_or_update(
        self,
        label: str,
        object_pose: ObjectPoseMap,
        observed_count: int,
        ttl_s: float,
        robot_pose: Optional[RobotPoseMap] = None,
        uncertainty_m: Optional[float] = None,
        track_id: Optional[int] = None,
        visible_track_ids: Optional[Sequence[int]] = None,
    ) -> AnomalyClusterSummary:
        visible_ids = set(visible_track_ids) if visible_track_ids is not None else None
        cluster = self._nearest_cluster(
            label, object_pose, track_id, visible_ids, robot_pose
        )
        if cluster is None:
            cluster = ActiveCluster(
                cluster_id=f"cluster_{self.next_cluster_index:05d}",
                marker_base_id=self.next_marker_base_id,
                label=label,
                object_pose=object_pose,
                count=max(1, observed_count),
                expires_at=time.monotonic() + ttl_s,
                ttl_s=ttl_s,
                robot_pose=robot_pose,
                uncertainty_m=uncertainty_m,
                track_id=track_id,
                ray_expires_at=time.monotonic() + self.ray_ttl_s,
            )
            self.next_cluster_index += 1
            self.next_marker_base_id += self.marker_stride
            self.active[cluster.cluster_id] = cluster
        else:
            if not self._is_farther_distance_outlier(
                cluster.object_pose, object_pose, robot_pose
            ):
                cluster.object_pose = self._blend_pose(
                    cluster.object_pose, object_pose, new_weight=0.20
                )
            cluster.count += max(1, observed_count)
            cluster.expires_at = time.monotonic() + ttl_s
            cluster.ttl_s = ttl_s
            cluster.robot_pose = robot_pose or cluster.robot_pose
            cluster.uncertainty_m = (
                uncertainty_m if uncertainty_m is not None else cluster.uncertainty_m
            )
            cluster.track_id = track_id if track_id is not None else cluster.track_id
            cluster.ray_expires_at = time.monotonic() + self.ray_ttl_s

        cluster = self._merge_overlapping_clusters(cluster, visible_ids)
        return self._summary(cluster)

    def build_marker_array(self) -> Dict[str, Any]:
        self._expire_old_markers()
        markers = []
        stamp = ros_time_now()
        now = time.monotonic()
        for cluster in self.active.values():
            markers.append(self._object_marker(cluster, stamp))
            markers.append(self._text_marker(cluster, stamp))
            if (
                self.ray_enabled
                and cluster.robot_pose is not None
                and cluster.ray_expires_at > now
            ):
                markers.append(self._ray_marker(cluster, stamp))
            if self.uncertainty_enabled and cluster.uncertainty_m is not None:
                markers.append(self._uncertainty_marker(cluster, stamp))
        while self.delete_queue:
            marker_id = self.delete_queue.pop(0)
            markers.append(self._delete_marker(marker_id, stamp))
        return {"markers": markers}

    def build_delete_all_marker_array(self) -> Dict[str, Any]:
        self.delete_queue.clear()
        return {"markers": [self._delete_all_marker(ros_time_now())]}

    def reset(self) -> None:
        self.active.clear()
        self.delete_queue.clear()

    def summaries(self) -> List[AnomalyClusterSummary]:
        return [self._summary(cluster) for cluster in self.active.values()]

    def _nearest_cluster(
        self,
        label: str,
        object_pose: ObjectPoseMap,
        track_id: Optional[int],
        visible_track_ids: Optional[set[int]],
        robot_pose: Optional[RobotPoseMap],
    ) -> Optional[ActiveCluster]:
        exact_track_matches = [
            cluster
            for cluster in self.active.values()
            if (
                cluster.label == label
                and track_id is not None
                and cluster.track_id == track_id
            )
        ]
        if exact_track_matches:
            return min(
                exact_track_matches,
                key=lambda cluster: math.hypot(
                    cluster.object_pose.x - object_pose.x,
                    cluster.object_pose.y - object_pose.y,
                ),
            )

        nearest = None
        nearest_distance = float("inf")
        for cluster in self.active.values():
            if cluster.label != label:
                continue
            distance = math.hypot(cluster.object_pose.x - object_pose.x, cluster.object_pose.y - object_pose.y)
            different_tracks = (
                track_id is not None
                and cluster.track_id is not None
                and track_id != cluster.track_id
            )
            if different_tracks and (
                visible_track_ids is None or cluster.track_id in visible_track_ids
            ):
                continue
            same_track = (
                track_id is not None and track_id == cluster.track_id
            )
            radius = (
                self.track_reassociation_radius_m
                if different_tracks or same_track
                else self.association_radius_m
            )
            ray_match = (
                different_tracks
                and visible_track_ids is not None
                and cluster.track_id not in visible_track_ids
                and self._ray_matches_cluster(
                    cluster.object_pose, object_pose, robot_pose
                )
            )
            if (distance <= radius or ray_match) and distance < nearest_distance:
                nearest = cluster
                nearest_distance = distance
        return nearest

    def _ray_matches_cluster(
        self,
        cluster_pose: ObjectPoseMap,
        observed_pose: ObjectPoseMap,
        robot_pose: Optional[RobotPoseMap],
    ) -> bool:
        if robot_pose is None:
            return False
        ray_x = observed_pose.x - robot_pose.x
        ray_y = observed_pose.y - robot_pose.y
        ray_length = math.hypot(ray_x, ray_y)
        if ray_length <= 0.05:
            return False
        unit_x, unit_y = ray_x / ray_length, ray_y / ray_length
        cluster_x = cluster_pose.x - robot_pose.x
        cluster_y = cluster_pose.y - robot_pose.y
        projection = cluster_x * unit_x + cluster_y * unit_y
        if projection <= 0.05:
            return False
        perpendicular = abs(cluster_x * unit_y - cluster_y * unit_x)
        return perpendicular <= self.track_reassociation_ray_tolerance_m

    def _is_farther_distance_outlier(
        self,
        current_pose: ObjectPoseMap,
        observed_pose: ObjectPoseMap,
        robot_pose: Optional[RobotPoseMap],
    ) -> bool:
        if robot_pose is None:
            return False
        current_distance = math.hypot(
            current_pose.x - robot_pose.x,
            current_pose.y - robot_pose.y,
        )
        observed_distance = math.hypot(
            observed_pose.x - robot_pose.x,
            observed_pose.y - robot_pose.y,
        )
        return observed_distance > current_distance + self.max_far_jump_m

    def _merge_overlapping_clusters(
        self,
        target: ActiveCluster,
        visible_track_ids: Optional[set[int]],
    ) -> ActiveCluster:
        merged = True
        while merged:
            merged = False
            for cluster_id, candidate in list(self.active.items()):
                if candidate is target or candidate.label != target.label:
                    continue
                distance = math.hypot(
                    target.object_pose.x - candidate.object_pose.x,
                    target.object_pose.y - candidate.object_pose.y,
                )
                different_tracks = (
                    target.track_id is not None
                    and candidate.track_id is not None
                    and target.track_id != candidate.track_id
                )
                same_track = (
                    target.track_id is not None
                    and target.track_id == candidate.track_id
                )
                if different_tracks and (
                    visible_track_ids is None
                    or candidate.track_id in visible_track_ids
                ):
                    continue
                radius = (
                    self.track_reassociation_radius_m
                    if different_tracks or same_track
                    else self.association_radius_m
                )
                if not same_track and distance > radius:
                    continue
                if (
                    same_track
                    and target.robot_pose is not None
                    and self._is_farther_distance_outlier(
                        candidate.object_pose,
                        target.object_pose,
                        target.robot_pose,
                    )
                ):
                    target.object_pose = candidate.object_pose
                elif not (
                    same_track
                    and self._is_farther_distance_outlier(
                        target.object_pose,
                        candidate.object_pose,
                        target.robot_pose,
                    )
                ):
                    target.object_pose = self._blend_pose(
                        target.object_pose,
                        candidate.object_pose,
                        new_weight=0.5,
                    )
                target.count += candidate.count
                target.expires_at = max(target.expires_at, candidate.expires_at)
                target.ttl_s = max(target.ttl_s, candidate.ttl_s)
                self.active.pop(cluster_id, None)
                self.delete_queue.extend(
                    candidate.marker_base_id + offset
                    for offset in range(self.marker_stride)
                )
                merged = True
                break
        return target

    def _expire_old_markers(self) -> None:
        now = time.monotonic()
        expired = [cluster_id for cluster_id, cluster in self.active.items() if cluster.expires_at <= now]
        for cluster_id in expired:
            cluster = self.active.pop(cluster_id)
            self.delete_queue.extend(
                cluster.marker_base_id + offset
                for offset in range(self.marker_stride)
            )

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

    def _ray_marker(self, cluster: ActiveCluster, stamp: Dict[str, int]) -> Dict[str, Any]:
        msg = self._base(cluster, cluster.marker_base_id + 2, MARKER_LINE_STRIP, stamp)
        msg["pose"]["position"] = {"x": 0.0, "y": 0.0, "z": 0.0}
        msg["lifetime"] = {
            "sec": int(self.ray_ttl_s),
            "nanosec": int((self.ray_ttl_s % 1.0) * 1_000_000_000),
        }
        robot_pose = cluster.robot_pose
        assert robot_pose is not None
        msg.update(
            {
                "scale": {"x": self.auxiliary_line_width_m, "y": 0.0, "z": 0.0},
                "color": {"r": 1.0, "g": 0.65, "b": 0.0, "a": 0.85},
                "points": [
                    {"x": robot_pose.x, "y": robot_pose.y, "z": 0.04},
                    {
                        "x": cluster.object_pose.x,
                        "y": cluster.object_pose.y,
                        "z": cluster.object_pose.z + 0.04,
                    },
                ],
            }
        )
        return msg

    def _uncertainty_marker(
        self, cluster: ActiveCluster, stamp: Dict[str, int]
    ) -> Dict[str, Any]:
        msg = self._base(cluster, cluster.marker_base_id + 3, MARKER_LINE_STRIP, stamp)
        msg["pose"]["position"] = {"x": 0.0, "y": 0.0, "z": 0.0}
        uncertainty = max(0.0, float(cluster.uncertainty_m or 0.0))
        radius = min(
            self.uncertainty_max_radius_m,
            max(self.uncertainty_min_radius_m, uncertainty * self.uncertainty_sigma_scale),
        )
        points = []
        for index in range(49):
            angle = 2.0 * math.pi * index / 48.0
            points.append(
                {
                    "x": cluster.object_pose.x + radius * math.cos(angle),
                    "y": cluster.object_pose.y + radius * math.sin(angle),
                    "z": cluster.object_pose.z + 0.025,
                }
            )
        msg.update(
            {
                "scale": {"x": self.auxiliary_line_width_m, "y": 0.0, "z": 0.0},
                "color": {"r": 1.0, "g": 0.85, "b": 0.0, "a": 0.9},
                "points": points,
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

    def _delete_all_marker(self, stamp: Dict[str, int]) -> Dict[str, Any]:
        msg = self._delete_marker(0, stamp)
        msg["action"] = MARKER_DELETE_ALL
        return msg

    def _summary(self, cluster: ActiveCluster) -> AnomalyClusterSummary:
        return AnomalyClusterSummary(
            cluster_id=cluster.cluster_id,
            label=cluster.label,
            object_pose=cluster.object_pose,
            count=cluster.count,
        )

    def _blend_pose(self, current: ObjectPoseMap, new_pose: ObjectPoseMap, new_weight: float) -> ObjectPoseMap:
        new_weight = max(0.0, min(1.0, float(new_weight)))
        current_weight = 1.0 - new_weight
        return ObjectPoseMap(
            x=(current.x * current_weight) + (new_pose.x * new_weight),
            y=(current.y * current_weight) + (new_pose.y * new_weight),
            z=(current.z * current_weight) + (new_pose.z * new_weight),
        )

    def _label_text(self, cluster: ActiveCluster) -> str:
        if self.text_compact and cluster.track_id is not None:
            label = f"#{cluster.track_id}"
            if self.text_show_count and cluster.count > 1:
                label += f" x{cluster.count}"
            return label
        label = cluster.label
        if cluster.track_id is not None:
            label += f" #{cluster.track_id}"
        if self.text_show_count and cluster.count > 1:
            label += f" x{cluster.count}"
        return label

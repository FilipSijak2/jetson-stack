from __future__ import annotations

import argparse
import json
import logging
import signal
import sys
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from .config import AppConfig, load_config
from .event_schema import EventJsonlWriter, build_event, build_readable_event
from .localization import (
    estimate_3d_bounds_camera,
    estimate_depth_measurement,
    estimate_laser_distance_m,
    estimate_object_pose_map,
)
from .map_snapshot import save_daily_map_summary
from .marker_manager import MarkerManager, build_detection_3d_marker_array
from .models import (
    AnomalyClusterSummary,
    CameraIntrinsics,
    Detection,
    DistanceEstimate,
    LaserScan,
    ObjectPoseMap,
    OccupancyGridMap,
    RobotPoseMap,
)
from .ros_messages import (
    compressed_image_msg,
    decode_compressed_image,
    decode_depth_image,
    encode_image,
    header_stamp_seconds,
    parse_camera_info,
    parse_occupancy_grid,
    parse_laser_scan,
    parse_robot_pose,
)
from .rosbridge_ws import RosbridgeClient
from .yolo_detector import YoloDetector


LOGGER = logging.getLogger("jetson_yolo_rosbridge_client")


@dataclass(frozen=True)
class LocatedDetection:
    detection: Detection
    object_pose: ObjectPoseMap
    distance_estimate: DistanceEstimate
    bearing_source: str


@dataclass(frozen=True)
class DetectionGroup:
    label: str
    detections: List[Detection]
    object_pose: ObjectPoseMap
    distance_estimate: DistanceEstimate
    bearing_source: str


@dataclass
class PendingAnomaly:
    label: str
    object_pose: ObjectPoseMap
    observations: int
    last_seen: float
    track_id: Optional[int] = None


@dataclass
class ReportedAnomaly:
    label: str
    object_pose: ObjectPoseMap
    track_id: Optional[int] = None


@dataclass
class InspectionCaptureState:
    request_id: str
    cluster_id: str
    label: str
    object_pose: ObjectPoseMap
    track_id: Optional[int]
    target_frames: int
    deadline: float
    frames_seen: int = 0
    best_score: float = -1.0
    best_image: Optional[np.ndarray] = None
    best_source_msg: Optional[Dict[str, Any]] = None


class JetsonYoloRosbridgeClient:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.artifact_root = Path(config.artifact_root)
        self.original_dir = self.artifact_root / "images" / "original"
        self.annotated_dir = self.artifact_root / "images" / "annotated"
        self.map_dir = self.artifact_root / "map_images"
        self.daily_map_dir = self.map_dir / "daily"
        self.inspection_dir = self.artifact_root / "images" / "inspection"
        self.event_log_path = self.artifact_root / "events.jsonl"
        self.inspection_log_path = self.artifact_root / "inspections.jsonl"
        paths = [self.map_dir, self.daily_map_dir]
        if config.save_per_event_images:
            paths.extend([self.original_dir, self.annotated_dir])
        if config.inspection_enabled:
            paths.append(self.inspection_dir)
        for path in paths:
            path.mkdir(parents=True, exist_ok=True)

        self.event_writer = EventJsonlWriter(self.event_log_path)
        self.detector = YoloDetector(
            model_path=config.yolo_model_path,
            confidence_threshold=config.confidence_threshold,
            anomaly_classes=config.anomaly_classes,
            mock_mode=config.mock_mode,
            logger=LOGGER,
            image_size=config.yolo_image_size,
            iou_threshold=config.yolo_iou_threshold,
            max_detections=config.yolo_max_detections,
            device=config.yolo_device,
            half=config.yolo_half,
            augment=config.yolo_augment,
            agnostic_nms=config.yolo_agnostic_nms,
            filter_classes=config.yolo_filter_classes,
            tracking_enabled=config.tracking_enabled,
            tracking_backend=config.tracking_backend,
            tracking_confidence_threshold=config.tracking_confidence_threshold,
            segmentation_enabled=config.segmentation_enabled,
        )
        self.markers = MarkerManager(
            frame_id=config.map_frame_id,
            merge_radius_m=config.cluster_merge_radius_m,
            association_radius_m=config.marker_association_radius_m,
            object_marker_size_m=config.marker_object_size_m,
            text_height_m=config.marker_text_height_m,
            text_z_offset_m=config.marker_text_z_offset_m,
            text_show_count=config.marker_text_show_count,
            text_compact=config.marker_text_compact,
            tracked_object_min_separation_m=config.tracked_object_min_separation_m,
            ray_enabled=config.marker_ray_enabled,
            ray_ttl_s=config.marker_ray_ttl_s,
            uncertainty_enabled=config.marker_uncertainty_enabled,
            uncertainty_sigma_scale=config.marker_uncertainty_sigma_scale,
            uncertainty_min_radius_m=config.marker_uncertainty_min_radius_m,
            uncertainty_max_radius_m=config.marker_uncertainty_max_radius_m,
            auxiliary_line_width_m=config.marker_aux_line_width_m,
        )
        self.rosbridge = RosbridgeClient(config.rosbridge_url, LOGGER)
        self.latest_map: Optional[OccupancyGridMap] = None
        self.latest_pose: Optional[RobotPoseMap] = None
        self.latest_scan: Optional[LaserScan] = None
        self.camera_intrinsics: Optional[CameraIntrinsics] = None
        self.depth_buffer = deque(maxlen=config.depth_buffer_size)
        self.last_event_by_label: Dict[str, float] = {}
        self.last_event_clusters: Dict[str, float] = {}
        self.reported_anomalies: List[ReportedAnomaly] = []
        self.pending_anomalies: List[PendingAnomaly] = []
        self.daily_clusters: Dict[str, AnomalyClusterSummary] = {}
        self.frame_count = 0
        self.event_counter = 0
        self.current_daily_key = _local_day_key()
        self._load_reported_anomalies()
        self.stop_requested = False
        self.next_marker_publish = 0.0
        self.next_daily_summary_refresh = 0.0
        self.next_debug_image_publish = 0.0
        self.next_privacy_image_publish = 0.0
        self.next_detection_3d_publish = 0.0
        self.last_missing_pose_log = 0.0
        self.last_missing_map_log = 0.0
        self.last_missing_scan_log = 0.0
        self.last_event_gate_log: Dict[str, float] = {}
        self.inspection_sequence = 0
        self.inspection_pending: Dict[str, Dict[str, Any]] = {}
        self.inspection_capture: Optional[InspectionCaptureState] = None
        self.inspection_retry_after: Dict[str, float] = {}
        self.inspected_locations: List[ReportedAnomaly] = []
        self._load_inspected_locations()

    def run_forever(self) -> None:
        while not self.stop_requested:
            try:
                self.rosbridge.connect()
                self._setup_rosbridge_topics()
                self._event_loop()
            except KeyboardInterrupt:
                self.stop_requested = True
            except Exception as exc:
                LOGGER.warning("rosbridge loop ended: %s", exc)
            finally:
                self.rosbridge.close()

            if not self.stop_requested:
                LOGGER.info("Reconnecting in %.1f s", self.config.reconnect_delay_s)
                time.sleep(self.config.reconnect_delay_s)

    def request_stop(self, *_args: Any) -> None:
        self.stop_requested = True
        self.rosbridge.close()

    def _setup_rosbridge_topics(self) -> None:
        self.rosbridge.subscribe(self.config.camera_topic, "sensor_msgs/CompressedImage", queue_length=1)
        self.rosbridge.subscribe(self.config.map_topic, "nav_msgs/OccupancyGrid", queue_length=1, throttle_rate=500)
        if self.config.use_depth_distance and self.config.depth_topic:
            self.rosbridge.subscribe(
                self.config.depth_topic,
                "sensor_msgs/Image",
                queue_length=1,
                throttle_rate=self.config.depth_throttle_ms,
            )
        if self.config.use_camera_intrinsics and self.config.camera_info_topic:
            self.rosbridge.subscribe(
                self.config.camera_info_topic,
                "sensor_msgs/CameraInfo",
                queue_length=1,
                throttle_rate=1000,
            )
        if self.config.use_laser_distance and self.config.scan_topic:
            self.rosbridge.subscribe(self.config.scan_topic, "sensor_msgs/LaserScan", queue_length=1, throttle_rate=100)
        pose_type = "geometry_msgs/PoseStamped"
        if self.config.robot_pose_topic == "/amcl_pose":
            pose_type = "geometry_msgs/PoseWithCovarianceStamped"
        self.rosbridge.subscribe(self.config.robot_pose_topic, pose_type, queue_length=1, throttle_rate=500)
        if self.config.inspection_enabled:
            self.rosbridge.subscribe(
                self.config.inspection_status_topic,
                "std_msgs/String",
                queue_length=10,
            )

        self.rosbridge.advertise(self.config.event_topic, "std_msgs/String")
        self.rosbridge.advertise(self.config.readable_event_topic, "std_msgs/String")
        self.rosbridge.advertise(self.config.marker_topic, "visualization_msgs/MarkerArray")
        if self.config.detection_3d_enabled:
            self.rosbridge.advertise(
                self.config.detection_3d_topic,
                "visualization_msgs/MarkerArray",
            )
        self.rosbridge.advertise(self.config.debug_image_topic, "sensor_msgs/CompressedImage")
        if self.config.privacy_image_enabled:
            self.rosbridge.advertise(
                self.config.privacy_image_topic, "sensor_msgs/CompressedImage"
            )
        if self.config.inspection_enabled:
            self.rosbridge.advertise(
                self.config.inspection_request_topic, "std_msgs/String"
            )
            self.rosbridge.advertise(
                self.config.inspection_result_topic, "std_msgs/String"
            )
            self.rosbridge.advertise(
                self.config.inspection_privacy_image_topic,
                "sensor_msgs/CompressedImage",
            )
        self.rosbridge.advertise(self.config.map_snapshot_topic, "sensor_msgs/CompressedImage")
        self._clear_existing_markers()

        LOGGER.info(
            "Subscribed camera=%s camera_info=%s depth=%s map=%s pose=%s scan=%s "
            "intrinsics=%s depth_distance=%s laser_distance=%s",
            self.config.camera_topic,
            self.config.camera_info_topic if self.config.use_camera_intrinsics else "disabled",
            self.config.depth_topic if self.config.use_depth_distance else "disabled",
            self.config.map_topic,
            self.config.robot_pose_topic,
            self.config.scan_topic if self.config.use_laser_distance else "disabled",
            self.config.use_camera_intrinsics,
            self.config.use_depth_distance,
            self.config.use_laser_distance,
        )
        LOGGER.info(
            "Publishing events=%s readable_events=%s markers=%s detections_3d=%s "
            "debug_image=%s privacy_image=%s map_snapshot=%s",
            self.config.event_topic,
            self.config.readable_event_topic,
            self.config.marker_topic,
            (
                self.config.detection_3d_topic
                if self.config.detection_3d_enabled
                else "disabled"
            ),
            self.config.debug_image_topic,
            (
                self.config.privacy_image_topic
                if self.config.privacy_image_enabled
                else "disabled"
            ),
            self.config.map_snapshot_topic,
        )
        if self.config.inspection_enabled:
            LOGGER.info(
                "Inspection enabled request=%s status=%s result=%s "
                "standoff=%.2fm capture_frames=%d",
                self.config.inspection_request_topic,
                self.config.inspection_status_topic,
                self.config.inspection_result_topic,
                self.config.inspection_standoff_m,
                self.config.inspection_capture_frames,
            )

    def _event_loop(self) -> None:
        self.next_marker_publish = time.monotonic()
        while not self.stop_requested:
            payload = self.rosbridge.recv_json()
            if payload is not None:
                self._handle_rosbridge_payload(payload)
            self._publish_markers_if_due()

    def _handle_rosbridge_payload(self, payload: Dict[str, Any]) -> None:
        if payload.get("op") != "publish":
            return
        topic = payload.get("topic")
        msg = payload.get("msg") or {}
        if not isinstance(msg, dict):
            LOGGER.warning("Ignoring malformed rosbridge message on %s", topic)
            return

        if topic == self.config.camera_topic:
            self._on_camera_image(msg)
        elif topic == self.config.depth_topic:
            self._on_depth_image(msg)
        elif topic == self.config.camera_info_topic:
            self._on_camera_info(msg)
        elif topic == self.config.map_topic:
            self._on_map(msg)
        elif topic == self.config.robot_pose_topic:
            self._on_pose(msg)
        elif topic == self.config.scan_topic:
            self._on_scan(msg)
        elif (
            self.config.inspection_enabled
            and topic == self.config.inspection_status_topic
        ):
            self._on_inspection_status(msg)

    def _on_map(self, msg: Dict[str, Any]) -> None:
        try:
            self.latest_map = parse_occupancy_grid(msg)
        except Exception as exc:
            LOGGER.warning("Failed to parse OccupancyGrid: %s", exc)

    def _on_pose(self, msg: Dict[str, Any]) -> None:
        try:
            self.latest_pose = parse_robot_pose(msg)
        except Exception as exc:
            LOGGER.warning("Failed to parse robot pose: %s", exc)

    def _on_scan(self, msg: Dict[str, Any]) -> None:
        try:
            self.latest_scan = parse_laser_scan(msg)
        except Exception as exc:
            LOGGER.warning("Failed to parse LaserScan: %s", exc)

    def _on_depth_image(self, msg: Dict[str, Any]) -> None:
        try:
            depth = decode_depth_image(msg)
            self.depth_buffer.append((header_stamp_seconds(msg), time.monotonic(), depth))
        except Exception as exc:
            LOGGER.warning("Failed to decode depth image: %s", exc)

    def _on_camera_info(self, msg: Dict[str, Any]) -> None:
        try:
            intrinsics = parse_camera_info(msg)
        except Exception as exc:
            LOGGER.warning("Failed to parse CameraInfo: %s", exc)
            return
        if intrinsics != self.camera_intrinsics:
            self.camera_intrinsics = intrinsics
            LOGGER.info(
                "Camera intrinsics fx=%.2f fy=%.2f cx=%.2f cy=%.2f size=%dx%d",
                intrinsics.fx,
                intrinsics.fy,
                intrinsics.cx,
                intrinsics.cy,
                intrinsics.width,
                intrinsics.height,
            )

    def _maybe_request_inspection(
        self,
        group: DetectionGroup,
        cluster: AnomalyClusterSummary,
    ) -> None:
        if not self.config.inspection_enabled:
            return
        if self.inspection_pending or self.inspection_capture is not None:
            return

        estimate = group.distance_estimate
        if (
            estimate.distance_m < self.config.inspection_min_distance_m
            or estimate.distance_m > self.config.inspection_max_distance_m
        ):
            return
        if (
            self.config.inspection_require_metric_distance
            and estimate.source not in {"depth", "laser"}
        ):
            self._log_event_gate("inspection_requires_depth_or_laser", group)
            return
        if (
            estimate.uncertainty_m is None
            or estimate.uncertainty_m > self.config.inspection_max_uncertainty_m
        ):
            self._log_event_gate("inspection_uncertainty_too_high", group)
            return

        track_id = _best_track_id(group.detections)
        if (
            self.config.inspection_once_per_cluster
            and self._inspection_already_completed(
                group.label, group.object_pose, track_id
            )
        ):
            return
        retry_key = self._inspection_location_key(group.label, group.object_pose)
        if time.monotonic() < self.inspection_retry_after.get(retry_key, 0.0):
            return

        robot_pose = self.latest_pose
        if robot_pose is None:
            return
        self.inspection_sequence += 1
        request_id = (
            f"inspect_{self.current_daily_key.replace('-', '')}_"
            f"{self.inspection_sequence:05d}"
        )
        request = {
            "request_id": request_id,
            "cluster_id": cluster.cluster_id,
            "timestamp": datetime.now().astimezone().isoformat(),
            "label": group.label,
            "track_id": track_id,
            "object_pose_map": {
                "x": group.object_pose.x,
                "y": group.object_pose.y,
                "z": group.object_pose.z,
            },
            "robot_pose_map": {
                "x": robot_pose.x,
                "y": robot_pose.y,
                "yaw": robot_pose.yaw,
            },
            "localization": {
                "distance_m": estimate.distance_m,
                "distance_source": estimate.source,
                "distance_uncertainty_m": estimate.uncertainty_m,
            },
            "standoff_m": self.config.inspection_standoff_m,
        }
        self.inspection_pending[request_id] = {
            "request": request,
            "object_pose": group.object_pose,
            "track_id": track_id,
            "retry_key": retry_key,
            "deadline": time.monotonic()
            + self.config.inspection_request_timeout_s,
        }
        self.rosbridge.publish(
            self.config.inspection_request_topic,
            {"data": json.dumps(request, separators=(",", ":"))},
        )
        LOGGER.info(
            "Inspection %s requested for %s track_id=%s at map=(%.2f, %.2f) "
            "distance=%.2fm source=%s",
            request_id,
            group.label,
            track_id if track_id is not None else "-",
            group.object_pose.x,
            group.object_pose.y,
            estimate.distance_m,
            estimate.source,
        )

    def _on_inspection_status(self, msg: Dict[str, Any]) -> None:
        try:
            status = json.loads(str(msg.get("data") or ""))
        except (json.JSONDecodeError, TypeError, ValueError):
            LOGGER.warning("Ignoring malformed inspection status: %r", msg.get("data"))
            return
        if not isinstance(status, dict):
            return
        request_id = str(status.get("request_id") or "")
        state = str(status.get("state") or "")
        metadata = self.inspection_pending.get(request_id)
        if metadata is None:
            return
        LOGGER.info(
            "Inspection %s state=%s reason=%s",
            request_id,
            state,
            status.get("reason") or "-",
        )

        if state == "capture_requested":
            if self.inspection_capture is not None:
                return
            request = metadata["request"]
            self.inspection_capture = InspectionCaptureState(
                request_id=request_id,
                cluster_id=str(request["cluster_id"]),
                label=str(request["label"]),
                object_pose=metadata["object_pose"],
                track_id=metadata["track_id"],
                target_frames=self.config.inspection_capture_frames,
                deadline=time.monotonic()
                + self.config.inspection_capture_timeout_s,
            )
            return

        if state in {"rejected", "failed", "canceled", "timeout", "capture_failed"}:
            self.inspection_retry_after[metadata["retry_key"]] = (
                time.monotonic() + self.config.inspection_retry_cooldown_s
            )
            self.inspection_pending.pop(request_id, None)
            if (
                self.inspection_capture is not None
                and self.inspection_capture.request_id == request_id
            ):
                self.inspection_capture = None
        elif state == "completed":
            self.inspection_pending.pop(request_id, None)

    def _expire_stale_inspection_request(self) -> None:
        if self.inspection_capture is not None:
            return
        now = time.monotonic()
        for request_id, metadata in list(self.inspection_pending.items()):
            if now < float(metadata.get("deadline", now + 1.0)):
                continue
            self.inspection_retry_after[metadata["retry_key"]] = (
                now + self.config.inspection_retry_cooldown_s
            )
            self.inspection_pending.pop(request_id, None)
            LOGGER.warning(
                "Inspection %s expired without a terminal coordinator status",
                request_id,
            )

    def _capture_inspection_frame_if_active(
        self,
        detections: List[Detection],
        frame: np.ndarray,
        source_msg: Dict[str, Any],
    ) -> None:
        capture = self.inspection_capture
        if capture is None:
            return
        if time.monotonic() >= capture.deadline:
            self._finish_inspection_capture(False, "capture_timeout")
            return

        detection = self._select_inspection_detection(
            capture, detections, frame.shape[1], frame.shape[0], source_msg
        )
        if detection is None:
            return
        privacy_frame = blur_except_detections(
            frame,
            [detection],
            kernel_size=self.config.privacy_blur_kernel_size,
            padding_ratio=self.config.privacy_bbox_padding_ratio,
            use_segmentation_masks=self.config.privacy_use_segmentation_masks,
            draw_track_id=self.config.privacy_draw_track_id,
            draw_mask_overlay=self.config.privacy_draw_mask_overlay,
            mask_overlay_alpha=self.config.privacy_mask_overlay_alpha,
        )
        score = _detection_sharpness(frame, detection)
        capture.frames_seen += 1
        if score > capture.best_score:
            capture.best_score = score
            capture.best_image = privacy_frame.copy()
            capture.best_source_msg = dict(source_msg)
        if capture.frames_seen >= capture.target_frames:
            self._finish_inspection_capture(True, "best_frame_selected")

    def _select_inspection_detection(
        self,
        capture: InspectionCaptureState,
        detections: List[Detection],
        image_width: int,
        image_height: int,
        source_msg: Dict[str, Any],
    ) -> Optional[Detection]:
        candidates = [item for item in detections if item.label == capture.label]
        if capture.track_id is not None:
            tracked = [
                item for item in candidates if item.track_id == capture.track_id
            ]
            if tracked:
                return max(tracked, key=lambda item: item.confidence)

        nearest: Optional[Detection] = None
        nearest_distance = float("inf")
        for candidate in candidates:
            try:
                located = self._locate_detection(
                    candidate, image_width, image_height, source_msg
                )
            except Exception:
                continue
            distance = _distance_xy(located.object_pose, capture.object_pose)
            if (
                distance <= self.config.marker_association_radius_m
                and distance < nearest_distance
            ):
                nearest = candidate
                nearest_distance = distance
        return nearest

    def _finish_inspection_capture(self, success: bool, reason: str) -> None:
        capture = self.inspection_capture
        if capture is None:
            return
        metadata = self.inspection_pending.get(capture.request_id)
        image_path: Optional[Path] = None
        if success and capture.best_image is not None:
            try:
                image_path = (
                    self.inspection_dir
                    / self.current_daily_key
                    / f"{capture.request_id}_{_slug(capture.label)}_privacy.jpg"
                )
                _write_image(
                    image_path,
                    capture.best_image,
                    quality=self.config.inspection_jpeg_quality,
                )
                self._publish_inspection_privacy_image(
                    capture.best_image,
                    capture.best_source_msg or {},
                )
                inspected = ReportedAnomaly(
                    capture.label, capture.object_pose, capture.track_id
                )
                self.inspected_locations.append(inspected)
            except Exception as exc:
                LOGGER.exception(
                    "Inspection %s privacy image save failed: %s",
                    capture.request_id,
                    exc,
                )
                image_path = None
                success = False
                reason = f"capture_save_failed: {exc}"
        else:
            success = False

        result = {
            "request_id": capture.request_id,
            "cluster_id": capture.cluster_id,
            "timestamp": datetime.now().astimezone().isoformat(),
            "success": success,
            "reason": reason,
            "label": capture.label,
            "track_id": capture.track_id,
            "object_pose_map": {
                "x": capture.object_pose.x,
                "y": capture.object_pose.y,
                "z": capture.object_pose.z,
            },
            "frames_evaluated": capture.frames_seen,
            "sharpness_score": (
                round(capture.best_score, 3) if capture.best_score >= 0.0 else None
            ),
            "privacy_image": str(image_path) if image_path is not None else None,
        }
        try:
            self.inspection_log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.inspection_log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(result, sort_keys=True) + "\n")
        except OSError as exc:
            LOGGER.error("Could not append inspection result log: %s", exc)
        self.rosbridge.publish(
            self.config.inspection_result_topic,
            {"data": json.dumps(result, separators=(",", ":"))},
        )
        LOGGER.info(
            "Inspection %s capture success=%s frames=%d sharpness=%s image=%s",
            capture.request_id,
            success,
            capture.frames_seen,
            result["sharpness_score"],
            image_path or "-",
        )
        if not success and metadata is not None:
            self.inspection_retry_after[metadata["retry_key"]] = (
                time.monotonic() + self.config.inspection_retry_cooldown_s
            )
        self.inspection_capture = None

    def _publish_inspection_privacy_image(
        self,
        image: np.ndarray,
        source_msg: Dict[str, Any],
    ) -> None:
        header = source_msg.get("header") or {}
        encoded = encode_image(
            image, ".jpg", quality=self.config.inspection_jpeg_quality
        )
        self.rosbridge.publish(
            self.config.inspection_privacy_image_topic,
            compressed_image_msg(
                encoded,
                "jpeg",
                frame_id=header.get("frame_id") or "camera",
                stamp=header.get("stamp"),
            ),
        )

    def _inspection_location_key(
        self, label: str, object_pose: ObjectPoseMap
    ) -> str:
        cell = max(0.05, self.config.reported_merge_radius_m)
        return (
            f"{label}:{round(object_pose.x / cell)}:"
            f"{round(object_pose.y / cell)}"
        )

    def _inspection_already_completed(
        self,
        label: str,
        object_pose: ObjectPoseMap,
        track_id: Optional[int],
    ) -> bool:
        radius = max(0.05, self.config.reported_merge_radius_m)
        for inspected in self.inspected_locations:
            if inspected.label != label:
                continue
            distance = _distance_xy(inspected.object_pose, object_pose)
            if (
                track_id is not None
                and inspected.track_id is not None
                and track_id != inspected.track_id
                and distance >= self.config.tracked_object_min_separation_m
            ):
                continue
            if distance <= radius:
                return True
        return False

    def _load_inspected_locations(self) -> None:
        try:
            lines = self.inspection_log_path.read_text(
                encoding="utf-8"
            ).splitlines()
        except FileNotFoundError:
            return
        except OSError as exc:
            LOGGER.warning(
                "Could not read inspection log %s: %s",
                self.inspection_log_path,
                exc,
            )
            return
        for line in lines:
            try:
                result = json.loads(line)
                if (
                    not result.get("success")
                    or _event_local_day_key(result) != self.current_daily_key
                ):
                    continue
                pose = result["object_pose_map"]
                track_id = result.get("track_id")
                self.inspected_locations.append(
                    ReportedAnomaly(
                        str(result["label"]),
                        ObjectPoseMap(
                            float(pose["x"]),
                            float(pose["y"]),
                            float(pose.get("z", 0.0)),
                        ),
                        int(track_id) if track_id is not None else None,
                    )
                )
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                continue

    def _rollover_daily_state_if_needed(self) -> None:
        day = _local_day_key()
        if day == self.current_daily_key:
            return
        previous_day = self.current_daily_key
        self.current_daily_key = day
        self.reported_anomalies.clear()
        self.pending_anomalies.clear()
        self.daily_clusters.clear()
        self.last_event_clusters.clear()
        self.last_event_gate_log.clear()
        self.inspected_locations.clear()
        self.inspection_retry_after.clear()
        self.next_daily_summary_refresh = 0.0
        self._clear_existing_markers()
        self.markers.reset()
        LOGGER.info(
            "Daily anomaly state reset: %s -> %s. "
            "De-duplication and daily map now start empty.",
            previous_day,
            day,
        )

    def _on_camera_image(self, msg: Dict[str, Any]) -> None:
        self._rollover_daily_state_if_needed()
        self.frame_count += 1
        if self.frame_count % self.config.inference_every_n_frames != 0:
            return

        try:
            frame = decode_compressed_image(msg)
        except Exception as exc:
            LOGGER.warning("Failed to decode compressed camera image: %s", exc)
            return

        detections = self.detector.detect(frame)
        anomalies = self._filter_anomalies(detections)
        self._expire_stale_inspection_request()
        self._capture_inspection_frame_if_active(anomalies, frame, msg)
        self._publish_debug_stream_if_due(anomalies, frame, msg)
        self._publish_privacy_stream_if_due(anomalies, frame, msg)
        self._publish_detection_3d_if_due(anomalies, frame, msg)
        if not anomalies:
            return

        if self.latest_pose is None:
            self._log_missing_pose()
            return

        located = [
            self._locate_detection(detection, frame.shape[1], frame.shape[0], msg)
            for detection in anomalies
        ]
        for group in self._group_located_detections(located):
            confirmed_group = self._confirm_detection_group(group)
            if confirmed_group is None:
                continue
            cluster = self.markers.add_or_update(
                label=confirmed_group.label,
                object_pose=confirmed_group.object_pose,
                observed_count=len(confirmed_group.detections),
                ttl_s=self.config.marker_ttl_s,
                robot_pose=self.latest_pose,
                uncertainty_m=confirmed_group.distance_estimate.uncertainty_m,
                track_id=_best_track_id(confirmed_group.detections),
            )
            self.daily_clusters[cluster.cluster_id] = cluster
            self._maybe_request_inspection(confirmed_group, cluster)
            track_id = _best_track_id(confirmed_group.detections)
            if self._already_reported(
                confirmed_group.label, cluster.object_pose, track_id
            ):
                self._log_event_gate("already_reported_today", confirmed_group)
                self._remember_reported(
                    confirmed_group.label, cluster.object_pose, track_id
                )
                self._refresh_daily_map_summary(msg)
                continue
            if not self._cooldown_ready(confirmed_group.label, cluster.object_pose):
                self._log_event_gate("cooldown", confirmed_group)
                continue
            try:
                self._create_anomaly_event(confirmed_group, cluster, frame, msg)
            except Exception as exc:
                LOGGER.exception("Failed to create anomaly event: %s", exc)

    def _filter_anomalies(self, detections: List[Detection]) -> List[Detection]:
        anomaly_labels = {label.strip() for label in self.config.anomaly_classes}
        return [
            detection
            for detection in detections
            if detection.label in anomaly_labels and detection.confidence >= self.config.confidence_threshold
        ]

    def _locate_detection(
        self,
        detection: Detection,
        image_width: int,
        image_height: int,
        source_msg: Dict[str, Any],
    ) -> LocatedDetection:
        robot_pose = self.latest_pose
        if robot_pose is None:
            raise RuntimeError("Cannot locate detection without robot pose")
        distance_estimate = self._distance_for_detection(
            detection, image_width, image_height, source_msg
        )
        intrinsics = self.camera_intrinsics if self.config.use_camera_intrinsics else None
        object_pose = estimate_object_pose_map(
            robot_pose=robot_pose,
            bbox_xyxy=detection.bbox_xyxy,
            image_width=image_width,
            default_distance_m=self.config.default_anomaly_distance_m,
            camera_horizontal_fov_deg=self.config.camera_horizontal_fov_deg,
            camera_yaw_offset_deg=self.config.camera_yaw_offset_deg,
            measured_distance_m=distance_estimate.distance_m,
            camera_intrinsics=intrinsics,
            measured_distance_is_axial=distance_estimate.source == "depth",
        )
        if distance_estimate.source == "depth":
            ray_distance = float(
                np.hypot(object_pose.x - robot_pose.x, object_pose.y - robot_pose.y)
            )
            scale = ray_distance / max(0.001, distance_estimate.distance_m)
            distance_estimate = DistanceEstimate(
                distance_m=ray_distance,
                source=distance_estimate.source,
                uncertainty_m=(
                    distance_estimate.uncertainty_m * scale
                    if distance_estimate.uncertainty_m is not None
                    else None
                ),
                valid_sample_count=distance_estimate.valid_sample_count,
                age_s=distance_estimate.age_s,
                axial_depth_m=distance_estimate.distance_m,
            )
        return LocatedDetection(
            detection=detection,
            object_pose=object_pose,
            distance_estimate=distance_estimate,
            bearing_source="camera_intrinsics" if intrinsics is not None else "horizontal_fov",
        )

    def _group_located_detections(self, located: List[LocatedDetection]) -> List[DetectionGroup]:
        groups: List[List[LocatedDetection]] = []
        for item in sorted(located, key=lambda value: value.detection.confidence, reverse=True):
            target_group = None
            for group in groups:
                if group[0].detection.label != item.detection.label:
                    continue
                center = self._group_center(group)
                distance = _distance_xy(center, item.object_pose)
                group_track_id = _best_track_id(
                    [group_item.detection for group_item in group]
                )
                if (
                    item.detection.track_id is not None
                    and group_track_id is not None
                    and item.detection.track_id != group_track_id
                    and distance >= self.config.tracked_object_min_separation_m
                ):
                    continue
                if distance <= self.config.cluster_merge_radius_m:
                    target_group = group
                    break
            if target_group is None:
                groups.append([item])
            else:
                target_group.append(item)

        return [
            DetectionGroup(
                label=group[0].detection.label,
                detections=[item.detection for item in group],
                object_pose=self._group_center(group),
                distance_estimate=group[0].distance_estimate,
                bearing_source=group[0].bearing_source,
            )
            for group in groups
        ]

    def _log_event_gate(self, reason: str, group: DetectionGroup) -> None:
        track_id = _best_track_id(group.detections)
        key = f"{reason}:{group.label}:{track_id}"
        now = time.monotonic()
        if now - self.last_event_gate_log.get(key, 0.0) < 5.0:
            return
        self.last_event_gate_log[key] = now
        detection = max(group.detections, key=lambda item: item.confidence)
        LOGGER.info(
            "Detection label=%s confidence=%.2f track_id=%s event_gate=%s "
            "object_map=(%.2f, %.2f) distance_source=%s",
            group.label,
            detection.confidence,
            track_id if track_id is not None else "-",
            reason,
            group.object_pose.x,
            group.object_pose.y,
            group.distance_estimate.source,
        )

    def _group_center(self, group: List[LocatedDetection]) -> ObjectPoseMap:
        count = max(1, len(group))
        return ObjectPoseMap(
            x=sum(item.object_pose.x for item in group) / count,
            y=sum(item.object_pose.y for item in group) / count,
            z=sum(item.object_pose.z for item in group) / count,
        )

    def _confirm_detection_group(self, group: DetectionGroup) -> Optional[DetectionGroup]:
        if self.config.anomaly_min_observations <= 1:
            return group
        pending = self._remember_pending_anomaly(group)
        if pending.observations < self.config.anomaly_min_observations:
            self._log_event_gate(
                f"pending_{pending.observations}_of_{self.config.anomaly_min_observations}",
                group,
            )
            return None
        return DetectionGroup(
            label=group.label,
            detections=group.detections,
            object_pose=pending.object_pose,
            distance_estimate=group.distance_estimate,
            bearing_source=group.bearing_source,
        )

    def _remember_pending_anomaly(self, group: DetectionGroup) -> PendingAnomaly:
        now = time.monotonic()
        self._drop_stale_pending_anomalies(now)
        track_id = _best_track_id(group.detections)
        index = self._pending_index(group.label, group.object_pose, track_id)
        if index is None:
            pending = PendingAnomaly(
                label=group.label,
                object_pose=group.object_pose,
                observations=1,
                last_seen=now,
                track_id=track_id,
            )
            self.pending_anomalies.append(pending)
            return pending

        pending = self.pending_anomalies[index]
        observations = pending.observations + 1
        pending.object_pose = _blend_pose(
            pending.object_pose,
            group.object_pose,
            new_weight=1.0 / float(observations),
        )
        pending.observations = observations
        pending.last_seen = now
        pending.track_id = track_id if track_id is not None else pending.track_id
        return pending

    def _drop_stale_pending_anomalies(self, now: float) -> None:
        ttl = self.config.anomaly_confirmation_ttl_s
        self.pending_anomalies = [
            pending
            for pending in self.pending_anomalies
            if now - pending.last_seen <= ttl
        ]

    def _pending_index(
        self,
        label: str,
        object_pose: ObjectPoseMap,
        track_id: Optional[int],
    ) -> Optional[int]:
        if track_id is not None:
            for index, pending in enumerate(self.pending_anomalies):
                if (
                    pending.label == label
                    and pending.track_id == track_id
                    and _distance_xy(pending.object_pose, object_pose)
                    <= self.config.marker_association_radius_m
                ):
                    return index
        radius = max(0.01, self.config.cluster_merge_radius_m)
        nearest_index = None
        nearest_distance = float("inf")
        for index, pending in enumerate(self.pending_anomalies):
            if pending.label != label:
                continue
            distance = _distance_xy(pending.object_pose, object_pose)
            if (
                track_id is not None
                and pending.track_id is not None
                and track_id != pending.track_id
                and distance >= self.config.tracked_object_min_separation_m
            ):
                continue
            if distance <= radius and distance < nearest_distance:
                nearest_index = index
                nearest_distance = distance
        return nearest_index

    def _cooldown_ready(self, label: str, object_pose: ObjectPoseMap) -> bool:
        now = time.monotonic()
        stale_keys = [
            key
            for key, last_seen in self.last_event_clusters.items()
            if now - last_seen >= max(self.config.detection_cooldown_s * 4.0, 60.0)
        ]
        for key in stale_keys:
            self.last_event_clusters.pop(key, None)

        key = self._cooldown_key(label, object_pose)
        last = self.last_event_clusters.get(key, 0.0)
        return now - last >= self.config.detection_cooldown_s

    def _mark_cooldown(self, label: str, object_pose: ObjectPoseMap) -> None:
        self.last_event_clusters[self._cooldown_key(label, object_pose)] = time.monotonic()

    def _cooldown_key(self, label: str, object_pose: ObjectPoseMap) -> str:
        cell = max(0.01, self.config.cluster_merge_radius_m)
        return f"{label}:{round(object_pose.x / cell)}:{round(object_pose.y / cell)}"

    def _already_reported(
        self,
        label: str,
        object_pose: ObjectPoseMap,
        track_id: Optional[int] = None,
    ) -> bool:
        return self._reported_index(label, object_pose, track_id) is not None

    def _reported_index(
        self,
        label: str,
        object_pose: ObjectPoseMap,
        track_id: Optional[int] = None,
    ) -> Optional[int]:
        radius = max(0.01, self.config.reported_merge_radius_m)
        for index, reported in enumerate(self.reported_anomalies):
            if reported.label != label:
                continue
            distance = _distance_xy(reported.object_pose, object_pose)
            if (
                track_id is not None
                and reported.track_id is not None
                and track_id != reported.track_id
                and distance >= self.config.tracked_object_min_separation_m
            ):
                continue
            if distance <= radius:
                return index
        return None

    def _remember_reported(
        self,
        label: str,
        object_pose: ObjectPoseMap,
        track_id: Optional[int] = None,
    ) -> None:
        index = self._reported_index(label, object_pose, track_id)
        if index is None:
            self.reported_anomalies.append(
                ReportedAnomaly(label, object_pose, track_id)
            )
        else:
            previous = self.reported_anomalies[index]
            self.reported_anomalies[index] = ReportedAnomaly(
                label,
                _blend_pose(previous.object_pose, object_pose, new_weight=0.25),
                track_id if track_id is not None else previous.track_id,
            )

    def _load_reported_anomalies(self) -> None:
        try:
            lines = self.event_log_path.read_text(encoding="utf-8").splitlines()
        except FileNotFoundError:
            return
        except OSError as exc:
            LOGGER.warning("Could not read anomaly event log %s: %s", self.event_log_path, exc)
            return

        loaded = 0
        max_event_counter = self.event_counter
        for line in lines:
            if not line.strip():
                continue
            try:
                event = json.loads(line)
                event_id = str(event.get("id") or "")
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                LOGGER.warning("Skipping invalid anomaly event log line: %s", exc)
                continue

            if event_id.startswith("anom_"):
                try:
                    max_event_counter = max(max_event_counter, int(event_id.rsplit("_", 1)[1]))
                except (IndexError, ValueError):
                    pass

            if _event_local_day_key(event) != self.current_daily_key:
                continue

            try:
                label = str(event.get("label") or "")
                track_id = event.get("track_id")
                track_id = int(track_id) if track_id is not None else None
                pose = event.get("object_pose_map") or {}
                object_pose = ObjectPoseMap(
                    x=float(pose["x"]),
                    y=float(pose["y"]),
                    z=float(pose.get("z", 0.0)),
                )
            except (KeyError, TypeError, ValueError) as exc:
                LOGGER.warning("Skipping invalid current-day anomaly event: %s", exc)
                continue

            if label:
                before = len(self.reported_anomalies)
                self._remember_reported(label, object_pose, track_id)
                if len(self.reported_anomalies) > before:
                    loaded += 1

        self.event_counter = max_event_counter
        LOGGER.info(
            "Loaded %d remembered anomaly locations for %s from %s "
            "(older days excluded from de-duplication)",
            loaded,
            self.current_daily_key,
            self.event_log_path,
        )

    def _create_anomaly_event(
        self,
        group: DetectionGroup,
        cluster: AnomalyClusterSummary,
        frame: np.ndarray,
        source_msg: Dict[str, Any],
    ) -> None:
        self.event_counter += 1
        event_id = f"anom_{self.event_counter:05d}"
        detection = max(group.detections, key=lambda value: value.confidence)
        label_slug = _slug(detection.label)
        original_path: Optional[Path] = None
        annotated_path: Optional[Path] = None

        robot_pose = self.latest_pose
        if robot_pose is None:
            return

        if self.config.save_per_event_images:
            original_path = self.original_dir / f"{event_id}_{label_slug}.jpg"
            annotated_path = self.annotated_dir / f"{event_id}_{label_slug}.jpg"
            annotated = annotate_frame(frame.copy(), group.detections)
            _write_image(original_path, frame)
            _write_image(annotated_path, annotated)

        snapshot_path: Optional[Path] = None
        refreshed_summary = self._refresh_daily_map_summary(
            source_msg,
            force=True,
            extra_reported=ReportedAnomaly(
                group.label,
                cluster.object_pose,
                _best_track_id(group.detections),
            ),
        )
        if refreshed_summary is not None:
            snapshot_path, _snapshot_image = refreshed_summary

        event = build_event(
            event_id=event_id,
            detection=detection,
            robot_pose=robot_pose,
            object_pose=cluster.object_pose,
            cluster_id=cluster.cluster_id,
            cluster_count=cluster.count,
            cluster_merge_radius_m=self.config.cluster_merge_radius_m,
            distance_estimate=group.distance_estimate,
            bearing_source=group.bearing_source,
            ttl_sec=self.config.marker_ttl_s,
            original_image=original_path,
            annotated_image=annotated_path,
            map_snapshot=snapshot_path,
            daily_map_summary=snapshot_path,
            event_log=self.event_log_path,
        )
        self.event_writer.append(event)
        self._remember_reported(
            group.label,
            cluster.object_pose,
            _best_track_id(group.detections),
        )
        self._publish_event(event)
        self._publish_debug_image(annotate_frame(frame.copy(), group.detections), source_msg)
        self._mark_cooldown(group.label, cluster.object_pose)
        LOGGER.info(
            "Anomaly %s label=%s confidence=%.2f cluster=%s count=%d "
            "original=%s annotated=%s daily_map=%s",
            event_id,
            detection.label,
            detection.confidence,
            cluster.cluster_id,
            cluster.count,
            original_path or "-",
            annotated_path or "-",
            snapshot_path or "-",
        )

    def _distance_for_detection(
        self,
        detection: Detection,
        image_width: int,
        image_height: int,
        source_msg: Dict[str, Any],
    ) -> DistanceEstimate:
        if self.config.use_depth_distance:
            measurement = self._depth_distance_for_detection(
                detection, image_width, image_height, source_msg
            )
            if measurement is not None:
                return measurement

        if self.config.use_laser_distance:
            if self.latest_scan is None:
                self._log_missing_scan()
            else:
                distance_m = estimate_laser_distance_m(
                    scan=self.latest_scan,
                    bbox_xyxy=detection.bbox_xyxy,
                    image_width=image_width,
                    camera_horizontal_fov_deg=self.config.camera_horizontal_fov_deg,
                    camera_yaw_offset_deg=self.config.camera_yaw_offset_deg,
                    half_window_deg=self.config.laser_window_deg,
                    min_distance_m=self.config.laser_min_distance_m,
                    max_distance_m=self.config.laser_max_distance_m,
                    camera_intrinsics=(
                        self.camera_intrinsics if self.config.use_camera_intrinsics else None
                    ),
                )
                if distance_m is not None:
                    return DistanceEstimate(
                        distance_m=distance_m,
                        source="laser",
                        uncertainty_m=self.config.laser_distance_uncertainty_m,
                    )
                LOGGER.info(
                    "No valid %s range for %s bbox; falling back to %.2f m",
                    self.config.scan_topic,
                    detection.label,
                    self.config.default_anomaly_distance_m,
                )
        return DistanceEstimate(
            distance_m=self.config.default_anomaly_distance_m,
            source="default",
            uncertainty_m=self.config.default_distance_uncertainty_m,
        )

    def _depth_distance_for_detection(
        self,
        detection: Detection,
        image_width: int,
        image_height: int,
        source_msg: Dict[str, Any],
    ) -> Optional[DistanceEstimate]:
        selected = self._select_depth_frame(source_msg)
        if selected is None:
            return None
        depth, age_s = selected
        try:
            measurement = estimate_depth_measurement(
                depth_image=depth,
                bbox_xyxy=detection.bbox_xyxy,
                image_width=image_width,
                image_height=image_height,
                min_distance_m=self.config.depth_min_distance_m,
                max_distance_m=self.config.depth_max_distance_m,
                roi_scale=self.config.depth_roi_scale,
                min_valid_pixels=self.config.depth_min_valid_pixels,
                percentile=self.config.depth_distance_percentile,
                age_s=age_s,
                object_mask=detection.mask if self.config.segmentation_enabled else None,
                mask_erode_px=self.config.segmentation_depth_mask_erode_px,
            )
        except Exception as exc:
            LOGGER.warning("Failed to estimate depth distance: %s", exc)
            return None

        if measurement is not None:
            LOGGER.debug(
                "Using depth distance %.2f m (uncertainty %.3f m, age %.3f s) for %s",
                measurement.distance_m,
                measurement.uncertainty_m or 0.0,
                measurement.age_s or 0.0,
                detection.label,
            )
        return measurement

    def _select_depth_frame(
        self, source_msg: Dict[str, Any]
    ) -> Optional[tuple[np.ndarray, float]]:
        if not self.depth_buffer:
            return None
        now = time.monotonic()
        camera_stamp = header_stamp_seconds(source_msg)
        candidates = [
            item for item in self.depth_buffer if now - item[1] <= self.config.depth_max_age_s
        ]
        if not candidates:
            return None

        if camera_stamp is not None and self.config.depth_sync_tolerance_s > 0.0:
            stamped = [item for item in candidates if item[0] is not None]
            if stamped:
                selected = min(stamped, key=lambda item: abs(float(item[0]) - camera_stamp))
                timestamp_delta = abs(float(selected[0]) - camera_stamp)
                if timestamp_delta > self.config.depth_sync_tolerance_s:
                    LOGGER.debug(
                        "Closest depth frame is %.3f s from RGB frame (limit %.3f s)",
                        timestamp_delta,
                        self.config.depth_sync_tolerance_s,
                    )
                    return None
                return selected[2], timestamp_delta

        selected = candidates[-1]
        return selected[2], now - selected[1]

    def _publish_debug_stream_if_due(
        self,
        detections: List[Detection],
        frame: np.ndarray,
        source_msg: Dict[str, Any],
    ) -> None:
        if not self.config.debug_image_always_stream and not (detections and self.config.debug_image_on_detection):
            return
        now = time.monotonic()
        if now < self.next_debug_image_publish:
            return
        self.next_debug_image_publish = now + (1.0 / self.config.debug_image_publish_hz)
        debug_frame = annotate_frame(frame.copy(), detections) if detections else frame
        self._publish_debug_image(debug_frame, source_msg)

    def _publish_privacy_stream_if_due(
        self,
        detections: List[Detection],
        frame: np.ndarray,
        source_msg: Dict[str, Any],
    ) -> None:
        if not self.config.privacy_image_enabled:
            return
        now = time.monotonic()
        if now < self.next_privacy_image_publish:
            return
        self.next_privacy_image_publish = now + (
            1.0 / self.config.privacy_image_publish_hz
        )
        privacy_frame = blur_except_detections(
            frame,
            detections,
            kernel_size=self.config.privacy_blur_kernel_size,
            padding_ratio=self.config.privacy_bbox_padding_ratio,
            use_segmentation_masks=self.config.privacy_use_segmentation_masks,
            draw_track_id=self.config.privacy_draw_track_id,
            draw_mask_overlay=self.config.privacy_draw_mask_overlay,
            mask_overlay_alpha=self.config.privacy_mask_overlay_alpha,
        )
        self._publish_privacy_image(privacy_frame, source_msg)

    def _publish_detection_3d_if_due(
        self,
        detections: List[Detection],
        frame: np.ndarray,
        source_msg: Dict[str, Any],
    ) -> None:
        if not self.config.detection_3d_enabled or not detections:
            return
        now = time.monotonic()
        if now < self.next_detection_3d_publish:
            return
        self.next_detection_3d_publish = now + (
            1.0 / self.config.detection_3d_publish_hz
        )
        intrinsics = self.camera_intrinsics
        selected = self._select_depth_frame(source_msg)
        if intrinsics is None or selected is None:
            return
        depth, _depth_age_s = selected
        height, width = frame.shape[:2]
        bounded = []
        for detection in detections:
            if self.config.detection_3d_require_mask and detection.mask is None:
                continue
            bounds = estimate_3d_bounds_camera(
                depth_image=depth,
                bbox_xyxy=detection.bbox_xyxy,
                image_width=width,
                image_height=height,
                intrinsics=intrinsics,
                object_mask=detection.mask,
                min_distance_m=self.config.depth_min_distance_m,
                max_distance_m=self.config.depth_max_distance_m,
                min_valid_points=self.config.detection_3d_min_valid_points,
                lower_percentile=self.config.detection_3d_lower_percentile,
                upper_percentile=self.config.detection_3d_upper_percentile,
                mask_erode_px=self.config.segmentation_depth_mask_erode_px,
                sample_stride=self.config.detection_3d_sample_stride,
                minimum_thickness_m=self.config.detection_3d_minimum_thickness_m,
            )
            if bounds is not None:
                bounded.append((detection, bounds))
        if not bounded:
            return

        header = source_msg.get("header") or {}
        frame_id = (
            self.config.detection_3d_frame_id
            or header.get("frame_id")
            or "camera_color_optical_frame"
        )
        marker_array = build_detection_3d_marker_array(
            bounded,
            frame_id=str(frame_id),
            stamp=header.get("stamp"),
            ttl_s=self.config.detection_3d_ttl_s,
            line_width_m=self.config.detection_3d_line_width_m,
            text_enabled=self.config.detection_3d_text_enabled,
            text_height_m=self.config.detection_3d_text_height_m,
            text_show_label=self.config.detection_3d_text_show_label,
            text_show_confidence=self.config.detection_3d_text_show_confidence,
            text_show_distance=self.config.detection_3d_text_show_distance,
        )
        self.rosbridge.publish(self.config.detection_3d_topic, marker_array)

    def _daily_map_path(self) -> Path:
        return self.daily_map_dir / f"anomalies_{self.current_daily_key}.png"

    def _reported_summaries(
        self,
        extra_reported: Optional[ReportedAnomaly] = None,
    ) -> List[AnomalyClusterSummary]:
        reported = list(self.reported_anomalies)
        if extra_reported is not None:
            if self._reported_index(
                extra_reported.label,
                extra_reported.object_pose,
                extra_reported.track_id,
            ) is None:
                reported.append(extra_reported)
        return [
            AnomalyClusterSummary(
                cluster_id=f"reported_{index:05d}",
                label=item.label,
                object_pose=item.object_pose,
                count=1,
            )
            for index, item in enumerate(reported, start=1)
        ]

    def _refresh_daily_map_summary(
        self,
        source_msg: Dict[str, Any],
        force: bool = False,
        extra_reported: Optional[ReportedAnomaly] = None,
    ) -> Optional[tuple[Path, np.ndarray]]:
        if not self.config.daily_map_summary:
            return None
        if self.latest_map is None:
            self._log_missing_map()
            return None
        now = time.monotonic()
        if not force and now < self.next_daily_summary_refresh:
            return None
        self.next_daily_summary_refresh = now + 15.0
        try:
            snapshot_path = self._daily_map_path()
            snapshot_image = save_daily_map_summary(
                self.latest_map,
                self._reported_summaries(extra_reported),
                snapshot_path,
            )
        except Exception as exc:
            LOGGER.warning("Failed to generate daily map summary: %s", exc)
            return None
        if self.config.daily_map_summary_topic_publish:
            self._publish_map_snapshot(snapshot_image, source_msg)
        return snapshot_path, snapshot_image

    def _publish_event(self, event: Dict[str, Any]) -> None:
        self.rosbridge.publish(self.config.event_topic, {"data": json.dumps(event, separators=(",", ":"))})
        self.rosbridge.publish(self.config.readable_event_topic, {"data": build_readable_event(event)})

    def _clear_existing_markers(self) -> None:
        self.rosbridge.publish(self.config.marker_topic, self.markers.build_delete_all_marker_array())

    def _publish_debug_image(self, image: np.ndarray, source_msg: Dict[str, Any]) -> None:
        header = source_msg.get("header") or {}
        stamp = header.get("stamp")
        frame_id = header.get("frame_id") or "camera"
        encoded = encode_image(image, ".jpg", quality=self.config.jpeg_quality)
        self.rosbridge.publish(
            self.config.debug_image_topic,
            compressed_image_msg(encoded, "jpeg", frame_id=frame_id, stamp=stamp),
        )

    def _publish_privacy_image(
        self, image: np.ndarray, source_msg: Dict[str, Any]
    ) -> None:
        header = source_msg.get("header") or {}
        stamp = header.get("stamp")
        frame_id = header.get("frame_id") or "camera"
        encoded = encode_image(image, ".jpg", quality=self.config.jpeg_quality)
        self.rosbridge.publish(
            self.config.privacy_image_topic,
            compressed_image_msg(encoded, "jpeg", frame_id=frame_id, stamp=stamp),
        )

    def _publish_map_snapshot(self, image: np.ndarray, source_msg: Dict[str, Any]) -> None:
        del source_msg
        encoded = encode_image(image, ".png")
        self.rosbridge.publish(
            self.config.map_snapshot_topic,
            compressed_image_msg(encoded, "png", frame_id=self.config.map_frame_id),
        )

    def _publish_markers_if_due(self) -> None:
        now = time.monotonic()
        if now < self.next_marker_publish:
            return
        self.next_marker_publish = now + (1.0 / self.config.marker_republish_hz)
        marker_array = self.markers.build_marker_array()
        if marker_array["markers"]:
            self.rosbridge.publish(self.config.marker_topic, marker_array)

    def _log_missing_pose(self) -> None:
        now = time.monotonic()
        if now - self.last_missing_pose_log >= 5.0:
            self.last_missing_pose_log = now
            LOGGER.warning("Bottle detected, but no %s pose has arrived yet", self.config.robot_pose_topic)

    def _log_missing_map(self) -> None:
        now = time.monotonic()
        if now - self.last_missing_map_log >= 5.0:
            self.last_missing_map_log = now
            LOGGER.warning("No %s map has arrived yet; event will not include a map snapshot", self.config.map_topic)

    def _log_missing_scan(self) -> None:
        now = time.monotonic()
        if now - self.last_missing_scan_log >= 5.0:
            self.last_missing_scan_log = now
            LOGGER.warning(
                "No %s scan has arrived yet; using default anomaly distance %.2f m",
                self.config.scan_topic,
                self.config.default_anomaly_distance_m,
            )


def annotate_frame(frame: np.ndarray, detections: List[Detection]) -> np.ndarray:
    height, width = frame.shape[:2]
    font_scale = max(0.35, min(0.55, width / 1280.0))
    thickness = max(1, int(round(width / 640.0)))
    for detection in detections:
        mask = _normalized_detection_mask(detection, height, width)
        if mask is not None:
            tint = np.zeros_like(frame)
            tint[:, :, 1] = 220
            blended = cv2.addWeighted(frame, 0.75, tint, 0.25, 0.0)
            frame[mask] = blended[mask]
            contours, _ = cv2.findContours(
                mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(frame, contours, -1, (0, 255, 0), thickness)
        x1, y1, x2, y2 = detection.bbox_xyxy
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), thickness)
        track_text = f" #{detection.track_id}" if detection.track_id is not None else ""
        label = f"{detection.label}{track_text} {detection.confidence:.2f}"
        (text_width, text_height), baseline = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            thickness,
        )
        label_x = max(0, min(x1, width - text_width - 6))
        label_y = max(text_height + baseline + 6, y1 - 6)
        label_y = min(label_y, height - 2)
        cv2.rectangle(
            frame,
            (label_x, label_y - text_height - baseline - 4),
            (label_x + text_width + 6, label_y + 3),
            (0, 0, 0),
            -1,
        )
        cv2.putText(
            frame,
            label,
            (label_x + 3, label_y - baseline),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 255),
            thickness,
            cv2.LINE_AA,
        )
    return frame


def blur_except_detections(
    frame: np.ndarray,
    detections: List[Detection],
    kernel_size: int,
    padding_ratio: float = 0.0,
    use_segmentation_masks: bool = True,
    draw_track_id: bool = False,
    draw_mask_overlay: bool = False,
    mask_overlay_alpha: float = 0.25,
) -> np.ndarray:
    if frame.size == 0:
        return frame.copy()

    kernel = max(3, int(kernel_size))
    if kernel % 2 == 0:
        kernel += 1
    output = cv2.GaussianBlur(frame, (kernel, kernel), 0)
    height, width = frame.shape[:2]
    padding = max(0.0, min(0.5, float(padding_ratio)))
    overlay_alpha = max(0.0, min(1.0, float(mask_overlay_alpha)))

    for detection in detections:
        x1, y1, x2, y2 = [int(value) for value in detection.bbox_xyxy]
        pad_x = int(round(max(0, x2 - x1) * padding))
        pad_y = int(round(max(0, y2 - y1) * padding))
        x1 = max(0, min(width, x1 - pad_x))
        y1 = max(0, min(height, y1 - pad_y))
        x2 = max(0, min(width, x2 + pad_x))
        y2 = max(0, min(height, y2 + pad_y))
        if x2 <= x1 or y2 <= y1:
            continue

        mask = (
            _normalized_detection_mask(detection, height, width)
            if use_segmentation_masks
            else None
        )
        if mask is not None:
            dilation_px = max(pad_x, pad_y)
            if dilation_px > 0:
                kernel_width = dilation_px * 2 + 1
                mask = cv2.dilate(
                    mask.astype(np.uint8),
                    np.ones((kernel_width, kernel_width), dtype=np.uint8),
                ).astype(bool)
            output[mask] = frame[mask]
            if draw_mask_overlay and overlay_alpha > 0.0:
                tint = np.zeros_like(output)
                tint[:, :, 1] = 220
                blended = cv2.addWeighted(
                    output, 1.0 - overlay_alpha, tint, overlay_alpha, 0.0
                )
                output[mask] = blended[mask]
                contours, _ = cv2.findContours(
                    mask.astype(np.uint8),
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE,
                )
                cv2.drawContours(output, contours, -1, (0, 255, 0), 2)
        else:
            output[y1:y2, x1:x2] = frame[y1:y2, x1:x2]

        if draw_track_id and detection.track_id is not None:
            label = f"{detection.label} #{detection.track_id}"
            text_y = max(18, y1 - 6)
            cv2.putText(
                output,
                label,
                (x1, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

    return output


def _normalized_detection_mask(
    detection: Detection, height: int, width: int
) -> Optional[np.ndarray]:
    if detection.mask is None or detection.mask.size == 0:
        return None
    mask = np.asarray(detection.mask, dtype=np.uint8)
    mask = np.squeeze(mask)
    if mask.ndim != 2:
        return None
    if mask.shape != (height, width):
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
    return mask.astype(bool)


def _best_track_id(detections: List[Detection]) -> Optional[int]:
    tracked = [item for item in detections if item.track_id is not None]
    if not tracked:
        return None
    return max(tracked, key=lambda item: item.confidence).track_id


def _detection_sharpness(frame: np.ndarray, detection: Detection) -> float:
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = [int(value) for value in detection.bbox_xyxy]
    x1, x2 = max(0, min(width, x1)), max(0, min(width, x2))
    y1, y2 = max(0, min(height, y1)), max(0, min(height, y2))
    if x2 <= x1 or y2 <= y1:
        return 0.0
    crop = frame[y1:y2, x1:x2]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    mask = _normalized_detection_mask(detection, height, width)
    crop_mask = mask[y1:y2, x1:x2].astype(np.uint8) if mask is not None else None
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    if crop_mask is not None and np.count_nonzero(crop_mask) >= 9:
        values = laplacian[crop_mask > 0]
        return float(np.var(values))
    return float(laplacian.var())


def _write_image(
    path: Path,
    image: np.ndarray,
    quality: Optional[int] = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    params = []
    if quality is not None and path.suffix.lower() in {".jpg", ".jpeg"}:
        params = [int(cv2.IMWRITE_JPEG_QUALITY), max(1, min(100, int(quality)))]
    if not cv2.imwrite(str(path), image, params):
        raise ValueError(f"Failed to write image: {path}")


def _distance_xy(first: ObjectPoseMap, second: ObjectPoseMap) -> float:
    return float(np.hypot(first.x - second.x, first.y - second.y))


def _blend_pose(current: ObjectPoseMap, new_pose: ObjectPoseMap, new_weight: float) -> ObjectPoseMap:
    current_weight = 1.0 - new_weight
    return ObjectPoseMap(
        x=(current.x * current_weight) + (new_pose.x * new_weight),
        y=(current.y * current_weight) + (new_pose.y * new_weight),
        z=(current.z * current_weight) + (new_pose.z * new_weight),
    )


def _slug(value: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in value.lower()).strip("_") or "anomaly"


def _local_day_key() -> str:
    return datetime.now().astimezone().date().isoformat()


def _event_local_day_key(event: Dict[str, Any]) -> Optional[str]:
    raw = str(event.get("timestamp") or "").strip()
    if not raw:
        return None
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.astimezone()
        return parsed.astimezone().date().isoformat()
    except ValueError:
        return None


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Jetson YOLO anomaly rosbridge client")
    parser.add_argument("--config", default="/workspace/config/anomaly_rosbridge.yaml", help="YAML config file")
    parser.add_argument("--log-level", default="INFO", help="Python logging level")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    config = load_config(args.config)
    client = JetsonYoloRosbridgeClient(config)
    signal.signal(signal.SIGINT, client.request_stop)
    signal.signal(signal.SIGTERM, client.request_stop)
    client.run_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())

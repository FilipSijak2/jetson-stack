from __future__ import annotations

import argparse
import json
import logging
import signal
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from .config import AppConfig, load_config
from .event_schema import EventJsonlWriter, build_event, build_readable_event
from .localization import estimate_laser_distance_m, estimate_object_pose_map
from .map_snapshot import save_daily_map_summary
from .marker_manager import MarkerManager
from .models import AnomalyClusterSummary, Detection, LaserScan, ObjectPoseMap, OccupancyGridMap, RobotPoseMap
from .ros_messages import (
    compressed_image_msg,
    decode_compressed_image,
    encode_image,
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


@dataclass(frozen=True)
class DetectionGroup:
    label: str
    detections: List[Detection]
    object_pose: ObjectPoseMap


class JetsonYoloRosbridgeClient:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.artifact_root = Path(config.artifact_root)
        self.original_dir = self.artifact_root / "images" / "original"
        self.annotated_dir = self.artifact_root / "images" / "annotated"
        self.map_dir = self.artifact_root / "map_images"
        self.daily_map_dir = self.map_dir / "daily"
        self.event_log_path = self.artifact_root / "events.jsonl"
        paths = [self.map_dir, self.daily_map_dir]
        if config.save_per_event_images:
            paths.extend([self.original_dir, self.annotated_dir])
        for path in paths:
            path.mkdir(parents=True, exist_ok=True)

        self.event_writer = EventJsonlWriter(self.event_log_path)
        self.detector = YoloDetector(
            model_path=config.yolo_model_path,
            confidence_threshold=config.confidence_threshold,
            anomaly_classes=config.anomaly_classes,
            mock_mode=config.mock_mode,
            logger=LOGGER,
        )
        self.markers = MarkerManager(
            frame_id=config.map_frame_id,
            merge_radius_m=config.cluster_merge_radius_m,
            object_marker_size_m=config.marker_object_size_m,
            text_height_m=config.marker_text_height_m,
            text_z_offset_m=config.marker_text_z_offset_m,
            text_show_count=config.marker_text_show_count,
        )
        self.rosbridge = RosbridgeClient(config.rosbridge_url, LOGGER)
        self.latest_map: Optional[OccupancyGridMap] = None
        self.latest_pose: Optional[RobotPoseMap] = None
        self.latest_scan: Optional[LaserScan] = None
        self.last_event_by_label: Dict[str, float] = {}
        self.last_event_clusters: Dict[str, float] = {}
        self.reported_anomalies: List[tuple[str, ObjectPoseMap]] = []
        self.daily_clusters: Dict[str, AnomalyClusterSummary] = {}
        self.frame_count = 0
        self.event_counter = 0
        self._load_reported_anomalies()
        self.stop_requested = False
        self.next_marker_publish = 0.0
        self.next_daily_summary_refresh = 0.0
        self.next_debug_image_publish = 0.0
        self.last_missing_pose_log = 0.0
        self.last_missing_map_log = 0.0
        self.last_missing_scan_log = 0.0

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
        if self.config.use_laser_distance and self.config.scan_topic:
            self.rosbridge.subscribe(self.config.scan_topic, "sensor_msgs/LaserScan", queue_length=1, throttle_rate=100)
        pose_type = "geometry_msgs/PoseStamped"
        if self.config.robot_pose_topic == "/amcl_pose":
            pose_type = "geometry_msgs/PoseWithCovarianceStamped"
        self.rosbridge.subscribe(self.config.robot_pose_topic, pose_type, queue_length=1, throttle_rate=500)

        self.rosbridge.advertise(self.config.event_topic, "std_msgs/String")
        self.rosbridge.advertise(self.config.readable_event_topic, "std_msgs/String")
        self.rosbridge.advertise(self.config.marker_topic, "visualization_msgs/MarkerArray")
        self.rosbridge.advertise(self.config.debug_image_topic, "sensor_msgs/CompressedImage")
        self.rosbridge.advertise(self.config.map_snapshot_topic, "sensor_msgs/CompressedImage")

        LOGGER.info(
            "Subscribed camera=%s map=%s pose=%s scan=%s laser_distance=%s",
            self.config.camera_topic,
            self.config.map_topic,
            self.config.robot_pose_topic,
            self.config.scan_topic if self.config.use_laser_distance else "disabled",
            self.config.use_laser_distance,
        )
        LOGGER.info(
            "Publishing events=%s readable_events=%s markers=%s debug_image=%s map_snapshot=%s",
            self.config.event_topic,
            self.config.readable_event_topic,
            self.config.marker_topic,
            self.config.debug_image_topic,
            self.config.map_snapshot_topic,
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
        elif topic == self.config.map_topic:
            self._on_map(msg)
        elif topic == self.config.robot_pose_topic:
            self._on_pose(msg)
        elif topic == self.config.scan_topic:
            self._on_scan(msg)

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

    def _on_camera_image(self, msg: Dict[str, Any]) -> None:
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
        self._publish_debug_stream_if_due(anomalies, frame, msg)
        if not anomalies:
            return

        if self.latest_pose is None:
            self._log_missing_pose()
            return

        located = [self._locate_detection(detection, frame.shape[1]) for detection in anomalies]
        for group in self._group_located_detections(located):
            cluster = self.markers.add_or_update(
                label=group.label,
                object_pose=group.object_pose,
                observed_count=len(group.detections),
                ttl_s=self.config.marker_ttl_s,
            )
            self.daily_clusters[cluster.cluster_id] = cluster
            if self._already_reported(group.label, cluster.object_pose):
                self._remember_reported(group.label, cluster.object_pose)
                self._refresh_daily_map_summary(msg)
                continue
            if not self._cooldown_ready(group.label, cluster.object_pose):
                continue
            try:
                self._create_anomaly_event(group, cluster, frame, msg)
            except Exception as exc:
                LOGGER.warning("Failed to create anomaly event: %s", exc)

    def _filter_anomalies(self, detections: List[Detection]) -> List[Detection]:
        anomaly_labels = {label.strip() for label in self.config.anomaly_classes}
        return [
            detection
            for detection in detections
            if detection.label in anomaly_labels and detection.confidence >= self.config.confidence_threshold
        ]

    def _locate_detection(self, detection: Detection, image_width: int) -> LocatedDetection:
        robot_pose = self.latest_pose
        if robot_pose is None:
            raise RuntimeError("Cannot locate detection without robot pose")
        object_pose = estimate_object_pose_map(
            robot_pose=robot_pose,
            bbox_xyxy=detection.bbox_xyxy,
            image_width=image_width,
            default_distance_m=self.config.default_anomaly_distance_m,
            camera_horizontal_fov_deg=self.config.camera_horizontal_fov_deg,
            camera_yaw_offset_deg=self.config.camera_yaw_offset_deg,
            measured_distance_m=self._distance_for_detection(detection, image_width),
        )
        return LocatedDetection(detection=detection, object_pose=object_pose)

    def _group_located_detections(self, located: List[LocatedDetection]) -> List[DetectionGroup]:
        groups: List[List[LocatedDetection]] = []
        for item in sorted(located, key=lambda value: value.detection.confidence, reverse=True):
            target_group = None
            for group in groups:
                if group[0].detection.label != item.detection.label:
                    continue
                center = self._group_center(group)
                distance = _distance_xy(center, item.object_pose)
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
            )
            for group in groups
        ]

    def _group_center(self, group: List[LocatedDetection]) -> ObjectPoseMap:
        count = max(1, len(group))
        return ObjectPoseMap(
            x=sum(item.object_pose.x for item in group) / count,
            y=sum(item.object_pose.y for item in group) / count,
            z=sum(item.object_pose.z for item in group) / count,
        )

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

    def _already_reported(self, label: str, object_pose: ObjectPoseMap) -> bool:
        return self._reported_index(label, object_pose) is not None

    def _reported_index(self, label: str, object_pose: ObjectPoseMap) -> Optional[int]:
        radius = max(0.01, self.config.cluster_merge_radius_m)
        for index, (reported_label, reported_pose) in enumerate(self.reported_anomalies):
            if reported_label == label and _distance_xy(reported_pose, object_pose) <= radius:
                return index
        return None

    def _remember_reported(self, label: str, object_pose: ObjectPoseMap) -> None:
        index = self._reported_index(label, object_pose)
        if index is None:
            self.reported_anomalies.append((label, object_pose))
        else:
            _, previous_pose = self.reported_anomalies[index]
            self.reported_anomalies[index] = (label, _blend_pose(previous_pose, object_pose, new_weight=0.25))

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
                label = str(event.get("label") or "")
                pose = event.get("object_pose_map") or {}
                event_id = str(event.get("id") or "")
                object_pose = ObjectPoseMap(
                    x=float(pose["x"]),
                    y=float(pose["y"]),
                    z=float(pose.get("z", 0.0)),
                )
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                LOGGER.warning("Skipping invalid anomaly event log line: %s", exc)
                continue

            if label:
                before = len(self.reported_anomalies)
                self._remember_reported(label, object_pose)
                if len(self.reported_anomalies) > before:
                    loaded += 1

            if event_id.startswith("anom_"):
                try:
                    max_event_counter = max(max_event_counter, int(event_id.rsplit("_", 1)[1]))
                except (IndexError, ValueError):
                    pass

        self.event_counter = max_event_counter
        if loaded:
            LOGGER.info("Loaded %d remembered anomaly locations from %s", loaded, self.event_log_path)

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

        self._remember_reported(group.label, cluster.object_pose)
        snapshot_path: Optional[Path] = None
        refreshed_summary = self._refresh_daily_map_summary(source_msg, force=True)
        if refreshed_summary is not None:
            snapshot_path, _snapshot_image = refreshed_summary

        if self.config.save_per_event_images:
            original_path = self.original_dir / f"{event_id}_{label_slug}.jpg"
            annotated_path = self.annotated_dir / f"{event_id}_{label_slug}.jpg"
            annotated = annotate_frame(frame.copy(), group.detections)
            _write_image(original_path, frame)
            _write_image(annotated_path, annotated)

        event = build_event(
            event_id=event_id,
            detection=detection,
            robot_pose=robot_pose,
            object_pose=cluster.object_pose,
            cluster_id=cluster.cluster_id,
            cluster_count=cluster.count,
            cluster_merge_radius_m=self.config.cluster_merge_radius_m,
            ttl_sec=self.config.marker_ttl_s,
            original_image=original_path,
            annotated_image=annotated_path,
            map_snapshot=snapshot_path,
            daily_map_summary=snapshot_path,
            event_log=self.event_log_path,
        )
        self.event_writer.append(event)
        self._publish_event(event)
        self._publish_debug_image(annotate_frame(frame.copy(), group.detections), source_msg)
        self._mark_cooldown(group.label, cluster.object_pose)
        LOGGER.info(
            "Anomaly %s label=%s confidence=%.2f cluster=%s count=%d",
            event_id,
            detection.label,
            detection.confidence,
            cluster.cluster_id,
            cluster.count,
        )

    def _distance_for_detection(self, detection: Detection, image_width: int) -> Optional[float]:
        if not self.config.use_laser_distance:
            return None
        if self.latest_scan is None:
            self._log_missing_scan()
            return None
        distance_m = estimate_laser_distance_m(
            scan=self.latest_scan,
            bbox_xyxy=detection.bbox_xyxy,
            image_width=image_width,
            camera_horizontal_fov_deg=self.config.camera_horizontal_fov_deg,
            camera_yaw_offset_deg=self.config.camera_yaw_offset_deg,
            half_window_deg=self.config.laser_window_deg,
            min_distance_m=self.config.laser_min_distance_m,
            max_distance_m=self.config.laser_max_distance_m,
        )
        if distance_m is None:
            LOGGER.info(
                "No valid %s range for %s bbox; falling back to %.2f m",
                self.config.scan_topic,
                detection.label,
                self.config.default_anomaly_distance_m,
            )
        return distance_m

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

    def _daily_map_path(self) -> Path:
        day = datetime.now().strftime("%Y-%m-%d")
        return self.daily_map_dir / f"anomalies_{day}.png"

    def _reported_summaries(self) -> List[AnomalyClusterSummary]:
        return [
            AnomalyClusterSummary(
                cluster_id=f"reported_{index:05d}",
                label=label,
                object_pose=pose,
                count=1,
            )
            for index, (label, pose) in enumerate(self.reported_anomalies, start=1)
        ]

    def _refresh_daily_map_summary(
        self,
        source_msg: Dict[str, Any],
        force: bool = False,
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
                self._reported_summaries(),
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

    def _publish_debug_image(self, image: np.ndarray, source_msg: Dict[str, Any]) -> None:
        header = source_msg.get("header") or {}
        stamp = header.get("stamp")
        frame_id = header.get("frame_id") or "camera"
        encoded = encode_image(image, ".jpg", quality=self.config.jpeg_quality)
        self.rosbridge.publish(
            self.config.debug_image_topic,
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
        x1, y1, x2, y2 = detection.bbox_xyxy
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), thickness)
        label = f"{detection.label} {detection.confidence:.2f}"
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


def _write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
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

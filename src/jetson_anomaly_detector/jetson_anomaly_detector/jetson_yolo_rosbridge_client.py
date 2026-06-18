from __future__ import annotations

import argparse
import json
import logging
import signal
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from .config import AppConfig, load_config
from .event_schema import EventJsonlWriter, build_event
from .localization import estimate_object_pose_map
from .map_snapshot import save_map_snapshot
from .marker_manager import MarkerManager
from .models import Detection, OccupancyGridMap, RobotPoseMap
from .ros_messages import (
    compressed_image_msg,
    decode_compressed_image,
    encode_image,
    parse_occupancy_grid,
    parse_robot_pose,
)
from .rosbridge_ws import RosbridgeClient
from .yolo_detector import YoloDetector


LOGGER = logging.getLogger("jetson_yolo_rosbridge_client")


class JetsonYoloRosbridgeClient:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.artifact_root = Path(config.artifact_root)
        self.original_dir = self.artifact_root / "images" / "original"
        self.annotated_dir = self.artifact_root / "images" / "annotated"
        self.map_dir = self.artifact_root / "map_images"
        self.event_log_path = self.artifact_root / "events.jsonl"
        for path in (self.original_dir, self.annotated_dir, self.map_dir):
            path.mkdir(parents=True, exist_ok=True)

        self.event_writer = EventJsonlWriter(self.event_log_path)
        self.detector = YoloDetector(
            model_path=config.yolo_model_path,
            confidence_threshold=config.confidence_threshold,
            anomaly_classes=config.anomaly_classes,
            mock_mode=config.mock_mode,
            logger=LOGGER,
        )
        self.markers = MarkerManager(frame_id=config.map_frame_id)
        self.rosbridge = RosbridgeClient(config.rosbridge_url, LOGGER)
        self.latest_map: Optional[OccupancyGridMap] = None
        self.latest_pose: Optional[RobotPoseMap] = None
        self.last_event_by_label: Dict[str, float] = {}
        self.frame_count = 0
        self.event_counter = 0
        self.stop_requested = False
        self.next_marker_publish = 0.0
        self.last_missing_pose_log = 0.0
        self.last_missing_map_log = 0.0

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
        pose_type = "geometry_msgs/PoseStamped"
        if self.config.robot_pose_topic == "/amcl_pose":
            pose_type = "geometry_msgs/PoseWithCovarianceStamped"
        self.rosbridge.subscribe(self.config.robot_pose_topic, pose_type, queue_length=1, throttle_rate=500)

        self.rosbridge.advertise(self.config.event_topic, "std_msgs/String")
        self.rosbridge.advertise(self.config.marker_topic, "visualization_msgs/MarkerArray")
        self.rosbridge.advertise(self.config.debug_image_topic, "sensor_msgs/CompressedImage")
        self.rosbridge.advertise(self.config.map_snapshot_topic, "sensor_msgs/CompressedImage")

        LOGGER.info("Subscribed camera=%s map=%s pose=%s", self.config.camera_topic, self.config.map_topic, self.config.robot_pose_topic)
        LOGGER.info(
            "Publishing events=%s markers=%s debug_image=%s map_snapshot=%s",
            self.config.event_topic,
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
        if not anomalies:
            return

        strongest = max(anomalies, key=lambda detection: detection.confidence)
        if not self._cooldown_ready(strongest.label):
            return

        if self.latest_pose is None:
            self._log_missing_pose()
            return

        try:
            self._create_anomaly_event(strongest, frame, msg)
        except Exception as exc:
            LOGGER.warning("Failed to create anomaly event: %s", exc)

    def _filter_anomalies(self, detections: List[Detection]) -> List[Detection]:
        anomaly_labels = {label.strip() for label in self.config.anomaly_classes}
        return [
            detection
            for detection in detections
            if detection.label in anomaly_labels and detection.confidence >= self.config.confidence_threshold
        ]

    def _cooldown_ready(self, label: str) -> bool:
        now = time.monotonic()
        last = self.last_event_by_label.get(label, 0.0)
        return now - last >= self.config.detection_cooldown_s

    def _create_anomaly_event(self, detection: Detection, frame: np.ndarray, source_msg: Dict[str, Any]) -> None:
        self.event_counter += 1
        event_id = f"anom_{self.event_counter:05d}"
        label_slug = _slug(detection.label)
        original_path = self.original_dir / f"{event_id}_{label_slug}.jpg"
        annotated_path = self.annotated_dir / f"{event_id}_{label_slug}.jpg"
        map_path = self.map_dir / f"{event_id}_{label_slug}_map.png"

        robot_pose = self.latest_pose
        if robot_pose is None:
            return
        object_pose = estimate_object_pose_map(
            robot_pose=robot_pose,
            bbox_xyxy=detection.bbox_xyxy,
            image_width=frame.shape[1],
            default_distance_m=self.config.default_anomaly_distance_m,
            camera_horizontal_fov_deg=self.config.camera_horizontal_fov_deg,
        )

        annotated = annotate_frame(frame.copy(), [detection])
        _write_image(original_path, frame)
        _write_image(annotated_path, annotated)

        snapshot_image = None
        snapshot_path: Optional[Path] = None
        if self.latest_map is not None:
            try:
                snapshot_image = save_map_snapshot(self.latest_map, object_pose, detection.label, map_path)
                snapshot_path = map_path
            except Exception as exc:
                LOGGER.warning("Failed to generate map snapshot: %s", exc)
        else:
            self._log_missing_map()

        event = build_event(
            event_id=event_id,
            detection=detection,
            robot_pose=robot_pose,
            object_pose=object_pose,
            ttl_sec=self.config.marker_ttl_s,
            original_image=original_path,
            annotated_image=annotated_path,
            map_snapshot=snapshot_path,
            event_log=self.event_log_path,
        )
        self.event_writer.append(event)
        self._publish_event(event)
        self._publish_debug_image(annotated, source_msg)
        if snapshot_image is not None:
            self._publish_map_snapshot(snapshot_image, source_msg)
        self.markers.add(event_id, self.event_counter * 2, detection.label, object_pose, self.config.marker_ttl_s)
        self.last_event_by_label[detection.label] = time.monotonic()
        LOGGER.info("Anomaly %s label=%s confidence=%.2f", event_id, detection.label, detection.confidence)

    def _publish_event(self, event: Dict[str, Any]) -> None:
        self.rosbridge.publish(self.config.event_topic, {"data": json.dumps(event, separators=(",", ":"))})

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


def annotate_frame(frame: np.ndarray, detections: List[Detection]) -> np.ndarray:
    for detection in detections:
        x1, y1, x2, y2 = detection.bbox_xyxy
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        label = f"ANOMALY: {detection.label} {detection.confidence:.2f}"
        cv2.putText(
            frame,
            label,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
    return frame


def _write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise ValueError(f"Failed to write image: {path}")


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

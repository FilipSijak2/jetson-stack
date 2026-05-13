#!/usr/bin/env python3
"""ROS 2 anomaly detector node for Jetson companion compute.

The node is intentionally dependency-light by default. In `mock` mode it verifies
ROS connectivity without ML packages. In `yolo` mode it imports ultralytics only
at runtime, so the package can still be built before the Jetson ML stack is ready.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import rclpy
import yaml
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import String


@dataclass
class Detection:
    label: str
    confidence: float
    bbox_xyxy: List[int]


class AnomalyDetectorNode(Node):
    def __init__(self) -> None:
        super().__init__('jetson_anomaly_detector')
        self.declare_parameter('config_file', '')
        config_file = self.get_parameter('config_file').get_parameter_value().string_value
        self.config = self._load_config(config_file)

        self.image_topic = self.config.get('image_topic', '/camera/realsense/color/image_raw')
        self.event_topic = self.config.get('event_topic', '/anomaly_events')
        self.debug_image_topic = self.config.get('debug_image_topic', '/anomaly/debug_image')
        self.backend = self.config.get('detector_backend', 'mock')
        self.min_confidence = float(self.config.get('min_confidence', 0.35))
        self.inference_every_n_frames = max(1, int(self.config.get('inference_every_n_frames', 5)))
        self.publish_debug_image = bool(self.config.get('publish_debug_image', True))
        self.floor_region_start_ratio = float(self.config.get('floor_region_start_ratio', 0.55))
        self.anomaly_labels = set(self.config.get('anomaly_labels', ['bottle', 'cup', 'backpack', 'chair', 'person']))

        self.bridge = CvBridge()
        self.frame_count = 0
        self.last_event_time = 0.0
        self.min_event_interval_s = float(self.config.get('min_event_interval_s', 1.0))
        self.yolo_model = None

        if self.backend == 'yolo':
            self._init_yolo()
        elif self.backend != 'mock':
            self.get_logger().warn(f"Unknown detector_backend '{self.backend}', falling back to mock")
            self.backend = 'mock'

        self.event_pub = self.create_publisher(String, self.event_topic, 10)
        self.debug_pub = self.create_publisher(Image, self.debug_image_topic, 10) if self.publish_debug_image else None
        self.image_sub = self.create_subscription(
            Image,
            self.image_topic,
            self._on_image,
            qos_profile_sensor_data,
        )
        self.get_logger().info(
            f"Jetson anomaly detector started: backend={self.backend}, image_topic={self.image_topic}, "
            f"event_topic={self.event_topic}"
        )

    def _load_config(self, config_file: str) -> Dict[str, Any]:
        if not config_file:
            self.get_logger().warn('No config_file parameter set; using defaults')
            return {}
        path = Path(config_file)
        if not path.exists():
            self.get_logger().warn(f'Config file not found: {config_file}; using defaults')
            return {}
        with path.open('r', encoding='utf-8') as handle:
            data = yaml.safe_load(handle) or {}
        return data

    def _init_yolo(self) -> None:
        try:
            from ultralytics import YOLO  # type: ignore
        except Exception as exc:  # pragma: no cover - depends on Jetson runtime
            self.get_logger().error(f'Failed to import ultralytics for YOLO mode: {exc}')
            self.get_logger().warn('Falling back to mock mode')
            self.backend = 'mock'
            return
        model_path = self.config.get('model_path', 'yolov8n.pt')
        self.get_logger().info(f'Loading YOLO model: {model_path}')
        self.yolo_model = YOLO(model_path)

    def _on_image(self, msg: Image) -> None:
        self.frame_count += 1
        if self.frame_count % self.inference_every_n_frames != 0:
            return

        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as exc:
            self.get_logger().warn(f'Failed to convert image: {exc}')
            return

        detections = self._detect(frame)
        anomalies = self._filter_anomalies(detections, frame.shape[0])

        if anomalies and time.time() - self.last_event_time >= self.min_event_interval_s:
            self.last_event_time = time.time()
            event = {
                'stamp': {
                    'sec': int(msg.header.stamp.sec),
                    'nanosec': int(msg.header.stamp.nanosec),
                },
                'frame_id': msg.header.frame_id,
                'source_image_topic': self.image_topic,
                'backend': self.backend,
                'anomaly_count': len(anomalies),
                'anomalies': [d.__dict__ for d in anomalies],
            }
            out = String()
            out.data = json.dumps(event)
            self.event_pub.publish(out)

        if self.debug_pub is not None:
            debug = self._draw_debug(frame, detections, anomalies)
            self.debug_pub.publish(self.bridge.cv2_to_imgmsg(debug, encoding='bgr8'))

    def _detect(self, frame) -> List[Detection]:
        if self.backend == 'mock':
            h, w = frame.shape[:2]
            return [Detection(label='mock_object', confidence=0.99, bbox_xyxy=[w // 3, int(h * 0.65), w // 2, int(h * 0.9)])]

        if self.backend == 'yolo' and self.yolo_model is not None:
            results = self.yolo_model.predict(frame, verbose=False, conf=self.min_confidence)
            detections: List[Detection] = []
            for result in results:
                names = getattr(result, 'names', {})
                boxes = getattr(result, 'boxes', None)
                if boxes is None:
                    continue
                for box in boxes:
                    cls_id = int(box.cls[0].item())
                    conf = float(box.conf[0].item())
                    xyxy = [int(v) for v in box.xyxy[0].tolist()]
                    detections.append(Detection(label=str(names.get(cls_id, cls_id)), confidence=conf, bbox_xyxy=xyxy))
            return detections

        return []

    def _filter_anomalies(self, detections: List[Detection], image_height: int) -> List[Detection]:
        floor_y = int(image_height * self.floor_region_start_ratio)
        anomalies: List[Detection] = []
        for det in detections:
            if det.confidence < self.min_confidence:
                continue
            _, y1, _, y2 = det.bbox_xyxy
            bottom_in_floor_region = y2 >= floor_y
            known_label_is_suspicious = det.label in self.anomaly_labels or self.backend == 'mock'
            if bottom_in_floor_region and known_label_is_suspicious:
                anomalies.append(det)
        return anomalies

    def _draw_debug(self, frame, detections: List[Detection], anomalies: List[Detection]):
        anomaly_boxes = {tuple(a.bbox_xyxy) for a in anomalies}
        h, w = frame.shape[:2]
        floor_y = int(h * self.floor_region_start_ratio)
        cv2.line(frame, (0, floor_y), (w, floor_y), (255, 255, 255), 2)
        for det in detections:
            x1, y1, x2, y2 = det.bbox_xyxy
            color = (0, 0, 255) if tuple(det.bbox_xyxy) in anomaly_boxes else (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame,
                f'{det.label} {det.confidence:.2f}',
                (x1, max(20, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )
        return frame


def main(args: Optional[List[str]] = None) -> None:
    rclpy.init(args=args)
    node = AnomalyDetectorNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

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
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import rclpy
import yaml
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage, Image
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

        self.image_topic = self.config.get('image_topic', '/camera/realsense/color/image_compressed')
        self.depth_topic = self.config.get(
            'depth_topic',
            '/camera/realsense/aligned_depth_to_color/image_raw',
        )
        self.event_topic = self.config.get('event_topic', '/anomaly_events')
        self.detections_topic = self.config.get('detections_topic', '/jetson_ai/detections')
        self.anomaly_category_topic = self.config.get(
            'anomaly_category_topic',
            '/jetson_ai/anomaly_category',
        )
        self.anomaly_detail_topic = self.config.get(
            'anomaly_detail_topic',
            '/jetson_ai/anomaly_detail',
        )
        self.debug_image_topic = self.config.get('debug_image_topic', '/anomaly/debug_image')
        self.backend = self.config.get('detector_backend', 'mock')
        self.min_confidence = float(self.config.get('min_confidence', 0.35))
        self.inference_every_n_frames = max(1, int(self.config.get('inference_every_n_frames', 5)))
        self.publish_debug_image = bool(self.config.get('publish_debug_image', True))
        self.image_is_compressed = self._topic_looks_compressed(self.image_topic)
        self.floor_region_start_ratio = float(self.config.get('floor_region_start_ratio', 0.55))
        self.anomaly_labels = set(self.config.get('anomaly_labels', ['bottle', 'cup', 'backpack', 'chair', 'person']))
        self.enable_depth_anomaly = bool(self.config.get('enable_depth_anomaly', True))
        self.near_obstacle_m = float(self.config.get('near_obstacle_m', 0.55))
        self.near_obstacle_min_ratio = float(self.config.get('near_obstacle_min_ratio', 0.08))
        self.center_roi_fraction = float(self.config.get('center_roi_fraction', 0.45))

        self.bridge = CvBridge()
        self.frame_count = 0
        self.last_event_time = 0.0
        self.min_event_interval_s = float(self.config.get('min_event_interval_s', 1.0))
        self.yolo_model = None
        self.latest_depth = None

        if self.backend == 'yolo':
            self._init_yolo()
        elif self.backend != 'mock':
            self.get_logger().warn(f"Unknown detector_backend '{self.backend}', falling back to mock")
            self.backend = 'mock'

        self.event_pub = self.create_publisher(String, self.event_topic, 10)
        self.detections_pub = self.create_publisher(String, self.detections_topic, 10)
        self.anomaly_category_pub = self.create_publisher(String, self.anomaly_category_topic, 10)
        self.anomaly_detail_pub = self.create_publisher(String, self.anomaly_detail_topic, 10)
        self.debug_pub = self.create_publisher(Image, self.debug_image_topic, 10) if self.publish_debug_image else None
        image_msg_type = CompressedImage if self.image_is_compressed else Image
        self.image_subs = [
            self.create_subscription(
                image_msg_type,
                topic,
                self._on_image,
                qos_profile_sensor_data,
            )
            for topic in self._image_topic_candidates(self.image_topic)
        ]
        self.depth_sub = None
        if self.enable_depth_anomaly:
            self.depth_sub = self.create_subscription(
                Image,
                self.depth_topic,
                self._on_depth,
                qos_profile_sensor_data,
            )
        self.get_logger().info(
            f"Jetson anomaly detector started: backend={self.backend}, image_topic={self.image_topic}, "
            f"image_topic_candidates={self._image_topic_candidates(self.image_topic)}, "
            f"event_topic={self.event_topic}, detections_topic={self.detections_topic}, "
            f"anomaly_category_topic={self.anomaly_category_topic}"
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

    @staticmethod
    def _topic_looks_compressed(topic: str) -> bool:
        return topic.endswith('/compressed') or topic.endswith('/image_compressed')

    @staticmethod
    def _image_topic_candidates(topic: str) -> List[str]:
        candidates = [topic]
        if topic.endswith('/image_compressed'):
            candidates.append(f"{topic[:-len('/image_compressed')]}/image_raw/compressed")
        elif topic.endswith('/image_raw/compressed'):
            candidates.append(f"{topic[:-len('/image_raw/compressed')]}/image_compressed")
        return list(dict.fromkeys(candidates))

    def _on_image(self, msg: Image | CompressedImage) -> None:
        self.frame_count += 1
        if self.frame_count % self.inference_every_n_frames != 0:
            return

        try:
            frame = self._image_msg_to_bgr(msg)
        except Exception as exc:
            self.get_logger().warn(f'Failed to convert image: {exc}')
            return

        detections = self._detect(frame)
        anomalies = self._filter_anomalies(detections, frame.shape[0])
        category, detail, depth_roi = self._categorize_anomaly(frame, detections, anomalies)

        self._publish_detections(msg, detections)
        self._publish_anomaly_state(category, detail)

        if category != 'none' and time.time() - self.last_event_time >= self.min_event_interval_s:
            self.last_event_time = time.time()
            event = {
                'stamp': self._stamp_dict(msg),
                'frame_id': msg.header.frame_id,
                'source_image_topic': self.image_topic,
                'backend': self.backend,
                'category': category,
                'detail': detail,
                'anomaly_count': len(anomalies),
                'anomalies': [d.__dict__ for d in anomalies],
                'detections': [d.__dict__ for d in detections],
            }
            out = String()
            out.data = json.dumps(event)
            self.event_pub.publish(out)

        if self.debug_pub is not None:
            debug = self._draw_debug(frame, detections, anomalies, category, detail, depth_roi)
            debug_msg = self.bridge.cv2_to_imgmsg(debug, encoding='bgr8')
            debug_msg.header = msg.header
            self.debug_pub.publish(debug_msg)

    def _image_msg_to_bgr(self, msg: Image | CompressedImage):
        if isinstance(msg, CompressedImage):
            encoded = np.frombuffer(msg.data, dtype=np.uint8)
            frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            if frame is None:
                raise ValueError('OpenCV could not decode compressed image')
            return frame
        return self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

    def _on_depth(self, msg: Image) -> None:
        try:
            self.latest_depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        except Exception as exc:
            self.get_logger().warn(f'Failed to convert depth image: {exc}')

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

    def _publish_detections(self, msg: Image, detections: List[Detection]) -> None:
        out = String()
        out.data = json.dumps({
            'stamp': self._stamp_dict(msg),
            'frame_id': msg.header.frame_id,
            'source_image_topic': self.image_topic,
            'backend': self.backend,
            'detections': [d.__dict__ for d in detections],
        })
        self.detections_pub.publish(out)

    def _publish_anomaly_state(self, category: str, detail: str) -> None:
        category_msg = String()
        category_msg.data = category
        self.anomaly_category_pub.publish(category_msg)

        detail_msg = String()
        detail_msg.data = detail
        self.anomaly_detail_pub.publish(detail_msg)

    def _categorize_anomaly(
        self,
        frame,
        detections: List[Detection],
        anomalies: List[Detection],
    ) -> Tuple[str, str, Optional[List[int]]]:
        if anomalies:
            strongest = max(anomalies, key=lambda det: det.confidence)
            return (
                'semantic_object',
                f'label={strongest.label} confidence={strongest.confidence:.2f}',
                strongest.bbox_xyxy,
            )

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if float(np.mean(gray)) < 6.0:
            return 'camera_visibility', 'dark_frame', None

        if not self.enable_depth_anomaly:
            return 'none', 'ok', None

        depth = self.latest_depth
        if depth is None:
            return 'none', 'waiting_for_depth', None

        depth_m = self._depth_to_meters(depth)
        if depth_m.size == 0:
            return 'none', 'invalid_depth', None

        h, w = depth_m.shape[:2]
        roi_fraction = max(0.1, min(1.0, self.center_roi_fraction))
        roi_w = int(w * roi_fraction)
        roi_h = int(h * roi_fraction)
        x1 = max(0, (w - roi_w) // 2)
        y1 = max(0, (h - roi_h) // 2)
        x2 = min(w, x1 + roi_w)
        y2 = min(h, y1 + roi_h)
        roi = depth_m[y1:y2, x1:x2]
        valid = np.isfinite(roi) & (roi > 0.05)
        if not np.any(valid):
            return 'none', 'no_valid_depth_in_roi', None

        near = valid & (roi < self.near_obstacle_m)
        ratio = float(np.count_nonzero(near)) / float(np.count_nonzero(valid))
        if ratio >= self.near_obstacle_min_ratio:
            median = float(np.median(roi[near]))
            return (
                'depth_near_object',
                f'median_m={median:.2f} ratio={ratio:.2f}',
                [x1, y1, x2, y2],
            )

        if detections:
            return 'none', f'objects_detected count={len(detections)}', None
        return 'none', 'ok', None

    @staticmethod
    def _depth_to_meters(depth) -> np.ndarray:
        arr = np.asarray(depth, dtype=np.float32)
        if arr.ndim == 3:
            arr = arr[:, :, 0]
        if arr.size == 0:
            return arr
        finite = arr[np.isfinite(arr)]
        if finite.size and float(np.max(finite)) > 50.0:
            arr = arr * 0.001
        return arr

    @staticmethod
    def _stamp_dict(msg: Image) -> Dict[str, int]:
        return {
            'sec': int(msg.header.stamp.sec),
            'nanosec': int(msg.header.stamp.nanosec),
        }

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

    def _draw_debug(
        self,
        frame,
        detections: List[Detection],
        anomalies: List[Detection],
        category: str,
        detail: str,
        depth_roi: Optional[List[int]],
    ):
        anomaly_boxes = {tuple(a.bbox_xyxy) for a in anomalies}
        h, w = frame.shape[:2]
        floor_y = int(h * self.floor_region_start_ratio)
        cv2.line(frame, (0, floor_y), (w, floor_y), (255, 255, 255), 2)
        if depth_roi is not None and category == 'depth_near_object':
            x1, y1, x2, y2 = depth_roi
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
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
        cv2.putText(
            frame,
            f'{category}: {detail}'[:110],
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255) if category != 'none' else (0, 180, 0),
            2,
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

import logging
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from jetson_anomaly_detector.jetson_yolo_rosbridge_client import blur_except_detections
from jetson_anomaly_detector.event_schema import build_event
from jetson_anomaly_detector.localization import estimate_depth_measurement
from jetson_anomaly_detector.marker_manager import MARKER_LINE_STRIP, MarkerManager
from jetson_anomaly_detector.models import (
    Detection,
    DistanceEstimate,
    ObjectPoseMap,
    RobotPoseMap,
)
from jetson_anomaly_detector.yolo_detector import YoloDetector


class TrackingAndSegmentationTest(unittest.TestCase):
    def test_mock_detector_produces_track_and_mask_when_enabled(self) -> None:
        detector = YoloDetector(
            model_path="unused.pt",
            confidence_threshold=0.5,
            anomaly_classes=["bottle"],
            mock_mode=True,
            logger=logging.getLogger("test"),
            tracking_enabled=True,
            segmentation_enabled=True,
        )
        detection = detector.detect(np.zeros((100, 200, 3), dtype=np.uint8))[0]
        self.assertEqual(detection.track_id, 1)
        self.assertIsNotNone(detection.mask)
        assert detection.mask is not None
        self.assertEqual(detection.mask.shape, (100, 200))
        self.assertTrue(np.any(detection.mask))

    def test_ultralytics_result_parser_pairs_box_mask_and_track(self) -> None:
        detector = YoloDetector(
            model_path="unused.pt",
            confidence_threshold=0.5,
            anomaly_classes=["bottle"],
            mock_mode=True,
            logger=logging.getLogger("test"),
            tracking_enabled=True,
            tracking_confidence_threshold=0.25,
            segmentation_enabled=True,
        )
        box = SimpleNamespace(
            cls=np.asarray([0]),
            conf=np.asarray([0.8]),
            xyxy=np.asarray([[10.0, 12.0, 40.0, 60.0]]),
            id=np.asarray([9]),
        )
        mask = np.zeros((1, 25, 50), dtype=np.float32)
        mask[:, 5:20, 10:30] = 1.0
        result = SimpleNamespace(
            names={0: "bottle"},
            boxes=[box],
            masks=SimpleNamespace(data=mask),
        )
        detection = detector._parse_results([result], (100, 200))[0]
        self.assertEqual(detection.track_id, 9)
        self.assertEqual(detection.bbox_xyxy, [10, 12, 40, 60])
        self.assertEqual(detection.mask.shape, (100, 200))
        self.assertTrue(np.any(detection.mask))

    def test_privacy_stream_preserves_mask_not_whole_bbox(self) -> None:
        rng = np.random.default_rng(7)
        frame = rng.integers(0, 256, size=(80, 120, 3), dtype=np.uint8)
        mask = np.zeros((80, 120), dtype=bool)
        mask[35:55, 50:70] = True
        detection = Detection(
            label="bottle",
            confidence=0.9,
            bbox_xyxy=[20, 15, 100, 70],
            track_id=4,
            mask=mask,
        )
        result = blur_except_detections(
            frame,
            [detection],
            kernel_size=21,
            use_segmentation_masks=True,
        )
        np.testing.assert_array_equal(result[40:50, 55:65], frame[40:50, 55:65])
        self.assertFalse(np.array_equal(result[20:30, 30:40], frame[20:30, 30:40]))

    def test_depth_estimation_uses_segmentation_mask(self) -> None:
        depth = np.full((100, 100), 4.0, dtype=np.float32)
        depth[35:65, 40:60] = 2.0
        mask = np.zeros((100, 100), dtype=bool)
        mask[35:65, 40:60] = True
        measurement = estimate_depth_measurement(
            depth_image=depth,
            bbox_xyxy=[20, 20, 80, 80],
            image_width=100,
            image_height=100,
            min_distance_m=0.15,
            max_distance_m=6.0,
            roi_scale=1.0,
            min_valid_pixels=20,
            object_mask=mask,
            mask_erode_px=1,
        )
        self.assertIsNotNone(measurement)
        assert measurement is not None
        self.assertAlmostEqual(measurement.distance_m, 2.0)

    def test_event_reports_track_and_mask_without_serializing_pixels(self) -> None:
        detection = Detection(
            "bottle",
            0.9,
            [10, 20, 30, 60],
            track_id=12,
            mask=np.ones((10, 10), dtype=bool),
        )
        event = build_event(
            event_id="anom_00001",
            detection=detection,
            robot_pose=RobotPoseMap(0.0, 0.0, 0.0),
            object_pose=ObjectPoseMap(1.0, 0.0),
            cluster_id="cluster_00001",
            cluster_count=2,
            cluster_merge_radius_m=1.0,
            distance_estimate=DistanceEstimate(1.0, "depth", 0.02),
            bearing_source="camera_intrinsics",
            ttl_sec=180.0,
            original_image=None,
            annotated_image=None,
            map_snapshot=None,
            daily_map_summary=None,
            event_log=Path("events.jsonl"),
        )
        self.assertEqual(event["track_id"], 12)
        self.assertTrue(event["segmentation_mask_used"])
        self.assertNotIn("mask", event)


class MarkerVisualizationTest(unittest.TestCase):
    def test_marker_array_contains_ray_uncertainty_and_track_id(self) -> None:
        manager = MarkerManager(
            ray_enabled=True,
            uncertainty_enabled=True,
            uncertainty_sigma_scale=2.0,
            uncertainty_min_radius_m=0.05,
            uncertainty_max_radius_m=1.0,
        )
        manager.add_or_update(
            label="bottle",
            object_pose=ObjectPoseMap(2.0, 1.0),
            observed_count=1,
            ttl_s=30.0,
            robot_pose=RobotPoseMap(0.5, -0.5, 0.0),
            uncertainty_m=0.2,
            track_id=7,
        )
        markers = manager.build_marker_array()["markers"]
        self.assertEqual(len(markers), 4)
        self.assertIn("#7", markers[1]["text"])

        line_markers = [marker for marker in markers if marker["type"] == MARKER_LINE_STRIP]
        self.assertEqual(len(line_markers), 2)
        ray = next(marker for marker in line_markers if len(marker["points"]) == 2)
        circle = next(marker for marker in line_markers if len(marker["points"]) == 49)
        self.assertEqual(ray["points"][0]["x"], 0.5)
        radius = circle["points"][0]["x"] - 2.0
        self.assertAlmostEqual(radius, 0.4)

    def test_auxiliary_markers_can_be_disabled(self) -> None:
        manager = MarkerManager(ray_enabled=False, uncertainty_enabled=False)
        manager.add_or_update(
            label="bottle",
            object_pose=ObjectPoseMap(1.0, 2.0),
            observed_count=1,
            ttl_s=30.0,
            robot_pose=RobotPoseMap(0.0, 0.0, 0.0),
            uncertainty_m=0.2,
        )
        self.assertEqual(len(manager.build_marker_array()["markers"]), 2)


if __name__ == "__main__":
    unittest.main()

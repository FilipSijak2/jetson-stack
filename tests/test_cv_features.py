import json
import logging
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from jetson_anomaly_detector.jetson_yolo_rosbridge_client import (
    DetectionGroup,
    InspectionCandidate,
    InspectionCaptureState,
    InspectionTarget,
    JetsonYoloRosbridgeClient,
    ReportedAnomaly,
    _detection_sharpness,
    _group_inspection_candidates,
    _local_day_key,
    _target_center,
    blur_except_detections,
)
from jetson_anomaly_detector.event_schema import build_event
from jetson_anomaly_detector.localization import (
    estimate_3d_bounds_camera,
    estimate_depth_measurement,
)
from jetson_anomaly_detector.marker_manager import (
    MARKER_LINE_LIST,
    MARKER_LINE_STRIP,
    MarkerManager,
    build_detection_3d_marker_array,
)
from jetson_anomaly_detector.models import (
    BoundingBox3D,
    CameraIntrinsics,
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

    def test_empty_segmentation_result_does_not_warn_about_missing_masks(self) -> None:
        logger = Mock()
        detector = YoloDetector(
            model_path="unused.pt",
            confidence_threshold=0.5,
            anomaly_classes=["bottle"],
            mock_mode=True,
            logger=logger,
            segmentation_enabled=True,
        )
        logger.reset_mock()
        result = SimpleNamespace(names={0: "bottle"}, boxes=[], masks=None)

        self.assertEqual(detector._parse_results([result], (100, 200)), [])
        logger.warning.assert_not_called()

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

    def test_deprojects_segmented_depth_to_3d_bounds(self) -> None:
        depth = np.full((100, 100), 2.0, dtype=np.float32)
        mask = np.zeros((100, 100), dtype=bool)
        mask[30:70, 40:60] = True
        bounds = estimate_3d_bounds_camera(
            depth_image=depth,
            bbox_xyxy=[35, 25, 65, 75],
            image_width=100,
            image_height=100,
            intrinsics=CameraIntrinsics(100.0, 100.0, 50.0, 50.0, 100, 100),
            object_mask=mask,
            min_distance_m=0.1,
            max_distance_m=6.0,
            min_valid_points=20,
            mask_erode_px=0,
            sample_stride=1,
        )
        self.assertIsNotNone(bounds)
        assert bounds is not None
        self.assertAlmostEqual(bounds.center_z, 2.0)
        self.assertAlmostEqual(bounds.center_x, -0.01, places=2)
        self.assertGreater(bounds.size_x, 0.3)
        self.assertGreater(bounds.size_y, 0.6)


class MarkerVisualizationTest(unittest.TestCase):
    def test_builds_live_3d_wireframe_in_camera_frame(self) -> None:
        detection = Detection("bottle", 0.91, [1, 2, 3, 4], track_id=8)
        bounds = BoundingBox3D(0.1, -0.2, 1.5, 0.3, 0.6, 0.1, 120)
        marker_array = build_detection_3d_marker_array(
            [(detection, bounds)],
            frame_id="realsense_color_optical_frame",
            stamp={"sec": 1, "nanosec": 2},
            ttl_s=0.75,
            line_width_m=0.01,
        )
        markers = marker_array["markers"]
        self.assertEqual(len(markers), 2)
        self.assertEqual(markers[0]["type"], MARKER_LINE_LIST)
        self.assertEqual(markers[0]["header"]["frame_id"], "realsense_color_optical_frame")
        self.assertEqual(len(markers[0]["points"]), 24)
        self.assertIn("#8", markers[1]["text"])

    def test_compact_3d_text_uses_track_and_distance_only(self) -> None:
        detection = Detection("bottle", 0.91, [1, 2, 3, 4], track_id=8)
        bounds = BoundingBox3D(0.1, -0.2, 1.5, 0.3, 0.6, 0.1, 120)
        markers = build_detection_3d_marker_array(
            [(detection, bounds)],
            frame_id="camera",
            stamp=None,
            ttl_s=0.75,
            line_width_m=0.01,
            text_height_m=0.035,
            text_show_label=False,
            text_show_confidence=False,
        )["markers"]
        self.assertEqual(markers[1]["text"], "#8 · 1.50 m")
        self.assertEqual(markers[1]["scale"]["z"], 0.035)

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

    def test_observation_ray_stops_after_its_short_ttl(self) -> None:
        manager = MarkerManager(
            ray_enabled=True,
            ray_ttl_s=2.0,
            uncertainty_enabled=False,
        )
        manager.add_or_update(
            label="bottle",
            object_pose=ObjectPoseMap(1.0, 0.0),
            observed_count=1,
            ttl_s=180.0,
            robot_pose=RobotPoseMap(0.0, 0.0, 0.0),
        )
        cluster = next(iter(manager.active.values()))
        cluster.ray_expires_at = 0.0
        markers = manager.build_marker_array()["markers"]
        self.assertFalse(
            any(
                marker["type"] == MARKER_LINE_STRIP
                and len(marker.get("points", [])) == 2
                for marker in markers
            )
        )

    def test_different_tracks_keep_separate_map_markers(self) -> None:
        manager = MarkerManager(
            association_radius_m=0.40,
            tracked_object_min_separation_m=0.12,
            text_compact=True,
            ray_enabled=False,
            uncertainty_enabled=False,
        )
        manager.add_or_update(
            "bottle", ObjectPoseMap(1.0, 1.0), 1, 30.0, track_id=10
        )
        manager.add_or_update(
            "bottle", ObjectPoseMap(1.2, 1.0), 1, 30.0, track_id=11
        )
        self.assertEqual(len(manager.active), 2)
        labels = {
            marker["text"]
            for marker in manager.build_marker_array()["markers"]
            if "text" in marker
        }
        self.assertEqual(labels, {"#10", "#11"})

    def test_replacement_track_reuses_marker_when_old_track_disappears(self) -> None:
        manager = MarkerManager(
            association_radius_m=0.40,
            track_reassociation_radius_m=1.00,
            ray_enabled=False,
            uncertainty_enabled=False,
        )
        manager.add_or_update(
            "bottle",
            ObjectPoseMap(1.0, 1.0),
            1,
            30.0,
            track_id=10,
            visible_track_ids={10},
        )
        manager.add_or_update(
            "bottle",
            ObjectPoseMap(1.7, 1.0),
            1,
            30.0,
            track_id=11,
            visible_track_ids={11},
        )

        self.assertEqual(len(manager.active), 1)
        self.assertEqual(next(iter(manager.active.values())).track_id, 11)

    def test_simultaneously_visible_tracks_do_not_reassociate(self) -> None:
        manager = MarkerManager(
            association_radius_m=0.40,
            track_reassociation_radius_m=1.00,
            ray_enabled=False,
            uncertainty_enabled=False,
        )
        manager.add_or_update(
            "bottle",
            ObjectPoseMap(1.0, 1.0),
            1,
            30.0,
            track_id=10,
            visible_track_ids={10},
        )
        manager.add_or_update(
            "bottle",
            ObjectPoseMap(1.2, 1.0),
            1,
            30.0,
            track_id=11,
            visible_track_ids={10, 11},
        )

        self.assertEqual(len(manager.active), 2)


class InspectionCaptureTest(unittest.TestCase):
    def test_sharpness_prefers_detailed_bottle_crop(self) -> None:
        sharp = np.zeros((80, 80, 3), dtype=np.uint8)
        sharp[20:60:2, 20:60] = 255
        blurred = np.full((80, 80, 3), 127, dtype=np.uint8)
        detection = Detection("bottle", 0.9, [20, 20, 60, 60])
        self.assertGreater(
            _detection_sharpness(sharp, detection),
            _detection_sharpness(blurred, detection),
        )

    def test_groups_nearby_bottles_into_one_inspection(self) -> None:
        def candidate(cluster_id, x, y, track_id):
            detection = Detection("bottle", 0.9, [0, 0, 20, 40], track_id)
            group = DetectionGroup(
                "bottle",
                [detection],
                ObjectPoseMap(x, y),
                DistanceEstimate(1.5, "depth", 0.05),
                "camera_intrinsics",
            )
            return InspectionCandidate(
                group,
                SimpleNamespace(cluster_id=cluster_id),
            )

        first = candidate("c1", 1.0, 0.0, 1)
        second = candidate("c2", 2.5, 0.0, 2)
        far = candidate("c3", 5.0, 0.0, 3)
        groups = _group_inspection_candidates([first, second, far], 2.0)
        self.assertEqual(sorted(len(group) for group in groups), [1, 2])

    def test_group_standoff_expands_to_fit_all_targets(self) -> None:
        client = JetsonYoloRosbridgeClient.__new__(JetsonYoloRosbridgeClient)
        client.config = SimpleNamespace(
            inspection_group_min_objects=2,
            inspection_standoff_m=0.70,
            camera_horizontal_fov_deg=69.0,
            inspection_group_fov_margin_ratio=1.25,
            inspection_group_max_standoff_m=2.50,
        )
        targets = [
            InspectionTarget("c1", "bottle", ObjectPoseMap(2.0, -0.5), 1),
            InspectionTarget("c2", "bottle", ObjectPoseMap(2.0, 0.5), 2),
        ]
        center = _target_center(targets)
        standoff = client._inspection_standoff_for_targets(targets, center)
        self.assertGreater(standoff, 0.70)
        self.assertLessEqual(standoff, 2.50)

    def test_group_capture_requires_each_tracked_bottle(self) -> None:
        client = JetsonYoloRosbridgeClient.__new__(JetsonYoloRosbridgeClient)
        client.config = SimpleNamespace(marker_association_radius_m=0.40)
        targets = [
            InspectionTarget("c1", "bottle", ObjectPoseMap(1.0, 0.0), 1),
            InspectionTarget("c2", "bottle", ObjectPoseMap(1.2, 0.0), 2),
        ]
        capture = InspectionCaptureState(
            request_id="inspect_1",
            cluster_id="group_1",
            label="bottle",
            object_pose=_target_center(targets),
            track_id=None,
            targets=targets,
            require_all_visible=True,
            target_frames=8,
            deadline=999.0,
        )
        both = [
            Detection("bottle", 0.9, [0, 0, 10, 20], 1),
            Detection("bottle", 0.8, [20, 0, 30, 20], 2),
        ]
        self.assertEqual(
            len(
                client._select_inspection_detections(
                    capture, both, 40, 30, {}
                )
            ),
            2,
        )
        self.assertEqual(
            client._select_inspection_detections(
                capture, both[:1], 40, 30, {}
            ),
            [],
        )


class AutomaticMetricsTest(unittest.TestCase):
    def test_writes_throttled_inference_sample(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            client = JetsonYoloRosbridgeClient.__new__(JetsonYoloRosbridgeClient)
            client.config = SimpleNamespace(
                evaluation_metrics_enabled=True,
                evaluation_metrics_sample_period_s=1.0,
            )
            client.performance_log_path = (
                Path(directory) / "evaluation" / "performance.jsonl"
            )
            client.next_evaluation_metrics_sample = 0.0
            client.last_evaluation_error_log = 0.0
            client.frame_count = 12
            detection = Detection(
                "bottle",
                0.9,
                [0, 0, 10, 20],
                track_id=4,
                mask=np.ones((20, 10), dtype=bool),
            )
            client._record_evaluation_metrics(
                frame=np.zeros((20, 30, 3), dtype=np.uint8),
                detections=[detection],
                anomalies=[detection],
                camera_fps=15.0,
                decode_ms=2.0,
                inference_ms=20.0,
                detection_stage_ms=22.0,
            )
            records = [
                json.loads(line)
                for line in client.performance_log_path.read_text(
                    encoding="utf-8"
                ).splitlines()
            ]
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["inference_ms"], 20.0)
            self.assertEqual(records[0]["tracked_detections"], 1)
            self.assertEqual(records[0]["segmented_detections"], 1)


class DailyStateTest(unittest.TestCase):
    def test_reported_event_reassociates_replacement_track(self) -> None:
        client = JetsonYoloRosbridgeClient.__new__(
            JetsonYoloRosbridgeClient
        )
        client.config = SimpleNamespace(
            reported_merge_radius_m=1.0,
            track_reassociation_radius_m=1.0,
        )
        client.reported_anomalies = [
            ReportedAnomaly("bottle", ObjectPoseMap(1.0, 1.0), 10)
        ]

        self.assertTrue(
            client._already_reported(
                "bottle",
                ObjectPoseMap(1.7, 1.0),
                track_id=11,
                visible_track_ids={11},
            )
        )

    def test_reported_event_keeps_two_visible_tracks_separate(self) -> None:
        client = JetsonYoloRosbridgeClient.__new__(
            JetsonYoloRosbridgeClient
        )
        client.config = SimpleNamespace(
            reported_merge_radius_m=1.0,
            track_reassociation_radius_m=1.0,
        )
        client.reported_anomalies = [
            ReportedAnomaly("bottle", ObjectPoseMap(1.0, 1.0), 10)
        ]

        self.assertFalse(
            client._already_reported(
                "bottle",
                ObjectPoseMap(1.2, 1.0),
                track_id=11,
                visible_track_ids={10, 11},
            )
        )

    def test_loads_only_current_day_for_dedup_but_keeps_global_counter(self) -> None:
        now = datetime.now().astimezone()
        today = now.isoformat()
        yesterday = (now - timedelta(days=1)).isoformat()
        events = [
            {
                "id": "anom_00041",
                "timestamp": yesterday,
                "label": "bottle",
                "object_pose_map": {"x": 1.0, "y": 1.0, "z": 0.0},
            },
            {
                "id": "anom_00042",
                "timestamp": today,
                "label": "bottle",
                "track_id": 10,
                "object_pose_map": {"x": 4.0, "y": 4.0, "z": 0.0},
            },
            {
                "id": "anom_00043",
                "timestamp": today,
                "label": "bottle",
                "track_id": 11,
                "object_pose_map": {"x": 4.7, "y": 4.0, "z": 0.0},
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            event_log = Path(directory) / "events.jsonl"
            event_log.write_text(
                "\n".join(json.dumps(event) for event in events) + "\n",
                encoding="utf-8",
            )
            client = JetsonYoloRosbridgeClient.__new__(JetsonYoloRosbridgeClient)
            client.event_log_path = event_log
            client.event_counter = 0
            client.current_daily_key = _local_day_key()
            client.reported_anomalies = []
            client.config = SimpleNamespace(
                reported_merge_radius_m=1.0,
                track_reassociation_radius_m=1.0,
            )
            client._load_reported_anomalies()

        self.assertEqual(client.event_counter, 43)
        self.assertEqual(len(client.reported_anomalies), 1)
        self.assertAlmostEqual(client.reported_anomalies[0].object_pose.x, 4.175)


if __name__ == "__main__":
    unittest.main()

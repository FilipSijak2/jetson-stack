import math
import unittest
from collections import deque
from types import SimpleNamespace

import numpy as np

from jetson_anomaly_detector.localization import (
    bbox_bearing_intrinsics_rad,
    estimate_depth_measurement,
    estimate_object_pose_map,
)
from jetson_anomaly_detector.jetson_yolo_rosbridge_client import (
    JetsonYoloRosbridgeClient,
    blur_except_detections,
)
from jetson_anomaly_detector.models import CameraIntrinsics, Detection, RobotPoseMap
from jetson_anomaly_detector.ros_messages import header_stamp_seconds, parse_camera_info


class LocalizationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.intrinsics = CameraIntrinsics(
            fx=600.0,
            fy=600.0,
            cx=320.0,
            cy=240.0,
            width=640,
            height=480,
        )

    def test_intrinsics_bearing_uses_principal_point(self) -> None:
        centered = bbox_bearing_intrinsics_rad(
            [300, 100, 340, 300], 640, self.intrinsics
        )
        right = bbox_bearing_intrinsics_rad(
            [500, 100, 540, 300], 640, self.intrinsics
        )
        self.assertAlmostEqual(centered, 0.0)
        self.assertLess(right, 0.0)

    def test_axial_depth_is_converted_to_ray_range(self) -> None:
        pose = estimate_object_pose_map(
            robot_pose=RobotPoseMap(0.0, 0.0, 0.0),
            bbox_xyxy=[500, 100, 540, 300],
            image_width=640,
            default_distance_m=1.5,
            camera_horizontal_fov_deg=69.0,
            measured_distance_m=2.0,
            camera_intrinsics=self.intrinsics,
            measured_distance_is_axial=True,
        )
        self.assertAlmostEqual(pose.x, 2.0, places=6)
        self.assertLess(pose.y, 0.0)
        self.assertGreater(math.hypot(pose.x, pose.y), 2.0)

    def test_depth_measurement_reports_robust_uncertainty(self) -> None:
        depth = np.full((480, 640), 2.0, dtype=np.float32)
        depth[200:210, 300:310] = 2.1
        measurement = estimate_depth_measurement(
            depth,
            [280, 120, 360, 420],
            640,
            480,
            0.15,
            6.0,
            0.6,
            20,
        )
        self.assertIsNotNone(measurement)
        assert measurement is not None
        self.assertEqual(measurement.source, "depth")
        self.assertGreater(measurement.valid_sample_count, 20)
        self.assertGreaterEqual(measurement.uncertainty_m or 0.0, 0.005)

    def test_camera_info_and_ros2_timestamp_parsing(self) -> None:
        info = parse_camera_info(
            {
                "width": 640,
                "height": 480,
                "k": [600, 0, 320, 0, 600, 240, 0, 0, 1],
                "p": [600, 0, 320, 0, 0, 600, 240, 0, 0, 0, 1, 0],
            }
        )
        self.assertEqual(info, self.intrinsics)
        self.assertEqual(
            header_stamp_seconds(
                {"header": {"stamp": {"sec": 10, "nanosec": 500_000_000}}}
            ),
            10.5,
        )


class DepthSynchronizationTest(unittest.TestCase):
    def test_selects_closest_depth_timestamp_and_rejects_mismatch(self) -> None:
        client = JetsonYoloRosbridgeClient.__new__(JetsonYoloRosbridgeClient)
        client.config = SimpleNamespace(
            depth_max_age_s=1.0,
            depth_sync_tolerance_s=0.05,
        )
        now = __import__("time").monotonic()
        first = np.full((2, 2), 1.0, dtype=np.float32)
        second = np.full((2, 2), 2.0, dtype=np.float32)
        client.depth_buffer = deque(
            [(10.00, now, first), (10.04, now, second)],
            maxlen=8,
        )

        selected = client._select_depth_frame(
            {"header": {"stamp": {"sec": 10, "nanosec": 30_000_000}}}
        )
        self.assertIsNotNone(selected)
        assert selected is not None
        np.testing.assert_array_equal(selected[0], second)
        self.assertAlmostEqual(selected[1], 0.01, places=6)

        rejected = client._select_depth_frame(
            {"header": {"stamp": {"sec": 11, "nanosec": 0}}}
        )
        self.assertIsNone(rejected)


class PrivacyImageTest(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(42)
        self.frame = rng.integers(0, 256, size=(80, 120, 3), dtype=np.uint8)

    def test_blurs_entire_frame_without_detection(self) -> None:
        result = blur_except_detections(self.frame, [], kernel_size=20)
        self.assertEqual(result.shape, self.frame.shape)
        self.assertFalse(np.array_equal(result, self.frame))

    def test_preserves_detected_bbox_and_blurs_surroundings(self) -> None:
        detection = Detection("bottle", 0.9, [40, 20, 80, 60])
        result = blur_except_detections(
            self.frame,
            [detection],
            kernel_size=21,
            padding_ratio=0.0,
        )
        np.testing.assert_array_equal(result[20:60, 40:80], self.frame[20:60, 40:80])
        self.assertFalse(np.array_equal(result[0:15, 0:30], self.frame[0:15, 0:30]))


if __name__ == "__main__":
    unittest.main()

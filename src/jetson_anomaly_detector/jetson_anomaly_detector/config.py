from __future__ import annotations

import os
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_str(name: str, default: str) -> str:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return raw


def _env_list(name: str, default: List[str]) -> List[str]:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return [item.strip() for item in raw.split(",") if item.strip()]


@dataclass(frozen=True)
class AppConfig:
    rosbridge_url: str = "ws://raspberry.local:9090"
    camera_topic: str = "/camera/realsense/color/image_raw/compressed"
    map_topic: str = "/map"
    robot_pose_topic: str = "/robot_pose_map"
    scan_topic: str = "/scan"
    depth_topic: str = (
        "/camera/realsense/aligned_depth_to_color/image_raw/compressedDepth"
    )
    camera_info_topic: str = "/camera/realsense/color/camera_info"
    event_topic: str = "/anomaly/events"
    readable_event_topic: str = "/anomaly/events/readable"
    marker_topic: str = "/anomaly/markers"
    detection_3d_topic: str = "/anomaly/detections_3d"
    detection_3d_frame_id: str = ""
    debug_image_topic: str = "/anomaly/debug_image/compressed"
    privacy_image_topic: str = "/anomaly/privacy_image/compressed"
    inspection_request_topic: str = "/anomaly/inspection/request"
    inspection_status_topic: str = "/anomaly/inspection/status"
    inspection_result_topic: str = "/anomaly/inspection/result"
    inspection_privacy_image_topic: str = "/anomaly/inspection/privacy_image/compressed"
    map_snapshot_topic: str = "/anomaly/map_snapshot/compressed"
    artifact_root: str = "/home/jetson/anomaly_logs"
    anomaly_classes: List[str] = None  # type: ignore[assignment]
    confidence_threshold: float = 0.5
    detection_cooldown_s: float = 5.0
    marker_ttl_s: float = 180.0
    marker_republish_hz: float = 1.0
    cluster_merge_radius_m: float = 1.00
    marker_association_radius_m: float = 2.00
    reported_merge_radius_m: float = 2.00
    track_reassociation_radius_m: float = 1.00
    track_reassociation_ray_tolerance_m: float = 0.35
    marker_max_far_jump_m: float = 0.60
    anomaly_min_observations: int = 2
    anomaly_confirmation_ttl_s: float = 6.0
    marker_object_size_m: float = 0.20
    marker_text_height_m: float = 0.08
    marker_text_z_offset_m: float = 0.18
    marker_text_show_count: bool = False
    marker_text_compact: bool = True
    tracked_object_min_separation_m: float = 0.01
    default_anomaly_distance_m: float = 1.5
    camera_horizontal_fov_deg: float = 69.0
    camera_yaw_offset_deg: float = 0.0
    use_camera_intrinsics: bool = True
    use_depth_distance: bool = True
    depth_throttle_ms: int = 100
    depth_max_age_s: float = 1.0
    depth_sync_tolerance_s: float = 0.35
    depth_buffer_size: int = 8
    depth_roi_scale: float = 0.60
    depth_min_valid_pixels: int = 20
    depth_distance_percentile: float = 50.0
    depth_min_distance_m: float = 0.15
    depth_max_distance_m: float = 6.0
    depth_track_filter_enabled: bool = True
    depth_track_max_far_jump_m: float = 0.60
    depth_track_filter_ttl_s: float = 3.0
    default_distance_uncertainty_m: float = 0.75
    use_laser_distance: bool = True
    laser_window_deg: float = 6.0
    laser_min_distance_m: float = 0.10
    laser_max_distance_m: float = 4.0
    laser_distance_uncertainty_m: float = 0.10
    yolo_model_path: str = "yolov8n.pt"
    yolo_image_size: int = 640
    yolo_iou_threshold: float = 0.70
    yolo_max_detections: int = 20
    yolo_device: str = "0"
    yolo_half: bool = True
    yolo_augment: bool = False
    yolo_agnostic_nms: bool = False
    yolo_filter_classes: bool = True
    tracking_enabled: bool = True
    tracking_backend: str = "bytetrack.yaml"
    tracking_confidence_threshold: float = 0.25
    segmentation_enabled: bool = True
    segmentation_depth_mask_erode_px: int = 2
    mock_mode: bool = False
    inference_every_n_frames: int = 1
    reconnect_delay_s: float = 3.0
    jpeg_quality: int = 85
    save_per_event_images: bool = True
    save_annotated_privacy_blur: bool = True
    save_documentation_images: bool = False
    save_tracking_documentation_sequence: bool = False
    daily_map_summary: bool = True
    daily_map_summary_topic_publish: bool = True
    debug_image_always_stream: bool = True
    debug_image_on_detection: bool = True
    debug_image_publish_hz: float = 2.0
    privacy_image_enabled: bool = True
    privacy_image_publish_hz: float = 2.0
    privacy_blur_kernel_size: int = 51
    privacy_bbox_padding_ratio: float = 0.03
    privacy_use_segmentation_masks: bool = True
    privacy_draw_track_id: bool = True
    privacy_draw_mask_overlay: bool = True
    privacy_mask_overlay_alpha: float = 0.25
    inspection_enabled: bool = False
    inspection_standoff_m: float = 0.70
    inspection_min_distance_m: float = 0.40
    inspection_max_distance_m: float = 3.0
    inspection_max_uncertainty_m: float = 0.30
    inspection_require_metric_distance: bool = True
    inspection_capture_frames: int = 8
    inspection_capture_timeout_s: float = 8.0
    inspection_request_timeout_s: float = 70.0
    inspection_jpeg_quality: int = 95
    inspection_once_per_cluster: bool = True
    inspection_retry_cooldown_s: float = 60.0
    inspection_group_enabled: bool = True
    inspection_group_radius_m: float = 2.0
    inspection_group_collection_s: float = 0.75
    inspection_group_min_objects: int = 2
    inspection_group_max_objects: int = 10
    inspection_group_fov_margin_ratio: float = 1.25
    inspection_group_max_standoff_m: float = 2.50
    inspection_group_require_all_visible: bool = True
    evaluation_metrics_enabled: bool = True
    evaluation_metrics_sample_period_s: float = 1.0
    marker_ray_enabled: bool = True
    marker_ray_ttl_s: float = 2.0
    marker_uncertainty_enabled: bool = True
    marker_uncertainty_sigma_scale: float = 2.0
    marker_uncertainty_min_radius_m: float = 0.05
    marker_uncertainty_max_radius_m: float = 1.0
    marker_aux_line_width_m: float = 0.025
    detection_3d_enabled: bool = True
    detection_3d_require_mask: bool = True
    detection_3d_publish_hz: float = 5.0
    detection_3d_ttl_s: float = 0.75
    detection_3d_min_valid_points: int = 30
    detection_3d_lower_percentile: float = 5.0
    detection_3d_upper_percentile: float = 95.0
    detection_3d_sample_stride: int = 2
    detection_3d_minimum_thickness_m: float = 0.05
    detection_3d_line_width_m: float = 0.01
    detection_3d_text_enabled: bool = True
    detection_3d_text_height_m: float = 0.035
    detection_3d_text_show_label: bool = False
    detection_3d_text_show_confidence: bool = False
    detection_3d_text_show_distance: bool = True
    map_frame_id: str = "map"

    def __post_init__(self) -> None:
        if self.anomaly_classes is None:
            object.__setattr__(self, "anomaly_classes", ["bottle"])


ENV_OVERRIDES = {
    "rosbridge_url": ("ROSBRIDGE_URL", _env_str),
    "camera_topic": ("CAMERA_TOPIC", _env_str),
    "map_topic": ("MAP_TOPIC", _env_str),
    "robot_pose_topic": ("ROBOT_POSE_TOPIC", _env_str),
    "scan_topic": ("SCAN_TOPIC", _env_str),
    "depth_topic": ("DEPTH_TOPIC", _env_str),
    "camera_info_topic": ("CAMERA_INFO_TOPIC", _env_str),
    "event_topic": ("EVENT_TOPIC", _env_str),
    "readable_event_topic": ("READABLE_EVENT_TOPIC", _env_str),
    "marker_topic": ("MARKER_TOPIC", _env_str),
    "detection_3d_topic": ("DETECTION_3D_TOPIC", _env_str),
    "detection_3d_frame_id": ("DETECTION_3D_FRAME_ID", _env_str),
    "debug_image_topic": ("DEBUG_IMAGE_TOPIC", _env_str),
    "privacy_image_topic": ("PRIVACY_IMAGE_TOPIC", _env_str),
    "inspection_request_topic": ("INSPECTION_REQUEST_TOPIC", _env_str),
    "inspection_status_topic": ("INSPECTION_STATUS_TOPIC", _env_str),
    "inspection_result_topic": ("INSPECTION_RESULT_TOPIC", _env_str),
    "inspection_privacy_image_topic": ("INSPECTION_PRIVACY_IMAGE_TOPIC", _env_str),
    "map_snapshot_topic": ("MAP_SNAPSHOT_TOPIC", _env_str),
    "artifact_root": ("JETSON_ARTIFACT_ROOT", _env_str),
    "anomaly_classes": ("ANOMALY_CLASSES", _env_list),
    "confidence_threshold": ("CONFIDENCE_THRESHOLD", _env_float),
    "detection_cooldown_s": ("DETECTION_COOLDOWN_S", _env_float),
    "marker_ttl_s": ("MARKER_TTL_S", _env_float),
    "marker_republish_hz": ("MARKER_REPUBLISH_HZ", _env_float),
    "cluster_merge_radius_m": ("CLUSTER_MERGE_RADIUS_M", _env_float),
    "marker_association_radius_m": ("MARKER_ASSOCIATION_RADIUS_M", _env_float),
    "reported_merge_radius_m": ("REPORTED_MERGE_RADIUS_M", _env_float),
    "track_reassociation_radius_m": ("TRACK_REASSOCIATION_RADIUS_M", _env_float),
    "track_reassociation_ray_tolerance_m": ("TRACK_REASSOCIATION_RAY_TOLERANCE_M", _env_float),
    "marker_max_far_jump_m": ("MARKER_MAX_FAR_JUMP_M", _env_float),
    "anomaly_min_observations": ("ANOMALY_MIN_OBSERVATIONS", _env_int),
    "anomaly_confirmation_ttl_s": ("ANOMALY_CONFIRMATION_TTL_S", _env_float),
    "marker_object_size_m": ("MARKER_OBJECT_SIZE_M", _env_float),
    "marker_text_height_m": ("MARKER_TEXT_HEIGHT_M", _env_float),
    "marker_text_z_offset_m": ("MARKER_TEXT_Z_OFFSET_M", _env_float),
    "marker_text_show_count": ("MARKER_TEXT_SHOW_COUNT", _env_bool),
    "marker_text_compact": ("MARKER_TEXT_COMPACT", _env_bool),
    "tracked_object_min_separation_m": ("TRACKED_OBJECT_MIN_SEPARATION_M", _env_float),
    "default_anomaly_distance_m": ("DEFAULT_ANOMALY_DISTANCE_M", _env_float),
    "camera_horizontal_fov_deg": ("CAMERA_HORIZONTAL_FOV_DEG", _env_float),
    "camera_yaw_offset_deg": ("CAMERA_YAW_OFFSET_DEG", _env_float),
    "use_camera_intrinsics": ("USE_CAMERA_INTRINSICS", _env_bool),
    "use_depth_distance": ("USE_DEPTH_DISTANCE", _env_bool),
    "depth_throttle_ms": ("DEPTH_THROTTLE_MS", _env_int),
    "depth_max_age_s": ("DEPTH_MAX_AGE_S", _env_float),
    "depth_sync_tolerance_s": ("DEPTH_SYNC_TOLERANCE_S", _env_float),
    "depth_buffer_size": ("DEPTH_BUFFER_SIZE", _env_int),
    "depth_roi_scale": ("DEPTH_ROI_SCALE", _env_float),
    "depth_min_valid_pixels": ("DEPTH_MIN_VALID_PIXELS", _env_int),
    "depth_distance_percentile": ("DEPTH_DISTANCE_PERCENTILE", _env_float),
    "depth_min_distance_m": ("DEPTH_MIN_DISTANCE_M", _env_float),
    "depth_max_distance_m": ("DEPTH_MAX_DISTANCE_M", _env_float),
    "depth_track_filter_enabled": ("DEPTH_TRACK_FILTER_ENABLED", _env_bool),
    "depth_track_max_far_jump_m": ("DEPTH_TRACK_MAX_FAR_JUMP_M", _env_float),
    "depth_track_filter_ttl_s": ("DEPTH_TRACK_FILTER_TTL_S", _env_float),
    "default_distance_uncertainty_m": ("DEFAULT_DISTANCE_UNCERTAINTY_M", _env_float),
    "use_laser_distance": ("USE_LASER_DISTANCE", _env_bool),
    "laser_window_deg": ("LASER_WINDOW_DEG", _env_float),
    "laser_min_distance_m": ("LASER_MIN_DISTANCE_M", _env_float),
    "laser_max_distance_m": ("LASER_MAX_DISTANCE_M", _env_float),
    "laser_distance_uncertainty_m": ("LASER_DISTANCE_UNCERTAINTY_M", _env_float),
    "yolo_model_path": ("YOLO_MODEL_PATH", _env_str),
    "yolo_image_size": ("YOLO_IMAGE_SIZE", _env_int),
    "yolo_iou_threshold": ("YOLO_IOU_THRESHOLD", _env_float),
    "yolo_max_detections": ("YOLO_MAX_DETECTIONS", _env_int),
    "yolo_device": ("YOLO_DEVICE", _env_str),
    "yolo_half": ("YOLO_HALF", _env_bool),
    "yolo_augment": ("YOLO_AUGMENT", _env_bool),
    "yolo_agnostic_nms": ("YOLO_AGNOSTIC_NMS", _env_bool),
    "yolo_filter_classes": ("YOLO_FILTER_CLASSES", _env_bool),
    "tracking_enabled": ("TRACKING_ENABLED", _env_bool),
    "tracking_backend": ("TRACKING_BACKEND", _env_str),
    "tracking_confidence_threshold": ("TRACKING_CONFIDENCE_THRESHOLD", _env_float),
    "segmentation_enabled": ("SEGMENTATION_ENABLED", _env_bool),
    "segmentation_depth_mask_erode_px": ("SEGMENTATION_DEPTH_MASK_ERODE_PX", _env_int),
    "mock_mode": ("MOCK_MODE", _env_bool),
    "inference_every_n_frames": ("INFERENCE_EVERY_N_FRAMES", _env_int),
    "reconnect_delay_s": ("ROSBRIDGE_RECONNECT_DELAY_S", _env_float),
    "jpeg_quality": ("DEBUG_JPEG_QUALITY", _env_int),
    "save_per_event_images": ("SAVE_PER_EVENT_IMAGES", _env_bool),
    "save_annotated_privacy_blur": ("SAVE_ANNOTATED_PRIVACY_BLUR", _env_bool),
    "save_documentation_images": ("SAVE_DOCUMENTATION_IMAGES", _env_bool),
    "save_tracking_documentation_sequence": (
        "SAVE_TRACKING_DOCUMENTATION_SEQUENCE",
        _env_bool,
    ),
    "daily_map_summary": ("DAILY_MAP_SUMMARY", _env_bool),
    "daily_map_summary_topic_publish": ("DAILY_MAP_SUMMARY_TOPIC_PUBLISH", _env_bool),
    "debug_image_always_stream": ("DEBUG_IMAGE_ALWAYS_STREAM", _env_bool),
    "debug_image_on_detection": ("DEBUG_IMAGE_ON_DETECTION", _env_bool),
    "debug_image_publish_hz": ("DEBUG_IMAGE_PUBLISH_HZ", _env_float),
    "privacy_image_enabled": ("PRIVACY_IMAGE_ENABLED", _env_bool),
    "privacy_image_publish_hz": ("PRIVACY_IMAGE_PUBLISH_HZ", _env_float),
    "privacy_blur_kernel_size": ("PRIVACY_BLUR_KERNEL_SIZE", _env_int),
    "privacy_bbox_padding_ratio": ("PRIVACY_BBOX_PADDING_RATIO", _env_float),
    "privacy_use_segmentation_masks": ("PRIVACY_USE_SEGMENTATION_MASKS", _env_bool),
    "privacy_draw_track_id": ("PRIVACY_DRAW_TRACK_ID", _env_bool),
    "privacy_draw_mask_overlay": ("PRIVACY_DRAW_MASK_OVERLAY", _env_bool),
    "privacy_mask_overlay_alpha": ("PRIVACY_MASK_OVERLAY_ALPHA", _env_float),
    "inspection_enabled": ("INSPECTION_ENABLED", _env_bool),
    "inspection_standoff_m": ("INSPECTION_STANDOFF_M", _env_float),
    "inspection_min_distance_m": ("INSPECTION_MIN_DISTANCE_M", _env_float),
    "inspection_max_distance_m": ("INSPECTION_MAX_DISTANCE_M", _env_float),
    "inspection_max_uncertainty_m": ("INSPECTION_MAX_UNCERTAINTY_M", _env_float),
    "inspection_require_metric_distance": ("INSPECTION_REQUIRE_METRIC_DISTANCE", _env_bool),
    "inspection_capture_frames": ("INSPECTION_CAPTURE_FRAMES", _env_int),
    "inspection_capture_timeout_s": ("INSPECTION_CAPTURE_TIMEOUT_S", _env_float),
    "inspection_request_timeout_s": ("INSPECTION_REQUEST_TIMEOUT_S", _env_float),
    "inspection_jpeg_quality": ("INSPECTION_JPEG_QUALITY", _env_int),
    "inspection_once_per_cluster": ("INSPECTION_ONCE_PER_CLUSTER", _env_bool),
    "inspection_retry_cooldown_s": ("INSPECTION_RETRY_COOLDOWN_S", _env_float),
    "inspection_group_enabled": ("INSPECTION_GROUP_ENABLED", _env_bool),
    "inspection_group_radius_m": ("INSPECTION_GROUP_RADIUS_M", _env_float),
    "inspection_group_collection_s": ("INSPECTION_GROUP_COLLECTION_S", _env_float),
    "inspection_group_min_objects": ("INSPECTION_GROUP_MIN_OBJECTS", _env_int),
    "inspection_group_max_objects": ("INSPECTION_GROUP_MAX_OBJECTS", _env_int),
    "inspection_group_fov_margin_ratio": ("INSPECTION_GROUP_FOV_MARGIN_RATIO", _env_float),
    "inspection_group_max_standoff_m": ("INSPECTION_GROUP_MAX_STANDOFF_M", _env_float),
    "inspection_group_require_all_visible": ("INSPECTION_GROUP_REQUIRE_ALL_VISIBLE", _env_bool),
    "evaluation_metrics_enabled": ("EVALUATION_METRICS_ENABLED", _env_bool),
    "evaluation_metrics_sample_period_s": ("EVALUATION_METRICS_SAMPLE_PERIOD_S", _env_float),
    "marker_ray_enabled": ("MARKER_RAY_ENABLED", _env_bool),
    "marker_ray_ttl_s": ("MARKER_RAY_TTL_S", _env_float),
    "marker_uncertainty_enabled": ("MARKER_UNCERTAINTY_ENABLED", _env_bool),
    "marker_uncertainty_sigma_scale": ("MARKER_UNCERTAINTY_SIGMA_SCALE", _env_float),
    "marker_uncertainty_min_radius_m": ("MARKER_UNCERTAINTY_MIN_RADIUS_M", _env_float),
    "marker_uncertainty_max_radius_m": ("MARKER_UNCERTAINTY_MAX_RADIUS_M", _env_float),
    "marker_aux_line_width_m": ("MARKER_AUX_LINE_WIDTH_M", _env_float),
    "detection_3d_enabled": ("DETECTION_3D_ENABLED", _env_bool),
    "detection_3d_require_mask": ("DETECTION_3D_REQUIRE_MASK", _env_bool),
    "detection_3d_publish_hz": ("DETECTION_3D_PUBLISH_HZ", _env_float),
    "detection_3d_ttl_s": ("DETECTION_3D_TTL_S", _env_float),
    "detection_3d_min_valid_points": ("DETECTION_3D_MIN_VALID_POINTS", _env_int),
    "detection_3d_lower_percentile": ("DETECTION_3D_LOWER_PERCENTILE", _env_float),
    "detection_3d_upper_percentile": ("DETECTION_3D_UPPER_PERCENTILE", _env_float),
    "detection_3d_sample_stride": ("DETECTION_3D_SAMPLE_STRIDE", _env_int),
    "detection_3d_minimum_thickness_m": ("DETECTION_3D_MINIMUM_THICKNESS_M", _env_float),
    "detection_3d_line_width_m": ("DETECTION_3D_LINE_WIDTH_M", _env_float),
    "detection_3d_text_enabled": ("DETECTION_3D_TEXT_ENABLED", _env_bool),
    "detection_3d_text_height_m": ("DETECTION_3D_TEXT_HEIGHT_M", _env_float),
    "detection_3d_text_show_label": ("DETECTION_3D_TEXT_SHOW_LABEL", _env_bool),
    "detection_3d_text_show_confidence": ("DETECTION_3D_TEXT_SHOW_CONFIDENCE", _env_bool),
    "detection_3d_text_show_distance": ("DETECTION_3D_TEXT_SHOW_DISTANCE", _env_bool),
    "map_frame_id": ("MAP_FRAME_ID", _env_str),
}


def load_config(config_file: Optional[str] = None) -> AppConfig:
    data: Dict[str, Any] = {}
    if config_file:
        path = Path(config_file)
        if path.exists():
            with path.open("r", encoding="utf-8") as handle:
                loaded = yaml.safe_load(handle) or {}
            if not isinstance(loaded, dict):
                raise ValueError(f"Config file must contain a mapping: {config_file}")
            data.update(loaded)

    defaults = AppConfig()
    allowed = {field.name for field in fields(AppConfig)}
    normalized = {key: value for key, value in data.items() if key in allowed}

    for key, (env_name, reader) in ENV_OVERRIDES.items():
        current = normalized.get(key, getattr(defaults, key))
        if os.getenv(env_name) not in (None, ""):
            normalized[key] = reader(env_name, current)

    if "anomaly_classes" not in normalized:
        normalized["anomaly_classes"] = list(defaults.anomaly_classes)
    elif isinstance(normalized["anomaly_classes"], str):
        normalized["anomaly_classes"] = [
            item.strip() for item in normalized["anomaly_classes"].split(",") if item.strip()
        ]
    else:
        normalized["anomaly_classes"] = list(normalized["anomaly_classes"])

    normalized["inference_every_n_frames"] = max(1, int(normalized.get("inference_every_n_frames", 1)))
    normalized["jpeg_quality"] = max(1, min(100, int(normalized.get("jpeg_quality", 85))))
    normalized["debug_image_publish_hz"] = max(0.1, float(normalized.get("debug_image_publish_hz", 2.0)))
    normalized["privacy_image_publish_hz"] = max(
        0.1, float(normalized.get("privacy_image_publish_hz", 2.0))
    )
    normalized["privacy_blur_kernel_size"] = max(
        3, int(normalized.get("privacy_blur_kernel_size", 51))
    )
    normalized["privacy_bbox_padding_ratio"] = max(
        0.0, min(0.5, float(normalized.get("privacy_bbox_padding_ratio", 0.03)))
    )
    normalized["privacy_mask_overlay_alpha"] = max(
        0.0, min(1.0, float(normalized.get("privacy_mask_overlay_alpha", 0.25)))
    )
    normalized["inspection_standoff_m"] = max(
        0.30, float(normalized.get("inspection_standoff_m", 0.70))
    )
    normalized["inspection_min_distance_m"] = max(
        0.10,
        float(normalized.get("inspection_min_distance_m", 0.40)),
    )
    normalized["inspection_max_distance_m"] = max(
        normalized["inspection_min_distance_m"],
        float(normalized.get("inspection_max_distance_m", 3.0)),
    )
    normalized["inspection_max_uncertainty_m"] = max(
        0.0, float(normalized.get("inspection_max_uncertainty_m", 0.30))
    )
    normalized["inspection_capture_frames"] = max(
        1, int(normalized.get("inspection_capture_frames", 8))
    )
    normalized["inspection_capture_timeout_s"] = max(
        1.0, float(normalized.get("inspection_capture_timeout_s", 8.0))
    )
    normalized["inspection_request_timeout_s"] = max(
        normalized["inspection_capture_timeout_s"],
        float(normalized.get("inspection_request_timeout_s", 70.0)),
    )
    normalized["inspection_jpeg_quality"] = max(
        1, min(100, int(normalized.get("inspection_jpeg_quality", 95)))
    )
    normalized["inspection_retry_cooldown_s"] = max(
        0.0, float(normalized.get("inspection_retry_cooldown_s", 60.0))
    )
    normalized["inspection_group_radius_m"] = max(
        0.10, float(normalized.get("inspection_group_radius_m", 2.0))
    )
    normalized["inspection_group_collection_s"] = max(
        0.0, float(normalized.get("inspection_group_collection_s", 0.75))
    )
    normalized["inspection_group_min_objects"] = max(
        2, int(normalized.get("inspection_group_min_objects", 2))
    )
    normalized["inspection_group_max_objects"] = max(
        normalized["inspection_group_min_objects"],
        int(normalized.get("inspection_group_max_objects", 10)),
    )
    normalized["inspection_group_fov_margin_ratio"] = max(
        1.0, float(normalized.get("inspection_group_fov_margin_ratio", 1.25))
    )
    normalized["inspection_group_max_standoff_m"] = max(
        normalized["inspection_standoff_m"],
        float(normalized.get("inspection_group_max_standoff_m", 2.50)),
    )
    normalized["evaluation_metrics_sample_period_s"] = max(
        0.1, float(normalized.get("evaluation_metrics_sample_period_s", 1.0))
    )
    normalized["marker_uncertainty_sigma_scale"] = max(
        0.0, float(normalized.get("marker_uncertainty_sigma_scale", 2.0))
    )
    normalized["marker_ray_ttl_s"] = max(
        0.1, float(normalized.get("marker_ray_ttl_s", 2.0))
    )
    normalized["marker_uncertainty_min_radius_m"] = max(
        0.0, float(normalized.get("marker_uncertainty_min_radius_m", 0.05))
    )
    normalized["marker_uncertainty_max_radius_m"] = max(
        normalized["marker_uncertainty_min_radius_m"],
        float(normalized.get("marker_uncertainty_max_radius_m", 1.0)),
    )
    normalized["marker_aux_line_width_m"] = max(
        0.005, float(normalized.get("marker_aux_line_width_m", 0.025))
    )
    normalized["detection_3d_publish_hz"] = max(
        0.1, float(normalized.get("detection_3d_publish_hz", 5.0))
    )
    normalized["detection_3d_ttl_s"] = max(
        0.1, float(normalized.get("detection_3d_ttl_s", 0.75))
    )
    normalized["detection_3d_min_valid_points"] = max(
        3, int(normalized.get("detection_3d_min_valid_points", 30))
    )
    normalized["detection_3d_lower_percentile"] = max(
        0.0, min(49.0, float(normalized.get("detection_3d_lower_percentile", 5.0)))
    )
    normalized["detection_3d_upper_percentile"] = max(
        51.0, min(100.0, float(normalized.get("detection_3d_upper_percentile", 95.0)))
    )
    normalized["detection_3d_sample_stride"] = max(
        1, int(normalized.get("detection_3d_sample_stride", 2))
    )
    normalized["detection_3d_minimum_thickness_m"] = max(
        0.01, float(normalized.get("detection_3d_minimum_thickness_m", 0.05))
    )
    normalized["detection_3d_line_width_m"] = max(
        0.002, float(normalized.get("detection_3d_line_width_m", 0.01))
    )
    normalized["detection_3d_text_height_m"] = max(
        0.01, float(normalized.get("detection_3d_text_height_m", 0.035))
    )
    normalized["marker_republish_hz"] = max(0.1, float(normalized.get("marker_republish_hz", 1.0)))
    normalized["cluster_merge_radius_m"] = max(0.01, float(normalized.get("cluster_merge_radius_m", 1.00)))
    normalized["marker_association_radius_m"] = max(
        normalized["cluster_merge_radius_m"],
        float(normalized.get("marker_association_radius_m", 2.00)),
    )
    normalized["reported_merge_radius_m"] = max(
        normalized["marker_association_radius_m"],
        float(normalized.get("reported_merge_radius_m", 2.00)),
    )
    normalized["track_reassociation_radius_m"] = max(
        normalized["marker_association_radius_m"],
        float(normalized.get("track_reassociation_radius_m", 1.00)),
    )
    normalized["track_reassociation_ray_tolerance_m"] = max(
        0.05,
        float(normalized.get("track_reassociation_ray_tolerance_m", 0.35)),
    )
    normalized["marker_max_far_jump_m"] = max(
        0.05, float(normalized.get("marker_max_far_jump_m", 0.60))
    )
    normalized["anomaly_min_observations"] = max(1, int(normalized.get("anomaly_min_observations", 2)))
    normalized["anomaly_confirmation_ttl_s"] = max(
        0.1,
        float(normalized.get("anomaly_confirmation_ttl_s", 6.0)),
    )
    normalized["depth_throttle_ms"] = max(0, int(normalized.get("depth_throttle_ms", 100)))
    normalized["depth_max_age_s"] = max(0.1, float(normalized.get("depth_max_age_s", 1.0)))
    normalized["depth_sync_tolerance_s"] = max(
        0.0, float(normalized.get("depth_sync_tolerance_s", 0.35))
    )
    normalized["depth_buffer_size"] = max(1, int(normalized.get("depth_buffer_size", 8)))
    normalized["depth_roi_scale"] = max(0.1, min(1.0, float(normalized.get("depth_roi_scale", 0.60))))
    normalized["depth_min_valid_pixels"] = max(1, int(normalized.get("depth_min_valid_pixels", 20)))
    normalized["depth_distance_percentile"] = max(
        0.0,
        min(100.0, float(normalized.get("depth_distance_percentile", 50.0))),
    )
    normalized["depth_min_distance_m"] = max(0.01, float(normalized.get("depth_min_distance_m", 0.15)))
    normalized["depth_max_distance_m"] = max(
        normalized["depth_min_distance_m"],
        float(normalized.get("depth_max_distance_m", 6.0)),
    )
    normalized["depth_track_max_far_jump_m"] = max(
        0.05, float(normalized.get("depth_track_max_far_jump_m", 0.60))
    )
    normalized["depth_track_filter_ttl_s"] = max(
        0.25, float(normalized.get("depth_track_filter_ttl_s", 3.0))
    )
    normalized["default_distance_uncertainty_m"] = max(
        0.0, float(normalized.get("default_distance_uncertainty_m", 0.75))
    )
    normalized["yolo_image_size"] = max(32, int(normalized.get("yolo_image_size", 640)))
    normalized["yolo_iou_threshold"] = max(
        0.0, min(1.0, float(normalized.get("yolo_iou_threshold", 0.70)))
    )
    normalized["yolo_max_detections"] = max(1, int(normalized.get("yolo_max_detections", 20)))
    normalized["tracking_backend"] = str(
        normalized.get("tracking_backend", "bytetrack.yaml")
    ).strip() or "bytetrack.yaml"
    normalized["tracking_confidence_threshold"] = max(
        0.0, min(1.0, float(normalized.get("tracking_confidence_threshold", 0.25)))
    )
    normalized["segmentation_depth_mask_erode_px"] = max(
        0, int(normalized.get("segmentation_depth_mask_erode_px", 2))
    )
    normalized["marker_object_size_m"] = max(0.05, float(normalized.get("marker_object_size_m", 0.20)))
    normalized["marker_text_height_m"] = max(0.01, float(normalized.get("marker_text_height_m", 0.08)))
    normalized["marker_text_z_offset_m"] = max(0.0, float(normalized.get("marker_text_z_offset_m", 0.18)))
    normalized["tracked_object_min_separation_m"] = max(
        0.01, float(normalized.get("tracked_object_min_separation_m", 0.01))
    )
    normalized["laser_window_deg"] = max(0.5, float(normalized.get("laser_window_deg", 6.0)))
    normalized["laser_min_distance_m"] = max(0.01, float(normalized.get("laser_min_distance_m", 0.10)))
    normalized["laser_max_distance_m"] = max(
        normalized["laser_min_distance_m"],
        float(normalized.get("laser_max_distance_m", 4.0)),
    )
    normalized["laser_distance_uncertainty_m"] = max(
        0.0, float(normalized.get("laser_distance_uncertainty_m", 0.10))
    )
    return AppConfig(**normalized)

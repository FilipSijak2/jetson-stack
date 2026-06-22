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
    camera_topic: str = "/camera/color/image/compressed"
    map_topic: str = "/map"
    robot_pose_topic: str = "/robot_pose_map"
    scan_topic: str = "/scan"
    event_topic: str = "/anomaly/events"
    readable_event_topic: str = "/anomaly/events/readable"
    marker_topic: str = "/anomaly/markers"
    debug_image_topic: str = "/anomaly/debug_image/compressed"
    map_snapshot_topic: str = "/anomaly/map_snapshot/compressed"
    artifact_root: str = "/home/jetson/anomaly_logs"
    anomaly_classes: List[str] = None  # type: ignore[assignment]
    confidence_threshold: float = 0.5
    detection_cooldown_s: float = 5.0
    marker_ttl_s: float = 180.0
    marker_republish_hz: float = 1.0
    cluster_merge_radius_m: float = 1.00
    marker_object_size_m: float = 0.20
    marker_text_height_m: float = 0.08
    marker_text_z_offset_m: float = 0.18
    marker_text_show_count: bool = False
    default_anomaly_distance_m: float = 1.5
    camera_horizontal_fov_deg: float = 69.0
    camera_yaw_offset_deg: float = 0.0
    use_laser_distance: bool = True
    laser_window_deg: float = 6.0
    laser_min_distance_m: float = 0.10
    laser_max_distance_m: float = 4.0
    yolo_model_path: str = "yolov8n.pt"
    mock_mode: bool = False
    inference_every_n_frames: int = 1
    reconnect_delay_s: float = 3.0
    jpeg_quality: int = 85
    save_per_event_images: bool = False
    daily_map_summary: bool = True
    daily_map_summary_topic_publish: bool = True
    debug_image_always_stream: bool = True
    debug_image_on_detection: bool = True
    debug_image_publish_hz: float = 2.0
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
    "event_topic": ("EVENT_TOPIC", _env_str),
    "readable_event_topic": ("READABLE_EVENT_TOPIC", _env_str),
    "marker_topic": ("MARKER_TOPIC", _env_str),
    "debug_image_topic": ("DEBUG_IMAGE_TOPIC", _env_str),
    "map_snapshot_topic": ("MAP_SNAPSHOT_TOPIC", _env_str),
    "artifact_root": ("JETSON_ARTIFACT_ROOT", _env_str),
    "anomaly_classes": ("ANOMALY_CLASSES", _env_list),
    "confidence_threshold": ("CONFIDENCE_THRESHOLD", _env_float),
    "detection_cooldown_s": ("DETECTION_COOLDOWN_S", _env_float),
    "marker_ttl_s": ("MARKER_TTL_S", _env_float),
    "marker_republish_hz": ("MARKER_REPUBLISH_HZ", _env_float),
    "cluster_merge_radius_m": ("CLUSTER_MERGE_RADIUS_M", _env_float),
    "marker_object_size_m": ("MARKER_OBJECT_SIZE_M", _env_float),
    "marker_text_height_m": ("MARKER_TEXT_HEIGHT_M", _env_float),
    "marker_text_z_offset_m": ("MARKER_TEXT_Z_OFFSET_M", _env_float),
    "marker_text_show_count": ("MARKER_TEXT_SHOW_COUNT", _env_bool),
    "default_anomaly_distance_m": ("DEFAULT_ANOMALY_DISTANCE_M", _env_float),
    "camera_horizontal_fov_deg": ("CAMERA_HORIZONTAL_FOV_DEG", _env_float),
    "camera_yaw_offset_deg": ("CAMERA_YAW_OFFSET_DEG", _env_float),
    "use_laser_distance": ("USE_LASER_DISTANCE", _env_bool),
    "laser_window_deg": ("LASER_WINDOW_DEG", _env_float),
    "laser_min_distance_m": ("LASER_MIN_DISTANCE_M", _env_float),
    "laser_max_distance_m": ("LASER_MAX_DISTANCE_M", _env_float),
    "yolo_model_path": ("YOLO_MODEL_PATH", _env_str),
    "mock_mode": ("MOCK_MODE", _env_bool),
    "inference_every_n_frames": ("INFERENCE_EVERY_N_FRAMES", _env_int),
    "reconnect_delay_s": ("ROSBRIDGE_RECONNECT_DELAY_S", _env_float),
    "jpeg_quality": ("DEBUG_JPEG_QUALITY", _env_int),
    "save_per_event_images": ("SAVE_PER_EVENT_IMAGES", _env_bool),
    "daily_map_summary": ("DAILY_MAP_SUMMARY", _env_bool),
    "daily_map_summary_topic_publish": ("DAILY_MAP_SUMMARY_TOPIC_PUBLISH", _env_bool),
    "debug_image_always_stream": ("DEBUG_IMAGE_ALWAYS_STREAM", _env_bool),
    "debug_image_on_detection": ("DEBUG_IMAGE_ON_DETECTION", _env_bool),
    "debug_image_publish_hz": ("DEBUG_IMAGE_PUBLISH_HZ", _env_float),
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
    normalized["marker_republish_hz"] = max(0.1, float(normalized.get("marker_republish_hz", 1.0)))
    normalized["cluster_merge_radius_m"] = max(0.01, float(normalized.get("cluster_merge_radius_m", 1.00)))
    normalized["marker_object_size_m"] = max(0.05, float(normalized.get("marker_object_size_m", 0.20)))
    normalized["marker_text_height_m"] = max(0.01, float(normalized.get("marker_text_height_m", 0.08)))
    normalized["marker_text_z_offset_m"] = max(0.0, float(normalized.get("marker_text_z_offset_m", 0.18)))
    normalized["laser_window_deg"] = max(0.5, float(normalized.get("laser_window_deg", 6.0)))
    normalized["laser_min_distance_m"] = max(0.01, float(normalized.get("laser_min_distance_m", 0.10)))
    normalized["laser_max_distance_m"] = max(
        normalized["laser_min_distance_m"],
        float(normalized.get("laser_max_distance_m", 4.0)),
    )
    return AppConfig(**normalized)

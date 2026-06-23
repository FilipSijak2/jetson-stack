from __future__ import annotations

import base64
import math
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import cv2
import numpy as np

from .models import LaserScan, OccupancyGridMap, RobotPoseMap


def ros_time_now() -> Dict[str, int]:
    now = time.time()
    sec = int(now)
    return {"sec": sec, "nanosec": int((now - sec) * 1_000_000_000)}


def iso_timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def decode_uint8_array(data: Any) -> bytes:
    if isinstance(data, str):
        return base64.b64decode(data)
    if isinstance(data, list):
        return bytes(int(value) & 0xFF for value in data)
    if isinstance(data, (bytes, bytearray)):
        return bytes(data)
    raise ValueError(f"Unsupported uint8[] payload type: {type(data)!r}")


def decode_compressed_image(msg: Dict[str, Any]) -> np.ndarray:
    raw = decode_uint8_array(msg.get("data", ""))
    encoded = np.frombuffer(raw, dtype=np.uint8)
    frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("OpenCV could not decode compressed image")
    return frame


def decode_depth_image(msg: Dict[str, Any]) -> np.ndarray:
    height = int(msg.get("height", 0))
    width = int(msg.get("width", 0))
    step = int(msg.get("step", 0))
    encoding = str(msg.get("encoding", "")).strip().lower()
    is_bigendian = bool(int(msg.get("is_bigendian", 0)))
    if height <= 0 or width <= 0:
        raise ValueError("Invalid depth image dimensions")

    if encoding in {"16uc1", "mono16"}:
        dtype = np.dtype(">u2" if is_bigendian else "<u2")
        scale = 0.001
    elif encoding == "32fc1":
        dtype = np.dtype(">f4" if is_bigendian else "<f4")
        scale = 1.0
    else:
        raise ValueError(f"Unsupported depth image encoding: {encoding!r}")

    row_bytes = width * dtype.itemsize
    if step <= 0:
        step = row_bytes
    if step < row_bytes:
        raise ValueError(f"Depth image step {step} is smaller than row bytes {row_bytes}")

    raw = decode_uint8_array(msg.get("data", ""))
    expected = height * step
    if len(raw) < expected:
        raise ValueError(f"Depth image payload {len(raw)} bytes is smaller than expected {expected}")

    rows = np.frombuffer(raw[:expected], dtype=np.uint8).reshape((height, step))
    packed = np.ascontiguousarray(rows[:, :row_bytes])
    depth = packed.view(dtype).reshape((height, width)).astype(np.float32)
    return depth * scale


def compressed_image_msg(
    encoded_bytes: bytes,
    image_format: str,
    frame_id: str,
    stamp: Optional[Dict[str, int]] = None,
) -> Dict[str, Any]:
    return {
        "header": {
            "stamp": stamp or ros_time_now(),
            "frame_id": frame_id,
        },
        "format": image_format,
        "data": base64.b64encode(encoded_bytes).decode("ascii"),
    }


def encode_image(frame: np.ndarray, extension: str, quality: int = 85) -> bytes:
    params = []
    if extension.lower() in {".jpg", ".jpeg"}:
        params = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
    ok, encoded = cv2.imencode(extension, frame, params)
    if not ok:
        raise ValueError(f"OpenCV could not encode image as {extension}")
    return encoded.tobytes()


def _decode_int8_array(data: Any) -> np.ndarray:
    if isinstance(data, str):
        raw = base64.b64decode(data)
        return np.frombuffer(raw, dtype=np.int8).astype(np.int16)
    if isinstance(data, list):
        return np.asarray(data, dtype=np.int16)
    raise ValueError(f"Unsupported int8[] payload type: {type(data)!r}")


def _float_or_default(value: Any, default: float) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def parse_occupancy_grid(msg: Dict[str, Any]) -> OccupancyGridMap:
    info = msg.get("info") or {}
    width = int(info.get("width", 0))
    height = int(info.get("height", 0))
    resolution = float(info.get("resolution", 0.0))
    origin = info.get("origin") or {}
    position = origin.get("position") or {}
    header = msg.get("header") or {}
    frame_id = header.get("frame_id") or "map"
    if width <= 0 or height <= 0 or resolution <= 0.0:
        raise ValueError("Invalid OccupancyGrid metadata")
    data = _decode_int8_array(msg.get("data", []))
    expected = width * height
    if data.size != expected:
        raise ValueError(f"OccupancyGrid data length {data.size} != {expected}")
    return OccupancyGridMap(
        width=width,
        height=height,
        resolution=resolution,
        origin_x=float(position.get("x", 0.0)),
        origin_y=float(position.get("y", 0.0)),
        frame_id=str(frame_id),
        data=data.reshape((height, width)),
    )


def parse_laser_scan(msg: Dict[str, Any]) -> LaserScan:
    ranges_raw = msg.get("ranges", [])
    if not isinstance(ranges_raw, list):
        raise ValueError(f"Unsupported LaserScan ranges payload type: {type(ranges_raw)!r}")
    ranges = np.asarray([_float_or_default(value, float("nan")) for value in ranges_raw], dtype=np.float32)
    angle_min = _float_or_default(msg.get("angle_min"), 0.0)
    angle_increment = _float_or_default(msg.get("angle_increment"), 0.0)
    angle_max = _float_or_default(msg.get("angle_max"), angle_min + angle_increment * max(0, ranges.size - 1))
    return LaserScan(
        angle_min=angle_min,
        angle_max=angle_max,
        angle_increment=angle_increment,
        range_min=_float_or_default(msg.get("range_min"), 0.0),
        range_max=_float_or_default(msg.get("range_max"), float("inf")),
        ranges=ranges,
    )


def quaternion_to_yaw(q: Dict[str, Any]) -> float:
    x = float(q.get("x", 0.0))
    y = float(q.get("y", 0.0))
    z = float(q.get("z", 0.0))
    w = float(q.get("w", 1.0))
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def parse_robot_pose(msg: Dict[str, Any]) -> RobotPoseMap:
    pose_container = msg.get("pose") or {}
    if isinstance(pose_container, dict) and "pose" in pose_container:
        pose_container = pose_container.get("pose") or {}
    position = pose_container.get("position") or {}
    orientation = pose_container.get("orientation") or {}
    return RobotPoseMap(
        x=float(position.get("x", 0.0)),
        y=float(position.get("y", 0.0)),
        yaw=quaternion_to_yaw(orientation),
    )


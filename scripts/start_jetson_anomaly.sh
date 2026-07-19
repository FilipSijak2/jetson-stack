#!/usr/bin/env bash
set -euo pipefail

: "${ANOMALY_CONFIG_FILE:=/workspace/config/anomaly_rosbridge.yaml}"
: "${ROSBRIDGE_URL:=ws://raspberry.local:9090}"
: "${CAMERA_TOPIC:=/camera/realsense/color/image_raw/compressed}"
: "${DEPTH_TOPIC:=/camera/realsense/aligned_depth_to_color/image_raw/compressedDepth}"
: "${MAP_TOPIC:=/map}"
: "${ROBOT_POSE_TOPIC:=/robot_pose_map}"
: "${EVENT_TOPIC:=/anomaly/events}"
: "${MARKER_TOPIC:=/anomaly/markers}"
: "${DEBUG_IMAGE_TOPIC:=/anomaly/debug_image/compressed}"
: "${MAP_SNAPSHOT_TOPIC:=/anomaly/map_snapshot/compressed}"
: "${JETSON_ARTIFACT_ROOT:=/home/jetson/anomaly_logs}"
: "${JETSON_LOG_DIR:=/workspace/logs}"
: "${SAVE_PER_EVENT_IMAGES:=1}"
: "${ANOMALY_MIN_OBSERVATIONS:=2}"
: "${REPORTED_MERGE_RADIUS_M:=2.00}"
: "${USE_DEPTH_DISTANCE:=1}"

mkdir -p "${JETSON_ARTIFACT_ROOT}"
mkdir -p "${JETSON_LOG_DIR}"
mkdir -p /workspace/models

LOG_FILE="${JETSON_LOG_DIR}/jetson_anomaly_$(date -u '+%Y%m%d').log"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "[jetson-anomaly] Starting rosbridge YOLO client"
echo "[jetson-anomaly] config=${ANOMALY_CONFIG_FILE}"
echo "[jetson-anomaly] rosbridge=${ROSBRIDGE_URL}"
echo "[jetson-anomaly] camera_topic=${CAMERA_TOPIC}"
echo "[jetson-anomaly] depth_topic=${DEPTH_TOPIC}"
echo "[jetson-anomaly] use_depth_distance=${USE_DEPTH_DISTANCE}"
echo "[jetson-anomaly] map_topic=${MAP_TOPIC}"
echo "[jetson-anomaly] robot_pose_topic=${ROBOT_POSE_TOPIC}"
echo "[jetson-anomaly] event_topic=${EVENT_TOPIC}"
echo "[jetson-anomaly] marker_topic=${MARKER_TOPIC}"
echo "[jetson-anomaly] artifact_root=${JETSON_ARTIFACT_ROOT}"
echo "[jetson-anomaly] save_per_event_images=${SAVE_PER_EVENT_IMAGES}"
echo "[jetson-anomaly] anomaly_min_observations=${ANOMALY_MIN_OBSERVATIONS}"
echo "[jetson-anomaly] reported_merge_radius_m=${REPORTED_MERGE_RADIUS_M}"
echo "[jetson-anomaly] log_file=${LOG_FILE}"
echo "[jetson-anomaly] Jetson runs as a plain Python WebSocket client; no ROS 2 DDS runtime is used"

python3 - <<'PY' || true
try:
    import torch
    print(
        "[jetson-anomaly] torch="
        f"{torch.__version__} torch_cuda={torch.version.cuda} "
        f"cuda_available={torch.cuda.is_available()}"
    )
except Exception as exc:
    print(f"[jetson-anomaly][WARN] torch import/check failed: {exc}")
PY

export PYTHONPATH="/workspace/src/jetson_anomaly_detector:${PYTHONPATH:-}"
cd /workspace/models
exec python3 -m jetson_anomaly_detector.jetson_yolo_rosbridge_client \
  --config "${ANOMALY_CONFIG_FILE}"

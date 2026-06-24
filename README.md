# Jetson YOLO Rosbridge Anomaly Client

Jetson Orin runs the complete YOLO anomaly detection and evidence generation
pipeline as a plain Python WebSocket client. It never joins the Raspberry Pi
ROS 2 DDS graph. It connects to the Raspberry Pi through `rosbridge_server`
WebSocket and publishes only the small visualization topics needed by Foxglove.

## Runtime Split

- Raspberry Pi runs ROS 2 navigation, SLAM, `/map`, `/tf`, `/tf_static`,
  `/odom`, `/scan`, camera publishing, `rosbridge_server`, and
  `foxglove_bridge`.
- Jetson runs `jetson_yolo_rosbridge_client`, YOLO inference, local image
  saving, event logging, marker generation, and map snapshot generation.
- Foxglove connects to the Raspberry Pi, usually `ws://raspberry.local:8765`.
- Jetson saves anomaly files under `/home/jetson/anomaly_logs` and does not copy
  images back to the Raspberry Pi.

## Topics

Jetson subscribes through rosbridge:

- `/camera/color/image/compressed` (`sensor_msgs/CompressedImage`)
- `/map` (`nav_msgs/OccupancyGrid`)
- `/robot_pose_map` (`geometry_msgs/PoseStamped`) or `/amcl_pose`
  (`geometry_msgs/PoseWithCovarianceStamped`) when configured
- `/scan` (`sensor_msgs/LaserScan`) when laser distance localization is enabled

Jetson publishes back through rosbridge:

- `/anomaly/events` (`std_msgs/String`, JSON)
- `/anomaly/events/readable` (`std_msgs/String`, human-readable summary)
- `/anomaly/markers` (`visualization_msgs/MarkerArray`, frame `map`)
- `/anomaly/debug_image/compressed` (`sensor_msgs/CompressedImage`)
- `/anomaly/map_snapshot/compressed` (`sensor_msgs/CompressedImage`)

Default anomaly class is only `bottle`. YOLO can detect other objects, but they
do not create events unless added to `ANOMALY_CLASSES`.

## Configuration

Compose-level settings are in `.env`:

```bash
cp .env.example .env
```

Runtime anomaly settings live in `config/containers/jetson_anomaly.env`, matching
the robot stack's `config/containers/*.env` layout. Important defaults:

```bash
ROSBRIDGE_URL=ws://raspberry.local:9090
CAMERA_TOPIC=/camera/color/image/compressed
DEPTH_TOPIC=/camera/realsense/aligned_depth_to_color/image_raw
MAP_TOPIC=/map
ROBOT_POSE_TOPIC=/robot_pose_map
SCAN_TOPIC=/scan
USE_DEPTH_DISTANCE=1
DEPTH_ROI_SCALE=0.60
DEPTH_MIN_VALID_PIXELS=20
USE_LASER_DISTANCE=1
LASER_WINDOW_DEG=6
ANOMALY_CLASSES=bottle
CONFIDENCE_THRESHOLD=0.5
DETECTION_COOLDOWN_S=5
CLUSTER_MERGE_RADIUS_M=1.00
MARKER_ASSOCIATION_RADIUS_M=2.00
REPORTED_MERGE_RADIUS_M=2.00
ANOMALY_MIN_OBSERVATIONS=2
ANOMALY_CONFIRMATION_TTL_S=6.0
SAVE_PER_EVENT_IMAGES=1
DAILY_MAP_SUMMARY=1
DEBUG_IMAGE_ALWAYS_STREAM=1
DEBUG_IMAGE_ON_DETECTION=1
DEBUG_IMAGE_PUBLISH_HZ=2
MARKER_TTL_S=180
MARKER_REPUBLISH_HZ=1
DEFAULT_ANOMALY_DISTANCE_M=1.5
CAMERA_HORIZONTAL_FOV_DEG=69
CAMERA_YAW_OFFSET_DEG=0
YOLO_MODEL_PATH=yolov8n.pt
JETSON_ARTIFACT_ROOT=/home/jetson/anomaly_logs
JETSON_LOG_DIR=/workspace/logs
```

The structured YAML defaults live in `config/anomaly_rosbridge.yaml`.
Environment variables from `config/containers/jetson_anomaly.env` override the
YAML values.

When RealSense aligned depth is available, Jetson estimates object distance
from valid depth pixels inside the central part of the detected bounding box.
This keeps a table leg or other obstacle in front of the object from being
mistaken for the bottle distance. If depth is unavailable or too sparse, Jetson
falls back to the laser range around the detected bounding-box bearing. If no
valid scan range is available either, it falls back to
`DEFAULT_ANOMALY_DISTANCE_M`.

Detections with the same label within `CLUSTER_MERGE_RADIUS_M` are merged into
one map square and marker text shows the observed count, for example
`bottle x3`. A map marker/event is created only after
`ANOMALY_MIN_OBSERVATIONS` spatially consistent observations within
`ANOMALY_CONFIRMATION_TTL_S`, which filters one-frame pose jumps from the
camera-to-map projection. Live markers use the wider
`MARKER_ASSOCIATION_RADIUS_M` to keep one noisy physical object on one stable
marker ID. Already reported objects are de-duplicated with
`REPORTED_MERGE_RADIUS_M`, intentionally at least as wide as the live marker
association radius. By default Jetson saves original/annotated camera images
for each new event and keeps a daily map summary at
`/home/jetson/anomaly_logs/map_images/daily/anomalies_YYYY-MM-DD.png` as new
anomaly clusters are detected.

If your active RealSense compressed topic is namespaced differently, set for
example:

```bash
CAMERA_TOPIC=/camera/realsense/color/image_raw/compressed
```

or:

```bash
CAMERA_TOPIC=/camera/realsense/color/image_compressed
```

## YOLO Dependencies

The Dockerfile uses the JetPack 6-compatible L4T base image
`nvcr.io/nvidia/l4t-jetpack:r36.4.0`, installs Jetson CUDA 12.6 PyTorch wheels
from `https://pypi.jetson-ai-lab.io/jp6/cu126/+simple/`, then installs
Ultralytics without allowing pip to replace `torch` or `torchvision`.

Default build pins:

```bash
L4T_BASE=nvcr.io/nvidia/l4t-jetpack:r36.4.0
PYTORCH_INDEX_URL=https://pypi.jetson-ai-lab.io/jp6/cu126/+simple/
TORCH_VERSION=2.8.0
TORCHVISION_VERSION=0.23.0
CUDSS_VERSION=0.5.0.16
CUSPARSELT_VERSION=0.7.1
SYMPY_VERSION=1.13.3
ULTRALYTICS_VERSION=8.3.40
```

Build on the Jetson with:

```bash
docker compose -f docker-compose.yaml -f docker-compose.build.yaml build jetson_anomaly
docker compose up -d --force-recreate jetson_anomaly
```

The YOLO/PyTorch image downloads several large Jetson CUDA wheels. If the build
fails with `No space left on device`, clean Docker's local build cache first:

```bash
docker builder prune -af
docker system prune -af --volumes
df -h
```

Verify the container sees CUDA:

```bash
docker exec jetson_anomaly_cont python3 - <<'PY'
import torch
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
PY
```

If you want to skip all YOLO/PyTorch installation for a mock-only image, build
with:

```bash
INSTALL_ULTRALYTICS=false docker compose \
  -f docker-compose.yaml \
  -f docker-compose.build.yaml \
  build jetson_anomaly
```

For real inference, `MOCK_MODE` must stay `0` and `YOLO_MODEL_PATH` should point
to `yolov8n.pt` or another compatible model. `MOCK_MODE=1` is only a fallback
for rosbridge and visualization debugging.

## Run

On the Raspberry Pi:

```bash
ros2 launch rosbridge_server rosbridge_websocket_launch.xml
ros2 run foxglove_bridge foxglove_bridge --ros-args -p port:=8765 -p address:=0.0.0.0
```

On the Jetson:

```bash
cd jetson-stack
docker compose up -d
```

The runtime Compose file intentionally has no `build:` section. GitHub Actions
builds and loads the image on Jetson; Compose only starts the loaded image. For
manual local builds, use the build override:

```bash
docker compose -f docker-compose.yaml -f docker-compose.build.yaml build jetson_anomaly
docker compose -f docker-compose.yaml -f docker-compose.build.yaml up -d
```

Local development without Compose:

```bash
export PYTHONPATH=$PWD/src/jetson_anomaly_detector:$PYTHONPATH
python3 -m jetson_anomaly_detector.jetson_yolo_rosbridge_client \
  --config config/anomaly_rosbridge.yaml
```

## Automated Deploy

GitHub Actions deploys from `devel`:

1. validates Compose, Python, shell scripts, and the topic contract
2. builds the `linux/arm64` Docker image
3. streams the Docker archive to `docker load` on Jetson over Tailscale SSH
4. verifies the generated and runtime image tags on Jetson
5. runs `scripts/pull-unprotected.sh origin devel` on Jetson
6. starts the runtime stack with `docker compose up -d --remove-orphans`

One-time Jetson setup:

```bash
cd ~
git clone <repo-url> jetson-stack
cd jetson-stack
cp .env.example .env
mkdir -p anomaly_logs logs
```

Keep Jetson-specific runtime config local:

```text
.env
config/anomaly_rosbridge.yaml
config/containers/jetson_anomaly.env
models/
```

Those paths are protected by `scripts/protected-files.txt`.

GitHub repository secrets:

- `TAILSCALE_AUTHKEY`
- `JETSON_SSH_USER`
- `JETSON_SSH_PRIVATE_KEY_B64` or `JETSON_SSH_PRIVATE_KEY`

GitHub repository variables:

- `JETSON_HOST`, default `100.125.121.125`
- `JETSON_STACK_DIR`, default `~/jetson-stack`

`scripts/pull-unprotected.sh` only updates files listed in
`scripts/runtime-files.txt`, and it skips protected files when they already
exist locally. This means a deploy updates the Compose/runtime wrapper without
overwriting Jetson-local topics, rosbridge URL, YOLO model path, or artifacts.

## Event JSON

Example `/anomaly/events` payload:

```json
{
  "id": "anom_00042",
  "timestamp": "2026-06-16T14:22:31Z",
  "label": "bottle",
  "type": "semantic_object_anomaly",
  "confidence": 0.87,
  "status": "active",
  "ttl_sec": 180,
  "bbox_xyxy": [312, 210, 390, 420],
  "robot_pose_map": {"x": 1.52, "y": -0.48, "yaw": 1.31},
  "object_pose_map": {"x": 2.10, "y": -0.92, "z": 0.0},
  "cluster": {"id": "cluster_00003", "count": 2, "merge_radius_m": 0.2},
  "jetson_files": {
    "original_image": "/home/jetson/anomaly_logs/images/original/anom_00042_bottle.jpg",
    "annotated_image": "/home/jetson/anomaly_logs/images/annotated/anom_00042_bottle.jpg",
    "map_snapshot": "/home/jetson/anomaly_logs/map_images/daily/anomalies_2026-06-20.png",
    "daily_map_summary": "/home/jetson/anomaly_logs/map_images/daily/anomalies_2026-06-20.png",
    "event_log": "/home/jetson/anomaly_logs/events.jsonl"
  }
}
```

For Foxglove, use `/anomaly/events/readable` when you want a compact
human-readable event summary. Keep `/anomaly/events` for machine parsing and
bag analysis.

## Saved Artifacts

Jetson writes:

- original frames: `/home/jetson/anomaly_logs/images/original/`
- annotated frames: `/home/jetson/anomaly_logs/images/annotated/`
- daily map summaries: `/home/jetson/anomaly_logs/map_images/daily/`
- map snapshots: `/home/jetson/anomaly_logs/map_images/`
- JSONL event log: `/home/jetson/anomaly_logs/events.jsonl`

The Compose file mounts this directory to `./anomaly_logs` on the Jetson repo
checkout for easy inspection.

Runtime stdout/stderr from `jetson_yolo_rosbridge_client` is also appended to:

```text
./logs/jetson_anomaly_YYYYMMDD.log
```

## Log Viewer

The Compose stack includes a lightweight File Browser webapp, matching the
robot stack pattern:

```bash
docker compose up -d jetson_log_viewer
```

Open:

```text
http://<jetson-ip>:8081
```

It exposes:

- `logs/`: daily Jetson anomaly client runtime logs
- `anomaly_logs/`: original images, annotated images, map snapshots, and
  `events.jsonl`

Change the host port in `.env`:

```bash
JETSON_LOG_VIEWER_PORT=8081
```

## Foxglove

Connect Foxglove to the Raspberry Pi:

```text
ws://raspberry.local:8765
```

Useful panels/topics:

- `/map`
- `/tf`, `/tf_static`
- `/scan` or `/scan_filtered`
- `/odom`
- `/robot_pose_map`
- `/anomaly/markers`
- `/anomaly/events`
- `/anomaly/debug_image/compressed`
- `/anomaly/map_snapshot/compressed`

Markers are republished at 1 Hz and expire after `MARKER_TTL_S`, default 180
seconds. Expired markers are deleted through `visualization_msgs/Marker.DELETE`.

## Test Procedure

1. Start the Raspberry Pi robot stack.
2. Confirm compressed camera publishing.
3. Start rosbridge on the Raspberry Pi, port `9090`.
4. Start foxglove_bridge on the Raspberry Pi, port `8765`.
5. Start `/robot_pose_map` publisher if `/amcl_pose` is not used directly.
6. Open Foxglove and connect to `ws://raspberry.local:8765`.
7. Start the Jetson YOLO rosbridge client.
8. Place a bottle in front of the camera.
9. Confirm Jetson logs a bottle detection.
10. Confirm original and annotated images are saved locally on Jetson.
11. Confirm `events.jsonl` is appended locally on Jetson.
12. Confirm a map snapshot PNG is saved locally on Jetson.
13. Confirm `/anomaly/events` publishes JSON.
14. Confirm `/anomaly/markers` publishes in `map` frame.
15. Confirm `/anomaly/debug_image/compressed` publishes.
16. Confirm `/anomaly/map_snapshot/compressed` publishes.
17. Confirm Foxglove shows `ANOMALY: bottle` on the map.
18. Confirm the marker remains visible for 180 seconds and then disappears.
19. Confirm existing robot navigation still works.

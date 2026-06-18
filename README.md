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

Jetson publishes back through rosbridge:

- `/anomaly/events` (`std_msgs/String`, JSON)
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
MAP_TOPIC=/map
ROBOT_POSE_TOPIC=/robot_pose_map
ANOMALY_CLASSES=bottle
CONFIDENCE_THRESHOLD=0.5
DETECTION_COOLDOWN_S=5
MARKER_TTL_S=180
MARKER_REPUBLISH_HZ=1
DEFAULT_ANOMALY_DISTANCE_M=1.5
CAMERA_HORIZONTAL_FOV_DEG=69
YOLO_MODEL_PATH=yolov8n.pt
JETSON_ARTIFACT_ROOT=/home/jetson/anomaly_logs
JETSON_LOG_DIR=/workspace/logs
```

The structured YAML defaults live in `config/anomaly_rosbridge.yaml`.
Environment variables from `config/containers/jetson_anomaly.env` override the
YAML values.

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

The Dockerfile installs `websocket-client` and, by default, `ultralytics`.
Jetson PyTorch wheels are sometimes JetPack-specific. If you already install
NVIDIA's Jetson PyTorch manually, build with:

```bash
INSTALL_ULTRALYTICS=false docker compose \
  -f docker-compose.yaml \
  -f docker-compose.build.yaml \
  build jetson_anomaly
```

Then install YOLO dependencies inside your Jetson Python environment:

```bash
python3 -m pip install --upgrade pip
python3 -m pip install ultralytics
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
3. copies the Docker archive to Jetson over Tailscale SSH
4. runs `docker load` on Jetson
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
  "jetson_files": {
    "original_image": "/home/jetson/anomaly_logs/images/original/anom_00042_bottle.jpg",
    "annotated_image": "/home/jetson/anomaly_logs/images/annotated/anom_00042_bottle.jpg",
    "map_snapshot": "/home/jetson/anomaly_logs/map_images/anom_00042_bottle_map.png",
    "event_log": "/home/jetson/anomaly_logs/events.jsonl"
  }
}
```

## Saved Artifacts

Jetson writes:

- original frames: `/home/jetson/anomaly_logs/images/original/`
- annotated frames: `/home/jetson/anomaly_logs/images/annotated/`
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

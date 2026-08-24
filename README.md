# Jetson Anomaly Detection Stack

This repository runs the deployed YOLO segmentation and anomaly-localization pipeline on a Jetson Orin. It connects to the Raspberry Pi ROS 2 graph through rosbridge; it does not join the robot DDS network.

## Runtime

```text
Raspberry Pi camera, depth, scan, map and pose topics
                        |
                  rosbridge :9090
                        |
                   Jetson Orin
                        |
       YOLO segmentation, tracking and localization
                        |
       images, JSONL events and visualization topics
```

The Compose stack starts:

- `jetson_anomaly`, using `filipsijak2/jetson_anomaly:jetson_anomaly-dev-local`
- `jetson_log_viewer`, exposing read-only artifact directories on port `8081`

## Active configuration

The application runtime configuration consists of two files:

- `config/anomaly_rosbridge.yaml` contains topic names, model settings, thresholds, tracking, privacy output, 3D detections and inspection settings.
- `config/containers/jetson_anomaly.env` contains the runtime environment loaded by Compose.

Environment values override equivalent YAML values. Keep both files synchronized when changing a duplicated setting.

The active model is `yolov8n-seg.pt`, the anomaly class is `bottle`, tracking uses `bytetrack.yaml`, and GPU inference uses device `0` with half precision.

The active Raspberry Pi endpoint is:

```env
ROSBRIDGE_URL=ws://raspberry.local:9090
```

The configured inputs are:

- `/camera/realsense/color/image_raw/compressed`
- `/camera/realsense/aligned_depth_to_color/image_raw/compressedDepth`
- `/camera/realsense/color/camera_info`
- `/map`
- `/robot_pose_map`
- `/scan`

The configured outputs are:

- `/anomaly/events`
- `/anomaly/events/readable`
- `/anomaly/markers`
- `/anomaly/detections_3d`
- `/anomaly/debug_image/compressed`
- `/anomaly/privacy_image/compressed`
- `/anomaly/map_snapshot/compressed`
- `/anomaly/inspection/*`

## Storage

Compose mounts:

- `./models` at `/workspace/models`
- `./anomaly_logs` at `/home/jetson/anomaly_logs`
- `./logs` at `/workspace/logs`

Model binaries, generated images and logs are ignored by Git. See [models/README.md](./models/README.md) for model placement.

## Start

```bash
docker compose pull
docker compose up -d
docker compose logs -f jetson_anomaly
```

For a local image build:

```bash
docker compose -f docker-compose.yaml -f docker-compose.build.yaml build
```

## Validation

```bash
python -m pip install -r requirements-yolo.txt -r requirements-rosbridge.txt
pytest -q
python -m compileall src/jetson_anomaly_detector/jetson_anomaly_detector
bash -n scripts/start_jetson_anomaly.sh
```

## Deployment workflow

The GitHub Actions deployment job requires these repository secrets:

- `TAILSCALE_AUTHKEY`
- `JETSON_SSH_USER`
- `JETSON_SSH_PRIVATE_KEY_B64` or `JETSON_SSH_PRIVATE_KEY`

It also requires the `JETSON_HOST` repository variable. `JETSON_STACK_DIR` is optional and defaults to `~/jetson-stack`.

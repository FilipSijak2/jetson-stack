# Jetson Anomaly Detector

ROS 2 companion stack for a Jetson Orin Nano in the diploma robot.

Responsibility split:

- Raspberry Pi 5 runs the robot runtime: LiDAR, EKF, AMCL, Nav2 and motor bridge.
- Hailo AI Kit remains responsible for navigation obstacle detection through
  `/ai_kit/obstacles`.
- Jetson runs object detection and anomaly categorization from RealSense data.

The Jetson stack intentionally does not publish `/ai_kit/obstacles`.

## Topic Contract

Inputs:

- `/camera/realsense/color/image_raw`
- `/camera/realsense/aligned_depth_to_color/image_raw`

Outputs:

- `/jetson_ai/detections` (`std_msgs/String`, JSON list of detections)
- `/jetson_ai/anomaly_category` (`std_msgs/String`)
- `/jetson_ai/anomaly_detail` (`std_msgs/String`)
- `/anomaly_events` (`std_msgs/String`, JSON event payload)
- `/anomaly/debug_image` (`sensor_msgs/Image`, optional)

## Modes

`mock` mode verifies ROS connectivity without ML dependencies. `yolo` mode can
be enabled after the Jetson ML stack is ready.

## Quick start

```bash
cd jetson-stack
docker compose up -d --build
```

For local ROS development:

```bash
colcon build --symlink-install
source install/setup.bash
export ROS_DOMAIN_ID=0
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
ros2 launch jetson_anomaly_detector anomaly_detector.launch.py config_file:=$PWD/config/anomaly_detector.yaml
```

Check:

```bash
ros2 topic hz /camera/realsense/color/image_raw
ros2 topic echo /anomaly_events
ros2 topic echo /jetson_ai/anomaly_category
ros2 topic echo /jetson_ai/detections
```

## CI Image Build

GitHub Actions builds a `linux/arm64` image on PRs. On `devel` pushes it
exports the image to a Docker archive, copies it to the Jetson over Tailscale
SSH, and runs `docker load` directly on the Jetson. No registry is required on
the Jetson.

The Jetson Tailscale IP is configured in the workflow as `100.125.121.125`.

The version tag follows the diploma repository convention:

- `vX.Y.rcN` is extracted from the latest commit message when present
- otherwise it falls back to `dev-<short-sha>`
- the generated Docker tag is
  `<github-owner>/jetson_anomaly:jetson_anomaly-<version>`

The workflow also tags the same loaded image with the image name defined
directly in `docker-compose.yaml`. Change the service `image:` line there when
you want Compose to use a specific local image version, for example:

```yaml
image: filipsijak2/jetson_anomaly:jetson_anomaly-v1.2.rc1
```

Required GitHub secrets:

- `TAILSCALE_AUTHKEY`
- `JETSON_SSH_USER`
- `JETSON_SSH_PRIVATE_KEY`

After a successful push build, run on the Jetson:

```bash
docker compose up -d
```

## Suggested next steps

1. Run mock mode against the live camera topic.
2. Enable a lightweight model on the Jetson.
3. Tune `anomaly_labels` and depth thresholds in `config/anomaly_detector.yaml`.
4. Save event snapshots.
5. Connect `/jetson_ai/*` events back to the logging/map workflow.

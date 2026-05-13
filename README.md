# Jetson Anomaly Starter

Starter project for a Jetson Orin Nano companion node for the diploma robot.

Goal:

- Raspberry Pi 5 runs the robot runtime: LiDAR, EKF, AMCL, Nav2 and motor bridge.
- Jetson runs camera processing and publishes high-level anomaly events.

This folder can later be moved into a standalone repository.

## First milestone

1. Jetson joins the same ROS 2 domain as the robot.
2. Jetson subscribes to the RealSense RGB topic.
3. Jetson publishes JSON events on `/anomaly_events`.
4. Foxglove shows the event stream and optional debug image.

## Modes

`mock` mode verifies ROS connectivity without ML dependencies.

`yolo` mode can be enabled later after the Jetson ML stack is ready.

## Quick start

```bash
cd jetson_anomaly
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
```

## Suggested next steps

1. Run mock mode against the live camera topic.
2. Enable a lightweight model on the Jetson.
3. Add depth filtering.
4. Save event snapshots.
5. Connect events back to the navigation/map workflow.

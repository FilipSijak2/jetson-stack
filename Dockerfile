FROM ros:humble

ENV DEBIAN_FRONTEND=noninteractive
ENV AMENT_TRACE_SETUP_FILES=""

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    python3-colcon-common-extensions \
    python3-numpy \
    python3-opencv \
    python3-yaml \
    ros-humble-cv-bridge \
    ros-humble-rmw-cyclonedds-cpp \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY src /workspace/src
COPY config /workspace/config

RUN . /opt/ros/humble/setup.sh && colcon build --symlink-install

CMD ["bash", "-lc", "source /opt/ros/humble/setup.bash && source /workspace/install/setup.bash && ros2 launch jetson_anomaly_detector anomaly_detector.launch.py config_file:=/workspace/config/anomaly_detector.yaml"]

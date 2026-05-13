#!/usr/bin/env bash
set -euo pipefail

: "${JETSON_DDS_INTERFACE:=tailscale0}"
: "${CYCLONEDDS_URI:=file:///tmp/cyclonedds.xml}"
: "${ROS_DOMAIN_ID:=0}"
: "${ROBOT_TAILSCALE_IP:?Set ROBOT_TAILSCALE_IP to the Raspberry Pi Tailscale IP in .env or docker-compose.yaml}"

if [[ "${CYCLONEDDS_URI}" == "file:///tmp/cyclonedds.xml" ]]; then
  cat > /tmp/cyclonedds.xml <<XML
<?xml version="1.0" encoding="UTF-8"?>
<CycloneDDS xmlns="https://cdds.omg.org/schema">
  <Domain id="any">
    <General>
      <Interfaces>
        <NetworkInterface name="${JETSON_DDS_INTERFACE}" priority="default" multicast="false" />
      </Interfaces>
      <AllowMulticast>false</AllowMulticast>
      <MaxMessageSize>65500B</MaxMessageSize>
      <FragmentSize>4000B</FragmentSize>
    </General>
    <Discovery>
      <Peers>
        <Peer Address="${ROBOT_TAILSCALE_IP}" />
      </Peers>
      <ParticipantIndex>auto</ParticipantIndex>
      <MaxAutoParticipantIndex>60</MaxAutoParticipantIndex>
    </Discovery>
    <Tracing>
      <Verbosity>warning</Verbosity>
      <OutputFile>stdout</OutputFile>
    </Tracing>
  </Domain>
</CycloneDDS>
XML
fi

echo "[jetson-anomaly] ROS_DOMAIN_ID=${ROS_DOMAIN_ID}"
echo "[jetson-anomaly] CYCLONEDDS_URI=${CYCLONEDDS_URI}"
echo "[jetson-anomaly] DDS interface=${JETSON_DDS_INTERFACE}"
echo "[jetson-anomaly] Robot DDS peer=${ROBOT_TAILSCALE_IP}"

set +u
# shellcheck disable=SC1091
source /opt/ros/humble/setup.bash
# shellcheck disable=SC1091
source /workspace/install/setup.bash
set -u

exec ros2 launch jetson_anomaly_detector anomaly_detector.launch.py \
  config_file:=/workspace/config/anomaly_detector.yaml

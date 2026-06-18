FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    python3-pip \
    python3-numpy \
    python3-opencv \
    python3-yaml \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY src /workspace/src
COPY config /workspace/config
COPY scripts /workspace/scripts
COPY requirements-rosbridge.txt /workspace/requirements-rosbridge.txt
COPY requirements-yolo.txt /workspace/requirements-yolo.txt

ARG INSTALL_ULTRALYTICS=true
RUN python3 -m pip install --no-cache-dir -r /workspace/requirements-rosbridge.txt && \
    if [ "${INSTALL_ULTRALYTICS}" = "true" ]; then \
    python3 -m pip install --no-cache-dir -r /workspace/requirements-yolo.txt; \
    else \
    echo "[jetson-anomaly] Skipping ultralytics install (INSTALL_ULTRALYTICS=false)"; \
    fi

RUN python3 -m pip install --no-cache-dir --no-deps -e /workspace/src/jetson_anomaly_detector
RUN chmod +x /workspace/scripts/start_jetson_anomaly.sh

CMD ["/workspace/scripts/start_jetson_anomaly.sh"]

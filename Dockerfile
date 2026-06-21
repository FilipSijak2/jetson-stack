ARG L4T_BASE=nvcr.io/nvidia/l4t-jetpack:r36.4.0
FROM ${L4T_BASE}

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ROOT_USER_ACTION=ignore
ENV NVIDIA_PYTHON_LIBS=/usr/local/lib/python3.10/dist-packages/nvidia
ENV LD_LIBRARY_PATH="${NVIDIA_PYTHON_LIBS}/cu12/lib:${NVIDIA_PYTHON_LIBS}/cudss/lib:${NVIDIA_PYTHON_LIBS}/cusparselt/lib:${LD_LIBRARY_PATH}"

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    libopenblas-dev \
    python3-matplotlib \
    python3-pip \
    python3-pil \
    python3-psutil \
    python3-requests \
    python3-scipy \
    python3-seaborn \
    python3-tqdm \
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
ARG PYTORCH_INDEX_URL=https://pypi.jetson-ai-lab.io/jp6/cu126/+simple/
ARG TORCH_VERSION=2.8.0
ARG TORCHVISION_VERSION=0.23.0
ARG ULTRALYTICS_VERSION=8.3.40
RUN python3 -m pip install --no-cache-dir -r /workspace/requirements-rosbridge.txt && \
    if [ "${INSTALL_ULTRALYTICS}" = "true" ]; then \
    python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel && \
    python3 -m pip install --no-cache-dir \
        numpy==1.26.1 \
        filelock \
        fsspec \
        jinja2 \
        networkx \
        packaging \
        py-cpuinfo \
        sympy \
        typing_extensions && \
    python3 -m pip install --no-cache-dir nvidia-cudss-cu12 nvidia-cusparselt-cu12 && \
    (python3 -m pip uninstall -y \
        nvidia-cublas-cu12 \
        nvidia-cuda-runtime-cu12 \
        nvidia-cusparse-cu12 \
        nvidia-nvjitlink-cu12 2>/dev/null || true) && \
    printf '%s\n' \
        "${NVIDIA_PYTHON_LIBS}/cu12/lib" \
        "${NVIDIA_PYTHON_LIBS}/cudss/lib" \
        "${NVIDIA_PYTHON_LIBS}/cusparselt/lib" \
        > /etc/ld.so.conf.d/nvidia-pip-libs.conf && \
    ldconfig && \
    python3 -m pip install --no-cache-dir \
        --no-deps \
        --index-url "${PYTORCH_INDEX_URL}" \
        "torch==${TORCH_VERSION}" \
        "torchvision==${TORCHVISION_VERSION}" && \
    python3 -m pip install --no-cache-dir --no-deps \
        -r /workspace/requirements-yolo.txt \
        "ultralytics==${ULTRALYTICS_VERSION}"; \
    else \
    echo "[jetson-anomaly] Skipping ultralytics install (INSTALL_ULTRALYTICS=false)"; \
    fi

RUN python3 -m pip install --no-cache-dir --no-deps -e /workspace/src/jetson_anomaly_detector
RUN chmod +x /workspace/scripts/start_jetson_anomaly.sh

CMD ["/workspace/scripts/start_jetson_anomaly.sh"]

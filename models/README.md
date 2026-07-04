# Models

Place Jetson detector assets here, for example:

- `yolov8n.pt`
- `yolov8n-seg.pt` (default; required for exact bottle masks)
- ONNX/TensorRT exports used by future backends

Large model files are ignored by Git. The Compose stack mounts this directory
at `/workspace/models` and starts the client from there, so an automatically
downloaded Ultralytics model persists across container recreation.

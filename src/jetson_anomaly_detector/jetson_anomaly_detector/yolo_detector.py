from __future__ import annotations

import logging
from typing import List, Sequence

import numpy as np

from .models import Detection


class YoloDetector:
    def __init__(
        self,
        model_path: str,
        confidence_threshold: float,
        anomaly_classes: Sequence[str],
        mock_mode: bool,
        logger: logging.Logger,
    ) -> None:
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.anomaly_classes = {label.strip() for label in anomaly_classes if label.strip()}
        self.mock_mode = mock_mode
        self.logger = logger
        self.model = None

        if self.mock_mode:
            self.logger.warning("MOCK_MODE is enabled; YOLO inference is disabled")
            return

        try:
            from ultralytics import YOLO  # type: ignore
        except Exception as exc:  # pragma: no cover - depends on Jetson runtime
            raise RuntimeError(
                "Ultralytics YOLO is required for real inference. "
                "Install Jetson-compatible PyTorch and ultralytics, or set MOCK_MODE=1 for debugging."
            ) from exc

        self.logger.info("Loading YOLO model: %s", self.model_path)
        self.model = YOLO(self.model_path)

    def detect(self, frame: np.ndarray) -> List[Detection]:
        if self.mock_mode:
            height, width = frame.shape[:2]
            return [
                Detection(
                    label="bottle",
                    confidence=0.99,
                    bbox_xyxy=[
                        int(width * 0.42),
                        int(height * 0.35),
                        int(width * 0.58),
                        int(height * 0.82),
                    ],
                )
            ]

        if self.model is None:
            return []

        results = self.model.predict(frame, verbose=False, conf=self.confidence_threshold)
        detections: List[Detection] = []
        for result in results:
            names = getattr(result, "names", {})
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            for box in boxes:
                cls_id = int(box.cls[0].item())
                confidence = float(box.conf[0].item())
                if confidence < self.confidence_threshold:
                    continue
                xyxy = [int(round(value)) for value in box.xyxy[0].tolist()]
                detections.append(
                    Detection(
                        label=str(names.get(cls_id, cls_id)),
                        confidence=confidence,
                        bbox_xyxy=xyxy,
                    )
                )
        return detections


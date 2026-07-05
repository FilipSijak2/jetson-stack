from __future__ import annotations

import logging
from typing import Any, List, Optional, Sequence

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
        image_size: int = 640,
        iou_threshold: float = 0.70,
        max_detections: int = 20,
        device: str = "0",
        half: bool = True,
        augment: bool = False,
        agnostic_nms: bool = False,
        filter_classes: bool = True,
        tracking_enabled: bool = True,
        tracking_backend: str = "bytetrack.yaml",
        tracking_confidence_threshold: float = 0.25,
        segmentation_enabled: bool = True,
    ) -> None:
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.anomaly_classes = {label.strip() for label in anomaly_classes if label.strip()}
        self.mock_mode = mock_mode
        self.logger = logger
        self.image_size = image_size
        self.iou_threshold = iou_threshold
        self.max_detections = max_detections
        self.device = device
        self.half = half and str(device).strip().lower() != "cpu"
        self.augment = augment
        self.agnostic_nms = agnostic_nms
        self.filter_classes = filter_classes
        self.tracking_enabled = tracking_enabled
        self.tracking_backend = tracking_backend
        self.tracking_confidence_threshold = tracking_confidence_threshold
        self.segmentation_enabled = segmentation_enabled
        self.model = None
        self.class_ids: List[int] = []
        self._missing_masks_warned = False

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
        names = getattr(self.model, "names", {})
        if isinstance(names, dict):
            self.class_ids = [
                int(class_id)
                for class_id, label in names.items()
                if str(label) in self.anomaly_classes
            ]
        elif isinstance(names, (list, tuple)):
            self.class_ids = [
                class_id for class_id, label in enumerate(names) if str(label) in self.anomaly_classes
            ]
        self.logger.info(
            "YOLO inference imgsz=%d conf=%.2f iou=%.2f max_det=%d device=%s half=%s "
            "augment=%s class_filter=%s tracking=%s tracker=%s tracking_conf=%.2f "
            "segmentation=%s",
            self.image_size,
            self.confidence_threshold,
            self.iou_threshold,
            self.max_detections,
            self.device,
            self.half,
            self.augment,
            self.class_ids if self.filter_classes else "disabled",
            self.tracking_enabled,
            self.tracking_backend,
            self.tracking_confidence_threshold,
            self.segmentation_enabled,
        )

    def detect(self, frame: np.ndarray) -> List[Detection]:
        if self.mock_mode:
            height, width = frame.shape[:2]
            x1, y1, x2, y2 = (
                int(width * 0.42),
                int(height * 0.35),
                int(width * 0.58),
                int(height * 0.82),
            )
            mask = None
            if self.segmentation_enabled:
                mask_image = np.zeros((height, width), dtype=np.uint8)
                center = ((x1 + x2) // 2, (y1 + y2) // 2)
                axes = (max(1, (x2 - x1) // 3), max(1, (y2 - y1) // 2))
                import cv2

                cv2.ellipse(mask_image, center, axes, 0, 0, 360, 1, -1)
                mask = mask_image.astype(bool)
            return [
                Detection(
                    label="bottle",
                    confidence=0.99,
                    bbox_xyxy=[x1, y1, x2, y2],
                    track_id=1 if self.tracking_enabled else None,
                    mask=mask,
                )
            ]

        if self.model is None:
            return []

        inference_confidence = (
            min(self.confidence_threshold, self.tracking_confidence_threshold)
            if self.tracking_enabled
            else self.confidence_threshold
        )
        predict_args = {
            "verbose": False,
            "conf": inference_confidence,
            "iou": self.iou_threshold,
            "imgsz": self.image_size,
            "max_det": self.max_detections,
            "device": self.device,
            "half": self.half,
            "augment": self.augment,
            "agnostic_nms": self.agnostic_nms,
        }
        if self.filter_classes and self.class_ids:
            predict_args["classes"] = self.class_ids
        if self.tracking_enabled:
            results = self.model.track(
                frame,
                persist=True,
                tracker=self.tracking_backend,
                **predict_args,
            )
        else:
            results = self.model.predict(frame, **predict_args)
        return self._parse_results(results, frame.shape[:2])

    def _parse_results(
        self, results: Any, frame_shape: Sequence[int]
    ) -> List[Detection]:
        detections: List[Detection] = []
        for result in results:
            names = getattr(result, "names", {})
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            masks = getattr(getattr(result, "masks", None), "data", None)
            try:
                box_count = len(boxes)
            except TypeError:
                box_count = 0
            if (
                self.segmentation_enabled
                and box_count > 0
                and masks is None
                and not self._missing_masks_warned
            ):
                self.logger.warning(
                    "SEGMENTATION_ENABLED=1 and detections exist, but the model "
                    "returned no masks. "
                    "Use a *-seg.pt/*-seg.engine model; falling back to bounding boxes."
                )
                self._missing_masks_warned = True
            for index, box in enumerate(boxes):
                cls_id = int(box.cls[0].item())
                confidence = float(box.conf[0].item())
                minimum_confidence = (
                    min(self.confidence_threshold, self.tracking_confidence_threshold)
                    if self.tracking_enabled
                    else self.confidence_threshold
                )
                if confidence < minimum_confidence:
                    continue
                xyxy = [int(round(value)) for value in box.xyxy[0].tolist()]
                track_id = _optional_scalar_int(getattr(box, "id", None))
                mask = None
                if self.segmentation_enabled and masks is not None:
                    mask = _extract_mask(masks, index, frame_shape)
                detections.append(
                    Detection(
                        label=_class_name(names, cls_id),
                        confidence=confidence,
                        bbox_xyxy=xyxy,
                        track_id=track_id,
                        mask=mask,
                    )
                )
        return detections


def _class_name(names: Any, class_id: int) -> str:
    if isinstance(names, dict):
        return str(names.get(class_id, class_id))
    if isinstance(names, (list, tuple)) and 0 <= class_id < len(names):
        return str(names[class_id])
    return str(class_id)


def _optional_scalar_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        if hasattr(value, "numel") and int(value.numel()) == 0:
            return None
        if hasattr(value, "reshape") and hasattr(value, "numel"):
            return int(value.reshape(-1)[0].item())
        if hasattr(value, "item"):
            return int(value.item())
        if isinstance(value, (list, tuple, np.ndarray)):
            return int(value[0]) if len(value) else None
        return int(value)
    except (IndexError, TypeError, ValueError):
        return None


def _extract_mask(
    masks: Any, index: int, frame_shape: Sequence[int]
) -> Optional[np.ndarray]:
    try:
        value = masks[index]
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "numpy"):
            value = value.numpy()
        mask = np.asarray(value, dtype=np.float32)
        mask = np.squeeze(mask)
        if mask.ndim != 2:
            return None
        height, width = int(frame_shape[0]), int(frame_shape[1])
        if mask.shape != (height, width):
            import cv2

            mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
        return mask > 0.5
    except (IndexError, TypeError, ValueError):
        return None


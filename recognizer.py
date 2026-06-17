from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


_IGNORED_STATIC_LABELS = {"J", "Z"}


class SignRecognizer:
    def __init__(self, model_path: str, conf_threshold: float = 0.5):
        model_file = Path(model_path)
        if not model_file.exists():
            raise FileNotFoundError(
                f"Model not found: {model_file}. Put your trained weight file in this path."
            )

        self.model = YOLO(str(model_file))
        self.conf_threshold = conf_threshold

    def predict(
        self,
        frame: np.ndarray,
        draw_labels: bool = True,
        draw_conf: bool = True,
    ) -> tuple[np.ndarray, str | None, float]:
        results = self.model.predict(frame, verbose=False, conf=self.conf_threshold)
        result = results[0]

        yolo_label: str | None = None
        yolo_conf: float = 0.0

        if hasattr(result, "probs") and result.probs is not None:
            yolo_label = self.model.names[int(result.probs.top1)]
            yolo_conf = float(result.probs.top1conf)
        elif result.boxes is not None and len(result.boxes) > 0:
            keep_indexes = [
                idx
                for idx, cls in enumerate(result.boxes.cls.cpu().numpy())
                if self.model.names[int(cls)] not in _IGNORED_STATIC_LABELS
            ]
            if len(keep_indexes) != len(result.boxes):
                result.boxes = result.boxes[keep_indexes]

            confidences = result.boxes.conf.cpu().numpy()
            if len(confidences) > 0:
                best_idx = int(np.argmax(confidences))
                yolo_conf = float(confidences[best_idx])
                yolo_label = self.model.names[int(result.boxes.cls[best_idx].item())]

        if yolo_label in _IGNORED_STATIC_LABELS:
            yolo_label = None
            yolo_conf = 0.0

        annotated = result.plot(labels=draw_labels, conf=draw_conf)
        return annotated, yolo_label, yolo_conf

    @staticmethod
    def open_camera(camera_index: int = 0) -> cv2.VideoCapture:
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            raise RuntimeError("Could not open webcam.")
        return cap

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


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
        annotated = result.plot(labels=draw_labels, conf=draw_conf)

        if hasattr(result, "probs") and result.probs is not None:
            label_index = int(result.probs.top1)
            confidence = float(result.probs.top1conf)
            label = self.model.names[label_index]
            return annotated, label, confidence

        if result.boxes is not None and len(result.boxes) > 0:
            confidences = result.boxes.conf.cpu().numpy()
            best_idx = int(np.argmax(confidences))
            confidence = float(confidences[best_idx])
            class_id = int(result.boxes.cls[best_idx].item())
            label = self.model.names[class_id]
            return annotated, label, confidence

        return annotated, None, 0.0

    @staticmethod
    def open_camera(camera_index: int = 0) -> cv2.VideoCapture:
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            raise RuntimeError("Could not open webcam.")
        return cap

from __future__ import annotations

import collections
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
    _MEDIAPIPE_AVAILABLE = True
except ImportError:
    _MEDIAPIPE_AVAILABLE = False

_TRACK_LEN = 64
_INDEX_TIP = 8
_DEFAULT_LANDMARKER = "models/hand_landmarker.task"


def recognize_gesture(points):
    if len(points) < 15:
        return None

    pts = list(points)
    n = len(pts)
    first = pts[:n // 3]
    mid = pts[n // 3 : 2 * n // 3]
    last = pts[2 * n // 3 :]
    first_half = pts[: n // 2]
    second_half = pts[n // 2 :]

    dy_first = first_half[-1][1] - first_half[0][1]
    dy_second = second_half[-1][1] - second_half[0][1]
    dx_second = second_half[-1][0] - second_half[0][0]
    dx_first_half = first_half[-1][0] - first_half[0][0]

    if (
        dy_first > 80
        and dy_second < -30
        and dx_second < -30
        and abs(dx_first_half) < 60
    ):
        return "J"

    dx_first_z = first[-1][0] - first[0][0]
    dx_mid = mid[-1][0] - mid[0][0]
    dy_mid = mid[-1][1] - mid[0][1]
    dx_last = last[-1][0] - last[0][0]
    dy_first_z = first[-1][1] - first[0][1]

    if (
        dx_first_z > 50
        and dx_mid < -30
        and dy_mid > 30
        and dx_last > 50
        and abs(dy_first_z) < 40
    ):
        return "Z"

    return None


def _is_index_extended(landmarks) -> bool:
    tip = landmarks[8]   # INDEX_FINGER_TIP
    pip = landmarks[6]   # INDEX_FINGER_PIP (중간 마디)
    mcp = landmarks[5]   # INDEX_FINGER_MCP (손가락 밑 관절)
    # 검지가 펴진 경우: 끝이 중간 마디보다 손바닥에서 더 멀리 있음
    tip_dist = ((tip.x - mcp.x) ** 2 + (tip.y - mcp.y) ** 2) ** 0.5
    pip_dist = ((pip.x - mcp.x) ** 2 + (pip.y - mcp.y) ** 2) ** 0.5
    return tip_dist > pip_dist * 1.2


class SignRecognizer:
    def __init__(
        self,
        model_path: str,
        conf_threshold: float = 0.5,
        landmarker_path: str = _DEFAULT_LANDMARKER,
    ):
        model_file = Path(model_path)
        if not model_file.exists():
            raise FileNotFoundError(
                f"Model not found: {model_file}. Put your trained weight file in this path."
            )

        self.model = YOLO(str(model_file))
        self.conf_threshold = conf_threshold
        self.track_points: collections.deque[tuple[int, int]] = collections.deque(
            maxlen=_TRACK_LEN
        )

        self._landmarker = None
        if _MEDIAPIPE_AVAILABLE:
            lm_file = Path(landmarker_path)
            if lm_file.exists():
                base_opts = mp_python.BaseOptions(model_asset_path=str(lm_file))
                options = mp_vision.HandLandmarkerOptions(
                    base_options=base_opts,
                    num_hands=1,
                )
                self._landmarker = mp_vision.HandLandmarker.create_from_options(options)

    def predict(
        self,
        frame: np.ndarray,
        draw_labels: bool = True,
        draw_conf: bool = True,
    ) -> tuple[np.ndarray, str | None, float]:
        results = self.model.predict(frame, verbose=False, conf=self.conf_threshold)
        result = results[0]
        annotated = result.plot(labels=draw_labels, conf=draw_conf)

        yolo_label: str | None = None
        yolo_conf: float = 0.0

        if hasattr(result, "probs") and result.probs is not None:
            yolo_label = self.model.names[int(result.probs.top1)]
            yolo_conf = float(result.probs.top1conf)
        elif result.boxes is not None and len(result.boxes) > 0:
            confidences = result.boxes.conf.cpu().numpy()
            best_idx = int(np.argmax(confidences))
            yolo_conf = float(confidences[best_idx])
            yolo_label = self.model.names[int(result.boxes.cls[best_idx].item())]

        # MediaPipe: 검지 끝(랜드마크 8) 추적
        if self._landmarker is not None:
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            detection = self._landmarker.detect(mp_image)

            if detection.hand_landmarks:
                lms = detection.hand_landmarks[0]
                if _is_index_extended(lms):
                    tip = lms[_INDEX_TIP]
                    self.track_points.append((int(tip.x * w), int(tip.y * h)))
                else:
                    self.track_points.clear()
            else:
                self.track_points.clear()

        # 파란색 궤적 선 그리기
        pts = list(self.track_points)
        for i in range(1, len(pts)):
            cv2.line(annotated, pts[i - 1], pts[i], (255, 0, 0), 2)

        # J/Z 궤적 감지 시 YOLO 결과 덮어쓰기
        gesture = recognize_gesture(self.track_points)
        if gesture is not None:
            return annotated, gesture, 1.0

        return annotated, yolo_label, yolo_conf

    @staticmethod
    def open_camera(camera_index: int = 0) -> cv2.VideoCapture:
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            raise RuntimeError("Could not open webcam.")
        return cap

from __future__ import annotations

import collections
import time
from enum import Enum
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
_PINKY_TIP = 20
_DEFAULT_LANDMARKER = "models/hand_landmarker.task"
_IGNORED_STATIC_LABELS = {"J", "Z"}
_MIN_DYNAMIC_POINTS = 8
_DYNAMIC_START_SPEED = 32.0
_DYNAMIC_STOP_SPEED = 10.0
_DYNAMIC_DOMINANCE_RATIO = 1.2
_DYNAMIC_TIMEOUT_SECONDS = 2.0
_DYNAMIC_COOLDOWN_SECONDS = 0.8


class GestureState(Enum):
    DETECTING = "detecting"
    MOVING = "moving"
    JUDGING = "judging"
    COOLDOWN = "cooldown"


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


def _distance(first: tuple[int, int] | None, second: tuple[int, int] | None) -> float:
    if first is None or second is None:
        return 0.0
    return float(np.hypot(second[0] - first[0], second[1] - first[1]))


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
        self.gesture_state = GestureState.DETECTING
        self.gesture_mode: str | None = None
        self.move_started_at = 0.0
        self.cooldown_until = 0.0
        self.prev_index_tip: tuple[int, int] | None = None
        self.prev_pinky_tip: tuple[int, int] | None = None

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

    def _reset_dynamic_state(self) -> None:
        self.gesture_state = GestureState.DETECTING
        self.gesture_mode = None
        self.move_started_at = 0.0
        self.track_points.clear()

    def _detect_hand_tips(
        self, frame: np.ndarray
    ) -> tuple[tuple[int, int] | None, tuple[int, int] | None]:
        if self._landmarker is None:
            return None, None

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        detection = self._landmarker.detect(mp_image)

        if not detection.hand_landmarks:
            self.prev_index_tip = None
            self.prev_pinky_tip = None
            return None, None

        lms = detection.hand_landmarks[0]
        index_tip = lms[_INDEX_TIP]
        pinky_tip = lms[_PINKY_TIP]
        return (
            (int(index_tip.x * w), int(index_tip.y * h)),
            (int(pinky_tip.x * w), int(pinky_tip.y * h)),
        )

    def _update_dynamic_state(
        self,
        frame: np.ndarray,
    ) -> tuple[str | None, bool]:
        now = time.monotonic()

        if self.gesture_state == GestureState.COOLDOWN:
            if now < self.cooldown_until:
                return None, True
            self._reset_dynamic_state()

        index_tip, pinky_tip = self._detect_hand_tips(frame)
        index_speed = _distance(self.prev_index_tip, index_tip)
        pinky_speed = _distance(self.prev_pinky_tip, pinky_tip)
        self.prev_index_tip = index_tip
        self.prev_pinky_tip = pinky_tip

        if index_tip is None or pinky_tip is None:
            self._reset_dynamic_state()
            return None, False

        if self.gesture_state == GestureState.DETECTING:
            index_is_dominant = index_speed > pinky_speed * _DYNAMIC_DOMINANCE_RATIO
            pinky_is_dominant = pinky_speed > index_speed * _DYNAMIC_DOMINANCE_RATIO

            if index_speed >= _DYNAMIC_START_SPEED and index_is_dominant:
                self.gesture_state = GestureState.MOVING
                self.gesture_mode = "Z"
                self.move_started_at = now
                self.track_points.clear()
                self.track_points.append(index_tip)
                return None, True

            if pinky_speed >= _DYNAMIC_START_SPEED and pinky_is_dominant:
                self.gesture_state = GestureState.MOVING
                self.gesture_mode = "J"
                self.move_started_at = now
                self.track_points.clear()
                self.track_points.append(pinky_tip)
                return None, True

            return None, False

        if self.gesture_state == GestureState.MOVING:
            active_tip = index_tip if self.gesture_mode == "Z" else pinky_tip
            active_speed = index_speed if self.gesture_mode == "Z" else pinky_speed
            self.track_points.append(active_tip)

            if now - self.move_started_at > _DYNAMIC_TIMEOUT_SECONDS:
                self._reset_dynamic_state()
                return None, False

            if (
                active_speed <= _DYNAMIC_STOP_SPEED
                and len(self.track_points) >= _MIN_DYNAMIC_POINTS
            ):
                self.gesture_state = GestureState.JUDGING
                gesture = recognize_gesture(self.track_points)
                if gesture == self.gesture_mode:
                    self.gesture_state = GestureState.COOLDOWN
                    self.cooldown_until = now + _DYNAMIC_COOLDOWN_SECONDS
                    self.track_points.clear()
                    return gesture, True

                self._reset_dynamic_state()
                return None, False

            return None, True

        return None, False

    def _predict_static_yolo(
        self,
        frame: np.ndarray,
        draw_labels: bool,
        draw_conf: bool,
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

    def predict(
        self,
        frame: np.ndarray,
        draw_labels: bool = True,
        draw_conf: bool = True,
    ) -> tuple[np.ndarray, str | None, float]:
        gesture, suppress_yolo = self._update_dynamic_state(frame)
        if suppress_yolo:
            annotated = frame.copy()
        else:
            annotated, yolo_label, yolo_conf = self._predict_static_yolo(
                frame,
                draw_labels=draw_labels,
                draw_conf=draw_conf,
            )

        # 파란색 궤적 선 그리기
        pts = list(self.track_points)
        for i in range(1, len(pts)):
            cv2.line(annotated, pts[i - 1], pts[i], (255, 0, 0), 2)

        if gesture is not None:
            return annotated, gesture, 1.0

        if suppress_yolo:
            return annotated, None, 0.0

        return annotated, yolo_label, yolo_conf

    @staticmethod
    def open_camera(camera_index: int = 0) -> cv2.VideoCapture:
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            raise RuntimeError("Could not open webcam.")
        return cap

from __future__ import annotations

import collections
import enum
import time
from pathlib import Path

import cv2
import numpy as np

try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
    _MP_AVAILABLE = True
except ImportError:
    _MP_AVAILABLE = False

try:
    import joblib
    from scipy.interpolate import interp1d
    _JOBLIB_AVAILABLE = True
except ImportError:
    _JOBLIB_AVAILABLE = False

_TRACK_LEN   = 64
_FIXED_LEN   = 30
_LABELS      = {0: "J", 1: "Z"}

_DEFAULT_LANDMARKER = "models/hand_landmarker.task"
_DEFAULT_SVM        = "models/gesture_svm.pkl"
_DEFAULT_SCALER     = "models/gesture_scaler.pkl"

_SPEED_THRESHOLD = 25   # px/frame — 이 이상이면 MOVING 진입
_TIMEOUT_SEC     = 1.5  # MOVING 최대 유지 시간


class State(enum.Enum):
    DETECTING = "DETECTING"
    MOVING    = "MOVING"
    JUDGING   = "JUDGING"
    COOLDOWN  = "COOLDOWN"


def _tip_speed(prev: tuple[int,int] | None, curr: tuple[int,int]) -> float:
    if prev is None:
        return 0.0
    return ((curr[0]-prev[0])**2 + (curr[1]-prev[1])**2) ** 0.5


def _preprocess(track_points) -> np.ndarray | None:
    pts = np.array(list(track_points), dtype=float)
    if len(pts) < 2:
        return None
    t     = np.linspace(0, 1, len(pts))
    t_new = np.linspace(0, 1, _FIXED_LEN)
    resampled = np.stack([
        interp1d(t, pts[:, 0])(t_new),
        interp1d(t, pts[:, 1])(t_new),
    ], axis=1)
    resampled -= resampled.min(axis=0)
    max_val = resampled.max()
    if max_val > 0:
        resampled /= max_val
    return resampled.flatten()


class GestureRecognizer:
    def __init__(
        self,
        svm_path: str = _DEFAULT_SVM,
        scaler_path: str = _DEFAULT_SCALER,
        landmarker_path: str = _DEFAULT_LANDMARKER,
        conf_threshold: float = 0.6,
        cooldown_sec: float = 2.0,
    ):
        self.conf_threshold = conf_threshold
        self.cooldown_sec   = cooldown_sec

        self.state: State = State.DETECTING
        self.track_points: collections.deque[tuple[int,int]] = collections.deque(maxlen=_TRACK_LEN)
        self._prev_index: tuple[int,int] | None = None
        self._prev_pinky: tuple[int,int] | None = None
        self._move_start: float = 0.0
        self._cooldown_start: float = 0.0

        self._model    = None
        self._scaler   = None
        self._landmarker = None

        if _JOBLIB_AVAILABLE:
            svm_file    = Path(svm_path)
            scaler_file = Path(scaler_path)
            if svm_file.exists() and scaler_file.exists():
                self._model  = joblib.load(str(svm_file))
                self._scaler = joblib.load(str(scaler_file))

        if _MP_AVAILABLE:
            lm_file = Path(landmarker_path)
            if lm_file.exists():
                base_opts = mp_python.BaseOptions(model_asset_path=str(lm_file))
                options   = mp_vision.HandLandmarkerOptions(
                    base_options=base_opts, num_hands=1
                )
                self._landmarker = mp_vision.HandLandmarker.create_from_options(options)

    @property
    def is_ready(self) -> bool:
        return self._model is not None and self._landmarker is not None

    def predict(self, frame: np.ndarray) -> tuple[np.ndarray, str | None, float]:
        annotated = frame.copy()

        if self._landmarker is None:
            return annotated, None, 0.0

        h, w     = frame.shape[:2]
        rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        detection = self._landmarker.detect(mp_image)

        now = time.time()

        # COOLDOWN 체크
        if self.state == State.COOLDOWN:
            if now - self._cooldown_start >= self.cooldown_sec:
                self.state = State.DETECTING
            self._draw_track(annotated)
            return annotated, None, 0.0

        if not detection.hand_landmarks:
            # 손 없으면 초기화
            self._prev_index = None
            self._prev_pinky = None
            if self.state == State.MOVING:
                self.state = State.JUDGING
            elif self.state == State.DETECTING:
                self.track_points.clear()
        else:
            lms = detection.hand_landmarks[0]

            index_pos = (int(lms[8].x * w),  int(lms[8].y * h))
            pinky_pos = (int(lms[20].x * w), int(lms[20].y * h))

            # 각 손가락 독립적으로 속도 계산
            index_speed = _tip_speed(self._prev_index, index_pos)
            pinky_speed = _tip_speed(self._prev_pinky, pinky_pos)
            speed = max(index_speed, pinky_speed)

            if self.state == State.DETECTING:
                # 첫 프레임(prev 없음)은 속도 0이므로 MOVING 진입 안 함
                if self._prev_index is not None and speed >= _SPEED_THRESHOLD:
                    self.track_points.clear()
                    self.state = State.MOVING
                    self._move_start = now
                    tip_pos = index_pos if index_speed >= pinky_speed else pinky_pos
                    self.track_points.append(tip_pos)
                self._prev_index = index_pos
                self._prev_pinky = pinky_pos

            elif self.state == State.MOVING:
                tip_pos = index_pos if index_speed >= pinky_speed else pinky_pos
                self.track_points.append(tip_pos)
                self._prev_index = index_pos
                self._prev_pinky = pinky_pos

                # 멈추거나 타임아웃 → JUDGING
                if speed < _SPEED_THRESHOLD or (now - self._move_start) >= _TIMEOUT_SEC:
                    self.state = State.JUDGING

        # JUDGING: 궤적으로 J/Z 판별
        if self.state == State.JUDGING:
            result, prob = self._judge()
            self.track_points.clear()
            self._prev_tip = None
            if result is not None:
                self._cooldown_start = now
                self.state = State.COOLDOWN
                self._draw_track(annotated)
                return annotated, result, prob
            else:
                self.state = State.DETECTING
                self._draw_track(annotated)
                return annotated, None, 0.0

        self._draw_track(annotated)
        return annotated, None, 0.0

    def _judge(self) -> tuple[str | None, float]:
        if self._model is None or len(self.track_points) < 15:
            return None, 0.0
        feat = _preprocess(self.track_points)
        if feat is None:
            return None, 0.0
        x    = self._scaler.transform([feat])
        pred = int(self._model.predict(x)[0])
        prob = float(self._model.predict_proba(x)[0].max())
        if prob < self.conf_threshold:
            return None, 0.0
        return _LABELS[pred], prob

    def _draw_track(self, frame: np.ndarray) -> None:
        pts = list(self.track_points)
        for i in range(1, len(pts)):
            cv2.line(frame, pts[i-1], pts[i], (255, 0, 0), 2)

    @property
    def current_state(self) -> str:
        return self.state.value

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

_TRACK_LEN      = 64
_FIXED_LEN      = 30
_LABELS         = {0: "Z", 1: "J"}
_SPEED_ENTER    = 5     # MOVING 진입 속도 (px/frame)
_STOP_FRAMES    = 15    # 멈춤 판단 프레임 수
_TIMEOUT_SEC    = 3.0   # MOVING 타임아웃
_COOLDOWN_SEC   = 2.0   # COOLDOWN 시간

_DEFAULT_LANDMARKER = "models/hand_landmarker.task"
_DEFAULT_SVM        = "models/gesture_svm.pkl"
_DEFAULT_SCALER     = "models/gesture_scaler.pkl"


class State(enum.Enum):
    DETECTING = "DETECTING"
    MOVING    = "MOVING"
    JUDGING   = "JUDGING"
    COOLDOWN  = "COOLDOWN"


def _dist(a, b):
    return ((a.x - b.x)**2 + (a.y - b.y)**2) ** 0.5


def _is_extended(lms, tip, pip, mcp, ratio=1.05):
    return _dist(lms[tip], lms[mcp]) > _dist(lms[pip], lms[mcp]) * ratio


def _finger_state(lms):
    """검지만 펴짐 → 'Z', 새끼만 펴짐 → 'J', 그 외 → None"""
    index  = _is_extended(lms, 8,  6,  5)
    middle = _is_extended(lms, 12, 10, 9)
    pinky  = _is_extended(lms, 20, 18, 17)

    # Z: 검지 펴짐 (새끼 상태 무관 — Z 제스처 시 새끼가 살짝 올라올 수 있음)
    if index:
        return "Z"
    # J: 새끼 펴짐 + 검지 안 펴짐
    if pinky and not index:
        return "J"
    return None


def _speed(prev, curr):
    if prev is None:
        return 0.0
    return ((curr[0] - prev[0])**2 + (curr[1] - prev[1])**2) ** 0.5


def _preprocess(pts):
    arr = np.array(list(pts), dtype=float)
    if len(arr) < 2:
        return None
    t     = np.linspace(0, 1, len(arr))
    t_new = np.linspace(0, 1, _FIXED_LEN)
    res = np.stack([
        interp1d(t, arr[:, 0])(t_new),
        interp1d(t, arr[:, 1])(t_new),
    ], axis=1)
    res -= res.min(axis=0)
    m = res.max()
    if m > 0:
        res /= m
    return res.flatten()


def _rule_judge(pts, mode):
    """모드 기반 판별 - 검지=Z, 새끼=J로 이미 구분되므로 이동량만 확인"""
    if len(pts) < 15:
        return None, 0.0

    # 전체 이동 거리 계산
    total_dist = sum(
        ((pts[i][0]-pts[i-1][0])**2 + (pts[i][1]-pts[i-1][1])**2)**0.5
        for i in range(1, len(pts))
    )

    # 충분히 움직였으면 모드에 따라 확정
    if total_dist > 80:
        if mode == "Z":
            return "Z", 0.9
        elif mode == "J":
            return "J", 0.9

    return None, 0.0


class GestureRecognizer:
    def __init__(
        self,
        svm_path: str = _DEFAULT_SVM,
        scaler_path: str = _DEFAULT_SCALER,
        landmarker_path: str = _DEFAULT_LANDMARKER,
        conf_threshold: float = 0.6,
        finger_ready_frames: int = 4,
    ):
        self.conf_threshold = conf_threshold
        self.state          = State.DETECTING
        self.track          = collections.deque(maxlen=_TRACK_LEN)
        self._mode          = None   # "J" or "Z"
        self._prev          = None   # 이전 프레임 손끝 좌표
        self._stop_cnt      = 0
        self._move_start    = 0.0
        self._cool_start    = 0.0
        self._finger_ready_cnt = 0          # 손가락 펴진 상태 유지 프레임 수
        self._FINGER_READY  = finger_ready_frames  # 이 프레임 수 이상 유지돼야 MOVING 진입
        self._last_fstate   = None   # 직전 비-None fstate 기억
        self._model         = None
        self._scaler        = None
        self._landmarker    = None

        if _JOBLIB_AVAILABLE:
            sp, sc = Path(svm_path), Path(scaler_path)
            if sp.exists() and sc.exists():
                self._model  = joblib.load(str(sp))
                self._scaler = joblib.load(str(sc))

        if _MP_AVAILABLE:
            lm = Path(landmarker_path)
            if lm.exists():
                opts = mp_vision.HandLandmarkerOptions(
                    base_options=mp_python.BaseOptions(model_asset_path=str(lm)),
                    num_hands=1,
                )
                self._landmarker = mp_vision.HandLandmarker.create_from_options(opts)

    @property
    def current_state(self):
        return self.state.value

    def predict(self, frame: np.ndarray):
        out = frame.copy()
        if self._landmarker is None:
            return out, None, 0.0

        h, w  = frame.shape[:2]
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res   = self._landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        now   = time.time()

        # ── COOLDOWN ──
        if self.state == State.COOLDOWN:
            if now - self._cool_start >= _COOLDOWN_SEC:
                self.state = State.DETECTING
                self.track.clear()
                self._prev = None
            self._draw(out)
            return out, None, 0.0

        # ── 손 없음 ──
        if not res.hand_landmarks:
            self._prev = None
            self._finger_ready_cnt = 0
            if self.state == State.MOVING:
                self.state = State.JUDGING
            else:
                self.track.clear()
            self._stop_cnt = 0

        else:
            lms  = res.hand_landmarks[0]
            fstate = _finger_state(lms)

            # 추적 손가락 좌표
            # DETECTING: 마지막으로 감지된 손가락 기준 유지 (fstate None일 때 튀는 것 방지)
            # MOVING 이후: self._mode 고정
            if self.state == State.DETECTING:
                if fstate is not None:
                    self._last_fstate = fstate
                tip_idx = 8 if self._last_fstate == "Z" else 20
            else:
                tip_idx = 8 if self._mode == "Z" else 20
            tip = lms[tip_idx]
            pos = (int(tip.x * w), int(tip.y * h))
            spd = _speed(self._prev, pos)

            if self.state == State.DETECTING:
                if fstate is not None:
                    self._finger_ready_cnt += 1
                else:
                    self._finger_ready_cnt = 0

                # 손가락 펴진 상태가 충분히 유지된 후 움직임 감지 시 MOVING 진입
                if (fstate is not None
                        and self._prev is not None
                        and self._finger_ready_cnt >= self._FINGER_READY
                        and spd >= _SPEED_ENTER):
                    self._mode       = fstate
                    self._stop_cnt   = 0
                    self._move_start = now
                    self.track.clear()
                    self.track.append(pos)
                    self.state = State.MOVING
                    self._finger_ready_cnt = 0
                self._prev = pos

            elif self.state == State.MOVING:
                # MOVING 중엔 손이 감지되는 한 궤적 수집 (fstate 조건 제거)
                self.track.append(pos)
                self._prev = pos

                if spd < 8:
                    self._stop_cnt += 1
                else:
                    self._stop_cnt = 0

                if self._stop_cnt >= _STOP_FRAMES or (now - self._move_start) >= _TIMEOUT_SEC:
                    self.state = State.JUDGING

        # ── JUDGING ──
        if self.state == State.JUDGING:
            label, prob = self._judge()
            self.track.clear()
            self._prev     = None
            self._stop_cnt = 0
            if label:
                self._cool_start = now
                self.state = State.COOLDOWN
                self._draw(out)
                return out, label, prob
            else:
                self.state = State.DETECTING

        self._draw(out)
        return out, None, 0.0

    def _judge(self):
        pts = list(self.track)
        if len(pts) < 15:
            return None, 0.0

        # 규칙 기반 먼저
        label, prob = _rule_judge(pts, self._mode)
        if label:
            return label, prob

        # SVM 시도
        if self._model is not None:
            feat = _preprocess(self.track)
            if feat is not None:
                x    = self._scaler.transform([feat])
                pred = int(self._model.predict(x)[0])
                prob = float(self._model.predict_proba(x)[0].max())
                if prob >= self.conf_threshold:
                    return _LABELS[pred], prob

        return None, 0.0

    def _draw(self, frame):
        pts = list(self.track)
        for i in range(1, len(pts)):
            cv2.line(frame, pts[i-1], pts[i], (255, 0, 0), 2)

from __future__ import annotations

from dataclasses import dataclass
from threading import Lock

import av
import cv2
import streamlit as st
from streamlit_shortcuts import shortcut_button
from streamlit_webrtc import WebRtcMode, webrtc_streamer

from postprocess import DebounceAccumulator
from recognizer import SignRecognizer


DEFAULT_MODEL_PATH = "models/asl6_yolov8n.pt"


@dataclass
class RuntimeConfig:
    model_path: str
    conf_threshold: float
    stable_frames: int
    cooldown_frames: int


class VideoProcessor:
    def __init__(self, config: RuntimeConfig):
        self.config = config
        self.lock = Lock()
        self.recognizer = SignRecognizer(
            model_path=config.model_path,
            conf_threshold=config.conf_threshold,
        )
        self.accumulator = DebounceAccumulator(
            min_stable_frames=config.stable_frames,
            cooldown_frames=config.cooldown_frames,
        )
        self.last_label: str | None = None
        self.last_score: float = 0.0

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        image = frame.to_ndarray(format="bgr24")
        image = cv2.flip(image, 1)

        annotated, label, score = self.recognizer.predict(
            image,
            draw_labels=True,
            draw_conf=False,
        )
        commit = self.accumulator.update(label)

        with self.lock:
            self.last_label = label
            self.last_score = score
            self.last_text = commit.text

        return av.VideoFrame.from_ndarray(annotated, format="bgr24")

    def reset_text(self) -> None:
        with self.lock:
            self.accumulator.clear()
            self.last_label = None
            self.last_score = 0.0
            self.last_text = ""

    def append_space(self) -> None:
        with self.lock:
            self.accumulator.append_space()
            self.last_text = self.accumulator.text

    def backspace(self) -> None:
        with self.lock:
            self.accumulator.backspace()
            self.last_text = self.accumulator.text

    def get_state(self) -> tuple[str | None, float, str]:
        with self.lock:
            return self.last_label, self.last_score, getattr(self, "last_text", "")

def render_output_box(placeholder, text: str) -> None:
    safe_text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    placeholder.markdown(
        f"""
        <div style="
            border: 1px solid #444;
            border-radius: 10px;
            padding: 16px;
            min-height: 260px;
            background-color: #111827;
            color: #f9fafb;
            font-size: 1.25rem;
            line-height: 1.7;
            white-space: pre-wrap;
            word-break: break-word;
        ">{safe_text}</div>
        """,
        unsafe_allow_html=True,
    )


def build_config() -> RuntimeConfig:
    with st.sidebar:
        st.subheader("설정")
        model_path = st.text_input("모델 경로", DEFAULT_MODEL_PATH)
        conf_threshold = st.slider("신뢰도 임계값", 0.1, 0.95, 0.5, 0.05)
        stable_frames = st.slider("글자 확정 프레임 수", 1, 20, 8)
        cooldown_frames = st.slider("중복 입력 방지 프레임 수", 0, 30, 10)

    return RuntimeConfig(
        model_path=model_path,
        conf_threshold=conf_threshold,
        stable_frames=stable_frames,
        cooldown_frames=cooldown_frames,
    )


def main() -> None:
    st.set_page_config(page_title="ASL Recognizer", layout="wide")
    st.title("FingerType")
    st.caption("실시간 수어 알파벳 인식 및 텍스트 조합 시스템")

    config = build_config()

    try:
        SignRecognizer(model_path=config.model_path, conf_threshold=config.conf_threshold)
    except Exception as exc:
        st.error(str(exc))
        st.stop()

    st.info(
        "1. 아래 `START`를 눌러 카메라를 시작하세요.\n"
        "2. 카메라 권한을 허용하세요.\n"
        "3. 손동작을 웹캠 앞에 보여주세요.\n"
        "4. 인식 결과는 오른쪽 패널, 누적 텍스트는 카메라 아래에 표시됩니다."
    )

    camera_col, info_col = st.columns([3, 2])
    with camera_col:
        ctx = webrtc_streamer(
            key="asl-recognizer",
            mode=WebRtcMode.SENDRECV,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
            video_processor_factory=lambda: VideoProcessor(config),
        )
        camera_status_placeholder = st.empty()
        camera_letter_placeholder = st.empty()
        camera_conf_placeholder = st.empty()

    with info_col:
        st.subheader("누적 텍스트")
        text_box_placeholder = st.empty()
        reset_clicked = st.button("전체 초기화", use_container_width=True)
        space_clicked = shortcut_button(
            "띄어쓰기",
            "space",
            key="space_button",
            use_container_width=True,
        )
        delete_clicked = shortcut_button(
            "한 글자 지우기",
            "backspace",
            key="delete_button",
            use_container_width=True,
        )
        st.caption("설정을 바꿨다면 STOP 후 다시 START 하세요.")
        st.caption("단축키: Space = 띄어쓰기, Backspace = 한 글자 지우기")

    if reset_clicked and ctx.video_processor:
        ctx.video_processor.reset_text()
        st.success("출력 텍스트를 초기화했습니다.")
    if space_clicked and ctx.video_processor:
        ctx.video_processor.append_space()
    if delete_clicked and ctx.video_processor:
        ctx.video_processor.backspace()

    @st.fragment(run_every=0.2)
    def render_live_status() -> None:
        if ctx.state.playing and ctx.video_processor:
            label, score, output_text = ctx.video_processor.get_state()
            camera_status_placeholder.success("인식 중")
            camera_letter_placeholder.metric("현재 인식 글자", label if label else "-")
            camera_conf_placeholder.progress(
                int(max(0.0, min(score, 1.0)) * 100),
                text=f"신뢰도: {score:.2f}" if label else "신뢰도: -",
            )
            render_output_box(text_box_placeholder, output_text)
        else:
            camera_status_placeholder.info("카메라 시작 대기 중")
            camera_letter_placeholder.metric("현재 인식 글자", "-")
            camera_conf_placeholder.progress(0, text="신뢰도: -")
            render_output_box(text_box_placeholder, "")

    render_live_status()

if __name__ == "__main__":
    main()

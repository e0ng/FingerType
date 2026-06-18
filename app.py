from __future__ import annotations

from dataclasses import dataclass
import tempfile
from threading import Lock

import av
import cv2
import streamlit as st
from streamlit_shortcuts import shortcut_button
from streamlit_webrtc import WebRtcMode, webrtc_streamer

from postprocess import DebounceAccumulator
from recognizer import SignRecognizer
from gesture_recognizer import GestureRecognizer


DEFAULT_MODEL_PATH = "models/asl6_yolov8n.pt"


@dataclass
class RuntimeConfig:
    model_path: str
    conf_threshold: float
    stable_frames: int
    cooldown_frames: int


def run_recognition_step(
    image,
    recognizer: SignRecognizer,
    gesture_recognizer: GestureRecognizer,
    accumulator: DebounceAccumulator,
) -> tuple[object, str | None, float, str, str | None, float, str]:
    annotated, gesture, gesture_score = gesture_recognizer.predict(image)
    state = gesture_recognizer.current_state

    if state == "DETECTING":
        yolo_annotated, label, score = recognizer.predict(
            annotated,
            draw_labels=True,
            draw_conf=False,
        )
        annotated = yolo_annotated
        if label in ("J", "Z"):
            label, score = None, 0.0
        commit = accumulator.update(label)
    else:
        label, score = None, 0.0
        if gesture is not None:
            commit = accumulator.force_commit(gesture)
        else:
            commit = accumulator.update(None)

    display_label = gesture if gesture else label
    display_score = gesture_score if gesture else score
    return (
        annotated,
        display_label,
        display_score,
        commit.text,
        gesture,
        gesture_score,
        state,
    )


class VideoProcessor:
    def __init__(self, config: RuntimeConfig):
        self.config = config
        self.lock = Lock()
        self.recognizer = SignRecognizer(
            model_path=config.model_path,
            conf_threshold=config.conf_threshold,
        )
        self.gesture_recognizer = GestureRecognizer()
        self.accumulator = DebounceAccumulator(
            min_stable_frames=config.stable_frames,
            cooldown_frames=config.cooldown_frames,
        )
        self.last_label: str | None = None
        self.last_score: float = 0.0
        self.last_gesture: str | None = None
        self.last_gesture_score: float = 0.0

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        image = frame.to_ndarray(format="bgr24")
        image = cv2.flip(image, 1)

        (
            annotated,
            label,
            score,
            output_text,
            gesture,
            gesture_score,
            state,
        ) = run_recognition_step(
            image,
            self.recognizer,
            self.gesture_recognizer,
            self.accumulator,
        )

        with self.lock:
            self.last_label  = label
            self.last_score  = score
            self.last_text   = output_text
            self.last_gesture = gesture
            self.last_gesture_score = gesture_score
            self.last_state  = state

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

    def get_state(self) -> tuple[str | None, float, str, str | None, float, str]:
        with self.lock:
            return (
                self.last_label,
                self.last_score,
                getattr(self, "last_text", ""),
                self.last_gesture,
                self.last_gesture_score,
                getattr(self, "last_state", "DETECTING"),
            )

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


def render_uploaded_video_tool(config: RuntimeConfig) -> None:
    st.divider()
    st.subheader("동영상 업로드 인식")

    uploaded_file = st.file_uploader(
        "인식할 동영상 파일",
        type=["mp4", "mov", "avi", "mkv"],
    )
    option_col1, option_col2 = st.columns(2)
    with option_col1:
        frame_stride = st.slider(
            "분석 프레임 간격",
            1,
            10,
            2,
            help="값이 클수록 빠르게 처리하지만 짧은 동작을 놓칠 수 있습니다.",
        )
    with option_col2:
        mirror_video = st.checkbox(
            "좌우 반전",
            value=True,
            help="웹캠 화면과 같은 방향으로 맞춰 분석합니다.",
        )

    if uploaded_file is None:
        return

    st.video(uploaded_file)
    if not st.button("업로드 영상 인식 시작", use_container_width=True):
        return

    recognizer = SignRecognizer(
        model_path=config.model_path,
        conf_threshold=config.conf_threshold,
    )
    gesture_recognizer = GestureRecognizer()
    accumulator = DebounceAccumulator(
        min_stable_frames=config.stable_frames,
        cooldown_frames=config.cooldown_frames,
    )

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as input_file:
        input_file.write(uploaded_file.getbuffer())
        input_path = input_file.name

    output_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    output_path = output_file.name
    output_file.close()

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        st.error("동영상 파일을 열 수 없습니다.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps / frame_stride if frame_stride > 1 else fps,
        (width, height),
    )

    progress = st.progress(0, text="동영상 인식 준비 중")
    preview_placeholder = st.empty()
    result_placeholder = st.empty()
    last_label: str | None = None
    last_score = 0.0
    last_state = "DETECTING"
    last_gesture: str | None = None
    last_gesture_score = 0.0
    output_text = ""
    processed_frames = 0
    frame_index = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_index % frame_stride != 0:
            frame_index += 1
            continue

        if mirror_video:
            frame = cv2.flip(frame, 1)

        (
            annotated,
            last_label,
            last_score,
            output_text,
            last_gesture,
            last_gesture_score,
            last_state,
        ) = run_recognition_step(
            frame,
            recognizer,
            gesture_recognizer,
            accumulator,
        )
        writer.write(annotated)
        processed_frames += 1

        if processed_frames % 10 == 0:
            preview_placeholder.image(
                cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                channels="RGB",
                caption="처리 중인 프레임",
            )

        if total_frames:
            progress.progress(
                min(frame_index / total_frames, 1.0),
                text=f"동영상 인식 중: {frame_index}/{total_frames} frames",
            )
        else:
            progress.progress(0, text=f"동영상 인식 중: {processed_frames} frames")

        frame_index += 1

    cap.release()
    writer.release()
    progress.progress(1.0, text="동영상 인식 완료")

    result_placeholder.markdown(
        f"""
        **최종 인식 글자:** `{last_label if last_label else "-"}`

        **신뢰도:** `{last_score:.2f}`

        **J/Z 상태:** `{last_state}`

        **J/Z 인식:** `{last_gesture if last_gesture else "-"}`

        **J/Z 신뢰도:** `{last_gesture_score:.2f}`
        """
    )
    render_output_box(st.empty(), output_text)

    with open(output_path, "rb") as video_file:
        st.video(video_file.read())


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

    # J/Z 인식 결과 상단 표시
    jz_col1, jz_col2 = st.columns(2)
    jz_label_ph = jz_col1.empty()
    jz_conf_ph  = jz_col2.empty()

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
            label, score, output_text, gesture, gesture_score, state = ctx.video_processor.get_state()
            # 상태 표시
            state_color = {"DETECTING": "🟢", "MOVING": "🟡", "JUDGING": "🔵", "COOLDOWN": "🔴"}
            jz_label_ph.metric("J/Z 인식", gesture if gesture else "-")
            jz_conf_ph.progress(
                int(gesture_score * 100),
                text=f"상태: {state_color.get(state,'')} {state}",
            )
            camera_status_placeholder.success("인식 중")
            camera_letter_placeholder.metric("현재 인식 글자", label if label else "-")
            camera_conf_placeholder.progress(
                int(max(0.0, min(score, 1.0)) * 100),
                text=f"신뢰도: {score:.2f}" if label else "신뢰도: -",
            )
            render_output_box(text_box_placeholder, output_text)
        else:
            jz_label_ph.metric("J/Z 인식", "-")
            jz_conf_ph.progress(0, text="상태: -")
            camera_status_placeholder.info("카메라 시작 대기 중")
            camera_letter_placeholder.metric("현재 인식 글자", "-")
            camera_conf_placeholder.progress(0, text="신뢰도: -")
            render_output_box(text_box_placeholder, "")

    render_live_status()
    render_uploaded_video_tool(config)

if __name__ == "__main__":
    main()

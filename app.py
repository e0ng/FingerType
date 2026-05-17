from __future__ import annotations

import time
from pathlib import Path

import cv2
import streamlit as st

from postprocess import DebounceAccumulator
from recognizer import SignRecognizer


DEFAULT_MODEL_PATH = "models/asl_yolov8n.pt"


def init_state() -> None:
    if "running" not in st.session_state:
        st.session_state.running = False
    if "accumulator" not in st.session_state:
        st.session_state.accumulator = DebounceAccumulator()


def main() -> None:
    st.set_page_config(page_title="ASL Recognizer", layout="wide")
    init_state()

    st.title("ASL Finger Spelling Recognizer")
    st.caption("Start with static A-Z recognition first, then add J/Z tracking later.")

    with st.sidebar:
        model_path = st.text_input("Model path", DEFAULT_MODEL_PATH)
        conf_threshold = st.slider("Confidence threshold", 0.1, 0.95, 0.5, 0.05)
        stable_frames = st.slider("Stable frames", 3, 20, 8)
        cooldown_frames = st.slider("Cooldown frames", 3, 30, 10)

        if st.button("Reset text"):
            st.session_state.accumulator.clear()

        start_clicked = st.button("Start recognition", type="primary")
        stop_clicked = st.button("Stop")

    if start_clicked:
        st.session_state.running = True
        st.session_state.accumulator = DebounceAccumulator(
            min_stable_frames=stable_frames,
            cooldown_frames=cooldown_frames,
        )
    if stop_clicked:
        st.session_state.running = False

    frame_col, text_col = st.columns([3, 2])
    frame_placeholder = frame_col.empty()
    label_placeholder = text_col.empty()
    text_placeholder = text_col.empty()

    if not st.session_state.running:
        text_placeholder.info(
            "1. Train or export your YOLO weight file.\n"
            "2. Put it at `models/asl_yolov8n.pt`.\n"
            "3. Click Start recognition."
        )
        return

    try:
        recognizer = SignRecognizer(model_path=model_path, conf_threshold=conf_threshold)
    except Exception as exc:
        st.session_state.running = False
        st.error(str(exc))
        return

    cap = SignRecognizer.open_camera()
    try:
        while st.session_state.running:
            ok, frame = cap.read()
            if not ok:
                st.warning("Failed to read webcam frame.")
                break

            frame = cv2.flip(frame, 1)
            annotated, label, score = recognizer.predict(frame)
            commit = st.session_state.accumulator.update(label)

            frame_placeholder.image(
                cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                channels="RGB",
                use_container_width=True,
            )
            label_placeholder.metric(
                "Current prediction",
                label if label is not None else "-",
                f"{score:.2f}" if label is not None else None,
            )
            text_placeholder.markdown(
                f"### Output text\n`{commit.text if commit.text else ''}`"
            )
            time.sleep(0.01)
    finally:
        cap.release()


if __name__ == "__main__":
    main()

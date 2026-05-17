from __future__ import annotations

import argparse

import cv2

from postprocess import DebounceAccumulator
from recognizer import SignRecognizer


def draw_overlay(frame, label: str | None, score: float, text: str) -> None:
    label_text = f"Label: {label or '-'}"
    score_text = f"Conf: {score:.2f}" if label is not None else "Conf: -"
    text_output = f"Text: {text}"

    cv2.rectangle(frame, (10, 10), (620, 110), (0, 0, 0), -1)
    cv2.putText(frame, label_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.putText(frame, score_text, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, text_output, (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/asl_yolov8n.pt")
    parser.add_argument("--conf", type=float, default=0.5)
    parser.add_argument("--stable-frames", type=int, default=8)
    parser.add_argument("--cooldown-frames", type=int, default=10)
    parser.add_argument("--camera-index", type=int, default=0)
    args = parser.parse_args()

    recognizer = SignRecognizer(model_path=args.model, conf_threshold=args.conf)
    accumulator = DebounceAccumulator(
        min_stable_frames=args.stable_frames,
        cooldown_frames=args.cooldown_frames,
    )

    cap = SignRecognizer.open_camera(camera_index=args.camera_index)
    print("Press 'q' to quit, 'c' to clear text.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Failed to read webcam frame.")
                break

            frame = cv2.flip(frame, 1)
            annotated, label, score = recognizer.predict(frame)
            commit = accumulator.update(label)
            draw_overlay(annotated, label, score, commit.text)

            cv2.imshow("ASL Recognition", annotated)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("c"):
                accumulator.clear()
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

# FingerType

# ASL Recognition Starter

This is a minimal starter for real-time ASL finger spelling recognition.

## What it does

- Loads a trained YOLO model
- Reads webcam frames
- Predicts the current sign label
- Accumulates stable predictions into text with debounce logic
- Shows the annotated frame and output text in Streamlit

## Project structure

- `app.py`: Streamlit app entry point
- `recognizer.py`: YOLO inference wrapper
- `postprocess.py`: Stable-frame and debounce logic

## Quick start

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Download the trained model separately and put it at:

```text
models/asl5_yolov8n.pt
```

For `J` and `Z` dynamic gesture tracking, also download the MediaPipe hand
landmarker model:

```bash
curl -L -o models/hand_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
```

3. Run the app:

```bash
streamlit run app.py
```

## Recommended build order

1. Train static A-Z recognition first
2. Verify live webcam inference works
3. Tune stable-frame and cooldown thresholds
4. Add tracking for `J` and `Z` as a second step

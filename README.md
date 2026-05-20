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

2. Put your trained model at:

```text
models/asl_yolov8n.pt
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

# FingerType

Real-time ASL finger spelling recognition with Streamlit, YOLO, and optional
MediaPipe-based dynamic gesture tracking for `J` and `Z`.

## What it does

- Loads a trained YOLO model
- Reads webcam frames
- Predicts the current sign label
- Accumulates stable predictions into text with debounce logic
- Shows the annotated frame and output text in Streamlit
- Tracks index-finger motion for dynamic `J` and `Z` gestures when the
  MediaPipe landmarker model is available

## Project structure

- `app.py`: Streamlit app entry point
- `recognizer.py`: YOLO inference wrapper
- `postprocess.py`: Stable-frame and debounce logic
- `webcam_infer.py`: Direct OpenCV webcam inference script
- `models/`: Local model files directory

## Quick start

### 1. Install dependencies

Install dependencies in the project root:

```bash
pip3 install -r requirements.txt
```

### 2. Prepare model files

The retrained YOLO weight file is not committed to this repository. Download the
final model separately and save it here:

```text
models/asl5_yolov8n.pt
```

For `J` and `Z` dynamic gesture tracking, download the MediaPipe hand
landmarker model:

```bash
curl -L -o models/hand_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
```

### 3. Run the Streamlit app

```bash
streamlit run app.py
```

Then:

1. Check that the model path is `models/asl5_yolov8n.pt`
2. Click `START`
3. Allow webcam access
4. Show a hand sign in front of the camera

The app displays:

- Current predicted letter
- Confidence score
- Accumulated output text

### 4. Optional direct webcam test

```bash
python3 webcam_infer.py --model models/asl5_yolov8n.pt
```

Controls:

- `q`: Quit
- `c`: Clear accumulated text

## Recommended build order

1. Train static A-Z recognition first
2. Verify live webcam inference works
3. Tune stable-frame and cooldown thresholds
4. Add tracking for `J` and `Z` as a second step

## Notes

- The retrained model is expected at `models/asl5_yolov8n.pt`.
- `J` and `Z` dynamic gesture tracking requires `models/hand_landmarker.task`.

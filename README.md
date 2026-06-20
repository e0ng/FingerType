# FingerType

FingerType is a Streamlit web app that recognizes ASL finger spelling from a
live webcam stream or an uploaded video. Static letters are detected with a
YOLOv8n object detection model, and dynamic `J` and `Z` gestures are handled
with MediaPipe hand landmarks and trajectory-based logic.

## Features

- Static ASL alphabet detection with YOLOv8n
- Dynamic `J` and `Z` gesture recognition with MediaPipe Hand Landmarker
- Live webcam recognition with `streamlit-webrtc`
- Uploaded video recognition with OpenCV frame processing
- Confidence threshold, stable-frame, and cooldown controls
- Accumulated text output with debounce-based postprocessing

## Project Structure

```text
.
├── app.py                  # Streamlit UI and input flow
├── recognizer.py           # YOLO static letter inference
├── gesture_recognizer.py   # J/Z dynamic gesture state machine
├── postprocess.py          # Debounce and cooldown logic
├── webcam_infer.py         # Optional direct webcam inference script
├── requirements.txt        # Runtime dependencies
└── models/                 # Local model files, not all weights are committed
```

## Environment

The project was developed with Python 3 and the packages listed in
`requirements.txt`.

Install dependencies from the project root:

```bash
pip install -r requirements.txt
```

If `streamlit run app.py` reports a missing package, reinstall dependencies in
the same Python environment that runs Streamlit.

## Model Files

Large or frequently updated model weights are not committed to this repository.
Download the final YOLO model and dynamic gesture classifier files from the
shared Google Drive folder, then place them in the `models` directory.

```text
models/asl6_yolov8n.pt
models/gesture_scaler.pkl
models/gesture_svm.pkl
```

Model file roles:

| File | Role | Git status |
|---|---|---|
| `models/asl_yolov8n.pt` | Baseline or reference model | Committed |
| `models/asl6_yolov8n.pt` | Final service model | Download separately |
| `models/gesture_scaler.pkl` | Feature scaler for J/Z trajectory classification | Download separately |
| `models/gesture_svm.pkl` | SVM classifier for J/Z trajectory classification | Download separately |
| `models/hand_landmarker.task` | MediaPipe hand landmark model for J/Z | Download separately if missing |

Model download folder:

https://drive.google.com/drive/folders/1IUOq2dYEz1PIXTiPFRucBRym74Ry0ffm?usp=share_link

MediaPipe hand landmarker download:

```bash
mkdir -p models
curl -L -o models/hand_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
```

After downloading, the model directory should include:

```text
models/asl6_yolov8n.pt
models/gesture_scaler.pkl
models/gesture_svm.pkl
models/hand_landmarker.task
```

## Run

Start the Streamlit app:

```bash
streamlit run app.py
```

In the sidebar, check that the model path is:

```text
models/asl6_yolov8n.pt
```

Then choose one of the app tabs:

- `실시간 인식`: Recognize letters from a live webcam stream
- `동영상 업로드`: Upload a video file and generate an annotated result video

## App Screenshot

Add a screenshot of the running app before final submission.

```text
Insert a screenshot showing the Streamlit app, webcam or upload tab, and recognition status panel.
```

## Data Pipeline

1. Collect and label ASL alphabet images in Roboflow.
2. Use bounding box labels for static A to Y hand shapes.
3. Exclude `J` and `Z` from static YOLO output because they require motion.
4. Add hard samples for letters that were repeatedly confused in webcam tests.
5. Create Roboflow dataset versions with different augmentation and sample
   selection strategies.
6. Train YOLOv8n models and compare mAP50, mAP50-95, latency, and confusion
   matrix results.
7. Use the final `asl6_yolov8n.pt` model in the Streamlit app.

The final static model was selected as the service model because it was trained
with additional hard samples for remaining failure classes while keeping
validation performance close to the baseline.

## Training Versions

| Version | Model file | Main change | Notes |
|---|---|---|---|
| v1 | `models/asl_yolov8n.pt` | Initial baseline | Reference model |
| v2 | `models/asl2_yolov8n.pt` | Roboflow augmentation for confused letters | Improved some classes but introduced new confusion |
| v3 | `models/asl3_yolov8n.pt` | Added custom right-hand images | High validation mAP but unstable webcam behavior |
| v4 | `models/asl4_yolov8n.pt` | Reduced unstable custom samples | More conservative dataset version |
| v5 | `models/asl5_yolov8n.pt` | Added remaining right-hand hard samples | Previous final candidate |
| v6 | `models/asl6_yolov8n.pt` | Added remaining failure-class samples | Current app default model |

## Team Roles

| Member | Main responsibility |
|---|---|
| 권예원 | Static YOLO model retraining, confusion analysis, Streamlit UI, uploaded video recognition |
| 한정현 | Dynamic J/Z gesture recognition, MediaPipe landmark processing, trajectory classification |

## Optional Direct Webcam Test

```bash
python webcam_infer.py --model models/asl6_yolov8n.pt
```

Controls:

- `q`: Quit
- `c`: Clear accumulated text

## Reproducibility Checklist

- Install dependencies with `pip install -r requirements.txt`.
- Download `models/asl6_yolov8n.pt`, `models/gesture_scaler.pkl`, and
  `models/gesture_svm.pkl` to the expected paths.
- Download `models/hand_landmarker.task` if dynamic J/Z recognition is needed.
- Run `streamlit run app.py` from the project root.
- Confirm there are no local absolute paths such as `C:\Users\...` or
  `/Users/...` in tracked source files.

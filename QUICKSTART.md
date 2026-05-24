# FingerType Quick Start

This repository includes the prototype code for real-time ASL finger spelling recognition.

## Included

- Streamlit app for live inference
- OpenCV webcam test script
- App code for loading a separately downloaded trained model

## Environment

Install dependencies in the project root:

```bash
pip3 install -r requirements.txt
```

## Model files

The retrained YOLO weight file is not committed to this repository. Download the
final model separately and save it here:

```text
models/asl5_yolov8n.pt
```

For `J` and `Z` dynamic gesture tracking, download the MediaPipe hand landmarker
model:

```bash
curl -L -o models/hand_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
```

## Option 1: Streamlit app

Run:

```bash
streamlit run app.py
```

Then:

1. Check that the model path is `models/asl5_yolov8n.pt`
2. Click `Start recognition`
3. Allow webcam access
4. Show a hand sign in front of the camera

The app displays:

- current predicted letter
- confidence score
- accumulated output text

## Option 2: Direct webcam test

Run:

```bash
python3 webcam_infer.py --model models/asl5_yolov8n.pt
```

Controls:

- `q`: quit
- `c`: clear accumulated text

## Notes

- The retrained model is expected at `models/asl5_yolov8n.pt`.
- `J` and `Z` dynamic gesture tracking requires `models/hand_landmarker.task`.

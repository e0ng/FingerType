# FingerType Quick Start

This repository includes the prototype code for real-time ASL finger spelling recognition.

## Included

- Streamlit app for live inference
- OpenCV webcam test script
- Trained baseline model at `models/asl_yolov8n.pt`

## Environment

Install dependencies in the project root:

```bash
pip3 install -r requirements.txt
```

## Option 1: Streamlit app

Run:

```bash
streamlit run app.py
```

Then:

1. Check that the model path is `models/asl_yolov8n.pt`
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
python3 webcam_infer.py --model models/asl_yolov8n.pt
```

Controls:

- `q`: quit
- `c`: clear accumulated text

## Notes

- This baseline works best for static A-Z recognition.
- `J` and `Z` are dynamic gestures, so additional tracking logic is still needed for robust recognition.
- The included model was trained from the Roboflow ASL dataset baseline.

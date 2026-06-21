# FingerType

FingerType은 실시간 웹캠 또는 업로드된 동영상에서 ASL 알파벳을 인식하고
누적 텍스트로 변환하는 Streamlit 웹 앱입니다. 정적 알파벳은 YOLOv8n 객체
탐지 모델로 처리하고, 동적 제스처인 `J`와 `Z`는 MediaPipe 손 랜드마크와
궤적 기반 로직을 이용해 판별합니다.
<img width="1710" height="1107" alt="스크린샷 2026-06-22 오전 6 30 31" src="https://github.com/user-attachments/assets/2d832f1b-d378-4143-a439-9eecb070694a" />

## 주요 기능

- YOLOv8n 기반 정적 ASL 알파벳 탐지
- MediaPipe Hand Landmarker 기반 `J`, `Z` 동적 제스처 인식
- `streamlit-webrtc`를 이용한 실시간 웹캠 인식
- OpenCV 기반 동영상 업로드 인식
- 신뢰도 임계값, 글자 확정 프레임 수, 중복 입력 방지 프레임 수 조정
- debounce 후처리를 통한 누적 텍스트 출력

## 프로젝트 구조

```text
.
├── app.py                  # Streamlit UI와 입력 흐름
├── recognizer.py           # YOLO 정적 알파벳 추론
├── gesture_recognizer.py   # J/Z 동적 제스처 상태 머신
├── postprocess.py          # debounce와 cooldown 후처리
├── requirements.txt        # 실행 의존성
└── models/                 # 로컬 모델 파일 디렉터리
```

## 실행 환경

프로젝트는 Python 3와 `requirements.txt`에 정리된 패키지를 기준으로
실행합니다.

프로젝트 루트에서 의존성을 설치합니다.

```bash
pip install -r requirements.txt
```

`streamlit run app.py` 실행 중 패키지 누락 오류가 발생하면, Streamlit을
실행하는 것과 동일한 Python 환경에서 의존성을 다시 설치하세요.

## 모델 파일

최종 YOLO 가중치 파일은 저장소에 포함하지 않습니다. 아래 Google Drive
폴더에서 최종 모델을 다운로드한 뒤 `models` 디렉터리에 넣어야 합니다.

```text
models/asl6_yolov8n.pt
```

모델 파일 역할은 다음과 같습니다.

| 파일 | 역할 | Git 상태 |
|---|---|---|
| `models/asl6_yolov8n.pt` | 최종 서비스 모델 | 별도 다운로드 |
| `models/gesture_scaler.pkl` | J/Z 궤적 분류용 feature scaler | 저장소 포함 |
| `models/gesture_svm.pkl` | J/Z 궤적 분류용 SVM 모델 | 저장소 포함 |
| `models/hand_landmarker.task` | J/Z용 MediaPipe 손 랜드마크 모델 | 저장소 포함 |

모델 다운로드 폴더:

https://drive.google.com/drive/folders/1IUOq2dYEz1PIXTiPFRucBRym74Ry0ffm?usp=share_link

다운로드 후 `models` 디렉터리는 아래 파일들을 포함해야 합니다.

```text
models/asl6_yolov8n.pt
models/gesture_scaler.pkl
models/gesture_svm.pkl
models/hand_landmarker.task
```

## 실행 방법

Streamlit 앱을 실행합니다.

```bash
streamlit run app.py
```

사이드바의 모델 경로가 아래 값인지 확인합니다.

```text
models/asl6_yolov8n.pt
```

앱에서는 두 가지 입력 방식을 선택할 수 있습니다.

- `실시간 인식`: 웹캠 스트림에서 알파벳 인식
- `동영상 업로드`: 업로드한 동영상 파일을 분석하고 결과 영상 생성

### 사이드바 설정

사이드바에서는 인식 민감도와 글자 확정 방식을 조정할 수 있습니다.

- `모델 경로`: 사용할 YOLO 모델 파일 경로입니다. 기본값은 `models/asl6_yolov8n.pt`입니다.
- `신뢰도 임계값`: 이 값보다 낮은 YOLO 예측은 글자 후보에서 제외합니다.
- `글자 확정 프레임 수`: 같은 글자가 몇 프레임 이상 유지되어야 텍스트로 확정할지 정합니다.
- `중복 입력 방지 프레임 수`: 한 번 입력된 글자가 연속으로 중복 입력되는 것을 막는 대기 프레임 수입니다.
- `동적 인식 진입 프레임 수`: J/Z 동적 인식을 시작하기 전 손가락이 펴진 상태로 유지되어야 하는 프레임 수입니다.
- `동적 인식 진입 속도`: 손끝 이동 속도가 이 값 이상이면 J/Z 동적 제스처 후보로 판단합니다.

### 실시간 인식 탭

`START` 버튼을 눌러 카메라를 시작한 뒤 브라우저 카메라 권한을 허용합니다.
인식 상태 패널에서는 현재 인식 글자, 신뢰도, J/Z 인식 상태, 누적 텍스트를 확인할 수 있습니다.

누적 텍스트 영역에서는 다음 버튼을 사용할 수 있습니다.

- `전체 초기화`: 누적 텍스트를 모두 지웁니다.
- `띄어쓰기`: 누적 텍스트에 공백을 추가합니다.
- `한 글자 지우기`: 마지막 글자 하나를 삭제합니다.

### 동영상 업로드 탭

`mp4`, `mov`, `avi`, `mkv` 형식의 동영상을 업로드할 수 있습니다.
업로드 후 `업로드 영상 인식 시작` 버튼을 누르면 프레임 단위로 분석을 진행합니다.

- `분석 프레임 간격`: 값이 클수록 빠르게 처리하지만 짧은 동작을 놓칠 수 있습니다.
- `좌우 반전`: 웹캠 화면과 같은 방향으로 맞춰 분석할 때 사용합니다.

처리 중에는 진행률과 preview frame을 표시하고, 완료 후에는 원본 영상이 표시되던 위치에 결과 영상을 보여줍니다.

## 데이터 파이프라인

1. Roboflow에서 ASL 알파벳 이미지를 수집하고 라벨링합니다.
2. 정적 A부터 Y 손모양에는 Bounding Box 라벨을 사용합니다.
3. `J`와 `Z`는 움직임이 필요한 글자이므로 정적 YOLO 결과에서 제외합니다.
4. 웹캠 테스트에서 반복적으로 혼동된 글자에 대해 직접 촬영 데이터를 추가합니다.
5. Roboflow 데이터셋 버전을 생성하며 augmentation과 샘플 구성을 달리합니다.
6. YOLOv8n 모델을 학습하고 mAP50, mAP50-95, latency, Confusion Matrix를 비교합니다.
7. 최종 서비스 모델로 `asl6_yolov8n.pt`를 사용합니다.

## 학습 스크립트

Colab에서 최종 정적 ASL 모델을 학습할 때 사용한 흐름은
`training/asl_train.ipynb`에 정리되어 있습니다. 
Roboflow API key는 코드에 직접 넣지 않고 
Colab Secrets에 `ROBOFLOW_API_KEY`로 등록해 사용합니다.

```bash
pip install roboflow
```


## 팀원 역할

| 팀원 | 주요 담당 |
|---|---|
| 권예원 | 정적 YOLO 모델 재학습, 직접 촬영 데이터 보강, Streamlit UI |
| 한정현 | 동적 J/Z 제스처 인식, MediaPipe 랜드마크 처리, 상태 머신 및 궤적 분류 |

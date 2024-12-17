# Pedestrian-based Camera Calibration Toolbox

Pedestrian-based Camera Calibration Toolbox는 보행자의 관절 정보를 활용하여 카메라를 자동으로 보정하는 Python 기반 도구입니다. 보정 과정에서 YOLO 기반 포즈 추적과 RANSAC 기반 보정 기법을 사용하며, 간단히 동영상을 입력하면 카메라의 내·외부 파라미터를 추출할 수 있습니다.

---

## 설치 및 필요 라이브러리

이 프로젝트를 실행하기 위해 다음 라이브러리들을 설치해야 합니다.

### **필요 라이브러리**
- `numpy`
- `opencv-python`
- `scipy`
- `matplotlib`
- `ultralytics` (YOLO11 모델 사용)
- `lap`

### **라이브러리 설치**
아래 명령어를 통해 필요한 패키지를 설치할 수 있습니다.
```bash
pip install numpy opencv-python scipy matplotlib ultralytics lap
```

---

## 주요 파일 및 설명

### **1. calib_extract_data.py**
- **목적:** 입력된 비디오에서 보행자의 상체 길이를 기준으로 300개의 라인 세그먼트를 추출하고 JSON 파일로 저장합니다.
- **주요 기능:**
  - YOLO를 사용하여 보행자의 관절 정보를 추적
  - Homography를 활용하여 라인의 시작점(머리)과 끝점(발)을 계산
  - 추출된 데이터는 `data/data.json` 파일에 저장
- **예제 코드:**
  ```python
  from calib_extract_data import DataExtractor

  data_extractor = DataExtractor(line_length=0.5)
  data_extractor.extract('data/input.avi')  # 'input.avi'는 분석할 동영상
  ```

---

### **2. main.py**
- **목적:** 추출된 데이터를 활용하여 카메라의 내·외부 파라미터를 계산하고 보정 결과를 저장합니다.
- **프로세스:**
  1. **데이터 추출:** `calib_extract_data.py`를 사용하여 JSON 데이터 생성.
  2. **데이터 로드:** 추출된 JSON 데이터를 로드하여 보정에 필요한 정보 준비.
  3. **데이터 샘플링:** SpatialSampler를 사용하여 데이터를 정제.
  4. **MSAC 기반 초기 보정:** RANSAC을 변형한 MSAC 알고리즘으로 내·외부 파라미터 계산.
  5. **최적화:** Bundle Adjustment로 최적의 결과 도출.
  6. **결과 저장:** 카메라의 회전행렬, 이동벡터, 내적 행렬 및 왜곡 계수를 텍스트 파일로 저장.
- **실행 방법:**
  ```bash
  python main.py
  ```

---

## 데이터 파일 구조
프로그램 실행 후 생성되는 데이터 파일은 다음과 같은 구조를 가집니다.

### **1. data/data.json**
- 추출된 보행자 데이터:
  - `a`: 발 위치 데이터
  - `b`: 머리 위치 데이터
  - `l`: 보행자의 상체 길이
  - `cam_w`: 카메라 해상도 (가로)
  - `cam_h`: 카메라 해상도 (세로)

### **2. data/calibration.txt**
- 카메라 보정 결과:
  - `RotationMatrices`: 카메라 회전 행렬
  - `TranslationVectors`: 카메라 이동 벡터
  - `IntrinsicMatrix`: 카메라 내부 파라미터 (초점 거리, 광축 중심)

### **3. data/dist_coef.txt**
- 렌즈 왜곡 계수:
  - `Distortion`: 왜곡 계수 값

---

## 실행 예제

### **비디오 데이터에서 데이터 추출**
1. `calib_extract_data.py` 실행:
   ```python
   from calib_extract_data import DataExtractor

   extractor = DataExtractor(line_length=0.5)
   extractor.extract('data/input.avi')
   ```

### **보정 실행**
2. `main.py` 실행:
   ```bash
   python main.py
   ```
   실행 후, 보정 결과는 `data/calibration.txt`와 `data/dist_coef.txt`에 저장됩니다.

---

## 주의사항
- **YOLO 모델**: `yolo11n-pose.pt` 모델이 필요한데, 이 파일은 [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)에서 다운로드하거나 제공받아야 합니다.
- **입력 동영상**: `data/input.avi` 경로에 보정할 동영상을 준비해야 합니다.
- **결과 파일 저장 경로**: 모든 결과는 `data/` 디렉터리에 저장되므로, 실행 전에 해당 폴더가 존재하는지 확인하십시오.

---

## 참고 문서
- [OpenCV Documentation](https://docs.opencv.org/)
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
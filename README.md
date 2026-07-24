# Smart CCTV Vehicle Type Classification

지능형 CCTV 환경에서 차량을 탐지하고, 탐지된 차량 영역을 기반으로 세부 차종을 분류하는 실시간 관제용 AI 프로젝트입니다.

이 프로젝트의 목표는 단순히 `car`/`truck`을 구분하는 것이 아니라, CCTV 프레임 안의 차량을 찾아낸 뒤 실제 관제 환경에서 필요한 **차종 단위의 식별 결과**를 제공하는 것입니다.

## 핵심 기능

- CCTV 또는 이미지 프레임에서 차량 객체 탐지
- 탐지된 bounding box 영역을 차량 crop으로 변환
- YOLO26s classification 모델로 차종 분류
- 예측 차종, confidence, top-5 후보를 CSV로 저장
- 실시간 카메라 관제 서비스에 붙일 수 있는 two-stage inference 구조 제공

## Pipeline

```text
CCTV frame
   |
   v
YOLO detector
car / truck bounding box detection
   |
   v
Vehicle crop
detected bbox region only
   |
   v
YOLO26s classifier
vehicle type classification
   |
   v
Prediction result
class name / confidence / top-5
```

1차 탐지는 COCO pretrained YOLO detector를 사용해 `car`, `truck` 객체만 선택합니다.  
2차 분류는 탐지된 차량 bbox 영역을 crop한 뒤, AIHub 기반 차량 이미지로 학습한 YOLO26s classification 모델이 차종을 예측합니다.

## Final Model

최종 제출용 차종 분류 모델은 아래 경로에 포함되어 있습니다.

```text
models/vehicle-classifier-yolo26s-full27-2/weights/best.pt
```

보관된 모델 구성:

```text
models/vehicle-classifier-yolo26s-full27-2/
├─ weights/
│  ├─ best.pt
│  └─ last.pt
├─ args.yaml
├─ results.csv
└─ train_batch*.jpg
```

일반 추론과 제출에는 `best.pt`를 사용합니다. `last.pt`는 학습 재개 또는 백업 용도입니다.

## Dataset

학습 데이터는 AIHub 차량 관련 원천/라벨링 데이터를 기반으로 구성했습니다.

라벨링 데이터의 bounding box와 차종명을 사용해 원천 이미지에서 차량 영역만 crop했고, YOLO classification 형식에 맞춰 아래 구조로 정리했습니다.

```text
vehicle_cls_crops_sampled/
├─ train/
│  ├─ class_a/
│  └─ class_b/
├─ val/
│  ├─ class_a/
│  └─ class_b/
└─ test/
   ├─ class_a/
   └─ class_b/
```

원천 데이터셋 전체는 용량과 라이선스 문제로 repository에 포함하지 않았습니다. 대신 데이터 전처리와 crop 생성 로직은 `src/`에 포함했습니다.

## Evaluation

Colab notebook에서 최종 모델 `yolo26s_cls_full-27-2`를 기준으로 학습 재개, 실제 CCTV crop 이미지 평가, 사용자 업로드 CCTV 이미지 추론을 수행했습니다.

노트북:

```text
notebooks/colab_train_predict_vehicle_cls.ipynb
```

현재 검증 방향은 다음 두 가지를 분리해 확인하는 것입니다.

- **분류 모델 자체 성능**: 이미 차량 영역만 crop된 이미지에서 차종을 잘 맞추는가
- **실서비스 파이프라인 성능**: detector가 선택한 bbox crop이 분류기에 충분히 좋은 입력인가

최근 검증에서는 실제 폐차장 CCTV에 가까운 단일 차량 crop 이미지에서는 모델이 의미 있는 예측을 보였고, 웹에서 가져온 보정된 차량 사진이나 다차량 full image에서는 오탐이 증가하는 경향을 확인했습니다. 이는 학습 데이터의 도메인과 테스트 이미지의 촬영 조건 차이가 성능에 영향을 준다는 점을 보여줍니다.

## Why This Matters

일반적인 차량 탐지는 `car`, `truck` 수준의 객체 인식에 머무르는 경우가 많습니다. 하지만 관제 서비스에서는 “차량이 있다”보다 “어떤 차종인가”가 더 유용한 정보가 될 수 있습니다.

이 프로젝트는 다음 상황을 목표로 합니다.

- 폐차장, 주차장, 출입구 CCTV 기반 차량 모니터링
- 특정 차종 출현 이력 저장
- 프레임별 탐지 결과와 차종 예측 로그 생성
- 향후 웹 UI에서 탐지 이미지 목록과 상세 프레임 조회

## Repository Structure

```text
configs/
  pipeline.yaml

models/
  vehicle-classifier-yolo26s-full27-2/

notebooks/
  colab_train_predict_vehicle_cls.ipynb

src/
  append_aihub_zip_to_cls_dataset.py
  crop_aihub_to_cls_dataset.py
  crop_aihub_zip_to_cls_dataset.py
  infer_two_stage.py
  predict_cls.py
  prepare_cls_dataset.py
  train.py

requirements.txt
```

## Quick Start

```bash
pip install -r requirements.txt
```

Two-stage inference:

```bash
python src/infer_two_stage.py \
  --det-weights yolo26s.pt \
  --cls-weights models/vehicle-classifier-yolo26s-full27-2/weights/best.pt \
  --source samples \
  --out runs/two_stage
```

Crop image classification:

```bash
python src/predict_cls.py \
  --weights models/vehicle-classifier-yolo26s-full27-2/weights/best.pt \
  --source samples/vehicle_crops \
  --out runs/vehicle_cls_predict \
  --imgsz 224 \
  --topk 5
```

## Limitations and Next Steps

- 비슷한 외관의 연식/파생 차종은 confusion이 발생할 수 있습니다.
- 학습 데이터와 다른 도메인, 예를 들어 홍보용 차량 사진이나 강한 보정/반사광이 있는 이미지는 성능이 낮아질 수 있습니다.
- 실제 서비스에서는 한 프레임에 차량이 여러 대일 때 관심 차량 선택 로직이 필요합니다.
- 향후에는 실시간 IP camera 입력, 결과 이미지 저장, 웹 기반 프레임 조회 UI를 통합할 예정입니다.

## Tech Stack

- Python
- Ultralytics YOLO
- OpenCV
- PyTorch
- Google Colab T4 GPU
- Google Drive based experiment storage

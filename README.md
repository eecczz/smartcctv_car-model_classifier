# car-cls

차량 탐지와 차종 분류를 분리해서 실험하는 Ultralytics YOLO26 기반 프로젝트입니다.

목표 흐름은 다음과 같습니다.

1. 1차: COCO pretrained `yolo26s.pt`로 `car`, `truck`만 탐지
2. 2차: 탐지된 차량 영역 crop을 `yolo26s-cls.pt` 기반 차종 분류기로 분류
3. Colab GPU에서 학습/추론하고, VS Code에서는 데이터 해석과 코드 관리를 진행

## 폴더 구조

```text
car-cls/
├── configs/
│   └── pipeline.yaml
├── notebooks/
│   └── colab_vehicle_pipeline.ipynb
├── src/
│   ├── infer_two_stage.py
│   ├── make_full_image_det_dataset.py
│   ├── prepare_cls_dataset.py
│   └── train.py
├── requirements.txt
└── README.md
```

## 먼저 결정할 것

이미 모아둔 AIHub 쪽 이미지가 “차량만 crop된 날이미지”라면, 차종 분류용으로는 JSON이나 bbox가 없어도 됩니다. Ultralytics classification 데이터셋은 클래스 폴더 구조만 있으면 됩니다.

다만 클래스명이 파일명에 들어있다면 아래처럼 파일명 파싱 규칙을 정해야 합니다.

```text
sedan_0001.jpg       -> sedan
truck-abc-0002.png   -> truck
SUV 003.jpeg         -> SUV
```

파일명이 불규칙하면 `--class-regex`로 직접 정규식을 줄 수 있습니다.

## 로컬 준비

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 1. 차종 분류 데이터셋 해석

원본 이미지가 `D:\aihub_vehicle_raw`에 있다고 가정합니다.

먼저 실제로 파일명에서 클래스가 어떻게 읽히는지 리포트만 봅니다.

```powershell
python src\prepare_cls_dataset.py `
  --src D:\aihub_vehicle_raw `
  --out data\vehicle_cls `
  --dry-run
```

문제가 없으면 YOLO classification 구조로 복사합니다.

```powershell
python src\prepare_cls_dataset.py `
  --src D:\aihub_vehicle_raw `
  --out data\vehicle_cls `
  --val-ratio 0.15 `
  --test-ratio 0.05
```

생성 결과:

```text
data/vehicle_cls/
├── train/
│   ├── sedan/
│   └── truck/
├── val/
│   ├── sedan/
│   └── truck/
└── test/
    ├── sedan/
    └── truck/
```

## 1-A. AIHub 원천/라벨링 zip에서 bbox crop 분류 데이터셋 만들기

AIHub에서 받은 원천데이터 zip과 라벨링데이터 zip이 있으면, JSON의 bbox만큼 원천 이미지를 잘라 차종명 폴더로 저장할 수 있습니다.

현재 `185.CCTV 기반 차량정보 및 교통정보 계측 데이터`처럼 `TS_*.zip`, `TL_*.zip` 한 쌍으로 들어있는 경우에는 zip을 전부 풀지 않고 바로 crop할 수 있습니다.

```powershell
python src\crop_aihub_zip_to_cls_dataset.py `
  --src "C:\Users\swh01\Downloads\185.CCTV 기반 차량정보 및 교통정보 계측 데이터" `
  --out data\vehicle_cls_crops_sampled `
  --class-field model_id `
  --exclude-class unknown `
  --frame-stride 1 `
  --min-class-count 1 `
  --dry-run
```

클래스 분포가 정상적으로 나오면 실제 데이터셋을 만듭니다.

```powershell
python src\crop_aihub_zip_to_cls_dataset.py `
  --src "C:\Users\swh01\Downloads\185.CCTV 기반 차량정보 및 교통정보 계측 데이터" `
  --out data\vehicle_cls_crops_sampled `
  --class-field model_id `
  --exclude-class unknown `
  --frame-stride 1 `
  --min-class-count 1 `
  --val-ratio 0.15 `
  --test-ratio 0.05
```

현재 생성된 `data/vehicle_cls_crops_sampled`는 이름과 달리 비산사거리와 인덕원사거리의 bbox를 프레임 건너뛰기 없이 모두 crop한 전체 버전입니다. 용량 중복을 피하기 위해 별도의 `full` 폴더는 남기지 않습니다.

- `model_id`의 trailing `(2015)`, `(unknown)` 같은 괄호 정보는 폴더명에서 제거
- `--frame-stride 1`: 모든 프레임 사용
- `--min-class-count 1`: 적은 수의 차종도 유지
- `unknown` 단독 클래스는 제외

아래 스크립트는 zip을 먼저 풀어둔 폴더나 일반 이미지/JSON 폴더에도 쓸 수 있는 범용 변환기입니다.

```powershell
python src\crop_aihub_to_cls_dataset.py `
  --src data\raw_aihub `
  --out data\vehicle_cls_crops `
  --extract-dir data\extracted\aihub_vehicle `
  --dry-run
```

`Top parsed classes`에 원하는 차종명이 제대로 나오면 실제 crop을 만듭니다.

```powershell
python src\crop_aihub_to_cls_dataset.py `
  --src data\raw_aihub `
  --out data\vehicle_cls_crops `
  --extract-dir data\extracted\aihub_vehicle `
  --val-ratio 0.15 `
  --test-ratio 0.05
```

생성 결과는 YOLO classification 학습에 바로 넣을 수 있는 구조입니다.

```text
data/vehicle_cls_crops/
├── train/
│   ├── sedan/
│   └── SUV/
├── val/
│   ├── sedan/
│   └── SUV/
└── test/
    ├── sedan/
    └── SUV/
```

라벨 JSON에서 차종 키가 자동으로 잘못 잡히면 `--class-key model`처럼 실제 키 이름을 지정합니다.

## 2. 차종 분류기 학습

Colab에서 바로 학습/예측까지 돌릴 때는 아래 노트북을 사용합니다.

```text
notebooks/colab_train_predict_vehicle_cls.ipynb
```

Drive에 `car-cls.zip`을 올린 뒤 노트북을 실행하면 됩니다. 노트북은 아래처럼 `/content`에 압축을 풀고 그 안의 `data/vehicle_cls_crops_sampled`를 학습 데이터로 사용합니다.

```bash
unzip /content/drive/MyDrive/car-cls.zip -d /content
```

```powershell
python src\train.py `
  --task classify `
  --model yolo26s-cls.pt `
  --data data\vehicle_cls_crops_sampled `
  --epochs 50 `
  --imgsz 224 `
  --project runs\vehicle_cls `
  --name yolo26s_cls
```

Colab에서는 같은 명령을 Linux 경로로 바꾸면 됩니다.

Crop된 차량 이미지 폴더에 대해 차종만 바로 예측하려면:

```powershell
python src\predict_cls.py `
  --weights runs\vehicle_cls\yolo26s_cls\weights\best.pt `
  --source samples\vehicle_crops `
  --out runs\vehicle_cls_predict `
  --imgsz 224 `
  --topk 5
```

## 3. 1차 탐지

COCO pretrained 모델은 이미 `car`, `truck` 클래스를 알고 있으므로, 1차 실험은 별도 학습 없이 `yolo26s.pt`에서 class filter만 걸어도 됩니다.

```powershell
python src\infer_two_stage.py `
  --det-weights yolo26s.pt `
  --cls-weights runs\vehicle_cls\yolo26s_cls\weights\best.pt `
  --source samples `
  --out runs\two_stage
```

COCO class id 기준:

- `car`: 2
- `truck`: 7

## 4. 객체탐지용 데이터셋이 꼭 필요할 때

현재 보유 이미지가 “차량만 crop된 이미지”라면 각 이미지 전체를 bbox로 간주해 YOLO detection 라벨을 만들 수 있습니다. 이건 실제 장면 속 차량 탐지 성능을 학습시키기엔 한계가 있지만, bbox 포맷 실험이나 파이프라인 검증에는 쓸 수 있습니다.

```powershell
python src\make_full_image_det_dataset.py `
  --src D:\aihub_vehicle_raw `
  --out data\vehicle_det_fullbox
```

## Ultralytics HUB 프로젝트를 새로 만들어야 하나?

Colab/Python 코드로만 학습한다면 HUB 프로젝트를 꼭 만들 필요는 없습니다. 로컬/Drive의 `runs/.../weights/best.pt`를 그대로 저장하고 불러오면 됩니다.

다만 배포 관리까지 HUB에서 할 거라면 분리하는 편이 좋습니다.

- `vehicle-detector-coco-car-truck`: pretrained detector 또는 detector fine-tune용
- `vehicle-type-classifier-yolo26s`: AIHub 차종 분류기 학습용

탐지기와 분류기는 task가 달라서 한 모델 프로젝트로 섞지 않는 게 관리가 쉽습니다.

## 참고 문서

- YOLO26: https://docs.ultralytics.com/models/yolo26/
- Classification dataset format: https://docs.ultralytics.com/datasets/classify/
- Detection dataset format: https://docs.ultralytics.com/datasets/detect/

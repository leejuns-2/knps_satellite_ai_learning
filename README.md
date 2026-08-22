# KNPS Satellite AI Learning Lab

국립공원 위성 모니터링 AI 챌린지를 준비하면서 **기초 수학 → PyTorch → CNN → Binary Segmentation → 실험/평가 → GeoRaster → Change Detection**을 직접 실행해보는 학습용 코드베이스입니다.

이 저장소의 목적은 단순히 코드를 복사하는 것이 아니라 다음 루프를 반복하는 것입니다.

```text
개념 이해
→ 최소 코드 실행
→ shape / dtype / 값 확인
→ 결과 시각화
→ 실험 조건 변경
→ 결과 해석
→ 포트폴리오 기록
```

## 0. 설치

Python 3.12 권장. 현재 검증 환경은 Python 3.12입니다.

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

동일한 패키지 버전으로 재현하려면 다음을 사용합니다.

```bash
pip install -r requirements-lock.txt
```

GPU가 없어도 모든 핵심 실습은 CPU에서 실행할 수 있도록 작게 구성했습니다.

## 1. 실행 순서

### A. AI 최소 수학
```bash
python labs/01_math_and_gradient.py
```

확인할 것:
- 함수와 미분
- 수치 미분과 autograd 비교
- gradient descent가 parameter를 어떻게 움직이는지

### B. Python · NumPy · PyTorch
```bash
python labs/02_numpy_torch_foundations.py
```

확인할 것:
- `[H,W,C]` ↔ `[C,H,W]`
- dtype
- Dataset / DataLoader
- `loss.backward()`
- `optimizer.step()`

### C. CNN
```bash
python labs/03_cnn_shapes.py
```

확인할 것:
- kernel / stride / padding
- feature map shape
- encoder에서 공간 해상도가 줄어드는 과정

### D. Binary Segmentation
```bash
python labs/04_segmentation_metrics.py
python train.py
python evaluate.py
python inference.py
```

확인할 것:
- logit → sigmoid → probability → threshold → binary mask
- BCEWithLogitsLoss
- Dice / IoU / Precision / Recall
- checkpoint 저장/복원
- validation과 inference 분리

### E. 실험 파이프라인
```bash
python labs/05_threshold_sweep.py
```

`outputs/threshold_sweep.csv`가 생성됩니다.

### F/G. Remote Sensing · GeoRaster
```bash
python labs/06_georaster_basics.py
```

synthetic GeoTIFF를 만들고 다음 metadata를 직접 확인합니다.
- CRS
- transform
- bounds
- nodata
- window read
- patch 저장

### H. Change Detection
```bash
python labs/07_change_detection.py
```

Before/After 두 영상을 6채널로 결합하고 변화 mask를 만드는 가장 단순한 구조를 확인합니다.

### 전체 실행
```bash
python run_all.py
```

### 자동 테스트

```bash
python -m unittest discover -s tests -v
```

### 실제 GeoTIFF image-mask 데이터

실제 데이터는 `data/geotiff/images`와 `data/geotiff/masks`에 같은 파일명으로
배치합니다. 상세한 규칙은 `data/README.md`를 확인하고, 센서별 band와 반사도
스케일을 `configs/geotiff_example.json`에서 수정합니다.

```bash
python train_geotiff.py
```

`GeoTiffSegmentationDataset`은 영상과 마스크의 크기, CRS, affine transform이
일치하는지 확인합니다. 이 스크립트는 실제 데이터를 넣기 전에는 실행되지 않는
골격이며, 공간적으로 인접한 patch가 학습/검증에 섞이지 않도록 실제 실험에서는
공원·지역 단위 분할로 교체해야 합니다.

## 2. 프로젝트 구조

```text
knps_satellite_ai_learning/
├── configs/
│   ├── baseline.json
│   └── geotiff_example.json
├── data/
├── labs/
│   ├── 01_math_and_gradient.py
│   ├── 02_numpy_torch_foundations.py
│   ├── 03_cnn_shapes.py
│   ├── 04_segmentation_metrics.py
│   ├── 05_threshold_sweep.py
│   ├── 06_georaster_basics.py
│   └── 07_change_detection.py
├── src/
│   ├── datasets.py
│   ├── losses.py
│   ├── metrics.py
│   ├── models.py
│   └── utils.py
├── tests/
│   └── test_core.py
├── checkpoints/
├── outputs/
├── portfolio/
│   └── PORTFOLIO_TEMPLATE.md
├── train.py
├── train_geotiff.py
├── evaluate.py
├── inference.py
├── run_all.py
├── requirements.txt
└── requirements-lock.txt
```

## 3. 학습할 때 지켜야 할 규칙

각 파일을 실행하기 전에 먼저 코드를 읽고 아래를 예상합니다.

1. 입력 shape은 무엇인가?
2. 출력 shape은 무엇인가?
3. dtype은 무엇인가?
4. 어떤 값 범위를 갖는가?
5. loss는 무엇을 줄이려고 하는가?
6. metric은 무엇을 측정하는가?
7. 코드 한 줄을 지우면 어떤 문제가 생기는가?

실행 후에는 예상과 실제 결과를 비교합니다.

## 4. 첫 번째 실험 과제

`configs/baseline.json`에서 한 번에 하나만 바꿉니다.

- `learning_rate`: `0.001 → 0.01`
- `batch_size`: `16 → 4`
- `threshold`: `0.5 → 0.3`, `0.7`
- `image_size`: `64 → 128`

한 번에 여러 개를 바꾸지 않습니다.

포트폴리오에는 다음처럼 기록합니다.

```text
가설:
threshold를 0.5에서 0.3으로 낮추면 Recall은 증가하고 Precision은 감소할 것이다.

변경:
threshold 0.5 → 0.3

결과:
Dice / IoU / Precision / Recall 기록

해석:
positive 판정 범위가 넓어지면서 FN은 감소했지만 FP가 증가했다.

다음 실험:
small-object 상황에서 threshold 변화가 더 크게 작용하는지 확인한다.
```

## 5. 실제 대회 데이터로 바꾸는 시점

지금 코드는 synthetic segmentation dataset을 사용합니다.

아래를 설명하고 구현할 수 있게 된 뒤 실제 위성 데이터로 넘어갑니다.

- Dataset과 DataLoader의 차이
- `[B,C,H,W]`
- logits와 probability의 차이
- BCEWithLogitsLoss가 sigmoid를 내부적으로 포함하는 이유
- `zero_grad → backward → step`
- train/eval 차이
- Dice와 IoU 계산
- checkpoint load
- threshold sweep

그 다음 `src/datasets.py`의 `SyntheticSegmentationDataset`을 실제 GeoTIFF/image-mask Dataset으로 교체하면 됩니다.

현재는 `GeoTiffSegmentationDataset`과 `train_geotiff.py`가 이 전환의 시작점을
제공합니다. 다만 실제 데이터의 좌표계 통일, patch 생성, class imbalance,
지역 단위 train/validation/test 분리는 데이터 특성에 맞게 추가해야 합니다.

## 6. 포트폴리오 산출물

실행하면서 최소 다음을 남깁니다.

- GitHub repository
- README
- 환경 파일
- 학습 곡선
- Dice/IoU 결과
- threshold sweep CSV
- prediction 시각화
- GeoTIFF metadata 출력
- change detection 입력 shape 증거
- 실패 사례와 해결 과정
- 실험표

`portfolio/PORTFOLIO_TEMPLATE.md`를 채우면 바로 프로젝트 설명 초안으로 사용할 수 있습니다.

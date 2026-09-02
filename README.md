# Satellite Segmentation Learning Lab

PyTorch segmentation과 위성영상 처리 기초를 단계별로 연습하는 학습용 저장소입니다. 현재 주 학습·평가 코드는 synthetic segmentation data를 사용합니다. GeoTIFF 지원은 실제 image-mask 파일을 읽고 기본 정합성을 검사하는 초기 구현이며, 실제 위성 데이터에 대한 모델 평가는 아직 완료하지 않았습니다.

이 저장소의 목적은 높은 benchmark 성능을 주장하는 것이 아니라, 데이터 shape부터 학습·평가·원격탐사 파일 처리까지 재현 가능한 작은 실험으로 이해하는 것입니다.

## Current Status

### Completed

- NumPy와 PyTorch tensor, gradient, DataLoader 실습
- CNN shape 확인과 작은 U-Net 형태의 binary segmentation 모델
- BCE/Dice loss와 Dice, IoU, Precision, Recall 계산
- synthetic data 학습·검증 loop와 checkpoint 저장/복원
- threshold sweep
- synthetic GeoTIFF 생성, metadata 확인, window read
- 6-channel before/after change-detection 입력 예제
- GeoTIFF image-mask loader의 크기·CRS·affine transform 검사
- 핵심 dataset, model, metric, GeoTIFF loader 단위 테스트

### Next

- 라이선스가 확인된 실제 위성 image-mask dataset 선정
- 인접 patch가 나뉘지 않도록 지역 단위 train/validation/test split
- class imbalance와 작은 객체 대응
- 실제 데이터 baseline Dice/IoU 측정
- prediction overlay와 geographic output 저장
- 정량적 failure-case 분석
- 센서별 band, nodata, reflectance scaling 검증

## Learning Path

```text
math and gradients
  -> tensors and DataLoader
  -> CNN shapes
  -> binary segmentation
  -> metrics and threshold sweep
  -> GeoTIFF basics
  -> before/after change detection
```

각 실습은 독립적으로 실행할 수 있습니다.

```bash
python labs/01_math_and_gradient.py
python labs/02_numpy_torch_foundations.py
python labs/03_cnn_shapes.py
python labs/04_segmentation_metrics.py
python labs/05_threshold_sweep.py
python labs/06_georaster_basics.py
python labs/07_change_detection.py
```

## Synthetic Segmentation Pipeline

```bash
python train.py
python evaluate.py
python inference.py
```

`train.py`, `evaluate.py`, `inference.py`는 `SyntheticSegmentationDataset`을 사용합니다. 생성되는 Dice/IoU는 코드 동작을 확인하기 위한 synthetic experiment 결과이며 실제 위성영상 성능이 아닙니다.

전체 학습 순서를 한 번에 실행하려면 다음 명령을 사용합니다.

```bash
python run_all.py
```

## GeoTIFF Extension

실제 image와 mask를 `data/geotiff/images`와 `data/geotiff/masks`에 같은 파일명으로 배치하면 `train_geotiff.py`가 쌍을 읽습니다.

```bash
python train_geotiff.py
```

`GeoTiffSegmentationDataset`은 크기, CRS, affine transform 일치를 검사합니다. 기본 설정은 `split_manifest.csv`를 사용해 완전한 장면/지역을 train, validation, reserved test로 명시적으로 분리합니다. `train_geotiff.py`는 reserved test를 평가하지 않으므로 validation을 보면서 설정을 정한 뒤 별도 final evaluation을 구현해야 합니다. `random_smoke_only`는 동작 확인용이며 benchmark 근거로 사용할 수 없습니다. 실제 실험에는 타일링, nodata 처리, 센서별 normalization, geographic output 복원이 추가로 필요합니다. 입력 규칙은 [`data/README.md`](data/README.md)에 있습니다.

## Metrics

- Dice와 IoU: 예측 mask와 정답 mask의 overlap
- Precision과 Recall: false positive와 false negative의 균형
- Threshold sweep: 같은 probability에서 이진 mask 기준값을 바꿔 metric 변화를 확인

실험 수치는 `portfolio/experiment_log.csv`에 기록합니다. 현재 행은 `pending` 상태이며 완료된 metric으로 해석하면 안 됩니다.

## Installation and Tests

Python 3.12 환경을 기준으로 작성했습니다.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m unittest discover -s tests -v
```

정확히 고정된 의존성을 사용하려면 `requirements-lock.txt`를 설치합니다.

## Repository Structure

```text
configs/       synthetic and GeoTIFF experiment settings
data/          local data placement guide; datasets are not committed
labs/          ordered learning exercises
portfolio/     experiment log and write-up template
src/           datasets, model, losses, metrics, utilities
tests/         unit tests for the core pipeline
train.py       synthetic training
train_geotiff.py  early GeoTIFF training path
evaluate.py    synthetic validation
inference.py   synthetic prediction examples
```

## Limitations

- 실제 위성 데이터 성능과 geographic generalization 결과가 없습니다.
- synthetic shapes는 실제 토지피복·재난 객체의 질감과 class imbalance를 재현하지 않습니다.
- GeoTIFF loader는 초기 구현이며 대용량 raster tiling과 prediction stitching을 지원하지 않습니다.
- spatial manifest는 파일 단위 분리를 강제하지만, 파일 자체가 인접 patch라면 작성자가 region 단위 격리를 확인해야 합니다.
- baseline experiment log의 metric은 아직 비어 있으며, 결과가 생성되기 전에는 성능을 주장하지 않습니다.

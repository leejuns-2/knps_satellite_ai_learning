# Satellite Segmentation Learning Lab

PyTorch 기반 semantic segmentation과 위성영상 처리 기초를
단계적으로 학습하고 실험하는 개인 학습용 저장소입니다.

현재 주요 학습·평가 코드는 **synthetic segmentation data**를 사용하고 있으며,
GeoTIFF 관련 코드는 실제 image-mask 파일을 읽고 기본적인 공간 정보 정합성을 확인하는 초기 단계입니다.

> This is a learning repository, not a benchmark project.  
> Current segmentation results are mainly based on synthetic data and should not be interpreted as real satellite-image performance.

---

## Project Context

- Type: Individual learning project
- Main focus:
  - PyTorch fundamentals
  - CNN and semantic segmentation
  - U-Net style architecture
  - Segmentation metrics
  - GeoTIFF handling
  - Basic geospatial image processing

이 저장소의 목적은 높은 성능을 주장하는 것이 아니라,
데이터 shape, 학습 loop, metric, checkpoint, GeoTIFF 처리 과정을
작은 실험 단위로 직접 확인하며 이해하는 것입니다.

---

## What I Practiced

현재까지 다음 내용을 실습했습니다.

- NumPy / PyTorch tensor 연산
- gradient와 기본적인 backpropagation 흐름
- `Dataset` / `DataLoader`
- CNN input-output shape 확인
- 작은 U-Net 형태의 binary segmentation model
- BCE / Dice loss
- Dice, IoU, Precision, Recall 계산
- training / validation loop
- checkpoint 저장 및 복원
- threshold sweep
- synthetic GeoTIFF 생성 및 metadata 확인
- raster window read
- 6-channel before / after change-detection 입력 구성
- GeoTIFF image-mask의 크기, CRS, affine transform 정합성 확인
- dataset / model / metric / GeoTIFF loader 단위 테스트

구현 과정에서는 **generative AI coding tools를 보조적으로 활용**했으며,
코드를 실행하고 결과를 확인하면서 각 구성요소의 역할과 오류 원인을 이해하는 데 집중했습니다.

---

## Current Status

### Completed

- PyTorch tensor, gradient, Dataset / DataLoader 기초 실습
- CNN shape 확인
- 작은 U-Net style binary segmentation pipeline
- BCE / Dice loss 구현 및 사용
- Dice / IoU / Precision / Recall 계산
- synthetic data training / validation
- checkpoint 저장 및 복원
- threshold sweep
- synthetic GeoTIFF 처리
- GeoTIFF image-mask loader 기본 구현
- 핵심 구성요소에 대한 unit test

### Not Completed Yet

- 실제 위성 image-mask dataset에서의 정식 baseline 성능 측정
- geographic generalization 평가
- 대규모 raster tiling 및 stitching
- sensor별 normalization
- 실제 데이터 기반 failure-case 분석

따라서 현재 저장소의 metric은
실제 위성영상 성능을 나타내는 benchmark 결과로 해석하지 않습니다.

---

## Learning Path

```text
Math & Gradients
        ↓
NumPy / PyTorch Tensors
        ↓
Dataset / DataLoader
        ↓
CNN Shapes
        ↓
Binary Segmentation
        ↓
Segmentation Metrics
        ↓
Threshold Sweep
        ↓
GeoTIFF Basics
        ↓
Before / After Change Detection
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

---

## Synthetic Segmentation Pipeline

현재 기본 segmentation 학습·평가는
`SyntheticSegmentationDataset`을 이용합니다.

```bash
python train.py
python evaluate.py
python inference.py
```

전체 학습 과정을 한 번에 실행하려면:

```bash
python run_all.py
```

현재 생성되는 Dice / IoU 등의 결과는
**코드와 학습 pipeline이 정상적으로 동작하는지 확인하기 위한 synthetic experiment 결과**입니다.

실제 위성 데이터에 대한 benchmark 성능이 아닙니다.

---

## Model

현재 기본 모델은 작은 U-Net style binary segmentation network입니다.

학습 과정에서 다음 흐름을 확인하는 데 초점을 두었습니다.

```text
Input Image
    ↓
Encoder
    ↓
Feature Representation
    ↓
Decoder
    ↓
Pixel-wise Prediction
    ↓
Binary Segmentation Mask
```

복잡한 architecture를 사용하는 것보다
segmentation model의 전체 학습 흐름을 이해하는 것을 우선했습니다.

---

## Loss and Metrics

### Loss

- Binary Cross Entropy (BCE)
- Dice Loss

### Evaluation Metrics

- **Dice**
  - prediction mask와 ground-truth mask의 overlap 정도

- **IoU**
  - prediction과 ground truth의 intersection / union

- **Precision**
  - 모델이 positive라고 판단한 픽셀 중 실제 positive 비율

- **Recall**
  - 실제 positive 픽셀 중 모델이 탐지한 비율

- **Threshold Sweep**
  - 동일한 probability output에서 threshold를 변경하며 metric 변화를 확인

---

## GeoTIFF Extension

실제 image와 mask를 다음 구조로 배치하면
`train_geotiff.py`에서 읽을 수 있도록 구성했습니다.

```text
data/
  geotiff/
    images/
    masks/
```

image와 mask는 동일한 파일명을 사용합니다.

실행:

```bash
python train_geotiff.py
```

`GeoTiffSegmentationDataset`은 다음 항목의 일치를 확인합니다.

- image / mask 크기
- CRS
- affine transform

기본 설정에서는 `split_manifest.csv`를 이용해
전체 scene 또는 region 단위로

```text
train
validation
reserved test
```

를 명시적으로 분리하도록 구성했습니다.

`train_geotiff.py`는 reserved test를 학습 과정에서 평가하지 않습니다.

validation 결과를 보면서 설정을 결정한 뒤
별도의 final evaluation을 수행하는 구조를 목표로 합니다.

`random_smoke_only` 옵션은 코드 동작 확인용이며,
실제 benchmark 결과로 사용하지 않습니다.

---

## Why Scene-level Split Matters

위성영상에서는 서로 인접한 patch를
무작위로 train과 validation에 나누면
매우 유사한 장면이 양쪽에 포함될 수 있습니다.

따라서 실제 데이터 실험에서는
가능한 한 **scene 또는 geographic region 단위 분리**가 필요합니다.

현재 `split_manifest.csv`는 파일 단위 분리를 지원하지만,
각 파일 자체가 인접 patch라면 작성자가
실제 geographic independence를 별도로 확인해야 합니다.

---

## Experiment Tracking

실험 결과는 다음 파일에 기록하도록 구성했습니다.

```text
portfolio/experiment_log.csv
```

현재 baseline experiment 행은 `pending` 상태이며,
실제 데이터에 대한 완료된 metric으로 해석하지 않습니다.

---

## Installation

Python 3.12 환경을 기준으로 작성했습니다.

가상환경 생성:

```bash
python -m venv .venv
```

활성화:

```bash
source .venv/bin/activate
```

Windows:

```powershell
.venv\Scripts\activate
```

dependency 설치:

```bash
pip install -r requirements.txt
```

정확히 고정된 dependency를 사용하려면:

```bash
pip install -r requirements-lock.txt
```

---

## Tests

핵심 pipeline이 정상적으로 동작하는지 확인하기 위해
unit test를 실행할 수 있습니다.

```bash
python -m unittest discover -s tests -v
```

현재 테스트 대상에는 다음 구성요소가 포함됩니다.

- Dataset
- Model
- Segmentation metrics
- GeoTIFF loader

---

## Repository Structure

```text
configs/          experiment configuration
data/             local data placement guide
labs/             ordered learning exercises
portfolio/        experiment log and write-up template
src/              datasets, model, losses, metrics, utilities
tests/            unit tests

train.py          synthetic training
evaluate.py       synthetic validation
inference.py      synthetic prediction
run_all.py        learning pipeline runner
train_geotiff.py  early GeoTIFF training path
```

실제 dataset은 저장소에 직접 포함하지 않는 것을 기본으로 합니다.

---

## Limitations

현재 저장소에는 다음 한계가 있습니다.

- 실제 위성 데이터에서의 정식 segmentation 성능이 아직 없습니다.
- synthetic shapes는 실제 해안·토지피복·재난 객체의 복잡한 질감을 재현하지 못합니다.
- 실제 데이터의 class imbalance와 small-object 문제를 충분히 다루지 않았습니다.
- GeoTIFF loader는 초기 구현 단계입니다.
- 대규모 raster tiling 및 prediction stitching을 지원하지 않습니다.
- sensor별 band 구성, nodata, reflectance scaling을 충분히 다루지 않았습니다.
- 실제 geographic generalization 성능을 평가하지 않았습니다.
- 현재 experiment log의 metric은 완료된 실제 benchmark 결과가 아닙니다.

---

## Next Steps

다음 단계에서는 실제 위성 데이터 실험으로 확장하는 것을 목표로 합니다.

- 라이선스가 확인된 실제 image-mask dataset 선정
- scene / region 단위 train-validation-test split
- 실제 데이터 baseline Dice / IoU 측정
- class imbalance 대응
- small-object segmentation 분석
- prediction overlay 생성
- geographic output 저장
- 정량적인 failure-case 분석
- sensor별 normalization 확인
- nodata 및 raster preprocessing 정리

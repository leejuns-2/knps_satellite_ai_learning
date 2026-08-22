# 위성영상 기반 변화 탐지 AI 학습 프로젝트

## 1. 프로젝트 한 줄 설명

위성영상 변화 탐지 문제를 해결하기 위해 기초 수학과 PyTorch부터 Binary Segmentation, GeoRaster, Change Detection까지 직접 구현하고 검증한 학습형 프로젝트.

## 2. 문제 정의

- 대상 문제:
- 입력:
- 출력:
- 핵심 난점:
- 평가 지표:

## 3. 내가 직접 구현한 범위

- [ ] Tensor / Dataset / DataLoader
- [ ] CNN shape 실험
- [ ] Tiny U-Net
- [ ] BCE + Dice Loss
- [ ] Dice / IoU / Precision / Recall
- [ ] Training / Validation 분리
- [ ] Checkpoint 저장/복원
- [ ] Threshold sweep
- [ ] Prediction visualization
- [ ] GeoTIFF metadata 확인
- [ ] Window read / patch 저장
- [ ] Before/After 6-channel change detection 입력

## 4. 가장 중요하게 이해한 개념

### Gradient
내 설명:

### Logit / Sigmoid / Threshold
내 설명:

### Dice / IoU
내 설명:

### GeoTIFF / CRS / Transform
내 설명:

### Change Detection
내 설명:

## 5. Baseline 실험

| Experiment | Change | Dice | IoU | Precision | Recall | Interpretation |
|---|---|---:|---:|---:|---:|---|
| baseline | lr=0.001, threshold=0.5 |  |  |  |  |  |
| exp-01 | threshold=0.3 |  |  |  |  |  |
| exp-02 | threshold=0.7 |  |  |  |  |  |

## 6. 가설 → 실험 → 결과 → 해석

### Experiment 01
**가설**

**변경 변수**

**고정 변수**

**결과**

**해석**

**다음 실험**

## 7. 실패 사례

### 사례 1
- 현상:
- 원인:
- 확인 방법:
- 해결:
- 배운 점:

## 8. 결과 이미지

아래 파일을 GitHub README 또는 포트폴리오에 첨부:

- `outputs/training_curve.png`
- `outputs/predictions/*`
- `outputs/07_change_difference.png`
- `outputs/07_change_prediction.png`

## 9. 대회 데이터로 확장할 내용

- 실제 image/mask pair 로더
- GeoTIFF tiling
- 좌표 보존
- before/after co-registration 확인
- overlap inference
- patch stitching
- class imbalance 대응
- small-object 분석
- pretrained encoder
- augmentation
- experiment tracking

## 10. 면접에서 설명할 문장

> 이 프로젝트에서는 모델 아키텍처를 바로 복사하지 않고, 먼저 데이터 shape과 mask 정합성, logit-to-probability 변환, loss와 metric의 차이, checkpoint 재현성을 직접 검증했습니다. 이후 GeoTIFF의 CRS와 affine transform을 유지한 patch 처리와 before/after 입력 구조까지 확장하여 위성영상 변화 탐지 문제에 필요한 전체 파이프라인을 단계적으로 구현했습니다.

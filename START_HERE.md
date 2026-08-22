# START HERE · 실행형 학습 체크리스트

이 파일은 Notion의 `01. 통합 학습 가이드`를 실제 코드 실행으로 연결하는 진입점입니다.

## 사용법

각 단계마다 반드시 아래 4개를 남깁니다.

1. **예상**: 코드를 실행하기 전에 shape / 출력 / metric 변화를 예상
2. **실행**: 명령어 실행
3. **증거**: 콘솔 출력, PNG, JSON, CSV 저장
4. **설명**: 왜 그런 결과가 나왔는지 본인 말로 3~5문장 작성

---

## A. AI 최소 수학

실행:

```bash
python labs/01_math_and_gradient.py
```

내가 설명해야 할 것:

- 함수란 무엇인가?
- 미분값이 왜 "변화율"인가?
- gradient의 부호가 parameter 이동 방향과 어떤 관계인가?
- `loss.backward()`가 무엇을 계산하는가?

포트폴리오 증거:

- `outputs/01_gradient_descent_loss.png`
- 최종 parameter가 2.0에 가까워지는 콘솔 출력

직접 수정:

- learning rate `0.1 → 0.01`
- learning rate `0.1 → 0.5`
- 시작값 `w=8 → w=-5`

기록 질문:

> learning rate가 너무 크거나 작으면 loss curve가 어떻게 달라졌는가?

---

## B. Python · NumPy · PyTorch

실행:

```bash
python labs/02_numpy_torch_foundations.py
```

내가 설명해야 할 것:

- `[H,W,C]`와 `[C,H,W]` 차이
- Dataset과 DataLoader 차이
- `zero_grad()`
- `backward()`
- `step()`

직접 수정:

- batch size 변경
- `lr=0.1 → 0.01`
- Dataset 샘플 수 변경

증거:

- optimizer step 전후 weight가 실제로 달라진 출력

---

## C. CNN

실행:

```bash
python labs/03_cnn_shapes.py
```

내가 설명해야 할 것:

- kernel / stride / padding
- feature map
- downsampling
- encoder

직접 수정:

- kernel `3 → 5`
- stride `1 → 2`
- padding `1 → 0`

기록:

| kernel | stride | padding | input H×W | output H×W | 해석 |
|---:|---:|---:|---|---|---|
| 3 | 1 | 1 | 64×64 |  |  |
| 3 | 2 | 1 | 64×64 |  |  |

---

## D. Binary Segmentation

먼저:

```bash
python labs/04_segmentation_metrics.py
```

그 다음 전체 baseline:

```bash
python train.py
python evaluate.py
python inference.py
```

내가 설명해야 할 것:

```text
logit
→ sigmoid
→ probability
→ threshold
→ binary mask
```

그리고:

- BCEWithLogitsLoss
- Dice
- IoU
- Precision
- Recall
- false positive / false negative

포트폴리오 증거:

- `outputs/training_curve.png`
- `outputs/evaluation_metrics.json`
- `outputs/predictions/`
- `checkpoints/best.pt`

---

## E. 실험 가능한 파이프라인

실행:

```bash
python labs/05_threshold_sweep.py
```

증거:

- `outputs/threshold_sweep.csv`

반드시 직접 해볼 실험:

### Experiment 1
- 변경: threshold `0.5 → 0.3`
- 예상: Recall 증가, Precision 감소 가능성
- 실제:
- 해석:

### Experiment 2
- 변경: threshold `0.5 → 0.7`
- 예상: Precision 증가, Recall 감소 가능성
- 실제:
- 해석:

### Experiment 3
`configs/baseline.json`에서 learning rate만 변경.

원칙:

> 한 실험에서 주요 변수 하나만 바꾼다.

---

## F/G. Remote Sensing · GeoRaster

실행:

```bash
python labs/06_georaster_basics.py
```

내가 설명해야 할 것:

- raster
- GeoTIFF
- CRS
- affine transform
- bounds
- nodata
- window read
- patch
- 왜 patch transform을 보존해야 하는가?

증거:

- `outputs/synthetic_satellite.tif`
- `outputs/synthetic_patch_256.tif`
- metadata 콘솔 출력

직접 수정:

- patch size `256 → 128`
- window 시작 좌표 변경
- pixel size 변경

---

## H. Change Detection

실행:

```bash
python labs/07_change_detection.py
```

내가 설명해야 할 것:

```text
Before RGB [3,H,W]
+
After RGB [3,H,W]
=
Early-fusion input [6,H,W]
```

증거:

- `outputs/07_change_difference.png`
- `outputs/07_change_prediction.png`
- `[1,6,H,W]` 입력 shape 출력

다음 단계:

`TinyUNet(in_channels=6)`로 바꿔 실제 학습형 change detection Dataset을 구현한다.

---

# 첫 GitHub 커밋 권장 범위

처음에는 아래까지만 올려도 충분합니다.

```text
README.md
START_HERE.md
requirements.txt
configs/
labs/
src/
train.py
evaluate.py
inference.py
portfolio/
```

실행 결과 중 포트폴리오에 보여줄 PNG/CSV/JSON만 선별해서 추가합니다.

현재 저장소에서는 가상환경, checkpoint, 원본 데이터, 자동 생성 output을
`.gitignore`로 제외합니다. 코드 변경 후에는 먼저 아래 테스트를 통과시킵니다.

```bash
python -m unittest discover -s tests -v
```

---

# 완료 기준

다음 질문에 코드 없이 말로 답할 수 있으면 1차 기반 학습을 통과한 것입니다.

- 왜 segmentation output은 `[B,1,H,W]`인가?
- logit과 probability의 차이는?
- threshold를 바꾸면 Precision/Recall이 왜 달라지는가?
- loss와 metric은 왜 별개인가?
- `model.train()`과 `model.eval()` 차이는?
- checkpoint를 왜 저장하는가?
- GeoTIFF의 CRS와 transform은 왜 중요한가?
- 큰 위성영상을 전체 resize하지 않고 patch로 나누는 이유는?
- Before/After 영상을 6채널로 합친다는 것은 무슨 뜻인가?
- 다음 실험에서 무엇을 하나만 바꿀 것인가?

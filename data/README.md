# 실제 GeoTIFF 데이터 배치

`train_geotiff.py`는 아래처럼 영상과 마스크의 파일명이 같은 쌍을 읽습니다.

```text
data/geotiff/
├── images/
│   ├── park_001.tif
│   └── park_002.tif
├── masks/
│   ├── park_001.tif
│   └── park_002.tif
└── split_manifest.csv
```

영상과 마스크는 width, height, CRS, affine transform이 같아야 합니다. 마스크는
첫 번째 band에서 `mask_threshold`보다 큰 픽셀을 1로 변환합니다. 영상 값의
스케일은 `configs/geotiff_example.json`의 `image_scale`을 실제 센서 데이터에
맞게 바꾸세요.

`split_manifest.csv`는 실제 benchmark 전에 지역/장면 단위로 작성합니다.

```csv
stem,split,region,note
park_001,train,region_a,complete scene assigned before patching
park_002,validation,region_b,geographically separate region
park_003,test,region_c,reserved; train_geotiff.py does not evaluate it
```

기본 `spatial_manifest` 전략은 각 파일 stem을 `train`, `validation`, `test` 중 하나에 정확히 한 번 배정하도록 강제합니다. 같은 큰 GeoTIFF에서 만든 인접 patch를 서로 다른 split에 넣지 마세요. `random_smoke_only` 옵션은 loader/training 동작 확인용이며 spatial generalization 결과로 보고할 수 없습니다.

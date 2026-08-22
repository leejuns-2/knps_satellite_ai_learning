# 실제 GeoTIFF 데이터 배치

`train_geotiff.py`는 아래처럼 영상과 마스크의 파일명이 같은 쌍을 읽습니다.

```text
data/geotiff/
├── images/
│   ├── park_001.tif
│   └── park_002.tif
└── masks/
    ├── park_001.tif
    └── park_002.tif
```

영상과 마스크는 width, height, CRS, affine transform이 같아야 합니다. 마스크는
첫 번째 band에서 `mask_threshold`보다 큰 픽셀을 1로 변환합니다. 영상 값의
스케일은 `configs/geotiff_example.json`의 `image_scale`을 실제 센서 데이터에
맞게 바꾸세요.

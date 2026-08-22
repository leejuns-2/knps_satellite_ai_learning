from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio.windows import Window


out = Path("outputs")
out.mkdir(exist_ok=True)

path = out / "synthetic_satellite.tif"

height = 512
width = 512
count = 3

rng = np.random.default_rng(42)
data = rng.integers(0, 10000, size=(count, height, width), dtype=np.uint16)

# 예시: 좌상단 경도 127, 위도 38 / pixel size 약 0.0001 degree
transform = from_origin(127.0, 38.0, 0.0001, 0.0001)

with rasterio.open(
    path,
    "w",
    driver="GTiff",
    height=height,
    width=width,
    count=count,
    dtype=data.dtype,
    crs="EPSG:4326",
    transform=transform,
    nodata=0,
) as dst:
    dst.write(data)

print("created:", path)

with rasterio.open(path) as src:
    print("\n=== metadata ===")
    print("width, height:", src.width, src.height)
    print("count:", src.count)
    print("crs:", src.crs)
    print("transform:", src.transform)
    print("bounds:", src.bounds)
    print("dtypes:", src.dtypes)
    print("nodata:", src.nodata)

    window = Window(col_off=128, row_off=128, width=256, height=256)
    patch = src.read(window=window)
    patch_transform = src.window_transform(window)

    print("\n=== window read ===")
    print("patch shape:", patch.shape)
    print("patch transform:", patch_transform)

    patch_path = out / "synthetic_patch_256.tif"
    profile = src.profile.copy()
    profile.update(
        width=256,
        height=256,
        transform=patch_transform,
    )

    with rasterio.open(patch_path, "w", **profile) as dst:
        dst.write(patch)

print("saved patch:", patch_path)
print("\n핵심: patch를 잘라도 CRS/transform을 함께 보존해야 지리적 위치를 잃지 않는다.")

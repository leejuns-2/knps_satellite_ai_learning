from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset


class SyntheticSegmentationDataset(Dataset):
    '''
    실제 대회 데이터가 없어도 segmentation pipeline을 완주하기 위한 Dataset.

    image: [3, H, W], float32, 0~1
    mask : [1, H, W], float32, 0 또는 1

    positive object는 원/사각형 형태로 생성되고,
    image의 positive 위치가 더 밝게 만들어져 작은 CNN도 학습할 수 있다.
    '''

    def __init__(self, n_samples: int = 128, image_size: int = 64, seed: int = 42):
        self.n_samples = n_samples
        self.image_size = image_size
        self.seed = seed

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int):
        rng = np.random.default_rng(self.seed + idx)
        h = w = self.image_size

        image = rng.normal(0.15, 0.05, size=(3, h, w)).astype(np.float32)
        image = np.clip(image, 0.0, 1.0)
        mask = np.zeros((1, h, w), dtype=np.float32)

        yy, xx = np.mgrid[:h, :w]

        if idx % 2 == 0:
            radius = int(rng.integers(max(3, h // 12), max(4, h // 5)))
            cy = int(rng.integers(radius, h - radius))
            cx = int(rng.integers(radius, w - radius))
            obj = (yy - cy) ** 2 + (xx - cx) ** 2 <= radius ** 2
        else:
            rh = int(rng.integers(max(4, h // 10), max(5, h // 4)))
            rw = int(rng.integers(max(4, w // 10), max(5, w // 4)))
            cy = int(rng.integers(rh, h - rh))
            cx = int(rng.integers(rw, w - rw))
            obj = (np.abs(yy - cy) <= rh) & (np.abs(xx - cx) <= rw)

        mask[0, obj] = 1.0

        for c in range(3):
            image[c, obj] += float(rng.uniform(0.45, 0.75))
        image += rng.normal(0.0, 0.03, size=image.shape).astype(np.float32)
        image = np.clip(image, 0.0, 1.0)

        return torch.from_numpy(image), torch.from_numpy(mask)


def discover_geotiff_pairs(
    data_root: str | Path,
    *,
    image_dir: str = "images",
    mask_dir: str = "masks",
) -> list[tuple[Path, Path]]:
    """파일명이 같은 GeoTIFF image-mask 쌍을 찾는다."""
    root = Path(data_root)
    images = root / image_dir
    masks = root / mask_dir
    if not images.is_dir() or not masks.is_dir():
        raise FileNotFoundError(
            f"{images}와 {masks} 디렉터리가 모두 필요합니다."
        )

    extensions = {".tif", ".tiff"}
    image_files = {
        path.stem: path
        for path in images.iterdir()
        if path.is_file() and path.suffix.lower() in extensions
    }
    mask_files = {
        path.stem: path
        for path in masks.iterdir()
        if path.is_file() and path.suffix.lower() in extensions
    }

    missing_masks = sorted(image_files.keys() - mask_files.keys())
    missing_images = sorted(mask_files.keys() - image_files.keys())
    if missing_masks or missing_images:
        raise ValueError(
            "GeoTIFF image-mask 파일명이 일치하지 않습니다. "
            f"mask 없음={missing_masks}, image 없음={missing_images}"
        )

    pairs = [(image_files[stem], mask_files[stem]) for stem in sorted(image_files)]
    if not pairs:
        raise ValueError(f"{images}에서 GeoTIFF 파일을 찾지 못했습니다.")
    return pairs


class GeoTiffSegmentationDataset(Dataset):
    """동일 격자의 GeoTIFF image-mask 쌍을 PyTorch tensor로 읽는다.

    image_scale은 센서의 반사도 스케일에 맞게 지정한다. 예를 들어 값 범위가
    0~10000인 영상은 기본값 10000을 사용하고, 이미 0~1이면 1을 사용한다.
    """

    def __init__(
        self,
        pairs: list[tuple[str | Path, str | Path]],
        *,
        bands: tuple[int, ...] | list[int] | None = None,
        image_scale: float = 10000.0,
        mask_threshold: float = 0.0,
        strict_georeferencing: bool = True,
    ):
        if not pairs:
            raise ValueError("최소 한 개의 image-mask 쌍이 필요합니다.")
        if image_scale <= 0:
            raise ValueError("image_scale은 0보다 커야 합니다.")

        self.pairs = [(Path(image), Path(mask)) for image, mask in pairs]
        self.bands = tuple(bands) if bands is not None else None
        self.image_scale = float(image_scale)
        self.mask_threshold = float(mask_threshold)
        self.strict_georeferencing = strict_georeferencing

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        image_path, mask_path = self.pairs[idx]
        with rasterio.open(image_path) as image_src, rasterio.open(mask_path) as mask_src:
            if image_src.width != mask_src.width or image_src.height != mask_src.height:
                raise ValueError(
                    f"영상과 마스크 크기가 다릅니다: {image_path.name}, {mask_path.name}"
                )
            if self.strict_georeferencing and (
                image_src.crs != mask_src.crs or image_src.transform != mask_src.transform
            ):
                raise ValueError(
                    f"영상과 마스크의 CRS/transform이 다릅니다: "
                    f"{image_path.name}, {mask_path.name}"
                )

            indexes = self.bands or tuple(range(1, image_src.count + 1))
            if not indexes or min(indexes) < 1 or max(indexes) > image_src.count:
                raise ValueError(
                    f"요청한 bands={indexes}가 영상 band 수 {image_src.count}와 맞지 않습니다."
                )

            image = image_src.read(indexes=indexes).astype(np.float32)
            mask = mask_src.read(1).astype(np.float32)

            if image_src.nodata is not None:
                image[image == image_src.nodata] = 0.0
            if mask_src.nodata is not None:
                mask[mask == mask_src.nodata] = 0.0

        image = np.clip(image / self.image_scale, 0.0, 1.0)
        mask = (mask > self.mask_threshold).astype(np.float32)[None, ...]
        return torch.from_numpy(image), torch.from_numpy(mask)

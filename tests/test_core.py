import tempfile
import unittest
from pathlib import Path

import numpy as np
import rasterio
import torch
from rasterio.transform import from_origin

from src.datasets import (
    GeoTiffSegmentationDataset,
    SyntheticSegmentationDataset,
    discover_geotiff_pairs,
)
from src.metrics import metric_dict
from src.models import TinyUNet
from src.utils import validate_baseline_config


class CorePipelineTests(unittest.TestCase):
    def test_synthetic_dataset_is_reproducible(self):
        first = SyntheticSegmentationDataset(n_samples=1, seed=7)[0]
        second = SyntheticSegmentationDataset(n_samples=1, seed=7)[0]
        self.assertTrue(torch.equal(first[0], second[0]))
        self.assertTrue(torch.equal(first[1], second[1]))

    def test_tiny_unet_preserves_spatial_shape(self):
        model = TinyUNet(in_channels=3, base=4)
        output = model(torch.zeros(2, 3, 64, 64))
        self.assertEqual(tuple(output.shape), (2, 1, 64, 64))

    def test_metrics_are_one_for_perfect_prediction(self):
        targets = torch.tensor([[[[0.0, 1.0], [1.0, 0.0]]]])
        logits = torch.where(targets == 1, torch.tensor(20.0), torch.tensor(-20.0))
        metrics = metric_dict(logits, targets)
        for value in metrics.values():
            self.assertAlmostEqual(value, 1.0, places=6)

    def test_precision_is_zero_when_nothing_is_predicted(self):
        targets = torch.ones(1, 1, 2, 2)
        logits = torch.full_like(targets, -20.0)
        metrics = metric_dict(logits, targets)
        self.assertEqual(metrics["precision"], 0.0)
        self.assertEqual(metrics["recall"], 0.0)

    def test_invalid_unet_image_size_is_rejected(self):
        config = {
            "seed": 42,
            "image_size": 65,
            "train_samples": 8,
            "val_samples": 2,
            "batch_size": 2,
            "epochs": 1,
            "learning_rate": 0.001,
            "threshold": 0.5,
            "in_channels": 3,
            "model_base": 4,
        }
        with self.assertRaisesRegex(ValueError, "4의 배수"):
            validate_baseline_config(config)

    def test_geotiff_pair_is_loaded_with_georeferencing(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image_dir = root / "images"
            mask_dir = root / "masks"
            image_dir.mkdir()
            mask_dir.mkdir()
            transform = from_origin(127.0, 38.0, 0.01, 0.01)
            image = np.full((3, 8, 8), 5000, dtype=np.uint16)
            mask = np.zeros((1, 8, 8), dtype=np.uint8)
            mask[:, 2:6, 2:6] = 1

            metadata = {
                "driver": "GTiff",
                "width": 8,
                "height": 8,
                "crs": "EPSG:4326",
                "transform": transform,
            }
            with rasterio.open(
                image_dir / "sample.tif", "w", count=3, dtype="uint16", **metadata
            ) as dst:
                dst.write(image)
            with rasterio.open(
                mask_dir / "sample.tif", "w", count=1, dtype="uint8", **metadata
            ) as dst:
                dst.write(mask)

            pairs = discover_geotiff_pairs(root)
            loaded_image, loaded_mask = GeoTiffSegmentationDataset(pairs)[0]
            self.assertEqual(tuple(loaded_image.shape), (3, 8, 8))
            self.assertEqual(tuple(loaded_mask.shape), (1, 8, 8))
            self.assertAlmostEqual(float(loaded_image.mean()), 0.5, places=6)
            self.assertEqual(float(loaded_mask.sum()), 16.0)


if __name__ == "__main__":
    unittest.main()

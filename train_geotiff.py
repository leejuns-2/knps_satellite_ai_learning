from __future__ import annotations

import csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split

from src.datasets import GeoTiffSegmentationDataset, discover_geotiff_pairs
from src.losses import bce_dice_loss
from src.metrics import metric_dict
from src.models import TinyUNet
from src.utils import get_device, load_config, save_json, set_seed


def validate_geotiff_config(config: dict) -> dict:
    required = {
        "data_root",
        "image_dir",
        "mask_dir",
        "image_scale",
        "mask_threshold",
        "split_strategy",
        "seed",
        "batch_size",
        "epochs",
        "learning_rate",
        "threshold",
        "model_base",
    }
    missing = sorted(required - config.keys())
    if missing:
        raise ValueError(f"GeoTIFF 설정에 필수 키가 없습니다: {', '.join(missing)}")
    if config["split_strategy"] not in {"spatial_manifest", "random_smoke_only"}:
        raise ValueError("split_strategy는 spatial_manifest 또는 random_smoke_only여야 합니다.")
    if config["split_strategy"] == "spatial_manifest" and not config.get("split_manifest"):
        raise ValueError("spatial_manifest 전략에는 split_manifest 경로가 필요합니다.")
    if config["split_strategy"] == "random_smoke_only" and not 0.0 < float(
        config.get("validation_fraction", 0)
    ) < 1.0:
        raise ValueError("random_smoke_only의 validation_fraction은 0과 1 사이여야 합니다.")
    if float(config["image_scale"]) <= 0.0:
        raise ValueError("image_scale은 0보다 커야 합니다.")
    if not 0.0 <= float(config["threshold"]) <= 1.0:
        raise ValueError("threshold는 0과 1 사이여야 합니다.")
    return config


def split_pairs_from_manifest(
    pairs: list[tuple[Path, Path]], manifest_path: str | Path
) -> tuple[list[tuple[Path, Path]], list[tuple[Path, Path]], list[tuple[Path, Path]]]:
    """Split complete scenes/regions using an explicit, reviewable manifest."""
    path = Path(manifest_path)
    if not path.exists():
        raise FileNotFoundError(f"Spatial split manifest not found: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or not {"stem", "split"}.issubset(rows[0]):
        raise ValueError("Split manifest must contain stem,split columns")

    assignments = {str(row["stem"]): str(row["split"]).lower() for row in rows}
    allowed = {"train", "validation", "test"}
    invalid = sorted(set(assignments.values()) - allowed)
    if invalid:
        raise ValueError(f"Unknown split labels: {invalid}")

    pair_stems = {image.stem for image, _ in pairs}
    missing = sorted(pair_stems - set(assignments))
    extra = sorted(set(assignments) - pair_stems)
    if missing or extra:
        raise ValueError(f"Manifest/pair mismatch: missing={missing}, extra={extra}")

    grouped = {
        split: [pair for pair in pairs if assignments[pair[0].stem] == split]
        for split in allowed
    }
    if not grouped["train"] or not grouped["validation"]:
        raise ValueError("Spatial manifest requires at least one train and validation scene")
    return grouped["train"], grouped["validation"], grouped["test"]


def main() -> None:
    config_path = Path("configs/geotiff_example.json")
    cfg = validate_geotiff_config(load_config(config_path))
    set_seed(int(cfg["seed"]))
    device = get_device()
    print("device:", device)

    try:
        pairs = discover_geotiff_pairs(
            cfg["data_root"],
            image_dir=cfg["image_dir"],
            mask_dir=cfg["mask_dir"],
        )
    except (FileNotFoundError, ValueError) as error:
        raise SystemExit(f"GeoTIFF 데이터 준비 오류: {error}") from error
    if len(pairs) < 2:
        raise SystemExit("학습/검증 분리를 위해 GeoTIFF 쌍이 최소 2개 필요합니다.")

    dataset_kwargs = {
        "bands": cfg.get("bands"),
        "image_scale": cfg["image_scale"],
        "mask_threshold": cfg["mask_threshold"],
    }
    full_dataset = GeoTiffSegmentationDataset(pairs, **dataset_kwargs)
    sample_image, _ = full_dataset[0]
    in_channels = int(sample_image.shape[0])

    if cfg["split_strategy"] == "spatial_manifest":
        train_pairs, val_pairs, test_pairs = split_pairs_from_manifest(
            pairs, cfg["split_manifest"]
        )
        train_ds = GeoTiffSegmentationDataset(train_pairs, **dataset_kwargs)
        val_ds = GeoTiffSegmentationDataset(val_pairs, **dataset_kwargs)
        train_count, val_count, test_count = len(train_pairs), len(val_pairs), len(test_pairs)
    else:
        print("WARNING: random_smoke_only is not valid evidence of spatial generalization.")
        val_count = max(1, round(len(full_dataset) * float(cfg["validation_fraction"])))
        val_count = min(val_count, len(full_dataset) - 1)
        train_count = len(full_dataset) - val_count
        test_count = 0
        generator = torch.Generator().manual_seed(int(cfg["seed"]))
        train_ds, val_ds = random_split(
            full_dataset,
            [train_count, val_count],
            generator=generator,
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg["batch_size"]),
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg["batch_size"]),
        shuffle=False,
        num_workers=0,
    )

    model = TinyUNet(in_channels=in_channels, base=int(cfg["model_base"])).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg["learning_rate"]))
    checkpoint_path = Path("checkpoints/geotiff_best.pt")
    checkpoint_path.parent.mkdir(exist_ok=True)

    history = []
    best_dice = -1.0
    for epoch in range(1, int(cfg["epochs"]) + 1):
        model.train()
        train_loss_sum = 0.0
        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = bce_dice_loss(logits, masks)
            loss.backward()
            optimizer.step()
            train_loss_sum += float(loss.item()) * images.size(0)

        model.eval()
        logits_list = []
        masks_list = []
        with torch.no_grad():
            for images, masks in val_loader:
                logits_list.append(model(images.to(device)).cpu())
                masks_list.append(masks)

        logits = torch.cat(logits_list)
        masks = torch.cat(masks_list)
        metrics = metric_dict(logits, masks, threshold=float(cfg["threshold"]))
        row = {
            "epoch": epoch,
            "train_loss": train_loss_sum / train_count,
            **metrics,
        }
        history.append(row)
        print(
            f"epoch={epoch:02d} train_loss={row['train_loss']:.4f} "
            f"dice={row['dice']:.4f} iou={row['iou']:.4f}"
        )

        if metrics["dice"] > best_dice:
            best_dice = metrics["dice"]
            torch.save(
                {
                    "model": model.state_dict(),
                    "config": cfg,
                    "in_channels": in_channels,
                    "epoch": epoch,
                    "metrics": metrics,
                },
                checkpoint_path,
            )

    save_json(
        {
            "config": cfg,
            "pair_count": len(pairs),
            "train_count": train_count,
            "val_count": val_count,
            "test_count_reserved_not_evaluated": test_count,
            "split_strategy": cfg["split_strategy"],
            "benchmark_status": "development only; no held-out real-data test evaluation",
            "history": history,
        },
        "outputs/geotiff_train_history.json",
    )
    print("best checkpoint:", checkpoint_path)
    print("best dice:", best_dice)


if __name__ == "__main__":
    main()

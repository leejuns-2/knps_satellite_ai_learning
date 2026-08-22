from __future__ import annotations

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
        "validation_fraction",
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
    if not 0.0 < float(config["validation_fraction"]) < 1.0:
        raise ValueError("validation_fraction은 0과 1 사이여야 합니다.")
    if float(config["image_scale"]) <= 0.0:
        raise ValueError("image_scale은 0보다 커야 합니다.")
    if not 0.0 <= float(config["threshold"]) <= 1.0:
        raise ValueError("threshold는 0과 1 사이여야 합니다.")
    return config


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

    dataset = GeoTiffSegmentationDataset(
        pairs,
        bands=cfg.get("bands"),
        image_scale=cfg["image_scale"],
        mask_threshold=cfg["mask_threshold"],
    )
    sample_image, _ = dataset[0]
    in_channels = int(sample_image.shape[0])

    val_count = max(1, round(len(dataset) * float(cfg["validation_fraction"])))
    val_count = min(val_count, len(dataset) - 1)
    train_count = len(dataset) - val_count
    generator = torch.Generator().manual_seed(int(cfg["seed"]))
    train_ds, val_ds = random_split(
        dataset,
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
            "history": history,
        },
        "outputs/geotiff_train_history.json",
    )
    print("best checkpoint:", checkpoint_path)
    print("best dice:", best_dice)


if __name__ == "__main__":
    main()

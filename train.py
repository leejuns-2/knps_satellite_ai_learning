from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from src.datasets import SyntheticSegmentationDataset
from src.losses import bce_dice_loss
from src.metrics import metric_dict
from src.models import TinyUNet
from src.utils import get_device, load_config, save_json, set_seed, validate_baseline_config


cfg = validate_baseline_config(load_config("configs/baseline.json"))
set_seed(cfg["seed"])
device = get_device()
print("device:", device)

train_ds = SyntheticSegmentationDataset(
    n_samples=cfg["train_samples"],
    image_size=cfg["image_size"],
    seed=cfg["seed"],
)
val_ds = SyntheticSegmentationDataset(
    n_samples=cfg["val_samples"],
    image_size=cfg["image_size"],
    seed=cfg["seed"] + 10000,
)

train_loader = DataLoader(
    train_ds,
    batch_size=cfg["batch_size"],
    shuffle=True,
    num_workers=0,
)
val_loader = DataLoader(
    val_ds,
    batch_size=cfg["batch_size"],
    shuffle=False,
    num_workers=0,
)

model = TinyUNet(in_channels=cfg["in_channels"], base=cfg["model_base"]).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])

history = []
best_dice = -1.0
checkpoint_path = Path("checkpoints/best.pt")
checkpoint_path.parent.mkdir(exist_ok=True)

for epoch in range(1, cfg["epochs"] + 1):
    model.train()
    train_loss_sum = 0.0

    for images, masks in train_loader:
        images = images.to(device)
        masks = masks.to(device)

        logits = model(images)
        loss = bce_dice_loss(logits, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss_sum += float(loss.item()) * images.size(0)

    train_loss = train_loss_sum / len(train_ds)

    model.eval()
    val_loss_sum = 0.0
    logits_list = []
    masks_list = []

    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(device)
            masks = masks.to(device)

            logits = model(images)
            loss = bce_dice_loss(logits, masks)

            val_loss_sum += float(loss.item()) * images.size(0)
            logits_list.append(logits.cpu())
            masks_list.append(masks.cpu())

    val_loss = val_loss_sum / len(val_ds)
    all_logits = torch.cat(logits_list)
    all_masks = torch.cat(masks_list)
    metrics = metric_dict(all_logits, all_masks, cfg["threshold"])

    row = {
        "epoch": epoch,
        "train_loss": train_loss,
        "val_loss": val_loss,
        **metrics,
    }
    history.append(row)

    print(
        f"epoch={epoch:02d} "
        f"train_loss={train_loss:.4f} "
        f"val_loss={val_loss:.4f} "
        f"dice={metrics['dice']:.4f} "
        f"iou={metrics['iou']:.4f}"
    )

    if metrics["dice"] > best_dice:
        best_dice = metrics["dice"]
        torch.save(
            {
                "model": model.state_dict(),
                "config": cfg,
                "epoch": epoch,
                "metrics": metrics,
            },
            checkpoint_path,
        )

Path("outputs").mkdir(exist_ok=True)
save_json({"config": cfg, "history": history}, "outputs/train_history.json")

epochs = [r["epoch"] for r in history]
train_losses = [r["train_loss"] for r in history]
val_losses = [r["val_loss"] for r in history]

plt.figure()
plt.plot(epochs, train_losses, marker="o", label="train")
plt.plot(epochs, val_losses, marker="o", label="validation")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.title("Training curve")
plt.legend()
plt.tight_layout()
plt.savefig("outputs/training_curve.png", dpi=150)
plt.close()

print("\nbest checkpoint:", checkpoint_path)
print("best dice:", best_dice)
print("saved: outputs/train_history.json")
print("saved: outputs/training_curve.png")

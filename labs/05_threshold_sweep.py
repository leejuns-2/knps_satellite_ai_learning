import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import csv
import torch
from torch.utils.data import DataLoader

from src.datasets import SyntheticSegmentationDataset
from src.metrics import metric_dict
from src.models import TinyUNet
from src.utils import get_device, validate_baseline_config


device = get_device()
checkpoint = Path("checkpoints/best.pt")

if not checkpoint.exists():
    raise SystemExit("먼저 `python train.py`를 실행해 checkpoints/best.pt를 만드세요.")

state = torch.load(checkpoint, map_location=device)
cfg = validate_baseline_config(state["config"])
dataset = SyntheticSegmentationDataset(
    n_samples=cfg["val_samples"],
    image_size=cfg["image_size"],
    seed=cfg["seed"] + 10000,
)
loader = DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=False)

model = TinyUNet(
    in_channels=cfg["in_channels"],
    base=cfg["model_base"],
).to(device)
model.load_state_dict(state["model"])
model.eval()

rows = []

with torch.no_grad():
    all_logits = []
    all_targets = []

    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)
        logits = model(images)
        all_logits.append(logits.cpu())
        all_targets.append(masks.cpu())

logits = torch.cat(all_logits)
targets = torch.cat(all_targets)

for threshold in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    metrics = metric_dict(logits, targets, threshold)
    row = {"threshold": threshold, **metrics}
    rows.append(row)
    print(row)

out = Path("outputs")
out.mkdir(exist_ok=True)
csv_path = out / "threshold_sweep.csv"

with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

best = max(rows, key=lambda r: r["dice"])
print("\nbest by Dice:", best)
print("saved:", csv_path)

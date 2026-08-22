import torch
from torch.utils.data import DataLoader

from src.datasets import SyntheticSegmentationDataset
from src.metrics import metric_dict
from src.models import TinyUNet
from src.utils import (
    assert_checkpoint_config_compatible,
    get_device,
    load_config,
    save_json,
    validate_baseline_config,
)


cfg = validate_baseline_config(load_config("configs/baseline.json"))
device = get_device()

checkpoint = torch.load("checkpoints/best.pt", map_location=device)
checkpoint_cfg = validate_baseline_config(checkpoint["config"])
assert_checkpoint_config_compatible(
    cfg,
    checkpoint_cfg,
    allowed_differences={"threshold"},
)

dataset = SyntheticSegmentationDataset(
    n_samples=cfg["val_samples"],
    image_size=cfg["image_size"],
    seed=cfg["seed"] + 10000,
)
loader = DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=False)

model = TinyUNet(
    in_channels=checkpoint_cfg["in_channels"],
    base=checkpoint_cfg["model_base"],
).to(device)
model.load_state_dict(checkpoint["model"])
model.eval()

logits_list = []
targets_list = []

with torch.no_grad():
    for images, masks in loader:
        logits = model(images.to(device))
        logits_list.append(logits.cpu())
        targets_list.append(masks)

logits = torch.cat(logits_list)
targets = torch.cat(targets_list)

metrics = metric_dict(logits, targets, threshold=cfg["threshold"])
print(metrics)

save_json(metrics, "outputs/evaluation_metrics.json")
print("saved: outputs/evaluation_metrics.json")

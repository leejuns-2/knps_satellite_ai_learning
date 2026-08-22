from pathlib import Path

import matplotlib.pyplot as plt
import torch

from src.datasets import SyntheticSegmentationDataset
from src.models import TinyUNet
from src.utils import (
    assert_checkpoint_config_compatible,
    get_device,
    load_config,
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

model = TinyUNet(
    in_channels=checkpoint_cfg["in_channels"],
    base=checkpoint_cfg["model_base"],
).to(device)
model.load_state_dict(checkpoint["model"])
model.eval()

dataset = SyntheticSegmentationDataset(
    n_samples=4,
    image_size=cfg["image_size"],
    seed=2026,
)

out = Path("outputs/predictions")
out.mkdir(parents=True, exist_ok=True)

for idx in range(len(dataset)):
    image, mask = dataset[idx]

    with torch.no_grad():
        logits = model(image.unsqueeze(0).to(device))
        prob = torch.sigmoid(logits)[0, 0].cpu()
        pred = (prob >= cfg["threshold"]).float()

    image_np = image.permute(1, 2, 0).numpy()

    # 각각 별도 파일로 저장
    plt.figure()
    plt.imshow(image_np)
    plt.title(f"sample {idx} - input")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out / f"{idx:02d}_input.png", dpi=150)
    plt.close()

    plt.figure()
    plt.imshow(mask[0].numpy())
    plt.title(f"sample {idx} - ground truth")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out / f"{idx:02d}_gt.png", dpi=150)
    plt.close()

    plt.figure()
    plt.imshow(prob.numpy())
    plt.title(f"sample {idx} - probability")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out / f"{idx:02d}_probability.png", dpi=150)
    plt.close()

    plt.figure()
    plt.imshow(pred.numpy())
    plt.title(f"sample {idx} - prediction")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out / f"{idx:02d}_prediction.png", dpi=150)
    plt.close()

print("saved predictions to:", out)

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


rng = np.random.default_rng(42)
h = w = 96

before = rng.normal(0.2, 0.03, size=(3, h, w)).astype(np.float32)
after = before.copy()
change_mask = np.zeros((1, h, w), dtype=np.float32)

# 새 시설물이 생겼다고 가정한 사각 영역
y1, y2 = 35, 58
x1, x2 = 50, 78

after[:, y1:y2, x1:x2] += 0.6
change_mask[:, y1:y2, x1:x2] = 1.0

before = np.clip(before, 0, 1)
after = np.clip(after, 0, 1)

six_channel = np.concatenate([before, after], axis=0)

print("before shape:", before.shape)
print("after shape :", after.shape)
print("6ch input   :", six_channel.shape)
print("GT mask     :", change_mask.shape)

x = torch.from_numpy(six_channel).unsqueeze(0)
y = torch.from_numpy(change_mask).unsqueeze(0)

print("\nPyTorch batch input :", x.shape)
print("PyTorch batch target:", y.shape)

# 가장 단순한 non-learning baseline: RGB absolute difference
diff = np.abs(after - before).mean(axis=0)
pred = (diff > 0.2).astype(np.float32)

intersection = (pred * change_mask[0]).sum()
union = pred.sum() + change_mask[0].sum() - intersection
iou = intersection / max(union, 1.0)
print("simple difference baseline IoU:", float(iou))

out = Path("outputs")
out.mkdir(exist_ok=True)

plt.figure()
plt.imshow(diff)
plt.title("Absolute RGB difference")
plt.axis("off")
plt.tight_layout()
plt.savefig(out / "07_change_difference.png", dpi=150)
plt.close()

plt.figure()
plt.imshow(pred)
plt.title("Thresholded change prediction")
plt.axis("off")
plt.tight_layout()
plt.savefig(out / "07_change_prediction.png", dpi=150)
plt.close()

print("saved outputs/07_change_difference.png")
print("saved outputs/07_change_prediction.png")

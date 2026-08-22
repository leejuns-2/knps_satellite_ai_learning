import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch.nn import functional as F

from src.metrics import metric_dict


target = torch.tensor(
    [[[
        [0, 0, 0, 0],
        [0, 1, 1, 0],
        [0, 1, 1, 0],
        [0, 0, 0, 0],
    ]]],
    dtype=torch.float32,
)

logits = torch.tensor(
    [[[
        [-3.0, -2.0, -1.0, -3.0],
        [-2.0,  2.5,  0.5, -2.0],
        [-1.5,  1.0,  3.0, -2.0],
        [-3.0, -2.0, -2.0, -3.0],
    ]]],
    dtype=torch.float32,
)

prob = torch.sigmoid(logits)

print("=== logit ===")
print(logits)

print("\n=== sigmoid probability ===")
print(prob.round(decimals=3))

print("\n=== BCEWithLogitsLoss ===")
loss = F.binary_cross_entropy_with_logits(logits, target)
print(float(loss))

for threshold in [0.3, 0.5, 0.7]:
    pred = (prob >= threshold).float()
    print(f"\n=== threshold={threshold} ===")
    print(pred)
    print(metric_dict(logits, target, threshold))

print("\n핵심:")
print("logit은 제한 없는 실수 점수")
print("sigmoid(logit)은 0~1 probability")
print("threshold 이후에야 최종 0/1 mask가 된다.")

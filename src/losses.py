import torch
from torch import nn


_bce = nn.BCEWithLogitsLoss()


def bce_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return _bce(logits, targets)


def soft_dice_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    dims = (1, 2, 3)
    intersection = (probs * targets).sum(dims)
    denominator = probs.sum(dims) + targets.sum(dims)
    dice = (2.0 * intersection + eps) / (denominator + eps)
    return 1.0 - dice.mean()


def bce_dice_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return bce_loss(logits, targets) + soft_dice_loss(logits, targets)

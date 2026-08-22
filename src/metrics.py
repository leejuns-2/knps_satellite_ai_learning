from __future__ import annotations

import torch


def _binary_from_logits(logits: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    return (probs >= threshold).float()


def confusion_counts(logits, targets, threshold: float = 0.5):
    preds = _binary_from_logits(logits, threshold)
    targets = (targets >= 0.5).float()

    dims = tuple(range(1, preds.ndim))
    tp = (preds * targets).sum(dims)
    fp = (preds * (1 - targets)).sum(dims)
    fn = ((1 - preds) * targets).sum(dims)
    tn = ((1 - preds) * (1 - targets)).sum(dims)
    return tp, fp, fn, tn


def metric_dict(logits, targets, threshold: float = 0.5, eps: float = 1e-7):
    tp, fp, fn, tn = confusion_counts(logits, targets, threshold)

    dice = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    iou = (tp + eps) / (tp + fp + fn + eps)
    precision_denominator = tp + fp
    recall_denominator = tp + fn
    precision = torch.where(
        precision_denominator > 0,
        tp / (precision_denominator + eps),
        torch.zeros_like(tp),
    )
    recall = torch.where(
        recall_denominator > 0,
        tp / (recall_denominator + eps),
        torch.zeros_like(tp),
    )

    return {
        "dice": float(dice.mean().item()),
        "iou": float(iou.mean().item()),
        "precision": float(precision.mean().item()),
        "recall": float(recall.mean().item()),
    }

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch


BASELINE_REQUIRED_KEYS = {
    "seed",
    "image_size",
    "train_samples",
    "val_samples",
    "batch_size",
    "epochs",
    "learning_rate",
    "threshold",
    "in_channels",
    "model_base",
}


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def validate_baseline_config(config: dict) -> dict:
    missing = sorted(BASELINE_REQUIRED_KEYS - config.keys())
    if missing:
        raise ValueError(f"설정에 필수 키가 없습니다: {', '.join(missing)}")

    positive_integer_keys = (
        "image_size",
        "train_samples",
        "val_samples",
        "batch_size",
        "epochs",
        "in_channels",
        "model_base",
    )
    for key in positive_integer_keys:
        if not isinstance(config[key], int) or config[key] <= 0:
            raise ValueError(f"{key}는 양의 정수여야 합니다: {config[key]!r}")

    if config["image_size"] % 4 != 0:
        raise ValueError("TinyUNet을 사용하려면 image_size가 4의 배수여야 합니다.")
    if not 0.0 <= float(config["threshold"]) <= 1.0:
        raise ValueError("threshold는 0과 1 사이여야 합니다.")
    if float(config["learning_rate"]) <= 0.0:
        raise ValueError("learning_rate는 0보다 커야 합니다.")
    return config


def assert_checkpoint_config_compatible(
    current: dict,
    saved: dict,
    *,
    allowed_differences: set[str] | None = None,
) -> None:
    allowed = allowed_differences or set()
    keys = BASELINE_REQUIRED_KEYS - allowed
    mismatches = {
        key: (saved.get(key), current.get(key))
        for key in keys
        if saved.get(key) != current.get(key)
    }
    if mismatches:
        details = ", ".join(
            f"{key}: checkpoint={old!r}, current={new!r}"
            for key, (old, new) in sorted(mismatches.items())
        )
        raise ValueError(
            "현재 설정과 체크포인트 설정이 다릅니다. 다시 학습하거나 설정을 복원하세요. "
            + details
        )


def save_json(obj: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

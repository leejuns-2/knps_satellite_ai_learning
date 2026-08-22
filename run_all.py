from __future__ import annotations

import subprocess
import sys


commands = [
    [sys.executable, "labs/01_math_and_gradient.py"],
    [sys.executable, "labs/02_numpy_torch_foundations.py"],
    [sys.executable, "labs/03_cnn_shapes.py"],
    [sys.executable, "labs/04_segmentation_metrics.py"],
    [sys.executable, "train.py"],
    [sys.executable, "evaluate.py"],
    [sys.executable, "inference.py"],
    [sys.executable, "labs/05_threshold_sweep.py"],
    [sys.executable, "labs/06_georaster_basics.py"],
    [sys.executable, "labs/07_change_detection.py"],
]

for cmd in commands:
    print("\n" + "=" * 80)
    print("RUN:", " ".join(cmd))
    print("=" * 80)
    subprocess.run(cmd, check=True)

print("\nALL LABS COMPLETED")

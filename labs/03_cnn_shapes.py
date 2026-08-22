import torch
from torch import nn


def conv_out_size(size, kernel, stride=1, padding=0, dilation=1):
    return ((size + 2*padding - dilation*(kernel-1) - 1) // stride) + 1


x = torch.randn(2, 3, 64, 64)
print("input:", x.shape)

experiments = [
    {"kernel_size": 3, "stride": 1, "padding": 1},
    {"kernel_size": 3, "stride": 2, "padding": 1},
    {"kernel_size": 5, "stride": 1, "padding": 0},
]

for cfg in experiments:
    conv = nn.Conv2d(3, 8, **cfg)
    y = conv(x)
    expected = conv_out_size(
        64,
        cfg["kernel_size"],
        cfg["stride"],
        cfg["padding"],
    )
    print("\nconfig:", cfg)
    print("output:", y.shape)
    print("formula expected spatial size:", expected)


encoder = nn.Sequential(
    nn.Conv2d(3, 8, 3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Conv2d(8, 16, 3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2),
)

z = x
print("\n=== encoder shape trace ===")
for i, layer in enumerate(encoder):
    z = layer(z)
    print(f"{i:02d} {layer.__class__.__name__:12s} -> {tuple(z.shape)}")

print("\n해석:")
print("채널 수는 feature 종류가 늘면서 증가할 수 있고,")
print("공간 크기 H,W는 pooling/stride 때문에 감소한다.")

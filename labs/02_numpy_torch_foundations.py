import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


print("=== 1. NumPy image [H,W,C] ===")
image_hwc = np.zeros((32, 48, 3), dtype=np.uint8)
print("shape:", image_hwc.shape, "dtype:", image_hwc.dtype)

print("\n=== 2. transpose [H,W,C] -> [C,H,W] ===")
image_chw = image_hwc.transpose(2, 0, 1)
print("shape:", image_chw.shape)

print("\n=== 3. torch.Tensor ===")
x = torch.from_numpy(image_chw).float() / 255.0
print("shape:", x.shape, "dtype:", x.dtype, "device:", x.device)


class ToyDataset(Dataset):
    def __init__(self, n=20):
        self.x = torch.linspace(-1, 1, n).unsqueeze(1)
        self.y = 3 * self.x + 0.5

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


print("\n=== 4. Dataset / DataLoader ===")
dataset = ToyDataset()
loader = DataLoader(dataset, batch_size=5, shuffle=True)
xb, yb = next(iter(loader))
print("batch x:", xb.shape, "batch y:", yb.shape)

print("\n=== 5. nn.Module / backward / optimizer.step ===")
model = nn.Linear(1, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
criterion = nn.MSELoss()

before = model.weight.detach().clone()

pred = model(xb)
loss = criterion(pred, yb)

optimizer.zero_grad()
loss.backward()

print("loss:", loss.detach().item())
print("gradient:", model.weight.grad)

optimizer.step()
after = model.weight.detach().clone()

print("weight before:", before)
print("weight after :", after)
print("parameter changed:", not torch.equal(before, after))

print("\n핵심:")
print("Dataset = sample 하나를 읽는 규칙")
print("DataLoader = sample 여러 개를 batch로 공급")
print("backward = gradient 계산")
print("optimizer.step = gradient를 사용해 parameter 변경")

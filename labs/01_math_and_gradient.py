from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def f(x):
    return x**2 + 2*x + 1


print("=== 1. 함수 ===")
for x in [-2, -1, 0, 1, 2]:
    print(f"x={x:>2}, f(x)={f(x):>4}")

print("\n=== 2. 수치 미분 ===")
x0 = 3.0
eps = 1e-4
numerical_grad = (f(x0 + eps) - f(x0 - eps)) / (2 * eps)
analytic_grad = 2*x0 + 2
print("numerical gradient:", numerical_grad)
print("analytic gradient :", analytic_grad)

print("\n=== 3. PyTorch autograd ===")
x = torch.tensor(3.0, requires_grad=True)
y = x**2 + 2*x + 1
y.backward()
print("x.grad =", x.grad.item())

print("\n=== 4. Gradient Descent ===")
w = torch.tensor(8.0, requires_grad=True)
lr = 0.1
history = []

for step in range(25):
    loss = (w - 2.0) ** 2
    history.append((step, float(w.detach()), float(loss.detach())))

    loss.backward()

    with torch.no_grad():
        w -= lr * w.grad

    w.grad.zero_()

print("final w:", float(w.detach()))
print("target w: 2.0")

out = Path("outputs")
out.mkdir(exist_ok=True)

steps = [r[0] for r in history]
losses = [r[2] for r in history]

plt.figure()
plt.plot(steps, losses, marker="o")
plt.xlabel("step")
plt.ylabel("loss")
plt.title("Gradient Descent: loss decreases")
plt.tight_layout()
plt.savefig(out / "01_gradient_descent_loss.png", dpi=150)
plt.close()

print("saved:", out / "01_gradient_descent_loss.png")

import torch
import math
x = torch.randn(3, requires_grad=True)
print(x)
#z = x + 3
z = torch.sigmoid(x)
print(z)
v = torch.tensor([1, 1, 1], dtype=torch.float32)
z.backward(v)
print(x.grad)


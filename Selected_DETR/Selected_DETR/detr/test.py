import torch

t = torch.tensor([[1, 2], [3, 4]])
f = torch.gather(t, 0, torch.tensor([[0, 0], [1, 0]]))
print(f)
print(f.shape)

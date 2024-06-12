import torch

mat = torch.randn(2, 3, 2, 2)
mat_1 = torch.randn(2, 3, 4)
print(mat_1)
mat_x = torch.randn(2, 3, 1)
new_mat_x, indices = mat_x.sort(descending = True, dim = 1)
print(indices)

mask = torch.zeros(2,3,4)
#mask = torch.zeros(4,2,3)
b = indices + mask
n = b.long()
# b = a+mask
# n = b.long()

k = torch.gather(mat_1,1,n)

print(k)
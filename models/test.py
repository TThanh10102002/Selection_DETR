import torch

from torch import nn

src = torch.randn(2, 2048, 32, 32)

clone_src = torch.randn(2, 2048, 32, 32)

norm_clone_src = nn.LayerNorm([2048, 32, 32])(clone_src)
flatten_clone_src = norm_clone_src.flatten(2).permute(0, 1, 2)
flatten_feature_dim = flatten_clone_src.size(2)
print(flatten_feature_dim)
first_transform = nn.Linear(flatten_feature_dim, flatten_feature_dim // 2)(flatten_clone_src)
first_activation = nn.GELU()(first_transform)
second_transform = nn.Linear(flatten_feature_dim // 2, flatten_feature_dim // 4)(first_activation)
second_activation = nn.GELU()(second_transform)
third_transform = nn.Linear(flatten_feature_dim // 4, flatten_feature_dim // 8)(second_activation)
third_activation = nn.GELU()(third_transform)
final_transform = nn.Linear(flatten_feature_dim // 8, 1)(third_activation)
final_activation = nn.GELU()(final_transform)

sorted_order, sorted_indices = final_activation.sort(descending=True, dim = 1)

print(sorted_order.size())
print(sorted_indices.size())
print(src)
src = src[:,sorted_indices.flatten(),:,:]
print(src)

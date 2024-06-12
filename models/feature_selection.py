"""
Features Selection modules.
"""

import torch
import torch.nn.functional as F
import torchvision
from torch import nn


class Features_Selection(nn.Module):
    def __init__(self, selection_ratio):
        super().__init__()
        self.selection_ratio = selection_ratio
        

    def forward(self, src, pos_embed, sel_ratio):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        sel_ratio = self.selection_ratio
        #src = src.flatten(2).permute(2, 0, 1)
        bs, c, h, w = src.shape
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        modified_pos_embed = pos_embed.permute(1, 2, 0).to(device)
        bs, c, h, w = src.shape
        #print(c, h, w)
        clone_src = src.clone().detach()
        
        # Scoring features for selection
        norm_clone_src = nn.LayerNorm([c, h, w], device=device)(clone_src)
        flatten_clone_src = norm_clone_src.flatten(2).permute(0, 1, 2).to(device)
        flatten_feature_dim = flatten_clone_src.size(2)
        first_transform = nn.Linear(flatten_feature_dim, flatten_feature_dim // 2, device=device)(flatten_clone_src)
        first_activation = nn.GELU()(first_transform)
        second_transform = nn.Linear(flatten_feature_dim // 2, flatten_feature_dim // 4, device=device)(first_activation)
        second_activation = nn.GELU()(second_transform)
        third_transform = nn.Linear(flatten_feature_dim // 4, flatten_feature_dim // 8, device=device)(second_activation)
        third_activation = nn.GELU()(third_transform)
        final_transform = nn.Linear(flatten_feature_dim // 8, 1, device=device)(third_activation)
        final_activation = nn.GELU()(final_transform)
        
        #Sorting the features
        sorted_features, sorted_indices = final_activation.sort(descending=True, dim = 1)

        #Kept elements with the selection ratio
        unsqueeze_indices = torch.unsqueeze(sorted_indices, 2)
        mask_features = torch.zeros(bs, c, h, w).to(device)
        unsqueeze_mask_indices = torch.add(unsqueeze_indices, mask_features)
        long_unsqueeze_mask_indices = unsqueeze_mask_indices.long()
        transform_features = torch.gather(src, 1, long_unsqueeze_mask_indices)
        total_elements = transform_features.size(1)

        #Calculate the number of elements that are kept
        if sel_ratio == 1./3.:
            kept_elements = total_elements // 3
        if sel_ratio == 1./2.:
            kept_elements = total_elements // 2
            
        selected_features = transform_features[:,:kept_elements,:,:]
        
        last_pos_dim = modified_pos_embed.size(2)
        pos_mask = torch.zeros(bs, c, last_pos_dim).to(device)
        pos_mask_indices = torch.add(sorted_indices, pos_mask)
        long_pos_mask_indices = pos_mask_indices.long()
        transform_pos_embed = torch.gather(modified_pos_embed, 1, long_pos_mask_indices)
        selected_pos_embed = transform_pos_embed[:,:kept_elements,:]

        selected_pos_embed = selected_pos_embed.permute(2, 0, 1)
        # # x_local, x_global = torch.split(x, self.latent_dim, dim = -1)
        # # x_global = x_global.mean(dim=1, keepdim=True).expand(-1, x_local.shape[1], -1)
        # # x = torch.cat([x_local, x_global], dim=-1)
        # # Sorting and selecting the features in ratio
        # selection_scores_clone[mask] = -1.
        
        # if sel_ratio==None:
        #     sel_ratio = self.selection_ratio
            
        # selection_range = ((~mask).sum(1) * sel_ratio).int()
        # max_selection_num = selection_range.max()
        # mask_top = torch.arange(max_selection_num).expand(len(selection_range), max_selection_num).to(selection_range.device) > (selection_range-1).unsqueeze(1)
            
        # sorted_order = selection_scores_clone.sort(descending=True, dim = 1)[1]
        # top_scores = sorted_order[:,:max_selection_num]
        
        # selection_loss = selection_scores.gather(1, top_scores).mean()
        # sel_src = src.gather(0, top_scores.permute(1,0)[...,None].expand(-1, -1, c)) * selection_scores.gather(1,top_scores).permute(1,0).unsqueeze(-1)
        # sel_pos_embed = pos_embed.gather(0,top_scores.permute(1,0)[...,None].expand(-1, -1, c))
        # sel_mask = mask_top
        
        return selected_features #, selected_pos_embed
    
    
def build_feature_selection(args):
    selection_ratio = args.selection_ratio
    if selection_ratio:
        selection_features = Features_Selection(selection_ratio=selection_ratio)
    else:
        selection_features = None
    return selection_features
    
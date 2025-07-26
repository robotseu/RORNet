import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict

class CrossModalAttention(nn.Module):
    """ Cross-Modal Attention. Adapted from: https://github.com/openai/CLIP/blob/main/clip/model.py#L56 """

    def __init__(self, embed_dim=1024, num_heads=8, output_dim=1024):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim)

    def forward(self, q, k, v, attn_mask=None):
        x, attn_weights = F.multi_head_attention_forward(
            query=q, key=k, value=v,
            embed_dim_to_check=v.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0.,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            need_weights=True,
            attn_mask=attn_mask
        )

        return x, attn_weights


class Enhancer(nn.Module):
    def __init__(self):
        super(Enhancer, self).__init__()
        self.text_to_visual_att = CrossModalAttention(embed_dim=512, num_heads=8, output_dim=512)
        self.visual_to_text_att = CrossModalAttention(embed_dim=512, num_heads=8, output_dim=512)

    def forward(self, img_feat, text_feat):
        text_to_visual_feat, _ = self.text_to_visual_att(text_feat, img_feat, img_feat)
        visual_to_text_feat, _ = self.visual_to_text_att(img_feat, text_feat, text_feat)

        return text_to_visual_feat, visual_to_text_feat

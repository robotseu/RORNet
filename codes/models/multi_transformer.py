import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict


"""Subclass torch's LayerNorm to handle fp16."""
class LayerNorm(nn.LayerNorm):
    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class CrossResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = CrossModalAttention(embed_dim=d_model, num_heads=n_head, output_dim=d_model)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=q.dtype, device=q.device) if self.attn_mask is not None else None
        attn_output, attn_weights = self.attn(q=q, k=k, v=v, attn_mask=self.attn_mask)
        return attn_output, attn_weights

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        attn_output, attn_weights = self.attention(self.ln_1(q), self.ln_1(k), self.ln_1(v))
        q = q + attn_output
        q = q + self.mlp(self.ln_2(q))
        return q, attn_weights


class CrossTransformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.ModuleList([CrossResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        for i, _ in enumerate(self.resblocks):
            q, attn_weights = self.resblocks[i](q, k, v)

        q = q.permute(1, 0, 2) # L'ND -> NL'D
        return q, attn_weights


class CrossModalAttention(nn.Module):
    """ Cross-Modal Attention. Adapted from: https://github.com/openai/CLIP/blob/main/clip/model.py#L56 """

    def __init__(self, embed_dim=512, num_heads=8, output_dim=512):
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


class MultiModalFusion(nn.Module):
    def __init__(self):
        super(MultiModalFusion, self).__init__()
        self.text_to_visual_att = CrossModalAttention(embed_dim=512, num_heads=8, output_dim=512)
        self.visual_to_text_att = CrossModalAttention(embed_dim=512, num_heads=8, output_dim=512)
        self.MultiTransformer = CrossTransformer(width=1024, layers=4, heads=8)

    def forward(self, tac_feat, img_feat, text_feat, grip_feat):
        # language-visual Feature Enhancer
        text_to_visual_feat, _ = self.text_to_visual_att(text_feat, img_feat, img_feat)
        visual_to_text_feat, _ = self.visual_to_text_att(img_feat, text_feat, text_feat)

        # visual-tactile feature
        visual_text_feat = torch.cat([text_to_visual_feat, visual_to_text_feat], dim=-1)

        # tactile-gripper Feature Fusion
        tactile_grip_feat = torch.cat([tac_feat, grip_feat], dim=-1)

        # Cross-Modal Decoder
        visual_text_feat = visual_text_feat.unsqueeze(-2).float().permute(1, 0, 2)
        tactile_grip_feat = tactile_grip_feat.unsqueeze(-2).float().permute(1, 0, 2)
        fusion_feat, _ = self.MultiTransformer(tactile_grip_feat, visual_text_feat, visual_text_feat)

        return fusion_feat.squeeze(1)


if __name__ == '__main__':
    tac_feat = torch.randn(2, 512)
    img_feat = torch.randn(2, 512)
    text_feat = torch.randn(2, 512)
    grip_feat = torch.randn(2, 512)

    mul = MultiModalFusion()

    total_params = sum([param.nelement() for param in mul.parameters()])
    print(total_params / 1e6)

    out = mul(tac_feat, img_feat, text_feat, grip_feat)
    print(out.shape)
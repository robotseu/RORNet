import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttention(nn.Module):
    def __init__(self):
        super().__init__()

        self.q_proj = nn.Sequential(
            nn.Conv2d(1024, 1024, kernel_size=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1)
        )
        self.ConvFuse = nn.Sequential(
            nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1)
        )

        self.drop2d = nn.Dropout2d(0.3)

    def forward(self, visual_feat, tactile_feat):
        # visual_feat.shape: [N, 512]
        # tactile_feat.shape: [N, 512]

        visual_feat = visual_feat.unsqueeze(2)
        tactile_feat = tactile_feat.unsqueeze(2)

        B, C, _ = visual_feat.shape  # [N, 512, 1]
        h1_temp = visual_feat.view(B, C, -1)  # [N, 512, 1]
        h3_temp = tactile_feat.view(B, C, -1)  # [N, 512, 1]

        crossh1_h3 = h3_temp @ h1_temp.transpose(-2, -1)
        crossh1_h3 = F.softmax(crossh1_h3, dim=-1)
        crossedh1_h3 = (crossh1_h3 @ h1_temp).contiguous()
        crossedh1_h3 = crossedh1_h3.view(B, C, H, W)

        h_concat = visual_feat + crossedh1_h3
        out = self.ConvFuse(self.drop2d(h_concat))

        return out

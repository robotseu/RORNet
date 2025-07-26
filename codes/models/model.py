import torch
import torch.nn as nn
import yaml

from codes.models.head import ContrastiveHead, SimCLRHead, ContrastiveCLIPHead
from codes.models.encoder.tactile_encoder import TimeSFormer
from codes.models.encoder.transformer_encoder import TransformerEncoder
from codes.models.enhancer import Enhancer
from codes.models.encoder.vivit import VIVIT
from codes.models.encoder.cnn import Basic_CNN
from codes.models.encoder.tcn import ResNet_TCN

import codes.models.encoder.clip.clip as clip


class VTmlp(nn.Module):
    def __init__(self):
        super(VTmlp, self).__init__()

        self.proj_1 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.1)
        )

        self.proj_2 = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.1)
        )

    def forward(self, x):
        x = self.proj_1(x)
        x = self.proj_2(x)

        return x


class TacGrasp_Net(nn.Module):
    def __init__(self, cfg):
        super(TacGrasp_Net, self).__init__()

        self.clip_encoder, self.preprocess = clip.load_clip(cfg["model"]["clip_model_path"], device='cuda')
        self.transformer_encoder = TransformerEncoder(hidden_size=512, num_heads=8, num_layers=3, mlp_dim=512, dropout_rate=0.1, n_frames=8)
        # self.tactile_encoder = VIVIT()

        self.tactile_encoder = TimeSFormer((224, 224), (16, 16), in_chans=3, num_classes=0, embed_dim=512,
                                           depth=6, num_heads=8, mlp_ratio=4, qkv_bias=False, qk_scale=None, drop_rate=0.1,
                                           attn_drop_rate=0.1, drop_path_rate=0.1, hybrid_backbone=None, norm_layer=nn.LayerNorm,
                                           num_frames=8, attention_type='divided_space_time', dropout=0.1)

        # self.tactile_encoder = ResNet_TCN()

        # self.enhancer = Enhancer()
        self.vt_proj = VTmlp()
        self.head = ContrastiveCLIPHead()

        self.frozen_module("clip_encoder")

    def frozen_module(self, module):
        for name, param in self.named_parameters():
            if module in name:
                param.requires_grad = False

    def encode_features(self, text, image_seq, tac_seq):
        B, N, C, H, W = image_seq.shape
        image_seq_merge = image_seq.contiguous().view(B * N, C, H, W)

        with torch.no_grad():
            image_merge_feat = self.clip_encoder.encode_image(image_seq_merge)  # [B * N, 512]
            text_feat = self.clip_encoder.encode_text(torch.squeeze(text, dim=1))   # [B, 512]
            image_feat = image_merge_feat.view(B, N, -1)         # [B, N, 512]

            image_feat = self.transformer_encoder(image_feat)    # [B, 512]
            tactile_feat = self.tactile_encoder(tac_seq)         # [B, 512]

            image_feat = image_feat.float()
            text_feat = text_feat.float()
            tactile_feat = tactile_feat.float()

            vt_feat = self.vt_proj(torch.cat([image_feat, tactile_feat], dim=1))

        return text_feat, vt_feat

    def forward(self, text, image_seq, tac_seq):
        # text [B, 1, 77]
        # image_seq [B, 8, 3, 224, 224]
        # tac_seq [B, 3, 8, 224, 224]
        B, N, C, H, W = image_seq.shape
        image_seq_merge = image_seq.contiguous().view(B * N, C, H, W)

        with torch.no_grad():
            image_merge_feat = self.clip_encoder.encode_image(image_seq_merge)  # [B * N, 512]
            text_feat = self.clip_encoder.encode_text(torch.squeeze(text, dim=1))   # [B, 512]
        image_feat = image_merge_feat.view(B, N, -1)         # [B, N, 512]

        image_feat = self.transformer_encoder(image_feat)    # [B, 512]
        tactile_feat = self.tactile_encoder(tac_seq)         # [B, 512]

        image_feat = image_feat.float()
        text_feat = text_feat.float()
        tactile_feat = tactile_feat.float()

        # image_feat, text_feat = self.enhancer(image_feat, text_feat)

        vt_feat = self.vt_proj(torch.cat([image_feat, tactile_feat], dim=1))
        logits_per_image, logits_per_text, cur_loss = self.head(vt_feat, text_feat)

        return logits_per_image, logits_per_text, cur_loss


if __name__ == '__main__':

    with open('/home/zzy/pycharm/projects/vlt_grasp/tac-grasp-net/config.yaml', encoding='utf-8') as config_file:
        cfg = yaml.load(config_file, Loader=yaml.FullLoader)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    text = torch.rand(2, 1, 77).to(device)
    img_seq = torch.rand(2, 8, 3, 224, 224).to(device)
    tac_seq = torch.rand(2, 3, 8, 224, 224).to(device)

    model = TacGrasp_Net(cfg).cuda()
    total_params = sum([param.nelement() for param in model.parameters() if param.requires_grad])
    print(total_params / 1e6)

    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(name)

    loss = model(text, img_seq, tac_seq)

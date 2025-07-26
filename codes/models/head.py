import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np


def contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    return nn.functional.cross_entropy(logits, torch.arange(len(logits), device=logits.device))


def clip_loss(similarity: torch.Tensor) -> torch.Tensor:
    caption_loss = contrastive_loss(similarity)
    image_loss = contrastive_loss(similarity.t())
    return (caption_loss + image_loss) / 2.0


class ContrastiveCLIPHead(nn.Module):
    def __init__(self):
        super(ContrastiveCLIPHead, self).__init__()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, image_embeds, text_embeds):
        image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits_per_text = torch.matmul(text_embeds, image_embeds.t()) * logit_scale
        logits_per_image = logits_per_text.t()

        loss = clip_loss(logits_per_text)

        return logits_per_image, logits_per_text, loss


class ContrastiveHead(nn.Module):
    def __init__(self):
        super(ContrastiveHead, self).__init__()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def get_logits(self, visual_tactile_feat, text_feat):
        # logits_per_visual_sample @ text_features.T 相似度矩阵
        logits_per_visual_sample = self.logit_scale * visual_tactile_feat @ text_feat.T
        logits_per_text = self.logit_scale * text_feat @ visual_tactile_feat.T
        return logits_per_visual_sample, logits_per_text

    def forward(self, visual_feat, text_feat, tac_feat):
        visual_tactile_features = torch.add(visual_feat, tac_feat)
        logits_per_image, logits_per_text = self.get_logits(visual_tactile_features, text_feat)
        labels = torch.arange(logits_per_image.shape[0], dtype=torch.long).cuda()
        total_loss = (F.cross_entropy(logits_per_image, labels) + F.cross_entropy(logits_per_text, labels)) / 2

        return logits_per_image, logits_per_text, total_loss


class SimCLRHead(nn.Module):
    def __init__(self, batch_size, temperature=0.5):
        super().__init__()
        self.batch_size = batch_size
        self.register_buffer("temperature", torch.tensor(temperature))
        self.register_buffer("negatives_mask", (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool)).float())

    def forward(self, emb_i, emb_j):
        """
        emb_i and emb_j are batches of embeddings, where corresponding indices are pairs
        z_i, z_j as per SimCLR paper
        """
        z_i = F.normalize(emb_i, dim=1)
        z_j = F.normalize(emb_j, dim=1)
        #
        # representations = torch.cat([z_i, z_j], dim=0)
        # similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)
        #
        # sim_ij = torch.diag(similarity_matrix, self.batch_size)
        # sim_ji = torch.diag(similarity_matrix, -self.batch_size)
        # positives = torch.cat([sim_ij, sim_ji], dim=0)
        #
        # nominator = torch.exp(positives / self.temperature)
        # denominator = self.negatives_mask * torch.exp(similarity_matrix / self.temperature)
        #
        # loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))
        # loss = torch.sum(loss_partial) / (2 * self.batch_size)
        # return loss

        b, device = z_i.shape[0], z_i.device
        n = b * 2
        projs = torch.cat((z_i, z_j))
        logits = projs @ projs.t()

        mask = torch.eye(n, device=device).bool()
        logits = logits[~mask].reshape(n, n - 1)
        logits /= self.temperature

        labels = torch.cat(((torch.arange(b, device=device) + b - 1), torch.arange(b, device=device)), dim=0)
        loss = F.cross_entropy(logits, labels, reduction='sum')
        loss /= n
        return loss


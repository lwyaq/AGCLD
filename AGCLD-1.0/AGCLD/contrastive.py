"""
Contrastive learning modules for AGCLD.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


def info_nce_loss(
    anchor: torch.Tensor,
    positive: torch.Tensor,
    negatives: torch.Tensor = None,
    temperature: float = 0.2,
    num_negatives: int = 256,
) -> torch.Tensor:
    """InfoNCE contrastive loss with in-batch negatives."""
    anchor = F.normalize(anchor, dim=1)
    positive = F.normalize(positive, dim=1)
    n = anchor.size(0)
    pos_sim = (anchor * positive).sum(dim=1, keepdim=True) / temperature

    if negatives is not None:
        negatives = F.normalize(negatives, dim=1)
        neg_sim = torch.mm(anchor, negatives.t()) / temperature
    else:
        num_neg = min(num_negatives, n - 1)
        if num_neg <= 0:
            return -pos_sim.mean()
        neg_sim = torch.mm(anchor, positive.t()) / temperature
        mask = torch.eye(n, device=anchor.device, dtype=torch.bool)
        neg_sim = neg_sim.masked_fill(mask, float("-inf"))

    logits = torch.cat([pos_sim, neg_sim], dim=1)
    labels = torch.zeros(n, dtype=torch.long, device=anchor.device)
    loss = F.cross_entropy(logits, labels)
    return loss


def symmetric_info_nce(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.2, num_negatives: int = 256) -> torch.Tensor:
    """Symmetric InfoNCE loss."""
    loss1 = info_nce_loss(z1, z2, temperature=temperature, num_negatives=num_negatives)
    loss2 = info_nce_loss(z2, z1, temperature=temperature, num_negatives=num_negatives)
    return 0.5 * (loss1 + loss2)


class DualGraphContrastiveLoss(nn.Module):
    """Contrastive loss between spatial and expression views."""

    def __init__(self, embed_dim: int, proj_dim: int = 64, temperature: float = 0.2, num_negatives: int = 256):
        super().__init__()
        self.temperature = temperature
        self.num_negatives = num_negatives
        self.proj_spatial = nn.Sequential(nn.Linear(embed_dim, proj_dim), nn.ReLU(), nn.Linear(proj_dim, proj_dim))
        self.proj_expr = nn.Sequential(nn.Linear(embed_dim, proj_dim), nn.ReLU(), nn.Linear(proj_dim, proj_dim))

    def forward(self, z_spatial: torch.Tensor, z_expr: torch.Tensor) -> torch.Tensor:
        h_spatial = self.proj_spatial(z_spatial)
        h_expr = self.proj_expr(z_expr)
        return symmetric_info_nce(h_spatial, h_expr, temperature=self.temperature, num_negatives=self.num_negatives)


class DeepGraphInfomaxLoss(nn.Module):
    """Deep Graph Infomax loss for self-supervised learning."""

    def __init__(self, embed_dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(embed_dim, embed_dim))
        nn.init.xavier_uniform_(self.weight)

    def summary(self, z: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(z.mean(dim=0))

    def discriminate(self, z: torch.Tensor, summary: torch.Tensor) -> torch.Tensor:
        value = torch.matmul(z, torch.matmul(self.weight, summary))
        return torch.sigmoid(value)

    def forward(self, z_pos: torch.Tensor, z_neg: torch.Tensor) -> torch.Tensor:
        summary = self.summary(z_pos)

        # Use binary cross entropy with logits for better numerical stability
        pos_logits = torch.matmul(z_pos, torch.matmul(self.weight, summary))
        neg_logits = torch.matmul(z_neg, torch.matmul(self.weight, summary))

        pos_loss = F.binary_cross_entropy_with_logits(pos_logits, torch.ones_like(pos_logits))
        neg_loss = F.binary_cross_entropy_with_logits(neg_logits, torch.zeros_like(neg_logits))

        return pos_loss + neg_loss

"""
Graph Attention Network encoders for AGCLD.
"""

from __future__ import annotations
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class TwoLevelGATEncoder(nn.Module):
    """Two-level GAT encoder with residual connections."""

    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, heads: int = 4, dropout: float = 0.1, negative_slope: float = 0.2):
        super().__init__()
        self.heads = heads
        self.dropout = dropout
        self.gat1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout, negative_slope=negative_slope, concat=True)
        self.norm1 = nn.LayerNorm(hidden_channels * heads)
        self.gat2 = GATConv(hidden_channels * heads, out_channels, heads=heads, dropout=dropout, negative_slope=negative_slope, concat=False)
        self.norm2 = nn.LayerNorm(out_channels)
        if in_channels != out_channels:
            self.residual_proj = nn.Linear(in_channels, out_channels)
        else:
            self.residual_proj = nn.Identity()
        self.act = nn.ELU()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: Optional[torch.Tensor] = None, return_attention: bool = False):
        identity = self.residual_proj(x)
        if return_attention:
            h1, alpha1 = self.gat1(x, edge_index, edge_attr=edge_weight, return_attention_weights=True)
        else:
            h1 = self.gat1(x, edge_index, edge_attr=edge_weight)
            alpha1 = None
        h1 = self.norm1(h1)
        h1 = self.act(h1)
        h1 = F.dropout(h1, p=self.dropout, training=self.training)
        if return_attention:
            h2, alpha2 = self.gat2(h1, edge_index, edge_attr=edge_weight, return_attention_weights=True)
        else:
            h2 = self.gat2(h1, edge_index, edge_attr=edge_weight)
            alpha2 = None
        h2 = self.norm2(h2)
        h = h2 + identity
        h = self.act(h)
        if return_attention:
            return h, (alpha1, alpha2)
        return h, None


class DualGraphGATEncoder(nn.Module):
    """Dual-branch GAT encoder for spatial and expression graphs."""

    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.spatial_encoder = TwoLevelGATEncoder(in_channels, hidden_channels, out_channels, heads=heads, dropout=dropout)
        self.expr_encoder = TwoLevelGATEncoder(in_channels, hidden_channels, out_channels, heads=heads, dropout=dropout)
        self.fusion_mlp = nn.Sequential(nn.Linear(out_channels * 2, out_channels), nn.ReLU(), nn.Linear(out_channels, 2))
        self.temperature = nn.Parameter(torch.tensor(1.0))

    def forward(
        self,
        x: torch.Tensor,
        edge_index_spatial: torch.Tensor,
        edge_index_expr: torch.Tensor,
        edge_weight_spatial: Optional[torch.Tensor] = None,
        edge_weight_expr: Optional[torch.Tensor] = None,
    ):
        z_spatial, _ = self.spatial_encoder(x, edge_index_spatial, edge_weight_spatial)
        z_expr, _ = self.expr_encoder(x, edge_index_expr, edge_weight_expr)
        concat = torch.cat([z_spatial, z_expr], dim=1)
        logits = self.fusion_mlp(concat)
        weights = F.softmax(logits / self.temperature.clamp(min=0.1), dim=1)
        z_fused = weights[:, 0:1] * z_spatial + weights[:, 1:2] * z_expr
        return z_fused, z_spatial, z_expr, weights

"""
AGCLD Backbone: Dual-Graph GAT with DGG for adaptive neighborhoods.
"""

from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn

from .contrastive import DualGraphContrastiveLoss, DeepGraphInfomaxLoss
from .dgg import DifferentiableGraphGenerator
from .gat_encoder import DualGraphGATEncoder


class AGCLD_Backbone(nn.Module):
    """AGCLD Backbone with Differentiable Graph Generator (DGG)."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        embed_dim: int = 64,
        gat_heads: int = 4,
        dropout: float = 0.1,
        temperature: float = 0.2,
        num_negatives: int = 256,
        contrast_weight: float = 1.0,
        dgi_weight: float = 0.5,
        spatial_reg_weight: float = 0.1,
        dgg_k_min: int = 2,
        dgg_k_max: int = 15,
        dgg_hidden: int = 64,
        dgg_edge_tau: float = 0.5,
        dgg_selector_lambda: float = 1.0,
        dgg_straight_through: bool = True,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
        self.contrast_weight = contrast_weight
        self.dgi_weight = dgi_weight
        self.spatial_reg_weight = spatial_reg_weight
        self.dgg_k_max = dgg_k_max

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
            nn.Dropout(dropout),
        )

        self.dgg_spatial = DifferentiableGraphGenerator(
            node_dim=hidden_dim,
            hidden_dim=dgg_hidden,
            k_min=dgg_k_min,
            k_max=dgg_k_max,
            use_distance=True,
            edge_tau=dgg_edge_tau,
            selector_lambda=dgg_selector_lambda,
            straight_through=dgg_straight_through,
            normalize=True,
        )

        self.dgg_expr = DifferentiableGraphGenerator(
            node_dim=hidden_dim,
            hidden_dim=dgg_hidden,
            k_min=dgg_k_min,
            k_max=dgg_k_max,
            use_distance=True,
            edge_tau=dgg_edge_tau,
            selector_lambda=dgg_selector_lambda,
            straight_through=dgg_straight_through,
            normalize=True,
        )

        self.dual_gat = DualGraphGATEncoder(
            in_channels=hidden_dim,
            hidden_channels=hidden_dim,
            out_channels=embed_dim,
            heads=gat_heads,
            dropout=dropout,
        )

        self.contrastive_loss = DualGraphContrastiveLoss(
            embed_dim=embed_dim,
            proj_dim=embed_dim,
            temperature=temperature,
            num_negatives=num_negatives,
        )

        self.dgi_spatial = DeepGraphInfomaxLoss(embed_dim)
        self.dgi_expr = DeepGraphInfomaxLoss(embed_dim)

        self.output_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def corruption(self, x: torch.Tensor) -> torch.Tensor:
        return x[torch.randperm(x.size(0), device=x.device)]

    def forward(
        self,
        x: torch.Tensor,
        edge_index_spatial: torch.Tensor,
        edge_index_expr: torch.Tensor,
        edge_dist_spatial: torch.Tensor,
        edge_dist_expr: torch.Tensor,
        coords: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        h = self.input_proj(x)
        h_corrupt = self.corruption(h)

        dgg_out_spatial = self.dgg_spatial(h, edge_index_spatial, edge_dist_spatial)
        dgg_out_expr = self.dgg_expr(h, edge_index_expr, edge_dist_expr)

        z_fused, z_spatial, z_expr, fusion_weights = self.dual_gat(
            h,
            edge_index_spatial,
            edge_index_expr,
            dgg_out_spatial.edge_weight,
            dgg_out_expr.edge_weight,
        )

        z_spatial_neg, _ = self.dual_gat.spatial_encoder(
            h_corrupt,
            edge_index_spatial,
            dgg_out_spatial.edge_weight,
        )
        z_expr_neg, _ = self.dual_gat.expr_encoder(
            h_corrupt,
            edge_index_expr,
            dgg_out_expr.edge_weight,
        )

        contrast_loss = self.contrastive_loss(z_spatial, z_expr)

        dgi_loss_spatial = self.dgi_spatial(z_spatial, z_spatial_neg)
        dgi_loss_expr = self.dgi_expr(z_expr, z_expr_neg)
        dgi_loss = 0.5 * (dgi_loss_spatial + dgi_loss_expr)

        spatial_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        if self.spatial_reg_weight > 0:
            src, dst = edge_index_spatial
            z_out = self.output_proj(z_fused)
            diff = (z_out[src] - z_out[dst]).pow(2).sum(dim=1)
            spatial_loss = (dgg_out_spatial.edge_weight * diff).mean()

        total_loss = (
            self.contrast_weight * contrast_loss + self.dgi_weight * dgi_loss + self.spatial_reg_weight * spatial_loss
        )

        embedding = self.output_proj(z_fused)

        k_spatial_mean = dgg_out_spatial.k_cont.mean()
        k_expr_mean = dgg_out_expr.k_cont.mean()
        eff_deg_spatial = dgg_out_spatial.edge_gate.view(-1, self.dgg_k_max).sum(dim=1).mean()
        eff_deg_expr = dgg_out_expr.edge_gate.view(-1, self.dgg_k_max).sum(dim=1).mean()

        return {
            "loss": total_loss,
            "contrast_loss": contrast_loss,
            "dgi_loss": dgi_loss,
            "spatial_loss": spatial_loss,
            "embedding": embedding.detach(),
            "z_spatial": z_spatial.detach(),
            "z_expr": z_expr.detach(),
            "fusion_weights": fusion_weights.detach(),
            "dgg_k_spatial_mean": k_spatial_mean,
            "dgg_k_expr_mean": k_expr_mean,
            "dgg_eff_deg_spatial": eff_deg_spatial,
            "dgg_eff_deg_expr": eff_deg_expr,
            "dgg_k_spatial": dgg_out_spatial.k_cont.detach(),
            "dgg_k_expr": dgg_out_expr.k_cont.detach(),
        }

    @torch.no_grad()
    def get_embedding(
        self,
        x: torch.Tensor,
        edge_index_spatial: torch.Tensor,
        edge_index_expr: torch.Tensor,
        edge_dist_spatial: torch.Tensor,
        edge_dist_expr: torch.Tensor,
    ) -> torch.Tensor:
        self.eval()
        h = self.input_proj(x)

        dgg_out_spatial = self.dgg_spatial(h, edge_index_spatial, edge_dist_spatial)
        dgg_out_expr = self.dgg_expr(h, edge_index_expr, edge_dist_expr)

        z_fused, _, _, _ = self.dual_gat(
            h,
            edge_index_spatial,
            edge_index_expr,
            dgg_out_spatial.edge_weight,
            dgg_out_expr.edge_weight,
        )

        return self.output_proj(z_fused)

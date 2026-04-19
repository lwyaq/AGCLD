"""
AGCLD Model: Modality-specific DVAE with Cross-modal Attention Fusion.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import AGCLD_Backbone


@dataclass
class ReconOutput:
    """Output of reconstruction module."""

    x_recon: torch.Tensor
    mask: torch.Tensor
    loss: torch.Tensor
    kl_loss: torch.Tensor


class ModalityDVAE(nn.Module):
    """Modality-specific Denoising Variational Autoencoder."""

    def __init__(self, input_dim: int, hidden_dim: int = 256, latent_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim),
        )

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z)
        return x_recon, mu, logvar


class AGCLD_Model(nn.Module):
    """AGCLD: Spatial Protein RNA OmUlti-modal Transformer."""

    def __init__(
        self,
        input_dims: List[int],
        hidden_dim: int = 128,
        embed_dim: int = 64,
        gat_heads: int = 4,
        contrast_weight: float = 1.0,
        dgi_weight: float = 0.5,
        spatial_reg_weight: float = 0.1,
        dgg_k_min: int = 2,
        dgg_k_max: int = 15,
        dgg_hidden: int = 64,
        dgg_edge_tau: float = 0.5,
        dgg_selector_lambda: float = 1.0,
        dgg_straight_through: bool = True,
        recon_weight: float = 1.0,
        kl_weight: float = 0.001,
        recon_mask_prob: float = 0.15,
        recon_hidden_dim: int = 256,
        recon_latent_dim: int = 128,
        recon_dropout: float = 0.1,
    ):
        super().__init__()

        self.input_dims = input_dims
        self.num_modalities = len(input_dims)
        self.recon_weight = float(recon_weight)
        self.kl_weight = float(kl_weight)
        self.recon_mask_prob = float(recon_mask_prob)

        self.modality_dvaes = nn.ModuleList(
            [
                ModalityDVAE(
                    input_dim=dim,
                    hidden_dim=recon_hidden_dim,
                    latent_dim=recon_latent_dim,
                    dropout=recon_dropout,
                )
                for dim in input_dims
            ]
        )

        self.modality_projections = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(dim, hidden_dim // 2),
                    nn.LayerNorm(hidden_dim // 2),
                    nn.GELU(),
                    nn.Dropout(0.1),
                )
                for dim in input_dims
            ]
        )

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim // 2,
            num_heads=4,
            dropout=0.1,
            batch_first=True,
        )

        fusion_input_dim = hidden_dim // 2 + len(input_dims) * (hidden_dim // 2)
        self.fusion_layer = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )

        self.backbone = AGCLD_Backbone(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            embed_dim=embed_dim,
            gat_heads=gat_heads,
            contrast_weight=contrast_weight,
            dgi_weight=dgi_weight,
            spatial_reg_weight=spatial_reg_weight,
            dgg_k_min=dgg_k_min,
            dgg_k_max=dgg_k_max,
            dgg_hidden=dgg_hidden,
            dgg_edge_tau=dgg_edge_tau,
            dgg_selector_lambda=dgg_selector_lambda,
            dgg_straight_through=dgg_straight_through,
        )

    def _mask_input(self, x: torch.Tensor) -> torch.Tensor:
        """Generate mask for denoising."""
        if not self.training or self.recon_mask_prob <= 0:
            return torch.zeros_like(x, dtype=torch.bool)
        prob = torch.full_like(x, fill_value=self.recon_mask_prob)
        return torch.bernoulli(prob).to(dtype=torch.bool)

    def process_modalities(self, modality_features: List[torch.Tensor]) -> Tuple[torch.Tensor, ReconOutput]:
        """Process each modality independently through its DVAE."""
        processed_features = []
        total_recon_loss = 0.0
        total_kl_loss = 0.0
        all_recons = []
        all_masks = []

        for x_i, dvae, proj in zip(modality_features, self.modality_dvaes, self.modality_projections):
            mask_i = self._mask_input(x_i)
            x_corrupt_i = x_i.clone()
            x_corrupt_i = x_corrupt_i.masked_fill(mask_i, 0.0)

            x_recon_i, mu_i, logvar_i = dvae(x_corrupt_i)

            if mask_i.any():
                recon_loss_i = F.mse_loss(x_recon_i[mask_i], x_i[mask_i])
            else:
                recon_loss_i = F.mse_loss(x_recon_i, x_i)

            kl_loss_i = -0.5 * torch.mean(torch.sum(1 + logvar_i - mu_i.pow(2) - logvar_i.exp(), dim=1))

            processed_i = proj(x_recon_i)
            processed_features.append(processed_i)

            all_recons.append(x_recon_i)
            all_masks.append(mask_i)
            total_recon_loss += recon_loss_i
            total_kl_loss += kl_loss_i

        if len(processed_features) > 1:
            stacked_features = torch.stack(processed_features, dim=1)
            attended_features, _ = self.cross_attention(stacked_features, stacked_features, stacked_features)
            fused_features = attended_features.mean(dim=1)
        else:
            fused_features = processed_features[0]

        concat_features = torch.cat(processed_features, dim=1)
        final_features = self.fusion_layer(torch.cat([fused_features, concat_features], dim=1))

        x_recon_combined = torch.cat(all_recons, dim=1)
        mask_combined = torch.cat(all_masks, dim=1)

        recon_output = ReconOutput(
            x_recon=x_recon_combined,
            mask=mask_combined,
            loss=total_recon_loss / self.num_modalities,
            kl_loss=total_kl_loss / self.num_modalities,
        )

        return final_features, recon_output

    def forward(
        self,
        modality_features: List[torch.Tensor],
        edge_index_spatial: torch.Tensor,
        edge_index_expr: torch.Tensor,
        edge_dist_spatial: torch.Tensor,
        edge_dist_expr: torch.Tensor,
        coords: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        fused_features, recon_out = self.process_modalities(modality_features)

        backbone_out = self.backbone(
            x=fused_features,
            edge_index_spatial=edge_index_spatial,
            edge_index_expr=edge_index_expr,
            edge_dist_spatial=edge_dist_spatial,
            edge_dist_expr=edge_dist_expr,
            coords=coords,
        )

        total_loss = backbone_out["loss"] + self.recon_weight * recon_out.loss + self.kl_weight * recon_out.kl_loss

        return {
            **backbone_out,
            "loss": total_loss,
            "recon_loss": recon_out.loss,
            "kl_loss": recon_out.kl_loss,
        }

    @torch.no_grad()
    def get_embedding(
        self,
        modality_features: List[torch.Tensor],
        edge_index_spatial: torch.Tensor,
        edge_index_expr: torch.Tensor,
        edge_dist_spatial: torch.Tensor,
        edge_dist_expr: torch.Tensor,
    ) -> torch.Tensor:
        self.eval()
        fused_features, _ = self.process_modalities(modality_features)
        return self.backbone.get_embedding(
            fused_features,
            edge_index_spatial,
            edge_index_expr,
            edge_dist_spatial,
            edge_dist_expr,
        )

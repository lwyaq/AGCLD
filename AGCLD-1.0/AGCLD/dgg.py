"""
Differentiable Graph Generator (DGG) for adaptive neighborhood selection.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import torch
import torch.nn as nn


def _logit_noise_like(x: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    """Gumbel-like noise for differentiable sampling."""
    u = torch.rand_like(x).clamp(min=eps, max=1.0 - eps)
    return torch.log(u) - torch.log1p(-u)


@dataclass(frozen=True)
class DGGOutput:
    """Output of DGG forward pass."""

    edge_weight: torch.Tensor
    edge_gate: torch.Tensor
    edge_energy: torch.Tensor
    k_cont: torch.Tensor


class DifferentiableGraphGenerator(nn.Module):
    """DGG-style adaptive neighborhood selection on a candidate kNN graph."""

    def __init__(
        self,
        node_dim: int,
        hidden_dim: int = 64,
        k_min: int = 2,
        k_max: int = 15,
        use_distance: bool = True,
        edge_tau: float = 0.5,
        selector_lambda: float = 1.0,
        straight_through: bool = True,
        normalize: bool = True,
    ):
        super().__init__()
        self.node_dim = node_dim
        self.hidden_dim = hidden_dim
        self.k_min = k_min
        self.k_max = k_max
        self.use_distance = use_distance
        self.edge_tau = edge_tau
        self.selector_lambda = selector_lambda
        self.straight_through = straight_through
        self.normalize = normalize

        edge_in = node_dim * 2
        if use_distance:
            edge_in += 1

        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_in, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        self.degree_mlp = nn.Sequential(
            nn.Linear(node_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_dist: Optional[torch.Tensor] = None,
    ) -> DGGOutput:
        n = x.size(0)
        e = edge_index.size(1)

        if e != n * self.k_max:
            raise ValueError(f"Expected E = N * k_max = {n * self.k_max}, got E = {e}")

        if self.use_distance and edge_dist is None:
            raise ValueError("edge_dist required when use_distance=True")

        src = edge_index[0]
        dst = edge_index[1]

        expected_dst = torch.arange(n, device=edge_index.device).repeat_interleave(self.k_max)
        if not torch.equal(dst, expected_dst):
            raise ValueError("edge_index must be grouped by dst")

        src_mat = src.view(n, self.k_max)
        x_dst = x.unsqueeze(1).expand(n, self.k_max, x.size(1))
        x_src = x[src_mat]

        feats = [x_dst, x_src]
        if self.use_distance:
            feats.append(edge_dist.view(n, self.k_max, 1))

        edge_feat = torch.cat(feats, dim=-1)
        edge_logits = self.edge_mlp(edge_feat).squeeze(-1)
        noise = _logit_noise_like(edge_logits) if self.training else 0
        edge_energy = torch.sigmoid((edge_logits + noise) / self.edge_tau)

        degree_hint = edge_energy.sum(dim=1, keepdim=True)
        deg_input = torch.cat([x, degree_hint], dim=1)
        k_raw = self.degree_mlp(deg_input).squeeze(1)
        k_cont = self.k_min + (self.k_max - self.k_min) * torch.sigmoid(k_raw)

        energy_sorted, perm = torch.sort(edge_energy, dim=1, descending=True)
        ranks = torch.arange(1, self.k_max + 1, device=x.device, dtype=x.dtype).view(1, -1)
        gate_soft = torch.sigmoid((k_cont.view(-1, 1) - ranks) / self.selector_lambda)

        if self.straight_through:
            k_int = torch.clamp(torch.round(k_cont), min=self.k_min, max=self.k_max)
            gate_hard = (ranks <= k_int.view(-1, 1)).to(gate_soft.dtype)
            gate = gate_soft + (gate_hard - gate_soft).detach()
        else:
            gate = gate_soft

        gated_sorted = energy_sorted * gate
        inv_perm = perm.argsort(dim=1)
        edge_gate = torch.gather(gate, dim=1, index=inv_perm)
        edge_weight = torch.gather(gated_sorted, dim=1, index=inv_perm)

        if self.normalize:
            denom = edge_weight.sum(dim=1, keepdim=True).clamp(min=1e-12)
            edge_weight = edge_weight / denom

        return DGGOutput(
            edge_weight=edge_weight.reshape(-1),
            edge_gate=edge_gate.reshape(-1),
            edge_energy=edge_energy.reshape(-1),
            k_cont=k_cont,
        )

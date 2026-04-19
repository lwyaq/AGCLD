"""
AGCLD Trainer: Self-contained training utilities.
"""

from __future__ import annotations
from typing import Dict, Optional, Tuple, List
import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from .model import AGCLD_Model


def normalize_counts(
    counts: np.ndarray,
    size_factors: Optional[np.ndarray] = None,
    log_transform: bool = True,
    scale: bool = True,
) -> np.ndarray:
    """Normalize count matrix."""
    counts = counts.astype(np.float32)
    if size_factors is None:
        lib = counts.sum(axis=1, keepdims=True)
        median = np.median(lib[lib > 0]) if np.any(lib > 0) else 1.0
        size_factors = lib / (median + 1e-8)
    else:
        size_factors = size_factors.reshape(-1, 1)

    normalized = counts / (size_factors + 1e-8)
    if log_transform:
        normalized = np.log1p(normalized)
    if scale:
        mean = normalized.mean(axis=0, keepdims=True)
        std = normalized.std(axis=0, keepdims=True) + 1e-8
        normalized = (normalized - mean) / std
    return normalized


def build_knn_edge_index_grouped(
    features,
    k: int,
    group_by: str = "dst",
    device: Optional[torch.device] = None,
    return_dist: bool = True,
):
    """Build kNN graph with edges grouped by destination."""
    if isinstance(features, torch.Tensor):
        feats_np = features.detach().cpu().numpy()
    else:
        feats_np = np.asarray(features, dtype=np.float32)

    if np.any(np.isnan(feats_np)) or np.any(np.isinf(feats_np)):
        print("Warning: Found NaN/Inf in features for kNN graph building, cleaning...")
        feats_np = np.nan_to_num(feats_np, nan=0.0, posinf=0.0, neginf=0.0)

    n = feats_np.shape[0]
    if k >= n:
        raise ValueError(f"k ({k}) must be < N ({n})")

    nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
    nn.fit(feats_np)
    dists_np, idx_np = nn.kneighbors(feats_np)

    indices = idx_np[:, 1:]
    dists = dists_np[:, 1:]

    if group_by == "dst":
        dst = np.repeat(np.arange(n), k)
        src = indices.reshape(-1)
    else:
        src = np.repeat(np.arange(n), k)
        dst = indices.reshape(-1)

    edge_index = torch.tensor(np.stack([src, dst], axis=0), dtype=torch.long, device=device)

    if return_dist:
        edge_dist = torch.tensor(dists.reshape(-1), dtype=torch.float32, device=device)
        return edge_index, edge_dist
    return edge_index, None


class AGCLD_Trainer:
    """Trainer for AGCLD model."""

    def __init__(
        self,
        model: AGCLD_Model,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        rebuild_expr_every: int = 50,
        recon_schedule: Optional[Dict[str, float]] = None,
    ):
        self.model = model
        self.device = next(model.parameters()).device
        self.k_max = model.backbone.dgg_k_max
        self.base_recon_weight = float(model.recon_weight)

        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.rebuild_expr_every = rebuild_expr_every
        self.recon_schedule = recon_schedule or {}
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=20, verbose=True
        )

    def _build_graphs(self, coords: torch.Tensor, features: torch.Tensor):
        edge_spatial, dist_spatial = build_knn_edge_index_grouped(
            coords, k=self.k_max, group_by="dst", device=self.device, return_dist=True
        )
        edge_expr, dist_expr = build_knn_edge_index_grouped(
            features, k=self.k_max, group_by="dst", device=self.device, return_dist=True
        )
        return edge_spatial, dist_spatial, edge_expr, dist_expr

    def _prepare_modality_features(self, modality_data: List[np.ndarray]) -> List[torch.Tensor]:
        modality_tensors = []
        for idx, data in enumerate(modality_data):
            data = data.copy().astype(np.float32)

            is_reduced_feature = data.shape[1] < 200 and np.abs(data.mean()) < 10

            if not is_reduced_feature and data.max() > 100:
                data = normalize_counts(data, log_transform=True, scale=True)

            data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

            tensor_cpu = torch.tensor(data, dtype=torch.float32)

            if torch.isnan(tensor_cpu).any() or torch.isinf(tensor_cpu).any():
                print(f"Warning: Modality {idx} has NaN/Inf after CPU conversion, cleaning...")
                tensor_cpu = torch.nan_to_num(tensor_cpu, nan=0.0, posinf=0.0, neginf=0.0)

            tensor = tensor_cpu.to(self.device)

            if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                print(f"Warning: Modality {idx} has NaN/Inf after GPU conversion, cleaning...")
                tensor = torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)

            modality_tensors.append(tensor)
        return modality_tensors

    def fit(
        self,
        coords: np.ndarray,
        modality_data: List[np.ndarray],
        epochs: int = 500,
        log_interval: int = 20,
        patience: int = 50,
        min_epochs: int = 100,
    ) -> Dict[str, list]:
        coords_t = torch.tensor(coords, dtype=torch.float32, device=self.device)
        modality_tensors = self._prepare_modality_features(modality_data)

        concat_features = torch.cat(modality_tensors, dim=1)
        edge_spatial, dist_spatial, edge_expr, dist_expr = self._build_graphs(coords_t, concat_features)

        history = {
            "loss": [],
            "contrast_loss": [],
            "dgi_loss": [],
            "spatial_loss": [],
            "recon_loss": [],
            "kl_loss": [],
            "recon_weight": [],
            "k_spatial_mean": [],
            "k_expr_mean": [],
            "eff_deg_spatial": [],
            "eff_deg_expr": [],
        }

        best_loss = float("inf")
        patience_counter = 0
        best_state = None
        self.model.train()

        for epoch in range(1, epochs + 1):
            scheduled_weight = self._get_recon_weight(epoch)
            if scheduled_weight is not None:
                self.model.recon_weight = scheduled_weight

            self.optimizer.zero_grad()
            out = self.model(
                modality_features=modality_tensors,
                edge_index_spatial=edge_spatial,
                edge_index_expr=edge_expr,
                edge_dist_spatial=dist_spatial,
                edge_dist_expr=dist_expr,
                coords=coords_t,
            )

            loss = out["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            history["loss"].append(float(loss.item()))
            history["contrast_loss"].append(float(out["contrast_loss"].item()))
            history["dgi_loss"].append(float(out["dgi_loss"].item()))
            history["spatial_loss"].append(float(out["spatial_loss"].item()))
            history["recon_loss"].append(float(out["recon_loss"].item()))
            history["kl_loss"].append(float(out["kl_loss"].item()))
            history["recon_weight"].append(float(self.model.recon_weight))
            history["k_spatial_mean"].append(float(out["dgg_k_spatial_mean"].item()))
            history["k_expr_mean"].append(float(out["dgg_k_expr_mean"].item()))
            history["eff_deg_spatial"].append(float(out["dgg_eff_deg_spatial"].item()))
            history["eff_deg_expr"].append(float(out["dgg_eff_deg_expr"].item()))

            self.scheduler.step(loss)

            if epoch >= min_epochs:
                if loss.item() < best_loss:
                    best_loss = loss.item()
                    patience_counter = 0
                    best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"Early stopping at epoch {epoch}")
                        break

            if self.rebuild_expr_every > 0 and epoch % self.rebuild_expr_every == 0:
                with torch.no_grad():
                    embedding = self.model.get_embedding(modality_tensors, edge_spatial, edge_expr, dist_spatial, dist_expr)
                edge_expr, dist_expr = build_knn_edge_index_grouped(
                    embedding, k=self.k_max, group_by="dst", device=self.device, return_dist=True
                )
                if log_interval:
                    print(f"[Epoch {epoch}] Rebuilt expression graph")

            if log_interval and epoch % log_interval == 0:
                print(
                    f"[Epoch {epoch}/{epochs}] "
                    f"loss={loss.item():.4f} "
                    f"recon={out['recon_loss'].item():.4f} "
                    f"kl={out['kl_loss'].item():.4f} "
                    f"contrast={out['contrast_loss'].item():.4f} "
                    f"dgi={out['dgi_loss'].item():.4f} "
                    f"k_sp={out['dgg_k_spatial_mean'].item():.2f} "
                    f"k_ex={out['dgg_k_expr_mean'].item():.2f}"
                )

        if best_state is not None:
            self.model.load_state_dict({k: v.to(self.device) for k, v in best_state.items()})
            print(f"Loaded best model with loss={best_loss:.4f}")

        return history

    def _get_recon_weight(self, epoch: int) -> Optional[float]:
        if not self.recon_schedule:
            return None
        start = float(self.recon_schedule.get("start", self.base_recon_weight))
        end = float(self.recon_schedule.get("end", start))
        warmup_epochs = max(1, int(self.recon_schedule.get("warmup_epochs", 1)))
        if epoch >= warmup_epochs:
            return end
        ratio = epoch / warmup_epochs
        return start + (end - start) * ratio

    @torch.no_grad()
    def get_embedding(self, coords: np.ndarray, modality_data: List[np.ndarray]) -> np.ndarray:
        self.model.eval()
        coords_t = torch.tensor(coords, dtype=torch.float32, device=self.device)
        modality_tensors = self._prepare_modality_features(modality_data)

        concat_features = torch.cat(modality_tensors, dim=1)
        edge_spatial, dist_spatial, edge_expr, dist_expr = self._build_graphs(coords_t, concat_features)

        embedding = self.model.get_embedding(modality_tensors, edge_spatial, edge_expr, dist_spatial, dist_expr)
        return embedding.cpu().numpy()

    def save_model(self, path: str):
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")

    def load_model(self, path: str):
        state_dict = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        print(f"Model loaded from {path}")

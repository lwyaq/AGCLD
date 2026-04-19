"""
Utility functions for AGCLD model.
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors


def normalize_counts(
    X: np.ndarray,
    target_sum: float = 1e4,
    log_transform: bool = True,
    scale: bool = True,
) -> np.ndarray:
    """Normalize count matrix."""
    X = np.asarray(X, dtype=np.float32)

    row_sums = X.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums > 0, row_sums, 1.0)
    X = X / row_sums * target_sum

    if log_transform:
        X = np.log1p(X)

    if scale:
        mean = X.mean(axis=0, keepdims=True)
        std = X.std(axis=0, keepdims=True)
        std = np.where(std > 0, std, 1.0)
        X = (X - mean) / std

    return X


def build_knn_edge_index_grouped(
    features: Union[np.ndarray, torch.Tensor],
    k: int,
    group_by: str = "dst",
    device: Optional[torch.device] = None,
    return_dist: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Build kNN graph edge index grouped by destination node."""
    if group_by != "dst":
        raise ValueError("group_by must be 'dst' for DGG compatibility")

    if isinstance(features, torch.Tensor):
        features = features.detach().cpu().numpy()

    features = np.asarray(features, dtype=np.float64)
    n = features.shape[0]

    nbrs = NearestNeighbors(n_neighbors=min(k + 1, n), algorithm="auto")
    nbrs.fit(features)
    distances, indices = nbrs.kneighbors(features)

    neighbors = indices[:, 1 : k + 1]
    dists = distances[:, 1 : k + 1]

    if neighbors.shape[1] < k:
        pad_width = k - neighbors.shape[1]
        neighbors = np.pad(neighbors, ((0, 0), (0, pad_width)), mode="edge")
        dists = np.pad(dists, ((0, 0), (0, pad_width)), mode="edge")

    src = neighbors.flatten()
    dst = np.repeat(np.arange(n), k)

    edge_index = np.stack([src, dst], axis=0)
    edge_dist = dists.flatten()

    if device is None:
        device = torch.device("cpu")

    edge_index = torch.tensor(edge_index, dtype=torch.long, device=device)

    if return_dist:
        edge_dist = torch.tensor(edge_dist, dtype=torch.float32, device=device)
        return edge_index, edge_dist

    return edge_index


def compute_morans_i(
    cluster_labels: np.ndarray,
    spatial_coords: np.ndarray,
    k: int = 6,
) -> float:
    """Compute Moran's I for spatial autocorrelation of cluster labels."""
    cluster_labels = np.asarray(cluster_labels)
    spatial_coords = np.asarray(spatial_coords, dtype=np.float64)

    unique_labels = sorted(set(cluster_labels.tolist()))
    if len(unique_labels) <= 1:
        return 0.0

    label_to_num = {label: i for i, label in enumerate(unique_labels)}
    x = np.array([label_to_num[label] for label in cluster_labels], dtype=np.float64)

    n = len(x)
    nbrs = NearestNeighbors(n_neighbors=min(k + 1, n), algorithm="auto").fit(spatial_coords)
    distances, indices = nbrs.kneighbors(spatial_coords)

    neigh = indices[:, 1:]
    dist = distances[:, 1:]
    w = np.where(dist > 0, 1.0 / dist, 0.0)
    row_sum = w.sum(axis=1, keepdims=True)
    w = np.divide(w, row_sum, out=np.zeros_like(w), where=row_sum > 0)
    s0 = float(w.sum())

    if s0 == 0:
        return 0.0

    x_mean = float(x.mean())
    x_centered = x - x_mean
    denom = float(np.sum(x_centered**2))

    if denom == 0:
        return 0.0

    neigh_vals = x_centered[neigh]
    num = float(np.sum(x_centered * np.sum(w * neigh_vals, axis=1)))

    return float((n / s0) * (num / denom))

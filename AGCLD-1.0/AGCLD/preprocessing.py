"""
Preprocessing utilities for AGCLD model.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import anndata
import scanpy as sc
from sklearn.decomposition import PCA
import scipy
from scipy.sparse import coo_matrix
import scipy.sparse
import sklearn.utils.extmath
import sklearn.preprocessing


def pca(
    adata: sc.AnnData,
    n_comps: int = 50,
    random_state: int = 42,
) -> np.ndarray:
    """Perform PCA on AnnData object."""
    X = adata.X
    if hasattr(X, "toarray"):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float32)

    n_comps = min(n_comps, X.shape[1] - 1, X.shape[0] - 1)
    if n_comps < 1:
        return X

    pca_model = PCA(n_components=n_comps, random_state=random_state)
    return pca_model.fit_transform(X).astype(np.float32)


def clustering(
    adata: sc.AnnData,
    key: str = "AGCLD",
    add_key: str = "AGCLD_cluster",
    n_clusters: int = 7,
    method: str = "mclust",
    random_state: int = 42,
) -> None:
    """Perform clustering on embeddings."""
    embedding = adata.obsm[key]

    if method == "mclust":
        try:
            import rpy2.robjects as ro
            from rpy2.robjects import numpy2ri
            from rpy2.robjects.packages import importr

            numpy2ri.activate()
            mclust = importr("mclust")

            ro.r.assign("data", embedding)
            ro.r.assign("G", n_clusters)
            ro.r(
                """
                set.seed(42)
                fit <- Mclust(data, G=G)
                labels <- fit$classification
            """
            )
            labels = np.array(ro.r["labels"], dtype=int) - 1
            numpy2ri.deactivate()

        except Exception as e:
            print(f"mclust failed: {e}, falling back to kmeans")
            from sklearn.cluster import KMeans

            labels = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10).fit_predict(embedding)
    else:
        from sklearn.cluster import KMeans

        labels = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10).fit_predict(embedding)

    adata.obs[add_key] = pd.Categorical(labels.astype(str))


def extract_coords(adata: sc.AnnData) -> np.ndarray:
    """Extract spatial coordinates from AnnData."""
    candidate_keys = ["spatial", "coords", "X_spatial", "X_umap", "X_pca"]
    for key in candidate_keys:
        if key in adata.obsm:
            arr = np.asarray(adata.obsm[key])
            if arr.ndim == 2 and arr.shape[1] >= 2:
                return arr[:, :2].astype(np.float32)

    cols = [c.lower() for c in adata.obs.columns]
    possible_pairs = [
        ("x", "y"),
        ("row", "col"),
        ("imagerow", "imagecol"),
        ("x_pos", "y_pos"),
        ("array_row", "array_col"),
    ]
    for x_name, y_name in possible_pairs:
        if x_name in cols and y_name in cols:
            x_col = adata.obs.columns[cols.index(x_name)]
            y_col = adata.obs.columns[cols.index(y_name)]
            return np.stack([adata.obs[x_col].values, adata.obs[y_col].values], axis=1).astype(np.float32)

    raise ValueError("Cannot find spatial coordinates in adata.obs/obsm")


def to_dense(X) -> np.ndarray:
    """Convert sparse matrix to dense array."""
    if hasattr(X, "toarray"):
        return X.toarray()
    return np.asarray(X)


def lsi(
    adata: anndata.AnnData,
    n_components: int = 20,
    use_highly_variable: Optional[bool] = None,
    **kwargs,
) -> None:
    r"""LSI analysis (following the Seurat v3 approach)."""
    if use_highly_variable is None:
        use_highly_variable = "highly_variable" in adata.var
    adata_use = adata[:, adata.var["highly_variable"]] if use_highly_variable else adata
    X = tfidf(adata_use.X)
    X_norm = sklearn.preprocessing.Normalizer(norm="l1").fit_transform(X)
    X_norm = np.log1p(X_norm * 1e4)
    X_lsi = sklearn.utils.extmath.randomized_svd(X_norm, n_components, **kwargs)[0]
    X_lsi -= X_lsi.mean(axis=1, keepdims=True)
    X_lsi /= X_lsi.std(axis=1, ddof=1, keepdims=True)
    adata.obsm["X_lsi"] = X_lsi[:, 1:]


def tfidf(X):
    r"""TF-IDF normalization (following the Seurat v3 approach)."""
    idf = X.shape[0] / X.sum(axis=0)
    if scipy.sparse.issparse(X):
        tf = X.multiply(1 / X.sum(axis=1))
        return tf.multiply(idf)
    else:
        tf = X / X.sum(axis=1, keepdims=True)
        return tf * idf

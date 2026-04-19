import warnings
from pathlib import Path
import gc
import sys

import h5py
import numpy as np
import pandas as pd
import scanpy as sc
import torch

from sklearn.cluster import KMeans

warnings.filterwarnings('ignore')
random_seed = 43
np.random.seed(random_seed)
torch.manual_seed(random_seed)

from AGCLD import AGCLD_Model, AGCLD_Trainer
from AGCLD.preprocessing import pca, clustering
from AGCLD.utils import compute_morans_i

DATA_PATH = r"data\\Mouse_E15_brain.h5"
f = h5py.File(DATA_PATH, "r")

X_gene = np.array(f['raw_mapped_gene_count'], dtype=np.float32)
X_atac = np.array(f['raw_peak_mat'], dtype=np.float32)
pos = np.array(f['pos'], dtype=np.float32)
Y = np.array(f['Y'], dtype=str)

gene_names = [g.decode('utf-8') for g in f['mapped_gene']]
peak_names = [p.decode('utf-8') for p in f['peak']]

f.close()

n_cells = X_gene.shape[0]
cell_ids = [f'cell_{i}' for i in range(n_cells)]

adata_rna = sc.AnnData(X_gene)
adata_rna.obs_names = cell_ids
adata_rna.var_names = gene_names
adata_rna.obsm['spatial'] = pos
adata_rna.obs['LayerName'] = Y

adata_atac = sc.AnnData(X_atac)
adata_atac.obs_names = cell_ids
adata_atac.var_names = peak_names
adata_atac.obsm['spatial'] = pos
adata_atac.obs['LayerName'] = Y

print('RNA:', adata_rna)
print('ATAC:', adata_atac)
print('Spatial pos:', pos.shape)

import scipy
from AGCLD.preprocessing import pca, lsi, extract_coords
from sklearn.preprocessing import QuantileTransformer

sc.pp.filter_genes(adata_rna, min_cells=10)
sc.pp.normalize_total(adata_rna, target_sum=1e4)
sc.pp.log1p(adata_rna)
sc.pp.highly_variable_genes(adata_rna, flavor='seurat_v3', n_top_genes=3000)
sc.pp.scale(adata_rna)
adata_rna_high = adata_rna[:, adata_rna.var['highly_variable']]
rna_features = pca(adata_rna_high, n_comps=50)

adata_atac = adata_atac[adata_rna.obs_names].copy()
if 'X_lsi' not in adata_atac.obsm.keys():
    sc.pp.highly_variable_genes(adata_atac, flavor="seurat_v3", n_top_genes=3000)
    lsi(adata_atac, use_highly_variable=False, n_components=51)

adata_atac.obsm['feat'] = adata_atac.obsm['X_lsi'].copy()

atac_features = adata_atac.obsm['feat']

coords = extract_coords(adata_atac)

modality_data = [rna_features, atac_features]
input_dims = [rna_features.shape[1], atac_features.shape[1]]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')
print(f'RNA features: {rna_features.shape}')
print(f'ATAC features: {atac_features.shape}')
print(f'Input dims: {input_dims}')

model = AGCLD_Model(
    input_dims=input_dims,
    hidden_dim=128,
    embed_dim=64,
    gat_heads=2,
    contrast_weight=1.0,
    dgi_weight=1.5,
    spatial_reg_weight=0.5,
    dgg_k_min=2,
    dgg_k_max=15,
    dgg_hidden=64,
    dgg_edge_tau=1.5,
    dgg_selector_lambda=0.1,
    dgg_straight_through=False,
    recon_weight=0.5,
    kl_weight=0.1,
    recon_mask_prob=0.05,
    recon_hidden_dim=128,
    recon_latent_dim=64,
).to(device)

trainer = AGCLD_Trainer(
    model=model,
    lr=1e-3,
    weight_decay=1e-5,
    rebuild_expr_every=50,
)

print(f'Model parameters: {sum(p.numel() for p in model.parameters()):,}')

history = trainer.fit(
    coords=coords,
    modality_data=modality_data,
    epochs=6000,
    log_interval=50,
    patience=50,
    min_epochs=120,
)

from sklearn.metrics import (
    adjusted_rand_score,
    adjusted_mutual_info_score,
    completeness_score,
    homogeneity_score,
    normalized_mutual_info_score,
    v_measure_score,
)

embedding = trainer.get_embedding(coords, modality_data)

adata = adata_atac.copy()
adata.obsm['AGCLD'] = embedding

n_clusters = 7

try:
    clustering(adata, key='AGCLD', add_key='AGCLD_cluster', n_clusters=n_clusters, method='mclust')
    used_method = 'mclust'
except Exception as e:
    print(f'mclust failed: {e}, using kmeans')
    labels = KMeans(n_clusters=n_clusters, random_state=random_seed, n_init=10).fit_predict(embedding)
    adata.obs['AGCLD_cluster'] = pd.Categorical(labels.astype(str))
    used_method = 'kmeans'

y_true = adata.obs['LayerName'].astype(str).values
y_pred = adata.obs['AGCLD_cluster'].astype(str).values

ari = adjusted_rand_score(y_true, y_pred)
nmi = normalized_mutual_info_score(y_true, y_pred, average_method='arithmetic')
ami = adjusted_mutual_info_score(y_true, y_pred)
homogeneity = homogeneity_score(y_true, y_pred)
completeness = completeness_score(y_true, y_pred)
v_measure = v_measure_score(y_true, y_pred)

print("\n聚类评估指标:")
print(f"Adjusted Rand Index (ARI): {ari:.4f}")
print(f"Normalized Mutual Information (NMI): {nmi:.4f}")
print(f"Adjusted Mutual Information (AMI): {ami:.4f}")
print(f"Homogeneity : {homogeneity:.4f}")
print(f"Completeness : {completeness:.4f}")
print(f"V_measure : {v_measure:.4f}")

adata.write('ours_E15.5_embedding.h5ad')

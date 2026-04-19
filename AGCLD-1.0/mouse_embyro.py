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
from sklearn.metrics import silhouette_score, davies_bouldin_score

warnings.filterwarnings('ignore')
random_seed = 42
np.random.seed(random_seed)
torch.manual_seed(random_seed)

PROJECT_ROOT = Path.cwd()
if PROJECT_ROOT.name == 'AGCLD':
    PROJECT_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))

from AGCLD import AGCLD_Model, AGCLD_Trainer
from AGCLD.preprocessing import pca, clustering
from AGCLD.utils import compute_morans_i

print('AGCLD loaded successfully!')
print(f'Project root: {PROJECT_ROOT}')

DATA_PATH = PROJECT_ROOT / 'data' / 'DBiT_Seq Mouse Embryo_RNA_Protein.h5'
OUTPUT_DIR = PROJECT_ROOT / 'AGCLD_Results' / 'mouse_embryo'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if not DATA_PATH.exists():
    raise FileNotFoundError(f'Data file not found: {DATA_PATH}')

print(f'Data: {DATA_PATH}')
print(f'Output: {OUTPUT_DIR}')

with h5py.File(DATA_PATH, 'r') as f:
    X_gene = np.array(f['X_gene'], dtype=np.float32)
    X_prot = np.array(f['X_protein'], dtype=np.float32)
    loc = np.array(f['pos'], dtype=np.float32)
    gene_names = [g.decode('utf-8') for g in list(f['gene'])]
    protein_names = [p.decode('utf-8').split('.')[0] for p in list(f['protein'])]

    n_cells = X_gene.shape[0]
    cell_ids = [f'cell_{i}' for i in range(n_cells)]

adata_rna = sc.AnnData(X_gene)
adata_rna.obs_names = cell_ids
adata_rna.var_names = gene_names
adata_rna.obsm['spatial'] = loc

adata_prot = sc.AnnData(X_prot)
adata_prot.obs_names = cell_ids
adata_prot.var_names = protein_names
adata_prot.obsm['spatial'] = loc

print('RNA:', adata_rna)
print('Protein:', adata_prot)
print('Spatial:', loc.shape)

adata_omics1 = adata_rna.copy()
adata_omics2 = adata_prot.copy()

print('Preprocessing RNA data...')
sc.pp.normalize_total(adata_omics1, target_sum=1e4)
sc.pp.log1p(adata_omics1)

print(f"NaN values in RNA data: {np.isnan(adata_omics1.X).sum()}")
if np.isnan(adata_omics1.X).sum() > 0:
    print('Filling NaN values with 0...')
    adata_omics1.X[np.isnan(adata_omics1.X)] = 0

adata_omics1.obsm['feat'] = pca(adata_omics1, n_comps=adata_omics2.n_vars)

print('Preprocessing protein data...')
sc.pp.log1p(adata_omics2)

print(f"NaN values in protein data: {np.isnan(adata_omics2.X).sum()}")
if np.isnan(adata_omics2.X).sum() > 0:
    print('Filling NaN values with 0...')
    adata_omics2.X[np.isnan(adata_omics2.X)] = 0

adata_omics2.obsm['feat'] = pca(adata_omics2, n_comps=adata_omics2.n_vars)

print(f"RNA data shape: {adata_omics1.shape}")
print(f"Protein data shape: {adata_omics2.shape}")
print(f"RNA features shape: {adata_omics1.obsm['feat'].shape}")
print(f"Protein features shape: {adata_omics2.obsm['feat'].shape}")

modality_data = [adata_omics1.obsm['feat'], adata_omics2.obsm['feat']]
input_dims = [adata_omics1.obsm['feat'].shape[1], adata_omics2.obsm['feat'].shape[1]]
coords = loc[:, :2].astype(np.float32)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')
print(f'RNA features: {adata_omics1.obsm["feat"].shape}')
print(f'Protein features: {adata_omics2.obsm["feat"].shape}')
print(f'Input dims: {input_dims}')
print(f'Coords: {coords.shape}')

model = AGCLD_Model(
    input_dims=input_dims,
    hidden_dim=128,
    embed_dim=64,
    gat_heads=2,
    contrast_weight=1.5,
    dgi_weight=0.8,
    spatial_reg_weight=0.2,
    dgg_k_min=2,
    dgg_k_max=15,
    dgg_hidden=64,
    dgg_edge_tau=0.5,
    dgg_selector_lambda=1.0,
    dgg_straight_through=True,
    recon_weight=2.0,
    kl_weight=0.01,
    recon_mask_prob=0.30,
    recon_hidden_dim=128,
    recon_latent_dim=64,
).to(device)

trainer = AGCLD_Trainer(
    model=model,
    lr=1e-3,
    weight_decay=1e-3,
    rebuild_expr_every=50,
)

print(f'Model parameters: {sum(p.numel() for p in model.parameters()):,}')

history = trainer.fit(
    coords=coords,
    modality_data=modality_data,
    epochs=1000,
    log_interval=50,
    patience=80,
    min_epochs=120,
)

embedding = trainer.get_embedding(coords, modality_data)

adata = adata_prot.copy()
adata.obsm['AGCLD'] = embedding

n_clusters = 6

try:
    clustering(adata, key='AGCLD', add_key='AGCLD_cluster', n_clusters=n_clusters, method='mclust')
    used_method = 'mclust'
except Exception as e:
    print(f'mclust failed: {e}, using kmeans')
    labels = KMeans(n_clusters=n_clusters, random_state=random_seed, n_init=10).fit_predict(embedding)
    adata.obs['AGCLD_cluster'] = pd.Categorical(labels.astype(str))
    used_method = 'kmeans'

cluster_labels = adata.obs['AGCLD_cluster'].astype(str).values
labels_int = pd.factorize(cluster_labels)[0]

sc_score = silhouette_score(embedding, labels_int)
db_score = davies_bouldin_score(embedding, labels_int)
morans = compute_morans_i(cluster_labels, coords, k=6)

print(f'Clustering method: {used_method}')
print(f'Number of clusters: {n_clusters}')
print(f'Silhouette Score: {sc_score:.4f}')
print(f'Davies-Bouldin Score: {db_score:.4f}')
print(f"Moran's I: {morans:.4f}")

out_path = OUTPUT_DIR / 'agcld_mouse_embyro_embedding.h5ad'
adata.write(out_path)

del trainer, model
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

print('\nAnalysis complete!')

import warnings
from pathlib import Path
import gc
import sys

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
from AGCLD.preprocessing import pca, clustering, extract_coords, to_dense
from AGCLD.utils import compute_morans_i

print('AGCLD loaded successfully!')
print(f'Project root: {PROJECT_ROOT}')

DATA_DIR = Path('E:/lyt/DVAE/data/human_lynode')
OUTPUT_DIR = Path('AGCLD_Results')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

rna_path = DATA_DIR / 'adata_RNA_with_annotation.h5ad'
adt_path = DATA_DIR / 'adata_ADT_with_annotation.h5ad'

if not rna_path.exists() or not adt_path.exists():
    print(f'Data files not found in {DATA_DIR}')
    print('Please ensure adata_RNA.h5ad and adata_ADT.h5ad exist')
else:
    print('Data files found!')
    print(f'RNA: {rna_path}')
    print(f'ADT: {adt_path}')

adata_rna = sc.read_h5ad(rna_path)
adata_prot = sc.read_h5ad(adt_path)

adata_rna.var_names_make_unique()
adata_prot.var_names_make_unique()

print('RNA:', adata_rna)
print('Protein:', adata_prot)

sc.pp.filter_genes(adata_rna, min_cells=10)
sc.pp.highly_variable_genes(adata_rna, flavor='seurat_v3', n_top_genes=3000)
sc.pp.normalize_total(adata_rna, target_sum=1e4)
sc.pp.log1p(adata_rna)
sc.pp.scale(adata_rna)
adata_rna_high = adata_rna[:, adata_rna.var['highly_variable']]
rna_features = pca(adata_rna_high, n_comps=30)

sc.pp.normalize_total(adata_prot, target_sum=1e4)
sc.pp.log1p(adata_prot)
sc.pp.scale(adata_prot)
n_prot_pca = min(50, adata_prot.n_vars - 1)
prot_features = pca(adata_prot, n_comps=n_prot_pca)

coords = extract_coords(adata_prot)

modality_data = [rna_features, prot_features]
input_dims = [rna_features.shape[1], prot_features.shape[1]]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')
print(f'RNA features: {rna_features.shape}')
print(f'Protein features: {prot_features.shape}')
print(f'Input dims: {input_dims}')

model = AGCLD_Model(
    input_dims=input_dims,
    hidden_dim=128,
    embed_dim=128,
    gat_heads=2,
    contrast_weight=1.5,
    dgi_weight=0.8,
    spatial_reg_weight=0.2,
    dgg_k_min=4,
    dgg_k_max=15,
    dgg_hidden=64,
    dgg_edge_tau=0.5,
    dgg_selector_lambda=1.0,
    dgg_straight_through=True,
    recon_weight=2.0,
    kl_weight=0.01,
    recon_mask_prob=0.50,
    recon_hidden_dim=128,
    recon_latent_dim=128,
).to(device)

trainer = AGCLD_Trainer(
    model=model,
    lr=1e-3,
    weight_decay=1e-5,
    rebuild_expr_every=0,
)

print(f'Model parameters: {sum(p.numel() for p in model.parameters()):,}')

history = trainer.fit(
    coords=coords,
    modality_data=modality_data,
    epochs=1000,
    log_interval=50,
    patience=50,
    min_epochs=120,
)

embedding = trainer.get_embedding(coords, modality_data)

adata = adata_prot.copy()
adata.obsm['AGCLD'] = embedding

n_clusters = 10

try:
    clustering(adata, key='AGCLD', add_key='AGCLD_cluster', n_clusters=n_clusters, method='mclust')
    used_method = 'mclust'
except Exception as e:
    print(f'mclust failed: {e}, using kmeans')
    labels = KMeans(n_clusters=n_clusters, random_state=random_seed, n_init=10).fit_predict(embedding)
    adata.obs['AGCLD_cluster'] = pd.Categorical(labels.astype(str))
    used_method = 'kmeans'

out_path = OUTPUT_DIR / 'agcld_human_lynode_embedding.h5ad'
adata.write(out_path)

del trainer, model
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

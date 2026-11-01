import scanpy as sc
import squidpy as sq
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib.colors as mcolors
import h5py
import warnings
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# 1. 加载数据
# ---------------------------------------------------------------------------
file_path = 'Mouse_E15_brain.h5'

with h5py.File(file_path, 'r') as f:
    X = f['raw_mapped_gene_count'][:].astype(np.float32)
    pos = f['pos'][:]
    Y = f['Y'][:]
    gene_names = f['mapped_gene'][:].astype(str)
    Y_str = np.array([y.decode('utf-8').strip() for y in Y])

adata = sc.AnnData(X=X)
adata.var_names = gene_names
adata.obsm["spatial"] = pos
adata.obs["cell type"] = pd.Categorical(Y_str)

sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

# ---------------------------------------------------------------------------
# 2. 配受体分析
# ---------------------------------------------------------------------------
if "cell type_ligrec" not in adata.uns:
    sq.gr.ligrec(
        adata, 
        cluster_key="cell type", 
        use_raw=False,
        n_perms=1000
    )

means = adata.uns["cell type_ligrec"]["means"]
tidy = means.stack(list(range(means.columns.nlevels))).reset_index()
tidy.columns = ["ligand", "receptor", "source", "target", "value"]

# ---------------------------------------------------------------------------
# 3. 35个信号
# ---------------------------------------------------------------------------
top_n = 35
df = (
    tidy.groupby(["source", "target", "ligand", "receptor"], as_index=False)["value"]
    .mean()
    .sort_values("value", ascending=False)
    .head(top_n)
)

full_cell_types = ["Cartilage", "Forebrain", "Hindbrain", "Mesenchyme", "Midbrain", "Muscle", "Skull"]
for ct in full_cell_types:
    if ct not in df["source"].unique():
        missing = tidy[tidy["source"]==ct].sort_values("value", ascending=False).head(1)
        df = pd.concat([df, missing], ignore_index=True)

df["signal"] = df["ligand"] + "/" + df["receptor"]

# ---------------------------------------------------------------------------
# 4. 颜色
# ---------------------------------------------------------------------------
cell_color_map = {
    "Cartilage": "#70EE65",
    "Forebrain": "#FD7D56",
    "Hindbrain": "#5ED9EC",
    "Mesenchyme": "#2BB8F9",
    "Midbrain": "#9269DC",
    "Muscle": "#FDB74F",
    "Skull": "#E4F73C"
}

# ---------------------------------------------------------------------------
# 5. 绘图
# ---------------------------------------------------------------------------
source_nodes = sorted(df["source"].unique())
signal_nodes = sorted(df["signal"].unique())
target_nodes = sorted(df["target"].unique())

labels = source_nodes + signal_nodes + target_nodes
node_colors = []

for s in source_nodes: node_colors.append(cell_color_map.get(s, "#d9d9d9"))
for s in signal_nodes: node_colors.append("#ecf0f1")
for t in target_nodes: node_colors.append(cell_color_map.get(t, "#d9d9d9"))

n_src, n_sig = len(source_nodes), len(signal_nodes)
idx_map = {}
for i, s in enumerate(source_nodes): idx_map[f"src_{s}"] = i
for i, s in enumerate(signal_nodes): idx_map[f"sig_{s}"] = n_src + i
for i, t in enumerate(target_nodes): idx_map[f"tgt_{t}"] = n_src + n_sig + i

def safe_to_rgba(c, alpha=0.25):
    try:
        rgba = mcolors.to_rgba(c, alpha)
        return f"rgba({int(rgba[0]*255)},{int(rgba[1]*255)},{int(rgba[2]*255)},{rgba[3]})"
    except:
        return "rgba(200,200,200,0.25)"

links_src, links_tgt, link_vals, link_colors = [], [], [], []
for _, row in df.iterrows():
    c = cell_color_map.get(row["source"], "#ccc")
    col = safe_to_rgba(c)
    links_src.append(idx_map[f"src_{row['source']}"])
    links_tgt.append(idx_map[f"sig_{row['signal']}"])
    link_vals.append(row["value"])
    link_colors.append(col)
    links_src.append(idx_map[f"sig_{row['signal']}"])
    links_tgt.append(idx_map[f"tgt_{row['target']}"])
    link_vals.append(row["value"])
    link_colors.append(col)

fig = go.Figure(go.Sankey(
    arrangement="snap",
    node=dict(pad=28, thickness=18, color=node_colors, label=[f"<b>{l}</b>" for l in labels]),
    link=dict(source=links_src, target=links_tgt, value=link_vals, color=link_colors)
))

fig.update_layout(
    font=dict(size=28, family="Arial"),
    width=1600, height=900, template="plotly_white"
)

# ----------------------------
# 🔥 唯一 100% 成功保存方式
# ----------------------------
fig.write_html("sankey_figure.html")

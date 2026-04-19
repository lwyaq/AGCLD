# AGCLD

This project provides an implementation of **AGCLD**, a deep learning framework for spatial multi-omics integration. The model integrates multiple modalities (e.g., RNA + Protein / RNA + ATAC) through modality-specific denoising VAEs, cross-modality attention fusion, and dual-graph graph attention networks with an adaptive differentiable graph generator.

## AGCLD Installation Steps

1. Clone the repository

```bash
git clone <[your-repo-url](https://github.com/lwyaq/AGCLD.git)>
```

2. Enter the project directory

```bash
cd AGCLD-1.0
```

3. Install dependencies

```bash
pip install -r requirements.txt
```

## Usage

Run one of the provided scripts:

```bash
python human_lynode_D1.py
python human_tonsil.py
python mouse_15_5.py
python mouse_embyro.py
python test_human_lynode.py
```

Outputs will be written under:

- `AGCLD_Results/`

## Project Structure

- `AGCLD/`
- `human_lynode_D1.py`
- `human_tonsil.py`
- `mouse_15_5.py`
- `mouse_embyro.py`
- `test_human_lynode.py`
- `data/`

## Notes

- Some scripts expect datasets under `data/` (see each script for the exact filename/path).
- GPU is used automatically if CUDA is available.

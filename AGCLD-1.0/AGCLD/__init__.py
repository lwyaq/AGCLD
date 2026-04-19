"""
AGCLD: Spatial Protein RNA OmUlti-modal Transformer

A deep learning framework for spatial multi-omics integration using:
1. Modality-specific Denoising Variational Autoencoders (DVAE)
2. Cross-modality attention fusion
3. Dual-graph Graph Attention Networks (GAT)
4. Differentiable Graph Generator (DGG) for adaptive neighborhoods
5. Contrastive learning for cross-view alignment
"""

from .model import AGCLD_Model
from .trainer import AGCLD_Trainer

__version__ = "1.0.0"
__all__ = ["AGCLD_Model", "AGCLD_Trainer"]

"""Reusable model-independent plots for lens, attention, and causal results.

Install ``vlmprobe[visualization]`` to draw figures. Importing this module
does not import matplotlib, NumPy, or Pillow. All functions return matplotlib
Axes, except the two path-diagram functions which return Figure,
``plot_attention_comparison`` which returns (Figure, axes), and
``coordinate_grid`` which returns (array, row_labels, column_labels).

Figures preserve native coordinates and measured values. Caller-supplied image
geometry and labels make the same API usable with different models/datasets.
"""
from .lenses import plot_lens_heatmap
from .attention import plot_attention_overlay, plot_attention_comparison, plot_input
from .causal import (coordinate_grid, plot_causal_heatmap,
                     plot_intervention_curves, plot_connections)

__all__ = ["plot_lens_heatmap", "plot_attention_overlay", "plot_attention_comparison",
           "plot_input", "coordinate_grid", "plot_causal_heatmap",
           "plot_intervention_curves", "plot_connections"]

from .path_graphs import plot_head_paths, plot_receiver_paths
__all__ += ["plot_head_paths", "plot_receiver_paths"]
from .sae import plot_sae_feature_overlay, plot_sae_activity
__all__ += ["plot_sae_feature_overlay", "plot_sae_activity"]
from .mj import plot_mj_codebook, plot_mj_modality_index
__all__ += ["plot_mj_codebook", "plot_mj_modality_index"]
from .mj_pretrained import plot_mj_visual_readout, plot_mj_text_comparison
__all__ += ["plot_mj_visual_readout", "plot_mj_text_comparison"]

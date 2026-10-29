"""Model-bound lens, attention, causal and sparse autoencoder analysis."""
from .core import BaseMethod, ForwardTrace, ProbeResult
from .adapters import BaseModelAdapter, HookPoint, ModelReadout, TorchModelAdapter
from .adapters import CapabilityError, HeadSite, ModelSpec, ProbeInputs, TokenLayout, register_adapter
from .prober import Prober
from .api.lenses import LensMethods
from .api.attention import AttentionMethods
from .api.causal import CausalMethods
from .api.sae import SAEMethods
from .metrics import TokenMargin, SequenceLogProb, ClassScore
from .geometry import visual_grid, preprocessing_report
from .api.mj import MJBatch, fit_visual_readout, decode_mj_codes
from . import visualization, evaluation

__version__ = "1.0.3.dev0"
__all__ = ["BaseMethod", "ProbeResult", "ForwardTrace", "BaseModelAdapter",
           "HookPoint", "ModelReadout", "TorchModelAdapter", "Prober", "ModelSpec",
           "ProbeInputs", "TokenLayout", "HeadSite", "CapabilityError", "register_adapter",
           "LensMethods", "AttentionMethods", "CausalMethods", "SAEMethods", "TokenMargin", "SequenceLogProb", "ClassScore",
           "visualization", "evaluation", "visual_grid", "preprocessing_report",
           "MJBatch", "fit_visual_readout", "decode_mj_codes"]

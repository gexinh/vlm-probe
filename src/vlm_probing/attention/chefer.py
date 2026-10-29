"""Chefer CVPR 2021 aggregation of genuine propagated attention relevance.

The first input is an LRP attention CAM, not an attention probability matrix.
Producing that input requires a model-specific relevance propagation backend.
"""

import torch
from torch import Tensor

from ..core.types import ProbeResult
from ..core.precision import analysis_dtype
from .base import BaseAttention


class CheferTransformerAttribution(BaseAttention):
    """Official positive CAM-gradient head averaging and residual rollout."""

    name = "chefer_transformer_attribution"

    def __init__(self, *, start_layer: int = 0):
        if type(start_layer) is not int or start_layer < 0:
            raise ValueError("start_layer must be a nonnegative integer")
        self.start_layer = start_layer

    def run(self, attention_relevance: Tensor, gradients: Tensor) -> ProbeResult:
        self._floating(attention_relevance, "LRP attention CAMs [L,B,H,T,T]", 5)
        self._floating(gradients, "attention gradients [L,B,H,T,T]", 5)
        if (attention_relevance.shape != gradients.shape
                or attention_relevance.device != gradients.device
                or attention_relevance.dtype != gradients.dtype):
            raise ValueError("CAMs and gradients must match shape, device, and dtype")
        if attention_relevance.shape[-1] != attention_relevance.shape[-2]:
            raise ValueError("the CVPR aggregation requires square self-attention CAMs")
        if self.start_layer >= len(attention_relevance):
            raise ValueError("start_layer must select an existing layer")
        if not (torch.isfinite(attention_relevance).all() and torch.isfinite(gradients).all()):
            raise ValueError("CAMs and gradients must be finite")
        dtype = analysis_dtype(attention_relevance, gradients)
        attention_relevance, gradients = attention_relevance.to(dtype), gradients.to(dtype)
        weighted = (attention_relevance * gradients).clamp_min(0).mean(2)
        identity = torch.eye(weighted.shape[-1], device=weighted.device, dtype=weighted.dtype)
        relevance = identity.expand(weighted.shape[1], -1, -1)
        layers = []
        for layer in weighted[self.start_layer:]:
            relevance = (identity + layer) @ relevance
            layers.append(relevance)
        if not torch.isfinite(relevance).all():
            raise ValueError("relevance aggregation overflowed; use a wider dtype")
        return ProbeResult(self.name,
                           {"relevance": relevance,
                            "layer_relevance": torch.stack(layers),
                            "weighted_attention_relevance": weighted},
                           {"paper": "https://arxiv.org/abs/2012.09838",
                            "analysis_dtype": str(dtype),
                            "input_semantics": "propagated_LRP_attention_CAMs",
                            "row_normalized": False, "start_layer": self.start_layer,
                            "scope": "aggregation only; caller supplies genuine LRP CAMs"})

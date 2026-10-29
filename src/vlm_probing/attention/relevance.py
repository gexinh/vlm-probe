"""An explicitly scoped gradient-weighted self-attention propagation variant."""

import torch
from torch import Tensor

from ..core.types import ProbeResult
from ..core.precision import analysis_dtype
from .base import BaseAttention


class AttentionRelevance(BaseAttention):
    """Positive-gradient self-attention relevance: C=mean_h(relu(A*dS/dA)).

    Start with R=I and update R <- R + C @ R in forward layer order, without
    probability normalization. This implements the self-attention recurrence
    used in generic attention explainability (https://arxiv.org/abs/2103.15679),
    not the full Chefer multimodal/cross-attention algorithm or layerwise LRP.
    Gradients must come from the same forward pass and an explicit target S.
    """

    name = "attention_relevance"

    def run(self, attention: Tensor, gradients: Tensor, *,
            valid_tokens: Tensor | None = None) -> ProbeResult:
        mask = self._stack(attention, valid_tokens)
        self._floating(gradients, "gradients", 5)
        if (gradients.shape != attention.shape or gradients.device != attention.device
                or gradients.dtype != attention.dtype):
            raise ValueError("gradients must match the attention shape, device, and dtype")
        if not torch.isfinite(gradients).all():
            raise ValueError("attention gradients must be finite")
        dtype = analysis_dtype(attention, gradients)
        attention, gradients = attention.to(dtype), gradients.to(dtype)
        weighted = (attention * gradients).clamp_min(0).mean(2)
        pair_mask = mask[:, :, None] & mask[:, None, :]
        weighted = weighted.masked_fill(~pair_mask[None], 0)
        relevance = torch.eye(attention.shape[-1], device=attention.device, dtype=attention.dtype)
        relevance = relevance[None] * mask[:, :, None]
        layers = []
        for layer in weighted:
            relevance = relevance + layer @ relevance
            layers.append(relevance)
        if not torch.isfinite(relevance).all():
            raise ValueError("relevance propagation overflowed; use a wider dtype")
        return ProbeResult(method="attention_relevance",
                           tensors={"relevance": relevance,
                                    "layer_relevance": torch.stack(layers),
                                    "weighted_attention": weighted},
                           metadata={"variant": "positive_gradient_self_attention_residual",
                                     "analysis_dtype": str(dtype),
                                     "row_normalized": False, "identity_included": True})

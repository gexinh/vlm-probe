"""Attention rollout with explicit residual and head aggregation conventions."""

import torch
from torch import Tensor

from ..core.types import ProbeResult
from ..core.precision import analysis_dtype
from .base import BaseAttention


class AttentionRollout(BaseAttention):
    """Compose normalized attention matrices in forward layer order.

    The default averages heads, adds identity, then normalizes each row, as in
    Abnar and Zuidema (2020): https://arxiv.org/abs/2005.00928.
    This descriptive propagation does not include value vectors or MLPs.
    """

    name = "attention_rollout"

    def __init__(self, *, head_reduction: str = "mean", residual: bool = True):
        if head_reduction not in {"mean", "max", "min"}:
            raise ValueError("head_reduction must be 'mean', 'max', or 'min'")
        self.head_reduction = head_reduction
        self.residual = residual

    def run(self, attention: Tensor, *, valid_tokens: Tensor | None = None) -> ProbeResult:
        """Input is [layer,batch,head,token,token], earliest layer first."""
        mask = self._stack(attention, valid_tokens)
        dtype = analysis_dtype(attention)
        attention = attention.to(dtype)
        identity = torch.eye(attention.shape[-1], device=attention.device, dtype=attention.dtype)
        identity = identity[None] * mask[:, :, None]
        pair_mask = mask[:, :, None] & mask[:, None, :]
        rollout = identity
        layers = []
        for layer in attention:
            if self.head_reduction == "mean":
                fused = layer.mean(1)
            elif self.head_reduction == "max":
                fused = layer.amax(1)
            else:
                fused = layer.amin(1)
            fused = fused.masked_fill(~pair_mask, 0)
            if self.residual:
                fused = fused + identity
            denominator = fused.sum(-1, keepdim=True)
            if (denominator.squeeze(-1)[mask] <= 0).any():
                raise ValueError("head reduction produced an empty valid row")
            transition = fused / denominator.masked_fill(~mask[:, :, None], 1)
            rollout = transition @ rollout
            layers.append(rollout)
        return ProbeResult(method="attention_rollout",
                           tensors={"rollout": rollout, "layer_rollouts": torch.stack(layers)},
                           metadata={"head_reduction": self.head_reduction,
                                     "residual": self.residual, "row_normalized": True,
                                     "analysis_dtype": str(dtype),
                                     "layer_order": "earliest_first"})

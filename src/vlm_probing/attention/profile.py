"""Descriptive attention statistics; these do not establish causal influence."""

from collections.abc import Mapping

import torch
from torch import Tensor

from ..core.types import ProbeResult
from ..core.precision import analysis_dtype
from .base import BaseAttention


class AttentionProfile(BaseAttention):
    """Measure entropy, concentration, and probability mass by token modality."""

    name = "attention_profile"

    def run(self, attention: Tensor, *, groups: Mapping[str, Tensor] | None = None,
            query_mask: Tensor | None = None,
            key_mask: Tensor | None = None) -> ProbeResult:
        """Group masks are disjoint [B,K]; ungrouped keys are permitted.

        Invalid query statistics are zero and explicitly identified by the
        returned ``valid_queries``. Entropy is measured in natural-log units.
        """
        qmask, kmask = self._probabilities(attention, query_mask, key_mask)
        dtype = analysis_dtype(attention)
        attention = attention.to(dtype)
        valid = qmask[:, None, :]
        log_prob = attention.clamp_min(torch.finfo(attention.dtype).tiny).log()
        tensors = {
            "entropy": (-(attention * log_prob).sum(-1)).masked_fill(~valid, 0),
            "concentration": attention.square().sum(-1).masked_fill(~valid, 0),
            "max_probability": attention.amax(-1).masked_fill(~valid, 0),
            "valid_queries": qmask,
        }
        names = list(groups or {})
        if names:
            masks = []
            covered = torch.zeros_like(kmask)
            for name, group in groups.items():
                if not isinstance(name, str) or not name:
                    raise ValueError("group names must be nonempty strings")
                group = self._mask(group, tuple(kmask.shape), attention, name)
                if (group & ~kmask).any() or (group & covered).any():
                    raise ValueError("modality groups must be valid, disjoint key masks")
                covered = covered | group
                masks.append(group)
            stacked = torch.stack(masks, dim=-1).to(attention.dtype)
            tensors["group_mass"] = torch.einsum("bhqk,bkg->bhqg", attention, stacked)
            tensors["group_mass"] = tensors["group_mass"].masked_fill(~valid[..., None], 0)
        return ProbeResult(method="attention_profile", tensors=tensors,
                           metadata={"groups": names, "entropy_units": "nats",
                                     "analysis_dtype": str(dtype),
                                     "statistic_axes": ["batch", "head", "query"]})

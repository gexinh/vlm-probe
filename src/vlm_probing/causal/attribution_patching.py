"""First-order estimate of an explicitly directed activation replacement."""

import torch
from torch import Tensor

from ..core.types import ProbeResult
from .base import BaseCausal, Runner, scalar_metric, selection_mask, validate_pair


class AttributionPatching(BaseCausal):
    """Compute ``(source - receiver) * grad(metric, receiver)``.

    Supply either a precomputed gradient evaluated at the receiver, or a scalar
    differentiable metric that replays computation from the receiver activation.
    Scores approximate source-into-receiver metric change; they are not exact
    patch effects. Positive values predict an increase in the caller's metric.
    """

    name = "attribution_patching"

    def run(
        self,
        receiver: Tensor,
        source: Tensor,
        *,
        gradient: Tensor | None = None,
        metric: Runner | None = None,
        mask: Tensor | None = None,
    ) -> ProbeResult:
        validate_pair(receiver, source)
        if (gradient is None) == (metric is None):
            raise ValueError("supply exactly one of gradient or metric")
        if metric is not None:
            with torch.enable_grad():
                point = receiver.detach().clone().requires_grad_(True)
                gradient = torch.autograd.grad(scalar_metric(metric, point), point)[0]
        validate_pair(receiver, gradient)
        selected = selection_mask(receiver, mask)
        # Difference, products and their sum use at least FP32 (FP64 is
        # preserved); reduced-precision rounding would bias the estimate.
        dtype = torch.promote_types(torch.float32, receiver.dtype)
        receiver_wide = receiver.to(dtype)
        delta = torch.where(selected, source.to(dtype) - receiver_wide, torch.zeros_like(receiver_wide))
        attribution = delta * gradient.to(dtype)
        return ProbeResult(
            method=self.name,
            tensors={"attribution": attribution, "estimated_effect": attribution.sum(),
                     "gradient": gradient, "delta": delta, "mask": selected},
            metadata={"approximation": "first_order", "gradient_at": "receiver",
                      "sign": "source_metric - receiver_metric", "analysis_dtype": str(dtype)},
        )

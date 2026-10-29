"""Shared contract and validation for causal tensor interventions."""

from collections.abc import Callable

import torch
from torch import Tensor

from ..core.base import BaseMethod
from ..core.types import ProbeResult
from ..core.precision import analysis_difference

Runner = Callable[[Tensor], Tensor]


def validate_activation(value: Tensor, name: str) -> None:
    if not isinstance(value, Tensor) or not value.is_floating_point():
        raise TypeError(f"{name} must be a floating-point torch.Tensor")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} must contain finite values")


def validate_pair(receiver: Tensor, source: Tensor) -> None:
    validate_activation(receiver, "receiver")
    validate_activation(source, "source")
    if source.shape != receiver.shape:
        raise ValueError("source and receiver must have exactly the same shape")
    if source.device != receiver.device or source.dtype != receiver.dtype:
        raise ValueError("source and receiver must share a device and dtype")


def selection_mask(value: Tensor, mask: Tensor | None) -> Tensor:
    """True selects entries; e.g. [B,T,1] selects tokens of [B,T,D]."""
    if mask is None:
        return torch.ones_like(value, dtype=torch.bool)
    if not isinstance(mask, Tensor) or mask.dtype != torch.bool:
        raise TypeError("mask must be a boolean torch.Tensor")
    if mask.device != value.device:
        raise ValueError("mask and activation must share a device")
    try:
        return torch.broadcast_to(mask, value.shape)
    except RuntimeError as exc:
        raise ValueError("mask must broadcast to the activation shape") from exc


def scalar_metric(metric: Runner, value: Tensor) -> Tensor:
    score = metric(value)
    if not isinstance(score, Tensor) or score.numel() != 1:
        raise ValueError("metric must return one scalar Tensor")
    if not torch.isfinite(score).all():
        raise ValueError("metric must return a finite scalar")
    if not score.requires_grad:
        raise ValueError("metric must be differentiable with respect to its input")
    return score.reshape(())


class BaseCausal(BaseMethod):
    """Base class for interventions and local causal-effect approximations.

    Tensor edits alone are not causal measurements. An optional pure ``runner``
    replays the surrounding computation with the original and edited tensor and
    returns a scalar or same-shaped vector of scores. Model hooks belong to the
    adapter, so kernels also work with non-transformer models.
    """

    family = "causal"

    def _intervention_result(
        self,
        original: Tensor,
        edited: Tensor,
        mask: Tensor,
        *,
        runner: Runner | None,
        metadata: dict | None = None,
        extra_tensors: dict[str, Tensor] | None = None,
    ) -> ProbeResult:
        tensors = {"original": original, "edited": edited, "mask": mask}
        tensors.update(extra_tensors or {})
        details = {"effect_measured": runner is not None, **(metadata or {})}
        if runner is not None:
            baseline = runner(original.clone())
            intervention = runner(edited.clone())
            if not isinstance(baseline, Tensor) or not isinstance(intervention, Tensor):
                raise TypeError("runner must return a Tensor")
            if baseline.shape != intervention.shape:
                raise ValueError("runner must return the same score shape for both runs")
            if not torch.isfinite(baseline).all() or not torch.isfinite(intervention).all():
                raise ValueError("runner must return finite scores")
            tensors.update(
                baseline_score=baseline,
                intervention_score=intervention,
                effect=analysis_difference(intervention, baseline),
            )
            details["effect_sign"] = "intervention_score - baseline_score"
            details["analysis_dtype"] = str(tensors["effect"].dtype)
        return ProbeResult(method=self.name, tensors=tensors, metadata=details)

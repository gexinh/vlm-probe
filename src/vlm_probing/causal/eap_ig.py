"""Integrated-gradient scoring on caller-defined computational edges."""

from collections.abc import Sequence

import torch
from torch import Tensor

from ..core.types import ProbeResult
from ..core.precision import analysis_difference, analysis_dtype
from .base import BaseCausal, Runner, scalar_metric, selection_mask, validate_pair


class EAPIG(BaseCausal):
    """Score actual computation edges along a straight edge-activation path.

    ``receiver`` and ``source`` have shape ``[edges, ...]``. Each named edge must
    be an independently replaceable message at a destination in the caller's
    computation graph. ``metric(edge_messages)`` must replay that graph and
    return a scalar. Ordinary node caches or correlations do not define edges.

    This kernel uses simultaneous interpolation in *edge activation space* and
    a trapezoidal gradient integral. It does not extract a transformer circuit,
    implement the original repository's input-embedding path, or validate edge
    identities on behalf of the caller. Exact edge knockout remains necessary
    to evaluate the circuit selected from these attribution scores.
    """

    name = "eap_ig"

    def run(
        self,
        receiver: Tensor,
        source: Tensor,
        *,
        edge_names: Sequence[str],
        metric: Runner,
        steps: int = 32,
        mask: Tensor | None = None,
    ) -> ProbeResult:
        validate_pair(receiver, source)
        if receiver.ndim == 0 or receiver.shape[0] == 0:
            raise ValueError("edge activations require a nonempty leading edge axis")
        if isinstance(edge_names, (str, bytes)):
            raise TypeError("edge_names must be a sequence of edge identifiers")
        names = tuple(edge_names)
        if len(names) != receiver.shape[0] or any(not isinstance(n, str) or not n for n in names):
            raise ValueError("provide one nonempty name per computational edge")
        if len(set(names)) != len(names):
            raise ValueError("computational edge names must be unique")
        if not isinstance(steps, int) or isinstance(steps, bool) or steps < 1:
            raise ValueError("steps must be a positive integer")
        selected = selection_mask(receiver, mask)
        # Keep path arithmetic and accumulation wide, but replay native messages.
        # Select endpoints directly: even a wide subtraction can lose a small
        # source next to a much larger receiver through cancellation.
        dtype = analysis_dtype(receiver, source)
        baseline = receiver.detach().to(dtype)
        delta = torch.where(selected, source.detach().to(dtype) - baseline, torch.zeros_like(baseline))
        endpoint = torch.where(selected, source.detach(), receiver.detach())
        gradient_sum = torch.zeros_like(baseline)
        endpoint_scores = []
        with torch.enable_grad():
            for index in range(steps + 1):
                if index == 0:
                    value = receiver.detach()
                elif index == steps:
                    value = endpoint
                else:
                    value = (baseline + (index / steps) * delta).to(receiver.dtype)
                point = value.detach().requires_grad_(True)
                score = scalar_metric(metric, point)
                gradient = torch.autograd.grad(score, point)[0]
                if not torch.isfinite(gradient).all():
                    raise ValueError("metric produced nonfinite gradients on the path")
                weight = 0.5 if index in (0, steps) else 1.0
                gradient_sum = gradient_sum + weight * gradient.detach().to(dtype)
                if index in (0, steps):
                    endpoint_scores.append(score.detach())
        average_gradient = gradient_sum / steps
        attribution = delta * average_gradient
        edge_scores = attribution.reshape(receiver.shape[0], -1).sum(dim=-1)
        effect = analysis_difference(endpoint_scores[1], endpoint_scores[0])
        return ProbeResult(
            method=self.name,
            tensors={"attribution": attribution, "edge_scores": edge_scores,
                     "average_gradient": average_gradient, "delta": delta,
                     "baseline_score": endpoint_scores[0], "intervention_score": endpoint_scores[1],
                     "effect": effect, "completeness_error": attribution.sum() - effect},
            metadata={"edge_names": list(names), "steps": steps,
                      "path": "simultaneous_linear_edge_activations",
                      "quadrature": "trapezoid", "sign": "source_metric - receiver_metric",
                      "analysis_dtype": str(dtype),
                      "graph_provider": "caller", "masked_source_endpoint": mask is not None},
        )

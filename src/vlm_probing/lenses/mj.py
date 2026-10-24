"""Modality-split Jacobian Lens, independently implemented from MJ-Lens Eq. (2).

Reference: ``Does the Workspace See? Testing a Multimodal Global Workspace in
Vision-Language Models`` (ICLR 2027 submission, OpenReview Xrp8mNku6R). Unlike
our original Jacobian Lens, this estimator averages admissible causal
source/target *pairs* within each prompt, then weights prompts equally. A
calibrated readout is an observational instrument, not a validated intervention
or evidence of a global workspace. No paper-scale reproduction is implied.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
import warnings

import torch
from torch import Tensor

from .base import BaseLens, decoded, floating_tensor, load_artifact, save_artifact
from ..core.types import ProbeResult


_MODALITIES = {"text", "visual"}


def _modality(value: str, name: str) -> str:
    if not isinstance(value, str) or value not in _MODALITIES:
        raise ValueError(f"{name} must be 'text' or 'visual'")
    return value


def _mask(value: Tensor, shape: tuple[int, int], name: str, device: torch.device) -> Tensor:
    if not isinstance(value, Tensor) or value.dtype != torch.bool or tuple(value.shape) != shape:
        raise ValueError(f"{name} must be boolean with shape {shape}")
    return value.to(device)


def _selection_report(source_mask: Tensor, target_mask: Tensor) -> tuple[Tensor, dict[str, object]]:
    # For each source position, count selected targets at that position or later.
    # Reverse cumsum needs only O(B*S) storage, rather than a B*S*S pair matrix.
    future_targets = target_mask.to(torch.int64).flip(-1).cumsum(-1).flip(-1)
    pair_counts = (future_targets * source_mask).sum(-1)
    report: dict[str, object] = {
        "pair_counts": pair_counts.cpu().tolist(),
        "source_counts": source_mask.sum(-1).cpu().tolist(),
        "target_counts": target_mask.sum(-1).cpu().tolist(),
        "n_prompts": int(source_mask.shape[0]),
        "group_counts": {},
    }
    failing = (pair_counts == 0).nonzero(as_tuple=True)[0]
    if len(failing):
        raise ValueError(
            "MJ-Lens calibration requires at least one admissible causal pair "
            "(target_position >= source_position) in every prompt; "
            f"failing rows={failing.cpu().tolist()}; "
            f"source_counts={report['source_counts']}; "
            f"target_counts={report['target_counts']}; "
            f"pair_counts={report['pair_counts']}. "
            "Check modality masks, padding and source/target order. "
            "Self pairs are retained; no prompts were dropped."
        )
    return pair_counts, report


def mj_jacobian_sums_from_graph(
    final: Tensor, sources: Mapping[int, Tensor], source_mask: Tensor,
    target_mask: Tensor, *, dim_batch: int = 32, group_ids: Tensor | None = None,
) -> tuple[dict[int, Tensor], dict[str, object]]:
    """Sum equally weighted per-prompt causal pair means for every source site.

    ``final`` is the differentiable input to the model's final normalization;
    each source is a residual ``[B,S,D_source]`` in that same graph. The graph
    must be causal along ``S`` and must not couple distinct batch examples.
    The caller establishes those properties; this helper cannot infer them.
    Masks identify the source/target channels and exclude padding explicitly.
    Target cotangents are summed. Causality supplies zero derivatives for
    earlier targets, and division uses the number of eligible ``t' >= t`` pairs,
    including self pairs. This is not a mean over a rectangular mask product.

    Returned matrices are CPU float64 sums, to be divided by ``n_prompts``
    after merging batches. Exact basis-vector VJPs are grouped by output
    dimension; there is no random Jacobian approximation. ``group_ids`` is an
    optional int64 vector of prompt groups (e.g. odd/even global prompt IDs).
    Group sums/counts share the same VJPs and need no extra backwards.
    """
    if torch.is_inference_mode_enabled():
        raise ValueError("MJ-Lens calibration cannot run inside torch.inference_mode()")
    floating_tensor(final, "final residual", ndim=3)
    if not isinstance(dim_batch, int) or isinstance(dim_batch, bool) or dim_batch < 1:
        raise ValueError("dim_batch must be a positive integer")
    if not final.requires_grad:
        raise ValueError("final residual must retain its differentiable graph")
    if not isinstance(sources, Mapping) or not sources:
        raise ValueError("sources must contain at least one differentiable residual site")
    shape = tuple(final.shape[:2])
    source_mask = _mask(source_mask, shape, "source_mask", final.device)
    target_mask = _mask(target_mask, shape, "target_mask", final.device)
    pair_counts, report = _selection_report(source_mask, target_mask)
    names, tensors = [], []
    for name, value in sources.items():
        floating_tensor(value, f"source residual at layer {name}", ndim=3)
        if tuple(value.shape[:2]) != shape or value.device != final.device or not value.requires_grad:
            raise ValueError(
                f"source residual at layer {name} must preserve batch/sequence {shape}, "
                "share final residual's device and retain its differentiable graph"
            )
        names.append(name)
        tensors.append(value)
    groups: dict[int, Tensor] = {}
    if group_ids is not None:
        if (not isinstance(group_ids, Tensor) or group_ids.dtype != torch.int64
                or tuple(group_ids.shape) != (shape[0],)):
            raise ValueError(f"group_ids must be an int64 tensor with shape {(shape[0],)}")
        group_ids = group_ids.to(final.device)
        groups = {int(group): group_ids == group for group in group_ids.unique().cpu().tolist()}
        report["group_counts"] = {group: int(mask.sum().item()) for group, mask in groups.items()}
    width = final.shape[-1]
    sums = {name: torch.empty(width, value.shape[-1], dtype=torch.float64)
            for name, value in sources.items()}
    group_sums = {group: {name: torch.empty_like(value) for name, value in sums.items()}
                  for group in groups}
    denominator = pair_counts.to(torch.float64)[None, :, None]
    for start in range(0, width, dim_batch):
        stop = min(start + dim_batch, width)
        cotangents = final.new_zeros((stop - start, *final.shape))
        for offset, dimension in enumerate(range(start, stop)):
            cotangents[offset, ..., dimension] = target_mask
        gradients = torch.autograd.grad(
            final, tensors, grad_outputs=cotangents, is_grads_batched=True,
            retain_graph=stop < width, allow_unused=True,
        )
        for name, gradient in zip(names, gradients):
            if gradient is None:
                raise ValueError(f"source residual at layer {name} is disconnected from the final graph")
            selected = gradient.masked_fill(~source_mask[None, ..., None], 0)
            per_prompt = selected.sum(2, dtype=torch.float64) / denominator
            if not torch.isfinite(per_prompt).all():
                raise ValueError(f"MJ-Lens calibration gradients at layer {name} are not finite")
            sums[name][start:stop] = per_prompt.sum(1).detach().cpu()
            for group, selected_prompts in groups.items():
                group_sums[group][name][start:stop] = per_prompt[:, selected_prompts].sum(1).detach().cpu()
    if groups:
        report["group_sums"] = group_sums
    return sums, report


class MJLens(BaseLens):
    """A modality-bound average Jacobian and its native output-channel readout.

    ``readout`` must apply the actual final normalization followed by the text
    head, or a calibrated visual-codebook head. Visual output requires a real
    generation alphabet; it is not supplied by an arbitrary vision encoder.
    ``binding`` identifies the model, site, readout, tokenizer/codebook and
    calibration data for strict save/load checks. Fitting replaces this lens's
    estimate; merge dataset batches using the graph helper and prompt counts.
    """

    name = "mj_lens"
    estimator = "mean_admissible_causal_pairs_within_prompt_mean_prompts"

    def __init__(
        self, readout: Callable[[Tensor], Tensor], *, source_modality: str = "visual",
        target_modality: str = "text", binding: Mapping[str, str] | None = None,
    ) -> None:
        if not callable(readout):
            raise TypeError("readout must be callable")
        self.readout = readout
        self.source_modality = _modality(source_modality, "source_modality")
        self.target_modality = _modality(target_modality, "target_modality")
        self.binding = dict(binding or {})
        self.jacobian: Tensor | None = None
        self.calibration: dict[str, object] = {}

    def _config(self) -> dict[str, str]:
        return {"estimator": self.estimator, "source_modality": self.source_modality,
                "target_modality": self.target_modality}

    def fit(
        self, activations: Tensor, downstream: Callable[[Tensor], Tensor], *,
        valid_mask: Tensor | None = None, source_mask: Tensor, target_mask: Tensor,
        dim_batch: int = 32,
    ) -> "MJLens":
        """Estimate Eq. (2) from a differentiable causal continuation of a batch.

        Masks are mandatory so channel identities are never guessed. There are
        no implicit first/last-token exclusions. ``valid_mask`` intersects both
        channel masks. Every prompt must contain an eligible causal pair.
        """
        if torch.is_inference_mode_enabled():
            raise ValueError("MJ-Lens calibration cannot run inside torch.inference_mode()")
        floating_tensor(activations, "activations", ndim=3)
        shape = tuple(activations.shape[:2])
        valid = torch.ones(shape, dtype=torch.bool, device=activations.device) if valid_mask is None else _mask(
            valid_mask, shape, "valid_mask", activations.device)
        sources = _mask(source_mask, shape, "source_mask", activations.device) & valid
        targets = _mask(target_mask, shape, "target_mask", activations.device) & valid
        _selection_report(sources, targets)  # Diagnose malformed selections before executing the model.
        if not isinstance(dim_batch, int) or isinstance(dim_batch, bool) or dim_batch < 1:
            raise ValueError("dim_batch must be a positive integer")
        if self.jacobian is not None:
            warnings.warn(
                "MJ-Lens fit replaces the existing estimate; it does not accumulate calibration batches. "
                "Use prompt-weighted dataset calibration for multiple batches.",
                RuntimeWarning, stacklevel=2,
            )
        with torch.enable_grad():
            hidden = activations.detach().clone().requires_grad_(True)
            final = downstream(hidden)
            sums, report = mj_jacobian_sums_from_graph(
                final, {0: hidden}, sources, targets, dim_batch=dim_batch,
            )
        estimate = sums[0] / int(report["n_prompts"])
        floating_tensor(estimate, "estimated MJ-Lens Jacobian", ndim=2)
        self.jacobian = estimate
        self.calibration = {
            **self._config(), **report, "dim_batch": dim_batch,
            "source_positions": [row.nonzero(as_tuple=True)[0].cpu().tolist() for row in sources],
            "target_positions": [row.nonzero(as_tuple=True)[0].cpu().tolist() for row in targets],
        }
        return self

    @torch.no_grad()
    def run(
        self, activations: Tensor, *, source_modality: str, layer: int | None = None,
    ) -> ProbeResult:
        """Read activations of the same source modality as this fitted transport."""
        actual = _modality(source_modality, "source_modality")
        if actual != self.source_modality:
            raise ValueError(
                f"MJ-Lens source modality mismatch: fitted {self.source_modality!r}, "
                f"received {actual!r}; use a transport fitted on the source modality being read"
            )
        floating_tensor(activations, "activations")
        if self.jacobian is None:
            raise RuntimeError("fit or load MJ-Lens before running it")
        floating_tensor(self.jacobian, "MJ-Lens Jacobian", ndim=2)
        if activations.shape[-1] != self.jacobian.shape[-1]:
            raise ValueError("activations do not match the fitted source hidden width")
        transported = activations @ self.jacobian.to(activations).T
        return ProbeResult(self.name, {"logits": decoded(self.readout, transported),
                                       "transported": transported}, {
            **self.calibration, **self._config(), "layer": layer, "binding": dict(self.binding),
        })

    @torch.no_grad()
    def dictionary(
        self, unembedding: Tensor, gain: Tensor, token_ids: Tensor | Sequence[int] | None = None,
    ) -> Tensor:
        """Return rows of ``W diag(gain) J`` in the source residual coordinates.

        ``gain`` is the learned final RMSNorm gain. Scalar RMS division is
        irrelevant to these dictionary directions, but is retained by the
        native ``readout`` when computing actual logits. This gain-only formula
        does not implement a generic LayerNorm or an uncalibrated image head.
        """
        if self.jacobian is None:
            raise RuntimeError("fit or load MJ-Lens before building its dictionary")
        floating_tensor(self.jacobian, "MJ-Lens Jacobian", ndim=2)
        floating_tensor(unembedding, "unembedding", ndim=2)
        floating_tensor(gain, "RMSNorm gain", ndim=1)
        if unembedding.shape[1] != self.jacobian.shape[0] or gain.shape[0] != unembedding.shape[1]:
            raise ValueError("unembedding and RMSNorm gain must match the fitted final hidden width")
        if token_ids is not None:
            ids = torch.as_tensor(token_ids, device=unembedding.device)
            if ids.ndim != 1 or ids.dtype not in (torch.int32, torch.int64) or not ids.numel():
                raise ValueError("token_ids must be a nonempty one-dimensional integer sequence")
            if (ids < 0).any() or (ids >= unembedding.shape[0]).any():
                raise ValueError("token_ids contain an out-of-range vocabulary or codebook index")
            unembedding = unembedding.index_select(0, ids.to(torch.int64))
        return (unembedding * gain.to(unembedding)) @ self.jacobian.to(unembedding)

    def save(self, path: str | Path) -> None:
        if self.jacobian is None:
            raise RuntimeError("fit MJ-Lens before saving it")
        floating_tensor(self.jacobian, "MJ-Lens Jacobian", ndim=2)
        save_artifact(path, self.name, self.binding, self._config(),
                      {"jacobian": self.jacobian.detach().cpu(), "calibration": self.calibration})

    def load(self, path: str | Path) -> "MJLens":
        state = load_artifact(path, self.name, self.binding, self._config())
        if not isinstance(state, Mapping) or "jacobian" not in state or not isinstance(state.get("calibration"), Mapping):
            raise ValueError("MJ-Lens artifact requires Jacobian and calibration metadata")
        floating_tensor(state["jacobian"], "artifact MJ-Lens Jacobian", ndim=2)
        self.jacobian = state["jacobian"]
        self.calibration = dict(state["calibration"])
        return self

"""Dataset calibration for MJ-Lens; exact causal pair expectations, frozen backbone."""
from pathlib import Path
import time

import torch

from .distribution import _atomic_json, _atomic_torch
from ..api.mj import batch_inputs, target_mask
from ..lenses.jacobian import _calibration_masks, _calibration_selection
from ..lenses.mj import mj_jacobian_sums_from_graph


def fit_mj_lens(method, input_batches, *, directory=None, dim_batch=32, resume=False,
                checkpoint_every=8, skip_first=0, exclude_last=False):
    """Mean admissible-pair Jacobians within prompts, then equal prompt means.

    All selected layers share a differentiable graph. Odd/even accumulators use
    the same VJPs. Resume requires deterministic/re-iterable inputs and immutable
    model/tokenizer/corpus identities; it never guesses whether a dataset changed.
    """
    if torch.is_inference_mode_enabled():
        raise ValueError("MJ calibration cannot run under inference_mode")
    if type(dim_batch) is not int or dim_batch < 1 or type(checkpoint_every) is not int or checkpoint_every < 1:
        raise ValueError("dim_batch and checkpoint_every must be positive integers")
    if directory is None and resume:
        raise ValueError("resume requires a directory")
    if directory is not None:
        method._require_binding()
    settings = {"layers": method.layers, "source": method.source, "target": method.target,
                "context": method.context, "binding": {str(i): method._binding(i) for i in method.layers},
                "dim_batch": dim_batch, "skip_first": skip_first, "exclude_last": exclude_last}
    root = None if directory is None else Path(directory)
    state_path = None if root is None else root / "mj_calibration_state.pt"
    sums, groups, group_counts = {}, {}, {0: 0, 1: 0}
    prompts, batches, pair_counts = 0, 0, []
    if state_path is not None and state_path.exists():
        if not resume:
            raise ValueError("MJ calibration state already exists; use resume=True or a new directory")
        state = torch.load(state_path, map_location="cpu", weights_only=True)
        if state.get("format_version") != 1 or state["settings"] != settings:
            raise ValueError("MJ resume settings/model/modality/context binding differ")
        sums, groups, group_counts = state["sums"], state["groups"], state["group_counts"]
        prompts, batches, pair_counts = state["prompts"], state["batches"], state["pair_counts"]
    previous_batches = batches
    previous_prompts = prompts
    visited_batches = 0
    probe, spec = method.probe, method.spec
    sites = list(dict.fromkeys([spec.sources[i] for i in method.layers] + [spec.final]))
    earliest = spec.sources[min(method.layers)]
    flags = [(parameter, parameter.requires_grad) for parameter in probe.model.parameters()]
    modes = [(module, module.training) for module in probe.model.modules()]
    started = time.monotonic()

    def checkpoint():
        if root is not None:
            root.mkdir(parents=True, exist_ok=True)
            _atomic_torch(state_path, {"format_version": 1, "settings": settings,
                "sums": sums, "groups": groups, "group_counts": group_counts,
                "prompts": prompts, "batches": batches, "pair_counts": pair_counts})

    try:
        probe.model.eval()
        probe.model.requires_grad_(False)
        for index, batch in enumerate(input_batches):
            visited_batches = index + 1
            if index < previous_batches:
                continue
            native = batch_inputs(batch, method.context)
            trace = probe._run(native, capture=sites, grad=True,
                interventions={earliest: lambda value: value.detach().requires_grad_(True)})
            layout = probe._layout(native, trace.logits)
            sources, targets = _calibration_masks(layout.valid,
                source_mask=_calibration_selection(layout, method.source),
                target_mask=target_mask(batch, layout, method.target), skip_first=skip_first,
                exclude_last=exclude_last,
                context=f"MJ batch={index}, {method.source}->{method.target}, context={method.context}")
            final = trace.activations[spec.final]
            values = {i: trace.activations[spec.sources[i]] for i in method.layers}
            ids = torch.arange(prompts, prompts + final.shape[0], device=final.device) % 2
            additions, report = mj_jacobian_sums_from_graph(final, values, sources, targets,
                                                           dim_batch=dim_batch, group_ids=ids)
            for layer, value in additions.items():
                sums[layer] = sums.get(layer, torch.zeros_like(value)) + value
            for group, layer_sums in report["group_sums"].items():
                groups.setdefault(group, {})
                for layer, value in layer_sums.items():
                    groups[group][layer] = groups[group].get(layer, torch.zeros_like(value)) + value
            for group, count in report["group_counts"].items():
                group_counts[group] = group_counts.get(group, 0) + count
            prompts += report["n_prompts"]
            batches = index + 1
            pair_counts.extend(report["pair_counts"])
            del trace, values, final, additions, report
            if batches % checkpoint_every == 0:
                checkpoint()
    finally:
        for parameter, flag in flags:
            parameter.requires_grad_(flag)
        for module, training in modes:
            module.training = training
    if not prompts:
        raise ValueError("MJ calibration iterable contains no eligible prompts")
    if visited_batches < previous_batches:
        raise ValueError("MJ resume iterable is shorter than the recorded calibration prefix")
    checkpoint()
    kernels, cosines = {}, {}
    for layer in method.layers:
        kernel = method._new_kernel(layer)
        kernel.jacobian = sums[layer] / prompts
        kernel.calibration = {"n_prompts": prompts, "n_batches": batches,
            "pair_counts": pair_counts, "estimator": kernel.estimator,
            "skip_first": skip_first, "exclude_last": exclude_last,
            "source_modality": method.source, "target_modality": method.target,
            "output_context": method.context, "tap_convention": "decoder_block_input"}
        kernels[layer] = kernel
        if group_counts.get(0) and group_counts.get(1):
            a, b = groups[0][layer].flatten(), groups[1][layer].flatten()
            denom = a.norm() * b.norm()
            cosines[str(layer)] = float((a @ b) / denom) if denom > 0 else None
    method.kernels = kernels
    method.calibration_report = {"status": "complete", "settings": settings,
        "n_prompts": prompts, "n_batches": batches, "pair_counts": pair_counts,
        "split_half_prompt_counts": group_counts, "split_half_cosine": cosines,
        "elapsed_seconds": time.monotonic() - started,
        "elapsed_scope": "this_fit_batches_call",
        "resumed": previous_batches > 0,
        "resumed_batches": previous_batches, "resumed_prompts": previous_prompts,
        "new_batches": batches - previous_batches, "new_prompts": prompts - previous_prompts,
        "interpretation": "numeric calibration diagnostic; not a semantic validity or steering test"}
    if root is not None:
        method.save(root)
        _atomic_json(root / "mj_calibration.json", method.calibration_report)
    return method

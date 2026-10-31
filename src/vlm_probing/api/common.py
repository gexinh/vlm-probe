"""Shared execution helpers for the model-bound method families."""
from collections.abc import Mapping
from functools import wraps

import torch

from ..adapters import CapabilityError, ProbeInputs
from ..core import ProbeResult


def model_eval(function):
    """Keep standalone readout/calibration calls in the same mode as forwards."""
    @wraps(function)
    def wrapped(self, *args, **kwargs):
        states = [(m, m.training) for m in self.probe.model.modules()]
        try:
            self.probe.model.eval()
            return function(self, *args, **kwargs)
        finally:
            for module, training in states:
                module.training = training
    return wrapped


def configuration(value):
    """JSON-safe public configuration; tensors are explicit caller parameters."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Mapping):
        return {str(k): configuration(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [configuration(v) for v in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return type(value).__name__


def metric_metadata(metric):
    if metric is None:
        return None
    return {"name": getattr(metric, "__name__", type(metric).__name__),
            "parameters": configuration(getattr(metric, "__dict__", {})) if hasattr(metric, "score") else {}}


def unpack(inputs):
    if isinstance(inputs, ProbeInputs):
        return dict(inputs.kwargs), inputs.layout
    if not isinstance(inputs, Mapping):
        raise TypeError("inputs must be native model kwargs or ProbeInputs")
    return dict(inputs), None


def layers_for(probe, layers, kind="residuals"):
    available = getattr(probe.spec, kind)
    if not available:
        raise CapabilityError(f"adapter lacks {kind}; inspect probe.describe()")
    chosen = sorted(available) if layers is None else ([layers] if type(layers) is int else list(layers))
    if not chosen or len(chosen) != len(set(chosen)) or any(type(i) is not int for i in chosen):
        raise ValueError("layers must be a nonempty selection of unique integer indices")
    missing = set(chosen) - available.keys()
    if missing:
        raise CapabilityError(f"adapter lacks {kind} at layers {sorted(missing)}; inspect probe.describe()")
    return chosen


def require(value, message):
    if value is None:
        raise CapabilityError(message)
    return value


def evaluate(metric, logits, layout):
    if metric is None:
        raise ValueError("provide a metric, e.g. TokenMargin(positive=..., negative=...)")
    score = metric.score(logits, layout) if hasattr(metric, "score") else metric(logits)
    if (not isinstance(score, torch.Tensor) or score.ndim > 1
            or (score.ndim == 1 and score.shape != (logits.shape[0],))
            or not torch.isfinite(score).all()):
        raise ValueError("metric must return a finite scalar or [batch] tensor")
    return score


def selected(hidden, mask):
    return hidden[mask.to(hidden.device)]


def pack(probe, method, labels, results, **metadata):
    keys = results[0].tensors.keys()
    if any(r.tensors.keys() != keys for r in results):
        raise ValueError("layer results have different tensor fields")
    tensors = {}
    for key in keys:
        values = [r.tensors[key].detach().to(probe.result_device) for r in results]
        if any(v.shape != values[0].shape for v in values):
            raise ValueError(f"layer results for {key} have different shapes; run layers separately")
        tensors[key] = torch.stack(values)
    return ProbeResult(method, tensors, {
        "model_id": probe.adapter.model_id, "layers": list(labels),
        "layer_results": [r.metadata for r in results], **metadata,
    })


def coordinates(probe, result, mask):
    result.tensors["positions"] = mask.nonzero().to(probe.result_device)
    result.metadata["position_axes"] = ["batch", "expanded_token"]
    return result


def check_alignment(probe, source_inputs, receiver_inputs, source_layout, receiver_layout,
                    alignment):
    """Structural alignment is checked; semantic pairing remains the experiment's choice."""
    if alignment not in {"strict", "position"}:
        raise ValueError("alignment must be 'strict' or explicitly 'position'")
    a, b = source_layout, receiver_layout
    for name in ("valid", "visual", "prompt"):
        x, y = getattr(a, name), getattr(b, name)
        if (x is None) != (y is None) or (x is not None and not torch.equal(x.cpu(), y.cpu())):
            raise ValueError(f"source/receiver {name} layouts differ; align inputs before patching")
    xargs, _ = unpack(source_inputs)
    yargs, _ = unpack(receiver_inputs)
    for key in probe.spec.alignment_keys:
        x, y = xargs.get(key), yargs.get(key)
        if isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor):
            equal = torch.equal(x.cpu(), y.cpu())
        else:
            equal = x is None and y is None
        if not equal:
            raise ValueError(f"source/receiver alignment metadata differs: {key}")
    if alignment == "strict":
        if a.token_ids is None or b.token_ids is None:
            raise ValueError("strict alignment needs layout.token_ids; explicitly use alignment='position' "
                             "only after checking semantic correspondence")
        valid = a.valid.cpu()
        if not torch.equal(a.token_ids.cpu()[valid], b.token_ids.cpu()[valid]):
            raise ValueError("source/receiver token IDs differ; align them or explicitly use alignment='position'")


class BoundMethod:
    def __init__(self, probe, name, layers):
        self.probe, self.name, self.layers = probe, name, layers

    def _finish(self, result):
        result.tensors = {k: v.detach().to(self.probe.result_device) for k, v in result.tensors.items()}
        result.metadata.update(model_id=self.probe.adapter.model_id, layers=self.layers)
        applicability = getattr(self, "applicability", None)
        if applicability is not None:
            result.metadata["applicability"] = applicability
        return result

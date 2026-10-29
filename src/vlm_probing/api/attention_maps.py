"""Target-conditioned maps with distinct attention and input integration paths."""
import torch

from .. import attention as kernels
from ..adapters import CapabilityError, ProbeInputs
from ..core.precision import analysis_dtype
from .common import BoundMethod, configuration, evaluate, metric_metadata, pack, unpack
from .attention_applicability import propagation_applicability


def _integration_dtype(*values):
    """Use at least FP32 for analysis arithmetic without narrowing FP64."""
    return analysis_dtype(*(value for value in values if value is not None))


def _integration_norm(value):
    return value.to(_integration_dtype(value)).norm(dim=-1)


class AttentionMapProbe(BoundMethod):
    def __init__(self, probe, name, layers, **options):
        super().__init__(probe, name, layers)
        self.options = options
        self.applicability = propagation_applicability(probe.spec, name)

    def _trace(self, inputs, capture, *, interventions=None):
        edits = dict(interventions or {})
        if self.probe.spec.embeddings is not None:
            site = self.probe.spec.embeddings
            # Gives frozen models a graph without changing parameter gradients.
            if site not in edits:
                edits[site] = lambda x: x if x.requires_grad else x.detach().requires_grad_(True)
        return self.probe._run(inputs, capture=capture, interventions=edits, grad=True)

    @staticmethod
    def _gradients(score, matrices):
        if not score.requires_grad or any(not x.requires_grad for x in matrices):
            raise CapabilityError("target score and captured attentions must be differentiable")
        values = torch.autograd.grad(score.sum(), matrices, allow_unused=True)
        if any(x is None for x in values):
            raise CapabilityError("an attention capture is diagnostic-only and does not feed the target score")
        return values

    @torch.enable_grad()
    def run(self, inputs, *, metric=None, baseline=None):
        p, o = self.probe, self.options
        if metric is None:
            raise ValueError(f"{self.name} requires an explicit fixed target metric")
        if baseline is not None and self.name not in {"tam", "beyond_intuition"}:
            raise ValueError("baseline is only an input-path endpoint for TAM/Beyond Intuition")
        sites = [p.spec.attentions[i] for i in self.layers]
        capture = list(sites)
        support_sites = [p.spec.attention_supports.get(i) for i in self.layers]
        if self.name == "grad_cam":
            capture += [site for site in support_sites if site is not None]
        if self.name == "beyond_intuition" and o["variant"] == "token":
            capture += [p.spec.attention_inputs[i] for i in self.layers]
            capture += [p.spec.attention_values[i] for i in self.layers]
        trace = self._trace(inputs, capture)
        layout = p._layout(inputs, trace.logits)
        matrices = [trace.activations[site] for site in sites]
        score = evaluate(metric, trace.logits, layout)
        base_score = score.detach()
        gradients = self._gradients(score, matrices) if not (
            self.name == "attribution" or self.name == "tam"
            or self.name == "beyond_intuition" and o["variant"] == "token") else None
        # Free baseline forward graphs before integration. No backward populates .grad.
        attentions = [x.detach() for x in matrices]
        supports = [trace.activations[site].detach() if site is not None else None
                    for site in support_sites] if self.name == "grad_cam" else []
        input_norms = value_norms = None
        if self.name == "beyond_intuition" and o["variant"] == "token":
            input_norms = torch.stack([
                _integration_norm(trace.activations[p.spec.attention_inputs[i]].detach())
                for i in self.layers])
            with torch.no_grad():
                value_norms = torch.stack([
                    _integration_norm(p.spec.project_values(
                        i, trace.activations[p.spec.attention_values[i]].detach()))
                    for i in self.layers])
        if gradients is not None:
            gradients = [x.detach() for x in gradients]
        del trace, matrices, score
        query = layout.select(o["queries"])
        key = layout.select(o["keys"])
        if self.name == "grad_cam":
            results = [kernels.AttentionGradCAM().run(a, g, query_mask=query.to(a.device),
                        key_mask=key.to(a.device), support_mask=None if support is None else support.to(a.device),
                        normalize=o["normalize"])
                       for a, g, support in zip(attentions, gradients, supports)]
            result = pack(p, results[0].method, self.layers, results)
        elif self.name == "attribution":
            results = []
            for site, a in zip(sites, attentions):
                total = torch.zeros_like(a, dtype=_integration_dtype(a))
                for alpha in self._alphas():
                    # The original zero-attention path does not renormalize rows.
                    # Interventions keep the model's native probability dtype.
                    leaf = (alpha * a).detach().requires_grad_(True)
                    sampled = self._trace(inputs, [site], interventions={site: lambda x: leaf})
                    sample_layout = p._layout(inputs, sampled.logits)
                    g, = self._gradients(evaluate(metric, sampled.logits, sample_layout),
                                         [sampled.activations[site]])
                    total += g.detach().to(total.dtype) / o["steps"]
                    del sampled, g, leaf
                results.append(kernels.AttentionAttribution().run(a, total,
                               query_mask=query.to(a.device), key_mask=key.to(a.device),
                               steps=o["steps"], quadrature=o["quadrature"]))
            result = pack(p, results[0].method, self.layers, results)
        else:
            total = self._input_integral(inputs, sites[-1], metric, baseline)
            dtype = _integration_dtype(total, *attentions, *(gradients or []), input_norms, value_norms)
            stack = torch.stack(attentions)
            total = total.to(dtype)
            valid = layout.valid.to(stack.device)
            if self.name == "tam":
                result = kernels.TransitionAttentionMaps(start_layer=o["start_layer"],
                    residual_normalize=o["residual_normalize"]).run(
                    stack, total, valid_tokens=valid, steps=o["steps"], quadrature=o["quadrature"])
            else:
                result = kernels.BeyondIntuition(variant=o["variant"], start_layer=o["start_layer"]).run(
                    stack, total, gradients=None if gradients is None else torch.stack(gradients),
                    input_norms=None if input_norms is None else input_norms.to(dtype),
                    projected_value_norms=None if value_norms is None else value_norms.to(dtype), valid_tokens=valid,
                    steps=o["steps"], quadrature=o["quadrature"])
            active = query.to(stack.device)[:, :, None] & key.to(stack.device)[:, None, :]
            result.tensors["map"] = result.tensors["relevance"].masked_fill(~active, 0)
            result = self._finish(result)
        result.tensors["query_mask"] = query.detach().to(p.result_device)
        result.tensors["key_mask"] = key.detach().to(p.result_device)
        result.tensors["baseline_score"] = base_score.to(p.result_device)
        result.metadata.update(configuration=configuration(o), metric=metric_metadata(metric),
                               target_fixed=True, observational_attribution=True)
        if self.applicability is not None:
            result.metadata["applicability"] = self.applicability
        if self.name in {"tam", "beyond_intuition"}:
            result.metadata.update(integration_path=f"{o['input_key']}: baseline -> processed input; other kwargs fixed",
                                   input_baseline="zero in processed-input coordinates" if baseline is None else "caller supplied",
                                   integration_steps=o["steps"], quadrature=o["quadrature"],
                                   transfer="VLM/language attention is a method transfer from vision classification"
                                   if p.spec.output_kind != "classification" else "native vision classification")
        return result

    def _alphas(self):
        n = self.options["steps"]
        if self.options["quadrature"] == "endpoints":
            return [i / (n-1) for i in range(n)]
        first = 1 if self.options["quadrature"] == "right" else 0
        return [i / n for i in range(first, n+first)]

    def _input_integral(self, inputs, site, metric, baseline):
        p, o = self.probe, self.options
        values, explicit = unpack(inputs)
        x = values.get(o["input_key"])
        if not isinstance(x, torch.Tensor) or not x.is_floating_point():
            raise CapabilityError(f"input-path integration requires a floating tensor {o['input_key']!r}")
        zero = torch.zeros_like(x) if baseline is None else baseline
        if not isinstance(zero, torch.Tensor) or zero.shape != x.shape or zero.device != x.device or zero.dtype != x.dtype:
            raise ValueError("input baseline must match processed input shape, device, and dtype")
        if not torch.isfinite(x).all() or not torch.isfinite(zero).all():
            raise ValueError("integration endpoints must be finite")
        dtype = _integration_dtype(x, zero)
        zero_wide, x_wide = zero.to(dtype), x.to(dtype)
        difference = x_wide - zero_wide
        total = None
        for alpha in self._alphas():
            # Select true endpoints directly: even a wide subtraction can lose
            # a much smaller endpoint through cancellation. Only the final
            # model-facing point is cast back to the processed input dtype.
            if alpha == 0:
                point = zero
            elif alpha == 1:
                point = x
            else:
                point = (zero_wide + alpha * difference).to(x.dtype)
            sample_values = {**values, o["input_key"]: point.detach().requires_grad_(True)}
            sampled = self._trace(ProbeInputs(sample_values, explicit), [site])
            layout = p._layout(ProbeInputs(sample_values, explicit), sampled.logits)
            g, = self._gradients(evaluate(metric, sampled.logits, layout), [sampled.activations[site]])
            if total is None:
                total = torch.zeros_like(g, dtype=_integration_dtype(g))
            total += g.detach().to(total.dtype) / o["steps"]
            del sampled, g
        return total

"""Attention observation and real value-path interventions."""
import torch
from .attention_maps import AttentionMapProbe
from .attention_applicability import propagation_applicability

from .. import attention as kernels
from ..adapters import CapabilityError
from ..core import ProbeResult
from .common import BoundMethod, configuration, evaluate, layers_for, metric_metadata, pack, require


class AttentionProbe(BoundMethod):
    def __init__(self, probe, name, layers, **options):
        super().__init__(probe, name, layers)
        self.options = options
        self.applicability = propagation_applicability(probe.spec, name)

    def run(self, inputs, *, metric=None):
        p, name, o = self.probe, self.name, self.options
        if name in {"profile", "rollout", "head_logits"} and metric is not None:
            raise ValueError(f"{name} is target-independent and does not accept a metric; "
                             "omit metric or use metric=None")
        mapping = p.spec.attention_scores if name == "temperature" else p.spec.attentions
        if name == "head_logits":
            mapping = p.spec.heads
        sites = [mapping[i] for i in self.layers]
        capture = list(sites)
        if name == "head_logits":
            final_site = p.spec.residuals[max(p.spec.residuals)]
            capture = list(dict.fromkeys([*capture, final_site]))
        if name in {"relevance", "reweight", "temperature"} and metric is None:
            raise ValueError(f"{name} requires a target metric")
        # Frozen model weights must still permit gradients through activations.
        edits = {}
        if name == "relevance" and p.spec.embeddings is not None:
            edits[p.spec.embeddings] = lambda x: x.detach().requires_grad_(True)
        trace = p._run(inputs, capture=capture, interventions=edits, grad=name == "relevance")
        layout = p._layout(inputs, trace.logits)
        matrices = [trace.activations[site] for site in sites]
        if name in {"rollout", "relevance"}:
            if self.layers != list(range(0, self.layers[-1] + 1)):
                raise ValueError("propagation requires consecutive layers from 0 in forward order")
            stack = torch.stack(matrices)
            valid = layout.valid.to(stack.device)
            if name == "rollout":
                result = kernels.AttentionRollout(**o).run(stack, valid_tokens=valid)
            else:
                score = evaluate(metric, trace.logits, layout).sum()
                grads = torch.autograd.grad(score, matrices)
                result = kernels.AttentionRelevance().run(stack, torch.stack(grads), valid_tokens=valid)
            result.metadata.update(configuration=configuration(o), metric=metric_metadata(metric),
                                   applicability=self.applicability)
            return self._finish(result)
        results = []
        if name == "profile":
            groups = o["groups"]
            if groups is None:
                groups = {"visual": "visual", "text": "text"} if layout.visual is not None else {}
            # Empty groups (text-only input) have a meaningful zero mass.
            group_masks = {}
            for key, selector in groups.items():
                if isinstance(selector, str) and selector in {"visual", "text"} and layout.visual is not None:
                    group_masks[key] = layout.visual if selector == "visual" else layout.valid & ~layout.visual
                else:
                    group_masks[key] = layout.select(selector)
            for matrix in matrices:
                result = kernels.AttentionProfile().run(
                    matrix, groups={k: v.to(matrix.device) for k, v in group_masks.items()},
                    query_mask=layout.select(o["queries"]).to(matrix.device),
                    key_mask=layout.valid.to(matrix.device))
                if o.get("include_attention", False):
                    result.tensors["attention"] = matrix
                results.append(result)
        elif name == "head_logits":
            full = trace.activations[final_site]
            weight, scale, center = p.spec.linear_readout(full)
            for layer, raw in zip(self.layers, matrices):
                heads = p._heads(layer, raw).permute(0, 2, 1, 3)
                results.append(kernels.HeadLogitAttribution().run(
                    heads, weight.to(heads), fixed_scale=scale.to(heads), center=center))
        else:
            baseline = evaluate(metric, trace.logits, layout)
            query_mask = layout.select(o["queries"])
            key_mask = layout.select(o["keys"]) if name == "reweight" else None
            for layer, site in zip(self.layers, sites):
                def edit(value):
                    active = query_mask.to(value.device)[:, None, :, None]
                    if name == "temperature":
                        result = kernels.AttentionTemperature(o["temperature"]).run(
                            value, query_mask=layout.valid.to(value.device))
                        return torch.where(active, result.tensors["logits"], value)
                    weights = torch.where(key_mask.to(value.device)[:, None, None, :], o["weight"], 1.)
                    result = kernels.AttentionReweight(renormalize=o["renormalize"]).run(
                        value, weights, query_mask=query_mask.to(value.device),
                        key_mask=layout.valid.to(value.device))
                    return result.tensors["attention"]
                edited = p._run(inputs, interventions={site: edit})
                score = evaluate(metric, edited.logits, layout)
                results.append(ProbeResult(f"attention_{name}", {"baseline_score": baseline,
                                                 "intervention_score": score, "effect": score - baseline},
                                           {"effect_measured": True}))
        return pack(p, results[0].method, self.layers, results,
                    configuration=configuration(o), metric=metric_metadata(metric),
                    sweep="independent per-layer runs" if name in {"reweight", "temperature"} else "observation")


class AttentionMethods:
    def __init__(self, probe):
        self.probe = probe

    def profile(self, *, layers=None, queries="all", groups=None, include_attention=False):
        return AttentionProbe(self.probe, "profile", layers_for(self.probe, layers, "attentions"),
                              queries=queries, groups=groups, include_attention=include_attention)

    def grad_cam(self, *, layers=None, queries=None, keys=None, normalize=False):
        """Grad-CAM on attention query rows; adaptation of the original feature CAM."""
        queries, keys = self._map_selection(queries, keys)
        return AttentionMapProbe(self.probe, "grad_cam", layers_for(self.probe, layers, "attentions"),
                                 queries=queries, keys=keys, normalize=normalize)

    def attribution(self, *, layers=None, queries=None, keys=None, steps=20, quadrature="right"):
        """Hao attention IG: independently integrate each layer from zero A to A."""
        self._integration(steps, quadrature)
        queries, keys = self._map_selection(queries, keys)
        chosen = layers_for(self.probe, layers, "attentions")
        if not set(chosen) <= self.probe.spec.editable_attention:
            raise CapabilityError("Attention Attribution requires consumed editable probabilities before A @ V")
        return AttentionMapProbe(self.probe, "attribution", chosen, queries=queries, keys=keys,
                                 steps=steps, quadrature=quadrature)

    def tam(self, *, layers=None, queries=None, keys=None, steps=20,
            quadrature="right", input_key="pixel_values", start_layer=0, residual_normalize=True):
        """TAM's backward Markov transitions and input-path final-attention feedback."""
        return self._input_map("tam", layers, queries, keys, steps, quadrature, input_key, start_layer,
                               residual_normalize=residual_normalize)

    def beyond_intuition(self, *, variant="head", layers=None, queries=None, keys=None,
                         steps=20, quadrature="right", input_key="pixel_values", start_layer=0):
        """Beyond Intuition headwise/tokenwise perception with input-path feedback."""
        if variant not in {"head", "token"}:
            raise ValueError("variant must be head or token")
        if variant == "token":
            spec = self.probe.spec
            require(spec.project_values, "token variant requires an explicit no-bias V/output projection")
            chosen = self._propagation_layers(layers)
            if not set(chosen) <= spec.attention_inputs.keys() & spec.attention_values.keys():
                raise CapabilityError("token variant requires pre-LN input and V captures at every selected layer")
        return self._input_map("beyond_intuition", layers, queries, keys, steps, quadrature,
                               input_key, start_layer, variant=variant)

    def dtd_lrp(self, *, method="transformer_attribution", start_layer=0, device=None):
        """Use Chefer's original-rule ViT backend with copied caller-owned weights."""
        from ..attention import CheferLRP, create_chefer_vit
        model = create_chefer_vit(source=self.probe.model)
        if device is not None:
            model = model.to(device)
        return CheferLRP(model, method=method, start_layer=start_layer, result_device=self.probe.result_device)

    @staticmethod
    def _integration(steps, quadrature):
        if type(steps) is not int or steps < 1:
            raise ValueError("steps must be a positive integer")
        if quadrature not in {"left", "right", "endpoints"}:
            raise ValueError("quadrature must be left, right, or endpoints")
        if quadrature == "endpoints" and steps < 2:
            raise ValueError("endpoint-inclusive integration requires steps >= 2")

    def _input_map(self, name, layers, queries, keys, steps, quadrature, input_key, start_layer, **extra):
        self._integration(steps, quadrature)
        queries, keys = self._map_selection(queries, keys)
        chosen = self._propagation_layers(layers)
        if type(start_layer) is not int or not 0 <= start_layer < len(chosen):
            raise ValueError("start_layer must be inside the selected consecutive stack")
        if not isinstance(input_key, str) or not input_key:
            raise ValueError("input_key must name a floating processed-input tensor")
        return AttentionMapProbe(self.probe, name, chosen, queries=queries, keys=keys,
                                 steps=steps, quadrature=quadrature, input_key=input_key,
                                 start_layer=start_layer, **extra)

    def _map_selection(self, queries, keys):
        vision = self.probe.spec.output_kind == "classification"
        return (([0] if vision else "last_prompt") if queries is None else queries,
                ("visual" if vision else "all") if keys is None else keys)

    def rollout(self, *, layers=None, head_reduction="mean", residual=True):
        return AttentionProbe(self.probe, "rollout", self._propagation_layers(layers),
                              head_reduction=head_reduction, residual=residual)

    def relevance(self, *, layers=None):
        return AttentionProbe(self.probe, "relevance", self._propagation_layers(layers))

    def _propagation_layers(self, layers):
        available = self.probe.describe()["methods"]["attention.rollout"]["sites"]
        if not available:
            raise CapabilityError("propagation requires attention layers starting at 0; "
                                  "hybrid linear-attention gaps cannot be skipped")
        chosen = layers_for(self.probe, available if layers is None else layers, "attentions")
        if not chosen or chosen != list(range(chosen[-1] + 1)):
            raise CapabilityError("propagation requires consecutive attention layers from 0; "
                                  "hybrid linear-attention gaps cannot be skipped")
        boundary = self.probe.spec.last_stream_kv_layer
        if boundary is not None and chosen[-1] > boundary:
            raise CapabilityError(f"propagation stops at layer {boundary}: later layers reuse stored K/V, "
                                  "so composing them would read positions from the wrong stream")
        return chosen

    def head_logits(self, *, layers=None):
        require(self.probe.spec.linear_readout, "head attribution requires spec.linear_readout")
        return AttentionProbe(self.probe, "head_logits", layers_for(self.probe, layers, "heads"))

    def reweight(self, *, layers=None, queries="text", keys="visual", weight=2., renormalize=True):
        chosen = layers_for(self.probe, layers, "attentions")
        if not set(chosen) <= self.probe.spec.editable_attention:
            raise CapabilityError("reweight requires editable probabilities before A @ V; "
                                  "returned diagnostic attentions are not editable")
        if not isinstance(weight, (int, float)) or not 0 <= weight < float("inf"):
            raise ValueError("weight must be finite and nonnegative")
        return AttentionProbe(self.probe, "reweight", chosen, queries=queries, keys=keys,
                              weight=weight, renormalize=renormalize)

    def temperature(self, *, layers=None, queries="all", temperature=1.):
        kernels.AttentionTemperature(temperature)  # validate before model execution
        return AttentionProbe(self.probe, "temperature", layers_for(self.probe, layers, "attention_scores"),
                              queries=queries, temperature=temperature)

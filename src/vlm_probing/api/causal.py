"""Independent layer interventions, scoring, and explicit edge attribution."""
import torch

from .. import causal as kernels
from ..adapters import CapabilityError, ProbeInputs, TokenLayout
from ..core import ProbeResult
from ..core.precision import analysis_difference, analysis_dtype
from .common import (BoundMethod, check_alignment, configuration, evaluate, layers_for,
                     metric_metadata, pack, unpack)


class CausalProbe(BoundMethod):
    def __init__(self, probe, name, layers, tokens="all", **options):
        super().__init__(probe, name, layers)
        self.tokens, self.options = tokens, options

    def run(self, inputs, *, metric, source=None, direction=None,
            alignment="strict"):
        p, name, o = self.probe, self.name, self.options
        if o.get("joint"):
            baseline_trace = p._run(inputs)
            layout = p._layout(inputs, baseline_trace.logits)
            baseline = evaluate(metric, baseline_trace.logits, layout)
            output = self.forward(inputs, direction=direction)
            score = evaluate(metric, output.logits, layout)
            return self._finish(ProbeResult(
                "steering" if name == "steer" else "attention_knockout",
                {"baseline_score": baseline, "intervention_score": score,
                 "effect": analysis_difference(score, baseline)},
                {"joint": True, "effect_measured": True, "actual_forward_passes": 2,
                 "configuration": configuration({"tokens": self.tokens, **o}),
                 "metric": metric_metadata(metric), "effect_sign": "intervention minus baseline"}))
        mapping = p.spec.attention_scores if name == "knockout" else p.spec.residuals
        sites = [mapping[i] for i in self.layers]
        paired = name in {"patch", "attribute"}
        if paired and source is None:
            raise ValueError(f"{name} requires source=clean_inputs")
        if name == "steer" and direction is None:
            raise ValueError("steer requires direction=... in residual coordinates")
        trace = p._run(inputs, capture=sites)
        layout = p._layout(inputs, trace.logits)
        baseline = evaluate(metric, trace.logits, layout)
        source_trace = None
        if paired:
            other = source
            source_trace = p._run(other, capture=sites)
            other_layout = p._layout(other, source_trace.logits)
            check_alignment(p, other, inputs, other_layout, layout, alignment)
        token_mask = layout.select(self.tokens) if name != "knockout" else None
        results = []
        for layer, site in zip(self.layers, sites):
            original = trace.activations[site]
            if name != "knockout" and original.shape[:2] != layout.valid.shape:
                raise ValueError("causal residual must match the expanded token layout")
            mask = token_mask.to(original.device)[..., None] if token_mask is not None else None
            src = source_trace.activations[site] if source_trace is not None else None
            if name == "attribute":
                def metric_hidden(value):
                    output = p._run(inputs, interventions={site: lambda _: value}, grad=True)
                    scores = evaluate(metric, output.logits, layout)
                    return scores.to(analysis_dtype(scores)).sum()
                result = kernels.AttributionPatching().run(original, src, mask=mask, metric=metric_hidden)
                # Retain useful scores, not full cached activations/gradients.
                results.append(ProbeResult(result.method, {
                    "token_scores": result.tensors["attribution"].sum(-1),
                    "estimated_effect": result.tensors["estimated_effect"],
                }, result.metadata))
                continue
            if name == "patch":
                edited = kernels.ActivationPatching().run(original, src, mask=mask).tensors["edited"]
            elif name == "steer":
                value = direction[layer] if isinstance(direction, dict) else direction
                value = value.to(original)
                edited = kernels.Steering().run(original, value, mask=mask,
                                                strength=o["strength"], preserve_norm=o["preserve_norm"]).tensors["edited"]
            else:
                query = layout.select(o["queries"]).to(original.device)
                key = layout.select(o["keys"]).to(original.device)
                blocked = query[:, None, :, None] & key[:, None, None, :]
                valid_query = layout.valid.to(original.device)[:, None, :, None]
                safe = original.masked_fill(~valid_query, 0)
                edited = kernels.AttentionKnockout().run(safe, blocked=blocked).tensors["edited"]
                edited = torch.where(valid_query, edited, original)
            output = p._run(inputs, interventions={site: lambda _: edited})
            score = evaluate(metric, output.logits, layout)
            method_name = {"patch": "activation_patching",
                           "steer": "steering", "knockout": "attention_knockout"}[name]
            results.append(ProbeResult(method_name, {"baseline_score": baseline, "intervention_score": score,
                                              "effect": analysis_difference(score, baseline)},
                                       {"effect_measured": True}))
        return pack(p, results[0].method, self.layers, results,
                    sweep="independent per-layer runs", alignment=alignment if paired else None,
                    configuration=configuration({"tokens": self.tokens, **o}),
                    metric=metric_metadata(metric),
                    effect_sign="source/intervention minus receiver")

    def forward(self, inputs, *, direction=None, capture=()):
        """One live joint forward, optionally retaining requested hook tensors.

        Later layers operate on the already intervened stream; no cached baseline
        activation replaces their inputs. Call ``run`` to measure a baseline too.
        """
        p, o = self.probe, self.options
        if not o.get("joint") or self.name not in {"steer", "knockout"}:
            raise ValueError("forward requires a joint steering or knockout configuration")
        if self.name == "steer" and direction is None:
            raise ValueError("steer requires direction=... in residual coordinates")
        mapping = p.spec.attention_scores if self.name == "knockout" else p.spec.residuals
        edits = {}
        for layer in self.layers:
            if self.name == "steer":
                value = direction[layer] if isinstance(direction, dict) else direction

                def edit(hidden, value=value):
                    layout = p._layout(inputs, hidden)
                    return kernels.Steering().run(
                        hidden, value.to(hidden), mask=layout.select(self.tokens)[..., None],
                        strength=o["strength"], preserve_norm=o["preserve_norm"]).tensors["edited"]
            else:
                def edit(scores):
                    layout = p._layout(inputs, scores, sequence_axis=2)
                    queries, keys = layout.select(o["queries"]), layout.select(o["keys"])
                    blocked = queries[:, None, :, None] & keys[:, None, None, :]
                    valid = layout.valid[:, None, :, None]
                    safe = scores.masked_fill(~valid, 0)
                    edited = kernels.AttentionKnockout().run(safe, blocked=blocked).tensors["edited"]
                    return torch.where(valid, edited, scores)
            edits[mapping[layer]] = edit
        return p._run(inputs, capture=capture, interventions=edits)

    def generate(self, inputs, *, direction, max_new_tokens=64, do_sample=False):
        """Uncached greedy generation with the same joint intervention each step.

        Native input IDs must already include expanded visual placeholders. This
        transparent reference implementation supports one unpadded example, and
        deliberately replays its entire prefix, including images, at every step.
        """
        if self.name != "steer" or not self.options.get("joint"):
            raise ValueError("generate requires joint steering")
        if do_sample or type(max_new_tokens) is not int or max_new_tokens < 1:
            raise ValueError("generate requires do_sample=False and positive max_new_tokens")
        values, explicit = unpack(inputs)
        ids = values.get("input_ids")
        if not isinstance(ids, torch.Tensor) or ids.ndim != 2 or ids.shape[0] != 1:
            raise ValueError("generate requires native input_ids for one example")
        attention = values.get("attention_mask", torch.ones_like(ids))
        if attention.ndim != 2 or not attention.bool().all():
            raise ValueError("generate requires an unpadded two-dimensional attention mask")
        initial_length = ids.shape[1]
        if explicit is None and self.tokens == "prediction":
            if self.probe.spec.layout is None:
                raise CapabilityError("prediction steering requires an explicit TokenLayout or layout provider")
            explicit = self.probe.spec.layout(values).checked(ids.shape, ids.device)
        config = getattr(self.probe.model, "generation_config", None)
        eos = getattr(config, "eos_token_id", None)
        eos = set(eos if isinstance(eos, (list, tuple)) else [eos])
        for _ in range(max_new_tokens):
            current = ProbeInputs(values, explicit) if explicit is not None else values
            trace = self.forward(current, direction=direction)
            token = trace.logits[:, -1].argmax(-1, keepdim=True)
            ids = torch.cat((values["input_ids"], token.to(ids.device)), dim=1)
            values = {**values, "input_ids": ids,
                      "attention_mask": torch.cat((attention, torch.ones_like(attention[:, :1])), dim=1)}
            attention = values["attention_mask"]
            for field in ("token_type_ids", "mm_token_type_ids", "position_ids"):
                if field in values:
                    previous = values[field]
                    if not isinstance(previous, torch.Tensor) or previous.shape[-1] != ids.shape[1] - 1:
                        raise ValueError(f"generate requires {field} aligned to the complete sequence")
                    if field == "position_ids":
                        extension = previous[..., -1:] + 1
                    elif field == "token_type_ids":
                        extension = previous[..., -1:]
                    else:  # Multimodal generated positions are text.
                        extension = torch.zeros_like(previous[..., -1:])
                    values[field] = torch.cat((previous, extension), dim=-1)
            if "cache_position" in values:
                values["cache_position"] = torch.arange(ids.shape[1], device=ids.device)
            if explicit is not None:
                # Fixed original prompt semantics; new positions are generated text.
                valid = torch.cat((explicit.valid, torch.ones_like(explicit.valid[:, :1])), dim=1)
                visual = (torch.cat((explicit.visual, torch.zeros_like(explicit.visual[:, :1])), dim=1)
                          if explicit.visual is not None else None)
                prompt = explicit.valid if explicit.prompt is None else explicit.prompt
                prompt = torch.cat((prompt, torch.zeros_like(prompt[:, :1])), dim=1)
                explicit = TokenLayout(valid, visual, prompt, ids)
            if token.item() in eos:
                break
        generated = ids[:, initial_length:].detach().cpu()
        tokenizer = getattr(self.probe.processor, "tokenizer", self.probe.processor)
        decoded = tokenizer.batch_decode(generated, skip_special_tokens=True) if tokenizer else None
        return self._finish(ProbeResult("steering_generation", {
            "generated_ids": generated, "fullsequence_ids": ids,
        }, {"joint": True, "decoded_text": decoded, "generation": "uncached greedy full-prefix replay",
            "actual_forward_passes": generated.shape[1],
            "configuration": configuration({"tokens": self.tokens, **self.options})}))


class EdgeProbe(BoundMethod):
    def __init__(self, probe, edges, steps):
        super().__init__(probe, "eap_ig", [])
        self.edges, self.steps = edges, steps

    def run(self, inputs, *, source, metric, alignment="strict"):
        p = self.probe
        sites = [p.spec.edges[name] for name in self.edges]
        receiver = p._run(inputs, capture=sites)
        donor = p._run(source, capture=sites)
        layout = p._layout(inputs, receiver.logits)
        other = p._layout(source, donor.logits)
        check_alignment(p, source, inputs, other, layout, alignment)
        original = torch.stack([receiver.activations[site] for site in sites])
        clean = torch.stack([donor.activations[site] for site in sites])

        def replay(messages):
            edits = {site: (lambda _, i=i: messages[i]) for i, site in enumerate(sites)}
            output = p._run(inputs, interventions=edits, grad=True)
            scores = evaluate(metric, output.logits, layout)
            return scores.to(analysis_dtype(scores)).sum()

        result = kernels.EAPIG().run(original, clean, edge_names=self.edges, metric=replay, steps=self.steps)
        # Avoid returning full message caches through the convenient interface.
        result.tensors = {k: v for k, v in result.tensors.items() if k in {
            "edge_scores", "effect", "baseline_score", "intervention_score", "completeness_error"}}
        result.metadata["alignment"] = alignment
        result.metadata["metric"] = metric_metadata(metric)
        return self._finish(result)


class PathProbe(BoundMethod):
    """Bind the four-stage path kernel to model inputs and sequence semantics."""

    def __init__(self, probe, senders, receivers, sender_tokens, donor_tokens, receiver_tokens):
        senders, receivers = kernels.PathPatching.endpoints(probe.spec, senders, receivers)
        super().__init__(probe, "path", sorted({layer for layer, _ in senders}))
        self.senders, self.receivers = senders, receivers
        self.sender_tokens = sender_tokens
        self.donor_tokens = sender_tokens if donor_tokens is None else donor_tokens
        self.receiver_tokens = receiver_tokens

    def _bindings(self, inputs, donor, metric, alignment):
        p, context = self.probe, {}

        def select(base, source):
            layout = p._layout(inputs, base.logits)
            other = p._layout(donor, source.logits)
            check_alignment(p, donor, inputs, other, layout, alignment)
            context.update(layout=layout, donor_layout=other, donor_trace=source)
            return (layout.select(self.sender_tokens), other.select(self.donor_tokens),
                    layout.select(self.receiver_tokens))

        def score(trace):
            layout = context["donor_layout"] if trace is context["donor_trace"] else context["layout"]
            return evaluate(metric, trace.logits, layout)

        return select, score

    def _finish_path(self, result, metric, alignment):
        result.metadata.update(alignment=alignment, metric=metric_metadata(metric),
                               configuration=configuration({
                                   "sender_tokens": self.sender_tokens,
                                   "donor_tokens": self.donor_tokens,
                                   "receiver_tokens": self.receiver_tokens,
                               }))
        return self._finish(result)

    def run(self, inputs, *, donor, metric, alignment="strict"):
        """Intervene on the configured sender set jointly in four forwards."""
        p = self.probe
        select, score = self._bindings(inputs, donor, metric, alignment)

        result = kernels.PathPatching().run(
            run_base=lambda **kwargs: p._run(inputs, **kwargs),
            run_donor=lambda **kwargs: p._run(donor, **kwargs), spec=p.spec,
            senders=self.senders, receivers=self.receivers, select=select, score=score,
        )
        return self._finish_path(result, metric, alignment)

    def sweep(self, inputs, *, donor, metric, alignment="strict", progress=None):
        """Measure each sender independently, sharing unmodified forward caches.

        ``intervention_score`` and ``effect`` have shape [paths, batch], or
        [paths] for scalar metrics. ``baseline_score`` and ``donor_score``
        retain the metric's original shape. ``senders`` is [paths, 2] in the
        configured order. Each path uses the same receiver set and exact
        control rules as :meth:`run`; only the baseline/donor runs are reused.

        ``progress(completed, total, sender)`` is called after each path, with
        a one-based completed count and the (layer, head) sender tuple.
        """
        if progress is not None and not callable(progress):
            raise TypeError("progress must be callable or None")
        p = self.probe
        # Revalidate capabilities in case the caller has modified the spec.
        senders, receivers = kernels.PathPatching.endpoints(p.spec, self.senders, self.receivers)
        receiver_sites = ([p.spec.path_final] if receivers == "residual" else
                          [p.spec.path_qkv[layer][kind].site for layer, _, kind in receivers])
        base_sites = list(dict.fromkeys([*(point.site for point in p.spec.path_heads.values()),
                                        *receiver_sites]))
        donor_sites = list(dict.fromkeys(p.spec.path_heads[layer].site for layer, _ in senders))
        base = p._run(inputs, capture=base_sites)
        source = p._run(donor, capture=donor_sites)
        select, score = self._bindings(inputs, donor, metric, alignment)

        def run_base(**kwargs):
            # The initial kernel call needs only captured unmodified values.
            # Controlled and receiver-only passes always execute the model.
            return base if kwargs.get("interventions") is None else p._run(inputs, **kwargs)

        results = []
        for completed, sender in enumerate(senders, 1):
            result = kernels.PathPatching().run(
                run_base=run_base, run_donor=lambda **_: source, spec=p.spec,
                senders=[sender], receivers=receivers, select=select, score=score,
            )
            results.append(result)
            if progress is not None:
                progress(completed, len(senders), sender)

        tensors = {name: results[0].tensors[name] for name in ("baseline_score", "donor_score")}
        tensors.update({name: torch.stack([result.tensors[name] for result in results])
                        for name in ("intervention_score", "effect")})
        tensors["senders"] = torch.tensor(senders, dtype=torch.long)
        metadata = {**results[0].metadata,
                    "senders": [list(sender) for sender in senders],
                    "sweep": "independent sender-to-receiver paths",
                    "shared_caches": ["baseline", "donor"],
                    "actual_forward_passes": 2 + 2 * len(senders),
                    "score_axes": ["path", "batch"] if tensors["effect"].ndim == 2 else ["path"],
                    "sender_axes": ["path", "layer_or_head"]}
        return self._finish_path(ProbeResult("path_patching", tensors, metadata), metric, alignment)


class CausalMethods:
    def __init__(self, probe):
        self.probe = probe

    def patch(self, *, layers=None, tokens="visual"):
        return CausalProbe(self.probe, "patch", layers_for(self.probe, layers), tokens)

    def path(self, *, senders, receivers="residual", sender_tokens="last_prompt",
             donor_tokens=None, receiver_tokens="last_prompt"):
        """Joint sender-to-receiver intervention; Q/K/V heads are physical axes."""
        return PathProbe(self.probe, senders, receivers, sender_tokens, donor_tokens, receiver_tokens)

    def attribute(self, *, layers=None, tokens="visual"):
        return CausalProbe(self.probe, "attribute", layers_for(self.probe, layers), tokens)

    def steer(self, *, layers=None, tokens="visual", strength=1., preserve_norm=False, joint=False):
        return CausalProbe(self.probe, "steer", layers_for(self.probe, layers), tokens,
                           strength=strength, preserve_norm=preserve_norm, joint=joint)

    def knockout(self, *, layers=None, queries="text", keys="visual", joint=False):
        return CausalProbe(self.probe, "knockout", layers_for(self.probe, layers, "attention_scores"),
                           queries=queries, keys=keys, joint=joint)

    def vsv(self, positive, negative, *, layers=None):
        """Construct image-specific visual residual directions without training."""
        return kernels.VisualSteering.from_inputs(self.probe, positive, negative, layers=layers)

    def eap_ig(self, *, edges=None, steps=32, graph="activation", quadrature="left"):
        if graph == "transformer":
            if edges is not None:
                raise ValueError("transformer graph supplies its own independently replaceable edges")
            from ..causal.transformer_eap_ig import TransformerEAPIG
            return TransformerEAPIG(self.probe, steps=steps, quadrature=quadrature)
        if graph != "activation":
            raise ValueError("graph must be 'activation' or 'transformer'")
        edges = list(self.probe.spec.edges) if edges is None else list(edges)
        if not edges or any(name not in self.probe.spec.edges for name in edges):
            raise CapabilityError("EAP-IG requires explicit computational edge sites in spec.edges")
        if len(set(edges)) != len(edges):
            raise ValueError("edge names must be unique")
        if type(steps) is not int or steps < 1:
            raise ValueError("steps must be a positive integer")
        return EdgeProbe(self.probe, edges, steps)

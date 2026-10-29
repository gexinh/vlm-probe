"""Original input-path EAP-IG on a native GPT-2 residual-message graph.

The graph has independent pre-LayerNorm Q/K/V inputs for every attention head,
MLP inputs, and the final residual input. Only input embeddings interpolate;
clean/corrupted source differences stay fixed throughout the gradient integral.
"""
import heapq

import torch
from torch.nn import functional as F

from ..adapters import CapabilityError
from ..api.common import BoundMethod, check_alignment, evaluate, metric_metadata
from ..core import ProbeResult
from ..core.precision import analysis_difference, analysis_dtype


class GPT2ResidualGraph:
    """Scoped, independently editable residual messages for native HF GPT-2.

    This provider splits the packed QKV projection into per-head computations
    from separate copies of its pre-LN input. No model weight is changed. Output
    projection biases are constant background contributions and cancel in edge
    replacements. Head outputs include their value bias through normal forward
    computation; MLP outputs include their output bias.
    """

    def __init__(self, probe):
        if type(probe.model).__name__ != "GPT2LMHeadModel":
            raise CapabilityError("the audited transformer EAP-IG graph currently supports native GPT2LMHeadModel")
        self.probe, self.model = probe, probe.model
        self.blocks = self.model.transformer.h
        cfg = self.model.config
        if cfg.add_cross_attention or getattr(cfg, "reorder_and_upcast_attn", False):
            raise CapabilityError("this graph supports standard GPT-2 self-attention only")
        self.layers, self.heads, self.width = len(self.blocks), cfg.n_head, cfg.n_embd
        self.source_names, self.destination_names = ["input"], []
        self.before_attention, self.before_mlp = [], []
        self.attention_destinations, self.mlp_destinations = [], []
        self.source_incoming = [None]
        edge_indices = []
        for layer in range(self.layers):
            previous = len(self.source_names)
            self.before_attention.append(previous)
            attention_start = len(self.destination_names)
            self.attention_destinations.append(attention_start)
            for kind in "qkv":
                for head in range(self.heads):
                    destination = len(self.destination_names)
                    self.destination_names.append(f"L{layer}.H{head}.{kind.upper()}")
                    edge_indices.extend((source, destination) for source in range(previous))
            for head in range(self.heads):
                self.source_names.append(f"L{layer}.H{head}")
                self.source_incoming.append([attention_start + kind * self.heads + head for kind in range(3)])
            self.before_mlp.append(len(self.source_names))
            destination = len(self.destination_names)
            self.mlp_destinations.append(destination)
            self.destination_names.append(f"L{layer}.MLP")
            edge_indices.extend((source, destination) for source in range(len(self.source_names)))
            self.source_names.append(f"L{layer}.MLP")
            self.source_incoming.append([destination])
        self.final_destination = len(self.destination_names)
        self.destination_names.append("readout")
        edge_indices.extend((source, self.final_destination) for source in range(len(self.source_names)))
        self.edge_indices = torch.tensor(edge_indices, dtype=torch.long)
        self.edge_names = [f"{self.source_names[u]} -> {self.destination_names[v]}" for u, v in edge_indices]
        self._active = False

    def describe(self):
        return {"provider": "native_hf_gpt2_pre_ln_residual_messages", "scope": "full_graph",
                "layers": self.layers, "heads_per_layer": self.heads,
                "source_nodes": self.source_names, "destinations": self.destination_names,
                "edge_count": len(self.edge_names), "message_positions": "all aligned valid sequence positions",
                "input": "token embeddings plus unchanged positional embeddings, before first block",
                "constant_background": "attention output projection biases; unchanged by edge replacement"}

    @staticmethod
    def _first(args, kwargs):
        return args[0] if args else kwargs["hidden_states"]

    @staticmethod
    def _replace_first(args, kwargs, value):
        if args:
            return (value, *args[1:]), kwargs
        return args, {**kwargs, "hidden_states": value}

    @staticmethod
    def _layer_norm(value, module):
        return F.layer_norm(value, module.normalized_shape, module.weight, module.bias, module.eps)

    def run(self, inputs, *, input_point=None, donor_sources=None, keep=None, grad=False):
        """Execute native forward with graph capture and optional edge knockout.

        At a destination v the input becomes its live residual plus
        sum_{excluded (u,v)}(donor_output[u] - live_output[u]). Thus retained
        edges use the current partially intervened forward, never clean caches.
        """
        if self._active:
            raise RuntimeError("a graph provider cannot be used concurrently or recursively")
        if (donor_sources is None) != (keep is None):
            raise ValueError("edge replacement needs donor_sources and keep together")
        if keep is not None and (keep.dtype != torch.bool or keep.shape != (len(self.edge_names),)):
            raise ValueError("keep must be one boolean per real graph edge")
        self._active = True
        context = {"sources": [], "receiver_blocks": [], "attention_inputs": {}}
        handles = []
        device = next(self.model.parameters()).device
        excluded = torch.zeros((len(self.source_names), len(self.destination_names)), device=device)
        if keep is not None:
            indices = self.edge_indices.to(device)
            excluded[indices[:, 0], indices[:, 1]] = (~keep.to(device)).float()

        def edit(value, previous, destinations):
            if donor_sources is None:
                return value
            live = torch.stack(context["sources"][:previous], dim=2)
            source = donor_sources[:, :, :previous].to(device=live.device)
            dtype = analysis_dtype(value, live, source)
            difference = source.to(dtype) - live.to(dtype)
            coefficients = excluded[:previous, destinations].to(difference)
            if coefficients.ndim == 1:
                change = torch.einsum("btsd,s->btd", difference, coefficients)
            else:
                change = torch.einsum("btsd,sv->btvd", difference, coefficients).reshape_as(value)
            return (value.to(dtype) + change).to(value.dtype)

        def first_block(module, args, kwargs):
            value = self._first(args, kwargs)
            value = value if input_point is None else input_point
            context["sources"].append(value)
            return self._replace_first(args, kwargs, value)

        handles.append(self.blocks[0].register_forward_pre_hook(first_block, with_kwargs=True))
        for layer, block in enumerate(self.blocks):
            def attention_input(module, args, layer=layer):
                context["attention_inputs"][layer] = args[0]

            def packed_projection(module, args, output, layer=layer, block=block):
                residual = context["attention_inputs"].pop(layer)
                copies = residual[:, :, None, None, :].expand(-1, -1, 3, self.heads, -1).clone()
                start = self.attention_destinations[layer]
                copies = edit(copies, self.before_attention[layer], slice(start, start + 3 * self.heads))
                if grad:
                    context["receiver_blocks"].append((start, copies))
                normalized = self._layer_norm(copies, block.ln_1)
                # Conv1D stores [input, output], packed in Q,K,V order.
                weights = module.weight.reshape(self.width, 3, self.heads, self.width // self.heads)
                result = torch.einsum("btqhd,dqhk->btqhk", normalized, weights)
                result = result + module.bias.reshape(3, self.heads, self.width // self.heads)
                return result.reshape(*residual.shape[:2], 3 * self.width)

            def head_outputs(module, args, layer=layer):
                values = args[0].reshape(*args[0].shape[:2], self.heads, self.width // self.heads)
                weights = module.weight.reshape(self.heads, self.width // self.heads, self.width)
                projected = torch.einsum("bthk,hkd->bthd", values, weights)
                context["sources"].extend(projected.unbind(2))

            def mlp_input(module, args, layer=layer):
                destination = self.mlp_destinations[layer]
                # A receiver input is an independent branch message. Without
                # cloning, autograd would also count the residual skip path
                # that consumes this same live residual tensor.
                value = edit(args[0].clone(), self.before_mlp[layer], destination)
                if grad:
                    context["receiver_blocks"].append((destination, value))
                return (value, *args[1:])

            def mlp_output(module, args, output):
                context["sources"].append(output)

            handles.extend([block.ln_1.register_forward_pre_hook(attention_input),
                            block.attn.c_attn.register_forward_hook(packed_projection),
                            block.attn.c_proj.register_forward_pre_hook(head_outputs),
                            block.ln_2.register_forward_pre_hook(mlp_input),
                            block.mlp.register_forward_hook(mlp_output)])

        def final_input(module, args):
            value = edit(args[0], len(self.source_names), self.final_destination)
            if grad:
                context["receiver_blocks"].append((self.final_destination, value))
            return (value, *args[1:])

        handles.append(self.model.transformer.ln_f.register_forward_pre_hook(final_input))
        try:
            trace = self.probe._run(inputs, grad=grad)
            if len(context["sources"]) != len(self.source_names):
                raise RuntimeError("native forward did not execute every declared graph source")
            sources = torch.stack(context["sources"], dim=2)
            return trace, sources, context["receiver_blocks"]
        finally:
            for handle in reversed(handles):
                handle.remove()
            self._active = False

    def gradient_scores(self, inputs, point, difference, metric, layout):
        point = point.detach().requires_grad_(True)
        trace, _, blocks = self.run(inputs, input_point=point, grad=True)
        metric_scores = evaluate(metric, trace.logits, layout)
        loss = -metric_scores.to(analysis_dtype(metric_scores)).mean()
        tensors = [value for _, value in blocks]
        gradients = torch.autograd.grad(loss, tensors)
        dtype = analysis_dtype(difference, *gradients)
        matrix = torch.zeros((len(self.source_names), len(self.destination_names)),
                             device=point.device, dtype=dtype)
        for (start, value), gradient in zip(blocks, gradients):
            count = self.before_attention[start // (3 * self.heads + 1)] if value.ndim == 5 else (
                len(self.source_names) if start == self.final_destination else
                self.before_mlp[self.mlp_destinations.index(start)])
            flattened = gradient.reshape(*gradient.shape[:2], -1, self.width)
            score = torch.einsum("btsd,btvd->sv", difference[:, :, :count].to(dtype),
                                 flattened.to(dtype))
            matrix[:count, start:start + flattened.shape[2]] = score.detach()
        indices = self.edge_indices.to(point.device)
        return matrix[indices[:, 0], indices[:, 1]], loss.detach()

    def select(self, scores, budget, *, strategy="greedy"):
        """Absolute-score backward greedy search, or a declared top-k baseline.

        Greedy expands parents of nodes connected to the readout, following the
        paper's backward frontier rule. The result is pruned to input→readout
        paths; the number remaining can be smaller than the requested budget.
        """
        if type(budget) is not int or not 0 <= budget <= len(self.edge_names):
            raise ValueError("budget must be between zero and the full real edge count")
        if scores.shape != (len(self.edge_names),) or not torch.isfinite(scores).all():
            raise ValueError("scores must be finite and match this graph")
        keep = torch.zeros(len(self.edge_names), dtype=torch.bool)
        if budget == len(self.edge_names):
            return ~keep
        values, indices = scores.detach().abs().cpu().tolist(), self.edge_indices.tolist()
        incoming = [[] for _ in self.destination_names]
        for edge, (_, destination) in enumerate(indices):
            incoming[destination].append(edge)
        if strategy == "topk":
            for edge in sorted(range(len(values)), key=lambda i: (-values[i], i))[:budget]:
                keep[edge] = True
        elif strategy == "greedy":
            frontier, expanded = [], set()

            def expand(source):
                destinations = self.source_incoming[source]
                for destination in destinations or []:
                    if destination not in expanded:
                        expanded.add(destination)
                        for edge in incoming[destination]:
                            heapq.heappush(frontier, (-values[edge], edge))

            for edge in incoming[self.final_destination]:
                heapq.heappush(frontier, (-values[edge], edge))
            for _ in range(budget):
                if not frontier:
                    break
                _, edge = heapq.heappop(frontier)
                keep[edge] = True
                expand(indices[edge][0])
        else:
            raise ValueError("strategy must be 'greedy' or 'topk'")
        # Prune dangling nodes. A Q/K/V endpoint belongs to its head output node.
        destination_source = {destination: source for source, destinations in enumerate(self.source_incoming)
                              for destination in destinations or []}
        forward = {0}
        for edge, (source, destination) in enumerate(indices):
            if keep[edge] and source in forward:
                forward.add(destination_source.get(destination, len(self.source_names)))
        backward = {len(self.source_names)}
        for edge in range(len(indices) - 1, -1, -1):
            source, destination = indices[edge]
            child = destination_source.get(destination, len(self.source_names))
            if keep[edge] and child in backward:
                backward.add(source)
        for edge, (source, destination) in enumerate(indices):
            child = destination_source.get(destination, len(self.source_names))
            if keep[edge] and not (source in forward and child in backward):
                keep[edge] = False
        return keep


class TransformerEAPIG(BoundMethod):
    """Model-bound original EAP-IG and actual full-complement evaluations.

    ``quadrature='left'`` matches the author's current implementation (alpha
    0..(m-1)/m); ``'right'`` matches paper equation 3 (alpha 1/m..1). Endpoint
    activation-space IG is a different method and is not called here.
    """

    def __init__(self, probe, *, steps=5, quadrature="left"):
        if type(steps) is not int or steps < 1:
            raise ValueError("steps must be a positive integer")
        if quadrature not in {"left", "right"}:
            raise ValueError("quadrature must be 'left' or 'right'")
        super().__init__(probe, "eap_ig", sorted(probe.spec.residuals))
        self.graph, self.steps, self.quadrature = GPT2ResidualGraph(probe), steps, quadrature

    def _pair(self, inputs, donor, metric, alignment):
        clean, sources, _ = self.graph.run(inputs)
        corrupt, donor_sources, _ = self.graph.run(donor)
        layout = self.probe._layout(inputs, clean.logits)
        donor_layout = self.probe._layout(donor, corrupt.logits)
        check_alignment(self.probe, donor, inputs, donor_layout, layout, alignment)
        return sources, donor_sources, layout, evaluate(metric, clean.logits, layout), evaluate(metric, corrupt.logits, donor_layout)

    def run(self, inputs, *, donor, metric, alignment="position", progress=None):
        clean, corrupt, layout, baseline, donor_score = self._pair(inputs, donor, metric, alignment)
        difference = analysis_difference(corrupt, clean)
        point_clean, point_corrupt = clean[:, :, 0], corrupt[:, :, 0]
        path_delta = analysis_difference(point_clean, point_corrupt)
        eap, _ = self.graph.gradient_scores(inputs, point_clean, difference, metric, layout)
        scores = torch.zeros_like(eap)
        offset = 0 if self.quadrature == "left" else 1
        alphas = [(index + offset) / self.steps for index in range(self.steps)]
        losses = []
        for index, alpha in enumerate(alphas, 1):
            if alpha == 0:
                point = point_corrupt
            elif alpha == 1:
                point = point_clean
            else:
                point = (point_corrupt.to(path_delta.dtype) + alpha * path_delta).to(point_corrupt.dtype)
            score, loss = self.graph.gradient_scores(inputs, point, difference, metric, layout)
            scores += score / self.steps
            losses.append(loss)
            if progress is not None:
                progress(index, self.steps, alpha)
        return self._finish(ProbeResult("eap_ig", {
            "edge_scores": scores, "eap_scores": eap, "edge_indices": self.graph.edge_indices,
            "baseline_score": baseline, "donor_score": donor_score,
            "path_loss": torch.stack(losses),
        }, {"graph": self.graph.describe(), "edge_names": self.graph.edge_names,
            "path": "linear_input_embeddings", "quadrature": self.quadrature,
            "alphas": alphas, "steps": self.steps, "metric": metric_metadata(metric),
            "score_sign": "estimated corrupted-edge loss increase; loss = -task metric",
            "alignment": alignment, "actual_forward_passes": 3 + self.steps,
            "actual_backward_passes": 1 + self.steps,
            "source_difference": "fixed corrupted minus clean source outputs at endpoints",
            "analysis_dtype": str(scores.dtype),
            "completeness_axiom": "not applicable: scores over graph edges are not input-feature IG"}))

    def evaluate(self, inputs, *, donor, metric, attribution, budgets, alignment="position",
                 rankings=("eap", "eap_ig"), strategy="greedy", epsilon=1e-6, progress=None):
        """Actually corrupt every edge outside each retained circuit.

        Returns raw per-example margins and (circuit-corrupt)/(clean-corrupt).
        Scores determine edge ranking only; curve values come from fresh native
        forwards. All-retained and none-retained endpoint checks are included.
        """
        if not isinstance(epsilon, (int, float)) or isinstance(epsilon, bool) or not 0 < epsilon < float("inf"):
            raise ValueError("epsilon must be a finite positive number")
        _, corrupt, layout, clean_score, corrupt_score = self._pair(inputs, donor, metric, alignment)
        if attribution.metadata.get("edge_names") != self.graph.edge_names:
            raise ValueError("attribution comes from a different graph")
        budgets = list(budgets)
        if not budgets or any(type(n) is not int or not 0 <= n <= len(self.graph.edge_names) for n in budgets):
            raise ValueError("budgets must be real edge counts")
        if not rankings or any(name not in {"eap", "eap_ig"} for name in rankings):
            raise ValueError("rankings must select 'eap' and/or 'eap_ig'")
        metrics, kept_counts, masks = [], [], []
        full_mask = torch.ones(len(self.graph.edge_names), dtype=torch.bool)
        full_trace, _, _ = self.graph.run(inputs, donor_sources=corrupt, keep=full_mask)
        none_trace, _, _ = self.graph.run(inputs, donor_sources=corrupt, keep=~full_mask)
        full_score = evaluate(metric, full_trace.logits, layout)
        none_score = evaluate(metric, none_trace.logits, layout)
        for name in rankings:
            values, counts, selected = [], [], []
            scores = attribution.tensors["eap_scores" if name == "eap" else "edge_scores"]
            for budget in budgets:
                keep = self.graph.select(scores, budget, strategy=strategy)
                trace, _, _ = self.graph.run(inputs, donor_sources=corrupt, keep=keep)
                values.append(evaluate(metric, trace.logits, layout))
                counts.append(int(keep.sum()))
                selected.append(keep)
                if progress is not None:
                    progress(name, budget, counts[-1])
            metrics.append(torch.stack(values))
            kept_counts.append(counts)
            masks.append(torch.stack(selected))
        metrics = torch.stack(metrics)
        normalizer = analysis_difference(clean_score, corrupt_score)
        eligible = normalizer.abs() > epsilon
        normalized = analysis_difference(metrics, corrupt_score) / torch.where(
            eligible, normalizer, torch.ones_like(normalizer))
        normalized = torch.where(eligible, normalized, torch.zeros_like(normalized))
        return self._finish(ProbeResult("eap_ig_faithfulness", {
            "circuit_score": metrics, "faithfulness": normalized, "eligible": eligible,
            "baseline_score": clean_score, "donor_score": corrupt_score,
            "requested_edges": torch.tensor(budgets), "retained_edges": torch.tensor(kept_counts),
            "retained_masks": torch.stack(masks),
            "all_retained_score": full_score, "none_retained_score": none_score,
            "all_retained_error": analysis_difference(full_score, clean_score),
            "none_retained_error": analysis_difference(none_score, corrupt_score),
        }, {"graph": self.graph.describe(), "rankings": list(rankings), "selection": strategy,
            "pruning": "retain only input-to-readout connected paths",
            "faithfulness_scope": "full graph complement: every excluded real edge corrupted",
            "normalization": "(circuit metric - corrupt metric) / (clean metric - corrupt metric)",
            "epsilon": epsilon, "metric": metric_metadata(metric),
            "actual_forward_passes": 4 + len(rankings) * len(budgets),
            "selection_and_evaluation": "caller-supplied evaluation inputs; equality with score-selection inputs is not inferred"}))

    def patch_edges(self, inputs, *, donor, metric, excluded_edges, alignment="position"):
        """Measure selected independent single-edge corruption effects exactly."""
        _, corrupt, layout, clean_score, corrupt_score = self._pair(inputs, donor, metric, alignment)
        edges = list(excluded_edges)
        if not edges or any(type(e) is not int or not 0 <= e < len(self.graph.edge_names) for e in edges):
            raise ValueError("excluded_edges must contain graph edge indices")
        values = []
        for edge in edges:
            keep = torch.ones(len(self.graph.edge_names), dtype=torch.bool)
            keep[edge] = False
            trace, _, _ = self.graph.run(inputs, donor_sources=corrupt, keep=keep)
            values.append(evaluate(metric, trace.logits, layout))
        scores = torch.stack(values)
        return self._finish(ProbeResult("edge_patching", {"edge_ids": torch.tensor(edges),
            "baseline_score": clean_score, "donor_score": corrupt_score, "intervention_score": scores,
            "loss_increase": analysis_difference(clean_score, scores)},
            {"edge_names": [self.graph.edge_names[e] for e in edges],
            "intervention": "one source residual message corrupted at one pre-LN destination",
            "actual_forward_passes": 2 + len(edges)}))

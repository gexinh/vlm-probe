"""IOI-style path control through four actual model forwards.

Wang et al., Appendix B: freeze head outputs, recompute MLPs/LayerNorm,
cache receiver inputs, then replay only those receivers in a fresh base run.
The callbacks execute the caller's real model; no gradient approximation is used.
"""
from collections import defaultdict

import torch

from ..adapters.spec import CapabilityError, ModelSpec
from ..core.types import ProbeResult
from ..core.precision import analysis_difference
from .base import BaseCausal, validate_activation, validate_pair


class PathPatching(BaseCausal):
    name = "path_patching"

    @staticmethod
    def endpoints(spec: ModelSpec, senders, receivers):
        """Validate explicit capabilities before executing the model."""
        missing = spec.residuals.keys() - spec.path_heads.keys()
        if missing:
            raise CapabilityError("IOI path patching requires editable head outputs at EVERY decoder "
                                  f"layer; missing layers {sorted(missing)}")
        senders = list(senders)
        if not senders:
            raise ValueError("senders must be a nonempty list of (layer, head) pairs")
        checked = []
        for sender in senders:
            if not isinstance(sender, (tuple, list)) or len(sender) != 2:
                raise ValueError("each sender must be (layer, head)")
            layer, head = sender
            if type(layer) is not int or layer not in spec.path_heads:
                raise CapabilityError(f"no editable head output at sender layer {layer!r}")
            if type(head) is not int or not 0 <= head < spec.path_heads[layer].heads:
                raise ValueError(f"sender head out of range at layer {layer}")
            checked.append((layer, head))
        if len(set(checked)) != len(checked):
            raise ValueError("sender pairs must be unique")
        if isinstance(receivers, str):
            if receivers != "residual":
                raise ValueError("receivers must be 'residual' or (layer, head, kind) triples")
            if spec.path_final is None:
                raise CapabilityError("adapter lacks a final residual receiver in path_final")
            return checked, receivers
        targets = []
        for receiver in receivers:
            if not isinstance(receiver, (tuple, list)) or len(receiver) != 3:
                raise ValueError("each receiver must be (layer, head, 'q'|'k'|'v')")
            layer, head, kind = receiver
            if type(layer) is not int or not isinstance(kind, str) or kind not in {"q", "k", "v"}:
                raise ValueError("receiver layer must be an integer and kind must be q, k, or v")
            point = spec.path_qkv.get(layer, {}).get(kind)
            if point is None:
                raise CapabilityError(f"adapter lacks receiver {kind} at layer {layer}")
            if type(head) is not int or not 0 <= head < point.heads:
                raise ValueError(f"receiver {kind} head out of range at layer {layer}; "
                                 "K/V indices refer to physical KV heads in GQA")
            targets.append((layer, head, kind))
        if not targets or len(set(targets)) != len(targets):
            raise ValueError("receivers must be nonempty and unique")
        return checked, targets

    def run(self, *, run_base, run_donor, spec, senders, receivers,
            select, score) -> ProbeResult:
        """Execute model callbacks with scoped capture/intervention arguments.

        ``select(base_trace, donor_trace)`` returns boolean [B,T] sender, donor,
        and receiver masks after checking the inputs' sequence alignment.
        ``score(trace)`` returns a finite scalar or per-example metric.
        Native model kwargs and layout policies remain in the bound API.
        """
        senders, receivers = self.endpoints(spec, senders, receivers)
        sender_heads = defaultdict(list)
        for layer, head in senders:
            sender_heads[layer].append(head)
        targets = defaultdict(list)
        if receivers == "residual":
            receiver_sites = [spec.path_final]
        else:
            receiver_sites = []
            for layer, head, kind in receivers:
                point = spec.path_qkv[layer][kind]
                targets[point.site].append(head)
                receiver_sites.append(point.site)
            receiver_sites = list(dict.fromkeys(receiver_sites))
        points = {point.site: point for mapping in spec.path_qkv.values() for point in mapping.values()}

        # A/B: unmodified base outputs and donor sender outputs.
        base_sites = list(dict.fromkeys([*(p.site for p in spec.path_heads.values()), *receiver_sites]))
        base = run_base(capture=base_sites)
        donor = run_donor(capture=[spec.path_heads[i].site for i in sender_heads])
        sender_mask, donor_mask, receiver_mask = select(base, donor)
        for name, mask in (("sender", sender_mask), ("donor", donor_mask), ("receiver", receiver_mask)):
            if not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool or mask.ndim != 2:
                raise ValueError(f"{name} selection must be a boolean [batch,token] tensor")
            if not mask.any(-1).all():
                raise ValueError(f"each example must select at least one {name} token")
        if (sender_mask.shape[0] != donor_mask.shape[0]
                or not torch.equal(sender_mask.sum(-1).cpu(), donor_mask.sum(-1).cpu())):
            raise ValueError("sender and donor selections need equal token counts in every example")
        if receiver_mask.shape != sender_mask.shape:
            raise ValueError("receiver selection must use base sequence coordinates")

        freezes = {}
        for layer, point in spec.path_heads.items():
            original = base.activations[point.site]
            validate_activation(original, "base head output")
            values = point.view(original).clone()
            if values.shape[:2] != sender_mask.shape:
                raise ValueError("head output must match the expanded base token layout")
            if layer in sender_heads:
                clean = donor.activations[point.site]
                validate_pair(original, clean)
                source = point.view(clean)
                if source.shape[:2] != donor_mask.shape:
                    raise ValueError("donor head output must match its expanded token layout")
                # Map sorted donor positions to sorted base positions PER example.
                # Final receiver replay below always reads BASE coordinates.
                for batch in range(values.shape[0]):
                    dst = sender_mask[batch].to(values.device)
                    src = donor_mask[batch].to(source.device)
                    for head in sender_heads[layer]:
                        values[batch, dst, head] = source[batch, src, head]
            frozen = values.reshape_as(original)
            def freeze(value, frozen=frozen):
                validate_pair(value, frozen)
                return frozen
            freezes[point.site] = freeze

        # C: Q/K/V are recomputed, even in a head whose output is frozen.
        # Each run owns/removes its hooks; MLPs and normalization are untouched.
        controlled = run_base(capture=receiver_sites, interventions=freezes)
        edits = {}
        for site in receiver_sites:
            cached = controlled.activations[site]
            validate_pair(base.activations[site], cached)
            if receivers == "residual":
                if cached.ndim != 3 or cached.shape[:2] != receiver_mask.shape:
                    raise ValueError("final residual must be [batch,token,residual_dim]")
                def inject(value, cached=cached):
                    validate_pair(value, cached)
                    return torch.where(receiver_mask.to(value.device)[..., None], cached, value)
            else:
                point, heads = points[site], targets[site]
                if point.view(cached).shape[:2] != receiver_mask.shape:
                    raise ValueError("receiver Q/K/V must match the expanded base token layout")
                def inject(value, cached=cached, point=point, heads=heads):
                    validate_pair(value, cached)
                    result = point.view(value).clone()
                    source = point.view(cached)
                    mask = receiver_mask.to(value.device)
                    for head in heads:
                        result[:, :, head][mask] = source[:, :, head][mask]
                    return result.reshape_as(value)
            edits[site] = inject

        # D: only receiver injections; no frozen outputs or donor sender hooks.
        intervention = run_base(interventions=edits)
        baseline_score, donor_score, intervention_score = score(base), score(donor), score(intervention)
        for value in (baseline_score, donor_score, intervention_score):
            if (not isinstance(value, torch.Tensor) or value.ndim > 1
                    or (value.ndim == 1 and value.shape != (sender_mask.shape[0],))
                    or not torch.isfinite(value).all()):
                raise ValueError("path metric must return a finite scalar or [batch] tensor")
        if baseline_score.shape != intervention_score.shape or donor_score.shape != baseline_score.shape:
            raise ValueError("path metric must return the same shape in base, donor, and intervention runs")
        return ProbeResult(self.name, {
            "baseline_score": baseline_score, "donor_score": donor_score,
            "intervention_score": intervention_score,
            "effect": analysis_difference(intervention_score, baseline_score),
        }, {
            "algorithm": "Wang et al. IOI-style path patching (Appendix B)",
            "stages": 4, "effect_measured": True,
            "effect_sign": "intervention_score - baseline_score",
            "control": "freeze all head outputs; replace sender; recompute MLP and LayerNorm",
            "final_run": "receiver injections only",
            "senders": [list(x) for x in senders],
            "receivers": receivers if isinstance(receivers, str) else [list(x) for x in receivers],
            "sender_positions": sender_mask.nonzero().cpu().tolist(),
            "donor_positions": donor_mask.nonzero().cpu().tolist(),
            "receiver_positions": receiver_mask.nonzero().cpu().tolist(),
            "position_axes": ["batch", "expanded_token"],
            "sweep": "joint sender set to joint receiver set",
        })

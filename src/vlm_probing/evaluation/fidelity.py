"""Explicit masked language-model fidelity under native residual replacement."""
import torch
import torch.nn.functional as F

from .types import EvaluationReport, LanguageFidelityBatch


def language_fidelity(method, batches, *, provenance=None):
    """Measure masked ΔNLL, KL and reconstruction loss recovered.

    Uses ORIGINAL model, SAE RECONSTRUCTION and native ZERO-residual controls.
    Zeroing SAE features is not the zero-residual control because decoder biases
    can remain. Token labels and prediction masks must be supplied explicitly;
    this function never guesses multimodal or padding alignment.
    """
    from .sae import _dictionary, _frozen_eval, _operation, _ids
    dictionary = _dictionary(method)
    probe = getattr(method, "probe", None)
    if probe is None or probe.spec.output_kind != "language":
        raise ValueError("language_fidelity requires a language-output model-bound SAE")
    site = probe.spec.residuals[method.layer]
    totals = {"n_tokens": 0, "baseline_nll_sum": 0., "reconstruction_nll_sum": 0.,
              "zero_nll_sum": 0., "kl_sum": 0.}
    examples = []
    seen_ids = set()
    with _frozen_eval(dictionary, probe.model):
        for batch in batches() if callable(batches) else batches:
            if not isinstance(batch, LanguageFidelityBatch):
                raise TypeError("use LanguageFidelityBatch with explicit token_ids and score_mask")
            result = _operation(method, "reconstruct").run(batch.inputs)
            baseline = result.tensors["baseline_logits"]
            reconstruction = result.tensors["edited_logits"]
            token_ids = torch.as_tensor(batch.token_ids).detach().cpu()
            mask = torch.as_tensor(batch.score_mask).detach().cpu()
            if (baseline.ndim != 3 or token_ids.shape != baseline.shape[:2]
                    or token_ids.dtype not in (torch.int32, torch.int64)
                    or mask.dtype != torch.bool or mask.shape != (baseline.shape[0], baseline.shape[1] - 1)
                    or not mask.any()):
                raise ValueError("token_ids must match logit coordinates; score_mask must select valid shifted predictions")
            scored_targets = token_ids[:, 1:][mask]
            if scored_targets.min() < 0 or scored_targets.max() >= baseline.shape[-1]:
                raise ValueError("scored targets must contain vocabulary IDs; exclude image placeholders using score_mask")
            ids = _ids(batch.ids, len(baseline), "")
            if len(set(ids)) != len(ids) or seen_ids & set(ids):
                raise ValueError("language fidelity requires unique original example IDs")
            seen_ids.update(ids)
            groups = ids if batch.group_ids is None else _ids(batch.group_ids, len(ids), "", name="group_ids")
            positions = result.tensors["positions"]

            def zero_selected(hidden):
                changed = hidden.clone()
                selected = positions.to(hidden.device)
                changed[selected[:, 0], selected[:, 1]] = 0
                return changed

            zero = probe._run(batch.inputs, interventions={site: zero_selected}).logits
            # Compute only selected positions; full [B,T,V] log-softmax is avoided.
            target = scored_targets.to(device=baseline.device, dtype=torch.long)
            # Public ProbeResult tensors may have been exported to CPU while
            # the zero-control trace remains on the native model device. Index
            # and normalize locally, then align selected distributions only.
            logp = [F.log_softmax(logits[:, :-1][mask.to(logits.device)].float(), dim=-1).to(baseline.device)
                    for logits in (baseline, reconstruction, zero)]
            nll = [-value.gather(-1, target[:, None]).squeeze(-1) for value in logp]
            kl = (logp[0].exp() * (logp[0] - logp[1])).sum(-1)
            if any(not torch.isfinite(value).all() for value in (*nll, kl)):
                raise ValueError("selected language predictions produced nonfinite NLL or KL")
            owners = mask.nonzero()[:, 0]
            for row, (sample_id, group_id) in enumerate(zip(ids, groups)):
                selected = (owners == row).to(nll[0].device)
                count = int(selected.sum())
                if not count:
                    continue
                values = [value[selected].double().sum().item() for value in nll]
                kl_sum = kl[selected].double().sum().item()
                examples.append({"id": sample_id, "group_id": group_id, "n_tokens": count,
                    "baseline_nll": values[0] / count, "reconstruction_nll": values[1] / count,
                    "zero_nll": values[2] / count, "delta_nll": (values[1] - values[0]) / count,
                    "kl_baseline_to_reconstruction": kl_sum / count})
                totals["n_tokens"] += count
                for key, value in zip(("baseline_nll_sum", "reconstruction_nll_sum", "zero_nll_sum"), values):
                    totals[key] += value
                totals["kl_sum"] += kl_sum
    n = totals["n_tokens"]
    if not n:
        raise ValueError("language fidelity requires selected prediction tokens")
    baseline, reconstructed, zero = [totals[key] / n for key in
                                   ("baseline_nll_sum", "reconstruction_nll_sum", "zero_nll_sum")]
    denominator = zero - baseline
    summary = {**totals, "n_examples": len(examples), "baseline_nll": baseline,
        "reconstruction_nll": reconstructed, "zero_nll": zero,
        "delta_nll": reconstructed - baseline, "kl_baseline_to_reconstruction": totals["kl_sum"] / n,
        "loss_recovered": (zero - reconstructed) / denominator if abs(denominator) > 1e-12 else None,
        "loss_recovered_denominator": denominator}
    return EvaluationReport(summary, examples, provenance={**(provenance or {}),
        "evaluation_kind": "language_fidelity", "binding": getattr(dictionary, "binding", {}),
        "token_scoring": "logits[:, :-1] predict token_ids[:, 1:], explicit boolean score_mask",
        "zero_control": "native selected residual vectors set to exact zero",
        "loss_recovered_definition": "(NLL_zero - NLL_reconstruction)/(NLL_zero - NLL_original)",
        "aggregation": "token-weighted; do not average per-batch losses",
        "sae_and_backbone_frozen": True})

"""Native, frozen SAE evaluation on model inputs or replayable activation caches.

FVU uses a single global centered denominator, never an average of batch FVU.
No upstream SAE toolkit is imported and no optimizer or calibration is run.
"""
from contextlib import contextmanager
from dataclasses import asdict
from collections.abc import Iterator, Mapping

import torch

from .types import ActivationBatch, EvaluationBatch, EvaluationReport, SAEEvaluationConfig, json_value
from ._statistics import ReconstructionStatistics


@contextmanager
def _frozen_eval(*modules):
    states = []
    first = modules[0] if modules else None
    buffers = ({key: value.detach().clone() for key, value in first.named_buffers()}
               if isinstance(first, torch.nn.Module) else {})
    seen = set()
    for root in modules:
        if isinstance(root, torch.nn.Module):
            for module in root.modules():
                if id(module) not in seen:
                    states.append((module, module.training))
                    seen.add(id(module))
            root.eval()
    try:
        with torch.no_grad():
            yield
    finally:
        for module, training in states:
            module.training = training
        changed = []
        for key, before in buffers.items():
            current = first.get_buffer(key)
            if current.shape != before.shape or current.dtype != before.dtype or not torch.equal(current, before):
                changed.append(key)
                # Even a misbehaving third-party dictionary cannot leave the
                # caller's calibration/threshold buffers silently changed.
                parent_name, _, name = key.rpartition(".")
                parent = first.get_submodule(parent_name) if parent_name else first
                parent._buffers[name] = before
        if changed:
            raise RuntimeError("evaluation changed SAE buffers; restored originals: " + ", ".join(changed))


def _dictionary(method):
    dictionary = getattr(method, "sae", method)
    if dictionary is None or not callable(getattr(dictionary, "encode", None)):
        raise TypeError("supply a fitted model-bound SAE method or an encode/decode dictionary")
    if not callable(getattr(dictionary, "decode", None)):
        raise TypeError("dictionary must provide encode and decode")
    if getattr(dictionary, "is_fitted", True) is False:
        raise ValueError("fit or load a dictionary before evaluation; evaluation does not fit it")
    return dictionary


def _encode_decode(dictionary, values):
    contextual = callable(getattr(dictionary, "encode_with_context", None))
    if contextual != callable(getattr(dictionary, "decode_with_context", None)):
        raise ValueError("context-aware dictionaries require paired encode/decode context methods")
    parameter = getattr(dictionary, "activation_parameter", None)
    if parameter is None:
        parameter = next(dictionary.parameters(), None)
    native = values if parameter is None else values.to(parameter)
    if contextual:
        features, context = dictionary.encode_with_context(native)
        reconstruction = dictionary.decode_with_context(features, context)
    else:
        features = dictionary.encode(native)
        reconstruction = dictionary.decode(features)
    if (not isinstance(features, torch.Tensor) or features.ndim != 2
            or len(features) != len(values) or features.shape[1] < 1
            or not torch.isfinite(features).all()):
        raise ValueError("dictionary encode must return finite [position,feature] values")
    if (not isinstance(reconstruction, torch.Tensor) or reconstruction.shape != values.shape
            or not torch.isfinite(reconstruction).all()):
        raise ValueError("dictionary decode must return finite native-width activations")
    # Metrics compare to the ORIGINAL cached activation, before dictionary dtype casting.
    return features, reconstruction


def _ids(value, count, prefix, *, name="ids"):
    if value is None:
        return [f"{prefix}:{i}" for i in range(count)]
    if isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must contain one ID per original example")
    result = list(value)
    if len(result) != count or any(item is None for item in result):
        raise ValueError(f"{name} must contain {count} non-null IDs")
    result = [str(item) for item in result]
    if any(not item for item in result):
        raise ValueError(f"{name} cannot contain empty IDs")
    return result


def _activation_rows(batch, index):
    if isinstance(batch, torch.Tensor):
        batch = ActivationBatch(batch)
    elif isinstance(batch, Mapping):
        batch = ActivationBatch(**batch)
    if not isinstance(batch, ActivationBatch):
        raise TypeError("activation batches must be tensors or ActivationBatch records")
    values = batch.activations
    if (not isinstance(values, torch.Tensor) or not values.is_floating_point()
            or values.ndim not in {2, 3} or values.shape[-1] < 1
            or not torch.isfinite(values).all()):
        raise ValueError("activations must be finite floating [position,hidden] or [example,token,hidden]")
    sample_ids = _ids(batch.ids, len(values), f"batch-{index}")
    groups = (sample_ids if batch.group_ids is None else
              _ids(batch.group_ids, len(values), "", name="group_ids"))
    if values.ndim == 3:
        repeats = values.shape[1]
        ids = [item for item in sample_ids for _ in range(repeats)]
        group_ids = [item for item in groups for _ in range(repeats)]
    else:
        ids, group_ids = sample_ids, groups
    if batch.mask is not None:
        mask = torch.as_tensor(batch.mask)
        if mask.dtype != torch.bool or tuple(mask.shape) != tuple(values.shape[:-1]):
            raise ValueError("mask must be boolean with the activation leading shape")
        selected = mask.reshape(-1).cpu().tolist()
        ids = [item for item, keep in zip(ids, selected) if keep]
        group_ids = [item for item, keep in zip(group_ids, selected) if keep]
        values = values.reshape(-1, values.shape[-1])[mask.reshape(-1).to(values.device)]
    else:
        values = values.reshape(-1, values.shape[-1])
    return values, ids, group_ids


def _contract(method):
    dictionary = _dictionary(method)
    binding = getattr(dictionary, "binding", {})
    # Different dictionaries can have different calibration IDs, but must attach
    # to exactly the same frozen model, preprocessing, hook and selected tokens.
    keys = ("model_id", "tokenizer_id", "preprocessing_id", "layer", "hook", "tokens")
    value = {key: binding.get(key) for key in keys}
    value["tokens"] = getattr(method, "tokens", value["tokens"])
    return json_value(value)


def _finish(statistics, method, config, provenance, *, kind, scores=None):
    summary, examples, features = statistics.finish()
    if scores:
        for name, values in scores.items():
            summary[name] = sum(values) / len(values) if values else None
    dictionary = _dictionary(method)
    recorded = {**(provenance or {}), "evaluation_kind": kind,
        "configuration": asdict(config), "binding": getattr(dictionary, "binding", {}),
        "source": getattr(dictionary, "source_metadata", {}),
        "comparison_contract": _contract(method),
        "selected_position_ids_sha256": statistics.sample_digest.hexdigest(),
        "native_activation_values_sha256": statistics.activation_digest.hexdigest(),
        "fvu_reference": "caller_supplied_mean" if statistics.reference_mean is not None else "evaluation_global_mean",
        "reference_mean": statistics.reference_mean,
        "inactive_definition": "zero nonzero activations on this evaluation split",
        "activity_definition": "feature != 0, including signed activations",
        "sae_and_backbone_frozen": True}
    return EvaluationReport(summary, examples, features, recorded)


def evaluate_activations(sae, batches, *, config=None, reference_mean=None, provenance=None):
    """Evaluate cached activations once, without storing all sparse feature rows.

    A replayable cache can be evaluated by many dictionaries using
    :func:`compare_activations`. Provide explicit IDs for comparisons across
    different batch partitions. A calibration mean fixes the FVU reference;
    otherwise the exact mean across all selected evaluation positions is used.
    Fidelity and intervention need full model inputs and are rejected here.
    """
    config = config or SAEEvaluationConfig()
    if set(config.metrics) & {"fidelity", "intervention"}:
        raise ValueError("fidelity/intervention require model inputs; use evaluate()")
    dictionary = _dictionary(sae)
    statistics = ReconstructionStatistics(reference_mean)
    with _frozen_eval(dictionary):
        for index, batch in enumerate(batches() if callable(batches) else batches):
            values, ids, groups = _activation_rows(batch, index)
            for start in range(0, len(values), config.chunk_size):
                stop = start + config.chunk_size
                features, reconstruction = _encode_decode(dictionary, values[start:stop])
                statistics.add(values[start:stop], features, reconstruction,
                               ids[start:stop], groups[start:stop])
    return _finish(statistics, sae, config, provenance, kind="cached_activations")


def _operation(method, operation, **kwargs):
    factory = getattr(method, operation, None)
    if callable(factory):
        return factory(**kwargs)
    probe = getattr(method, "probe", None)
    if probe is None:
        raise TypeError("model evaluation requires a model-bound SAE method")
    return getattr(probe.sae, operation)(layer=method.layer, tokens=method.tokens,
        sae=method.sae, binding=method.sae.binding, **kwargs)


def _scores(result, ids, name, examples, collected):
    tensors = result.tensors
    for key in ("baseline_score", "edited_score", "score_change"):
        if key not in tensors:
            raise ValueError("a per-example task metric is required for fidelity/intervention")
        values = tensors[key].detach().double().cpu().reshape(-1)
        if len(values) != len(ids) or not torch.isfinite(values).all():
            raise ValueError("task metric must return one finite score per original example")
        label = name + "." + key
        collected.setdefault(label, []).extend(values.tolist())
        for sample_id, value in zip(ids, values.tolist()):
            examples[sample_id][label] = value


def matched_random_features(feature_statistics, feature_ids, *, repeats=3, seed=0):
    """Pick count/frequency-matched controls using INDEPENDENT calibration stats.

    Match each target to one of the 32 closest available firing frequencies.
    Target features and previously sampled controls are excluded. All control
    sets are disjoint; ``repeats * len(feature_ids)`` distinct non-target
    features must be available.
    Save returned IDs before evaluating the held-out split.
    """
    frequency = torch.as_tensor(feature_statistics["frequency"]).double().cpu()
    target = list(feature_ids)
    if (frequency.ndim != 1 or not len(frequency) or not torch.isfinite(frequency).all()
            or (frequency < 0).any() or (frequency > 1).any()):
        raise ValueError("frequency must be a finite one-dimensional vector in [0,1]")
    if (not target or len(set(target)) != len(target) or any(type(i) is not int or not 0 <= i < len(frequency) for i in target)
            or len(frequency) - len(target) < len(target)):
        raise ValueError("targets must be unique valid IDs with enough non-target control features")
    if type(repeats) is not int or repeats < 0:
        raise ValueError("repeats must be nonnegative")
    if repeats * len(target) > len(frequency) - len(target):
        raise ValueError("not enough distinct non-target features for disjoint random controls")
    generator = torch.Generator().manual_seed(seed)
    controls = []
    previously_selected = set(target)
    for _ in range(repeats):
        excluded = set(previously_selected)
        chosen = []
        for index in target:
            distance = (frequency - frequency[index]).abs()
            distance[list(excluded)] = float("inf")
            available = len(frequency) - len(excluded)
            candidates = distance.argsort()[:min(32, available)]
            selected = int(candidates[torch.randint(len(candidates), (1,), generator=generator)])
            chosen.append(selected)
            excluded.add(selected)
        controls.append(tuple(chosen))
        previously_selected.update(chosen)
    return tuple(controls)


def evaluate(method, batches, *, config=None, reference_mean=None, provenance=None):
    """Evaluate a loaded ``probe.sae`` method on prepared, identified batches.

    Reconstruction/sparsity is always recorded. Fidelity/intervention additionally
    require ``EvaluationBatch.metric``. Random feature IDs may be chosen on a
    separate calibration split with :func:`matched_random_features`; otherwise
    controls are matched only by feature count and are labeled accordingly.
    """
    config = config or SAEEvaluationConfig()
    dictionary = _dictionary(method)
    statistics = ReconstructionStatistics(reference_mean)
    scores = {}
    seen_ids = set()
    random_controls = tuple(tuple(row) for row in config.random_feature_ids)
    if "intervention" in config.metrics:
        width = dictionary.config.d_sae
        if any(index >= width for index in config.feature_ids):
            raise ValueError("feature_ids exceed dictionary width")
        if not random_controls and config.random_control_repeats:
            pool = [i for i in range(width) if i not in config.feature_ids]
            required = len(config.feature_ids) * config.random_control_repeats
            if len(pool) < required:
                raise ValueError("not enough distinct non-target features for count-matched controls")
            generator = torch.Generator().manual_seed(config.seed)
            order = torch.randperm(len(pool), generator=generator)[:required].tolist()
            size = len(config.feature_ids)
            random_controls = tuple(tuple(pool[i] for i in order[start:start + size])
                                    for start in range(0, required, size))
        for row in random_controls:
            if (len(row) != len(config.feature_ids) or len(set(row)) != len(row)
                    or any(type(i) is not int or not 0 <= i < width or i in config.feature_ids for i in row)):
                raise ValueError("random_feature_ids must be valid count-matched disjoint control sets")
        flattened_controls = [index for row in random_controls for index in row]
        if len(flattened_controls) != len(set(flattened_controls)):
            raise ValueError("random_feature_ids must be disjoint across all control sets")
    model = getattr(getattr(method, "probe", None), "model", None)
    with _frozen_eval(dictionary, model):
        for index, batch in enumerate(batches() if callable(batches) else batches):
            if isinstance(batch, Mapping):
                batch = EvaluationBatch(**batch)
            if not isinstance(batch, EvaluationBatch):
                raise TypeError("model evaluation needs EvaluationBatch(inputs, ids, metric=...)")
            result = method.run(batch.inputs)
            positions = result.tensors["positions"].detach().cpu()
            # Keep backbone precision as the metric reference. ``activation``
            # retains the dictionary's inference dtype for compatibility;
            # casting to a lower SAE dtype must count toward reconstruction
            # error rather than silently changing the evaluated observation.
            values = result.tensors.get("native_activation", result.tensors["activation"])
            features = result.tensors["features"]
            reconstruction = result.tensors["reconstruction"]
            supplied_ids = list(batch.ids) if not isinstance(batch.ids, (str, bytes)) else batch.ids
            ids = _ids(supplied_ids, len(supplied_ids), f"batch-{index}")
            if len(set(ids)) != len(ids) or seen_ids.intersection(ids):
                raise ValueError("model evaluation requires unique example IDs; use group_ids for related examples")
            seen_ids.update(ids)
            groups = ids if batch.group_ids is None else _ids(batch.group_ids, len(ids), "", name="group_ids")
            if (positions.ndim != 2 or positions.shape[1] != 2 or len(positions) != len(values)
                    or (len(positions) and (positions[:, 0].min() < 0 or positions[:, 0].max() >= len(ids)))):
                raise ValueError("result positions must map selected rows to original example IDs")
            if set(positions[:, 0].tolist()) != set(range(len(ids))):
                raise ValueError("each model example must have at least one selected SAE position")
            row_ids = [ids[i] for i in positions[:, 0].tolist()]
            row_groups = [groups[i] for i in positions[:, 0].tolist()]
            for start in range(0, len(values), config.chunk_size):
                stop = start + config.chunk_size
                statistics.add(values[start:stop], features[start:stop], reconstruction[start:stop],
                               row_ids[start:stop], row_groups[start:stop])
            if set(config.metrics) & {"fidelity", "intervention"}:
                if batch.metric is None:
                    raise ValueError("fidelity/intervention require EvaluationBatch.metric")
                reconstructed = _operation(method, "reconstruct").run(batch.inputs, metric=batch.metric)
                _scores(reconstructed, ids, "reconstruction", statistics.examples, scores)
            if "intervention" in config.metrics:
                identity = _operation(method, "edit", feature_ids=list(config.feature_ids), scale=1.,
                                      mode="error_preserving").run(batch.inputs, metric=batch.metric)
                _scores(identity, ids, "no_op", statistics.examples, scores)
                for scale in config.intervention_scales:
                    edited = _operation(method, "edit", feature_ids=list(config.feature_ids), scale=scale,
                                        mode="error_preserving").run(batch.inputs, metric=batch.metric)
                    _scores(edited, ids, f"target_scale_{scale:g}", statistics.examples, scores)
                    for repeat, control in enumerate(random_controls):
                        edited = _operation(method, "edit", feature_ids=list(control), scale=scale,
                                            mode="error_preserving").run(batch.inputs, metric=batch.metric)
                        _scores(edited, ids, f"random_{repeat}_scale_{scale:g}", statistics.examples, scores)
    provenance = {**(provenance or {}), "random_feature_ids": random_controls,
                  "random_control_matching": "caller_calibration_selection" if config.random_feature_ids else "feature_count_only"}
    return _finish(statistics, method, config, provenance, kind="model_inputs", scores=scores)


def _compare(methods, batches, evaluator, **kwargs):
    if not isinstance(methods, Mapping) or not methods:
        raise ValueError("methods must be a nonempty name-to-method mapping")
    if not callable(batches) and isinstance(batches, Iterator):
        raise ValueError("comparison needs replayable batches or a factory returning a fresh iterator")
    contracts = [_contract(method) for method in methods.values()]
    if any(contract != contracts[0] for contract in contracts[1:]):
        raise ValueError("comparison requires the same backbone, preprocessing, hook and token scope")
    reports = {}
    alignment = None
    for name, method in methods.items():
        report = evaluator(method, batches, **kwargs)
        signature = (report.summary["n_positions"], report.provenance["selected_position_ids_sha256"],
                     report.provenance["native_activation_values_sha256"])
        if alignment is not None and signature != alignment:
            raise ValueError("comparison batches changed example IDs/order, groups or selected positions")
        alignment = signature
        reports[str(name)] = report
    return reports


def compare(methods, batches, *, config=None, reference_mean=None, provenance=None):
    """Evaluate named methods on identical replayed model inputs; return reports."""
    return _compare(methods, batches, evaluate, config=config, reference_mean=reference_mean,
                    provenance=provenance)


def compare_activations(methods, batches, *, config=None, reference_mean=None, provenance=None):
    """Compare named dictionaries using one shared, replayable activation cache."""
    return _compare(methods, batches, evaluate_activations, config=config,
                    reference_mean=reference_mean, provenance=provenance)


evaluate_sae = evaluate
compare_saes = compare

# Additional evaluations have explicit inputs/capabilities instead of guessing
# labels or image geometry from a reconstruction batch.
from .localization import localization
from .probing import sparse_probe, pca_features
from .fidelity import language_fidelity
from .types import LanguageFidelityBatch

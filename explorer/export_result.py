"""Export VLMProbe results for local import into Probe Explorer.

Run from the repository root, using the Python environment that ran the probes.
Point the CLI at trusted directories written by ``ProbeResult.save()``. Results
listed together should use the same input, preprocessing, and batch ordering.

    python explorer/export_result.py outputs/logit outputs/grad_cam \
        --output comparison.json --target-token-id 123 \
        --query 51 --image processed.png --grid 2 3 \
        --visual-positions 8 9 10 11 12 13

From Python, call ``build_bundle([result], ...)`` with live ProbeResult objects
and write the returned dictionary with ``json.dump(..., allow_nan=False)``.
"""

from __future__ import annotations

import argparse
import base64
from dataclasses import dataclass
from io import BytesIO
import json
from pathlib import Path
import re
from tempfile import NamedTemporaryFile

import torch


LABELS = {
    "activation_patching": "Activation patching",
    "attention_attribution": "Attention Attribution",
    "attention_grad_cam": "Grad-CAM",
    "attention_knockout": "Attention knockout",
    "attention_lens": "Attention Lens",
    "attention_profile": "Attention profile",
    "attention_relevance": "Attention relevance",
    "attention_reweight": "Attention reweighting",
    "attention_rollout": "Attention rollout",
    "attention_temperature": "Attention temperature",
    "attribution_patching": "Attribution patching",
    "beyond_intuition": "Beyond Intuition",
    "chefer_transformer_attribution": "Chefer transformer attribution",
    "chefer_rollout": "Chefer LRP rollout",
    "chefer_last_layer": "Chefer last layer",
    "eap_ig": "EAP-IG",
    "eap_ig_faithfulness": "Circuit faithfulness",
    "edge_patching": "Edge patching",
    "embed_lens": "Embed Lens",
    "head_logit_attribution": "Head logit attribution",
    "jacobian_lens": "Jacobian Lens",
    "logit_lens": "Logit Lens",
    "mj_lens": "MJ-Lens",
    "patchscope": "Patchscope",
    "path_patching": "Path patching",
    "sae.edit": "SAE feature edit",
    "sae.features": "SAE feature activations",
    "sae.profile": "SAE feature profile",
    "sae.reconstruct": "SAE reconstruction",
    "steering": "Steering",
    "transition_attention_maps": "TAM",
    "tuned_lens": "Tuned Lens",
}


@dataclass(frozen=True)
class SavedResult:
    method: str
    tensors: dict[str, torch.Tensor]
    metadata: dict


def load_saved_result(directory: Path) -> SavedResult:
    """Read a trusted result saved by ProbeResult.save()."""
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    if not isinstance(metadata, dict) or not isinstance(metadata.get("method"), str):
        raise ValueError(f"{directory}: metadata.json has no method name")
    tensors = torch.load(directory / "tensors.pt", map_location="cpu", weights_only=True)
    if not isinstance(tensors, dict) or any(not isinstance(t, torch.Tensor) for t in tensors.values()):
        raise ValueError(f"{directory}: tensors.pt does not contain a tensor dictionary")
    method = metadata.pop("method")
    if metadata.get("backend") == "explicit_author_ViT" and not method.startswith("chefer_"):
        method = f"chefer_{method}"
    return SavedResult(method, tensors, metadata)


@dataclass(frozen=True)
class Selection:
    batch: int
    query: int | None
    head: int | None
    target_token_id: int | None
    grid_shape: tuple[int, int] | None
    visual_positions: tuple[int, ...] | None
    feature_ids: tuple[int, ...] | None
    top_k: int
    tokenizer: object | None

    def batch_values(self, values):
        return values.detach().cpu().select(0, self.batch)

    def token_label(self, index, alphabet="token"):
        if self.tokenizer is not None and alphabet == "token":
            return self.tokenizer.decode([index])
        return f"{alphabet.capitalize()} {index}"


def _json_values(values):
    """Keep finite values and mark missing tensor cells as JSON null."""
    def scrub(value):
        if isinstance(value, list):
            return [scrub(item) for item in value]
        if isinstance(value, float) and not abs(value) < float("inf"):
            return None
        return value

    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().tolist()
    return scrub(values)


def _matrix(title, values, rows, columns, unit, *, signed=False, rank=False):
    return {"type": "matrix", "title": title, "values": _json_values(values),
            "rows": rows, "columns": columns, "unit": unit,
            "signed": signed, "lowerIsBetter": rank}


def _series(title, x, label, values, unit, *, signed=False):
    return {"type": "series", "title": title, "x": x,
            "series": [{"label": label, "values": _json_values(values)}],
            "unit": unit, "signed": signed}


def _table(title, columns, rows):
    return {"type": "table", "title": title, "columns": columns, "rows": rows}


def _bars(title, labels, values, unit, *, signed=False):
    return {"type": "bars", "title": title, "entries": [
        {"label": label, "value": value} for label, value in zip(labels, _json_values(values))],
        "unit": unit, "signed": signed}


def _layers(result, count):
    layers = result.metadata.get("layers")
    if isinstance(layers, (list, tuple)) and len(layers) == count:
        return [f"Layer {layer}" for layer in layers]
    layer = result.metadata.get("layer")
    if count == 1 and layer is not None:
        return [f"Layer {layer}"]
    return [f"Captured slice {index}" for index in range(count)]


def _packed_positions(result, selection):
    positions = result.tensors["positions"].detach().cpu()
    rows = torch.where(positions[:, 0] == selection.batch)[0]
    if len(rows) == 0:
        raise ValueError("selected batch item has no captured positions")
    return rows, positions[rows, 1].tolist()


def _lens_views(result, selection):
    tensors = result.tensors
    unpacked = result.method == "patchscope"
    if unpacked:
        values = tensors["logits"].detach().cpu().select(1, selection.batch)
        positions = list(range(values.shape[1]))
    else:
        packed, positions = _packed_positions(result, selection)
        if "similarities" in tensors:
            similarities = tensors["similarities"].detach().cpu()[0, packed][-64:]
            token_ids = tensors["token_ids"].detach().cpu()[0, packed][-64:]
            count = min(selection.top_k, token_ids.shape[1])
            rows = [
                [position, rank + 1, index, selection.token_label(index), score]
                for position, ids, scores in zip(positions[-64:], token_ids.tolist(), _json_values(similarities))
                for rank, (index, score) in enumerate(zip(ids[:count], scores[:count]))
            ]
            return [_table("Nearest vocabulary embeddings", ["Position", "Rank", "Token ID", "Token", "Cosine"], rows)]
        packed = packed[-64:]
        positions = positions[-64:]
        if selection.head is not None and "head_logits" in tensors:
            values = tensors["head_logits"].detach().cpu()[:, packed, selection.head]
        else:
            values = tensors["logits"].detach().cpu()[:, packed]
    if values.ndim != 3 or values.shape[1] == 0:
        raise ValueError("expected model-bound layer/position/vocabulary logits")
    alphabet = "code" if result.metadata.get("target_modality") == "visual" else "token"
    layers = _layers(result, values.shape[0])
    values = values[:, -64:]
    positions = positions[-64:]
    views = []
    target = selection.target_token_id
    if target is not None:
        if not 0 <= target < values.shape[-1]:
            raise ValueError(f"target {alphabet} ID must be inside the readout alphabet")
        chosen = values[..., target]
        ranks = 1 + (values > chosen.unsqueeze(-1)).sum(-1)
        views.append(_matrix(f"{selection.token_label(target, alphabet)} rank", ranks, layers,
                             [str(position) for position in positions], "Rank", rank=True))
    show_at = -1
    if unpacked and result.metadata.get("layer_results"):
        position = result.metadata["layer_results"][0].get("target_position")
        if position in positions:
            show_at = positions.index(position)
    last_values, last_ids = values[:, show_at].topk(min(selection.top_k, values.shape[-1]), dim=-1)
    rows = [[layer, rank + 1, index, selection.token_label(index, alphabet), score]
            for layer, ids, scores in zip(layers, last_ids.tolist(), _json_values(last_values))
            for rank, (index, score) in enumerate(zip(ids, scores))]
    views.append(_table(f"Top readouts at position {positions[show_at]}",
                        ["Layer", "Rank", f"{alphabet.capitalize()} ID", alphabet.capitalize(), "Logit"], rows))
    return views


def _mask_positions(result, selection, key):
    mask = result.tensors.get(key)
    if mask is None:
        return None
    if mask.ndim == 3:  # packed attention profile
        mask = mask[0]
    return torch.where(selection.batch_values(mask))[0].tolist()


def _query_position(result, selection, count):
    available = _mask_positions(result, selection, "query_mask")
    if available is None:
        available = _mask_positions(result, selection, "valid_queries")
    query = selection.query
    if query is None:
        if available:
            return available[-1]
        if available == []:
            raise ValueError("selected batch item has no captured query positions")
        return count - 1
    if query >= count or (available is not None and query not in available):
        raise ValueError(f"query position {query} was not measured in this result")
    return query


def _spatial(title, values, unit, signed, **identifiers):
    return {"type": "spatial", "title": title, "values": _json_values(values),
            "unit": unit, "signed": signed, **identifiers}


def _attention_map_views(result, selection, image):
    tensors = result.tensors
    if "patch_relevance" in tensors:
        values = selection.batch_values(tensors["patch_relevance"])
        if values.ndim != 2:
            raise ValueError("patch relevance must have measured height and width axes")
        if selection.grid_shape is not None and values.shape != selection.grid_shape:
            raise ValueError("declared grid differs from measured patch relevance shape")
        if image is not None:
            return [_spatial("Patch relevance", values, "Relevance", bool((values < 0).any()))]
        return [_matrix("Patch relevance", values, [str(i) for i in range(values.shape[0])],
                        [str(i) for i in range(values.shape[1])], "Relevance", signed=bool((values < 0).any()))]

    map_fields = ("attention", "cam", "token_attribution", "layer_rollouts", "layer_relevance", "map", "relevance", "rollout")
    field = next((name for name in map_fields if name in tensors), None)
    if field is None:
        return []
    values = tensors[field].detach().cpu()
    if field == "attention":
        if values.ndim == 4:
            values = values.unsqueeze(0)
        query = _query_position(result, selection, values.shape[-2])
        values = values[:, selection.batch, :, query]
        if selection.head is None:
            accumulator = torch.float64 if values.dtype == torch.float64 else torch.float32
            values = values.mean(dim=1, dtype=accumulator)
        else:
            values = values[:, selection.head]
    else:
        if field == "token_attribution" and selection.head is not None and "attribution" in tensors:
            per_head = tensors["attribution"].detach().cpu()
            if per_head.ndim == 4:
                per_head = per_head.unsqueeze(0)
            values = per_head[:, :, selection.head]
        if values.ndim == 3:
            values = values.unsqueeze(0)
        if values.ndim != 4:
            raise ValueError("expected a model-bound query/key attention map")
        query = _query_position(result, selection, values.shape[-2])
        values = values[:, selection.batch, query]
    key_mask = _mask_positions(result, selection, "key_mask")
    if key_mask == []:
        raise ValueError("selected batch item has no measured key positions")
    signed = field == "token_attribution" or bool((values < 0).any())
    unit = "Attention probability" if field == "attention" else "Attribution score"
    if field in {"rollout", "layer_rollouts"}:
        unit = "Rollout weight"
    scope = f"Query {query}"
    if field == "attention":
        scope += " · mean across heads" if selection.head is None else f" · head {selection.head}"
    elif field == "token_attribution":
        scope += " · sum across heads" if selection.head is None else f" · head {selection.head}"
    layers = _layers(result, values.shape[0])
    if field in {"map", "relevance", "rollout"} and values.shape[0] == 1:
        layers = ["Final map"]
    views = []
    visual = selection.visual_positions
    if visual is not None:
        if max(visual) >= values.shape[-1]:
            raise ValueError("visual_positions exceeds the key axis")
        if key_mask is not None and not set(visual).issubset(key_mask):
            raise ValueError("visual positions include keys outside the measured selector")
    if image is not None and selection.grid_shape is not None and visual is None:
        raise ValueError("query/key overlays require explicit visual positions in native grid order")
    if image is not None and selection.grid_shape is not None:
        grids = values[:, list(visual)].reshape(-1, *selection.grid_shape)
        native_layers = result.metadata.get("layers")
        if not isinstance(native_layers, (list, tuple)) or len(native_layers) != len(grids):
            native_layers = [None] * len(grids)
        for layer, native_layer, grid in reversed(list(zip(layers, native_layers, grids))):
            view = _spatial(f"{layer} · {scope}", grid, unit, signed)
            if type(native_layer) is int:
                view["layer"] = native_layer
            views.append(view)
    keys = visual or key_mask or range(values.shape[-1])
    keys = list(keys)[:128]
    if keys:
        views.append(_matrix(f"Key positions · {scope}", values[:, keys], layers,
                             [str(key) for key in keys], unit, signed=signed))
    return views


def _attention_views(result, selection, image):
    tensors = result.tensors
    if result.method == "head_logit_attribution":
        values = tensors["head_logits"].detach().cpu().select(1, selection.batch)
        query = _query_position(result, selection, values.shape[-2])
        token = selection.target_token_id
        if token is None or token >= values.shape[-1]:
            raise ValueError("head logit attribution requires a valid --target-token-id")
        columns = [f"Head {index}" for index in range(values.shape[1])]
        return [_matrix(f"{selection.token_label(token)} contribution · Query {query}",
                        values[:, :, query, token], _layers(result, values.shape[0]), columns,
                        "Logit contribution", signed=True)]
    views = _attention_map_views(result, selection, image)
    if "entropy" in tensors:
        entropy = tensors["entropy"].detach().cpu()
        if entropy.ndim == 3:
            entropy = entropy.unsqueeze(0)
        query = _query_position(result, selection, entropy.shape[-1])
        by_head = entropy[:, selection.batch, :, query]
        labels = _layers(result, by_head.shape[0])
        if "group_mass" in tensors:
            masses = tensors["group_mass"].detach().cpu()
            groups = result.metadata.get("groups")
            if masses.ndim == 4:
                masses = masses.unsqueeze(0)
            elif groups is None and result.metadata.get("layer_results"):
                groups = result.metadata["layer_results"][0].get("groups")
            group_labels = groups or [f"Group {index}" for index in range(masses.shape[-1])]
            by_group = masses[:, selection.batch, :, query]
            pooled = by_group.mean(1) if selection.head is None else by_group[:, selection.head]
            head = "mean across heads" if selection.head is None else f"head {selection.head}"
            views.append(_matrix(f"Group mass · Query {query} · {head}", pooled, labels, group_labels,
                                 "Attention mass"))
        views.append(_matrix(f"Entropy · Query {query}", by_head, labels,
                             [f"Head {index}" for index in range(by_head.shape[1])], "Nats"))
    return views


def _metric_name(result):
    metric = result.metadata.get("metric")
    if isinstance(metric, dict):
        return metric.get("name")
    return metric if isinstance(metric, str) else None


def _effect(result, selection, name="effect"):
    values = result.tensors[name].detach().cpu()
    tensors, meta = result.tensors, result.metadata
    unit = _metric_name(result) or "Score change"
    if "senders" in tensors:
        coordinates = tensors["senders"].detach().cpu().tolist()
        effects = values[:, selection.batch] if values.ndim == 2 else values
        layers = sorted({layer for layer, _ in coordinates})
        heads = sorted({head for _, head in coordinates})
        grid = [[None] * len(heads) for _ in layers]
        for (layer, head), effect in zip(coordinates, effects.tolist()):
            grid[layers.index(layer)][heads.index(head)] = effect
        return [_matrix("Sender effects", grid, [f"Layer {layer}" for layer in layers],
                        [f"Head {head}" for head in heads], unit, signed=True)]
    if values.ndim == 2:
        values = values[:, selection.batch]
    elif values.ndim == 0:
        values = values.reshape(1)
    elif meta.get("joint") or result.method.startswith("sae.") or result.method == "path_patching":
        values = values[selection.batch:selection.batch + 1]
    if values.numel() == 0:
        raise ValueError("batch index exceeds the measured intervention batch")
    title = "Estimated effect" if name == "estimated_effect" else "Change from baseline"
    if len(values) == 1:
        label = "Joint intervention" if meta.get("joint") else "Selected intervention"
        if not meta.get("joint") and len(meta.get("layers", [])) == 1:
            label = _layers(result, 1)[0]
        return [_bars(title, [label], values, unit, signed=True)]
    return [_series(title, _layers(result, len(values)), "Effect", values, unit, signed=True)]


def _causal_views(result, selection):
    tensors, meta = result.tensors, result.metadata
    if "edge_scores" in tensors:
        scores = tensors["edge_scores"].detach().cpu()
        ids = scores.abs().topk(min(selection.top_k, len(scores))).indices.tolist()
        names = meta.get("edge_names") or [f"Edge {i}" for i in range(len(scores))]
        view = _bars("Largest estimated edge effects", [str(names[i]) for i in ids], scores[ids],
                     _metric_name(result) or "Edge score", signed=True)
        view["description"] = "Edge estimates from the full run batch, ordered by absolute score."
        return [view]
    if "faithfulness" in tensors:
        eligible = tensors["eligible"].detach().cpu()
        values = tensors["faithfulness"].detach().cpu()
        if values.ndim == 3:
            selected = values[:, :, selection.batch]
            included = bool(eligible[selection.batch])
        else:
            selected = values
            included = bool(eligible)
        if not included:
            raise ValueError("selected example has no measurable clean-to-corrupt normalizer")
        return [{"type": "series", "title": "Measured circuit faithfulness",
                 "x": tensors["requested_edges"].detach().cpu().tolist(),
                 "series": [{"label": label, "values": values} for label, values in
                            zip(meta["rankings"], _json_values(selected))],
                 "unit": "Recovered fraction", "signed": True}]
    if "loss_increase" in tensors and "edge_ids" in tensors:
        values = tensors["loss_increase"].detach().cpu()
        if values.ndim == 2:
            values = values[:, selection.batch]
        names = meta.get("edge_names") or tensors["edge_ids"].detach().cpu().tolist()
        return [_bars("Measured effect of edge corruption", [str(name) for name in names], values,
                      "Loss increase", signed=True)]
    if "effect" in tensors:
        return _effect(result, selection)
    if "estimated_effect" in tensors:
        views = _effect(result, selection, "estimated_effect")
        views[0]["description"] = "First-order estimate, summed over the original run batch."
        scores = tensors["token_scores"].detach().cpu()[:, selection.batch, :128]
        views.append(_matrix("Estimated effect by position", scores, _layers(result, len(scores)),
                             [str(i) for i in range(scores.shape[1])], "Attribution score", signed=True))
        return views
    return []


def _sae_views(result, selection, image):
    tensors, meta = result.tensors, result.metadata
    if "score_change" in tensors:
        views = _effect(result, selection, "score_change")
        if meta.get("feature_ids"):
            ids = ", ".join(map(str, meta["feature_ids"]))
            views[0]["description"] = f"Feature IDs {ids} · scale {meta['scale']} · {meta['mode']}"
        return views
    if "firing_frequency" in tensors:
        frequency = tensors["firing_frequency"].detach().cpu()
        ids = list(selection.feature_ids or frequency.topk(min(selection.top_k, len(frequency))).indices.tolist())
        view = _bars("Feature firing frequency", [f"Feature {i}" for i in ids], frequency[ids], "Fraction")
        view["description"] = "Aggregated over the original run's selected positions and examples."
        return [view]
    if "features" not in tensors:
        return []
    packed, positions = _packed_positions(result, selection)
    values = tensors["features"].detach().cpu()[packed]
    ids = selection.feature_ids or values.abs().amax(dim=0).topk(min(selection.top_k, values.shape[1])).indices.tolist()
    ids = list(ids)
    if max(ids) >= values.shape[1]:
        raise ValueError("feature IDs exceed the dictionary width")
    views = []
    if image is not None and selection.grid_shape is not None:
        visual = selection.visual_positions
        if visual is not None:
            rows = {position: row for row, position in enumerate(positions)}
            if not set(visual).issubset(rows):
                raise ValueError("declared visual positions were not captured by the SAE")
            spatial = values[[rows[pos] for pos in visual]]
        elif meta.get("visual_only"):
            spatial = values
        else:
            raise ValueError("mixed SAE positions require explicit visual positions for a spatial view")
        if len(spatial) != selection.grid_shape[0] * selection.grid_shape[1]:
            raise ValueError("captured SAE features do not match the declared grid")
        for feature in ids:
            grid = spatial[:, feature].reshape(selection.grid_shape)
            views.append(_spatial(f"Feature {feature}", grid, "Activation", bool((grid < 0).any()), feature=feature))
    measured = values[:128, ids].T
    views.append(_matrix("Feature activity by position", measured, [f"Feature {i}" for i in ids],
                         [str(i) for i in positions[:128]], "Activation", signed=bool((measured < 0).any())))
    frequency = (values[:, ids] != 0).float().mean(0)
    views.append(_bars("Feature firing frequency", [f"Feature {i}" for i in ids], frequency, "Fraction"))
    return views


def _encoded_image(image_path):
    from PIL import Image

    data = Path(image_path).read_bytes()
    mime_types = {"PNG": "image/png", "JPEG": "image/jpeg", "WEBP": "image/webp"}
    with Image.open(BytesIO(data)) as image:
        if image.format not in mime_types or getattr(image, "is_animated", False):
            raise ValueError("image must be a single PNG, JPEG or WebP image")
        width, height = image.size
        mime = mime_types[image.format]
        orientation = image.getexif().get(274, 1)
        if orientation != 1:
            raise ValueError("save the processed image in its display orientation first")
        image.load()
    encoded = base64.b64encode(data).decode("ascii")
    return {"src": f"data:{mime};base64,{encoded}", "width": width, "height": height, "boxes": []}


def _family(method):
    if method.startswith("sae."):
        return "sae"
    if method.endswith("_lens") or method == "patchscope":
        return "lens"
    if method in {"attention_knockout", "attention_reweight", "attention_temperature"}:
        return "causal"
    if method.startswith(("attention_", "chefer_")) or method in {
        "beyond_intuition", "transition_attention_maps", "head_logit_attribution",
    }:
        return "attention"
    return "causal"


def build_bundle(results, *, model=None, title="Imported results", prompt="", target="", metric=None,
                 image_path=None, grid_shape=None, visual_positions=None, batch_index=0,
                 query=None, head=None, target_token_id=None, feature_ids=None, top_k=5, tokenizer=None):
    """Create an Explorer JSON bundle for one input across supported ProbeResults.

    Supply the exact processed image and native grid for overlays. Key positions
    must follow its row-major order. ``query`` is an expanded token coordinate.
    Optional ``tokenizer`` decodes vocabulary IDs in live Python use. Every
    exported family becomes a separate case. Dense charts are capped to 64
    lens or 128 attention/SAE positions for browser readability.
    """
    if not isinstance(title, str) or not title.strip():
        raise ValueError("title must be a nonempty string")
    indices = (query, head, target_token_id)
    if type(batch_index) is not int or batch_index < 0 or any(
        value is not None and (type(value) is not int or value < 0) for value in indices
    ):
        raise ValueError("batch, query, head and target token ID must be nonnegative integers")
    if type(top_k) is not int or not 1 <= top_k <= 20:
        raise ValueError("top_k must be an integer from 1 to 20")
    if grid_shape is not None:
        grid_shape = tuple(grid_shape)
        if len(grid_shape) != 2 or any(type(value) is not int or value <= 0 for value in grid_shape):
            raise ValueError("grid_shape must contain two positive integers")
    if visual_positions is not None:
        visual_positions = tuple(visual_positions)
        if not visual_positions or len(set(visual_positions)) != len(visual_positions) or any(
            type(position) is not int or position < 0 for position in visual_positions
        ):
            raise ValueError("visual positions must be unique nonnegative integers")
        if grid_shape is not None and len(visual_positions) != grid_shape[0] * grid_shape[1]:
            raise ValueError("visual position count does not match the declared grid")
    if feature_ids is not None:
        feature_ids = tuple(feature_ids)
        if not feature_ids or len(set(feature_ids)) != len(feature_ids) or any(
            type(feature) is not int or feature < 0 for feature in feature_ids
        ):
            raise ValueError("feature_ids must be unique nonnegative integers")
    selection = Selection(batch_index, query, head, target_token_id, grid_shape, visual_positions,
                          feature_ids, top_k, tokenizer)
    image = _encoded_image(image_path) if image_path is not None else None
    results = list(results)
    if not results:
        raise ValueError("provide at least one ProbeResult")
    models = {result.metadata["model_id"] for result in results if result.metadata.get("model_id")}
    if len(models) > 1:
        raise ValueError("results with distinct model IDs must be exported in separate bundles")
    model = model or next(iter(models), "Unspecified model")
    cases = {}
    case_metrics = {}
    for result in results:
        family = _family(result.method)
        try:
            if family == "attention":
                views = _attention_views(result, selection, image)
            elif family == "lens":
                views = _lens_views(result, selection)
            elif family == "sae":
                views = _sae_views(result, selection, image)
            else:
                views = _causal_views(result, selection)
        except (IndexError, KeyError, RuntimeError, ValueError) as error:
            raise ValueError(f"{result.method}: {error}") from error
        if not views:
            suffix = " Rerun with a target metric to export intervention scores." if family == "sae" else ""
            raise ValueError(f"{result.method}: no supported visualization in this result.{suffix}")
        case = cases.setdefault(family, {
            "id": f"imported-{family}", "family": family, "title": title, "model": str(model),
            "prompt": prompt, "target": target,
            "source": {"path": "docs/API.md#results-and-persistence",
                       "description": f"VLMProbe result export · batch index {batch_index}"},
            "methods": [],
        })
        if image is not None:
            case["image"] = image
        method_label = LABELS.get(result.method, result.method.replace("_", " ").title())
        method_id = re.sub(r"[^a-z0-9]+", "-", result.method.lower()).strip("-")
        peers = [method for method in case["methods"] if method["id"].startswith(method_id + "-") or
                 method["id"] == method_id]
        if peers:
            method_id += f"-{len(peers) + 1}"
            method_label += f" ({len(peers) + 1})"
        result_metric = _metric_name(result)
        if result_metric:
            case_metrics.setdefault(family, set()).add(result_metric)
        description = f"VLMProbe · {result.method}"
        if result_metric:
            description += f" · {result_metric}"
        if family == "sae" and result.metadata.get("architecture"):
            description += f" · {result.metadata['architecture']}"
        case["methods"].append({"id": method_id, "label": method_label,
                                "description": description, "views": views})
    for family, case in cases.items():
        case["metric"] = metric or " / ".join(sorted(case_metrics.get(family, ())))
    return {"schema": 1, "cases": list(cases.values())}


def _write_bundle(destination, bundle):
    payload = json.dumps(bundle, ensure_ascii=False, allow_nan=False) + "\n"
    temp_path = None
    try:
        with NamedTemporaryFile(dir=destination.parent, mode="w", encoding="utf-8", delete=False) as stream:
            temp_path = Path(stream.name)
            stream.write(payload)
        temp_path.replace(destination)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description="Export trusted ProbeResult directories for Probe Explorer.")
    parser.add_argument("directories", metavar="RESULT", type=Path, nargs="+")
    parser.add_argument("--output", "-o", required=True, type=Path, help="destination .json file")
    parser.add_argument("--title", default="Imported results")
    parser.add_argument("--model", help="optional model display label")
    parser.add_argument("--prompt", default="")
    parser.add_argument("--target", default="", help="human-readable target label")
    parser.add_argument("--metric", help="optional metric display label")
    parser.add_argument("--batch", type=int, default=0, help="zero-based batch item")
    parser.add_argument("--query", type=int, help="expanded query position (default: final selected query)")
    parser.add_argument("--head", type=int, help="head index for raw attention, ATTATTR or Attention Lens")
    parser.add_argument("--target-token-id", type=int, help="vocabulary/codebook ID for lens ranks")
    parser.add_argument("--image", type=Path, help="processed PNG, JPEG or WebP to embed")
    parser.add_argument("--grid", type=int, nargs=2, metavar=("HEIGHT", "WIDTH"), help="native patch grid")
    parser.add_argument("--visual-positions", type=int, nargs="+", help="expanded positions in native grid order")
    parser.add_argument("--features", type=int, nargs="+", help="SAE dictionary IDs to display")
    parser.add_argument("--top-k", type=int, default=5, help="entries per ranking (1–20)")
    args = parser.parse_args()
    try:
        if any(args.output.resolve() == (directory / filename).resolve()
               for directory in args.directories for filename in ("metadata.json", "tensors.pt")):
            raise ValueError("output must differ from the original saved result files")
        bundle = build_bundle([load_saved_result(directory) for directory in args.directories],
                              model=args.model, title=args.title, prompt=args.prompt, target=args.target,
                              metric=args.metric, image_path=args.image, grid_shape=args.grid,
                              visual_positions=args.visual_positions, batch_index=args.batch, query=args.query,
                              head=args.head, target_token_id=args.target_token_id, feature_ids=args.features,
                              top_k=args.top_k)
        _write_bundle(args.output, bundle)
    except (OSError, ImportError, ValueError, RuntimeError) as error:
        parser.exit(2, f"Export failed. {error}\n")
    print(f"Wrote {len(bundle['cases'])} case(s) to {args.output}")


if __name__ == "__main__":
    main()

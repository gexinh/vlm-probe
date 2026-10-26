"""Native-coordinate MJ text readouts from pretrained-model experiments.

These are output-token readouts, not attention or gradient attribution maps.
For ordinary image-before-question chat order, visual source states cannot
depend on the later question. A chosen answer-token display does not change
that causal ordering. Supply the actual processed image/crop and its declared
visual grid; this module does not infer or undo processor geometry.
"""
import torch

from ..core import ProbeResult


def _result(result, source):
    if (not isinstance(result, ProbeResult) or result.method != "mj_lens"
            or result.metadata.get("source_modality") != source
            or result.metadata.get("target_modality") != "text"):
        raise ValueError(f"expected a {source}-to-text MJ-Lens result")
    logits = result.tensors.get("logits")
    if (not isinstance(logits, torch.Tensor) or not logits.is_floating_point()
            or logits.ndim != 3 or min(logits.shape) < 1):
        raise ValueError("MJ logits must be a nonempty floating [layer,position,vocabulary] tensor")
    layers = result.metadata.get("layers")
    if (not isinstance(layers, (list, tuple)) or len(layers) != logits.shape[0]
            or any(type(i) is not int or i < 0 for i in layers)
            or len(set(layers)) != len(layers)):
        raise ValueError("MJ layers must declare one unique block-input index per captured layer")
    return logits, list(layers)


def _token_id(token_id, width, name):
    if type(token_id) is not int or not 0 <= token_id < width:
        raise ValueError(f"{name} must be an integer inside the output vocabulary")


def _positions(result, count):
    positions = result.tensors.get("positions")
    if (not isinstance(positions, torch.Tensor) or positions.dtype != torch.long
            or tuple(positions.shape) != (count, 2) or (positions < 0).any()):
        raise ValueError("MJ positions must be nonnegative int64 [packed_position,2] coordinates")
    if torch.unique(positions, dim=0).shape[0] != count:
        raise ValueError("MJ source coordinates cannot contain duplicate positions")
    return positions


def region_token_readout(
    result, tokenizer, *, grid_shape, image_size, bbox, top_k=5,
    batch_index=0, reference_token_id=None,
):
    """Decode every visual patch overlapping an explicitly supplied image bbox.

    ``result`` is a packed vocabulary-lens result with ``logits[L,N,V]``, native
    ``positions[N,2]`` and declared ``metadata['layers']``. MJ results must be
    visual-to-text; ordinary Logit, Tuned or Jacobian results must have been run
    with visual positions selected by the caller. ``image_size=(width,height)``
    and ``bbox=(x,y,width,height)`` are in the actual processed image geometry;
    ``grid_shape=(height,width)`` declares its row-major visual grid. A packed
    or tiled sequence is not automatically interpreted as an original image.

    Every positive-area bbox/patch intersection is retained. Per-patch Top-k is
    selected from the entire vocabulary. The region distribution is the
    intersection-area-weighted mean of per-patch full-vocabulary softmaxes,
    rather than a decoded mean hidden vector or a softmax restricted to Top-k.
    ``reference_token_id`` only adds post-hoc rank/probability measurements for
    that exact vocabulary ID; case/leading-space variants and translated or
    synonymous words are not pooled into a concept score. It never selects
    patches, vocabulary candidates or layers. Raw token strings,
    reprs, IDs and probabilities are returned in a JSON-safe dictionary. No
    special tokens, whitespace, punctuation or unrelated words are filtered.
    """
    import math

    if not isinstance(result, ProbeResult):
        raise TypeError("result must be a packed vocabulary-lens ProbeResult")
    if result.method == "mj_lens":
        logits, layers = _result(result, "visual")
    elif result.method in {"logit_lens", "tuned_lens", "jacobian_lens"}:
        logits = result.tensors.get("logits")
        layers = result.metadata.get("layers")
        if (not isinstance(logits, torch.Tensor) or not logits.is_floating_point()
                or logits.ndim != 3 or min(logits.shape) < 1):
            raise ValueError("vocabulary logits must be nonempty floating [layer,position,vocabulary]")
        if (not isinstance(layers, (list, tuple)) or len(layers) != logits.shape[0]
                or any(type(i) is not int or i < 0 for i in layers)
                or len(set(layers)) != len(layers)):
            raise ValueError("result layers must declare one unique site index per captured layer")
        layers = list(layers)
    else:
        raise ValueError("region decoding requires MJ, Logit, Tuned or Jacobian vocabulary readouts")
    if type(top_k) is not int or not 1 <= top_k <= logits.shape[-1]:
        raise ValueError("top_k must be an integer within the full vocabulary width")
    if not callable(getattr(tokenizer, "decode", None)):
        raise TypeError("tokenizer must provide its native decode method")
    if type(batch_index) is not int or batch_index < 0:
        raise ValueError("batch_index must be a nonnegative integer")
    for name, pair in (("grid_shape", grid_shape), ("image_size", image_size)):
        if (not isinstance(pair, (tuple, list)) or len(pair) != 2
                or any(type(value) is not int or value <= 0 for value in pair)):
            raise ValueError(f"{name} must contain two positive integers")
    try:
        box = tuple(float(value) for value in bbox)
    except (TypeError, ValueError):
        raise ValueError("bbox must contain finite (x,y,width,height) values") from None
    if len(box) != 4 or not all(math.isfinite(value) for value in box) or min(box[2:]) <= 0:
        raise ValueError("bbox must contain finite (x,y,width,height) with positive size")
    if reference_token_id is not None:
        _token_id(reference_token_id, logits.shape[-1], "reference_token_id")
    positions = _positions(result, logits.shape[1])
    ordered = (positions[:, 0] == batch_index).nonzero(as_tuple=True)[0]
    height, width = grid_shape
    if len(ordered) != height * width:
        raise ValueError("selected source count does not match the declared visual grid_shape")
    ordered = ordered[positions[ordered, 1].argsort()]
    image_width, image_height = image_size
    patch_width, patch_height = image_width / width, image_height / height
    selected_patches = []
    for patch_index in range(height * width):
        row, column = divmod(patch_index, width)
        x0, y0 = column * patch_width, row * patch_height
        x1, y1 = (column + 1) * patch_width, (row + 1) * patch_height
        area = (max(0., min(x1, box[0] + box[2]) - max(x0, box[0]))
                * max(0., min(y1, box[1] + box[3]) - max(y0, box[1])))
        if area > 0:
            packed_index = int(ordered[patch_index])
            selected_patches.append({"patch_index": patch_index, "row": row, "column": column,
                "sequence_position": int(positions[packed_index, 1]), "packed_index": packed_index,
                "patch_bbox_xywh": [x0, y0, patch_width, patch_height], "intersection_area": area})
    if not selected_patches:
        raise ValueError("bbox does not overlap any patch in the supplied processed image")
    total_area = sum(patch["intersection_area"] for patch in selected_patches)
    for patch in selected_patches:
        patch["region_weight"] = patch["intersection_area"] / total_area
    weights = torch.tensor([patch["region_weight"] for patch in selected_patches], dtype=torch.float32)
    indices = torch.tensor([patch["packed_index"] for patch in selected_patches], device=logits.device)
    decoded = {}

    def token_record(token_id, probability, *, logit=None):
        token_id = int(token_id)
        if token_id not in decoded:
            # Native decoding, retaining special IDs and whitespace. A padded
            # vocabulary row can legitimately decode to an empty string.
            decoded[token_id] = tokenizer.decode([token_id], skip_special_tokens=False,
                                                  clean_up_tokenization_spaces=False)
        token = decoded[token_id]
        if not isinstance(token, str):
            raise TypeError("native tokenizer.decode must return a string")
        value = {"token_id": token_id, "token": token, "token_repr": repr(token),
                 "probability": float(probability)}
        if logit is not None:
            value["logit"] = float(logit)
        return value

    reports = []
    for index, layer in enumerate(layers):
        selected_logits = logits[index].index_select(0, indices).detach().float().cpu()
        if not torch.isfinite(selected_logits).all():
            raise ValueError("selected region vocabulary logits must be finite")
        probabilities = selected_logits.softmax(-1)
        mixture = (probabilities * weights[:, None]).sum(0)
        region_ids = mixture.topk(top_k).indices
        layer_report = {"layer": layer,
            "region_top_tokens": [token_record(i, mixture[i]) for i in region_ids], "patches": []}
        if reference_token_id is not None:
            reference = token_record(reference_token_id, mixture[reference_token_id])
            reference["rank"] = int((mixture > mixture[reference_token_id]).sum()) + 1
            layer_report["region_reference"] = reference
        for patch, raw, probability in zip(selected_patches, selected_logits, probabilities):
            ids = raw.topk(top_k).indices
            patch_report = {"patch_index": patch["patch_index"],
                "top_tokens": [token_record(i, probability[i], logit=raw[i]) for i in ids]}
            if reference_token_id is not None:
                reference = token_record(reference_token_id, probability[reference_token_id],
                                         logit=raw[reference_token_id])
                reference["rank"] = int((raw > raw[reference_token_id]).sum()) + 1
                patch_report["reference"] = reference
            layer_report["patches"].append(patch_report)
        reports.append(layer_report)
    return {"method": result.method, "layers": layers,
        "tap_convention": result.metadata.get("tap_convention", "native_residual_site"),
        "grid_shape": list(grid_shape), "image_size": list(image_size), "bbox_xywh": list(box),
        "batch_index": batch_index, "vocabulary_size": logits.shape[-1], "top_k": top_k,
        "selection": "all positive-area intersections of bbox with native visual grid",
        "aggregation": "intersection-area-weighted mean of per-patch full-vocabulary softmax",
        "special_tokens_filtered": False, "reference_used_for_selection": False,
        "patches": selected_patches, "layer_readouts": reports}


def _region_font():
    from matplotlib import font_manager
    names = {font.name for font in font_manager.fontManager.ttflist}
    return ["DejaVu Sans", *(["Noto Sans CJK SC"] if "Noto Sans CJK SC" in names else [])]


def plot_region_tokens(image, report, *, top_k=3, layer_labels=None, title=None, axes=None):
    """Plot the processed image/grid and a full-vocabulary bbox readout per layer.

    ``report`` is the dictionary returned by :func:`region_token_readout`.
    All selected patch indices are drawn in native row-major coordinates. The
    table uses raw token reprs and mixture probabilities; reference ranks are
    post-hoc annotations. ``layer_labels`` may explain matched sites across
    methods without changing the stored native layer IDs. Optional ``axes``
    contains one image axis followed by one axis per layer. Returns figure and
    that one-dimensional axes array. The figure does not infer new semantics.
    """
    import textwrap
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from .attention import _image

    if type(top_k) is not int or not 1 <= top_k <= report["top_k"]:
        raise ValueError("plot top_k must be within the extracted report top_k")
    count = len(report["layer_readouts"])
    if layer_labels is not None and len(layer_labels) != count:
        raise ValueError("layer_labels must match extracted layer count")
    rgb = _image(image)
    if [rgb.shape[1], rgb.shape[0]] != report["image_size"]:
        raise ValueError("display image dimensions must match the declared processed image_size")
    created = axes is None
    if created:
        figure, axes = plt.subplots(1, count + 1, figsize=(3.3 + count * 2.6, 4.3),
                                    gridspec_kw={"width_ratios": [1.3] + [1.] * count})
    axes = np.asarray(axes, dtype=object)
    if axes.ndim != 1 or len(axes) != count + 1:
        raise ValueError("axes must contain one image axis and one axis per layer")
    figure = axes[0].figure
    if any(axis.figure is not figure for axis in axes):
        raise ValueError("region plot axes must share one figure")
    image_axis = axes[0]
    image_axis.imshow(rgb)
    grid_height, grid_width = report["grid_shape"]
    image_width, image_height = report["image_size"]
    for x in np.linspace(0, image_width, grid_width + 1):
        image_axis.axvline(x - .5, color="#58788f", alpha=.23, linewidth=.4)
    for y in np.linspace(0, image_height, grid_height + 1):
        image_axis.axhline(y - .5, color="#58788f", alpha=.23, linewidth=.4)
    for patch in report["patches"]:
        x, y, width, height = patch["patch_bbox_xywh"]
        image_axis.add_patch(Rectangle((x, y), width, height, fill=False,
                                      edgecolor="#4dd9d3", linewidth=.8))
        image_axis.text(x + width / 2, y + height / 2, f"p{patch['patch_index']}",
            ha="center", va="center", fontsize=5.5, color="white",
            bbox={"facecolor": "#10213b", "edgecolor": "none", "alpha": .75, "pad": .35})
    x, y, width, height = report["bbox_xywh"]
    image_axis.add_patch(Rectangle((x, y), width, height, fill=False,
                                  edgecolor="#ef8a21", linewidth=1.8))
    image_axis.axis("off")
    image_axis.set_title(title or "Processed image · bbox + patch IDs", fontsize=10)
    fonts = _region_font()
    for index, (axis, layer) in enumerate(zip(axes[1:], report["layer_readouts"])):
        axis.set(xlim=(0, 1), ylim=(0, 1))
        axis.axis("off")
        axis.add_patch(Rectangle((0, .015), 1, .95, transform=axis.transAxes,
            facecolor="#f7fafc", edgecolor="#dbe5ed", linewidth=.8))
        label = layer_labels[index] if layer_labels is not None else f"Layer {layer['layer']}"
        axis.text(.05, .91, str(label), fontsize=10.5, weight="bold", color="#10213b")
        displayed = [(token, textwrap.wrap(token["token_repr"], width=25,
                       replace_whitespace=False, drop_whitespace=False) or ["''"])
                     for token in layer["region_top_tokens"][:top_k]]
        step = min(.055, .66 / sum(len(lines) + 1.4 for _, lines in displayed))
        word_size, score_size = min(9., step / .055 * 9.), min(7.6, step / .055 * 7.6)
        cursor = .79
        for rank, (token, token_lines) in enumerate(displayed, 1):
            axis.text(.05, cursor, f"{rank}. " + "\n".join(token_lines), fontsize=word_size,
                      fontfamily=fonts, va="top", color="#123950")
            cursor -= step * len(token_lines)
            axis.text(.08, cursor, f"id {token['token_id']} · p={token['probability']:.3g}",
                      fontsize=score_size, va="top", color="#63758a")
            cursor -= step * 1.4
        reference = layer.get("region_reference")
        if reference is not None:
            text = (f"Exact QA token {reference['token_repr']}:\n"
                    f"id {reference['token_id']} · rank {reference['rank']} · p={reference['probability']:.3g}")
            axis.text(.05, .05, text, fontsize=7.5, va="bottom", color="#6f5b3c", fontfamily=fonts)
    if created:
        figure.subplots_adjust(left=.015, right=.99, top=.92, bottom=.16, wspace=.12)
        figure.text(.5, .035, "BBox readout: area-weighted mean of per-patch full-vocabulary probabilities. "
                    "Raw token reprs; no vocabulary filtering.", ha="center", fontsize=8, color="#5a6e83")
    return figure, axes


def plot_region_patch_tokens(report, *, top_k=3, layer_labels=None, title=None):
    """Show every overlapping patch's raw Top-k words across captured layers.

    Each cell retains the original token repr, vocabulary ID and probability.
    No patch or decoded token is chosen for visual appeal. Top-k logits and
    post-hoc ranks remain in the extracted report/CSV. Returns figure and axis.
    """
    import textwrap
    import matplotlib.pyplot as plt

    if type(top_k) is not int or not 1 <= top_k <= report["top_k"]:
        raise ValueError("plot top_k must be within the extracted report top_k")
    layers = report["layer_readouts"]
    if layer_labels is not None and len(layer_labels) != len(layers):
        raise ValueError("layer_labels must match extracted layer count")
    rows = []
    for patch_index in range(len(report["patches"])):
        cells = []
        for layer in layers:
            tokens = layer["patches"][patch_index]["top_tokens"][:top_k]
            cells.append("\n".join(
                textwrap.fill(token["token_repr"], width=26, replace_whitespace=False,
                              drop_whitespace=False)
                + f"\n#{token['token_id']} · p={token['probability']:.2g}" for token in tokens))
        rows.append(cells)
    max_lines = max(cell.count("\n") + 1 for row in rows for cell in row)
    figure, axis = plt.subplots(figsize=(max(9., 3.1 * len(layers)), max(2.5, .145 * max_lines * len(rows) + 1.1)))
    axis.axis("off")
    labels = [f"p{patch['patch_index']}  (r{patch['row']}, c{patch['column']})" for patch in report["patches"]]
    columns = layer_labels or [f"Layer {layer['layer']}" for layer in layers]
    table = axis.table(cellText=rows, rowLabels=labels, colLabels=columns,
                       cellLoc="left", loc="center", bbox=(.09, .02, .9, .91))
    table.auto_set_font_size(False)
    table.set_fontsize(7.5)
    for (row, column), cell in table.get_celld().items():
        cell.set_edgecolor("#d9e2eb")
        cell.set_linewidth(.5)
        cell.get_text().set_fontfamily(_region_font())
        if row == 0:
            cell.set_facecolor("#e9f2f8")
            cell.get_text().set_weight("bold")
        elif row % 2 == 0:
            cell.set_facecolor("#f8fafc")
    axis.set_title(title or "Every bbox-overlapping visual patch · raw full-vocabulary Top-k", fontsize=11, pad=14)
    figure.subplots_adjust(left=.015, right=.99, top=.95, bottom=.02)
    return figure, axis


def plot_mj_visual_readout(
    image, result, *, positive_token_id, negative_token_id=None, grid_shape,
    layer, bbox=None, token_label=None, negative_label=None, image_alpha=.65,
    batch_index=0, alpha=.85, colorbar=True, cmap=None, vmin=None, vmax=None,
    ax=None,
):
    """Overlay a chosen token logit or logit margin at visual source positions.

    ``layer`` is an actual MJ block-input index from ``result.metadata['layers']``,
    not a position in the captured layer list. Positions for ``batch_index``
    are sorted by their native expanded-sequence coordinates and must match the
    explicit ``(height,width)`` grid exactly. Multi-image/tiled inputs must first
    be split by the caller. ``bbox`` is optional ``(x,y,width,height)`` in the
    supplied processed image's coordinates; no bbox transform is inferred.

    The measured logit/margin is preserved, including its sign. Color limits
    affect display only and are computed separately for each map by default;
    color intensity is not a shared measure of semantic strength across layers
    or models. Positive values use cyan/blue, negative values pale gray, with an orange
    annotation box. Labels are caller-provided token descriptions; a subword
    score is not automatically a whole-word concept score. Returns Axes.
    """
    from .attention import plot_attention_overlay
    from matplotlib.patches import Rectangle
    from matplotlib.colors import LinearSegmentedColormap
    import numpy as np

    logits, layers = _result(result, "visual")
    _token_id(positive_token_id, logits.shape[-1], "positive_token_id")
    if negative_token_id is not None:
        _token_id(negative_token_id, logits.shape[-1], "negative_token_id")
    if type(layer) is not int or layer not in layers:
        raise ValueError(f"layer must be an MJ block-input index from {layers}")
    if type(batch_index) is not int or batch_index < 0:
        raise ValueError("batch_index must be a nonnegative integer")
    if (not isinstance(grid_shape, (tuple, list)) or len(grid_shape) != 2
            or any(type(n) is not int or n <= 0 for n in grid_shape)):
        raise ValueError("grid_shape must contain two positive integers")
    positions = _positions(result, logits.shape[1])
    indices = (positions[:, 0] == batch_index).nonzero(as_tuple=True)[0]
    if len(indices) != grid_shape[0] * grid_shape[1]:
        raise ValueError("selected visual source count does not match the declared grid_shape")
    # Native source order, rather than the order of an externally packed result.
    indices = indices[positions[indices, 1].argsort()]
    row = logits[layers.index(layer)]
    indices = indices.to(logits.device)
    values = row[:, positive_token_id].index_select(0, indices)
    if negative_token_id is not None:
        values = values - row[:, negative_token_id].index_select(0, indices)
    if not torch.isfinite(values).all():
        raise ValueError("selected MJ readout values must be finite")
    # Only the selected scalar map crosses to CPU, not the full vocabulary tensor.
    values = values.detach().float().cpu().reshape(*grid_shape).numpy().copy()
    signed = negative_token_id is not None or bool((values < 0).any())
    positive = str(positive_token_id) if token_label is None else str(token_label)
    label = f"MJ token logit: {positive}"
    if negative_token_id is not None:
        negative = str(negative_token_id) if negative_label is None else str(negative_label)
        label = f"MJ token-logit margin: {positive} − {negative}"
    if cmap is None and signed:
        cmap = LinearSegmentedColormap.from_list("mj_readout_contrast", [
            (0., "#aebbc6"), (.5, "#ffffff"), (.72, "#9bdce3"), (1., "#1764a0"),
        ])
    ax = plot_attention_overlay(
        image, values, grid_shape=tuple(grid_shape), signed=signed,
        alpha=alpha, image_alpha=image_alpha,
        cmap=cmap or "Blues", vmin=vmin, vmax=vmax,
        ax=ax, colorbar=colorbar, value_label=label,
        title=f"MJ V→T · block input {layer}",
    )
    if bbox is not None:
        try:
            box = np.asarray(bbox, dtype=float)
        except (TypeError, ValueError):
            raise ValueError("bbox must be finite (x,y,width,height) in the supplied image coordinates") from None
        if box.shape != (4,) or not np.isfinite(box).all() or min(box[2:]) <= 0:
            raise ValueError("bbox must be finite (x,y,width,height) in the supplied image coordinates")
        ax.add_patch(Rectangle(box[:2], box[2], box[3], fill=False,
                               edgecolor="#ef8a21", linewidth=1.6, zorder=3))
    return ax


def _single_text_logits(value, layers, width, name):
    if not isinstance(value, torch.Tensor) or not value.is_floating_point():
        raise ValueError(f"{name} must be a floating tensor")
    if value.ndim == 3 and value.shape[1] == 1:
        value = value[:, 0]
    if tuple(value.shape) != (layers, width):
        raise ValueError(f"{name} must be [layers,vocabulary] or [layers,1,vocabulary]")
    return value


def plot_mj_text_comparison(
    mj_result, plain_logits, native_logits, *, positive_token_id,
    negative_token_id, labels=None, ax=None,
):
    """Compare token-logit margins at one text state across MJ block inputs.

    ``mj_result`` must contain exactly one selected text position per layer.
    ``plain_logits`` is the native final norm/head applied directly to those
    same block-input states, before MJ transport; it must use the same layer
    order and vocabulary. ``native_logits`` is the final model vector for that
    same last-prompt/prediction position, not an averaged sequence or another
    example. Supply a vector, or singleton-leading axes. Values remain raw
    signed margins; no per-line normalization is applied. Returns Axes.
    """
    from ._common import axes

    logits, layers = _result(mj_result, "text")
    if logits.shape[1] != 1:
        raise ValueError("text comparison requires exactly one selected text prediction position")
    _positions(mj_result, 1)
    width = logits.shape[-1]
    _token_id(positive_token_id, width, "positive_token_id")
    _token_id(negative_token_id, width, "negative_token_id")
    plain = _single_text_logits(plain_logits, len(layers), width, "plain_logits")
    if (not isinstance(native_logits, torch.Tensor) or not native_logits.is_floating_point() or native_logits.ndim < 1
            or native_logits.shape[-1] != width or native_logits.numel() != width):
        raise ValueError("native_logits must contain exactly one final vocabulary vector")
    native = native_logits.reshape(width)
    mj_margin = logits[:, 0, positive_token_id] - logits[:, 0, negative_token_id]
    plain_margin = plain[:, positive_token_id] - plain[:, negative_token_id]
    native_margin = native[positive_token_id] - native[negative_token_id]
    if not all(torch.isfinite(value).all() for value in (mj_margin, plain_margin, native_margin)):
        raise ValueError("selected MJ, plain and native margins must be finite")
    if labels is not None and len(labels) != len(layers):
        raise ValueError("labels must match the number of MJ block-input layers")
    ax = axes(ax, figsize=(7, 3.5))
    ax.plot(layers, mj_margin.detach().float().cpu().tolist(), marker="o",
            color="#2166ac", label="MJ-Lens T→T")
    ax.plot(layers, plain_margin.detach().float().cpu().tolist(), marker="s",
            linestyle="--", color="#7c8796", label="Plain readout · same inputs")
    ax.axhline(float(native_margin.detach().cpu()), color="#d97722", linestyle=":",
               linewidth=1.5, label="Native final output")
    ax.axhline(0, color="#d5dae2", linewidth=.8, zorder=0)
    ax.set_xticks(layers, layers if labels is None else labels)
    ax.set(xlabel="Decoder block input", ylabel="Answer-token logit margin",
           title=f"Text-state readout: token {positive_token_id} − token {negative_token_id}")
    ax.legend(frameon=False, fontsize=8)
    return ax

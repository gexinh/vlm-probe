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

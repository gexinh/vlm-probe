"""Visual codebook readouts and controlled modality-index displays."""


def plot_mj_codebook(result, *, grid, decoder=None, batch_index=0):
    """Display layerwise code grids or their native VQ decodes.

    Requires an explicit source grid and visual-to-visual MJ result. Decodes
    show per-position readout, which may echo teacher-forced codes; they are
    neither fresh autoregressive generations nor grounding maps.
    Returns (Figure, axes).
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from ..api.mj import decode_mj_codes
    if type(batch_index) is not int or batch_index < 0:
        raise ValueError("batch_index must be nonnegative")
    if decoder is None:
        def display_grid(codes):
            return codes.reshape(len(codes), *grid)
        images, codes = decode_mj_codes(result, grid=grid, decoder=display_grid)
    else:
        images, codes = decode_mj_codes(result, grid=grid, decoder=decoder)
    if batch_index >= images.shape[1]:
        raise ValueError("batch_index is absent from MJ result")
    layers = result.metadata["layers"]
    figure, axes = plt.subplots(1, len(layers), figsize=(2.3 * len(layers), 2.7), squeeze=False)
    for index, (layer, ax) in enumerate(zip(layers, axes[0])):
        value = images[index, batch_index].detach().cpu().numpy()
        if decoder is None:
            ax.imshow(value, interpolation="nearest", cmap="tab20")
            for (y, x), code in np.ndenumerate(codes[index, batch_index].cpu().numpy()):
                ax.text(x, y, str(int(code)), ha="center", va="center", fontsize=8)
        else:
            if value.ndim == 3 and value.shape[-1] not in (1, 3, 4):
                if value.shape[0] not in (1, 3, 4):
                    raise ValueError("native decoded images need declared HWC or CHW color channels")
                value = value.transpose(1, 2, 0)
            if value.dtype.kind == "f" and (value.min() < 0 or value.max() > 1):
                raise ValueError("native decoder must supply display-ready [0,1] floats or uint8 images")
            ax.imshow(value)
        ax.set_title(f"Block input {layer}", fontsize=10)
        ax.axis("off")
    figure.suptitle("Visual codebook readout" if decoder is None else "Native VQ decode of MJ readout", fontsize=11)
    figure.tight_layout()
    return figure, axes[0]


def plot_mj_modality_index(result, *, labels=None, ax=None):
    """Compare raw, rotated-control and excess dictionary mass at each position."""
    import numpy as np
    from ._common import axes
    if result.method != "mj_modality_index":
        raise ValueError("expected a controlled MJ modality-index result")
    names = ("raw_mi", "rotated_mi", "excess_mi")
    rows = [result.tensors[name].detach().float().cpu().flatten().numpy() for name in names]
    if not rows[0].size:
        raise ValueError("no positions to display")
    if labels is not None and len(labels) != len(rows[0]):
        raise ValueError("labels must match the selected activation count")
    ax = axes(ax, figsize=(max(5, len(rows[0]) * .6), 3))
    x = np.arange(len(rows[0]))
    for index, (name, row, color) in enumerate(zip(names, rows, ("#2a9d8f", "#c6cbd5", "#335caa"))):
        ax.bar(x + (index - 1) * .24, row, width=.24, color=color,
               label={"raw_mi": "Raw", "rotated_mi": "Rotated control", "excess_mi": "Excess"}[name])
    ax.axhline(0, color="#637083", linewidth=.7)
    ax.set_xticks(x, labels if labels is not None else [str(i) for i in x])
    ax.set(xlabel="Selected source position", ylabel="Dictionary mass fraction",
           title="MJ modality index with rotated control")
    ax.legend(frameon=False, fontsize=8)
    return ax

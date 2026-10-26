"""Compact plots for native SAE EvaluationReport objects."""


def plot_sae_evaluation(report, *, top_k=20, ax=None):
    """Plot measured firing frequency; inactive features remain zero."""
    import torch
    from ._common import axes
    frequency = torch.as_tensor(report.feature_statistics.get("frequency", []))
    if frequency.ndim != 1 or not len(frequency):
        raise ValueError("report must contain per-feature firing frequency")
    if type(top_k) is not int or top_k < 1:
        raise ValueError("top_k must be a positive integer")
    top_k = min(top_k, len(frequency))
    scores, indices = frequency.topk(top_k)
    ax = axes(ax, figsize=(8, 3))
    ax.bar(range(top_k), scores.cpu().numpy(), color="#7350cf")
    ax.set_xticks(range(top_k), [str(i) for i in indices.tolist()], rotation=45)
    ax.set(xlabel="SAE feature ID", ylabel="Fraction of positions active", ylim=(0, 1))
    return ax


def plot_sae_comparison(reports, *, x="l0", y="fvu", ax=None):
    """Plot named reports, grouping checkpoint curves by provenance method.

    ``reports`` maps checkpoint labels to EvaluationReport. A report may declare
    ``provenance['method']`` to join its sparsity variants into one curve.
    Missing/undefined values are skipped rather than plotted as zero.
    """
    import math
    from collections import OrderedDict
    from ._common import axes
    ax = axes(ax, figsize=(7, 4))
    groups = OrderedDict()
    for name, report in reports.items():
        x_value, y_value = report.summary.get(x), report.summary.get(y)
        if x_value is None or y_value is None:
            continue
        if not math.isfinite(x_value) or not math.isfinite(y_value):
            continue
        group = str(report.provenance.get("method", name))
        groups.setdefault(group, []).append((x_value, y_value))
    if not groups:
        raise ValueError("no finite comparison points for the requested metrics")
    for name, points in groups.items():
        points.sort()
        ax.plot([p[0] for p in points], [p[1] for p in points], marker="o", label=name)
    labels = {"l0": "Mean active features per position", "fvu": "Fraction of variance unexplained",
              "mse": "Reconstruction MSE", "delta_nll": "Change in next-token NLL"}
    ax.set(xlabel=labels.get(x, x), ylabel=labels.get(y, y))
    ax.legend(frameon=False)
    ax.grid(alpha=.2)
    return ax

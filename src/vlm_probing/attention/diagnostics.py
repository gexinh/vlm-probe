"""Explicit mass-distribution checks on caller-selected visual scores."""

from collections.abc import Sequence
from numbers import Integral

import torch
from torch import Tensor

from ..core import ProbeResult


def _positive_sizes(value, name, *, length=None):
    if (not isinstance(value, Sequence) or isinstance(value, (str, bytes))
            or not value or (length is not None and len(value) != length)
            or any(isinstance(n, bool) or not isinstance(n, Integral) or n <= 0 for n in value)):
        suffix = f" {length}" if length is not None else ""
        raise ValueError(f"{name} must contain{suffix} positive integers")
    return [int(n) for n in value]


def _ratio(numerator, denominator):
    # Retain complete one-sided concentration as +inf. Two inactive bands are
    # undefined rather than equal; replacing 0/0 with 1 would hide zero maps.
    return torch.where(denominator > 0, numerator / denominator,
                       torch.where(numerator > 0, torch.full_like(numerator, float("inf")),
                                   torch.full_like(numerator, float("nan"))))


def _mass_ratio(numerator_scores, denominator_scores, *, mean=False):
    if not numerator_scores.shape[-1] or not denominator_scores.shape[-1]:
        return torch.full_like(numerator_scores.sum(-1), float("nan"))
    # Use ONE scale for both bands, so it cancels from the ratio. Forming raw
    # sums first can overflow even for finite scores; tiny scores can underflow
    # during aggregation or a mean. Neither the input nor returned raw masses
    # is normalized by this numerical device.
    scale = torch.maximum(numerator_scores.amax(-1), denominator_scores.amax(-1))
    scale = scale.masked_fill(scale == 0, 1)[..., None]
    numerator = (numerator_scores / scale).sum(-1)
    denominator = (denominator_scores / scale).sum(-1)
    if mean:
        numerator = numerator / numerator_scores.shape[-1]
        denominator = denominator / denominator_scores.shape[-1]
    return _ratio(numerator, denominator)


def attention_position_diagnostics(
    visual_scores: Tensor, *, grid_shape=None, tile_sizes=None,
) -> ProbeResult:
    """Measure sequence/spatial/tile mass ratios without correcting any scores.

    ``visual_scores`` is a finite, nonnegative floating tensor with native
    visual-token order on its last axis. Select the desired layer, query rows,
    batch item and visual keys *before* calling this function. Leading axes
    are preserved; no query selection, head reduction, normalization, floor
    subtraction, positive-part transform or geometry inference is performed.

    Every result includes first/last visual-sequence thirds of equal token
    count. Optional ``grid_shape=(height,width)`` declares one row-major image
    grid and adds top/bottom thirds of equal row count. Optional ``tile_sizes``
    declares consecutive native tile groups and adds their masses and the
    first/last ratio. Unequal tile sizes can by themselves change mass ratios;
    the mean-per-token tile ratio is returned separately. Packed/tiled model
    outputs must not be relabelled as one image grid without a valid mapping.

    A ratio is +inf for positive numerator/zero denominator and NaN when both
    are zero (also for thirds containing no tokens/rows). Ratios are descriptive
    statistics, not proof of positional bias, content independence or causal
    localization. Compare matched inputs and mask/weight controls to study
    those questions. Raw masses accumulate in FP64; ratios use a common band
    scale to avoid overflow/underflow and never mutate inputs. Raw FP64 masses
    can still be +inf when their exact sum exceeds FP64's representable range;
    a finite ratio can remain valid in that case.
    """
    if not isinstance(visual_scores, Tensor) or not visual_scores.is_floating_point():
        raise TypeError("visual_scores must be a floating-point tensor")
    if visual_scores.ndim < 1 or any(n == 0 for n in visual_scores.shape):
        raise ValueError("visual_scores must have nonempty dimensions and visual tokens on the last axis")
    if not torch.isfinite(visual_scores).all() or (visual_scores < 0).any():
        raise ValueError("visual_scores must be finite and nonnegative; select any signed-evidence transform explicitly")
    scores = visual_scores.detach().to(torch.float64)
    count = scores.shape[-1]
    third = count // 3
    first_scores = scores[..., :third]
    last_scores = scores[..., count - third:] if third else scores[..., :0]
    first, last = first_scores.sum(-1), last_scores.sum(-1)
    tensors = {
        "total_mass": scores.sum(-1),
        "sequence_first_third_mass": first,
        "sequence_last_third_mass": last,
        "sequence_first_last_third_ratio": _mass_ratio(first_scores, last_scores),
    }
    metadata = {
        "input": "caller_selected_nonnegative_visual_scores_in_native_order",
        "visual_token_count": count,
        "sequence_third_token_count": third,
        "input_dtype": str(visual_scores.dtype),
        "mass_accumulation_dtype": "torch.float64",
        "ratio_arithmetic": "shared band scale, preserving the original mass ratio",
        "score_transformation": "none",
        "ratio_zero_denominator": "+inf for positive/zero; NaN for zero/zero",
        "interpretation": (
            "Descriptive mass distribution only. A ratio different from one does not by itself "
            "identify a causal-mask effect or establish content independence or localization. "
            "Use matched content and mask/weight controls."
        ),
    }
    if grid_shape is not None:
        height, width = _positive_sizes(grid_shape, "grid_shape", length=2)
        if height * width != count:
            raise ValueError("grid_shape must match the explicitly selected visual-token count")
        row_third = height // 3
        grid = scores.reshape(*scores.shape[:-1], height, width)
        top_scores = grid[..., :row_third, :].reshape(*scores.shape[:-1], row_third * width)
        bottom_scores = (grid[..., height - row_third:, :].reshape(*scores.shape[:-1], row_third * width)
                         if row_third else top_scores)
        top, bottom = top_scores.sum(-1), bottom_scores.sum(-1)
        tensors.update(top_mass=top, bottom_mass=bottom,
                       top_bottom_ratio=_mass_ratio(top_scores, bottom_scores))
        metadata.update(grid_shape=[height, width], spatial_third_row_count=row_third,
                        spatial_order="caller_declared_row_major")
    if tile_sizes is not None:
        sizes = _positive_sizes(tile_sizes, "tile_sizes")
        if len(sizes) < 2 or sum(sizes) != count:
            raise ValueError("tile_sizes requires at least two groups covering the selected visual tokens exactly")
        parts = scores.split(sizes, dim=-1)
        mass = torch.stack([part.sum(-1) for part in parts], dim=-1)
        tensors.update(tile_mass=mass, first_last_tile_ratio=_mass_ratio(parts[0], parts[-1]),
                       first_last_tile_mean_ratio=_mass_ratio(parts[0], parts[-1], mean=True))
        metadata.update(tile_sizes=sizes, tile_order="caller_declared_native_sequence_order")
    return ProbeResult("attention_position_diagnostics", tensors, metadata)

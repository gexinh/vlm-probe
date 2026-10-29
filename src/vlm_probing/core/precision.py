"""Precision for analysis arithmetic, independent of the model's native dtype."""
import torch
from torch import Tensor


def analysis_dtype(*values: Tensor) -> torch.dtype:
    """Use at least FP32 and preserve a caller's wider floating-point dtype."""
    dtype = torch.float32
    for value in values:
        dtype = torch.promote_types(dtype, value.dtype)
    return dtype


def analysis_difference(source: Tensor, receiver: Tensor) -> Tensor:
    """Subtract after promotion so narrowing cannot corrupt a measured effect."""
    # Exact intervention runners also accept integral scores. Widening those
    # to FP32 would discard integer differences above 2**24.
    if not any(value.is_floating_point() or value.is_complex() for value in (source, receiver)):
        return source - receiver
    dtype = analysis_dtype(source, receiver)
    return source.to(dtype) - receiver.to(dtype)

"""Grad-CAM over attention heads, as used by the official ViT baselines."""

import torch
from torch import Tensor

from ..core.types import ProbeResult
from ..core.precision import analysis_dtype
from ._attribution_utils import finite_result, matching_gradient
from .base import BaseAttention


class AttentionGradCAM(BaseAttention):
    """Use heads as Grad-CAM channels for each selected attention query.

    For query q, alpha[h,q] = mean_accessible_selected_keys(dS/dA[h,q,k]) and
    CAM[q,k] = relu(mean_heads(alpha[h,q] * A[h,q,k])). The comparison's
    ViT baseline selects the final CLS query and excludes the CLS key.
    This is an attention-map Grad-CAM variant, not CAM over image features.
    """

    name = "attention_grad_cam"

    def run(self, attention: Tensor, gradients: Tensor, *,
            query_mask: Tensor | None = None, key_mask: Tensor | None = None,
            support_mask: Tensor | None = None,
            normalize: bool = False) -> ProbeResult:
        """Return CAM [batch,query,key]; key_mask selects averaging/display keys.

        ``support_mask`` declares structurally accessible pairs, independently
        of probability magnitude. It is boolean [B or 1,H or 1,Q,K]; None
        declares unrestricted support. Adapters supply the native causal,
        padding or window mask. Do not infer support from ``attention > 0``.
        Masks select the map; attention is not renormalized over selected keys.
        Pass gradients from this attention's forward and an explicit target.
        """
        qmask, _ = self._probabilities(attention, query_mask)
        batch, heads, queries, keys = attention.shape
        kmask = self._mask(key_mask, (batch, keys), attention, "key_mask")
        if not kmask.any(-1).all():
            raise ValueError("each example needs at least one selected key")
        matching_gradient(attention, gradients, "gradients")
        dtype = analysis_dtype(attention, gradients)
        attention, gradients = attention.to(dtype), gradients.to(dtype)
        selected = qmask[:, None, :, None] & kmask[:, None, None, :]
        if support_mask is not None:
            if (not isinstance(support_mask, Tensor) or support_mask.dtype != torch.bool
                    or support_mask.device != attention.device or support_mask.ndim != 4
                    or support_mask.shape[0] not in (1, batch)
                    or support_mask.shape[1] not in (1, heads)
                    or tuple(support_mask.shape[-2:]) != (queries, keys)):
                raise ValueError("support_mask must be boolean [B or 1,H or 1,Q,K] on the attention device")
            selected = selected & support_mask
        head_weights = gradients.masked_fill(~selected, 0).sum(-1)
        # Empty intersections (e.g. a text query preceding every visual key)
        # have zero weight and map, rather than an undefined mean.
        head_weights = head_weights / selected.sum(-1).clamp_min(1)
        cam = (attention * head_weights[..., None]).masked_fill(~selected, 0).mean(1).clamp_min(0)
        pair_mask = selected.any(1)
        cam = cam.masked_fill(~pair_mask, 0)
        if normalize:
            valid = pair_mask.any(-1, keepdim=True)
            minimum = cam.masked_fill(~pair_mask, float("inf")).amin(-1, keepdim=True).masked_fill(~valid, 0)
            maximum = cam.masked_fill(~pair_mask, float("-inf")).amax(-1, keepdim=True).masked_fill(~valid, 0)
            width = maximum - minimum
            cam = (cam - minimum) / width.masked_fill(width == 0, 1)
            cam = cam.masked_fill(~pair_mask, 0)
        finite_result(cam)
        return ProbeResult(self.name, {"cam": cam, "head_weights": head_weights},
                           {"variant": "attention_heads_as_channels",
                            "analysis_dtype": str(dtype),
                            "gradient_pooling": "accessible_selected_keys_per_query",
                            "structural_support_supplied": support_mask is not None,
                            "head_reduction": "mean", "relu_after_head_reduction": True,
                            "normalized_for_display": normalize,
                            "reference": "https://arxiv.org/abs/1610.02391",
                            "attention_baseline_code": "https://github.com/hila-chefer/Transformer-Explainability"})

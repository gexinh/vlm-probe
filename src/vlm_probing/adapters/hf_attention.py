"""Audited consumed-attention taps for Transformers 5.3 and Gemma 4 in 5.5.4.

The numerical operations and their order match ``eager_attention_forward``.
The backend override lasts only for an attention call: decoder mask construction
must still see the ordinary eager backend. Adapter execution is non-reentrant.
"""
from functools import wraps
from importlib import import_module
from inspect import signature

import torch
from torch import nn

from .spec import CapabilityError


_AUDITED_CLASSES = {
    (f"transformers.models.{family}.modeling_{family}", name)
    for family, name in (
        ("llama", "LlamaAttention"),
        ("mistral", "MistralAttention"),
        ("qwen2", "Qwen2Attention"),
        ("qwen3", "Qwen3Attention"),
        ("qwen2_vl", "Qwen2VLAttention"),
        ("qwen2_5_vl", "Qwen2_5_VLAttention"),
        ("qwen3_vl", "Qwen3VLTextAttention"),
        ("qwen3_5", "Qwen3_5Attention"),
        ("gemma4", "Gemma4TextAttention"),
    )
}
_SUPPORT_ONLY_CLASSES = {
    ("transformers.models.gpt2.modeling_gpt2", "GPT2Attention"),
    ("transformers.models.llama.modeling_llama", "LlamaAttention"),
    ("transformers.models.qwen2_5_vl.modeling_qwen2_5_vl", "Qwen2_5_VLAttention"),
}


def _attention_support(mask, *, batch, heads, queries, keys, device):
    """Decode native eager masks without confusing bias/underflow with masking.

    HF's audited mask constructors use negative infinity or the mask dtype's
    minimum finite value for unavailable pairs. Other finite additive biases
    remain allowed, even if their softmax probability later underflows to zero.
    """
    if mask is None:
        return torch.ones((batch, 1, queries, keys), dtype=torch.bool, device=device)
    if (mask.ndim != 4 or mask.shape[0] not in (1, batch)
            or mask.shape[1] not in (1, heads) or mask.shape[2] not in (1, queries)
            or mask.shape[3] != keys or mask.device != device):
        raise CapabilityError("native attention support requires an aligned 4D attention mask")
    if mask.dtype == torch.bool:
        allowed = mask
    elif mask.is_floating_point():
        allowed = ~(torch.isneginf(mask) | (mask == torch.finfo(mask.dtype).min))
    else:
        raise CapabilityError("native attention support requires a boolean or floating attention mask")
    return allowed.expand(batch, mask.shape[1], queries, keys)


def _support_requested(module):
    # Capture hooks are temporary. Avoid allocating a dense mask in ordinary
    # model forwards and in probes which do not request pairwise support.
    site = module._probe_support
    return bool(site._forward_hooks or site._forward_pre_hooks)


def _emit_support(module, mask, *, batch, heads, queries, keys, device, causal_bias=None):
    if not _support_requested(module):
        return
    support = _attention_support(mask, batch=batch, heads=heads,
                                 queries=queries, keys=keys, device=device)
    if causal_bias is not None:
        # Legacy GPT-2 applies its registered native bias inside eager attention.
        support = support & causal_bias[:, :, keys-queries:keys, :keys].bool()
    module._probe_support(support)


def install_support_only(attention):
    """Expose support for the audited legacy adapters without altering arithmetic.

    The same wrapper serves GPT-2 in 5.3 and the registered 4.57 GPT-2,
    Llama and Qwen2.5-VL contracts. GPT-2 alone may supply a native causal bias.
    """
    identity = type(attention).__module__, type(attention).__name__
    if (identity not in _SUPPORT_ONLY_CLASSES
            or attention.config._attn_implementation != "eager"):
        return False
    if hasattr(attention, "_probe_support"):
        return True
    attention.add_module("_probe_support", nn.Identity())
    original = attention.forward
    parameters = signature(original)

    @wraps(original)
    def forward(*args, **kwargs):
        # If consumed-probability taps are installed later, their backend emits
        # support once. Retain this wrapper without creating a second capture.
        if _support_requested(attention) and not hasattr(attention, "_probe_probs"):
            bound = parameters.bind_partial(*args, **kwargs).arguments
            if bound.get("past_key_values") is not None or bound.get("layer_past") is not None:
                raise CapabilityError("native support capture requires an uncached attention call")
            hidden = bound["hidden_states"]
            heads = (attention.num_heads if identity[1] == "GPT2Attention"
                     else attention.config.num_attention_heads)
            _emit_support(attention, bound.get("attention_mask"),
                          batch=hidden.shape[0], heads=heads,
                          queries=hidden.shape[1], keys=hidden.shape[1], device=hidden.device,
                          causal_bias=getattr(attention, "bias", None) if identity[1] == "GPT2Attention" else None)
        return original(*args, **kwargs)

    attention.forward = forward
    return True


def install_eager_taps(attention):
    """Return whether this exact eager architecture supports editable scores/probabilities."""
    if ((type(attention).__module__, type(attention).__name__) not in _AUDITED_CLASSES
            or attention.config._attn_implementation != "eager"):
        return False
    if hasattr(attention, "_probe_scores"):
        return True
    native = import_module(type(attention).__module__)
    functions, repeat_kv = native.ALL_ATTENTION_FUNCTIONS, native.repeat_kv

    def tapped_eager(module, query, key, value, attention_mask, scaling, dropout=0., **kwargs):
        _emit_support(module, attention_mask, batch=query.shape[0], heads=query.shape[1],
                      queries=query.shape[2], keys=key.shape[2], device=query.device)
        key = repeat_kv(key, module.num_key_value_groups)
        value = repeat_kv(value, module.num_key_value_groups)
        scores = torch.matmul(query, key.transpose(2, 3)) * scaling
        if attention_mask is not None:
            scores = scores + attention_mask
        scores = module._probe_scores(scores)
        weights = nn.functional.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
        weights = nn.functional.dropout(weights, p=dropout, training=module.training)
        weights = module._probe_probs(weights)
        output = torch.matmul(weights, value).transpose(1, 2).contiguous()
        return output, weights

    backend = "vlm_probing_consumed_eager"
    functions.register(backend, tapped_eager)
    attention.add_module("_probe_scores", nn.Identity())
    attention.add_module("_probe_probs", nn.Identity())
    if not hasattr(attention, "_probe_support"):
        attention.add_module("_probe_support", nn.Identity())
    original = attention.forward

    @wraps(original)
    def forward(*args, **kwargs):
        previous = attention.config._attn_implementation
        # Respect a caller who switches away from eager after constructing Prober.
        if previous != "eager":
            raise ValueError("editable native attention requires attn_implementation='eager'")
        try:
            attention.config._attn_implementation = backend
            return original(*args, **kwargs)
        finally:
            attention.config._attn_implementation = previous

    attention.forward = forward
    return True

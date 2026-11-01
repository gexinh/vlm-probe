"""Architectural caveats for attention propagation without changing its formula."""
import warnings


class CausalAttentionWarning(UserWarning):
    """Attention propagation is being transferred to a causally masked stack."""


_PROPAGATION = frozenset({"rollout", "relevance", "tam", "beyond_intuition"})
_CAUSAL_CAVEAT = (
    "Causal attention propagation can favor early sequence positions because "
    "later tokens can access earlier ones. Maps still depend on attention "
    "weights and, for target-conditioned methods, target gradients; they are "
    "not a content-independent baseline or a causal-effect measurement. "
    "The method's propagation formula is unchanged."
)


def propagation_applicability(spec, method):
    """Return explicit mask semantics and warn only for declared causal stacks.

    Unknown custom adapters remain unknown: language-shaped outputs do not imply
    causal attention, and bidirectional attention is not a promise of no bias.
    """
    if method not in _PROPAGATION:
        return None
    causal = spec.causal_attention
    if causal is True:
        caveat = _CAUSAL_CAVEAT
        warnings.warn(f"{method}: {caveat}", CausalAttentionWarning, stacklevel=3)
    elif causal is False:
        caveat = ("The adapter declares bidirectional attention. This excludes the "
                  "causal-mask caveat, but does not establish spatial faithfulness "
                  "or causal influence.")
    else:
        caveat = ("The adapter has not declared causal_attention; attention-mask "
                  "applicability is unknown. Inspect the model's actual masks "
                  "before interpreting propagation maps.")
    result = {"causal_attention": causal,
              "attention_pattern": "unspecified" if causal is None else "causal" if causal else "bidirectional",
              "propagation_formula_modified": False,
              "caveat": caveat}
    if spec.last_stream_kv_layer is not None:
        result["propagation_limit"] = spec.last_stream_kv_layer
        result["caveat"] = (f"{caveat} Propagation stops at layer {spec.last_stream_kv_layer} "
                            "because later layers reuse stored K/V.")
    return result

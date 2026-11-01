"""Capability reports for discovery; method execution validates separately."""


def describe_methods(probe):
    """Report method availability from the bound adapter without executing it."""
    s = probe.spec
    residual = sorted(s.residuals)
    readout = getattr(probe.adapter, "readout", None) is not None
    methods = {}
    propagation = []
    for i in range(max(s.residuals) + 1):
        if i not in s.attentions or (s.last_stream_kv_layer is not None
                                     and i > s.last_stream_kv_layer):
            break
        propagation.append(i)
    rules = {
        "sae.features": (residual, "a fitted dictionary bound to an explicit residual hook"),
        "sae.edit": (residual, "a fitted dictionary and an editable residual hook"),
        "sae.profile": (residual, "a fitted dictionary and explicit token positions"),
        "sae.reconstruct": (residual, "a fitted same-space dictionary and an editable residual hook"),
        "lens.logit": (residual if readout else [], "residual sites and model readout"),
        "lens.tuned": (residual if readout else [], "residual sites and model readout"),
        "lens.jacobian": (residual if readout else [], "residual sites and model readout"),
        "lens.mj": (sorted(s.mj.sources) if s.mj is not None and s.causal_attention is True else [],
                    "causal block-input sites, actual final RMSNorm target and explicit modality layout"),
        "lens.embed": (["embeddings"] if s.embeddings and s.input_embeddings is not None else [], "embedding-space site and input embeddings"),
        "lens.attention": (sorted(s.heads) if s.input_embeddings is not None else [], "projected heads and vocabulary dimensions"),
        "lens.patchscope": (residual, "residual sites"),
        "attention.profile": (sorted(s.attentions), "observable attention probabilities"),
        "attention.rollout": (propagation, "observable self-attention at consecutive layers from 0"),
        "attention.relevance": (propagation, "differentiable self-attention at consecutive layers from 0"),
        "attention.grad_cam": (sorted(s.attentions), "differentiable target-conditioned attention"),
        "attention.attribution": (sorted(s.editable_attention), "editable consumed probabilities for attention-path integration"),
        "attention.tam": (propagation, "consecutive attention stack and floating input-path tensor"),
        "attention.beyond_intuition": (propagation, "consecutive attention stack; token variant also needs pre-LN inputs and projected V"),
        "attention.dtd_lrp": (residual if s.output_kind == "classification" and
                             type(probe.model).__name__ == "ViTForImageClassification" else [],
                             "native HF ViT converted to Chefer's original-rule backend"),
        "attention.head_logits": (sorted(s.heads) if s.linear_readout else [], "projected heads and fixed-scale readout"),
        "attention.reweight": (sorted(s.editable_attention), "editable probabilities before A @ V"),
        "attention.temperature": (sorted(s.attention_scores), "editable masked attention logits"),
        "causal.patch": (residual, "residual sites"),
        "causal.path": (sorted(s.path_heads) if set(s.path_heads) == set(s.residuals)
                        and (s.path_final or s.path_qkv) else [],
                        "editable outputs at every decoder layer and receiver Q/K/V or final residual"),
        "causal.attribute": (residual, "differentiable residual sites"),
        "causal.steer": (residual, "residual sites"),
        "causal.knockout": (sorted(s.attention_scores), "editable masked attention logits"),
        "causal.eap_ig": (list(s.edges), "explicit independently replaceable edge sites"),
    }
    for name, (sites, need) in rules.items():
        if s.output_kind == "representation" and not name.startswith("sae."):
            sites, need = [], "representation encoder currently exposes SAE methods; no language/classification logits"
        methods[name] = {"available": bool(sites), "sites": sites,
                         "requires": need, "missing": None if sites else need}
    native_gpt2 = (type(probe.model).__module__ == "transformers.models.gpt2.modeling_gpt2"
                   and type(probe.model).__name__ == "GPT2LMHeadModel"
                   and not probe.model.config.add_cross_attention
                   and not getattr(probe.model.config, "reorder_and_upcast_attn", False))
    edge_method = methods["causal.eap_ig"]
    edge_method["graphs"] = {"activation": bool(s.edges), "transformer": native_gpt2}
    if native_gpt2:
        edge_method.update(available=True, missing=None,
                           requires="graph='transformer' for the native GPT-2 residual-message graph")
    if s.mj is not None:
        methods["lens.mj"]["tap_convention"] = "decoder_block_input"
        methods["lens.mj"]["output_channels"] = ["text"] + (["visual"] if s.mj.visual_head is not None else [])
    return methods

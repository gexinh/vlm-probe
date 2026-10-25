"""Explicit Hugging Face contracts, tested against Transformers 5.3.0.

Only native Transformers classes are registered. Audited eager decoder taps
expose probabilities actually consumed by A @ V, including Qwen and Mistral.
"""
from importlib.metadata import version

import torch

from .spec import CapabilityError, HeadSite, MJSpec, ModelSpec, TokenLayout
from .torch import HookPoint, ModelReadout, TorchModelAdapter
from .hf_attention import install_eager_taps, install_support_only


# Exact module/class identities; wrapper structure is never guessed from shapes.
MODEL_CONTRACTS = {
    (f"transformers.models.{module}.modeling_{module}", name): (prefix, visual_ids)
    for module, name, prefix, visual_ids in (
        ("llama", "LlamaForCausalLM", "model", ()),
        ("qwen2", "Qwen2ForCausalLM", "model", ()),
        ("qwen3", "Qwen3ForCausalLM", "model", ()),
        ("gpt2", "GPT2LMHeadModel", "transformer", ()),
        ("qwen2_vl", "Qwen2VLForConditionalGeneration", "model.language_model", ("image_token_id", "video_token_id")),
        ("qwen2_5_vl", "Qwen2_5_VLForConditionalGeneration", "model.language_model", ("image_token_id", "video_token_id")),
        ("qwen3_vl", "Qwen3VLForConditionalGeneration", "model.language_model", ("image_token_id", "video_token_id")),
        ("qwen3_5", "Qwen3_5ForConditionalGeneration", "model.language_model", ("image_token_id", "video_token_id")),
        ("idefics3", "Idefics3ForConditionalGeneration", "model.text_model", ("image_token_id",)),
        ("internvl", "InternVLForConditionalGeneration", "model.language_model", ("image_token_id",)),
        ("llava", "LlavaForConditionalGeneration", "model.language_model", ("image_token_index",)),
        ("janus", "JanusForConditionalGeneration", "model.language_model", ("image_token_id",)),
    )
}


def make_adapter(model):
    installed = version("transformers")
    release = installed.split(".")[:2]
    name = type(model).__name__
    legacy = release == ["4", "57"] and name in {
        "LlamaForCausalLM", "Qwen2_5_VLForConditionalGeneration", "GPT2LMHeadModel"
    }
    if release != ["5", "3"] and not legacy:
        raise CapabilityError(f"HF adapters require transformers 5.3.x (tested: 5.3.0), got {installed}; "
                              "install vlmprobe[transformers] or pass an explicit adapter")
    prefix, visual_fields = MODEL_CONTRACTS[(type(model).__module__, name)]
    decoder = model.get_submodule(prefix)
    # Composite architectures allow other text backbones; only audited ones count.
    allowed = {"Idefics3ForConditionalGeneration": {"llama"},
               "InternVLForConditionalGeneration": {"qwen2", "llama"},
               "LlavaForConditionalGeneration": {"llama", "mistral"},
               "JanusForConditionalGeneration": {"llama"}}
    if name in allowed and decoder.config.model_type not in allowed[name]:
        raise CapabilityError(f"{name} text backbone {decoder.config.model_type!r} is not supported")
    hybrid = name == "Qwen3_5ForConditionalGeneration"
    gpt2 = name == "GPT2LMHeadModel"
    if gpt2 and model.config.add_cross_attention:
        raise CapabilityError("GPT2 cross-attention configurations require an explicit adapter")
    layers = decoder.h if gpt2 else decoder.layers
    norm = decoder.ln_f if gpt2 else decoder.norm
    embedding = decoder.wte if gpt2 else decoder.embed_tokens
    layer_path = f"{prefix}.h" if gpt2 else f"{prefix}.layers"
    norm_path = f"{prefix}.ln_f" if gpt2 else f"{prefix}.norm"
    sites, residuals, heads, attentions, attention_scores = {}, {}, {}, {}, {}
    path_heads, path_qkv = {}, {}
    attention_inputs, attention_values, editable_attention = {}, {}, set()
    attention_supports = {}
    for i, layer in enumerate(layers):
        residuals[i] = f"residual.{i}"
        # Include Qwen3-VL's DeepStack addition after the block. This also avoids
        # relying on decoder-layer output containers that differ by HF version.
        following = f"{layer_path}.{i + 1}" if i + 1 < len(layers) else norm_path
        sites[residuals[i]] = HookPoint(following, kind="input", selector=0)
        if hybrid and layer.layer_type != "full_attention":
            continue
        heads[i] = f"heads.{i}"
        attn = layer.attn if gpt2 else layer.self_attn
        attn_path = f"{layer_path}.{i}.attn" if gpt2 else f"{layer_path}.{i}.self_attn"
        projection = "c_proj" if gpt2 else "o_proj"
        sites[heads[i]] = HookPoint(f"{attn_path}.{projection}", kind="input", selector=0)
        if not hybrid:
            count = attn.num_heads if gpt2 else attn.config.num_attention_heads
            kv_count = count if gpt2 else attn.config.num_key_value_heads
            path_heads[i] = HeadSite(heads[i], count, attn.head_dim)
            path_qkv[i] = {}
            for index, kind in enumerate(("q", "k", "v")):
                site = f"{kind}.{i}"
                if gpt2:
                    start = index * attn.split_size
                    sites[site] = HookPoint(f"{attn_path}.c_attn",
                                            tensor_slice=(start, start + attn.split_size))
                else:
                    # Qwen3 and Qwen3-VL normalize each Q/K head before rotary position
                    # encoding. Capture the actual normalized receiver input.
                    normalized = name in {"Qwen3ForCausalLM", "Qwen3VLForConditionalGeneration"} and kind in {"q", "k"}
                    component = f"{kind}_norm" if normalized else f"{kind}_proj"
                    sites[site] = HookPoint(f"{attn_path}.{component}")
                path_qkv[i][kind] = HeadSite(site, count if kind == "q" else kv_count,
                                            attn.head_dim)
        if attn.config._attn_implementation == "eager":
            attentions[i] = f"attention.{i}"
            sites[attentions[i]] = HookPoint(attn_path, selector=1)
            if release == ["5", "3"] and install_eager_taps(attn):
                attention_scores[i] = f"scores.{i}"
                sites[attention_scores[i]] = HookPoint(f"{attn_path}._probe_scores")
                sites[attentions[i]] = HookPoint(f"{attn_path}._probe_probs")
                editable_attention.add(i)
            if hasattr(attn, "_probe_support") or install_support_only(attn):
                attention_supports[i] = f"attention_support.{i}"
                sites[attention_supports[i]] = HookPoint(f"{attn_path}._probe_support")
        if not hybrid and not gpt2:
            attention_inputs[i] = residuals[i-1] if i else "attention_input.0"
            if not i:
                # The embedding alias already names this same pre-LN tensor.
                attention_inputs[i] = "embeddings"
            attention_values[i] = path_qkv[i]["v"].site
    sites["embeddings"] = HookPoint(f"{layer_path}.0", kind="input", selector=0)

    def layout(inputs):
        ids = inputs.get("input_ids")
        if ids is None or ids.ndim != 2:
            raise CapabilityError("HF auto layout requires input_ids; supply ProbeInputs(layout=...) otherwise")
        mask = inputs.get("attention_mask")
        valid = torch.ones_like(ids, dtype=torch.bool) if mask is None else mask.bool()
        if valid.shape != ids.shape:
            raise CapabilityError("HF auto layout requires a 2D attention_mask aligned with input_ids")
        visual = torch.zeros_like(ids, dtype=torch.bool)
        for field in visual_fields:
            visual |= ids == getattr(model.config, field)
        return TokenLayout(valid, visual & valid, token_ids=ids)

    def project_heads(i, hidden):
        attn = layers[i].attn if gpt2 else layers[i].self_attn
        count = attn.num_heads if gpt2 else attn.config.num_attention_heads
        # H * head_dim can differ from residual width. Qwen3.5 applies its output
        # gate before o_proj, so this site captures the actual gated values.
        values = hidden.reshape(*hidden.shape[:2], count, attn.head_dim)
        if gpt2:
            # HF Conv1D stores its matrix as [input, output], unlike nn.Linear.
            weights = attn.c_proj.weight.reshape(count, attn.head_dim, attn.embed_dim)
            return torch.einsum("bthd,hdo->btho", values, weights)
        weights = attn.o_proj.weight.reshape(attn.o_proj.out_features, count, attn.head_dim)
        return torch.einsum("bthd,ohd->btho", values, weights)

    def linear_readout(full):
        norm_weight = 1.0 + norm.weight if hybrid else norm.weight
        eps = norm.eps if hybrid else norm.variance_epsilon
        weight = model.lm_head.weight * norm_weight
        scale = (full.float().square().mean(-1, keepdim=True) + eps).rsqrt()
        return weight.to(full), scale.to(full), False

    def project_values(i, raw):
        attn = layers[i].self_attn
        kv = raw.reshape(*raw.shape[:2], attn.config.num_key_value_heads, attn.head_dim)
        expanded = kv.repeat_interleave(attn.num_key_value_groups, dim=2).flatten(2)
        return torch.nn.functional.linear(expanded, attn.o_proj.weight)

    notes = [
        "Residual layer i is measured after block i and visual injection, before the next block/final norm.",
        "Attention observation requires eager attention; projected heads exclude output bias.",
        ("Audited native eager scores and consumed probabilities before A @ V are editable."
         if attention_scores else "Native attention-score intervention sites are unavailable for this backbone."),
        "Computational edges require an explicit graph adapter.",
        "Causal attention propagation can favor early sequence positions; weights and target gradients still affect the maps.",
        f"Uncached full-sequence image/text forwards; Transformers {installed}.",
    ]
    if hybrid:
        notes.append("Qwen3.5 head/attention methods cover full_attention layers only; propagation cannot skip linear layers.")
        notes.append("IOI-style path patching is unavailable: linear-attention outputs are not declared as independently frozen heads.")
    else:
        notes.extend([
            "Path sender outputs are frozen before output projection; the fixed linear projection preserves each head contribution.",
            "Path Q/K receivers are measured before positional rotation when present; Qwen3 and Qwen3-VL use normalized Q/K, and K/V index physical KV heads.",
        ])
    if gpt2:
        notes.append("GPT2 Q/K/V use disjoint packed-projection slices; fixed-scale head logit attribution is not exposed for its affine LayerNorm.")
    if name == "Qwen3VLForConditionalGeneration":
        notes.append("Rollout/relevance summarize language self-attention; DeepStack injection is not an attention edge.")
    mj = None
    if not gpt2 and release == ["5", "3"]:
        mj = MJSpec(
            sources={i: "embeddings" if i == 0 else residuals[i - 1] for i in range(len(layers))},
            final=residuals[len(layers) - 1], norm=norm,
            text_unembedding=model.lm_head.weight,
            norm_gain=1.0 + norm.weight if hybrid else norm.weight,
            visual_head=model.model.generation_head if name == "JanusForConditionalGeneration" else None,
            decode_visual_codes=model.decode_image_tokens if name == "JanusForConditionalGeneration" else None,
        )
        notes.append("MJ-Lens uses decoder block-input indices; layer 0 is the merged embedding input.")
    if name == "JanusForConditionalGeneration":
        notes.append("Janus image-output replay requires explicit generation embeddings and token layout; codebook IDs are not text IDs.")
    spec = ModelSpec(
        residuals, attentions=attentions, attention_scores=attention_scores,
        heads=heads, embeddings="embeddings",
        input_embeddings=embedding.weight, layout=layout,
        project_heads=project_heads, linear_readout=None if gpt2 else linear_readout,
        editable_attention=editable_attention, attention_inputs=attention_inputs,
        attention_values=attention_values, project_values=None if hybrid or gpt2 else project_values,
        path_heads=path_heads, path_qkv=path_qkv,
        path_final=residuals[len(layers)-1] if not hybrid else None,
        alignment_keys=("image_grid_thw", "video_grid_thw", "position_ids", "second_per_grid_ts",
                        "pixel_attention_mask", "image_sizes", "token_type_ids", "mm_token_type_ids"),
        forward_defaults={"use_cache": False, "return_dict": True}, notes=tuple(notes),
        attention_supports=attention_supports, causal_attention=True, mj=mj,
    )
    model_id = getattr(model.config, "_name_or_path", "") or name
    return TorchModelAdapter(model, sites, model_id=model_id,
                             readout=ModelReadout(model.lm_head, norm=norm), spec=spec)

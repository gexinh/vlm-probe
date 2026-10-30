"""Native Gemma 4 image/text contracts, audited against Transformers 5.5.4.

Keep this version boundary separate from the existing Transformers 5.3 adapters.
Shared K/V must work without a decode cache; 5.5.0 does not preserve that path.
"""
from importlib.metadata import version

import torch

from .hf_attention import install_eager_taps
from .spec import CapabilityError, ModelSpec, TokenLayout
from .torch import HookPoint, ModelReadout, TorchModelAdapter


MODEL_CONTRACTS = {
    ("transformers.models.gemma4.modeling_gemma4", "Gemma4ForConditionalGeneration"):
        "model.language_model",
    ("transformers.models.gemma4.modeling_gemma4", "Gemma4ForCausalLM"): "model",
}


class Gemma4Adapter(TorchModelAdapter):
    """Image/text execution only; do not silently classify audio/video as text."""

    def run(self, inputs, **kwargs):
        config = self.model.config
        if any(inputs.get(key) is not None for key in (
            "input_features", "input_features_mask", "pixel_values_videos", "video_position_ids",
        )):
            raise CapabilityError("Gemma 4 adapter supports image/text only; audio/video inputs are not audited")
        ids = inputs.get("input_ids")
        if ids is not None:
            for field in ("audio_token_id", "video_token_id"):
                token = getattr(config, field, None)
                if token is not None and (ids == token).any():
                    raise CapabilityError("Gemma 4 adapter supports image/text only; audio/video tokens are not audited")
        types = inputs.get("mm_token_type_ids")
        if types is not None and ((types != 0) & (types != 1)).any():
            raise CapabilityError("Gemma 4 adapter supports image/text mm_token_type_ids only (0 or 1)")
        return super().run(inputs, **kwargs)


def make_adapter(model):
    installed = version("transformers")
    # The optional extra pins this exact patch release. Reject newer runtimes
    # until their native execution paths have been validated independently.
    if installed != "5.5.4":
        raise CapabilityError(
            f"Gemma 4 adapters require transformers 5.5.4, got {installed}; "
            "install vlmprobe[gemma4] in a separate environment"
        )
    prefix = MODEL_CONTRACTS[(type(model).__module__, type(model).__name__)]
    decoder = model.get_submodule(prefix)
    multimodal = type(model).__name__ == "Gemma4ForConditionalGeneration"
    sites, residuals, heads = {}, {}, {}
    attentions, scores, supports = {}, {}, {}
    layers = decoder.layers
    for i, layer in enumerate(layers):
        residuals[i] = f"residual.{i}"
        following = f"{prefix}.layers.{i + 1}" if i + 1 < len(layers) else f"{prefix}.norm"
        # Capture after the whole block: post-attention/feed-forward RMSNorm,
        # per-layer input branch and layer_scalar have all been applied.
        sites[residuals[i]] = HookPoint(following, kind="input", selector=0)
        path = f"{prefix}.layers.{i}.self_attn"
        heads[i] = f"heads.{i}"
        sites[heads[i]] = HookPoint(f"{path}.o_proj", kind="input", selector=0)
        if install_eager_taps(layer.self_attn):
            attentions[i], scores[i], supports[i] = f"attention.{i}", f"scores.{i}", f"support.{i}"
            sites[attentions[i]] = HookPoint(f"{path}._probe_probs")
            sites[scores[i]] = HookPoint(f"{path}._probe_scores")
            sites[supports[i]] = HookPoint(f"{path}._probe_support")
    sites["embeddings"] = HookPoint(f"{prefix}.layers.0", kind="input", selector=0)

    def layout(inputs):
        ids = inputs.get("input_ids")
        if ids is None or ids.ndim != 2:
            raise CapabilityError("Gemma 4 auto layout requires input_ids; supply ProbeInputs(layout=...) otherwise")
        mask = inputs.get("attention_mask")
        valid = torch.ones_like(ids, dtype=torch.bool) if mask is None else mask.bool()
        if valid.shape != ids.shape:
            raise CapabilityError("Gemma 4 auto layout requires a 2D attention_mask aligned with input_ids")
        visual = (ids == model.config.image_token_id) if multimodal else torch.zeros_like(valid)
        return TokenLayout(valid, visual & valid, token_ids=ids)

    def project_heads(i, hidden):
        attention = layers[i].self_attn
        count = decoder.config.num_attention_heads
        values = hidden.reshape(*hidden.shape[:2], count, attention.head_dim)
        weight = attention.o_proj.weight.reshape(attention.o_proj.out_features, count, attention.head_dim)
        return torch.einsum("bthd,ohd->btho", values, weight)

    def output_transform(logits):
        cap = decoder.config.final_logit_softcapping
        if cap is not None:
            return torch.tanh(logits / cap) * cap
        return logits

    notes = (
        f"Gemma 4 image/text or text-only forwards; Transformers {installed}; uncached full sequence.",
        "Residual i follows the entire decoder block, including per-layer inputs and layer_scalar.",
        "Readout applies the native final RMSNorm, LM head and configured logit soft-capping.",
        "Embedding capture follows scaled text embeddings and image merge; cosine EmbedLens uses the native vocabulary directions.",
        "Eager attention taps expose actual masked scores, structural support and consumed probabilities, including shared-K/V layers.",
        "Projected heads sum to the bias-free o_proj output BEFORE post_attention_layernorm; they are not additive residual writes.",
        "IOI path patching and fixed-scale head logit attribution are not declared: post-attention normalization couples heads, and K/V may be shared.",
        "MJ-Lens, token-weighted Beyond Intuition and edge-level EAP-IG require additional explicit contracts.",
        "Attention propagation describes the native local/global masks; vision blocks may attend bidirectionally. It does not establish spatial faithfulness.",
        "Audio, video, automatic image grids, quantization, sharding and compiled execution are not validated.",
    )
    spec = ModelSpec(
        residuals=residuals, heads=heads, embeddings="embeddings",
        input_embeddings=decoder.embed_tokens.weight, layout=layout, project_heads=project_heads,
        attentions=attentions, attention_scores=scores, attention_supports=supports,
        editable_attention=set(attentions),
        alignment_keys=("image_position_ids", "position_ids", "mm_token_type_ids"),
        forward_defaults={"use_cache": False, "return_dict": True}, notes=notes,
        causal_attention=decoder.config.use_bidirectional_attention != "all",
    )
    model_id = getattr(model.config, "_name_or_path", "") or type(model).__name__
    return Gemma4Adapter(
        model, sites, model_id=model_id, spec=spec,
        readout=ModelReadout(model.lm_head, norm=decoder.norm, output_transform=output_transform),
    )

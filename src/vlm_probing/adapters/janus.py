"""Explicit teacher-forced image-output inputs for native Transformers Janus."""
from collections.abc import Mapping

import torch

from .spec import CapabilityError, ProbeInputs, TokenLayout


def prepare_janus_generation(probe, prompt_inputs, codes, *, context="image_output"):
    """Build a full-sequence Janus replay without mixing token namespaces.

    ``prompt_inputs`` contains an unpadded text ``input_ids`` batch, optionally
    its all-one attention mask. ``codes`` is a [batch, positions] long tensor
    of already generated visual code IDs, with at least two positions. These
    codes may be a prefix; a complete VQ grid is needed only when rendering.

    Visual targets are states carrying code i and predicting code i+1. The
    final code is excluded because its successor is unavailable. The last
    text prompt state predicts the first image code but is deliberately not a
    visual-position target under the paper's position-modality convention.
    The backbone and its training flags are unchanged. No images are generated
    or decoded by this helper; callers supply their own sampled code sequence.
    """
    model = probe.model
    if (type(model).__module__, type(model).__name__) != (
            "transformers.models.janus.modeling_janus", "JanusForConditionalGeneration"):
        raise CapabilityError("prepare_janus_generation requires native Transformers JanusForConditionalGeneration")
    if context != "image_output":
        raise ValueError("Janus generation replay requires context='image_output'")
    explicit = None
    if isinstance(prompt_inputs, ProbeInputs):
        values, explicit = dict(prompt_inputs.kwargs), prompt_inputs.layout
    elif isinstance(prompt_inputs, Mapping):
        values = dict(prompt_inputs)
    else:
        raise TypeError("prompt_inputs must be text model kwargs or ProbeInputs")
    unexpected = set(values) - {"input_ids", "attention_mask", "use_cache", "return_dict"}
    if unexpected:
        raise ValueError(f"Janus replay expects text prompt IDs only; unsupported kwargs: {sorted(unexpected)}")
    if values.get("use_cache") is True:
        raise ValueError("Janus replay requires use_cache=False")
    ids = values.get("input_ids")
    if not isinstance(ids, torch.Tensor) or ids.dtype != torch.long or ids.ndim != 2 or not ids.numel():
        raise ValueError("Janus prompt input_ids must be a nonempty long [batch, tokens] tensor")
    embedding = model.get_input_embeddings()
    if bool(((ids < 0) | (ids >= embedding.num_embeddings)).any()):
        raise ValueError("Janus prompt IDs are outside the text vocabulary")
    if bool((ids == model.config.image_token_id).any()):
        raise ValueError("Janus image-output prompt must be text-only; image placeholder IDs are not allowed")
    attention_mask = values.get("attention_mask")
    if attention_mask is not None:
        if (not isinstance(attention_mask, torch.Tensor) or attention_mask.shape != ids.shape
                or not bool((attention_mask == 1).all())):
            raise ValueError("Janus generation replay currently requires unpadded prompts with a 2D all-one attention_mask")
    if explicit is not None:
        explicit = explicit.checked(ids.shape, ids.device)
        if (not bool(explicit.valid.all())
                or (explicit.visual is not None and bool(explicit.visual.any()))
                or (explicit.prompt is not None and not bool(explicit.prompt.all()))
                or (explicit.token_ids is not None and not torch.equal(explicit.token_ids, ids))):
            raise ValueError("Janus generation replay requires an unpadded text-only prompt layout aligned with input_ids")
    if (not isinstance(codes, torch.Tensor) or codes.dtype != torch.long or codes.ndim != 2
            or codes.shape[0] != ids.shape[0] or codes.shape[1] < 2):
        raise ValueError("Janus visual codes must be long [batch, positions], matching the prompt batch, with at least two codes")
    generation_embedding = model.model.generation_embeddings
    if bool(((codes < 0) | (codes >= generation_embedding.num_embeddings)).any()):
        raise ValueError("Janus visual code IDs are outside the visual codebook")
    with torch.no_grad():
        prompt_embeddings = embedding(ids.to(embedding.weight.device))
        code_embeddings = model.prepare_embeddings_for_image_generation(
            codes.to(generation_embedding.weight.device))
        # Sharded models may place the generation aligner separately from the
        # text embeddings; the decoder's input space is the shared destination.
        code_embeddings = code_embeddings.to(prompt_embeddings)
        inputs_embeds = torch.cat((prompt_embeddings, code_embeddings), dim=1).detach()
    valid = torch.ones(inputs_embeds.shape[:2], dtype=torch.bool, device=inputs_embeds.device)
    visual = torch.zeros_like(valid)
    visual[:, ids.shape[1]:] = True
    prompt = valid & ~visual
    targets = visual.clone()
    targets[:, -1] = False
    inputs = ProbeInputs(
        {"inputs_embeds": inputs_embeds, "attention_mask": valid.long(),
         "use_cache": False, "return_dict": True},
        TokenLayout(valid=valid, visual=visual, prompt=prompt, token_ids=None),
    )
    # Import lazily: the adapter contract itself does not depend on the public
    # MJ-Lens runner, and normal Janus probing needs no generation replay.
    from ..api.mj import MJBatch
    return MJBatch(inputs=inputs, target_masks={"visual": targets, "text": torch.zeros_like(valid)},
                   context=context)

"""Explicit native DINO vision hooks for published Prisma patch dictionaries.

The caller loads source-matched weights and preprocessing. An adapter contract
does not establish checkpoint equivalence merely from matching hidden width.
"""
from importlib.metadata import version

import torch

from .spec import CapabilityError, ModelSpec, TokenLayout
from .torch import HookPoint, TorchModelAdapter


class _DinoVisionAdapter(TorchModelAdapter):
    def run(self, *args, **kwargs):
        trace = super().run(*args, **kwargs)
        hidden = trace.output.last_hidden_state
        # Match Prisma DINO pre_logits: retain normalized CLS and mean patches
        # separately. SAE sites below are pre-final-normalization residuals.
        trace.output = torch.stack((hidden[:, 0], hidden[:, 1:].mean(1)), dim=-1)
        return trace


def make_adapter(model, *, source_model_name):
    """Expose native DINO block outputs before final norm, with CLS at zero.

    Instantiate native HF ``ViTModel`` with the exact ``facebook/dino-vitb16``
    checkpoint, and pass its source-matched processor separately to ``Prober``.
    This explicit adapter does not register unrelated ViT models automatically.
    Source/native tensor and preprocessing equivalence remain caller obligations.
    """
    if version("transformers").split(".")[:2] != ["5", "3"]:
        raise CapabilityError("DINO vision residual hooks are audited for transformers 5.3.x")
    identity = (type(model).__module__, type(model).__name__)
    if identity != ("transformers.models.vit.modeling_vit", "ViTModel"):
        raise CapabilityError("DINO vision adapter requires native HF ViTModel")
    if source_model_name != "facebook/dino-vitb16":
        raise CapabilityError("declare the audited DINO ViT-B/16 source checkpoint")
    config = model.config
    if type(config.image_size) is not int or type(config.patch_size) is not int:
        raise CapabilityError("DINO vision adapter requires scalar image and patch sizes")
    if config.image_size % config.patch_size:
        raise CapabilityError("vision image size must be divisible by patch size")
    grid = config.image_size // config.patch_size
    residuals = {i: f"vision.residual.{i}" for i in range(len(model.encoder.layer))}
    sites = {site: HookPoint(f"encoder.layer.{i}") for i, site in residuals.items()}
    sites["vision.embeddings"] = HookPoint("encoder.layer.0", kind="input", selector=0)

    def layout(inputs):
        pixels = inputs.get("pixel_values")
        if not isinstance(pixels, torch.Tensor) or pixels.ndim != 4:
            raise CapabilityError("vision SAE requires pixel_values[B,C,H,W]")
        if inputs.get("interpolate_pos_encoding", False):
            raise CapabilityError("interpolated visual grids require a separately verified source binding")
        if tuple(pixels.shape[-2:]) != (config.image_size, config.image_size):
            raise ValueError("vision pixels must match the source checkpoint's fixed native image size")
        valid = torch.ones(pixels.shape[0], grid * grid + 1, dtype=torch.bool, device=pixels.device)
        visual = valid.clone()
        visual[:, 0] = False
        return TokenLayout(valid, visual)

    notes = (
        "Vision layer i is DINO block-i post-MLP residual output, before final normalization.",
        f"CLS is position 0; positions 1..{grid * grid} are row-major {grid}x{grid} patches.",
        "Published patch dictionaries require exact source weights, normalization and preprocessing.",
        "Output retains normalized CLS and pooled patch vectors as [batch,hidden,2], not language logits.",
    )
    spec = ModelSpec(residuals, embeddings="vision.embeddings", layout=layout,
        output_kind="representation", forward_defaults={"return_dict": True},
        notes=notes, causal_attention=False)
    adapter = _DinoVisionAdapter(model, sites, spec=spec,
        model_id=source_model_name)
    adapter.visual_grid = (grid, grid)
    adapter.vision_prefix_tokens = 1
    adapter.site_representations = {site: "native_vision_residual_output" for site in residuals.values()}
    return adapter

"""One-time tensor conversion into native SAE artifacts.

These importers read documented JSON/tensor formats directly. They do not load
upstream Python classes, reorder features, or infer model-hook compatibility
from matching widths. Unsupported variants raise instead of approximating.
"""
from __future__ import annotations

import json
import pickle
import re
from pathlib import Path

import torch


class _PassivePrismaConfig:
    """Data-only replacement for a historical pickled Prisma config class.

    The source class is never imported or executed. PyTorch's restricted
    unpickler may only populate this plain object's attribute dictionary.
    """


def _require_prisma_legacy_loader():
    """Named safe-global aliases were added to PyTorch's loader in 2.6."""
    match = re.match(r"(\d+)\.(\d+)", str(torch.__version__))
    supported = match is not None and tuple(map(int, match.groups())) >= (2, 6)
    if not supported or not callable(getattr(torch.serialization, "safe_globals", None)):
        raise RuntimeError(
            "Loading historical Prisma checkpoints with embedded config objects requires "
            "PyTorch >=2.6 for named weights-only loader aliases. "
            "Upgrade the SAE dependencies with: python -m pip install --upgrade 'vlmprobe[sae]'. "
            "The checkpoint is never loaded with weights_only=False."
        )


def _weights(path, *, source_format=None):
    path = Path(path)
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file
        state = load_file(str(path), device="cpu")
    else:
        try:
            state = torch.load(path, map_location="cpu", weights_only=True)
        except pickle.UnpicklingError:
            if source_format != "prisma":
                raise
            _require_prisma_legacy_loader()
            # Some published Prisma files wrap tensors with a pickled config.
            # Permit only its historical name, mapped to our inert container;
            # all other Python globals still require weights-only approval.
            with torch.serialization.safe_globals([
                (_PassivePrismaConfig, "vit_prisma.sae.config.VisionModelSAERunnerConfig"),
            ]):
                state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise ValueError("checkpoint must contain a tensor dictionary or a supported safe envelope")
    return state


def _validate_prisma_envelope_config(state, config):
    """Reject conflicting external and embedded inference/binding metadata."""
    embedded = state.get("cfg")
    if embedded is None:
        return
    if type(embedded) is _PassivePrismaConfig:
        embedded = vars(embedded)
    if not isinstance(embedded, dict):
        raise ValueError("Prisma envelope config must be a plain data dictionary")
    fields = (
        "model_name", "hook_point_layer", "layer_subtype", "hook_point_head_index",
        "architecture", "activation_fn_str", "activation_fn_kwargs", "normalize_activations",
        "context_size", "cls_token_only", "use_patches_only", "image_size", "is_transcoder",
        "d_out", "out_hook_point_layer", "layer_out_subtype", "apply_b_dec_to_input",
    )
    for name in fields:
        if (name in embedded) != (name in config):
            raise ValueError(f"Prisma embedded configuration differs from config.json: missing {name}")
        if name in embedded and embedded[name] != config[name]:
            raise ValueError(f"Prisma embedded configuration differs from config.json: {name}")
    if _dimensions(embedded) != _dimensions(config):
        raise ValueError("Prisma embedded configuration differs from config.json: dimensions")


def _tensor(state, key, shape=None):
    value = state.get(key)
    if not isinstance(value, torch.Tensor) or not value.is_floating_point() or not torch.isfinite(value).all():
        raise ValueError(f"source checkpoint requires finite floating {key}")
    if shape is not None and tuple(value.shape) != tuple(shape):
        raise ValueError(f"source {key} shape differs from declared dimensions {tuple(shape)}")
    return value


def _dimensions(config):
    d_in = config.get("d_in")
    d_sae = config.get("d_sae", d_in * config.get("expansion_factor", 0) if type(d_in) is int else None)
    if any(type(value) is not int or value < 1 for value in (d_in, d_sae)):
        raise ValueError("source configuration must explicitly declare positive activation/dictionary dimensions")
    return d_in, d_sae


def _binding(config, binding, *, source_format):
    from ..models import validate_sae_binding
    binding = validate_sae_binding(binding)
    model = config.get("model_name", config.get("metadata", {}).get("model_name"))
    hook = config.get("hook_name", config.get("hook_point", config.get("metadata", {}).get("hook_name")))
    layer = config.get("hook_layer", config.get("hook_point_layer", config.get("metadata", {}).get("hook_layer")))
    if source_format == "prisma":
        layer = config.get("hook_point_layer")
        hook = f"blocks.{layer}.{config.get('layer_subtype')}"
    if model is not None and binding.get("backend_model_name", binding["model_id"].split("@", 1)[0]) != model:
        raise ValueError("source checkpoint model differs from the declared native/source binding")
    if hook is not None and binding.get("backend_hook_name", binding["hook"]["module"]) != hook:
        raise ValueError("declare the exact source hook separately from the verified native hook")
    if layer is not None and layer != binding["layer"]:
        raise ValueError("source checkpoint layer differs from the binding")
    if config.get("hook_head_index", config.get("hook_point_head_index")) is not None:
        raise ValueError("head-specific checkpoints require an independently verified adapter and are not imported here")
    return binding


def _construct(config, binding, weights, *, architecture, source_format):
    from ..models import SAEConfig, SparseAutoencoder
    d_in, d_sae = _dimensions(config)
    normalization = config.get("normalize_activations", "none")
    if normalization in {None, "none"}:
        normalization = "none"
    elif source_format in {"prisma", "saelens"} and normalization == "constant_norm_rescale":
        normalization = "constant_norm"
    elif source_format in {"prisma", "saelens"} and normalization == "layer_norm":
        normalization = "layer_norm"
    else:
        raise ValueError("source runtime/folded normalization must be explicitly converted; this variant is unsupported")
    k = config.get("k", config.get("activation_fn_kwargs", {}).get("k", min(32, d_sae)))
    if config.get("reshape_activations", "none") != "none" or config.get("is_transcoder", False):
        raise ValueError("head reshaping and cross-space transcoders are not same-hook dictionary imports")
    if config.get("d_out", d_in) != d_in:
        raise ValueError("source decoder does not reconstruct the same activation space")
    native = SAEConfig(architecture=architecture, d_in=d_in, d_sae=d_sae, k=k,
        normalize_decoder=False, input_normalization=normalization,
        normalization_std_correction=1 if source_format in {"prisma", "saelens"} else 0,
        apply_decoder_bias_to_input=config.get("apply_b_dec_to_input", True),
        use_decoder_bias=True, rescale_acts_by_decoder_norm=config.get("rescale_acts_by_decoder_norm", False))
    dtype = _tensor(weights, "W_enc", (d_in, d_sae)).dtype
    model = SparseAutoencoder(native, binding, dtype=dtype)
    with torch.no_grad():
        model.encoder.weight.copy_(weights["W_enc"].T)
        model.decoder.weight.copy_(_tensor(weights, "W_dec", (d_sae, d_in)).T)
        model.decoder.bias.copy_(_tensor(weights, "b_dec", (d_in,)))
        if architecture == "gated":
            model.encoder.bias.copy_(_tensor(weights, "b_gate", (d_sae,)))
            model.magnitude_bias.copy_(_tensor(weights, "b_mag", (d_sae,)))
            model.log_magnitude_scale.copy_(_tensor(weights, "r_mag", (d_sae,)))
        else:
            model.encoder.bias.copy_(_tensor(weights, "b_enc", (d_sae,)))
        if architecture == "jumprelu":
            threshold = _tensor(weights, "threshold", (d_sae,))
            if (threshold <= 0).any():
                raise ValueError("JumpReLU source thresholds must be positive")
            model.threshold.copy_(threshold)
        model.input_scale.fill_(1.)
    return model


def _saelens(config, weights, binding, *, legacy_config=None):
    config = dict(config)
    if legacy_config is not None and legacy_config != "feature_splitting_v2":
        raise ValueError("unsupported legacy SAELens config variant")
    if "architecture" not in config:
        if legacy_config != "feature_splitting_v2":
            raise ValueError("legacy config lacks an architecture; specify the reviewed legacy_config variant")
        config.update(architecture="standard", normalize_activations="none", apply_b_dec_to_input=True)
    activation = config.get("activation_fn", config.get("activation_fn_str", "relu"))
    # SAELens <6 represented OpenAI's TopK checkpoints as Standard + TopK.
    # Follow its pinned handle_pre_6_0_config migration rather than replacing
    # the sparse activation with ReLU based on the old architecture label.
    if config["architecture"] == "standard" and activation == "topk":
        if type(config.get("activation_fn_kwargs", {}).get("k")) is not int:
            raise ValueError("legacy Standard/TopK checkpoint requires explicit activation_fn_kwargs.k")
        config["architecture"] = "topk"
        config["k"] = config["activation_fn_kwargs"]["k"]
    if type(config.get("normalize_activations")) is bool:
        config["normalize_activations"] = "expected_average_only_in" if config["normalize_activations"] else "none"
    architecture = {"standard": "relu", "gated": "gated", "topk": "topk", "abstopk": "abstopk", "jumprelu": "jumprelu"}.get(config["architecture"])
    if architecture is None:
        raise ValueError("this source architecture is not yet a supported exact native tensor conversion")
    for name in ("scaling_factor", "finetuning_scaling_factor"):
        if name in weights and not torch.equal(weights[name], torch.ones_like(weights[name])):
            raise ValueError("nontrivial finetuning scaling requires an explicit converter and is not silently discarded")
    if activation not in {"relu", "topk"} or (architecture == "gated" and activation != "relu"):
        raise ValueError("unsupported legacy activation function")
    binding = _binding(config, binding, source_format="saelens")
    return _construct(config, binding, weights, architecture=architecture, source_format="saelens")


def _prisma(config, weights, binding):
    if config.get("is_transcoder", False):
        raise ValueError("Prisma transcoders require distinct input/output bindings")
    architecture = config.get("architecture")
    if architecture == "gated":
        if config.get("activation_fn_str") != "relu":
            raise ValueError("gated TopK magnitude paths are a distinct source variant and are unsupported")
        architecture = "gated"
    elif architecture in {"standard", "vanilla"}:
        architecture = {"relu": "relu", "topk": "topk"}.get(config.get("activation_fn_str"))
    else:
        architecture = None
    if architecture is None:
        raise ValueError("unsupported Prisma architecture/activation combination")
    binding = _binding(config, binding, source_format="prisma")
    return _construct(config, binding, weights, architecture=architecture, source_format="prisma")


def _overcomplete(envelope, binding):
    """Exact inference conversion of linear v3 bridge artifacts.

    Nonlinear/normalizing encoders and constrained archetypal factors are not
    approximated by a linear layer. Published pretrained SAEs are not implied.
    """
    from ..models import SAEConfig, SparseAutoencoder, validate_sae_binding
    if envelope.get("schema") != "vlmprobe-overcomplete-sae-v1":
        raise ValueError("Overcomplete import expects the reviewed safe v3 bridge envelope")
    if validate_sae_binding(envelope.get("binding")) != binding:
        raise ValueError("source artifact binding differs")
    cfg = envelope["config"]
    architecture = cfg["architecture"]
    if architecture not in {"relu", "topk", "jumprelu"}:
        raise ValueError("this Overcomplete source variant needs a separate exact threshold/dictionary converter")
    state = envelope["state_dict"]
    d_in, d_sae = cfg["d_in"], cfg["d_sae"]
    weight = _tensor(state, "sae.encoder.final_block.0.weight", (d_sae, d_in))
    bias = _tensor(state, "sae.encoder.final_block.0.bias", (d_sae,))
    encoder_keys = {key for key in state if key.startswith("sae.encoder.")}
    if encoder_keys != {"sae.encoder.final_block.0.weight", "sae.encoder.final_block.0.bias"}:
        raise ValueError("normalizing or nonlinear source encoders cannot be imported as a linear encoder")
    dictionary = envelope.get("auxiliary_tensors", {}).get("fused_dictionary")
    if not isinstance(dictionary, torch.Tensor) or tuple(dictionary.shape) != (d_sae, d_in):
        raise ValueError("exact source eval dictionary must be materialized as fused_dictionary")
    if not torch.isfinite(dictionary).all():
        raise ValueError("source eval dictionary must be finite")
    mean, scale = state["input_mean"], state["input_scale"]
    if any(value.numel() not in {1, d_in} for value in (mean, scale)) or not torch.isfinite(mean).all() or not torch.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError("source affine normalization must be finite scalars or coordinate vectors with positive scale")
    k = cfg.get("kwargs", {}).get("top_k", max(d_sae // 10, 1))
    model = SparseAutoencoder(SAEConfig(architecture=architecture, d_in=d_in, d_sae=d_sae,
        k=k, normalize_decoder=False, apply_decoder_bias_to_input=False, input_normalization="none"), binding, dtype=weight.dtype)
    with torch.no_grad():
        model.encoder.weight.copy_(weight / scale)
        model.encoder.bias.copy_(bias - weight @ (mean / scale).expand(d_in))
        model.decoder.weight.copy_(dictionary.T * scale.reshape(-1, 1) if scale.numel() > 1 else dictionary.T * scale)
        model.decoder.bias.copy_(mean.expand(d_in))
        if architecture == "jumprelu":
            model.threshold.copy_(_tensor(state, "sae.thresholds", (d_sae,)).exp())
    return model


def import_checkpoint(*, source_format, path, binding=None, config_path=None,
                      output_dir=None, source_metadata=None, model_card=None,
                      device="cpu", legacy_config=None):
    """Convert supported source tensor files once into the native SAE class.

    This function imports weights, not a model implementation. ``binding`` is
    required for external files and declares separately verified activation
    coordinates. Imported artifacts never claim automatic backbone equivalence.
    """
    from ..artifacts import file_sha256, load_artifact, save_artifact
    from ..models import validate_sae_binding
    path = Path(path)
    if source_format == "native_v1":
        model = load_artifact(path, expected_binding=binding, device=device)
    else:
        binding = validate_sae_binding(binding)
        weights = _weights(path, source_format=source_format)
        if source_format == "overcomplete":
            model = _overcomplete(weights, binding)
        else:
            if config_path is None:
                config_path = path.with_name("cfg.json" if source_format == "saelens" else "config.json")
            config_path = Path(config_path)
            config = json.loads(config_path.read_text())
            if source_format == "prisma":
                _validate_prisma_envelope_config(weights, config)
            if "state_dict" in weights:
                # Tensor envelope; Prisma config was decoded into an inert container above.
                if (not isinstance(weights["state_dict"], dict)
                        or not all(isinstance(value, torch.Tensor) for value in weights["state_dict"].values())):
                    raise ValueError("source state_dict must contain only tensors")
                weights = weights["state_dict"]
            if source_format == "saelens":
                model = _saelens(config, weights, binding, legacy_config=legacy_config)
            elif source_format == "prisma":
                model = _prisma(config, weights, binding)
            else:
                raise ValueError("source_format must be saelens, prisma, overcomplete or native_v1")
            model.source_metadata = {"source_config_sha256": file_sha256(config_path)}
        model.is_fitted = True
        model.source_metadata = {**getattr(model, "source_metadata", {}), **(source_metadata or {}),
            "conversion": "direct_tensor_mapping_to_native_sae", "source_format": source_format,
            "source_checkpoint_sha256": file_sha256(path), "feature_order": "preserved",
            "backbone_compatibility": "not_verified_by_weight_conversion"}
        model.to(device)
        model.eval()
    if output_dir is not None:
        save_artifact(model, output_dir, model_card=model_card)
    return model

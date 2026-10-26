"""External tensor formats become the same native class without source SDKs."""
import json
import os
import sys
import types
from pathlib import Path

import pytest
import torch

from vlm_probing.sae.importers import import_checkpoint
from vlm_probing.sae.artifacts import load_artifact
from vlm_probing.sae.models import SparseAutoencoder


def _save_legacy_prisma_envelope(path, config, weights, monkeypatch):
    """Write the historical class name without installing the source package."""
    with monkeypatch.context() as patch:
        for name in ("vit_prisma", "vit_prisma.sae", "vit_prisma.sae.config"):
            patch.setitem(sys.modules, name, types.ModuleType(name))
        cls = type("VisionModelSAERunnerConfig", (), {
            "__module__": "vit_prisma.sae.config",
        })
        sys.modules["vit_prisma.sae.config"].VisionModelSAERunnerConfig = cls
        instance = cls()
        instance.__dict__.update(config)
        torch.save({"cfg": instance, "state_dict": weights}, path)


def test_prisma_legacy_config_envelope_loads_without_official_import(tmp_path, monkeypatch):
    import builtins
    config = {"architecture": "standard", "activation_fn_str": "topk",
        "activation_fn_kwargs": {"k": 2}, "normalize_activations": "layer_norm",
        "d_in": 4, "d_sae": 7, "model_name": "source-model", "hook_point_layer": 0,
        "layer_subtype": "hook_resid_post", "cls_token_only": False, "use_patches_only": True}
    weights = source_weights()
    source = tmp_path / "weights.pt"
    _save_legacy_prisma_envelope(source, config, weights, monkeypatch)
    (tmp_path / "config.json").write_text(json.dumps(config))
    original_import = builtins.__import__
    def blocked_import(name, *args, **kwargs):
        if name.startswith("vit_prisma"):
            raise AssertionError("the official package must not be imported")
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", blocked_import)
    native = import_checkpoint(source_format="prisma", path=source, binding=binding())
    raw = torch.randn(5, 4, dtype=torch.float64)
    expected_features, expected_reconstruction = reference(raw, weights, norm="layer_norm", activation="topk")
    actual_reconstruction, actual_features = native(raw)
    torch.testing.assert_close(actual_features, expected_features)
    torch.testing.assert_close(actual_reconstruction, expected_reconstruction)


def test_prisma_legacy_config_envelope_rejects_conflicting_sidecar(tmp_path, monkeypatch):
    config = {"architecture": "standard", "activation_fn_str": "relu",
        "d_in": 4, "d_sae": 7, "model_name": "source-model", "hook_point_layer": 0,
        "layer_subtype": "hook_resid_post", "normalize_activations": None}
    source = tmp_path / "weights.pt"
    _save_legacy_prisma_envelope(source, config, source_weights(), monkeypatch)
    (tmp_path / "config.json").write_text(json.dumps({**config, "normalize_activations": "layer_norm"}))
    with pytest.raises(ValueError, match="embedded configuration.*normalize_activations"):
        import_checkpoint(source_format="prisma", path=source, binding=binding())


@pytest.mark.parametrize("change", ["omit_normalization", "add_bias_flag", "change_dimensions"])
def test_prisma_legacy_config_rejects_incomplete_or_changed_sidecar(tmp_path, monkeypatch, change):
    config = {"architecture": "standard", "activation_fn_str": "relu",
        "d_in": 4, "d_sae": 7, "model_name": "source-model", "hook_point_layer": 0,
        "layer_subtype": "hook_resid_post", "normalize_activations": "layer_norm"}
    source = tmp_path / "weights.pt"
    _save_legacy_prisma_envelope(source, config, source_weights(), monkeypatch)
    sidecar = dict(config)
    if change == "omit_normalization":
        sidecar.pop("normalize_activations")
    elif change == "add_bias_flag":
        sidecar["apply_b_dec_to_input"] = False
    else:
        sidecar["d_sae"] = 8
    (tmp_path / "config.json").write_text(json.dumps(sidecar))
    with pytest.raises(ValueError, match="embedded configuration differs"):
        import_checkpoint(source_format="prisma", path=source, binding=binding())


def test_prisma_legacy_config_does_not_allow_other_pickled_classes(tmp_path):
    import pickle
    from pathlib import PosixPath
    source = tmp_path / "weights.pt"
    torch.save({"cfg": PosixPath("unapproved-object"), "state_dict": source_weights()}, source)
    with pytest.raises(pickle.UnpicklingError, match="Unsupported global"):
        import_checkpoint(source_format="prisma", path=source, binding=binding())


def test_prisma_legacy_config_refuses_executable_pickle_reducer(tmp_path):
    import pickle
    marker = tmp_path / "must-not-exist"

    class ExecutableReducer:
        def __reduce__(self):
            expression = f"__import__('pathlib').Path({str(marker)!r}).write_text('executed')"
            return eval, (expression,)

    source = tmp_path / "weights.pt"
    torch.save({"cfg": ExecutableReducer(), "state_dict": source_weights()}, source)
    with pytest.raises(pickle.UnpicklingError, match="Unsupported global"):
        import_checkpoint(source_format="prisma", path=source, binding=binding())
    assert not marker.exists()


@pytest.mark.parametrize("version,missing_api", [("2.2.2", True), ("2.5.1", False), ("2.6.0", True)])
def test_prisma_legacy_config_reports_required_torch_loader_version(tmp_path, monkeypatch, version, missing_api):
    config = {"architecture": "standard", "activation_fn_str": "relu",
        "d_in": 4, "d_sae": 7, "model_name": "source-model", "hook_point_layer": 0,
        "layer_subtype": "hook_resid_post", "normalize_activations": None}
    source = tmp_path / "weights.pt"
    _save_legacy_prisma_envelope(source, config, source_weights(), monkeypatch)
    monkeypatch.setattr(torch, "__version__", version)
    if missing_api:
        monkeypatch.delattr(torch.serialization, "safe_globals", raising=False)
    with pytest.raises(RuntimeError, match=r"requires PyTorch >=2\.6.*vlmprobe\[sae\]"):
        import_checkpoint(source_format="prisma", path=source, binding=binding())


def test_prisma_tensor_only_checkpoint_does_not_require_legacy_loader_api(tmp_path, monkeypatch):
    weights = source_weights()
    source = tmp_path / "weights.pt"
    torch.save(weights, source)
    monkeypatch.setattr(torch, "__version__", "2.2.2")
    monkeypatch.delattr(torch.serialization, "safe_globals", raising=False)
    from vlm_probing.sae.importers import _weights
    actual = _weights(source, source_format="prisma")
    for key, value in weights.items():
        torch.testing.assert_close(actual[key], value, rtol=0, atol=0)


def binding(model="source-model", hook="blocks.0.hook_resid_post", layer=0):
    return {"model_id": model + "@source-revision", "tokenizer_id": "source-tokenizer@revision",
        "preprocessing_id": "verified-native-transform@v1", "calibration_id": "author-training-data@revision",
        "site": hook, "layer": layer, "backend_model_name": model, "backend_hook_name": hook,
        "hook": {"module": hook, "kind": "output", "selector": None, "tensor_slice": None}}


def source_weights(d=4, f=7, gated=False):
    generator = torch.Generator().manual_seed(27)
    values = {"W_enc": torch.randn(d, f, generator=generator, dtype=torch.float64),
              "W_dec": torch.randn(f, d, generator=generator, dtype=torch.float64),
              "b_dec": torch.randn(d, generator=generator, dtype=torch.float64)}
    if gated:
        for key in ("b_gate", "b_mag", "r_mag"):
            values[key] = torch.randn(f, generator=generator, dtype=torch.float64) * .2
    else:
        values["b_enc"] = torch.randn(f, generator=generator, dtype=torch.float64)
    return values


def reference(raw, weights, *, norm=None, activation="relu", k=2, apply_bias=True):
    context = None
    if norm == "layer_norm":
        mean = raw.mean(-1, keepdim=True)
        std = (raw - mean).std(-1, keepdim=True)
        raw = (raw - mean) / (std + 1e-5)
        context = (mean, std)
    elif norm == "constant_norm_rescale":
        coefficient = raw.shape[-1] ** .5 / raw.norm(dim=-1, keepdim=True)
        raw = raw * coefficient
        context = coefficient
    centered = raw - weights["b_dec"] if apply_bias else raw
    if activation == "gated":
        gate = centered @ weights["W_enc"] + weights["b_gate"]
        features = (gate > 0) * (centered @ (weights["W_enc"] * weights["r_mag"].exp()) + weights["b_mag"]).relu()
    else:
        pre = centered @ weights["W_enc"] + weights["b_enc"]
        if activation in {"topk", "abstopk"}:
            rank = pre.abs() if activation == "abstopk" else pre
            indices = rank.topk(k, sorted=False).indices
            values = pre.gather(-1, indices)
            features = torch.zeros_like(pre).scatter(-1, indices, values if activation == "abstopk" else values.relu())
        elif activation == "jumprelu":
            features = pre * (pre > weights["threshold"])
        else:
            features = pre.relu()
    output = features @ weights["W_dec"] + weights["b_dec"]
    if norm == "layer_norm":
        output = output * context[1] + context[0]
    elif norm == "constant_norm_rescale":
        output = output / context
    return features, output


@pytest.mark.parametrize("activation", ["relu", "topk", "gated"])
@pytest.mark.parametrize("normalization", [None, "layer_norm", "constant_norm_rescale"])
def test_prisma_conversion_exact_nonsquare_tensors_and_context(activation, normalization, tmp_path):
    weights = source_weights(gated=activation == "gated")
    config = {"model_name": "source-model", "hook_point_layer": 0, "layer_subtype": "hook_resid_post",
        "d_in": 4, "d_sae": 7, "architecture": "gated" if activation == "gated" else "standard",
        "activation_fn_str": "relu" if activation == "gated" else activation,
        "activation_fn_kwargs": {"k": 2}, "normalize_activations": normalization}
    path, config_path = tmp_path / "weights.pt", tmp_path / "config.json"
    torch.save(weights, path)
    config_path.write_text(json.dumps(config))
    model = import_checkpoint(source_format="prisma", path=path, config_path=config_path,
        binding=binding(), output_dir=tmp_path / "canonical")
    assert type(model) is SparseAutoencoder
    raw = torch.randn(3, 4, dtype=torch.float64)
    expected_features, expected_output = reference(raw, weights, norm=normalization, activation=activation)
    actual_features, context = model.encode_with_context(raw)
    torch.testing.assert_close(actual_features, expected_features, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(model.decode_with_context(actual_features, context), expected_output, rtol=1e-12, atol=1e-12)
    restored = load_artifact(tmp_path / "canonical", expected_binding=binding())
    assert torch.equal(restored.encoder.weight, weights["W_enc"].T)
    assert model.source_metadata["feature_order"] == "preserved"


@pytest.mark.parametrize("architecture", ["standard", "topk", "abstopk", "gated", "jumprelu"])
def test_saelens_conversion_preserves_bias_and_signed_features(architecture, tmp_path):
    weights = source_weights(gated=architecture == "gated")
    if architecture == "jumprelu":
        weights["threshold"] = torch.linspace(.1, .3, 7, dtype=torch.float64)
    config = {"architecture": architecture, "d_in": 4, "d_sae": 7, "k": 2,
        "normalize_activations": "none", "apply_b_dec_to_input": False,
        "metadata": {"model_name": "source-model", "hook_name": "blocks.0.hook_resid_post", "hook_layer": 0}}
    path, cfg = tmp_path / "weights.pt", tmp_path / "cfg.json"
    torch.save(weights, path); cfg.write_text(json.dumps(config))
    model = import_checkpoint(source_format="saelens", path=path, config_path=cfg, binding=binding())
    raw = torch.randn(5, 4, dtype=torch.float64)
    expected_features, expected_output = reference(raw, weights,
        activation="relu" if architecture == "standard" else architecture, apply_bias=False)
    actual_features = model.encode(raw)
    torch.testing.assert_close(actual_features, expected_features, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(model.decode(actual_features), expected_output, rtol=1e-12, atol=1e-12)


def test_import_rejects_silent_approximations(tmp_path):
    weights = source_weights()
    path, cfg = tmp_path / "weights.pt", tmp_path / "cfg.json"
    torch.save(weights, path)
    config = {"architecture": "standard", "d_in": 4, "d_sae": 7, "normalize_activations": "covariance_whitening"}
    cfg.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="normalization"):
        import_checkpoint(source_format="saelens", path=path, config_path=cfg, binding=binding())

    config.update(normalize_activations="none")
    cfg.write_text(json.dumps(config))
    weights["scaling_factor"] = torch.full((7,), 2., dtype=torch.float64)
    torch.save(weights, path)
    with pytest.raises(ValueError, match="scaling"):
        import_checkpoint(source_format="saelens", path=path, config_path=cfg, binding=binding())


@pytest.mark.parametrize("normalization", ["none", "layer_norm", "constant_norm_rescale"])
@pytest.mark.parametrize("activation_field", ["activation_fn_str", "activation_fn"])
def test_saelens_legacy_topk_normalization_matches_source_equations(normalization, activation_field, tmp_path):
    weights = source_weights()
    config = {"architecture": "standard", activation_field: "topk",
        "activation_fn_kwargs": {"k": 2}, "d_in": 4, "d_sae": 7,
        "model_name": "source-model", "hook_name": "blocks.0.hook_resid_post", "hook_layer": 0,
        "normalize_activations": normalization, "apply_b_dec_to_input": True}
    path, cfg = tmp_path / "weights.pt", tmp_path / "cfg.json"
    torch.save(weights, path)
    cfg.write_text(json.dumps(config))
    model = import_checkpoint(source_format="saelens", path=path, config_path=cfg, binding=binding())
    assert model.config.architecture == "topk" and model.config.k == 2
    raw = torch.randn(5, 4, dtype=torch.float64) * 3 + 7
    expected_features, expected_output = reference(raw, weights, norm=normalization, activation="topk")
    actual_features, context = model.encode_with_context(raw)
    torch.testing.assert_close(actual_features, expected_features, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(model.decode_with_context(actual_features, context), expected_output, rtol=1e-12, atol=1e-12)
    assert (actual_features != 0).sum(-1).max() <= 2


def test_overcomplete_linear_affine_conversion(tmp_path):
    d, f = 4, 7
    weight, bias, dictionary = torch.randn(f, d), torch.randn(f), torch.randn(f, d)
    mean, scale = torch.randn(d), torch.rand(d) + .5
    envelope = {"schema": "vlmprobe-overcomplete-sae-v1", "binding": binding(),
        "config": {"architecture": "relu", "d_in": d, "d_sae": f, "kwargs": {"encoder_module": "linear"}},
        "state_dict": {"sae.encoder.final_block.0.weight": weight, "sae.encoder.final_block.0.bias": bias,
                       "input_mean": mean, "input_scale": scale},
        "auxiliary_tensors": {"fused_dictionary": dictionary}}
    path = tmp_path / "old.pt"; torch.save(envelope, path)
    model = import_checkpoint(source_format="overcomplete", path=path, binding=binding())
    raw = torch.randn(3, d)
    expected = torch.nn.functional.linear((raw - mean) / scale, weight, bias).relu()
    torch.testing.assert_close(model.encode(raw), expected)
    torch.testing.assert_close(model.decode(model.encode(raw)), expected @ dictionary * scale + mean)


@pytest.mark.skipif(os.getenv("VLMPROBE_RUN_PRETRAINED_IMPORT_TESTS") != "1", reason="opt-in real checkpoint conversion; weights are never bundled")
@pytest.mark.parametrize("source_format", ["saelens", "prisma"])
def test_real_author_checkpoint_conversion_parity(source_format, tmp_path):
    if source_format == "prisma":
        path = Path("/tmp/vlmprobe-prisma-weights/weights.pt")
        cfg = path.with_name("config.json")
        source_binding = binding("open-clip:laion/CLIP-ViT-B-32-DataComp.XL-s13B-b90K", "blocks.11.hook_resid_post", 11)
        weights = torch.load(path, map_location="cpu", weights_only=True)
        kwargs = {}
    else:
        parent = Path("/tmp/vlmprobe-saelens-v2-hf/hub/models--jbloom--GPT2-Small-Feature-Splitting-Experiment-Layer-8/snapshots/174d298f1633aab71e440685c3292ce8108a5a73/blocks.8.hook_resid_pre_768")
        path, cfg = parent / "sae_weights.safetensors", parent / "cfg.json"
        source_binding = binding("gpt2-small", "blocks.8.hook_resid_pre", 8)
        from safetensors.torch import load_file
        weights = load_file(str(path))
        kwargs = {"legacy_config": "feature_splitting_v2"}
    if not path.exists():
        pytest.skip("author checkpoint is not available in the local verification cache")
    model = import_checkpoint(source_format=source_format, path=path, config_path=cfg,
        binding=source_binding, output_dir=tmp_path / "converted", **kwargs)
    raw = torch.randn(3, 768, generator=torch.Generator().manual_seed(42))
    features, output = reference(raw, weights)
    torch.testing.assert_close(model.encode(raw), features, rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(model.decode(model.encode(raw)), output, rtol=2e-5, atol=2e-5)
    assert type(load_artifact(tmp_path / "converted")) is SparseAutoencoder

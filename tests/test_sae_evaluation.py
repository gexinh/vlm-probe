"""Evaluation invariants: global denominators, frozen state and held-out inputs."""
from pathlib import Path
from types import SimpleNamespace
import copy
import json
import sys
import tempfile

import pytest
import torch
from torch import nn

from vlmprobe import evaluation, Prober, TokenMargin
from vlmprobe.sae import SAEConfig, SparseAutoencoder
from vlmprobe.sae.binding import build_sae_binding
from vlmprobe.sae.training import train_sae

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
from tiny_model import TinyModel

E = evaluation.sae


class SignedIdentity(nn.Module):
    def __init__(self, scale=.5):
        super().__init__()
        self.register_buffer("counter", torch.tensor(0))
        self.scale = scale
        self.binding = {"model_id": "fixed", "hook": "resid", "tokens": "all"}
        self.config = SimpleNamespace(d_sae=2)

    def encode(self, values):
        return values

    def decode(self, values):
        return values * self.scale


def test_global_metrics_partition_invariance_and_signed_activity():
    x = torch.tensor([[-5., 0.], [-1., 2.], [3., -2.], [9., 0.]], dtype=torch.float64)
    sae = SignedIdentity()
    first = E.evaluate_activations(sae, [E.ActivationBatch(x, ids=["a", "b", "c", "d"])])
    second = E.evaluate_activations(sae, [E.ActivationBatch(x[:1], ["a"]),
        E.ActivationBatch(x[1:], ["b", "c", "d"])], config=E.SAEEvaluationConfig(chunk_size=1))
    expected = (x - x * .5).square().sum() / (x - x.mean(0)).square().sum()
    assert first.summary["fvu"] == pytest.approx(expected.item())
    assert first.summary["fvu"] == pytest.approx(second.summary["fvu"], abs=1e-12)
    assert first.summary["l0"] == 1.5
    assert first.provenance["selected_position_ids_sha256"] == second.provenance["selected_position_ids_sha256"]
    assert first.provenance["native_activation_values_sha256"] == second.provenance["native_activation_values_sha256"]
    torch.testing.assert_close(first.feature_statistics["frequency"], torch.tensor([1., .5], dtype=torch.float64))
    torch.testing.assert_close(first.feature_statistics["variance"], x.var(0, correction=0))


def test_reference_mean_and_constant_denominators():
    x = torch.ones(4, 2)
    report = E.evaluate_activations(SignedIdentity(), [x])
    assert report.summary["fvu"] is None
    assert report.summary["fvu_defined"] is False
    report = E.evaluate_activations(SignedIdentity(), [x], reference_mean=torch.zeros(2))
    assert report.summary["fvu"] == pytest.approx(.25)
    with pytest.raises(ValueError, match="reference_mean"):
        E.evaluate_activations(SignedIdentity(), [x], reference_mean=torch.zeros(3))
    with pytest.raises(ValueError, match="at least one"):
        E.evaluate_activations(SignedIdentity(), [])


def test_masks_and_groups_preserve_original_example_rows():
    x = torch.tensor([[[1., 0.], [2., 0.]], [[3., 0.], [4., 0.]]])
    mask = torch.tensor([[True, False], [True, True]])
    report = E.evaluate_activations(SignedIdentity(), [E.ActivationBatch(x,
        ids=["image1", "image2"], group_ids=["patient1", "patient2"], mask=mask)])
    assert report.summary["n_examples"] == 2
    assert report.summary["n_positions"] == 3
    assert [row["n_positions"] for row in report.per_example] == [1, 2]
    assert report.per_example[1]["group_id"] == "patient2"
    with pytest.raises(ValueError, match="different source"):
        E.evaluate_activations(SignedIdentity(), [E.ActivationBatch(x, ["same", "same"], ["a", "b"])])


def test_frozen_buffers_and_training_flags_are_restored():
    sae = SignedIdentity().train()
    E.evaluate_activations(sae, [torch.ones(2, 2)])
    assert sae.training
    assert int(sae.counter) == 0

    class Mutating(SignedIdentity):
        def encode(self, values):
            self.counter.add_(1)
            return values

    sae = Mutating()
    with pytest.raises(RuntimeError, match="changed SAE buffers"):
        E.evaluate_activations(sae, [torch.ones(2, 2)])
    assert int(sae.counter) == 0


def test_compare_rejects_consumed_generators_and_unmatched_inputs():
    methods = {"a": SignedIdentity(), "b": SignedIdentity(.7)}
    with pytest.raises(ValueError, match="replayable"):
        E.compare_activations(methods, iter([torch.ones(2, 2)]))
    reports = E.compare_activations(methods, lambda: iter([E.ActivationBatch(torch.ones(2, 2), ["a", "b"])]))
    assert set(reports) == {"a", "b"}
    methods["b"].binding["hook"] = "different"
    with pytest.raises(ValueError, match="same backbone"):
        E.compare_activations(methods, [torch.ones(2, 2)])
    methods["b"].binding["hook"] = "resid"
    count = 0
    def factory():
        nonlocal count
        count += 1
        return [E.ActivationBatch(torch.ones(2, 2), [str(count), "b"])]
    with pytest.raises(ValueError, match="changed example"):
        E.compare_activations(methods, factory)
    value_calls = 0
    def changed_values():
        nonlocal value_calls
        value_calls += 1
        return [E.ActivationBatch(torch.ones(2, 2) * value_calls, ["a", "b"])]
    with pytest.raises(ValueError, match="changed example"):
        E.compare_activations(methods, changed_values)


def test_report_strict_json_and_csv(tmp_path):
    report = E.evaluate_activations(SignedIdentity(), [torch.ones(2, 2)])
    files = report.save(tmp_path)
    assert set(files) == {"summary", "per_example", "feature_statistics"}
    value = json.loads(files["summary"].read_text())
    assert value["summary"]["fvu"] is None
    assert "feature_id" in files["feature_statistics"].read_text()


def test_real_native_model_fit_load_evaluate_and_intervene(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(41)
    model = TinyModel()
    probe = Prober(model)
    binding = build_sae_binding(probe, 0, model_id="tiny-fixed", tokenizer_id="integer-v1",
        preprocessing_id="synthetic-v1", calibration_id="test-fixture-only")
    sae = SparseAutoencoder(SAEConfig(architecture="relu", d_in=6, d_sae=12, k=2), binding=binding)
    train_sae(sae, [torch.randn(8, 6)], [torch.randn(8, 6)], tmp_path / "fit",
              epochs=1, patience=1, lr=.001, progress_every=20)
    checkpoint = tmp_path / "sae.pt"
    sae.save(checkpoint)
    method = probe.sae.features(layer=0, checkpoint=checkpoint, binding=binding)
    inputs = {"input_ids": torch.tensor([[1, 2, 3], [4, 5, 6]]), "image_tokens": torch.randn(2, 2, 6)}
    before = copy.deepcopy(method.sae.state_dict())
    model_before = copy.deepcopy(model.state_dict())
    flags_before = [module.training for module in model.modules()]
    config = E.SAEEvaluationConfig(metrics=("reconstruction", "sparsity", "fidelity", "intervention"),
        feature_ids=(0,), intervention_scales=(0., 1.), random_control_repeats=1)
    report = E.evaluate(method, [E.EvaluationBatch(inputs, ["one", "two"], metric=TokenMargin(4, 5))], config=config)
    assert report.summary["n_examples"] == 2
    assert report.summary["no_op.score_change"] == 0
    assert "reconstruction.score_change" in report.summary
    assert "random_0_scale_0.score_change" in report.summary
    report.save(tmp_path / "evaluation")
    for key, value in method.sae.state_dict().items():
        assert torch.equal(value, before[key])
    for key, value in model.state_dict().items():
        assert torch.equal(value, model_before[key])
    assert [module.training for module in model.modules()] == flags_before
    expanded = torch.cat([torch.zeros(2, 2, dtype=torch.long), inputs["input_ids"]], dim=1)
    prediction_mask = torch.tensor([[False, True, True, True], [False, True, True, True]])
    fidelity = E.language_fidelity(method, [E.LanguageFidelityBatch(inputs, ["one", "two"], expanded, prediction_mask)])
    assert fidelity.summary["n_tokens"] == 6
    assert fidelity.summary["delta_nll"] == pytest.approx(
        fidelity.summary["reconstruction_nll"] - fidelity.summary["baseline_nll"])
    assert fidelity.summary["kl_baseline_to_reconstruction"] >= -1e-7


def test_model_metrics_keep_native_precision_for_lower_precision_dictionary():
    torch.manual_seed(73)
    probe = Prober(TinyModel())
    binding = build_sae_binding(probe, 0, model_id="fixed-precision-fixture",
        tokenizer_id="integers", preprocessing_id="native", calibration_id="fixed-weights")
    dictionary = SparseAutoencoder(SAEConfig(architecture="relu", d_in=6, d_sae=12,
        k=2, input_normalization="none"), binding=binding).bfloat16()
    dictionary.is_fitted = True  # Fixed test weights; no report training.
    method = probe.sae.features(layer=0, sae=dictionary, binding=binding)
    inputs = {"input_ids": torch.tensor([[1, 2, 3]]), "image_tokens": torch.randn(1, 2, 6)}
    result = method.run(inputs)
    assert result.tensors["native_activation"].dtype == torch.float32
    assert result.tensors["activation"].dtype == torch.bfloat16
    expected = (result.tensors["native_activation"].double()
                - result.tensors["reconstruction"].double()).square().mean().item()
    report = E.evaluate(method, [E.EvaluationBatch(inputs, ["image"])],
                        config=E.SAEEvaluationConfig(chunk_size=1))
    assert report.summary["mse"] == pytest.approx(expected, abs=1e-12)
    # Comparison authenticates the actual backbone observations, independent
    # of each dictionary's inference precision.
    full_precision = copy.deepcopy(dictionary).float()
    second = probe.sae.features(layer=0, sae=full_precision, binding=binding)
    reports = E.compare({"bf16": method, "fp32": second},
                        [E.EvaluationBatch(inputs, ["image"])])
    assert len(reports) == 2
    with pytest.raises(ValueError, match="unique example IDs"):
        E.evaluate(method, [E.EvaluationBatch(inputs, ["image"])] * 2)


def test_sparse_probe_requires_each_class_in_validation_and_test():
    torch.manual_seed(17)
    x = torch.randn(8, 4)
    with pytest.raises(ValueError, match="validation must contain every class"):
        E.sparse_probe(x, [0, 1] * 4, x, [0] * 8, x, [0, 1] * 4)


def test_calibration_random_controls_are_disjoint_across_repeats():
    controls = E.matched_random_features({"frequency": torch.ones(12)}, [0, 1], repeats=3, seed=71)
    flattened = [feature for group in controls for feature in group]
    assert len(flattened) == len(set(flattened)) == 6
    assert not set(flattened).intersection({0, 1})
    assert controls == E.matched_random_features({"frequency": torch.ones(12)}, [0, 1], repeats=3, seed=71)
    with pytest.raises(ValueError, match="not enough distinct"):
        E.matched_random_features({"frequency": torch.ones(3)}, [0], repeats=3)


def test_explicit_random_controls_reject_partial_overlap_before_model_execution():
    dictionary = SignedIdentity()
    dictionary.config.d_sae = 6
    method = SimpleNamespace(sae=dictionary)
    config = E.SAEEvaluationConfig(metrics=("intervention",), feature_ids=(0, 1),
        random_feature_ids=((2, 3), (3, 4)))
    # Different control sets sharing a feature still violate independent,
    # without-replacement controls, even though neither set repeats exactly.
    with pytest.raises(ValueError, match="disjoint across all control sets"):
        E.evaluate(method, [], config=config)


def test_localization_inactive_zero_maps_and_resizing():
    maps = torch.tensor([[[0., 1.], [0., 0.]], [[0., 0.], [0., 0.]]])
    masks = torch.tensor([[[False, True], [False, False]], [[True, False], [False, False]]])
    report = E.localization(maps, masks)
    assert report.summary["coverage"] == .5
    assert report.summary["point_at_1"] == .5
    assert report.summary["mass_inside_active"] == 1
    assert report.per_example[1]["mass_inside"] is None
    with pytest.raises(ValueError, match="nonempty annotated"):
        E.localization(maps, torch.zeros_like(masks))


def test_sparse_probe_no_leakage_and_pca_train_only():
    pytest.importorskip("sklearn")
    generator = torch.Generator().manual_seed(4)
    train = torch.randn(60, 6, generator=generator)
    validation = torch.randn(30, 6, generator=generator)
    test = torch.randn(30, 6, generator=generator)
    labels = [(value[:, 0] > 0).int().tolist() for value in (train, validation, test)]
    report = E.sparse_probe(train, labels[0], validation, labels[1], test, labels[2], feature_counts=(1, 2), regularization=(.1, 1.))
    assert report.summary["results"][0]["macro_auroc"] > .9
    assert report.provenance["selected_features"]["1"]["0"]["feature_ids"] == [0]
    changed = E.sparse_probe(train, labels[0], validation, labels[1], test * 100, labels[2], feature_counts=(1,), regularization=(.1, 1.))
    assert changed.provenance["selected_features"]["1"]["0"]["feature_ids"] == [0]
    with pytest.raises(ValueError, match="disjoint"):
        E.sparse_probe(train, labels[0], validation, labels[1], test, labels[2],
            train_group_ids=["same"] * 60, validation_group_ids=["same"] * 30, test_group_ids=["other"] * 30)
    pca = E.pca_features(train, validation, test, n_components=2)
    pca_changed = E.pca_features(train, validation * 10, test * 100, n_components=2)
    torch.testing.assert_close(pca[0], pca_changed[0])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA needed for CPU-result/native-GPU regression")
def test_language_fidelity_cpu_results_native_cuda_and_unscored_placeholders():
    torch.manual_seed(12)
    model = TinyModel().cuda()
    probe = Prober(model)  # The public API intentionally exports results to CPU.
    binding = build_sae_binding(probe, 0, model_id="tiny-cuda", tokenizer_id="integer-v1",
        preprocessing_id="synthetic-v1", calibration_id="fixed-fixture")
    sae = SparseAutoencoder(SAEConfig(architecture="relu", d_in=6, d_sae=12, k=2),
                            binding=binding, device="cuda")
    sae.is_fitted = True  # Fixed dictionary fixture; no experiment/model is trained.
    method = probe.sae.features(layer=0, sae=sae, binding=binding)
    inputs = {"input_ids": torch.tensor([[1, 2, 3]], device="cuda"),
              "image_tokens": torch.randn(1, 2, 6, device="cuda")}
    expanded = torch.tensor([[-200, -200, 1, 2, 3]])
    mask = torch.tensor([[False, True, True, True]])
    report = E.language_fidelity(method, [E.LanguageFidelityBatch(inputs, ["one"], expanded, mask)])
    assert report.summary["n_tokens"] == 3
    assert report.summary["kl_baseline_to_reconstruction"] >= -1e-6
    assert isinstance(report.summary["zero_nll"], float)

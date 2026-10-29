"""Sparse feature API contracts on real CPU forwards and trained dictionaries."""
import copy
from pathlib import Path
import sys
import tempfile
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
from tiny_model import TinyModel
from vlmprobe import ProbeInputs, Prober, TokenLayout, TokenMargin
from vlmprobe import visualization as viz
from vlmprobe.sae import SAEConfig, SparseAutoencoder
from vlmprobe.sae.binding import build_sae_binding
from vlmprobe.sae.training import train_sae


class SAEAPITests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(87)
        torch.set_num_threads(1)
        self.model = TinyModel()
        self.probe = Prober(self.model)
        self.inputs = {"input_ids": torch.tensor([[1, 2, 3], [4, 5, 6]]),
                       "image_tokens": torch.randn(2, 2, 6)}
        self.binding = build_sae_binding(
            self.probe, 0, model_id="tiny-seed87-weights-v1", tokenizer_id="integer-ids-v1",
            preprocessing_id="two-visual-then-three-text-v1", calibration_id="synthetic-train-seed87-v1")
        self.config = SAEConfig(architecture="relu", d_in=6, d_sae=12, k=3,
                                l1_coefficient=.01, seed=87)
        self.sae = SparseAutoencoder(self.config, binding=self.binding)
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.train_dir = Path(self.directory.name) / "fit"
        train_sae(self.sae, [torch.randn(16, 6), torch.randn(16, 6)],
                  [torch.randn(16, 6)], self.train_dir, epochs=2, patience=2,
                  lr=.001, seed=87, progress_every=20)
        self.metric = TokenMargin(4, 5)

    def test_feature_values_shapes_and_padding_coordinates(self):
        valid = torch.tensor([[False, True, True, True, False], [True, True, True, True, True]])
        inputs = ProbeInputs(self.inputs, TokenLayout(valid))
        result = self.probe.sae.features(layer=0, sae=self.sae, binding=self.binding).run(inputs)
        raw = self.probe.adapter.run(self.inputs, capture=["residual.0"]).activations["residual.0"][valid]
        encoded = self.sae.encode(raw)
        torch.testing.assert_close(result.tensors["features"], encoded)
        torch.testing.assert_close(result.tensors["activation"], raw)
        torch.testing.assert_close(result.tensors["reconstruction_error"], raw - self.sae.decode(encoded))
        self.assertEqual(result.tensors["positions"].tolist(), valid.nonzero().tolist())
        self.assertEqual(result.tensors["features"].shape, (8, 12))
        self.assertEqual(result.tensors["reconstruction"].shape, (8, 6))
        self.assertEqual(result.tensors["l0"].tolist(), (encoded != 0).sum(-1).tolist())
        self.assertFalse(result.tensors["features"].requires_grad)

    def test_untrained_dictionary_is_rejected_by_bound_method(self):
        fresh = SparseAutoencoder(self.config, binding=self.binding)
        with self.assertRaisesRegex(ValueError, "trained|fit"):
            self.probe.sae.features(layer=0, sae=fresh, binding=self.binding).run(self.inputs)

    def test_actual_hook_binding_is_checked_even_if_expected_and_sae_both_wrong(self):
        wrong = copy.deepcopy(self.binding)
        wrong["hook"]["module"] = "layers.1"
        mislabeled = copy.deepcopy(self.sae)
        mislabeled.binding = wrong
        with self.assertRaisesRegex(ValueError, "hook|runtime"):
            self.probe.sae.features(layer=0, sae=mislabeled, binding=wrong).run(self.inputs)
        with self.assertRaisesRegex(ValueError, "binding|identit"):
            self.probe.sae.features(layer=0, sae=self.sae,
                binding={**self.binding, "calibration_id": "different-training-corpus"}).run(self.inputs)

    def test_one_artifact_layer_and_exactly_one_source_are_required(self):
        with self.assertRaises((ValueError, TypeError)):
            self.probe.sae.features(layer=[0], sae=self.sae, binding=self.binding)
        with self.assertRaisesRegex(ValueError, "exactly one"):
            self.probe.sae.features(layer=0, binding=self.binding)
        with self.assertRaisesRegex(ValueError, "exactly one"):
            self.probe.sae.features(layer=0, sae=self.sae, checkpoint="anything.pt", binding=self.binding)

    def test_fitted_portable_checkpoint_round_trip_and_wrong_identity_rejection(self):
        checkpoint = Path(self.directory.name) / "portable.pt"
        self.sae.save(checkpoint)
        expected = self.probe.sae.features(layer=0, sae=self.sae, binding=self.binding).run(self.inputs)
        loaded = self.probe.sae.features(layer=0, checkpoint=checkpoint, binding=self.binding).run(self.inputs)
        torch.testing.assert_close(loaded.tensors["features"], expected.tensors["features"])
        with self.assertRaisesRegex(ValueError, "binding|identit"):
            self.probe.sae.features(layer=0, checkpoint=checkpoint,
                binding={**self.binding, "model_id": "other-model-weights"}).run(self.inputs)

    def test_model_parameters_grads_and_individual_training_flags_are_restored(self):
        self.model.train()
        self.model.norm.eval()
        flags = [module.training for module in self.model.modules()]
        parameters = {key: value.detach().clone() for key, value in self.model.named_parameters()}
        self.probe.sae.features(layer=0, sae=self.sae, binding=self.binding).run(self.inputs)
        self.probe.sae.edit(layer=0, sae=self.sae, binding=self.binding,
                            feature_ids=[0], scale=0).run(self.inputs, metric=self.metric)
        self.assertEqual([module.training for module in self.model.modules()], flags)
        for key, value in self.model.named_parameters():
            torch.testing.assert_close(value, parameters[key])
            self.assertIsNone(value.grad)

    def test_error_preserving_scale_one_is_bitwise_exact_noop(self):
        result = self.probe.sae.edit(layer=0, sae=self.sae, binding=self.binding,
            feature_ids=[0, 1], scale=1., mode="error_preserving").run(self.inputs, metric=self.metric)
        self.assertTrue(torch.equal(result.tensors["baseline_logits"], result.tensors["edited_logits"]))
        self.assertTrue(torch.equal(result.tensors["score_change"], torch.zeros(2)))

    def test_inactive_feature_ablation_preserves_native_precision_without_fastpath(self):
        self.model.double()
        inputs = {**self.inputs, "image_tokens": self.inputs["image_tokens"].double()}
        sae = copy.deepcopy(self.sae).float()
        with torch.no_grad():
            sae.encoder.weight[0].zero_()
            sae.encoder.bias[0].fill_(-10000.)
        hidden = self.probe.adapter.run(inputs, capture=["residual.0"]).activations["residual.0"]
        self.assertEqual(hidden.dtype, torch.float64)
        self.assertEqual(next(sae.parameters()).dtype, torch.float32)
        self.assertTrue((sae.encode(hidden)[..., 0] == 0).all())
        # Scale zero bypasses the identity fast path. Only adding the exact zero
        # decoded delta to the original float64 residual preserves native bits.
        edited = self.probe.sae.edit(layer=0, sae=sae, binding=self.binding,
            feature_ids=[0], scale=0., mode="error_preserving").run(inputs, metric=self.metric)
        self.assertTrue(torch.equal(edited.tensors["baseline_logits"], edited.tensors["edited_logits"]))
        self.assertTrue(torch.equal(edited.tensors["score_change"], torch.zeros(2, dtype=torch.float64)))

    def test_mixed_precision_ablation_adds_decoded_delta_to_original_native_tensor(self):
        self.model.double()
        inputs = {**self.inputs, "image_tokens": self.inputs["image_tokens"].double()}
        sae = copy.deepcopy(self.sae).float()
        with torch.no_grad():
            sae.encoder.weight[0].zero_()
            sae.encoder.bias[0].fill_(5.)
        raw = self.probe.adapter.run(inputs, capture=["residual.0"]).activations["residual.0"]
        features = sae.encode(raw)
        changed = features.clone()
        changed[..., 0] = 0
        # The dictionary projections remain FP32, but subtract their represented
        # values in FP64 before adding to the original FP64 residual. Rounding
        # the delta first would defeat the wider native precision contract.
        decoded_delta = sae.decode(changed).to(raw) - sae.decode(features).to(raw)
        replacement = raw + decoded_delta.to(raw)

        def manual(hidden):
            return replacement.clone()

        expected = self.probe.adapter.run(inputs, interventions={"residual.0": manual})
        edited = self.probe.sae.edit(layer=0, sae=sae, binding=self.binding,
            feature_ids=[0], scale=0., mode="error_preserving").run(inputs)
        self.assertTrue(torch.equal(edited.tensors["edited_logits"], expected.logits))
        self.assertFalse(torch.equal(edited.tensors["baseline_logits"], edited.tensors["edited_logits"]))

    def test_ablation_and_reconstruction_match_native_manual_hook_separately(self):
        raw = self.probe.adapter.run(self.inputs, capture=["residual.0"]).activations["residual.0"]
        selected = torch.zeros(raw.shape[:2], dtype=torch.bool)
        selected[:, :2] = True
        values = raw[selected]
        features = self.sae.encode(values)
        edited_features = features.clone()
        edited_features[:, [0, 1]] = 0
        for mode in ("error_preserving", "reconstruction"):
            replacement = self.sae.decode(edited_features)
            if mode == "error_preserving":
                replacement = values + (replacement - self.sae.decode(features)).to(values)

            def manual(hidden):
                changed = hidden.clone()
                changed[selected] = replacement
                return changed

            expected = self.probe.adapter.run(self.inputs, interventions={"residual.0": manual})
            actual = self.probe.sae.edit(layer=0, sae=self.sae, binding=self.binding,
                tokens="visual", feature_ids=[0, 1], scale=0., mode=mode).run(self.inputs, metric=self.metric)
            torch.testing.assert_close(actual.tensors["edited_logits"], expected.logits)
            torch.testing.assert_close(actual.tensors["score_change"],
                actual.tensors["edited_score"] - actual.tensors["baseline_score"])

    def test_reconstruction_scale_one_is_reconstruction_control_not_native_noop(self):
        result = self.probe.sae.edit(layer=0, sae=self.sae, binding=self.binding,
            feature_ids=[0], scale=1., mode="reconstruction").run(self.inputs)
        self.assertFalse(torch.equal(result.tensors["baseline_logits"], result.tensors["edited_logits"]))
        self.assertNotIn("baseline_score", result.tensors)

    def test_invalid_ids_scale_and_modes_fail_before_model_edit(self):
        for ids in ([], [-1], [12], [0, 0], [True]):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                self.probe.sae.edit(layer=0, sae=self.sae, binding=self.binding, feature_ids=ids)
        for scale in (float("nan"), float("inf")):
            with self.subTest(scale=scale), self.assertRaises(ValueError):
                self.probe.sae.edit(layer=0, sae=self.sae, binding=self.binding, feature_ids=[0], scale=scale)
        with self.assertRaises(ValueError):
            self.probe.sae.edit(layer=0, sae=self.sae, binding=self.binding, feature_ids=[0], mode="other")

    def test_built_in_visualization_uses_actual_feature_positions(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from PIL import Image
        result = self.probe.sae.features(layer=0, sae=self.sae, binding=self.binding,
                                         tokens="visual").run(self.inputs)
        self.assertTrue(result.metadata["visual_only"])
        ax = viz.plot_sae_activity(result, top_k=4)
        self.assertEqual(len(ax.patches), 4)
        plt.close(ax.figure)
        ax = viz.plot_sae_feature_overlay(Image.new("RGB", (40, 20)), result,
                                          feature_id=0, grid_shape=[1, 2], batch_index=1)
        self.assertEqual(len(ax.images), 2)
        plt.close(ax.figure)
        with self.assertRaisesRegex(ValueError, "grid"):
            viz.plot_sae_feature_overlay(Image.new("RGB", (40, 20)), result,
                                          feature_id=0, grid_shape=[2, 2], batch_index=1)

    def test_nonvisual_overlay_is_rejected_even_with_matching_grid_count(self):
        from PIL import Image
        for tokens, grid in (("text", [1, 3]), ("all", [1, 5])):
            with self.subTest(tokens=tokens):
                result = self.probe.sae.features(layer=0, sae=self.sae, binding=self.binding,
                                                 tokens=tokens).run(self.inputs)
                self.assertFalse(result.metadata["visual_only"])
                with self.assertRaisesRegex(ValueError, "visual token"):
                    viz.plot_sae_feature_overlay(Image.new("RGB", (40, 20)), result,
                        feature_id=0, grid_shape=grid, batch_index=0)


if __name__ == "__main__":
    unittest.main()

"""Native DINO hook geometry, representation output and SAE controls."""
from importlib.metadata import PackageNotFoundError, version
import unittest

import torch

from vlmprobe import Prober, CapabilityError
from vlmprobe.adapters.prisma_vision import make_adapter
from vlmprobe.sae import SAEConfig, SparseAutoencoder
from vlmprobe.sae.binding import build_sae_binding

try:
    HAS_V5 = version("transformers").split(".")[:2] == ["5", "3"]
except PackageNotFoundError:
    HAS_V5 = False


@unittest.skipUnless(HAS_V5, "DINO adapter targets native Transformers 5.3.x")
class DinoSAEAdapterTests(unittest.TestCase):
    def setUp(self):
        from transformers import ViTConfig, ViTModel
        torch.manual_seed(412)
        torch.set_num_threads(1)
        config = ViTConfig(image_size=16, patch_size=8, hidden_size=16,
            num_hidden_layers=2, num_attention_heads=2, intermediate_size=32,
            hidden_dropout_prob=0., attention_probs_dropout_prob=0.)
        config._attn_implementation = "eager"
        self.model = ViTModel(config).eval().requires_grad_(False)
        self.adapter = make_adapter(self.model, source_model_name="facebook/dino-vitb16")
        self.probe = Prober(self.model, adapter=self.adapter)
        self.inputs = {"pixel_values": torch.rand(2, 3, 16, 16)}

    def test_native_output_residual_hook_and_visual_coordinates(self):
        native = self.model(**self.inputs).last_hidden_state
        trace = self.adapter.run(self.inputs, capture=list(self.probe.spec.residuals.values()))
        torch.testing.assert_close(trace.output[:, :, 0], native[:, 0])
        torch.testing.assert_close(trace.output[:, :, 1], native[:, 1:].mean(1))
        self.assertEqual(trace.activations["vision.residual.0"].shape, (2, 5, 16))
        layout = self.probe._layout(self.inputs, trace.activations["vision.residual.0"])
        self.assertTrue(layout.valid.all())
        self.assertFalse(layout.visual[:, 0].any())
        self.assertTrue(layout.visual[:, 1:].all())
        self.assertFalse(self.probe.spec.causal_attention)
        self.assertEqual(self.adapter.visual_grid, (2, 2))
        with self.assertRaisesRegex(TypeError, "representation"):
            _ = trace.logits

    def test_patch_sae_reconstruction_and_exact_noop(self):
        binding = build_sae_binding(self.probe, 0, model_id="fixed-DINO-fixture",
            tokenizer_id="vision-only", preprocessing_id="RGB16", calibration_id="fixed-weights")
        binding["tokens"] = "visual"
        dictionary = SparseAutoencoder(SAEConfig("relu", 16, 32,
            input_normalization="none", normalize_decoder=False), binding)
        dictionary.is_fitted = True
        method = self.probe.sae.features(layer=0, sae=dictionary, binding=binding, tokens="visual")
        result = method.run(self.inputs)
        self.assertEqual(result.tensors["features"].shape, (8, 32))
        self.assertTrue(result.metadata["visual_only"])
        baseline = self.adapter.run(self.inputs).output.detach()
        metric = lambda output: torch.nn.functional.cosine_similarity(
            output.flatten(1), baseline.flatten(1), dim=-1)
        feature = int(result.tensors["features"].sum(0).argmax())
        noop = self.probe.sae.edit(layer=0, feature_ids=[feature], sae=dictionary,
            binding=binding, tokens="visual", scale=1.).run(self.inputs, metric=metric)
        torch.testing.assert_close(noop.tensors["baseline_output"], noop.tensors["edited_output"],
                                   rtol=0, atol=0)
        reconstruction = self.probe.sae.reconstruct(layer=0, sae=dictionary,
            binding=binding, tokens="visual").run(self.inputs, metric=metric)
        self.assertEqual(reconstruction.tensors["score_change"].shape, (2,))
        self.assertTrue(torch.isfinite(reconstruction.tensors["score_change"]).all())
        self.assertFalse(torch.equal(reconstruction.tensors["baseline_output"],
                                     reconstruction.tensors["edited_output"]))

    def test_unreviewed_source_or_geometry_rejects(self):
        with self.assertRaisesRegex(CapabilityError, "audited DINO"):
            make_adapter(self.model, source_model_name="unreviewed-vit")
        with self.assertRaisesRegex(CapabilityError, "interpolated"):
            self.probe.spec.layout({**self.inputs, "interpolate_pos_encoding": True})
        with self.assertRaisesRegex(ValueError, "fixed native image size"):
            self.probe.spec.layout({"pixel_values": torch.rand(1, 3, 32, 16)})

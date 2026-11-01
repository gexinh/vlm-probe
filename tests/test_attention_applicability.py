"""Propagation warnings state a mask caveat without changing measured tensors."""
import importlib.util
from importlib.metadata import version
import unittest
import warnings

import torch

from examples.tiny_model import TinyModel
from vlmprobe import CapabilityError, ModelSpec, Prober, TokenMargin
from vlm_probing.api.attention_applicability import CausalAttentionWarning
from vlm_probing.api.residual_applicability import ResidualInterventionWarning

HAS_V5 = (importlib.util.find_spec("transformers") is not None
          and version("transformers").split(".")[:2] == ["5", "3"])


class AttentionApplicabilityTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        torch.set_num_threads(1)
        self.model = TinyModel().eval().requires_grad_(False)
        self.probe = Prober(self.model)
        self.inputs = {"input_ids": torch.tensor([[1, 2, 3]]),
                       "image_tokens": torch.randn(1, 2, 6)}

    def methods(self):
        return {
            "rollout": lambda: self.probe.attention.rollout(),
            "relevance": lambda: self.probe.attention.relevance(),
            "tam": lambda: self.probe.attention.tam(steps=2, input_key="image_tokens"),
            "beyond_intuition": lambda: self.probe.attention.beyond_intuition(steps=2, input_key="image_tokens"),
        }

    def test_explicit_mask_contract_preserves_unknown_default_and_rejects_nonbool(self):
        self.assertIsNone(ModelSpec({0: "residual"}).causal_attention)
        for invalid in (0, 1, "causal"):
            with self.subTest(value=invalid):
                with self.assertRaisesRegex(TypeError, "causal_attention"):
                    ModelSpec({0: "residual"}, causal_attention=invalid).validate({"residual": None})

    def test_declared_residual_limits_warn_and_undeclared_stacks_stay_silent(self):
        for name, value in (("residual_bypass", 1), ("last_stream_kv_layer", 99)):
            with self.subTest(field=name), self.assertRaises((TypeError, ValueError)):
                ModelSpec({0: "residual"}, **{name: value}).validate({"residual": None})
        final = max(self.probe.spec.residuals)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.assertIsNone(self.probe.causal.patch(layers=[0]).applicability)
        self.assertFalse(caught)
        self.probe.spec.residual_bypass = True
        with self.assertWarnsRegex(ResidualInterventionWarning, "does not restore"):
            self.probe.causal.patch(layers=[0])
        # Nothing runs after the final residual, and steering restores no source.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.probe.causal.patch(layers=[final])
            self.probe.causal.steer(layers=[0])
            self.probe.causal.knockout(layers=[0])
        self.assertFalse(caught)

    def test_declared_kv_boundary_limits_propagation(self):
        self.assertEqual(self.probe.attention.rollout().layers, sorted(self.probe.spec.attentions))
        self.probe.spec.last_stream_kv_layer = 0
        self.assertEqual(self.probe.describe()["methods"]["attention.rollout"]["sites"], [0])
        probe = self.probe.attention.rollout()
        self.assertEqual(probe.layers, [0])
        self.assertEqual(probe.applicability["propagation_limit"], 0)
        self.assertIn("stored K/V", probe.applicability["caveat"])
        with self.assertRaisesRegex(CapabilityError, "stops at layer 0"):
            self.probe.attention.relevance(layers=[0, 1])

    def test_only_declared_causal_propagation_warns_at_construction(self):
        for causal in (True, False, None):
            self.probe.spec.causal_attention = causal
            for name, factory in self.methods().items():
                with self.subTest(causal=causal, method=name), warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    factory()
                    matching = [entry for entry in caught if issubclass(entry.category, CausalAttentionWarning)]
                    self.assertEqual(len(matching), int(causal is True))
                    if matching:
                        self.assertIn("can favor early sequence positions", str(matching[0].message))
                        self.assertIn("still depend on attention weights", str(matching[0].message))
                        self.assertIn("formula is unchanged", str(matching[0].message))
        self.probe.spec.causal_attention = True
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.probe.attention.profile()
            self.probe.attention.grad_cam()
            self.probe.attention.attribution(steps=2)
        self.assertFalse(caught)

    def test_results_record_applicability_without_changing_raw_formulas(self):
        # Unknown versus declared causal uses the identical model and input. Only
        # warning/metadata change; no mask correction or spatial normalization.
        for name, factory in self.methods().items():
            with self.subTest(method=name):
                self.probe.spec.causal_attention = None
                unknown = factory().run(self.inputs, **({} if name == "rollout" else {"metric": TokenMargin(4, 5)}))
                self.assertEqual(unknown.metadata["applicability"]["attention_pattern"], "unspecified")
                self.assertIn("unknown", unknown.metadata["applicability"]["caveat"])
                self.probe.spec.causal_attention = True
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", CausalAttentionWarning)
                    causal = factory().run(self.inputs, **({} if name == "rollout" else {"metric": TokenMargin(4, 5)}))
                self.assertTrue(causal.metadata["applicability"]["causal_attention"])
                self.assertFalse(causal.metadata["applicability"]["propagation_formula_modified"])
                for key in unknown.tensors:
                    torch.testing.assert_close(causal.tensors[key], unknown.tensors[key], rtol=0, atol=0)

    @unittest.skipUnless(HAS_V5, "native adapters audited for Transformers 5.3.x")
    def test_native_language_and_vit_declare_different_mask_semantics(self):
        from transformers import GPT2Config, GPT2LMHeadModel, ViTConfig, ViTForImageClassification
        decoder = Prober(GPT2LMHeadModel(GPT2Config(
            vocab_size=16, n_embd=8, n_layer=1, n_head=2, n_positions=8,
            attn_implementation="eager")))
        vision = Prober(ViTForImageClassification(ViTConfig(
            image_size=8, patch_size=4, hidden_size=8, intermediate_size=16,
            num_hidden_layers=1, num_attention_heads=2, num_labels=2,
            attn_implementation="eager")))
        self.assertIs(decoder.spec.causal_attention, True)
        self.assertIs(vision.spec.causal_attention, False)
        with self.assertWarns(CausalAttentionWarning):
            decoder.attention.rollout()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = vision.attention.rollout().run({"pixel_values": torch.randn(1, 3, 8, 8)})
        self.assertFalse(caught)
        self.assertEqual(result.metadata["applicability"]["attention_pattern"], "bidirectional")
        self.assertIn("does not establish spatial faithfulness", result.metadata["applicability"]["caveat"])


if __name__ == "__main__":
    unittest.main()

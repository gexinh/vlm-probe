"""Native Gemma mixed visual masks require an explicit method-scope declaration."""
import importlib.util
from importlib.metadata import version
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
from gemma4_smoke import make_gemma4
from vlmprobe import CapabilityError, Prober

HAS_GEMMA4 = (importlib.util.find_spec("transformers") is not None
              and version("transformers") == "5.5.4")


@unittest.skipUnless(HAS_GEMMA4, "Gemma 4 requires the isolated Transformers 5.5.4 runtime")
class Gemma4MixedAttentionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(71)
        torch.set_num_threads(1)

    def test_future_visual_support_and_native_derivative_are_preserved(self):
        for bidirectional in (None, "vision"):
            with self.subTest(bidirectional=bidirectional):
                torch.manual_seed(71)
                model, inputs = make_gemma4(shared_layers=0, per_layer_width=0,
                                            bidirectional=bidirectional)
                model.requires_grad_(False)
                inputs = {key: value[:1] for key, value in inputs.items()}
                native = model(**inputs, use_cache=False).logits.detach()
                probe = Prober(model)
                source_site, final_site = probe.spec.residuals[0], probe.spec.residuals[3]
                trace = probe._run(inputs, capture=[source_site, probe.spec.attention_supports[2],
                                                  probe.spec.attention_supports[1]])
                torch.testing.assert_close(trace.logits, native, atol=0, rtol=0)
                sliding = trace.activations[probe.spec.attention_supports[2]]
                global_ = trace.activations[probe.spec.attention_supports[1]]
                self.assertEqual(bool(sliding[0, 0, 2, 5]), bidirectional == "vision")
                self.assertFalse(global_[0, 0, 2, 5])
                source = trace.activations[source_site].detach().requires_grad_(True)
                final = probe._run(inputs, capture=[final_site], grad=True,
                                   interventions={source_site: lambda _: source}).activations[final_site]
                gradient = torch.autograd.grad(final[0, 2, 0], source)[0][0, 5]
                if bidirectional == "vision":
                    self.assertGreater(gradient.abs().max().item(), 1e-6)
                    self.assertIsNone(probe.spec.causal_attention)
                    self.assertEqual(probe.spec.attention_pattern, "mixed")
                    applicability = probe.attention.rollout().run(inputs).metadata["applicability"]
                    self.assertEqual(applicability["attention_pattern"], "mixed")
                    self.assertIsNone(applicability["causal_attention"])
                    with patch.object(probe, "_run", side_effect=AssertionError("forward must not run")):
                        with self.assertRaisesRegex(CapabilityError, "mixed causal/bidirectional"):
                            probe.lens.jacobian(layers=[0], tokens=[5])
                    self.assertFalse(probe.describe()["methods"]["lens.jacobian"]["available"])
                else:
                    self.assertEqual(gradient.abs().max().item(), 0)
                    self.assertIs(probe.spec.causal_attention, True)
                    self.assertEqual(probe.spec.attention_pattern, "causal")
                    target = torch.zeros_like(inputs["input_ids"], dtype=torch.bool)
                    target[:, 2] = True
                    fitted = probe.lens.jacobian(layers=[0], tokens=[5]).fit(
                        inputs, target_mask=target, skip_first=0, exclude_last=False)
                    self.assertEqual(fitted.kernels[0].jacobian.abs().max().item(), 0)
                    self.assertTrue(probe.describe()["methods"]["lens.jacobian"]["available"])

    def test_vision_mode_stays_conservative_for_text_only_inputs(self):
        model, _ = make_gemma4(bidirectional="vision")
        probe = Prober(model)
        ids = torch.tensor([[2, 4, 5, 3]])
        inputs = {"input_ids": ids, "attention_mask": torch.ones_like(ids)}
        native = model(**inputs, use_cache=False).logits.detach()
        actual = probe.lens.logit(layers=[3], tokens="all").run(inputs)
        torch.testing.assert_close(actual.tensors["logits"][0], native[0], atol=0, rtol=0)
        self.assertEqual(probe.spec.attention_pattern, "mixed")
        with self.assertRaisesRegex(CapabilityError, "separately validated estimator"):
            probe.lens.jacobian(layers=[0])

    def test_mixed_mask_metadata_keeps_the_shared_kv_boundary(self):
        model, inputs = make_gemma4(bidirectional="vision")
        probe = Prober(model)
        result = probe.attention.rollout().run(inputs)
        self.assertEqual(result.metadata["layers"], [0, 1])
        applicability = result.metadata["applicability"]
        self.assertEqual(applicability["attention_pattern"], "mixed")
        self.assertEqual(applicability["propagation_limit"], 1)
        self.assertIn("bidirectional visual blocks", applicability["caveat"])
        self.assertIn("stored K/V", applicability["caveat"])
        with self.assertRaisesRegex(CapabilityError, "stops at layer 1"):
            probe.attention.rollout(layers=[0, 1, 2])


if __name__ == "__main__":
    unittest.main()

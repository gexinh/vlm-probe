"""Mixed masks retain their identity and cannot use the causal Jacobian recipe."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import warnings

import torch

from examples.tiny_model import TinyModel
from vlmprobe import CapabilityError, ModelSpec, Prober
from vlm_probing.api.attention_applicability import CausalAttentionWarning


class MixedAttentionContractTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        torch.set_num_threads(1)
        self.model = TinyModel().eval().requires_grad_(False)
        self.probe = Prober(self.model)
        self.inputs = {"input_ids": torch.tensor([[1, 2, 3]]),
                       "image_tokens": torch.randn(1, 2, 6)}

    def test_pattern_validation_preserves_legacy_and_rejects_contradictions(self):
        for causal in (None, True, False):
            ModelSpec({0: "residual"}, causal_attention=causal).validate({"residual": None})
        for pattern, causal in (("causal", True), ("bidirectional", False), ("mixed", None),
                                ("causal", None), ("bidirectional", None)):
            with self.subTest(pattern=pattern, causal=causal):
                ModelSpec({0: "residual"}, causal_attention=causal,
                          attention_pattern=pattern).validate({"residual": None})
        for invalid in (True, 1, [], {}):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(TypeError, "attention_pattern"):
                ModelSpec({0: "residual"}, attention_pattern=invalid).validate({"residual": None})
        for pattern, causal in (("unidirectional", None), ("causal", False),
                                ("bidirectional", True), ("mixed", True), ("mixed", False)):
            with self.subTest(pattern=pattern, causal=causal):
                with self.assertRaisesRegex(ValueError, "attention_pattern"):
                    ModelSpec({0: "residual"}, causal_attention=causal,
                              attention_pattern=pattern).validate({"residual": None})

    def test_mixed_metadata_preserves_the_measured_rollout(self):
        baseline = self.probe.attention.rollout().run(self.inputs)
        self.probe.spec.attention_pattern = "mixed"
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            actual = self.probe.attention.rollout().run(self.inputs)
        self.assertFalse([item for item in caught if issubclass(item.category, CausalAttentionWarning)])
        self.assertEqual(actual.tensors.keys(), baseline.tensors.keys())
        for name in actual.tensors:
            torch.testing.assert_close(actual.tensors[name], baseline.tensors[name], atol=0, rtol=0)
        applicability = actual.metadata["applicability"]
        self.assertEqual(applicability["attention_pattern"], "mixed")
        self.assertIsNone(applicability["causal_attention"])
        self.assertFalse(applicability["propagation_formula_modified"])
        self.assertIn("causal text with bidirectional visual blocks", applicability["caveat"])
        self.assertIn("structural support", applicability["caveat"])

    def test_explicit_pure_patterns_match_legacy_warning_and_metadata(self):
        for pattern, expected_causal in (("causal", True), ("bidirectional", False)):
            with self.subTest(pattern=pattern), warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                self.probe.spec.attention_pattern = pattern
                result = self.probe.attention.rollout().run(self.inputs)
                self.assertEqual(sum(issubclass(item.category, CausalAttentionWarning)
                                     for item in caught), int(expected_causal))
                self.assertIs(result.metadata["applicability"]["causal_attention"], expected_causal)
                self.assertEqual(result.metadata["applicability"]["attention_pattern"], pattern)

    def test_mixed_jacobian_factory_is_unavailable_before_forward(self):
        self.probe.spec.attention_pattern = "mixed"
        with patch.object(self.probe, "_run", side_effect=AssertionError("forward must not run")) as run:
            capability = self.probe.describe()["methods"]["lens.jacobian"]
            self.assertFalse(capability["available"])
            self.assertEqual(capability["sites"], [])
            self.assertIn("mixed attention", capability["missing"])
            with self.assertRaisesRegex(CapabilityError, "mixed causal/bidirectional"):
                self.probe.lens.jacobian(layers=[0])
            run.assert_not_called()

    def test_changed_declaration_cannot_bypass_fit_run_or_artifact_checks(self):
        # An existing calibration must not be reused after changing its contract.
        method = self.probe.lens.jacobian(layers=[0]).fit(
            self.inputs, skip_first=0, exclude_last=False)
        previous = method.kernels[0].jacobian.clone()
        self.probe.spec.attention_pattern = "mixed"
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "no-artifact"
            with patch.object(self.probe, "_run", side_effect=AssertionError("forward must not run")) as run:
                for action in (lambda: method.fit(self.inputs, skip_first=0, exclude_last=False),
                               lambda: method.run(self.inputs), lambda: method._capture(self.inputs),
                               lambda: method.load(target), lambda: method.save(target)):
                    with self.subTest(action=action), self.assertRaisesRegex(CapabilityError, "causal continuation"):
                        action()
                run.assert_not_called()
            self.assertFalse(target.exists())
        torch.testing.assert_close(method.kernels[0].jacobian, previous, atol=0, rtol=0)

    def test_unknown_custom_adapters_keep_existing_jacobian_behavior(self):
        self.assertIsNone(self.probe.spec.causal_attention)
        self.assertIsNone(self.probe.spec.attention_pattern)
        self.assertTrue(self.probe.describe()["methods"]["lens.jacobian"]["available"])
        method = self.probe.lens.jacobian(layers=[0]).fit(
            self.inputs, skip_first=0, exclude_last=False)
        result = method.run(self.inputs)
        self.assertEqual(method.kernels[0].calibration["estimator"],
                         "sum_valid_causal_targets_mean_sources_mean_prompts")
        self.assertTrue(torch.isfinite(result.tensors["logits"]).all())


if __name__ == "__main__":
    unittest.main()

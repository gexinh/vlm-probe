"""Explicit mass diagnostics distinguish distribution, geometry and causal claims."""
import unittest
import json
from pathlib import Path
import tempfile

import torch

from vlmprobe.attention import (
    AttentionRollout, AttentionRelevance, BeyondIntuition,
    TransitionAttentionMaps, attention_position_diagnostics,
)


class AttentionPositionDiagnosticsTests(unittest.TestCase):
    def test_selected_query_axes_raw_masses_and_explicit_geometry(self):
        scores = torch.arange(1, 25, dtype=torch.float64).reshape(2, 12).requires_grad_()
        before = scores.detach().clone()
        result = attention_position_diagnostics(scores, grid_shape=(3, 4), tile_sizes=[4, 8])
        tensors = result.tensors
        torch.testing.assert_close(tensors["top_mass"], torch.tensor([10., 58.], dtype=torch.float64))
        torch.testing.assert_close(tensors["bottom_mass"], torch.tensor([42., 90.], dtype=torch.float64))
        torch.testing.assert_close(tensors["top_bottom_ratio"], torch.tensor([10 / 42, 58 / 90], dtype=torch.float64))
        torch.testing.assert_close(tensors["tile_mass"], torch.tensor([[10., 68.], [58., 164.]], dtype=torch.float64))
        torch.testing.assert_close(tensors["first_last_tile_ratio"], torch.tensor([10 / 68, 58 / 164], dtype=torch.float64))
        torch.testing.assert_close(tensors["first_last_tile_mean_ratio"], 2 * tensors["first_last_tile_ratio"])
        torch.testing.assert_close(scores.detach(), before)
        self.assertIsNone(scores.grad)
        self.assertFalse(tensors["tile_mass"].requires_grad)
        self.assertEqual(result.metadata["score_transformation"], "none")
        self.assertIn("does not by itself", result.metadata["interpretation"])

    def test_zero_mass_tiny_grids_and_equal_uniform_bands(self):
        uniform = attention_position_diagnostics(torch.ones(24, dtype=torch.bfloat16), grid_shape=(6, 4), tile_sizes=[8, 16])
        self.assertEqual(uniform.tensors["top_bottom_ratio"].dtype, torch.float64)
        self.assertEqual(float(uniform.tensors["top_bottom_ratio"]), 1.)
        self.assertEqual(float(uniform.tensors["first_last_tile_ratio"]), .5)
        self.assertEqual(float(uniform.tensors["first_last_tile_mean_ratio"]), 1.)
        zero = attention_position_diagnostics(torch.zeros(12), grid_shape=(3, 4))
        self.assertTrue(torch.isnan(zero.tensors["top_bottom_ratio"]))
        with tempfile.TemporaryDirectory() as directory:
            zero.save(directory)
            restored = torch.load(Path(directory) / "tensors.pt", weights_only=True)
            self.assertTrue(torch.isnan(restored["top_bottom_ratio"]))
            metadata = json.loads((Path(directory) / "metadata.json").read_text())
            self.assertEqual(metadata["ratio_zero_denominator"], "+inf for positive/zero; NaN for zero/zero")
        one_sided = attention_position_diagnostics(torch.tensor([1., 0., 0.]), grid_shape=(3, 1))
        self.assertTrue(torch.isposinf(one_sided.tensors["top_bottom_ratio"]))
        tiny = attention_position_diagnostics(torch.ones(2), grid_shape=(1, 2))
        self.assertTrue(torch.isnan(tiny.tensors["sequence_first_last_third_ratio"]))
        self.assertTrue(torch.isnan(tiny.tensors["top_bottom_ratio"]))

    def test_ratios_are_stable_at_extreme_finite_scales_without_changing_raw_mass(self):
        # FP32 sums used to overflow on the first case. Even FP64 raw sums can
        # overflow on the second, so ratios must not divide aggregated masses.
        cases = ((3e38, torch.float32), (1e308, torch.float64),
                 (1e-300, torch.float64), (5e-324, torch.float64))
        for scale, dtype in cases:
            with self.subTest(scale=scale, dtype=dtype):
                values = torch.full((12,), scale, dtype=dtype)
                before = values.clone()
                result = attention_position_diagnostics(values, grid_shape=(3, 4), tile_sizes=[4, 8])
                tensors = result.tensors
                self.assertEqual(float(tensors["sequence_first_last_third_ratio"]), 1.)
                self.assertEqual(float(tensors["top_bottom_ratio"]), 1.)
                self.assertEqual(float(tensors["first_last_tile_ratio"]), .5)
                self.assertEqual(float(tensors["first_last_tile_mean_ratio"]), 1.)
                torch.testing.assert_close(values, before, rtol=0, atol=0)
                if scale == 3e38:
                    expected = values.double().sum()
                    self.assertTrue(torch.isfinite(tensors["total_mass"]))
                    torch.testing.assert_close(tensors["total_mass"], expected, rtol=0, atol=0)
                if scale == 1e308:
                    self.assertTrue(torch.isposinf(tensors["total_mass"]))

    def test_invalid_inputs_and_geometry_do_not_guess_or_clip(self):
        for values in (torch.ones(3, dtype=torch.long), [1., 2., 3.]):
            with self.assertRaises(TypeError):
                attention_position_diagnostics(values)
        for values in (torch.tensor(1.), torch.empty(0), torch.tensor([float("nan")]), torch.tensor([-1., 2., 3.])):
            with self.assertRaises(ValueError):
                attention_position_diagnostics(values)
        for kwargs in ({"grid_shape": (2, 2)}, {"grid_shape": (True, 12)},
                       {"grid_shape": (0, 12)}, {"tile_sizes": [12]},
                       {"tile_sizes": [3, 3]}, {"tile_sizes": [4, 0, 8]}):
            with self.assertRaises(ValueError):
                attention_position_diagnostics(torch.ones(12), **kwargs)

    def test_causal_mask_control_is_biased_but_remains_weight_dependent(self):
        # A content-free lower-triangular control alone produces early-key mass.
        # Reweighting the SAME permitted edges changes it: the formulas are not
        # mathematically content independent, nor is relevance generally immune.
        tokens, layers = 48, 6
        causal = torch.ones(tokens, tokens, dtype=torch.float64).tril()
        causal /= causal.sum(-1, keepdim=True)
        bidirectional = torch.full_like(causal, 1 / tokens)
        edge_weights = torch.linspace(.05, 4., tokens, dtype=torch.float64).square()
        changed = causal * edge_weights
        changed /= changed.sum(-1, keepdim=True)
        stacks = {"causal": causal.expand(layers, 1, 1, tokens, tokens),
                  "bidirectional": bidirectional.expand(layers, 1, 1, tokens, tokens),
                  "changed": changed.expand(layers, 1, 1, tokens, tokens)}
        ratios = {}
        for name, stack in stacks.items():
            row = AttentionRollout().run(stack).tensors["rollout"][0, -1, :-1]
            ratios[name] = attention_position_diagnostics(row).tensors["sequence_first_last_third_ratio"]
        self.assertGreater(float(ratios["causal"]), 10)
        self.assertAlmostEqual(float(ratios["bidirectional"]), 1., places=12)
        self.assertGreater(float(ratios["causal"]), 10 * float(ratios["changed"]))
        stack = stacks["causal"]
        gradient = torch.ones_like(stack)
        rows = [AttentionRelevance().run(stack, gradient).tensors["relevance"][0, -1, :-1],
                TransitionAttentionMaps().run(stack, gradient[-1]).tensors["relevance"][0, -1, :-1],
                BeyondIntuition().run(stack, gradient[-1], gradients=gradient).tensors["relevance"][0, -1, :-1]]
        for row in rows:
            ratio = attention_position_diagnostics(row).tensors["sequence_first_last_third_ratio"]
            self.assertGreater(float(ratio), 10.)


if __name__ == "__main__":
    unittest.main()

"""Analytic checks for MJ-Lens causal pair weighting and readout contracts."""
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest.mock import Mock, patch

import torch

from vlm_probing.lenses.mj import MJLens, mj_jacobian_sums_from_graph


BINDING = {"model_id": "causal-model@fixed", "site": "block-0-input",
           "readout_id": "native-rms-and-text-head@fixed", "tokenizer_id": "vocabulary@fixed",
           "calibration_id": "held-out-prompts@fixed"}


class MJLensTests(unittest.TestCase):
    def test_pair_weighted_matrix_matches_analytic_unequal_prompt_masks(self):
        # Rectangular masks have impossible earlier-target pairs, excluded from
        # the denominator; different prompt scales expose global pair weighting.
        states = torch.ones(2, 3, 2, dtype=torch.float64)
        sources = torch.tensor([[1, 0, 1], [0, 1, 0]], dtype=torch.bool)
        targets = torch.tensor([[0, 1, 1], [1, 1, 1]], dtype=torch.bool)
        weights = torch.tensor([[[1., 0., 0.], [2., 3., 0.], [4., 5., 6.]],
                                [[7., 0., 0.], [8., 9., 0.], [10., 11., 12.]]], dtype=torch.float64)
        head = torch.tensor([[1., 2.], [-3., 4.], [5., -6.]], dtype=torch.float64)
        def downstream(hidden):
            return torch.bmm(weights, hidden) @ head.T
        lens = MJLens(lambda x: x).fit(states, downstream, source_mask=sources,
                                        target_mask=targets, dim_batch=2)
        # Prompt 0: (2+4+6)/3 = 4. Prompt 1: (9+11)/2 = 10.
        torch.testing.assert_close(lens.jacobian, 7 * head, rtol=0, atol=0)
        self.assertEqual(lens.calibration["pair_counts"], [3, 2])
        self.assertEqual(lens.calibration["n_prompts"], 2)
        self.assertEqual(lens.calibration["source_positions"], [[0, 2], [1]])
        self.assertEqual(lens.calibration["target_positions"], [[1, 2], [0, 1, 2]])

    def test_identity_includes_single_position_self_pair(self):
        lens = MJLens(lambda x: x).fit(torch.ones(1, 1, 2), lambda x: x,
            source_mask=torch.ones(1, 1, dtype=torch.bool),
            target_mask=torch.ones(1, 1, dtype=torch.bool))
        torch.testing.assert_close(lens.jacobian, torch.eye(2, dtype=torch.float64), rtol=0, atol=0)
        self.assertEqual(lens.calibration["pair_counts"], [1])

    def test_zero_causal_effects_remain_in_pair_expectation(self):
        # Identity only influences self. All six admissible causal pairs count,
        # so three identity self derivatives yield I/2, rather than I.
        masks = torch.ones(1, 3, dtype=torch.bool)
        lens = MJLens(lambda x: x).fit(torch.ones(1, 3, 2), lambda x: x,
                                      source_mask=masks, target_mask=masks)
        torch.testing.assert_close(lens.jacobian, .5 * torch.eye(2, dtype=torch.float64), rtol=0, atol=0)
        self.assertEqual(lens.calibration["pair_counts"], [6])

    def test_intermediate_sources_share_vjps_and_split_groups(self):
        states = torch.ones(3, 3, 2, dtype=torch.float64, requires_grad=True)
        middle = 2 * states.cumsum(1)
        final = middle.cumsum(1)
        masks = torch.ones(3, 3, dtype=torch.bool)
        with patch("torch.autograd.grad", wraps=torch.autograd.grad) as backward:
            sums, report = mj_jacobian_sums_from_graph(final, {0: states, 1: middle},
                masks, masks, dim_batch=1, group_ids=torch.tensor([0, 1, 0]))
        self.assertEqual(backward.call_count, 2)
        # Sum_{s<=t} 2*(t-s+1) = 20, /6 pairs, summed over 3 prompts =>10I.
        torch.testing.assert_close(sums[0], 10 * torch.eye(2, dtype=torch.float64))
        torch.testing.assert_close(sums[1], 3 * torch.eye(2, dtype=torch.float64))
        self.assertEqual(report["group_counts"], {0: 2, 1: 1})
        self.assertEqual(report["pair_counts"], [6, 6, 6])
        for layer in sums:
            self.assertEqual(sums[layer].device.type, "cpu")
            self.assertEqual(sums[layer].dtype, torch.float64)
            torch.testing.assert_close(report["group_sums"][0][layer] + report["group_sums"][1][layer], sums[layer])
            torch.testing.assert_close(report["group_sums"][0][layer] / 2,
                                       report["group_sums"][1][layer])

    def test_prompt_sum_merges_unequal_batches_without_reweighting(self):
        states = torch.tensor([1., 3., 8.], dtype=torch.float64)[:, None, None]
        def estimate(values):
            values = values.clone().requires_grad_(True)
            mask = torch.ones(values.shape[:2], dtype=torch.bool)
            sums, report = mj_jacobian_sums_from_graph(values.square(), {4: values}, mask, mask)
            return sums[4], report["n_prompts"]
        joint, total = estimate(states)
        left, nleft = estimate(states[:2])
        right, nright = estimate(states[2:])
        torch.testing.assert_close((left + right) / (nleft + nright), joint / total)
        torch.testing.assert_close(joint / total, torch.tensor([[8.]], dtype=torch.float64))

    def test_valid_mask_intersects_channels_before_pair_counting(self):
        mask = torch.ones(2, 3, dtype=torch.bool)
        valid = torch.tensor([[1, 1, 0], [0, 1, 1]], dtype=torch.bool)
        lens = MJLens(lambda x: x).fit(torch.ones(2, 3, 1), lambda x: x.cumsum(1),
            source_mask=mask, target_mask=mask, valid_mask=valid)
        self.assertEqual(lens.calibration["pair_counts"], [3, 3])
        self.assertEqual(lens.calibration["source_positions"], [[0, 1], [1, 2]])
        torch.testing.assert_close(lens.jacobian, torch.ones(1, 1, dtype=torch.float64))

    def test_all_source_target_modality_pairs_use_the_same_pair_estimator(self):
        mask = torch.ones(1, 2, dtype=torch.bool)
        for source in ("visual", "text"):
            for target in ("visual", "text"):
                with self.subTest(source=source, target=target):
                    lens = MJLens(lambda x: x, source_modality=source, target_modality=target)
                    lens.fit(torch.ones(1, 2, 2), lambda x: x.cumsum(1), source_mask=mask, target_mask=mask)
                    result = lens.run(torch.tensor([[2., 3.]]), source_modality=source, layer=7)
                    self.assertEqual(result.method, "mj_lens")
                    self.assertEqual(result.metadata["source_modality"], source)
                    self.assertEqual(result.metadata["target_modality"], target)
                    self.assertEqual(result.metadata["layer"], 7)
                    torch.testing.assert_close(result.tensors["transported"], torch.tensor([[2., 3.]]))

    def test_no_admissible_pairs_fails_before_downstream(self):
        downstream = Mock(side_effect=AssertionError("must not run"))
        source = torch.tensor([[1, 0, 0], [0, 0, 1]], dtype=torch.bool)
        target = torch.tensor([[0, 1, 0], [1, 0, 0]], dtype=torch.bool)
        with self.assertRaisesRegex(ValueError, r"failing rows=\[1\]") as caught:
            MJLens(lambda x: x).fit(torch.ones(2, 3, 1), downstream, source_mask=source, target_mask=target)
        self.assertIn("pair_counts=[1, 0]", str(caught.exception))
        self.assertIn("no prompts were dropped", str(caught.exception))
        downstream.assert_not_called()

    def test_source_modality_mismatch_precedes_readout(self):
        readout = Mock(side_effect=AssertionError("must not run"))
        lens = MJLens(readout, source_modality="visual", target_modality="text")
        lens.jacobian = torch.eye(2)
        with self.assertRaisesRegex(ValueError, "source modality mismatch"):
            lens.run(torch.ones(2), source_modality="text")
        readout.assert_not_called()
        with self.assertRaisesRegex(ValueError, "source_modality"):
            MJLens(readout, source_modality="audio")
        with self.assertRaisesRegex(ValueError, "target_modality"):
            MJLens(readout, target_modality="image")

    def test_run_preserves_native_readout_normalization_and_caller_dtype(self):
        gain = torch.tensor([2., -3.])
        def readout(value):
            return value / value.square().mean(-1, keepdim=True).sqrt() * gain
        lens = MJLens(readout)
        lens.jacobian = 4 * torch.eye(2, dtype=torch.float64)
        result = lens.run(torch.tensor([[3., 4.]], dtype=torch.float32), source_modality="visual")
        self.assertEqual(result.tensors["logits"].dtype, torch.float32)
        torch.testing.assert_close(result.tensors["logits"], readout(torch.tensor([[12., 16.]])))
        with self.assertRaisesRegex(ValueError, "source hidden width"):
            lens.run(torch.ones(3), source_modality="visual")

    def test_dictionary_keeps_nonuniform_signed_rms_gain_and_selected_rows(self):
        lens = MJLens(lambda x: x)
        lens.jacobian = torch.tensor([[2., 1.], [3., -4.]], dtype=torch.float64)
        unembedding = torch.tensor([[1., 0.], [0., 1.], [2., 3.]])
        gain = torch.tensor([-2., 5.])
        expected = (unembedding * gain) @ lens.jacobian.float()
        torch.testing.assert_close(lens.dictionary(unembedding, gain), expected)
        torch.testing.assert_close(lens.dictionary(unembedding, gain, token_ids=[2, 0]), expected[[2, 0]])
        self.assertFalse(torch.allclose(expected, unembedding @ lens.jacobian.float()))
        for bad in ([3], [-1], [1.5], [True], [], [[0]]):
            with self.subTest(ids=bad), self.assertRaisesRegex(ValueError, "token_ids"):
                lens.dictionary(unembedding, gain, token_ids=bad)
        with self.assertRaisesRegex(ValueError, "final hidden width"):
            lens.dictionary(torch.ones(4, 3), torch.ones(3))

    def test_save_load_requires_same_binding_modalities_and_estimator(self):
        lens = MJLens(lambda x: x, binding=BINDING)
        mask = torch.ones(1, 2, dtype=torch.bool)
        lens.fit(torch.ones(1, 2, 2), lambda x: x.cumsum(1), source_mask=mask, target_mask=mask)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "lens.pt"
            lens.save(path)
            loaded = MJLens(lambda x: x, binding=BINDING).load(path)
            torch.testing.assert_close(loaded.jacobian, lens.jacobian)
            self.assertEqual(loaded.calibration, lens.calibration)
            for source, target in (("text", "text"), ("visual", "visual")):
                with self.subTest(source=source, target=target), self.assertRaisesRegex(ValueError, "algorithm configuration"):
                    MJLens(lambda x: x, binding=BINDING, source_modality=source, target_modality=target).load(path)
            with self.assertRaisesRegex(ValueError, "artifact binding"):
                MJLens(lambda x: x, binding={**BINDING, "site": "another-site"}).load(path)
            artifact = torch.load(path, weights_only=True)
            self.assertEqual(artifact["config"]["estimator"], lens.estimator)
            artifact["config"]["estimator"] = "sum_targets_mean_sources"
            torch.save(artifact, path)
            with self.assertRaisesRegex(ValueError, "algorithm configuration"):
                loaded.load(path)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, "fit MJ-Lens"):
                MJLens(lambda x: x).save(Path(directory) / "unfitted.pt")
            unbound = MJLens(lambda x: x)
            unbound.jacobian = torch.eye(2)
            with self.assertRaisesRegex(ValueError, "binding"):
                unbound.save(Path(directory) / "unbound.pt")

    def test_replacing_fit_warns_once_before_running_continuation(self):
        masks = torch.ones(1, 1, dtype=torch.bool)
        lens = MJLens(lambda x: x).fit(torch.ones(1, 1, 1), lambda x: x,
                                      source_mask=masks, target_mask=masks)
        downstream = Mock(side_effect=AssertionError("must not run"))
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            with self.assertRaisesRegex(RuntimeWarning, "replaces the existing estimate"):
                lens.fit(torch.ones(1, 1, 1), downstream, source_mask=masks, target_mask=masks)
        downstream.assert_not_called()

    def test_malformed_graph_masks_dimensions_and_groups_fail_clearly(self):
        hidden = torch.ones(2, 2, 1, requires_grad=True)
        mask = torch.ones(2, 2, dtype=torch.bool)
        for dim_batch in (0, -1, 1.2, True):
            with self.subTest(dim_batch=dim_batch), self.assertRaisesRegex(ValueError, "dim_batch"):
                mj_jacobian_sums_from_graph(hidden, {0: hidden}, mask, mask, dim_batch=dim_batch)
        with self.assertRaisesRegex(ValueError, "source_mask must be boolean"):
            mj_jacobian_sums_from_graph(hidden, {0: hidden}, mask.long(), mask)
        for bad in (torch.tensor([0, 1], dtype=torch.int32), torch.tensor([[0, 1]]), torch.tensor([0])):
            with self.subTest(group_shape=bad.shape), self.assertRaisesRegex(ValueError, "group_ids"):
                mj_jacobian_sums_from_graph(hidden, {0: hidden}, mask, mask, group_ids=bad)
        detached = hidden.detach()
        with self.assertRaisesRegex(ValueError, "differentiable graph"):
            mj_jacobian_sums_from_graph(detached, {0: hidden}, mask, mask)
        unrelated = torch.ones_like(hidden, requires_grad=True)
        with self.assertRaisesRegex(ValueError, "disconnected"):
            mj_jacobian_sums_from_graph(hidden * 2, {9: unrelated}, mask, mask)
        with torch.inference_mode(), self.assertRaisesRegex(ValueError, "inference_mode"):
            MJLens(lambda x: x).fit(hidden, lambda x: x, source_mask=mask, target_mask=mask)


if __name__ == "__main__":
    unittest.main()

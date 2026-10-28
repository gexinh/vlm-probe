"""Analytic precision regressions for causal kernels; no model downloads."""
import unittest

import torch

from vlm_probing.causal import EAPIG, AttributionPatching


def wide(dtype):
    return torch.float64 if dtype == torch.float64 else torch.float32


class CausalIntegrationPrecisionTests(unittest.TestCase):
    def setUp(self):
        self.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    def tearDown(self):
        torch.set_num_threads(self.old_threads)

    def test_eap_ig_constant_gradient_completeness_across_dtypes_and_steps(self):
        # A linear metric has a constant gradient, so the exact attribution is
        # delta at every step count; this isolates accumulation from quadrature.
        values = [[1., -2., .5], [3., .25, -1.]]
        for dtype in (torch.bfloat16, torch.float32, torch.float64):
            receiver = torch.zeros(2, 3, dtype=dtype)
            source = torch.tensor(values, dtype=dtype)
            expected = torch.tensor(values, dtype=wide(dtype))
            tolerance = 1e-12 if dtype == torch.float64 else 2e-6
            for steps in (20, 64, 256, 512, 1024):
                with self.subTest(dtype=dtype, steps=steps):
                    seen = set()

                    def metric(edges):
                        seen.add(edges.dtype)
                        return edges.sum()

                    result = EAPIG().run(receiver, source, edge_names=["a->z", "b->z"],
                                         metric=metric, steps=steps)
                    torch.testing.assert_close(result.tensors["attribution"], expected,
                                               rtol=tolerance, atol=tolerance)
                    torch.testing.assert_close(result.tensors["edge_scores"], expected.sum(-1),
                                               rtol=tolerance, atol=tolerance)
                    for key in ("attribution", "edge_scores", "average_gradient", "delta"):
                        self.assertEqual(result.tensors[key].dtype, wide(dtype))
                    self.assertEqual(result.metadata["analysis_dtype"], str(wide(dtype)))
                    # Replayed messages stay in the caller's native dtype.
                    self.assertEqual(seen, {dtype})

    def test_eap_ig_path_endpoints_are_exact_receiver_and_source(self):
        # 1.0078125 - 256 is not representable in BF16; a native-dtype delta
        # would move the alpha=1 endpoint away from the source activation.
        receiver = torch.tensor([[256.]], dtype=torch.bfloat16)
        source = torch.tensor([[1.0078125]], dtype=torch.bfloat16)
        points = []

        def metric(edges):
            points.append(edges.detach().clone())
            return edges.double().sum()

        result = EAPIG().run(receiver, source, edge_names=["a->z"], metric=metric, steps=4)
        torch.testing.assert_close(points[0], receiver, rtol=0, atol=0)
        torch.testing.assert_close(points[-1], source, rtol=0, atol=0)
        self.assertEqual(result.tensors["intervention_score"].item(), 1.0078125)
        self.assertEqual(result.tensors["effect"].item(), 1.0078125 - 256)
        self.assertEqual(result.tensors["completeness_error"].item(), 0.0)

    def test_eap_ig_masked_entries_stay_at_receiver(self):
        receiver = torch.tensor([[256., 3.]], dtype=torch.bfloat16)
        source = torch.tensor([[1.0078125, 5.]], dtype=torch.bfloat16)
        points = []

        def metric(edges):
            points.append(edges.detach().clone())
            return edges.double().sum()

        result = EAPIG().run(receiver, source, edge_names=["a->z"], metric=metric,
                             mask=torch.tensor([[True, False]]), steps=4)
        self.assertTrue(all(point[0, 1].item() == 3. for point in points))
        self.assertEqual(result.tensors["edge_scores"].item(), 1.0078125 - 256)

    def test_attribution_patching_difference_and_sum_avoid_native_rounding(self):
        # BF16 rounds 1.0078125 - 256 to -255 and a 257-term unit sum to 256.
        cases = (
            (torch.tensor([256., 0.]), torch.tensor([1.0078125, 1.]), 1.0078125 - 256 + 1),
            (torch.zeros(257), torch.ones(257), 257.),
        )
        for receiver, source, expected in cases:
            for dtype in (torch.bfloat16, torch.float32, torch.float64):
                with self.subTest(dtype=dtype, size=receiver.numel()):
                    receiver_, source_ = receiver.to(dtype), source.to(dtype)
                    result = AttributionPatching().run(receiver_, source_,
                                                       gradient=torch.ones_like(receiver_))
                    effect = result.tensors["estimated_effect"]
                    self.assertEqual(effect.dtype, wide(dtype))
                    self.assertEqual(effect.item(), expected)
                    self.assertEqual(result.tensors["gradient"].dtype, dtype)
                    self.assertEqual(result.metadata["analysis_dtype"], str(wide(dtype)))

    def test_attribution_patching_metric_gradient_promotes_after_native_backward(self):
        receiver = torch.tensor([256., 0.], dtype=torch.bfloat16)
        source = torch.tensor([1.0078125, 1.], dtype=torch.bfloat16)
        seen = set()

        def metric(value):
            seen.add(value.dtype)
            return value.sum()

        result = AttributionPatching().run(receiver, source, metric=metric)
        self.assertEqual(seen, {torch.bfloat16})
        self.assertEqual(result.tensors["estimated_effect"].item(), 1.0078125 - 256 + 1)


if __name__ == "__main__":
    unittest.main()

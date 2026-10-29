"""Analytic precision checks for attention analysis and native input replay."""
import math
import unittest

import torch
from torch import nn

from vlmprobe import HookPoint, ModelSpec, Prober, TokenLayout, TokenMargin, TorchModelAdapter
from vlmprobe.attention import (AttentionGradCAM, AttentionProfile, AttentionRelevance,
                                AttentionRollout, CheferTransformerAttribution)


class RecordedAttentionModel(nn.Module):
    """Expose real differentiable attention taps with known target gradients."""

    def __init__(self, attention, *, layers=1, coefficient=1., input_dependent=False):
        super().__init__()
        self.register_buffer("attention", attention)
        self.embedding = nn.Identity()
        self.taps = nn.ModuleList([nn.Identity() for _ in range(layers)])
        self.residual = nn.Identity()
        self.coefficient, self.input_dependent = coefficient, input_dependent
        self.points, self.forward_dtypes, self.sample_ids = [], [], []

    def forward(self, pixel_values, sample_id):
        self.points.append(pixel_values.detach().clone())
        self.sample_ids.append(sample_id.detach().clone())
        hidden = self.embedding(pixel_values)
        matrices = [tap(self.attention + 0 * hidden.sum()) for tap in self.taps]
        self.forward_dtypes.extend(matrix.dtype for matrix in matrices)
        score = sum(matrix.sum((1, 3)) for matrix in matrices) * self.coefficient
        if self.input_dependent:
            score = score * hidden.reshape(-1, 1)
        return {"logits": self.residual(torch.stack((score, score * 0), -1))}

    def probing_adapter(self):
        tokens = self.attention.shape[-1]
        spec = ModelSpec(
            {i: "residual" for i in range(len(self.taps))},
            attentions={i: f"attention.{i}" for i in range(len(self.taps))},
            embeddings="embedding",
            layout=lambda _: TokenLayout(torch.ones(1, tokens, dtype=torch.bool)),
        )
        return TorchModelAdapter(self, {
            "embedding": HookPoint("embedding"), "residual": HookPoint("residual"),
            **{f"attention.{i}": HookPoint(f"taps.{i}") for i in range(len(self.taps))},
        }, spec=spec)


class AttentionAnalysisPrecisionTests(unittest.TestCase):
    def setUp(self):
        self.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    def tearDown(self):
        torch.set_num_threads(self.old_threads)

    @staticmethod
    def expected_dtype(dtype):
        return torch.float64 if dtype == torch.float64 else torch.float32

    def test_rollout_preserves_probability_mass_over_many_layers(self):
        # Every represented input row sums to exactly one. Long propagation
        # must not accumulate native BF16 rounding into a spurious 1.0625 sum.
        for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                matrix = torch.tensor([[255 / 256, 1 / 256],
                                       [1 / 256, 255 / 256]], dtype=dtype)
                attention = matrix.expand(32, 1, 1, 2, 2).clone()
                original = attention.clone()
                result = AttentionRollout().run(attention)
                value = result.tensors["rollout"]
                expected = torch.linalg.matrix_power((matrix.double() + torch.eye(2)) / 2, 32)
                torch.testing.assert_close(value.double(), expected[None], rtol=2e-6, atol=2e-6)
                torch.testing.assert_close(value.double().sum(-1), torch.ones(1, 2, dtype=torch.float64),
                                           rtol=2e-6, atol=2e-6)
                self.assertEqual(value.dtype, self.expected_dtype(dtype))
                self.assertEqual(result.metadata["analysis_dtype"], str(value.dtype))
                torch.testing.assert_close(attention, original, rtol=0, atol=0)

    def test_relevance_and_chefer_keep_small_residual_increments(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            attention = torch.eye(2, dtype=dtype).expand(32, 1, 1, 2, 2).clone()
            gradients = torch.full_like(attention, 1 / 256)
            expected = torch.eye(2, dtype=torch.float64)[None] * (1 + 1 / 256) ** 32
            for kernel in (AttentionRelevance(), CheferTransformerAttribution()):
                with self.subTest(dtype=dtype, kernel=kernel.name):
                    result = kernel.run(attention, gradients)
                    value = result.tensors["relevance"]
                    torch.testing.assert_close(value.double(), expected, rtol=2e-6, atol=2e-6)
                    self.assertEqual(value.dtype, self.expected_dtype(dtype))
                    self.assertEqual(result.metadata["analysis_dtype"], str(value.dtype))

    def test_grad_cam_keeps_small_products_and_query_masks(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                attention = torch.full((1, 1, 8, 8), 1 / 8, dtype=dtype)
                gradients = torch.full_like(attention, 2 ** -23)
                query = torch.zeros(1, 8, dtype=torch.bool)
                query[:, -1] = True
                result = AttentionGradCAM().run(attention, gradients, query_mask=query)
                expected = torch.zeros(1, 8, 8, dtype=torch.float64)
                expected[:, -1] = 2 ** -26
                torch.testing.assert_close(result.tensors["cam"].double(), expected, rtol=0, atol=1e-16)
                self.assertEqual(result.tensors["cam"].dtype, self.expected_dtype(dtype))

    def test_profile_long_uniform_rows_preserve_entropy_and_concentration(self):
        for count in (8192, 32768):
            for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
                with self.subTest(count=count, dtype=dtype):
                    attention = torch.full((1, 1, 1, count), 1 / count, dtype=dtype)
                    group = torch.arange(count)[None] < count // 2
                    result = AttentionProfile().run(attention, groups={"half": group})
                    self.assertAlmostEqual(result.tensors["concentration"].item(), 1 / count, delta=1e-12)
                    tolerance = 1e-12 if dtype == torch.float64 else 5e-6
                    self.assertAlmostEqual(result.tensors["entropy"].item(), math.log(count), delta=tolerance)
                    self.assertEqual(result.tensors["group_mass"].item(), .5)
                    self.assertEqual(result.tensors["entropy"].dtype, self.expected_dtype(dtype))

    def test_validation_uses_original_probability_and_gradient_dtypes(self):
        # Native BF16 1/3 rows are valid although their FP32 sum differs from1.
        attention = torch.full((1, 1, 3, 3), 1 / 3, dtype=torch.bfloat16)
        AttentionRollout().run(attention[None])
        AttentionProfile().run(attention)
        AttentionRelevance().run(attention[None], torch.ones_like(attention[None]))
        with self.assertRaisesRegex(ValueError, "dtype"):
            AttentionRelevance().run(attention[None], torch.ones_like(attention[None], dtype=torch.float32))
        with self.assertRaisesRegex(ValueError, "sum to one"):
            AttentionRollout().run((attention * 2)[None])

    @staticmethod
    def inputs(dtype, value=1.):
        return {"pixel_values": torch.tensor([[value]], dtype=dtype),
                "sample_id": torch.tensor([37], dtype=torch.long)}

    def test_public_propagation_and_grad_cam_preserve_native_model_dtypes(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            inputs = self.inputs(dtype)
            matrix = torch.tensor([[255 / 256, 1 / 256], [1 / 256, 255 / 256]], dtype=dtype)
            model = RecordedAttentionModel(matrix.reshape(1, 1, 2, 2), layers=32)
            result = Prober(model).attention.rollout().run(inputs)
            torch.testing.assert_close(result.tensors["rollout"].double().sum(-1),
                                       torch.ones(1, 2, dtype=torch.float64), rtol=2e-6, atol=2e-6)
            self.assertEqual(set(model.forward_dtypes), {dtype})
            model = RecordedAttentionModel(torch.eye(2, dtype=dtype).reshape(1, 1, 2, 2),
                                            layers=32, coefficient=1 / 256)
            result = Prober(model).attention.relevance().run(inputs, metric=TokenMargin(0, 1))
            self.assertAlmostEqual(result.tensors["relevance"][0, -1, -1].item(),
                                   (1 + 1 / 256) ** 32, delta=2e-6)
            self.assertEqual(set(model.forward_dtypes), {dtype})
            model = RecordedAttentionModel(torch.full((1, 1, 8, 8), 1 / 8, dtype=dtype),
                                            coefficient=2 ** -23)
            result = Prober(model).attention.grad_cam(queries="all", keys="all").run(
                inputs, metric=TokenMargin(0, 1))
            self.assertEqual(result.tensors["cam"][0, 0, -1, 0].item(), 2 ** -26)
            self.assertEqual(result.tensors["cam"].dtype, self.expected_dtype(dtype))
            self.assertEqual(set(model.forward_dtypes), {dtype})

    def test_input_path_replays_exact_endpoints_and_wide_interior_points(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            for name in ("tam", "beyond_intuition"):
                for quadrature in ("left", "right", "endpoints"):
                    with self.subTest(dtype=dtype, method=name, quadrature=quadrature):
                        model = RecordedAttentionModel(torch.full((1, 1, 2, 2), .5, dtype=dtype),
                                                       input_dependent=True)
                        probe = Prober(model)
                        inputs = self.inputs(dtype, 1e-5 if dtype == torch.float16 else 1e-10)
                        baseline = torch.ones_like(inputs["pixel_values"])
                        method = getattr(probe.attention, name)(layers=[0], queries="all", keys="all",
                                                               steps=3, quadrature=quadrature)
                        result = method.run(inputs, metric=TokenMargin(0, 1), baseline=baseline)
                        points = model.points[-3:]
                        alphas = {"left": [0., 1 / 3, 2 / 3], "right": [1 / 3, 2 / 3, 1.],
                                  "endpoints": [0., .5, 1.]}[quadrature]
                        wide = self.expected_dtype(dtype)
                        for point, alpha in zip(points, alphas):
                            if alpha == 0:
                                expected = baseline
                            elif alpha == 1:
                                expected = inputs["pixel_values"]
                            else:
                                expected = (baseline.to(wide) + alpha * (
                                    inputs["pixel_values"].to(wide) - baseline.to(wide))).to(dtype)
                            torch.testing.assert_close(point, expected, rtol=0, atol=0)
                            self.assertEqual(point.dtype, dtype)
                        self.assertEqual(set(model.forward_dtypes), {dtype})
                        self.assertTrue(all(torch.equal(value, inputs["sample_id"]) for value in model.sample_ids))
                        self.assertEqual(result.tensors["feedback"].dtype, wide)
                        self.assertTrue(torch.isfinite(result.tensors["map"]).all())


if __name__ == "__main__":
    unittest.main()

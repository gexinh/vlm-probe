"""Analysis metrics retain finite differences and caller-supplied FP64 detail."""
import math
import unittest

import torch

from vlmprobe.metrics import ClassScore, sequence_logprob, token_logit_margin
from vlmprobe.causal import ActivationPatching


class MetricAnalysisPrecisionTests(unittest.TestCase):
    def test_exact_integer_intervention_scores_keep_integer_differences(self):
        receiver = torch.tensor([0.])
        source = torch.tensor([1.])
        result = ActivationPatching().run(receiver, source,
            runner=lambda value: torch.tensor(2**25 + int(value.item()), dtype=torch.int64))
        self.assertEqual(result.tensors['effect'].dtype, torch.int64)
        self.assertEqual(result.tensors['effect'].item(), 1)

    def test_integer_class_logits_and_margins_keep_exact_values(self):
        logits = torch.tensor([[2**25, 2**25 + 1]], dtype=torch.int64)
        selected = ClassScore(1).score(logits)
        self.assertEqual(selected.dtype, torch.int64)
        self.assertEqual(selected.item(), 2**25 + 1)
        self.assertEqual(token_logit_margin(logits, 1, 0).item(), 1)

    def test_half_margin_is_finite_and_backward_remains_native(self):
        logits = torch.tensor([[60000., -60000.]], dtype=torch.float16, requires_grad=True)
        margin = token_logit_margin(logits, 0, 1)
        self.assertEqual(margin.dtype, torch.float32)
        self.assertEqual(margin.item(), 120000.)
        gradient, = torch.autograd.grad(margin.sum(), logits)
        torch.testing.assert_close(gradient, torch.tensor([[1., -1.]], dtype=torch.float16))

    def test_bfloat_margin_retains_difference_between_represented_logits(self):
        logits = torch.tensor([[.10009765625, -.8984375]], dtype=torch.bfloat16)
        margin = token_logit_margin(logits, 0, 1)
        self.assertEqual(margin.dtype, torch.float32)
        self.assertEqual(margin.item(), .99853515625)

    def test_class_scores_promote_half_and_preserve_double(self):
        for dtype in [torch.float16, torch.bfloat16, torch.float32, torch.float64]:
            logits = torch.tensor([[0., 1.000000001]], dtype=dtype)
            expected_dtype = torch.float64 if dtype == torch.float64 else torch.float32
            for kind in ['logit', 'probability']:
                result = ClassScore(1, kind=kind).score(logits)
                self.assertEqual(result.dtype, expected_dtype)
                expected = logits[0, 1].item()
                if kind == 'probability':
                    expected = 1 / (1 + math.exp(-expected))
                self.assertAlmostEqual(result.item(), expected,
                                       delta=1e-14 if dtype == torch.float64 else 1e-7)

    def test_sequence_logprob_preserves_small_double_logit_difference(self):
        logits = torch.tensor([[[0., 1e-10], [0., 0.]]], dtype=torch.float64)
        ids = torch.tensor([[0, 1]])
        mask = torch.tensor([[False, True]])
        result = sequence_logprob(logits, ids, mask)
        self.assertEqual(result.dtype, torch.float64)
        self.assertAlmostEqual(result.item(), -math.log1p(math.exp(-1e-10)), delta=1e-15)


if __name__ == '__main__':
    unittest.main()

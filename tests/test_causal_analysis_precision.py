"""Analysis precision and exact native endpoints, independently of model dtype."""
import unittest

import torch
from torch import nn

from vlm_probing import HeadSite, HookPoint, ModelSpec, Prober, TokenLayout, TorchModelAdapter
from vlm_probing.causal import ActivationPatching, EAPIG

try:
    from transformers import GPT2Config, GPT2LMHeadModel
except ImportError:
    GPT2LMHeadModel = None


class NativeMessages(nn.Module):
    def __init__(self):
        super().__init__()
        self.message = nn.Identity()
        self.residual = nn.Identity()
        self.dtypes = set()
        self.forward_calls = 0

    def forward(self, input_ids, messages):
        self.forward_calls += 1
        value = self.residual(self.message(messages))
        self.dtypes.add(value.dtype)
        return {"logits": value}

    def probing_adapter(self):
        def layout(inputs):
            ids = inputs["input_ids"]
            valid = torch.ones_like(ids, dtype=torch.bool)
            return TokenLayout(valid, visual=valid, token_ids=ids)
        return TorchModelAdapter(self, {"message": HookPoint("message"),
                                        "residual": HookPoint("residual")},
                                 spec=ModelSpec({0: "residual"}, layout=layout,
                                                edges={"source_to_output": "message"},
                                                path_heads={0: HeadSite("message", 1, 1)},
                                                path_final="residual"),
                                 model_id="native-message-precision")


def native_inputs(value):
    return {"input_ids": torch.zeros(value.shape[:2], dtype=torch.long), "messages": value}


class CausalAnalysisPrecisionTests(unittest.TestCase):
    def setUp(self):
        self.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    def tearDown(self):
        torch.set_num_threads(self.threads)

    def test_eap_endpoint_does_not_reconstruct_source_from_a_cancelled_difference(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                # Even FP64 subtraction can lose this source next to the baseline.
                small = 1e-4 if dtype == torch.float16 else (1e-30 if dtype == torch.float64 else 1e-10)
                receiver = torch.tensor([[1., 3.]], dtype=dtype)
                source = torch.tensor([[small, 7.]], dtype=dtype)
                points = []

                def metric(value):
                    points.append(value.detach().clone())
                    # Both requested endpoints are positive; a spurious zero
                    # from cancellation would make this finite metric fail.
                    return value.double().log().sum()

                result = EAPIG().run(receiver, source, edge_names=["edge"],
                                     metric=metric, mask=torch.tensor([[True, False]]), steps=4)
                torch.testing.assert_close(points[0], receiver, rtol=0, atol=0)
                expected = torch.tensor([[source[0, 0].item(), 3.]], dtype=dtype)
                torch.testing.assert_close(points[-1], expected, rtol=0, atol=0)
                self.assertTrue(all(value.dtype == dtype for value in points))
                self.assertTrue(all(value[0, 1].item() == 3 for value in points))
                self.assertEqual(result.tensors["intervention_score"].item(),
                                 expected.double().log().sum().item())

    def test_native_metric_effect_subtracts_after_promotion(self):
        receiver = torch.tensor([[256.]], dtype=torch.bfloat16)
        source = torch.tensor([[1.0078125]], dtype=torch.bfloat16)
        result = EAPIG().run(receiver, source, edge_names=["edge"],
                             metric=lambda value: value.sum(), steps=32)
        expected = 1.0078125 - 256
        self.assertEqual(result.tensors["baseline_score"].dtype, torch.bfloat16)
        self.assertEqual(result.tensors["effect"].dtype, torch.float32)
        self.assertEqual(result.tensors["effect"].item(), expected)
        self.assertEqual(result.tensors["completeness_error"].item(), 0)

    def test_exact_patch_keeps_native_activations_and_finite_wide_effect(self):
        for dtype in (torch.float16, torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                receiver = torch.tensor([40000.], dtype=dtype)
                source = -receiver
                result = ActivationPatching().run(receiver, source, runner=lambda value: value.sum())
                self.assertEqual(result.tensors["edited"].dtype, dtype)
                expected_dtype = torch.float64 if dtype == torch.float64 else torch.float32
                self.assertEqual(result.tensors["effect"].dtype, expected_dtype)
                self.assertEqual(result.tensors["effect"].item(), -80000)

    def test_public_patch_promotes_effect_without_widening_replayed_messages(self):
        model = NativeMessages()
        probe = Prober(model)
        receiver = torch.tensor([[[40000.]]], dtype=torch.float16)
        source = -receiver
        result = probe.causal.patch(layers=[0]).run(
            native_inputs(receiver), source=native_inputs(source), metric=lambda logits: logits[:, -1, 0])
        self.assertEqual(result.tensors["effect"].dtype, torch.float32)
        self.assertEqual(result.tensors["effect"].item(), -80000)
        self.assertEqual(model.dtypes, {torch.float16})

    def test_public_eap_sums_native_batch_metrics_in_analysis_precision(self):
        model = NativeMessages()
        probe = Prober(model)
        receiver = torch.zeros((2, 1, 1), dtype=torch.float16)
        source = torch.full_like(receiver, 40000.)
        result = probe.causal.eap_ig(steps=2).run(
            native_inputs(receiver), source=native_inputs(source), metric=lambda logits: logits[:, -1, 0])
        self.assertEqual(result.tensors["edge_scores"].dtype, torch.float32)
        self.assertEqual(result.tensors["edge_scores"].item(), 80000)
        self.assertEqual(result.tensors["effect"].item(), 80000)
        self.assertEqual(model.dtypes, {torch.float16})

    def test_public_path_promotes_effect_without_changing_four_native_runs(self):
        model = NativeMessages()
        probe = Prober(model)
        receiver = torch.tensor([[[40000.]]], dtype=torch.float16)
        source = -receiver
        result = probe.causal.path(senders=[(0, 0)]).run(
            native_inputs(receiver), donor=native_inputs(source), metric=lambda logits: logits[:, -1, 0])
        self.assertEqual(result.tensors["baseline_score"].dtype, torch.float16)
        self.assertEqual(result.tensors["effect"].dtype, torch.float32)
        self.assertEqual(result.tensors["effect"].item(), -80000)
        self.assertEqual(model.dtypes, {torch.float16})
        self.assertEqual(model.forward_calls, 4)


@unittest.skipIf(GPT2LMHeadModel is None, "optional transformers dependency")
class TransformerGraphAnalysisPrecisionTests(unittest.TestCase):
    def setUp(self):
        self.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    def tearDown(self):
        torch.set_num_threads(self.threads)

    @staticmethod
    def model(dtype):
        torch.manual_seed(27)
        model = GPT2LMHeadModel(GPT2Config(n_layer=2, n_head=2, n_embd=8,
                                         vocab_size=23, n_positions=12,
                                         attn_pdrop=0, resid_pdrop=0, embd_pdrop=0,
                                         _attn_implementation="eager"))
        return model.to(dtype).eval().requires_grad_(False)

    def test_graph_scores_promote_source_difference_and_preserve_fp64(self):
        clean = {"input_ids": torch.tensor([[1, 2, 3, 2]]),
                 "attention_mask": torch.ones(1, 4, dtype=torch.long)}
        donor = {**clean, "input_ids": torch.tensor([[1, 2, 3, 4]])}
        for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                probe = Prober(self.model(dtype))
                method = probe.causal.eap_ig(graph="transformer", steps=2)
                metric = lambda logits: logits[:, -1, 5].double() - logits[:, -1, 6].double()
                source, corrupted, *_ = method._pair(clean, donor, metric, "position")
                expected_dtype = torch.float64 if dtype == torch.float64 else torch.float32
                expected = corrupted.to(expected_dtype) - source.to(expected_dtype)
                if dtype in (torch.float16, torch.bfloat16):
                    # Ensure this fixture would catch a native subtract-before-cast.
                    self.assertTrue((expected != (corrupted - source).to(expected_dtype)).any())
                seen = []
                original = method.graph.gradient_scores

                def inspect(inputs, point, difference, metric, layout):
                    seen.append((point.dtype, difference.detach().clone()))
                    return original(inputs, point, difference, metric, layout)

                method.graph.gradient_scores = inspect
                result = method.run(clean, donor=donor, metric=metric)
                self.assertEqual(result.tensors["edge_scores"].dtype, expected_dtype)
                self.assertEqual(result.tensors["eap_scores"].dtype, expected_dtype)
                for point_dtype, difference in seen:
                    self.assertEqual(point_dtype, dtype)
                    torch.testing.assert_close(difference, expected, rtol=0, atol=0)

    def test_right_input_path_uses_exact_native_clean_endpoint(self):
        model = self.model(torch.bfloat16)
        with torch.no_grad():
            model.transformer.wpe.weight.zero_()
            model.transformer.wte.weight[1].fill_(1e-10)
            model.transformer.wte.weight[2].fill_(1)
        probe = Prober(model)
        method = probe.causal.eap_ig(graph="transformer", steps=1, quadrature="right")
        clean = {"input_ids": torch.tensor([[1]]), "attention_mask": torch.ones(1, 1, dtype=torch.long)}
        donor = {**clean, "input_ids": torch.tensor([[2]])}
        points = []
        original = method.graph.gradient_scores

        def inspect(inputs, point, difference, metric, layout):
            points.append(point.detach().clone())
            return original(inputs, point, difference, metric, layout)

        method.graph.gradient_scores = inspect
        method.run(clean, donor=donor, metric=lambda logits: logits[:, -1, 3].float())
        expected = model.transformer.wte.weight[1].reshape(1, 1, -1)
        torch.testing.assert_close(points[-1], expected, rtol=0, atol=0)
        self.assertEqual(points[-1].dtype, torch.bfloat16)


if __name__ == "__main__":
    unittest.main()

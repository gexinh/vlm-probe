"""Native Gemma attention soft-capping must survive probe instrumentation."""
import importlib.util
from importlib.metadata import version
from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
from gemma4_smoke import make_gemma4
from vlmprobe import Prober


HAS_GEMMA4 = (importlib.util.find_spec("transformers") is not None
              and version("transformers") == "5.5.4")


def unpadded_inputs(inputs, dtype):
    # Avoid fully masked left-padding queries: native Gemma's FP32 softmax can
    # produce NaNs for those rows in an FP64 model, independently of probes.
    values = {key: value[:1] for key, value in inputs.items()}
    for key in ("input_ids", "attention_mask", "mm_token_type_ids"):
        if key in values:
            values[key] = values[key][:, :-1]
    return {key: value.to(dtype) if value.is_floating_point() else value
            for key, value in values.items()}


@unittest.skipUnless(HAS_GEMMA4, "Gemma 4 adapters require Transformers 5.5.4")
class Gemma4AttentionSoftcapTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_native_logits_and_gradients_with_attention_softcap(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            for multimodal in (False, True):
                for softcap in (None, 0.01):
                    with self.subTest(dtype=dtype, multimodal=multimodal, softcap=softcap):
                        torch.manual_seed(91)
                        model, inputs = make_gemma4(
                            multimodal=multimodal, alternative_attention=True,
                        )
                        model.to(dtype=dtype)
                        inputs = unpadded_inputs(inputs, dtype)
                        if softcap is not None:
                            inputs["softcap"] = softcap
                        differentiation = [model.get_input_embeddings().weight]
                        if multimodal:
                            inputs["pixel_values"].requires_grad_(True)
                            differentiation.append(inputs["pixel_values"])
                        native = model(**inputs, use_cache=False).logits
                        native_gradients = torch.autograd.grad(
                            native[:, -1, 4].sum(), differentiation,
                        )

                        probe = Prober(model)
                        actual = model(**inputs, use_cache=False).logits
                        actual_gradients = torch.autograd.grad(
                            actual[:, -1, 4].sum(), differentiation,
                        )
                        torch.testing.assert_close(actual, native, rtol=0, atol=0)
                        for expected, observed in zip(native_gradients, actual_gradients):
                            torch.testing.assert_close(observed, expected, rtol=0, atol=0)
                        torch.testing.assert_close(
                            probe._run(inputs).logits, native.detach(), rtol=0, atol=0,
                        )
                        final = max(probe.spec.residuals)
                        lens = probe.lens.logit(layers=[final], tokens="all").run(inputs)
                        torch.testing.assert_close(
                            lens.tensors["logits"][0], native.detach().flatten(0, 1),
                            rtol=0, atol=0,
                        )
                        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))

    def test_softcapped_scores_support_and_consumed_probabilities(self):
        torch.manual_seed(91)
        model, inputs = make_gemma4(alternative_attention=True)
        inputs = {**unpadded_inputs(inputs, torch.float32), "softcap": 0.01}
        native = model(**inputs, use_cache=False, output_attentions=True)
        probe = Prober(model)
        sites = [*probe.spec.attentions.values(), *probe.spec.attention_scores.values(),
                 *probe.spec.attention_supports.values()]
        trace = probe._run(inputs, capture=sites)
        for layer, expected in enumerate(native.attentions):
            with self.subTest(layer=layer):
                probabilities = trace.activations[probe.spec.attentions[layer]]
                scores = trace.activations[probe.spec.attention_scores[layer]]
                support = trace.activations[probe.spec.attention_supports[layer]]
                torch.testing.assert_close(probabilities, expected, rtol=0, atol=0)
                torch.testing.assert_close(scores.softmax(-1), probabilities, rtol=0, atol=0)
                self.assertLessEqual(scores[support.expand_as(scores)].abs().max().item(), 0.010001)
                self.assertEqual(probabilities.masked_select(~support).abs().sum().item(), 0)
        site = probe.spec.attentions[3]  # A shared-K/V layer still consumes this probability tensor.
        identity = probe._run(inputs, interventions={site: lambda value: value})
        torch.testing.assert_close(identity.logits, native.logits.detach(), rtol=0, atol=0)
        edited = probe._run(inputs, capture=[site], interventions={site: torch.zeros_like})
        self.assertEqual(edited.activations[site].abs().sum().item(), 0)
        self.assertGreater((edited.logits - native.logits.detach()).abs().max().item(), 1e-6)

    def test_other_native_kernels_preserve_ignored_softcap_keyword(self):
        # The shared wrapper must not apply Gemma semantics to kernels which
        # only accept this keyword through **kwargs. This tests instrumentation
        # directly and does not claim a 5.5.4 public Llama adapter contract.
        from transformers import LlamaConfig, LlamaForCausalLM
        from vlm_probing.adapters.hf_attention import install_eager_taps

        torch.manual_seed(91)
        config = LlamaConfig(vocab_size=40, hidden_size=16, intermediate_size=24,
                             num_hidden_layers=2, num_attention_heads=2,
                             num_key_value_heads=1, head_dim=8)
        config._attn_implementation = "eager"
        model = LlamaForCausalLM(config).eval()
        inputs = {"input_ids": torch.tensor([[2, 4, 5, 3]]), "use_cache": False}
        native = model(**inputs, softcap=0.01).logits
        torch.testing.assert_close(native, model(**inputs).logits, rtol=0, atol=0)
        for layer in model.model.layers:
            self.assertTrue(install_eager_taps(layer.self_attn))
        torch.testing.assert_close(model(**inputs, softcap=0.01).logits, native, rtol=0, atol=0)

    def test_different_native_kernels_remain_independent_in_either_install_order(self):
        from transformers import LlamaConfig, LlamaForCausalLM
        from vlm_probing.adapters.hf_attention import install_eager_taps

        for gemma_first in (False, True):
            with self.subTest(gemma_first=gemma_first):
                torch.manual_seed(91)
                gemma, gemma_inputs = make_gemma4(alternative_attention=True)
                gemma_inputs = {**unpadded_inputs(gemma_inputs, torch.float32), "softcap": 0.01}
                config = LlamaConfig(vocab_size=40, hidden_size=16, intermediate_size=24,
                                     num_hidden_layers=2, num_attention_heads=2,
                                     num_key_value_heads=1, head_dim=8)
                config._attn_implementation = "eager"
                llama = LlamaForCausalLM(config).eval()
                llama_inputs = {"input_ids": torch.tensor([[2, 4, 5, 3]]),
                                "use_cache": False, "softcap": 0.01}
                native_gemma = gemma(**gemma_inputs, use_cache=False).logits
                native_llama = llama(**llama_inputs).logits

                if gemma_first:
                    probe = Prober(gemma)
                for layer in llama.model.layers:
                    self.assertTrue(install_eager_taps(layer.self_attn))
                if not gemma_first:
                    probe = Prober(gemma)

                # The shared Transformers registry must not let the most recent
                # family installation change either earlier model's semantics.
                torch.testing.assert_close(
                    gemma(**gemma_inputs, use_cache=False).logits, native_gemma, rtol=0, atol=0,
                )
                torch.testing.assert_close(probe._run(gemma_inputs).logits, native_gemma, rtol=0, atol=0)
                torch.testing.assert_close(llama(**llama_inputs).logits, native_llama, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()

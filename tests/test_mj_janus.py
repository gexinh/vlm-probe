"""MJ adapter contracts and native Janus teacher-forced image replay."""
import importlib.util
from importlib.metadata import version
from dataclasses import replace
from pathlib import Path
import sys
import unittest

import torch

from vlm_probing import CapabilityError, Prober, ProbeInputs, TokenLayout
from vlm_probing.adapters import MJSpec, ModelSpec, prepare_janus_generation


HAS_V5 = (importlib.util.find_spec("transformers") is not None
          and version("transformers").split(".")[:2] == ["5", "3"])


def tiny_janus():
    from transformers import (JanusConfig, JanusForConditionalGeneration, JanusVisionConfig,
                              JanusVQVAEConfig, LlamaConfig)
    config = JanusConfig(
        text_config=LlamaConfig(vocab_size=64, hidden_size=16, intermediate_size=32,
            num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
            max_position_embeddings=64, attn_implementation="eager"),
        vision_config=JanusVisionConfig(hidden_size=16, num_hidden_layers=1,
            num_attention_heads=2, image_size=16, patch_size=8,
            projection_dim=16, num_image_tokens=4),
        vq_config=JanusVQVAEConfig(embed_dim=8, num_embeddings=32, latent_channels=32,
            base_channels=32, channel_multiplier=[1], num_res_blocks=1,
            projection_dim=16, image_token_embed_dim=16, num_patches=2),
        image_token_id=63,
    )
    return JanusForConditionalGeneration(config).eval()


class MJSpecTests(unittest.TestCase):
    def test_explicit_contract_checks_norm_width_and_hook_names(self):
        mj = MJSpec(sources={0: "source"}, final="final", norm=torch.nn.Identity(),
                    text_unembedding=torch.ones(7, 3), norm_gain=torch.ones(3))
        spec = ModelSpec(residuals={0: "final"}, mj=mj)
        spec.validate({"source": None, "final": None})
        with self.assertRaisesRegex(ValueError, "readout hidden width"):
            replace(mj, norm_gain=torch.ones(2)).validate()
        with self.assertRaisesRegex(ValueError, "unknown hook sites"):
            spec.validate({"final": None})
        with self.assertRaisesRegex(ValueError, "block-input"):
            replace(mj, sources={-1: "source"}).validate()
        with self.assertRaisesRegex(TypeError, "callable"):
            replace(mj, norm=None).validate()


@unittest.skipUnless(HAS_V5, "native Janus contracts require Transformers 5.3.x")
class NativeJanusMJTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(71)
        torch.set_num_threads(1)
        self.model = tiny_janus()

    def test_native_block_input_and_final_norm_parity(self):
        inputs = {"input_ids": torch.tensor([[1, 63, 63, 63, 63, 5, 6, 7]]),
                  "pixel_values": torch.randn(1, 3, 16, 16), "use_cache": False}
        native = {}
        handles = []
        for i, block in enumerate(self.model.model.language_model.layers):
            def capture(module, args, index=i):
                native[index] = args[0].detach().clone()
            handles.append(block.register_forward_pre_hook(capture))
        handles.append(self.model.model.language_model.norm.register_forward_pre_hook(
            lambda module, args: native.__setitem__("final", args[0].detach().clone())))
        try:
            with torch.no_grad():
                expected = self.model(**inputs).logits
        finally:
            for handle in handles:
                handle.remove()
        probe = Prober(self.model)
        mj = probe.spec.mj
        self.assertEqual(mj.sources, {0: "embeddings", 1: "residual.0"})
        self.assertEqual(mj.final, "residual.1")
        trace = probe._run(inputs, capture=[*mj.sources.values(), mj.final])
        torch.testing.assert_close(trace.logits, expected, rtol=0, atol=0)
        for layer, site in mj.sources.items():
            torch.testing.assert_close(trace.activations[site], native[layer], rtol=0, atol=0)
        torch.testing.assert_close(trace.activations[mj.final], native["final"], rtol=0, atol=0)
        torch.testing.assert_close(mj.norm(trace.activations[mj.final]) @ mj.text_unembedding.T,
                                   expected, rtol=0, atol=0)
        self.assertIs(mj.visual_head, self.model.model.generation_head)
        self.assertEqual(probe._layout(inputs, trace.logits).visual.sum().item(), 4)

    def test_native_gradient_and_vq_decoder_contract(self):
        probe = Prober(self.model)
        mj = probe.spec.mj
        inputs = {"input_ids": torch.tensor([[1, 5, 6, 7]]), "use_cache": False}
        trace = probe._run(inputs, capture=[mj.sources[0], mj.final], grad=True,
                           interventions={mj.sources[0]: lambda hidden: hidden.detach().requires_grad_(True)})
        actual = torch.autograd.grad(trace.activations[mj.final].sum(),
                                     trace.activations[mj.sources[0]])[0]
        self.assertEqual(actual.shape, (1, 4, 16))
        self.assertTrue(torch.isfinite(actual).all())
        self.assertTrue((actual != 0).any())
        decoded = mj.decode_visual_codes(torch.tensor([[1, 2, 3, 4]]))
        self.assertEqual(decoded.shape, (1, 2, 2, 3))
        self.assertTrue(all(parameter.grad is None for parameter in self.model.parameters()))

    def test_generation_replay_matches_native_embeddings_and_position_targets(self):
        probe = Prober(self.model)
        ids = torch.tensor([[1, 5, 6], [1, 7, 8]])
        codes = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]])
        prompt = {"input_ids": ids, "attention_mask": torch.ones_like(ids)}
        self.model.train()
        self.model.lm_head.eval()
        modes = [(module, module.training) for module in self.model.modules()]
        hooks = [(tuple(module._forward_hooks), tuple(module._forward_pre_hooks))
                 for module in self.model.modules()]
        batch = prepare_janus_generation(probe, prompt, codes)
        self.assertEqual(batch.context, "image_output")
        self.assertIsNone(batch.inputs.layout.token_ids)
        self.assertEqual(batch.inputs.layout.visual[0].tolist(),
                         [False, False, False, True, True, True, True])
        self.assertEqual(batch.inputs.layout.prompt[0].tolist(),
                         [True, True, True, False, False, False, False])
        self.assertEqual(batch.target_masks["visual"][0].tolist(),
                         [False, False, False, True, True, True, False])
        self.assertFalse(batch.target_masks["text"].any())
        self.assertEqual([(module, module.training) for module in self.model.modules()], modes)
        self.assertEqual(hooks, [(tuple(module._forward_hooks), tuple(module._forward_pre_hooks))
                                 for module in self.model.modules()])
        embeddings = batch.inputs.kwargs["inputs_embeds"]
        self.assertFalse(embeddings.requires_grad)
        torch.testing.assert_close(embeddings[:, :3], self.model.get_input_embeddings()(ids))
        torch.testing.assert_close(embeddings[:, 3:], self.model.prepare_embeddings_for_image_generation(codes))
        trace = probe._run(batch.inputs, capture=[probe.spec.mj.final])
        logits = probe.spec.mj.visual_head(probe.spec.mj.norm(trace.activations[probe.spec.mj.final]))
        self.assertEqual(logits.shape, (2, 7, 32))
        self.assertEqual(probe._layout(batch.inputs, trace.logits).visual.sum().item(), 8)
        torch.testing.assert_close(prompt["input_ids"], ids)
        self.assertTrue(all(parameter.grad is None for parameter in self.model.parameters()))

    def test_generation_rejects_wrong_code_ids_shape_and_dtype(self):
        probe = Prober(self.model)
        prompt = {"input_ids": torch.tensor([[1, 5]])}
        bad = (torch.tensor([[1, 32]]), torch.tensor([[1, -1]]), torch.tensor([[1]]),
               torch.tensor([1, 2]), torch.tensor([[1., 2.]]), torch.tensor([[1, 2], [3, 4]]))
        for codes in bad:
            with self.subTest(codes=codes), self.assertRaisesRegex(ValueError, "visual code"):
                prepare_janus_generation(probe, prompt, codes)

    def test_generation_rejects_padding_images_misaligned_layout_and_context(self):
        probe = Prober(self.model)
        ids = torch.tensor([[1, 5]])
        codes = torch.tensor([[1, 2]])
        prompts = (
            {"input_ids": ids, "attention_mask": torch.tensor([[1, 0]])},
            {"input_ids": ids, "pixel_values": torch.ones(1, 3, 16, 16)},
            {"input_ids": torch.tensor([[1, 63]])},
            {"input_ids": ids.float()},
            {"input_ids": torch.tensor([[1, 64]])},
            {"input_ids": ids, "use_cache": True},
            ProbeInputs({"input_ids": ids}, TokenLayout(torch.ones_like(ids, dtype=torch.bool),
                        visual=torch.tensor([[False, True]]))),
        )
        for prompt in prompts:
            with self.subTest(prompt=prompt), self.assertRaises(ValueError):
                prepare_janus_generation(probe, prompt, codes)
        with self.assertRaisesRegex(ValueError, "image_output"):
            prepare_janus_generation(probe, {"input_ids": ids}, codes, context="text_output")
        batch = prepare_janus_generation(probe, ProbeInputs({"input_ids": ids},
            TokenLayout(torch.ones_like(ids, dtype=torch.bool), token_ids=ids.clone())), codes)
        self.assertEqual(batch.target_masks["visual"].sum().item(), 1)

    def test_rms_language_contract_and_affine_layer_norm_exclusion(self):
        from transformers import GPT2Config, GPT2LMHeadModel, LlamaConfig, LlamaForCausalLM
        llama = LlamaForCausalLM(LlamaConfig(vocab_size=32, hidden_size=16,
            intermediate_size=32, num_hidden_layers=2, num_attention_heads=2,
            num_key_value_heads=2, attn_implementation="eager"))
        probe = Prober(llama)
        self.assertEqual(probe.spec.mj.sources, {0: "embeddings", 1: "residual.0"})
        self.assertIsNone(probe.spec.mj.visual_head)
        with self.assertRaisesRegex(CapabilityError, "native Transformers Janus"):
            prepare_janus_generation(probe, {"input_ids": torch.tensor([[1, 5]])}, torch.tensor([[1, 2]]))
        gpt = GPT2LMHeadModel(GPT2Config(vocab_size=32, n_embd=16, n_layer=1, n_head=2,
                                      n_positions=32, attn_implementation="eager"))
        self.assertIsNone(Prober(gpt).spec.mj)

    def test_existing_multimodal_rms_contracts_use_native_norm_and_block_inputs(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
        from hf_model_matrix import make_model
        for family in ("qwen2_vl", "qwen2_5_vl", "qwen3_vl", "qwen3_5",
                       "smolvlm", "internvl", "llava"):
            with self.subTest(family=family):
                model, inputs = make_model(family)
                probe = Prober(model)
                mj = probe.spec.mj
                self.assertIsNotNone(mj)
                self.assertIsNone(mj.visual_head)
                self.assertEqual(mj.sources[0], "embeddings")
                for i in range(1, len(probe.spec.residuals)):
                    self.assertEqual(mj.sources[i], probe.spec.residuals[i - 1])
                expected_gain = mj.norm.weight + 1 if family == "qwen3_5" else mj.norm.weight
                torch.testing.assert_close(mj.norm_gain, expected_gain, rtol=0, atol=0)
                trace = probe._run(inputs, capture=[mj.final])
                actual = mj.norm(trace.activations[mj.final]) @ mj.text_unembedding.T
                torch.testing.assert_close(actual, trace.logits, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()

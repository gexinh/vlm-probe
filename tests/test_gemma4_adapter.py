"""Native Gemma 4: shared K/V, PLE, local/global masks and actual interventions."""
import importlib.util
from importlib.metadata import version
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import warnings

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
from gemma4_smoke import make_gemma4
from vlmprobe import CapabilityError, Prober, SequenceLogProb, TokenMargin
from vlm_probing.api.residual_applicability import ResidualInterventionWarning

HAS_GEMMA4 = (importlib.util.find_spec("transformers") is not None
              and version("transformers") == "5.5.4")


def hooks(model):
    return [(tuple(m._forward_hooks), tuple(m._forward_pre_hooks)) for m in model.modules()]


@unittest.skipUnless(HAS_GEMMA4, "Gemma 4 adapters are audited for Transformers 5.5.4")
class Gemma4AdapterTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(71)
        torch.set_num_threads(1)

    def test_native_logits_gradients_readout_and_weights(self):
        for dtype in (torch.float32, torch.bfloat16):
            with self.subTest(dtype=dtype):
                model, inputs = make_gemma4()
                model.to(dtype=dtype)
                with torch.no_grad():
                    model.model.language_model.norm.weight.copy_(torch.linspace(0.7, 1.3, 16))
                    for i, layer in enumerate(model.model.language_model.layers):
                        layer.layer_scalar.fill_(0.7 + i * 0.2)
                pixels = inputs["pixel_values"].to(dtype=dtype).requires_grad_(True)
                inputs = {**inputs, "pixel_values": pixels}
                embedding = model.get_input_embeddings().weight
                native = model(**inputs, use_cache=False).logits
                native_grads = torch.autograd.grad(native[:, -1, 4].sum(), (pixels, embedding))
                state = {name: value.detach().clone() for name, value in model.state_dict().items()}
                probe = Prober(model)
                actual = model(**inputs, use_cache=False).logits
                actual_grads = torch.autograd.grad(actual[:, -1, 4].sum(), (pixels, embedding))
                torch.testing.assert_close(actual, native, atol=0, rtol=0)
                for a, b in zip(actual_grads, native_grads):
                    torch.testing.assert_close(a, b, atol=0, rtol=0)
                self.assertGreater(native_grads[0].abs().max().item(), 0)
                result = probe.lens.logit(layers=[3], tokens="all").run(inputs)
                torch.testing.assert_close(result.tensors["logits"][0],
                                           native.detach()[inputs["attention_mask"].bool()], atol=0, rtol=0)
                self.assertEqual(state.keys(), model.state_dict().keys())
                for name, value in model.state_dict().items():
                    torch.testing.assert_close(value, state[name], atol=0, rtol=0)
                self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_embedding_merge_padding_and_complete_answer_score(self):
        model, inputs = make_gemma4()
        probe = Prober(model)
        expected = {}

        def capture(_module, args):
            expected["embeddings"] = args[0].detach().clone()

        handle = model.model.language_model.layers[0].register_forward_pre_hook(capture)
        try:
            model(**inputs, use_cache=False)
        finally:
            handle.remove()
        trace = probe._run(inputs, capture=[probe.spec.embeddings])
        torch.testing.assert_close(trace.activations["embeddings"], expected["embeddings"])
        layout = probe._layout(inputs, trace.logits)
        visual = inputs["input_ids"] == 40
        torch.testing.assert_close(layout.visual, visual)
        torch.testing.assert_close(layout.valid, inputs["attention_mask"].bool())
        text = layout.valid & ~visual
        native_embeddings = model.get_input_embeddings()(inputs["input_ids"].masked_fill(visual, 0))
        torch.testing.assert_close(trace.activations["embeddings"][text], native_embeddings[text])
        nearest = probe.lens.embed(tokens="visual").run(inputs, top_k=2)
        self.assertEqual(nearest.tensors["positions"].tolist(), visual.nonzero().tolist())
        answer = torch.zeros_like(visual)
        answer[0, 7], answer[1, 8] = True, True
        scores = SequenceLogProb(answer).score(trace.logits, layout)
        expected_scores = torch.stack([trace.logits[0, 6].log_softmax(-1)[3],
                                       trace.logits[1, 7].log_softmax(-1)[3]])
        torch.testing.assert_close(scores, expected_scores)

    def test_softcap_is_part_of_readout(self):
        for cap in (None, 0.05):
            with self.subTest(cap=cap):
                model, inputs = make_gemma4(softcap=cap)
                probe = Prober(model)
                sites = [probe.spec.residuals[i] for i in sorted(probe.spec.residuals)]
                trace = probe._run(inputs, capture=sites)
                decoder = model.model.language_model
                for site in sites:  # Intermediate lens layers use the same capped readout.
                    raw = model.lm_head(decoder.norm(trace.activations[site]))
                    expected = raw if cap is None else torch.tanh(raw / cap) * cap
                    torch.testing.assert_close(probe.adapter.readout(trace.activations[site]), expected)
                    if cap is not None:
                        self.assertGreater((raw - expected).abs().max().item(), 1e-3)
                torch.testing.assert_close(trace.logits, expected)

    def test_head_widths_and_capability_boundaries(self):
        model, inputs = make_gemma4(alternative_attention=True)
        probe = Prober(model)
        for i, width in ((0, 8), (1, 16), (2, 8), (3, 16)):
            site = probe.spec.heads[i]
            raw = probe._run(inputs, capture=[site]).activations[site]
            self.assertEqual(raw.shape[-1], 2 * width)
            projection = model.get_submodule(probe.adapter.sites[site].module)
            expected = projection(raw)
            if projection.bias is not None:
                expected = expected - projection.bias
            torch.testing.assert_close(probe._heads(i, raw).sum(-2), expected)
        capabilities = probe.describe()["methods"]
        for name in ("causal.path", "attention.head_logits", "lens.mj", "causal.eap_ig"):
            self.assertFalse(capabilities[name]["available"], name)
        with self.assertRaises(CapabilityError):
            probe.causal.path(senders=[(0, 0)])
        with self.assertRaises(CapabilityError):
            probe.attention.head_logits()
        with self.assertRaises(CapabilityError):
            probe.attention.beyond_intuition(variant="token")

    def test_local_global_and_bidirectional_visual_support(self):
        for bidirectional in (None, "vision"):
            with self.subTest(bidirectional=bidirectional):
                model, inputs = make_gemma4(bidirectional=bidirectional)
                probe = Prober(model)
                trace = probe._run(inputs, capture=list(probe.spec.attention_supports.values()))
                local = trace.activations[probe.spec.attention_supports[0]]
                global_ = trace.activations[probe.spec.attention_supports[1]]
                self.assertFalse(local[0, 0, 7, 2])  # outside the local window
                self.assertTrue(global_[0, 0, 7, 2])
                self.assertFalse(global_[0, 0, 2, 7])  # future text
                self.assertFalse(global_[1, 0, 8, 0])  # left padding
                self.assertEqual(bool(local[0, 0, 2, 5]), bidirectional == "vision")

    def test_shared_layer_probabilities_are_consumed(self):
        model, inputs = make_gemma4()
        probe = Prober(model)
        site = probe.spec.attentions[3]
        base = probe._run(inputs)
        edited = probe._run(inputs, capture=[site], interventions={site: torch.zeros_like})
        self.assertEqual(edited.activations[site].abs().sum().item(), 0)
        self.assertGreater((base.logits - edited.logits).abs().max().item(), 1e-6)
        identity = probe._run(inputs, interventions={site: lambda x: x})
        torch.testing.assert_close(base.logits, identity.logits, atol=0, rtol=0)

    def test_dense_without_ple_or_kv_sharing(self):
        model, inputs = make_gemma4(shared_layers=0, per_layer_width=0,
                                    bidirectional="vision", alternative_attention=True)
        native = model(**inputs, use_cache=False).logits
        probe = Prober(model)
        torch.testing.assert_close(probe._run(inputs).logits, native, atol=0, rtol=0)
        result = probe.lens.logit(layers=[3], tokens="all").run(inputs)
        torch.testing.assert_close(result.tensors["logits"][0], native[inputs["attention_mask"].bool()])
        effect = probe.causal.knockout(layers=[3], queries="last_prompt", keys="visual").run(
            inputs, metric=TokenMargin(4, 5))
        self.assertGreater(effect.tensors["effect"].abs().max().item(), 1e-7)

    def test_residual_and_attention_interventions_with_frozen_weights(self):
        model, inputs = make_gemma4()
        model.requires_grad_(False)
        probe = Prober(model)
        before = hooks(model)
        metric = TokenMargin(4, 5)
        corrupt = {**inputs, "pixel_values": 1 - inputs["pixel_values"]}
        identity = probe.causal.patch(layers=[0, 3]).run(inputs, source=inputs, metric=metric)
        torch.testing.assert_close(identity.tensors["effect"], torch.zeros(2, 2))
        patched = probe.causal.patch(layers=[0]).run(corrupt, source=inputs, metric=metric)
        self.assertGreater(patched.tensors["effect"].abs().max().item(), 1e-7)
        probe.causal.attribute(layers=[0]).run(corrupt, source=inputs, metric=metric)
        direction = torch.linspace(-0.2, 0.2, 16)
        steering = probe.causal.steer(layers=[0, 3], joint=True).run(inputs, direction=direction, metric=metric)
        self.assertGreater(steering.tensors["effect"].abs().max().item(), 1e-6)
        for result in (
            probe.causal.knockout(layers=[1, 3], queries="last_prompt", keys="visual", joint=True).run(inputs, metric=metric),
            probe.attention.reweight(layers=[1], queries="last_prompt", weight=0).run(inputs, metric=metric),
            probe.attention.temperature(layers=[1], queries="last_prompt", temperature=2).run(inputs, metric=metric),
        ):
            self.assertGreater(result.tensors["effect"].abs().max().item(), 1e-7)
        probe.attention.grad_cam(layers=[1], queries="last_prompt", keys="visual").run(inputs, metric=metric)
        probe.attention.attribution(layers=[1], queries="last_prompt", keys="visual", steps=3).run(inputs, metric=metric)
        probe.attention.relevance(layers=[0, 1]).run(inputs, metric=metric)
        probe.attention.rollout(layers=[0, 1]).run(inputs)
        probe.attention.tam(layers=[0, 1], queries="last_prompt", keys="visual", steps=2).run(inputs, metric=metric)
        probe.attention.beyond_intuition(layers=[0, 1], queries="last_prompt", keys="visual", steps=2).run(inputs, metric=metric)
        self.assertEqual(before, hooks(model))
        self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_residual_restoration_matches_declared_bypass(self):
        for options, bypass in (({}, True), ({"shared_layers": 0}, True), ({"per_layer_width": 0}, True),
                                ({"shared_layers": 0, "per_layer_width": 0}, False)):
            with self.subTest(options=options):
                model, inputs = make_gemma4(softcap=None, **options)
                probe = Prober(model)
                self.assertEqual(probe.spec.residual_bypass, bypass)
                corrupt = {**inputs, "pixel_values": 1 - inputs["pixel_values"]}
                source = probe._run(inputs, capture=list(probe.spec.residuals.values()))
                self.assertGreater((probe._run(corrupt).logits - source.logits).abs().max().item(), 0)
                for layer, site in probe.spec.residuals.items():
                    value = source.activations[site]
                    patched = probe._run(corrupt, interventions={site: lambda _, value=value: value})
                    gap = (patched.logits - source.logits).abs().max().item()
                    if bypass and layer < 3:
                        self.assertGreater(gap, 0)
                    else:
                        self.assertEqual(gap, 0)

    def test_shared_kv_residual_edits_cannot_reach_other_positions(self):
        model, inputs = make_gemma4()
        probe = Prober(model)
        self.assertEqual(probe.spec.last_stream_kv_layer, 1)
        corrupt = {**inputs, "pixel_values": 1 - inputs["pixel_values"]}
        with self.assertWarnsRegex(ResidualInterventionWarning, "does not restore"):
            patch = probe.causal.patch(layers=[0, 1, 2, 3])
        result = patch.run(corrupt, source=inputs, metric=TokenMargin(4, 5))
        self.assertGreater(result.tensors["effect"][0].abs().max().item(), 0)
        self.assertEqual(result.tensors["effect"][1:].abs().max().item(), 0)
        self.assertEqual(result.metadata["applicability"]["position_local_layers"], [1, 2, 3])
        self.assertEqual(result.metadata["applicability"]["restoration_incomplete_layers"], [0, 1, 2])
        # A later attention edit still reaches them: the zero is about the residual route.
        knockout = probe.causal.knockout(layers=[3], queries="last_prompt", keys="visual").run(
            inputs, metric=TokenMargin(4, 5))
        self.assertGreater(knockout.tensors["effect"].abs().max().item(), 0)
        for factory in (lambda: probe.causal.steer(layers=[2]),
                        lambda: probe.causal.attribute(layers=[2]),
                        lambda: probe.lens.patchscope(layers=[0], source_position=3,
                                                      target_layer=2, target_position=3)):
            with self.assertWarnsRegex(ResidualInterventionWarning, "edited positions"):
                factory()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            probe.causal.steer(layers=[0])
        self.assertFalse([item for item in caught if issubclass(item.category, ResidualInterventionWarning)])

    def test_propagation_stops_at_last_stream_kv_layer(self):
        model, inputs = make_gemma4()
        probe = Prober(model)
        self.assertEqual(probe.describe()["methods"]["attention.rollout"]["sites"], [0, 1])
        result = probe.attention.rollout().run(inputs)
        self.assertEqual(result.metadata["layers"], [0, 1])
        self.assertEqual(result.metadata["applicability"]["propagation_limit"], 1)
        for factory in (lambda: probe.attention.rollout(layers=[0, 1, 2]),
                        lambda: probe.attention.relevance(layers=[0, 1, 2]),
                        lambda: probe.attention.tam(layers=[0, 1, 2]),
                        lambda: probe.attention.beyond_intuition(layers=[0, 1, 2])):
            with self.assertRaisesRegex(CapabilityError, "stops at layer 1"):
                factory()
        # Single-layer maps and interventions remain available on shared layers.
        probe.attention.profile(layers=[3], queries="last_prompt").run(inputs)
        model, inputs = make_gemma4(shared_layers=0)
        probe = Prober(model)
        self.assertIsNone(probe.spec.last_stream_kv_layer)
        self.assertEqual(probe.attention.rollout().run(inputs).metadata["layers"], [0, 1, 2, 3])

    def test_declared_mask_contract_and_moe_scope(self):
        with self.assertRaisesRegex(CapabilityError, "not audited"):
            Prober(make_gemma4(bidirectional="all")[0])
        model, inputs = make_gemma4(bidirectional="vision")
        probe = Prober(model)
        self.assertTrue(probe.spec.causal_attention)
        without = {key: value for key, value in inputs.items() if key != "mm_token_type_ids"}
        with self.assertRaisesRegex(CapabilityError, "require mm_token_type_ids"):
            probe._run(without)
        model, inputs = make_gemma4(moe=True, shared_layers=0, per_layer_width=0, bidirectional="vision",
                                    alternative_attention=True)
        native = model(**inputs, use_cache=False).logits
        probe = Prober(model)
        torch.testing.assert_close(probe._run(inputs).logits, native, atol=0, rtol=0)
        self.assertTrue(any("MoE" in note for note in probe.spec.notes))

    def test_text_only_prompt_on_multimodal_class(self):
        for bidirectional in (None, "vision"):
            with self.subTest(bidirectional=bidirectional):
                model, _ = make_gemma4(bidirectional=bidirectional)
                ids = torch.tensor([[2, 4, 5, 3]])
                inputs = dict(input_ids=ids, attention_mask=torch.ones_like(ids))
                native = model(**inputs, use_cache=False).logits
                probe = Prober(model)
                torch.testing.assert_close(probe._run(inputs).logits, native, atol=0, rtol=0)
                self.assertFalse(probe.spec.layout(inputs).visual.any())

    def test_calibrated_lenses_and_patchscope(self):
        model, inputs = make_gemma4()
        probe = Prober(model)
        probe.lens.tuned(layers=[0]).fit(inputs, steps=2).run(inputs)
        probe.lens.attention(layers=[1]).fit(inputs, steps=2).run(inputs)
        probe.lens.jacobian(layers=[0]).fit(inputs, skip_first=0, exclude_last=False).run(inputs)
        probe.lens.patchscope(layers=[0], source_position=3, target_layer=0, target_position=3).run(inputs, target_inputs=inputs)

    def test_joint_generation_replays_per_layer_inputs_and_visual_masks(self):
        model, batch = make_gemma4()
        inputs = {key: value[:1] for key, value in batch.items()}
        for key in ("input_ids", "attention_mask", "mm_token_type_ids"):
            inputs[key] = inputs[key][:, :-1]
        probe = Prober(model)
        result = probe.causal.steer(layers=[0], joint=True, tokens="prediction").generate(
            inputs, direction=torch.zeros(16), max_new_tokens=3)
        # A zero direction must match actual full-prefix greedy generation.
        values = dict(inputs)
        expected = []
        for _ in range(result.tensors["generated_ids"].shape[1]):
            token = model(**values, use_cache=False).logits[:, -1].argmax(-1, keepdim=True)
            expected.append(token)
            values["input_ids"] = torch.cat((values["input_ids"], token), dim=1)
            values["attention_mask"] = torch.ones_like(values["input_ids"])
            values["mm_token_type_ids"] = torch.cat((values["mm_token_type_ids"], torch.zeros_like(token)), dim=1)
        torch.testing.assert_close(result.tensors["generated_ids"], torch.cat(expected, dim=1))

    def test_text_only_and_sdpa(self):
        for multimodal in (False, True):
            for backend in ("eager", "sdpa"):
                with self.subTest(multimodal=multimodal, backend=backend):
                    model, inputs = make_gemma4(multimodal=multimodal, backend=backend)
                    native = model(**inputs, use_cache=False).logits.detach()
                    probe = Prober(model)
                    result = probe.lens.logit(layers=[3], tokens="all").run(inputs)
                    torch.testing.assert_close(result.tensors["logits"][0], native[inputs["attention_mask"].bool()])
                    self.assertEqual(bool(probe.spec.attentions), backend == "eager")
                    if not multimodal:
                        self.assertFalse(probe.spec.layout(inputs).visual.any())
                        decoded = probe.lens.embed(tokens="all").run(inputs, top_k=1)
                        torch.testing.assert_close(decoded.tensors["token_ids"][0, :, 0],
                                                   inputs["input_ids"][inputs["attention_mask"].bool()])

    def test_failure_restores_hooks_modes_backend_and_logits(self):
        model, inputs = make_gemma4()
        probe = Prober(model)
        native = probe._run(inputs).logits
        model.train()
        model.model.vision_tower.eval()
        modes = [m.training for m in model.modules()]
        before = hooks(model)

        def fail(_):
            raise RuntimeError("deliberate")

        with self.assertRaisesRegex(RuntimeError, "deliberate"):
            probe._run(inputs, interventions={probe.spec.attentions[3]: fail})
        self.assertEqual(before, hooks(model))
        self.assertEqual(modes, [m.training for m in model.modules()])
        self.assertEqual(model.config.text_config._attn_implementation, "eager")
        torch.testing.assert_close(probe._run(inputs).logits, native, atol=0, rtol=0)

    def test_version_and_modality_boundaries(self):
        model, inputs = make_gemma4()
        with patch("vlm_probing.adapters.gemma4.version", return_value="5.5.0"):
            with self.assertRaisesRegex(CapabilityError, "5.5.4"):
                Prober(model)
        probe = Prober(model)
        for extra in ({"input_features": torch.zeros(1)}, {"pixel_values_videos": torch.zeros(1)},
                      {"mm_token_type_ids": torch.full_like(inputs["input_ids"], 2)},
                      {"input_ids": inputs["input_ids"].masked_fill(inputs["input_ids"] == 40, 41)}):
            with self.subTest(extra=list(extra)), self.assertRaisesRegex(CapabilityError, "image/text"):
                probe._run({**inputs, **extra})
        with self.assertRaisesRegex(ValueError, "without KV cache"):
            probe._run({**inputs, "use_cache": True})
        with self.assertRaisesRegex(CapabilityError, "2D attention_mask"):
            probe.spec.layout({**inputs, "attention_mask": torch.ones(2, 1, 9, 9)})


if __name__ == "__main__":
    unittest.main()

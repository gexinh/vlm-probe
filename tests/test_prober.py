"""Public API contracts on actual forwards, including intervention controls."""
import sys
import tempfile
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
from tiny_model import TinyModel
from vlm_probing import (CapabilityError, ProbeInputs, Prober, SequenceLogProb,
                         TokenLayout, TokenMargin, register_adapter)
from vlm_probing.lenses import EmbedLens


class ProberTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(19)
        torch.set_num_threads(1)
        self.model = TinyModel()
        self.probe = Prober(self.model)
        self.inputs = {"input_ids": torch.tensor([[1, 2, 3], [4, 5, 6]]),
                       "image_tokens": torch.randn(2, 2, 6)}
        self.bad = {**self.inputs, "image_tokens": torch.zeros(2, 2, 6)}
        self.metric = TokenMargin(4, 5)
        self.binding = {"model_id": "tiny-seed19", "tokenizer_id": "ids",
                        "calibration_id": "fixture", "readout_id": "norm-head"}

    def test_auto_and_describe_are_read_only(self):
        summary = self.probe.describe()
        self.assertEqual(sum(x["available"] for x in summary["methods"].values()), 26)
        self.assertTrue(summary["methods"]["sae.features"]["available"])
        self.assertTrue(summary["methods"]["sae.edit"]["available"])
        self.assertTrue(self.model.training)
        with self.assertRaises(CapabilityError):
            Prober(torch.nn.Linear(2, 2))

    def test_describe_reports_missing_readout_without_running_model(self):
        self.probe.adapter.readout = None
        handle = self.model.register_forward_pre_hook(
            lambda *_: self.fail("describe must not run the model"))
        try:
            methods = self.probe.describe()["methods"]
        finally:
            handle.remove()
        for name in ("lens.logit", "lens.tuned", "lens.jacobian"):
            self.assertEqual(methods[name], {
                "available": False, "sites": [],
                "requires": "residual sites and model readout",
                "missing": "residual sites and model readout",
            })
        self.assertTrue(methods["causal.patch"]["available"])

    def test_describe_representation_keeps_only_sae_available(self):
        self.probe.spec.output_kind = "representation"
        methods = self.probe.describe()["methods"]
        for name, method in methods.items():
            with self.subTest(method=name):
                self.assertEqual(method["available"], name.startswith("sae."))
                if not name.startswith("sae."):
                    self.assertEqual(method["sites"], [])
                    self.assertEqual(method["missing"],
                        "representation encoder currently exposes SAE methods; "
                        "no language/classification logits")

    def test_describe_limits_propagation_without_hiding_observable_layers(self):
        self.probe.spec.last_stream_kv_layer = 0
        methods = self.probe.describe()["methods"]
        for name in ("rollout", "relevance", "tam", "beyond_intuition"):
            self.assertEqual(methods["attention." + name]["sites"], [0])
        self.assertEqual(methods["attention.profile"]["sites"], [0, 1])
        self.probe.spec.attentions.pop(0)
        methods = self.probe.describe()["methods"]
        self.assertFalse(methods["attention.rollout"]["available"])
        self.assertEqual(methods["attention.profile"]["sites"], [1])

    def test_registration_and_explicit_adapter(self):
        class Custom(TinyModel):
            pass
        register_adapter(Custom, lambda m: m.probing_adapter())
        m = Custom()
        self.assertIs(Prober(m).model, m)
        with self.assertRaises(ValueError):
            register_adapter(Custom, lambda m: m.probing_adapter())
        with self.assertRaisesRegex(ValueError, "exact supplied"):
            Prober(self.model, adapter=m.probing_adapter())

    def test_final_lens_matches_model_and_coordinates(self):
        result = self.probe.lens.logit(layers=[1]).run(self.inputs)
        expected = self.probe.adapter.run(self.inputs).logits[:, -1]
        torch.testing.assert_close(result.tensors["logits"][0], expected)
        self.assertEqual(result.tensors["positions"].tolist(), [[0, 4], [1, 4]])
        self.assertFalse(result.tensors["logits"].requires_grad)

    def test_explicit_prompt_masks_and_padding_selection(self):
        layout = TokenLayout(torch.tensor([[0, 1, 1, 1, 0], [1, 1, 1, 1, 1]], dtype=torch.bool),
                             prompt=torch.tensor([[0, 1, 1, 0, 0], [1, 1, 1, 1, 0]], dtype=torch.bool))
        batch = ProbeInputs(self.inputs, layout)
        result = self.probe.lens.logit(layers=[1]).run(batch)
        expected = self.probe.adapter.run(self.inputs).logits[[0, 1], [2, 3]]
        torch.testing.assert_close(result.tensors["logits"][0], expected)
        self.assertEqual(result.tensors["positions"].tolist(), [[0, 2], [1, 3]])
        with self.assertRaisesRegex(ValueError, "padding"):
            self.probe.lens.logit(tokens=[0]).run(batch)
        with self.assertRaises(CapabilityError):
            self.probe.lens.logit(tokens="visual").run(batch)

    def test_embed_uses_actual_embedding_space(self):
        result = self.probe.lens.embed().run(self.inputs, top_k=3)
        expected = EmbedLens(self.model.embedding.weight).run(self.inputs["image_tokens"].reshape(-1, 6), top_k=3)
        torch.testing.assert_close(result.tensors["similarities"][0], expected.tensors["similarities"])
        torch.testing.assert_close(result.tensors["token_ids"][0], expected.tensors["token_ids"])

    def test_fitted_lenses_save_load_and_do_not_train_model(self):
        original = {n: p.detach().clone() for n, p in self.model.named_parameters()}
        for name in ("tuned", "attention"):
            factory = getattr(self.probe.lens, name)
            method = factory(layers=[0], binding=self.binding)
            with self.assertRaises(RuntimeError):
                method.run(self.inputs)
            method.fit(self.inputs, steps=30, lr=0.02)
            self.assertLess(method.losses[0][-1], method.losses[0][0])
            expected = method.run(self.inputs).tensors["logits"]
            with tempfile.TemporaryDirectory() as path:
                method.save(path)
                loaded = factory(layers=[0], binding=self.binding).load(path)
                torch.testing.assert_close(loaded.run(self.inputs).tensors["logits"], expected)
                with self.assertRaisesRegex(ValueError, "binding"):
                    factory(layers=[0], binding={**self.binding, "model_id": "different"}).load(path)
        for n, p in self.model.named_parameters():
            torch.testing.assert_close(original[n], p)
            self.assertIsNone(p.grad)

    def test_artifacts_require_explicit_identities(self):
        method = self.probe.lens.tuned(layers=[0]).fit(self.inputs, steps=1)
        with tempfile.TemporaryDirectory() as path:
            with self.assertRaisesRegex(ValueError, "identities"):
                method.save(path)
            self.assertEqual(list(Path(path).iterdir()), [])

    def test_jacobian_final_site_is_identity_and_loads(self):
        method = self.probe.lens.jacobian(layers=[1], binding=self.binding)
        method.fit(self.inputs, skip_first=0, exclude_last=False)
        torch.testing.assert_close(method.kernels[1].jacobian, torch.eye(6))
        result = method.run(self.inputs)
        expected = self.probe.lens.logit(layers=[1], tokens="all").run(self.inputs)
        torch.testing.assert_close(result.tensors["logits"], expected.tensors["logits"])
        with tempfile.TemporaryDirectory() as path:
            method.save(path)
            loaded = self.probe.lens.jacobian(layers=[1], binding=self.binding).load(path)
            torch.testing.assert_close(loaded.run(self.inputs).tensors["logits"], result.tensors["logits"])

    def test_patchscope_runs_target_and_self_patch_is_noop(self):
        method = self.probe.lens.patchscope(layers=[0], source_position=2, target_layer=0, target_position=2)
        result = method.run(self.inputs, target_inputs=self.inputs)
        torch.testing.assert_close(result.tensors["logits"][0], self.probe.adapter.run(self.inputs).logits)
        with self.assertRaisesRegex(ValueError, "cross-model"):
            self.probe.lens.patchscope(layers=[0], source_position=2, target_layer=0,
                                       target_position=2, target=Prober(TinyModel()))

    def test_patch_sweep_is_independent_and_matches_manual_hooks(self):
        result = self.probe.causal.patch(layers=[0, 1]).run(self.bad, source=self.inputs, metric=self.metric)
        source = self.probe.adapter.run(self.inputs, capture=["residual.0", "residual.1"])
        for row, layer in enumerate([0, 1]):
            site = f"residual.{layer}"
            def edit(x):
                y = x.clone()
                y[:, :2] = source.activations[site][:, :2]
                return y
            expected = self.probe.adapter.run(self.bad, interventions={site: edit})
            layout = self.probe._layout(self.bad, expected.logits)
            torch.testing.assert_close(result.tensors["intervention_score"][row], self.metric.score(expected.logits, layout))
        noop = self.probe.causal.patch().run(self.inputs, source=self.inputs, metric=self.metric)
        torch.testing.assert_close(noop.tensors["effect"], torch.zeros(2, 2))

    def test_pair_alignment_rejects_token_or_visual_layout_changes(self):
        changed = {**self.inputs, "input_ids": self.inputs["input_ids"] + 1}
        method = self.probe.causal.patch(layers=[0])
        with self.assertRaisesRegex(ValueError, "token IDs"):
            method.run(self.inputs, source=changed, metric=self.metric)
        method.run(self.inputs, source=changed, metric=self.metric, alignment="position")
        layout = self.model.probing_adapter().spec.layout(self.inputs)
        layout.visual = ~layout.visual
        with self.assertRaisesRegex(ValueError, "visual layouts"):
            method.run(self.inputs, source=ProbeInputs(self.inputs, layout), metric=self.metric)

    def test_zero_strength_steering_control(self):
        result = self.probe.causal.steer(layers=[0], strength=0.).run(
            self.inputs, direction=torch.randn(6), metric=self.metric)
        torch.testing.assert_close(result.tensors["effect"], torch.zeros(1, 2))

    def test_attribution_predicts_small_patch(self):
        near = {**self.inputs, "image_tokens": self.inputs["image_tokens"] + 1e-4}
        estimated = self.probe.causal.attribute(layers=[0]).run(self.inputs, source=near, metric=self.metric)
        exact = self.probe.causal.patch(layers=[0]).run(self.inputs, source=near, metric=self.metric)
        torch.testing.assert_close(estimated.tensors["estimated_effect"][0], exact.tensors["effect"].sum(), atol=1e-5, rtol=0.02)

    def test_knockout_and_edge_completeness(self):
        result = self.probe.causal.knockout(layers=[0]).run(self.inputs, metric=self.metric)
        def edit(x):
            y = x.clone()
            y[:, :, 2:, :2] = -torch.inf
            return y
        manual = self.probe.adapter.run(self.inputs, interventions={"scores.0": edit})
        torch.testing.assert_close(result.tensors["intervention_score"][0],
                                   self.metric.score(manual.logits, self.probe._layout(self.inputs, manual.logits)))
        result = self.probe.causal.eap_ig(steps=64).run(self.bad, source=self.inputs, metric=self.metric)
        self.assertLess(result.tensors["completeness_error"].abs().item(), 1e-3)

    def test_attention_noop_controls_and_targeted_temperature(self):
        for method in (self.probe.attention.reweight(layers=[0], weight=1.),
                       self.probe.attention.temperature(layers=[0], temperature=1.)):
            result = method.run(self.inputs, metric=self.metric)
            torch.testing.assert_close(result.tensors["effect"], torch.zeros(1, 2), atol=1e-6, rtol=0)
        result = self.probe.attention.temperature(layers=[0], queries=[4], temperature=0.5).run(self.inputs, metric=self.metric)
        def edit(x):
            y = x.clone()
            y[:, :, 4] = y[:, :, 4] / 0.5
            return y
        manual = self.probe.adapter.run(self.inputs, interventions={"scores.0": edit})
        torch.testing.assert_close(result.tensors["intervention_score"][0],
                                   self.metric.score(manual.logits, self.probe._layout(self.inputs, manual.logits)))

    def test_attention_observation_and_frozen_gradient_capture(self):
        profile = self.probe.attention.profile().run(self.inputs)
        torch.testing.assert_close(profile.tensors["group_mass"].sum(-1), torch.ones(2, 2, 1, 5))
        rollout = self.probe.attention.rollout().run(self.inputs)
        torch.testing.assert_close(rollout.tensors["rollout"].sum(-1), torch.ones(2, 5))
        self.model.requires_grad_(False)
        result = self.probe.attention.relevance().run(self.inputs, metric=self.metric)
        self.assertTrue(torch.isfinite(result.tensors["relevance"]).all())
        self.probe.attention.head_logits(layers=[1]).run(self.inputs)
        with self.assertRaisesRegex(ValueError, "consecutive"):
            self.probe.attention.rollout(layers=[1]).run(self.inputs)

    def test_unsupported_edit_rejected_before_execution(self):
        self.probe.spec.editable_attention = set()
        self.assertFalse(self.probe.describe()["methods"]["attention.reweight"]["available"])
        with self.assertRaises(CapabilityError):
            self.probe.attention.reweight()
        self.probe.spec.attention_scores.clear()
        with self.assertRaises(CapabilityError):
            self.probe.causal.knockout()

    def test_sequence_scores_shift_and_mask_visual_sentinels(self):
        layout = self.probe.spec.layout(self.inputs)
        layout.token_ids[:, :2] = -200
        answer = torch.zeros_like(layout.valid)
        answer[:, 3:] = True
        logits = self.probe.adapter.run(self.inputs).logits
        scores = SequenceLogProb(answer).score(logits, layout)
        expected = logits[:, 2:4].log_softmax(-1).gather(-1, self.inputs["input_ids"][:, 1:, None]).sum((1, 2))
        torch.testing.assert_close(scores, expected)
        result = self.probe.causal.patch(layers=[0]).run(
            ProbeInputs(self.inputs, layout), source=ProbeInputs(self.inputs, layout), metric=SequenceLogProb(answer))
        torch.testing.assert_close(result.tensors["effect"], torch.zeros(1, 2))

    def test_prepare_uses_processor_without_mutating_inputs(self):
        calls = []
        def processor(**kwargs):
            calls.append(kwargs)
            return dict(self.inputs)
        probe = Prober(self.model, processor=processor)
        batch = probe.prepare(prompt="formatted", image="image", padding=True, device="cpu")
        self.assertEqual(calls[0], {"text": "formatted", "images": "image", "padding": True, "return_tensors": "pt"})
        probe.lens.logit().run(batch)
        self.assertEqual(set(self.inputs), {"input_ids", "image_tokens"})

    def test_custom_config_does_not_imply_hf_forward_arguments(self):
        from types import SimpleNamespace
        self.model.config = SimpleNamespace(name="custom-model")
        # This forward intentionally has no use_cache keyword.
        self.probe.lens.logit(layers=[0]).run(self.inputs)

    def test_hooks_modes_and_result_serialization(self):
        self.model.layers[0].eval()
        states = [m.training for m in self.model.modules()]
        def failure(logits):
            raise RuntimeError("metric failed")
        with self.assertRaisesRegex(RuntimeError, "metric failed"):
            self.probe.causal.path(senders=[(0, 0)]).run(self.inputs, donor=self.inputs, metric=failure)
        self.assertEqual(states, [m.training for m in self.model.modules()])
        self.assertTrue(all(not m._forward_hooks and not m._forward_pre_hooks for m in self.model.modules()))
        result = self.probe.lens.logit().run(self.inputs)
        with tempfile.TemporaryDirectory() as path:
            result.save(path)
            self.assertTrue((Path(path) / "metadata.json").exists())

    def test_standalone_readout_uses_eval_and_restores_dropout_mode(self):
        self.model.lm_head = torch.nn.Sequential(torch.nn.Dropout(0.9), torch.nn.Linear(6, 12))
        self.model.train()
        probe = Prober(self.model)
        expected = probe.adapter.run(self.inputs).logits[:, -1]
        actual = probe.lens.logit(layers=[1]).run(self.inputs).tensors["logits"][0]
        torch.testing.assert_close(actual, expected)
        self.assertTrue(self.model.lm_head[0].training)
        probe.lens.tuned(layers=[0]).fit(self.inputs, steps=2)
        self.assertTrue(self.model.lm_head[0].training)


if __name__ == "__main__":
    unittest.main()

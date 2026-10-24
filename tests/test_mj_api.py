"""Native model-bound MJ calibration and artifact regression contracts."""
from pathlib import Path
import tempfile
import unittest

import torch

from vlm_probing import MJBatch, Prober, decode_mj_codes, fit_visual_readout
from vlm_probing.adapters import prepare_janus_generation
from vlm_probing.lenses import LinearizedVisualReadout
from test_mj_janus import HAS_V5, tiny_janus


BINDING = {"model_revision": "random-fixture-seed-312", "tokenizer_id": "fixture-text64",
           "calibration_id": "independent-training-images", "codebook_id": "fixture-vq32"}


@unittest.skipUnless(HAS_V5, "MJ native integration requires Transformers 5.3.x")
class MJAPITests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(312)
        self.model = tiny_janus()
        self.probe = Prober(self.model)
        self.inputs = []
        for seed in (7, 8, 9):
            self.inputs.append({"input_ids": torch.tensor([[1, 63, 63, 63, 63, 2, seed, 4]]),
                                "pixel_values": torch.rand(1, 3, 16, 16)})

    def text_lens(self, **kwargs):
        return self.probe.lens.mj(layers=[0, 1], source="visual", target="text",
                                  binding=BINDING, **kwargs)

    def generation_batches(self):
        return [prepare_janus_generation(self.probe, {"input_ids": torch.tensor([[1, 2, 4]])},
                                          torch.tensor([[1, 2, 3, 4]]) + offset)
                for offset in range(3)]

    def visual_readout(self, batches):
        return fit_visual_readout(self.probe, batches,
            binding={"model_id": "fixture-Janus", "readout_id": "native-generation-head",
                     "calibration_id": "training-visual-output-positions"})

    def test_unequal_batching_and_resume_equal_joint_prompt_mean(self):
        a, b, c = self.inputs
        joined = {key: torch.cat([b[key], c[key]]) for key in a}
        reference = self.text_lens().fit_batches([a, joined], dim_batch=5)
        singleton = self.text_lens().fit_batches([a, b, c], dim_batch=5)
        for layer in reference.layers:
            torch.testing.assert_close(reference.kernels[layer].jacobian,
                                       singleton.kernels[layer].jacobian, atol=2e-6, rtol=2e-5)
        self.assertEqual(reference.calibration_report["split_half_prompt_counts"], {0: 2, 1: 1})
        with tempfile.TemporaryDirectory() as directory:
            self.text_lens().fit_batches([a], directory=directory, dim_batch=5)
            resumed = self.text_lens().fit_batches([a, joined], directory=directory,
                                                  dim_batch=5, resume=True)
            for layer in reference.layers:
                torch.testing.assert_close(reference.kernels[layer].jacobian,
                                           resumed.kernels[layer].jacobian)
            with self.assertRaisesRegex(ValueError, "shorter"):
                self.text_lens().fit_batches([a], directory=directory, dim_batch=5, resume=True)
            with self.assertRaisesRegex(ValueError, "already exists"):
                self.text_lens().fit_batches([a, joined], directory=directory, dim_batch=5)

    def test_save_load_context_and_model_binding_are_enforced(self):
        fitted = self.text_lens().fit_batches(self.inputs, dim_batch=8)
        expected = fitted.run(self.inputs[0])
        with tempfile.TemporaryDirectory() as directory:
            fitted.save(directory)
            loaded = self.text_lens().load(directory)
            torch.testing.assert_close(expected.tensors["logits"], loaded.run(self.inputs[0]).tensors["logits"])
            wrong = self.probe.lens.mj(layers=[0, 1], binding={**BINDING, "model_revision": "other-model"})
            with self.assertRaisesRegex(ValueError, "binding"):
                wrong.load(directory)
        with self.assertRaisesRegex(ValueError, "context mismatch"):
            fitted.run(MJBatch(self.inputs[0], context="image_output"))
        with self.assertRaisesRegex(ValueError, "source-modality"):
            fitted.run(self.inputs[0], tokens="text")

    def test_binding_is_validated_before_expensive_capture(self):
        method = self.probe.lens.mj(layers=[0])
        calls = []
        handle = self.model.register_forward_pre_hook(lambda *_: calls.append(1))
        try:
            with tempfile.TemporaryDirectory() as directory:
                with self.assertRaisesRegex(ValueError, "immutable binding"):
                    method.fit_batches(self.inputs, directory=directory)
        finally:
            handle.remove()
        self.assertEqual(calls, [])

    def test_failure_restores_modes_flags_hooks_and_existing_fit(self):
        method = self.text_lens().fit(self.inputs[0])
        before = method.kernels[0].jacobian.clone()
        self.model.train()
        self.model.model.language_model.layers[0].eval()
        next(self.model.parameters()).requires_grad_(False)
        modes = [m.training for m in self.model.modules()]
        flags = [p.requires_grad for p in self.model.parameters()]
        hooks = [(tuple(m._forward_hooks), tuple(m._forward_pre_hooks)) for m in self.model.modules()]
        bad = MJBatch(self.inputs[0], target_masks={"text": torch.zeros(1, 8, dtype=torch.bool)})
        with self.assertRaisesRegex(ValueError, "retain"):
            method.fit_batches([bad])
        self.assertEqual(modes, [m.training for m in self.model.modules()])
        self.assertEqual(flags, [p.requires_grad for p in self.model.parameters()])
        self.assertEqual(hooks, [(tuple(m._forward_hooks), tuple(m._forward_pre_hooks))
                                 for m in self.model.modules()])
        self.assertTrue(all(p.grad is None for p in self.model.parameters()))
        torch.testing.assert_close(before, method.kernels[0].jacobian)

    def test_visual_channel_calibration_all_pairs_and_native_decode(self):
        batches = self.generation_batches()
        readout = self.visual_readout(batches)
        self.assertEqual(readout.n_samples, 9)  # three positions, three rollouts
        expected = []
        for source in ("text", "visual"):
            method = self.probe.lens.mj(layers=[0, 1], source=source, target="visual",
                                      visual_readout=readout, binding=BINDING)
            method.fit_batches(batches, dim_batch=8)
            result = method.run(batches[0])
            self.assertEqual(result.metadata["readout_kind"], "visual_codebook")
            self.assertEqual(result.tensors["logits"].shape[-1], 32)
            if source == "visual":
                images, codes = decode_mj_codes(result, grid=(2, 2), decoder=self.probe.spec.mj.decode_visual_codes)
                self.assertEqual(tuple(codes.shape), (2, 1, 2, 2))
                self.assertEqual(tuple(images.shape[:2]), (2, 1))
                with self.assertRaisesRegex(ValueError, "grid"):
                    decode_mj_codes(result, grid=(1, 2), decoder=self.probe.spec.mj.decode_visual_codes)
                with tempfile.TemporaryDirectory() as directory:
                    method.save(directory)
                    restored = self.probe.lens.mj(layers=[0, 1], source=source, target="visual",
                        visual_readout=readout, binding=BINDING).load(directory)
                    torch.testing.assert_close(result.tensors["logits"], restored.run(batches[0]).tensors["logits"])
            else:
                with self.assertRaisesRegex(ValueError, "visual-to-visual"):
                    decode_mj_codes(result, grid=(2, 2), decoder=self.probe.spec.mj.decode_visual_codes)
            expected.append(method.calibration_report["pair_counts"])
        self.assertNotEqual(expected[0], expected[1])
        text = self.probe.lens.mj(layers=[0], source="text", target="text", binding=BINDING)
        self.assertEqual(text.fit(self.inputs[0]).run(self.inputs[1]).tensors["logits"].shape[-1], 64)

    def test_visual_source_metadata_and_native_head_roles_are_checked(self):
        batches = self.generation_batches()
        readout = self.visual_readout(batches)
        with self.assertRaisesRegex(ValueError, "non-text-output"):
            self.probe.lens.mj(layers=[0], target="visual", visual_readout=readout).fit(self.inputs[0])
        head = self.model.model.generation_head
        wrong = LinearizedVisualReadout(self.probe.spec.mj.norm, head.proj_out,
                                       torch.nn.GELU(approximate="tanh"), head.vision_head)
        with torch.no_grad():
            trace = self.probe._run(batches[0].inputs, capture=[self.probe.spec.mj.final])
            wrong.fit(self.probe.spec.mj.norm(trace.activations[self.probe.spec.mj.final]))
        with self.assertRaisesRegex(ValueError, "exact native"):
            self.probe.lens.mj(layers=[0], target="visual", visual_readout=wrong)

    def test_recalibrating_shared_visual_readout_invalidates_bound_artifact_identity(self):
        batches = self.generation_batches()
        readout = self.visual_readout(batches)
        method = self.probe.lens.mj(layers=[0], target="visual", visual_readout=readout,
                                    binding=BINDING).fit_batches(batches)
        readout.update(torch.randn(5, 16))
        for call in (lambda: method.run(batches[0]), lambda: method.dictionary(0)):
            with self.assertRaisesRegex(ValueError, "binding changed"):
                call()
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "binding changed"):
                method.save(directory)


if __name__ == "__main__":
    unittest.main()

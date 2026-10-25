"""Jacobian calibration diagnostics, replacement warnings, and dataset weighting."""
import importlib.util
import json
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest.mock import Mock, patch

import torch

from vlmprobe import Prober
from vlmprobe.lenses import JacobianLens
from vlmprobe.training import fit_jacobian_lenses


BINDING = {"model_id": "tiny-gpt2@1", "tokenizer_id": "integer-vocab@1",
           "readout_id": "native-final-norm-and-head", "calibration_id": "disjoint-prompts@1"}
HAS_TRANSFORMERS = importlib.util.find_spec("transformers") is not None


def native_probe():
    from transformers import GPT2Config, GPT2LMHeadModel

    torch.manual_seed(47)
    model = GPT2LMHeadModel(GPT2Config(vocab_size=32, n_positions=40, n_embd=4,
        n_layer=2, n_head=1, resid_pdrop=0, embd_pdrop=0, attn_pdrop=0,
        attn_implementation="eager"))
    return Prober(model)


class JacobianContractTests(unittest.TestCase):
    def test_mixed_short_rows_are_named_before_downstream_and_never_dropped(self):
        valid = torch.arange(20)[None] < torch.tensor([20, 3])[:, None]
        downstream = Mock(side_effect=AssertionError("must not run"))
        with self.assertRaises(ValueError) as caught:
            JacobianLens(lambda x: x).fit(torch.ones(2, 20, 2), downstream, valid_mask=valid)
        message = str(caught.exception)
        for text in ("failing rows=[1]", "valid_counts=[20, 3]",
                     "retained_source_counts=[3, 0]", "retained_target_counts=[3, 0]",
                     "skip_first=16, exclude_last=True", "at least 18", "No rows were dropped"):
            self.assertIn(text, message)
        downstream.assert_not_called()

    def test_default_threshold_preserves_reference_exclusions(self):
        for length in (1, 16, 17):
            with self.subTest(length=length), self.assertRaisesRegex(ValueError, "at least 18"):
                JacobianLens(lambda x: x).fit(torch.ones(1, length, 2), lambda x: x)
        lens = JacobianLens(lambda x: x).fit(torch.ones(1, 18, 2), lambda x: x)
        self.assertEqual(lens.calibration["source_positions"], [[16]])
        self.assertEqual(lens.calibration["target_positions"], [[16]])
        torch.testing.assert_close(lens.jacobian, torch.eye(2), rtol=0, atol=0)

    def test_long_empty_and_disjoint_masks_report_actual_counts(self):
        states = torch.ones(2, 20, 2)
        for name in ("valid_mask", "source_mask", "target_mask"):
            for all_empty in (False, True):
                with self.subTest(name=name, all_empty=all_empty):
                    mask = torch.ones(2, 20, dtype=torch.bool)
                    mask[1] = False
                    if all_empty:
                        mask[0] = False
                    with self.assertRaises(ValueError) as caught:
                        JacobianLens(lambda x: x).fit(states, lambda x: x, **{name: mask})
                    message = str(caught.exception)
                    self.assertIn("failing rows=[0, 1]" if all_empty else "failing rows=[1]", message)
                    self.assertIn("retained_source_counts=", message)
                    self.assertIn("retained_target_counts=", message)
        first_only = torch.zeros(2, 20, dtype=torch.bool)
        first_only[:, 0] = True
        with self.assertRaisesRegex(ValueError, r"retained_source_counts=\[0, 0\]"):
            JacobianLens(lambda x: x).fit(states, lambda x: x, source_mask=first_only)

    def test_replacement_warns_and_remains_last_batch_only(self):
        lens = JacobianLens(lambda x: x)
        first, last = torch.ones(1, 3, 2), torch.full((1, 3, 2), 2.)
        with warnings.catch_warnings(record=True) as first_warnings:
            warnings.simplefilter("always")
            lens.fit(first, lambda x: x.square(), skip_first=0, exclude_last=False)
        self.assertEqual(first_warnings, [])
        with self.assertWarnsRegex(RuntimeWarning, "vlmprobe.training.fit_jacobian_lenses"):
            lens.fit(last, lambda x: x.square(), skip_first=0, exclude_last=False)
        torch.testing.assert_close(lens.jacobian, 4 * torch.eye(2), rtol=0, atol=0)
        self.assertEqual(lens.calibration["n_prompts"], 1)
        joint = JacobianLens(lambda x: x).fit(torch.cat([first, last]),
            lambda x: x.square(), skip_first=0, exclude_last=False)
        torch.testing.assert_close(joint.jacobian, 3 * torch.eye(2), rtol=0, atol=0)

    def test_kernel_replacement_warning_precedes_downstream(self):
        lens = JacobianLens(lambda x: x).fit(torch.ones(1, 2, 2), lambda x: x,
            skip_first=0, exclude_last=False)
        old = lens.jacobian.clone()
        downstream = Mock(side_effect=AssertionError("must not run"))
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            with self.assertRaisesRegex(RuntimeWarning, "replaces the existing estimate"):
                lens.fit(torch.ones(1, 2, 2), downstream, skip_first=0, exclude_last=False)
        downstream.assert_not_called()
        torch.testing.assert_close(lens.jacobian, old, rtol=0, atol=0)

    @unittest.skipUnless(HAS_TRANSFORMERS, "optional transformers extra not installed")
    def test_public_selectors_report_empty_retained_sources_before_gradients(self):
        probe = native_probe()
        inputs = {"input_ids": torch.arange(20)[None]}
        for tokens in ("last_prompt", torch.zeros(1, 20, dtype=torch.bool)):
            with self.subTest(tokens=type(tokens).__name__):
                method = probe.lens.jacobian(layers=[0], tokens=tokens)
                with patch.object(probe, "_run", wraps=probe._run) as run:
                    with self.assertRaisesRegex(ValueError, "source_selector=") as caught:
                        method.fit(inputs)
                self.assertIn("failing rows=[0]", str(caught.exception))
                self.assertIn("retained_source_counts=[0]", str(caught.exception))
                self.assertFalse(any(call.kwargs.get("grad") for call in run.call_args_list))
                self.assertEqual(method.kernels, {})

    @unittest.skipUnless(HAS_TRANSFORMERS, "optional transformers extra not installed")
    def test_public_refit_warns_once_for_multiple_layers_before_capture(self):
        probe = native_probe()
        first = {"input_ids": torch.tensor([[1, 2, 3, 4]])}
        last = {"input_ids": torch.tensor([[5, 6, 7, 8]])}
        method = probe.lens.jacobian(layers=[0, 1], tokens="all", binding=BINDING)
        with warnings.catch_warnings(record=True) as initial:
            warnings.simplefilter("always")
            method.fit(first, skip_first=0, exclude_last=False)
        self.assertFalse(any(issubclass(item.category, RuntimeWarning) for item in initial))
        with patch.object(probe, "_run", side_effect=AssertionError("capture must not run")) as run:
            with warnings.catch_warnings():
                warnings.simplefilter("error", RuntimeWarning)
                with self.assertRaises(RuntimeWarning):
                    method.fit(last, skip_first=0, exclude_last=False)
            run.assert_not_called()
        with warnings.catch_warnings(record=True) as repeated:
            warnings.simplefilter("always")
            method.fit(last, skip_first=0, exclude_last=False)
        runtime = [item for item in repeated if issubclass(item.category, RuntimeWarning)]
        self.assertEqual(len(runtime), 1)
        fresh = probe.lens.jacobian(layers=[0, 1], tokens="all", binding=BINDING)
        fresh.fit(last, skip_first=0, exclude_last=False)
        for layer in method.layers:
            torch.testing.assert_close(method.kernels[layer].jacobian,
                fresh.kernels[layer].jacobian, rtol=0, atol=0)

    @unittest.skipUnless(HAS_TRANSFORMERS, "optional transformers extra not installed")
    def test_streaming_matches_joint_native_estimate_with_unequal_batches(self):
        probe = native_probe()
        inputs = {"input_ids": torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])}
        batches = [{"input_ids": inputs["input_ids"][:2]}, {"input_ids": inputs["input_ids"][2:]}]
        joint = probe.lens.jacobian(layers=[0, 1], tokens="all", binding=BINDING)
        joint.fit(inputs, skip_first=0, exclude_last=False)
        with tempfile.TemporaryDirectory() as directory:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                streamed, report = fit_jacobian_lenses(probe, batches, layers=[0, 1],
                    binding=BINDING, directory=directory, tokens="all", dim_batch=2,
                    skip_first=0, exclude_last=False, checkpoint_every=1)
            self.assertFalse(any(issubclass(item.category, RuntimeWarning) for item in caught))
            self.assertEqual((report["prompts"], report["batches"]), (3, 2))
            for layer in joint.layers:
                torch.testing.assert_close(streamed.kernels[layer].jacobian, joint.kernels[layer].jacobian)
            loaded = probe.lens.jacobian(layers=[0, 1], tokens="all", binding=BINDING).load(directory)
            for layer in joint.layers:
                torch.testing.assert_close(loaded.kernels[layer].jacobian, streamed.kernels[layer].jacobian)

    @unittest.skipUnless(HAS_TRANSFORMERS, "optional transformers extra not installed")
    def test_streaming_errors_include_batch_selector_rows_and_restore_model(self):
        for tokens in ("all", "last_prompt", torch.zeros(2, 20, dtype=torch.bool)):
            with self.subTest(tokens=type(tokens).__name__), tempfile.TemporaryDirectory() as directory:
                probe = native_probe()
                probe.model.train()
                flags = [p.requires_grad for p in probe.model.parameters()]
                inputs = {"input_ids": torch.arange(20)[None].expand(2, -1),
                          "attention_mask": (torch.arange(20)[None] < torch.tensor([20, 3])[:, None]).long()}
                with patch("torch.autograd.grad", side_effect=AssertionError("VJP must not run")):
                    with self.assertRaises(ValueError) as caught:
                        fit_jacobian_lenses(probe, [inputs], layers=[0], binding=BINDING,
                            directory=directory, tokens=tokens, resume=False)
                message = str(caught.exception)
                for text in ("batch=0", "source_selector=", "failing rows=", "valid_counts=[20, 3]",
                             "retained_source_counts=", "retained_target_counts=", "skip_first=16"):
                    self.assertIn(text, message)
                self.assertTrue(probe.model.training)
                self.assertEqual(flags, [p.requires_grad for p in probe.model.parameters()])
                self.assertFalse((Path(directory) / "calibration_state.pt").exists())

    @unittest.skipUnless(HAS_TRANSFORMERS, "optional transformers extra not installed")
    def test_streaming_resume_accumulates_new_prompts_and_complete_replay_does_no_work(self):
        probe = native_probe()
        batches = [{"input_ids": torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])},
                   {"input_ids": torch.tensor([[9, 10, 11, 12]])}]
        options = dict(layers=[0, 1], binding=BINDING, tokens="all", dim_batch=2,
                       skip_first=0, exclude_last=False, checkpoint_every=1)
        with tempfile.TemporaryDirectory() as directory:
            _, first = fit_jacobian_lenses(probe, batches[:1], directory=directory, **options)
            self.assertFalse(first["resumed"])
            self.assertEqual((first["new_prompts"], first["new_batches"]), (2, 1))
            resumed, report = fit_jacobian_lenses(probe, batches, directory=directory, **options)
            self.assertTrue(report["resumed"])
            self.assertEqual((report["resumed_prompts"], report["resumed_batches"]), (2, 1))
            self.assertEqual((report["new_prompts"], report["new_batches"]), (1, 1))
            joint = probe.lens.jacobian(layers=[0, 1], tokens="all", binding=BINDING)
            joint.fit({"input_ids": torch.cat([b["input_ids"] for b in batches])},
                      skip_first=0, exclude_last=False)
            for layer in joint.layers:
                torch.testing.assert_close(resumed.kernels[layer].jacobian, joint.kernels[layer].jacobian)
            with patch.object(probe, "_run", side_effect=AssertionError("completed replay must not capture")):
                replay, replay_report = fit_jacobian_lenses(probe, batches, directory=directory, **options)
            self.assertEqual((replay_report["new_prompts"], replay_report["new_batches"]), (0, 0))
            for layer in joint.layers:
                torch.testing.assert_close(replay.kernels[layer].jacobian, joint.kernels[layer].jacobian)

    @unittest.skipUnless(HAS_TRANSFORMERS, "optional transformers extra not installed")
    def test_streaming_short_resume_rejects_exhausted_iterators_and_preserves_artifacts(self):
        probe = native_probe()
        batches = [{"input_ids": torch.tensor([[1, 2, 3, 4]])},
                   {"input_ids": torch.tensor([[5, 6, 7, 8]])}]
        options = dict(layers=[0], binding=BINDING, tokens="all", dim_batch=2,
                       skip_first=0, exclude_last=False, checkpoint_every=1)
        with tempfile.TemporaryDirectory() as directory:
            fit_jacobian_lenses(probe, batches, directory=directory, **options)
            before = {p.name: p.read_bytes() for p in Path(directory).iterdir()}
            probe.model.train()
            probe.model.transformer.h[0].eval()
            next(probe.model.parameters()).requires_grad_(False)
            modes = [m.training for m in probe.model.modules()]
            flags = [p.requires_grad for p in probe.model.parameters()]
            exhausted = iter(batches)
            list(exhausted)
            for incomplete in ([], batches[:1], exhausted):
                with self.subTest(kind=type(incomplete).__name__):
                    with patch.object(probe, "_run", side_effect=AssertionError("short prefix must not capture")):
                        with self.assertRaisesRegex(ValueError, "shorter than the recorded calibration prefix"):
                            fit_jacobian_lenses(probe, incomplete, directory=directory, **options)
                    self.assertEqual(before, {p.name: p.read_bytes() for p in Path(directory).iterdir()})
                    self.assertEqual(modes, [m.training for m in probe.model.modules()])
                    self.assertEqual(flags, [p.requires_grad for p in probe.model.parameters()])

    @unittest.skipUnless(HAS_TRANSFORMERS, "optional transformers extra not installed")
    def test_streaming_tensor_selector_is_serializable_and_resumes_by_values(self):
        probe = native_probe()
        inputs = {"input_ids": torch.tensor([[1, 2, 3, 4]])}
        selector = torch.tensor([[True, True, False, True]])
        options = dict(layers=[0], binding=BINDING, dim_batch=2,
                       skip_first=0, exclude_last=False, checkpoint_every=1)
        with tempfile.TemporaryDirectory() as directory:
            fitted, report = fit_jacobian_lenses(probe, [inputs], tokens=selector,
                directory=directory, **options)
            recorded = json.loads((Path(directory) / "status.json").read_text())
            self.assertEqual(recorded["settings"]["tokens"], {
                "kind": "tensor", "dtype": "torch.bool", "shape": [1, 4],
                "values": [[True, True, False, True]],
            })
            self.assertEqual(fitted.kernels[0].calibration["source_selector"],
                             report["settings"]["tokens"])
            # Older versions checkpointed the valid estimate before failing
            # to serialize the Tensor into status.json. Recover that work.
            state_path = Path(directory) / "calibration_state.pt"
            legacy = torch.load(state_path, weights_only=True)
            legacy["settings"]["tokens"] = selector.clone()
            torch.save(legacy, state_path)
            # An independently reconstructed mask has the same immutable
            # identity, while changing a retained source changes that identity.
            with patch.object(probe, "_run", side_effect=AssertionError("resume must not capture")):
                replay, replay_report = fit_jacobian_lenses(probe, [inputs], tokens=selector.clone(),
                    directory=directory, **options)
                altered = selector.clone()
                altered[0, 0] = False
                with self.assertRaisesRegex(ValueError, "resume settings/binding differ"):
                    fit_jacobian_lenses(probe, [inputs], tokens=altered, directory=directory, **options)
            self.assertEqual(replay_report["new_prompts"], 0)
            torch.testing.assert_close(replay.kernels[0].jacobian, fitted.kernels[0].jacobian)
            self.assertEqual(torch.load(state_path, weights_only=True)["settings"]["tokens"],
                             recorded["settings"]["tokens"])

    @unittest.skipUnless(HAS_TRANSFORMERS, "optional transformers extra not installed")
    def test_generic_fit_batches_points_to_existing_dataset_estimator(self):
        probe = native_probe()
        method = probe.lens.jacobian(layers=[0])
        with patch.object(probe, "_run", side_effect=AssertionError("must not run")):
            with self.assertRaisesRegex(ValueError, "vlmprobe.training.fit_jacobian_lenses"):
                method.fit_batches([], [], "unused")


if __name__ == "__main__":
    unittest.main()

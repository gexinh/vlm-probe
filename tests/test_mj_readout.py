"""Small analytic controls for MJ-Lens readouts, not a paper-scale replication."""
from pathlib import Path
import tempfile
import unittest

import torch
from torch import nn
from torch.nn import functional as F

from vlm_probing.lenses.mj_readout import LinearizedVisualReadout, modality_index


BINDING = {"model_id": "toy-unified@1", "readout_id": "visual-codebook@1",
           "calibration_id": "visual-rollouts@1"}


class VisualReadoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def make_readout(self, approximate="none", binding=None):
        torch.manual_seed(11)
        return LinearizedVisualReadout(nn.LayerNorm(3, dtype=torch.float64),
                                       nn.Linear(3, 5, dtype=torch.float64),
                                       nn.GELU(approximate=approximate),
                                       nn.Linear(5, 4, dtype=torch.float64),
                                       binding=BINDING if binding is None else binding)

    def test_exact_and_tanh_slope_match_native_autograd(self):
        states = torch.tensor([[-3., 1., 4.], [2., 0., -1.]], dtype=torch.float64)
        for approximate in ("none", "tanh"):
            readout = self.make_readout(approximate)
            normalized = readout.norm(states)
            preactivation = readout.first_projection(normalized).detach().requires_grad_(True)
            slope = torch.autograd.grad(readout.activation(preactivation).sum(), preactivation)[0]
            readout.fit(normalized)
            torch.testing.assert_close(readout.mean_slope, slope.mean(0))
            expected = (readout.last_projection.weight * slope.mean(0)) @ readout.first_projection.weight
            torch.testing.assert_close(readout.weight, expected)
            torch.testing.assert_close(readout.dictionary_weight, expected)

    def test_mean_slope_is_not_derivative_at_mean(self):
        first, last = nn.Linear(1, 1, dtype=torch.float64), nn.Linear(1, 1, dtype=torch.float64)
        with torch.no_grad():
            first.weight.fill_(1)
            first.bias.zero_()
            last.weight.fill_(1)
        readout = LinearizedVisualReadout(nn.Identity(), first, nn.GELU(), last)
        states = torch.tensor([[0.], [3.]], dtype=torch.float64)
        readout.fit(states)
        point = states.mean(0).requires_grad_()
        at_mean = torch.autograd.grad(F.gelu(point).sum(), point)[0]
        self.assertGreater(float((readout.mean_slope - at_mean).abs().item()), .2)

    def test_native_transformers_gelu_uses_exact_derivative_and_original_object(self):
        try:
            from transformers.activations import GELUActivation
        except ImportError:
            self.skipTest("Transformers is optional")
        for python_variant in (False, True):
            reference = self.make_readout()
            activation = GELUActivation(use_gelu_python=python_variant)
            readout = LinearizedVisualReadout(reference.norm, reference.first_projection,
                                            activation, reference.last_projection)
            states = torch.randn(2, 3, dtype=torch.float64)
            reference.fit(states)
            readout.fit(states)
            self.assertIs(readout.activation, activation)
            torch.testing.assert_close(readout.mean_slope, reference.mean_slope)
            torch.testing.assert_close(readout.native(states), reference.native(states))

    def test_activation_configuration_changes_are_rejected(self):
        readout = self.make_readout().fit(torch.randn(2, 3, dtype=torch.float64))
        readout.activation.approximate = "tanh"
        with self.assertRaisesRegex(ValueError, "changed"):
            readout.update(torch.randn(2, 3, dtype=torch.float64))
        with self.assertRaisesRegex(ValueError, "recalibrate"):
            readout(torch.randn(2, 3, dtype=torch.float64))
        readout.fit(torch.randn(2, 3, dtype=torch.float64))
        self.assertEqual(readout._gelu_approximate, "tanh")

    def test_native_parameter_changes_require_recalibration_at_readout(self):
        readout = self.make_readout().fit(torch.randn(2, 3, dtype=torch.float64))
        with torch.no_grad():
            readout.last_projection.weight.add_(1)
        with self.assertRaisesRegex(ValueError, "recalibrate"):
            _ = readout.weight
        with self.assertRaisesRegex(ValueError, "recalibrate"):
            readout(torch.randn(2, 3, dtype=torch.float64))

    def test_readout_drops_both_biases_and_keeps_norm_gradient(self):
        first, last = nn.Linear(2, 2, dtype=torch.float64), nn.Linear(2, 2, dtype=torch.float64)
        with torch.no_grad():
            first.weight.copy_(torch.eye(2))
            first.bias.fill_(2)
            last.weight.copy_(torch.eye(2) * 3)
            last.bias.fill_(5)
        readout = LinearizedVisualReadout(lambda x: x * 2, first, nn.GELU(), last)
        readout.fit(torch.tensor([[1., 2.]], dtype=torch.float64))
        states = torch.tensor([[0., 1.]], dtype=torch.float64, requires_grad=True)
        actual = readout(states)
        torch.testing.assert_close(actual, F.linear(states * 2, readout.weight))
        torch.testing.assert_close(readout(torch.zeros_like(states)), torch.zeros_like(states))
        self.assertFalse(torch.allclose(actual, readout.native(states)))
        gradient = torch.autograd.grad(actual.sum(), states)[0]
        torch.testing.assert_close(gradient, readout.weight.sum(0).unsqueeze(0) * 2)

    def test_streaming_is_position_weighted_and_fit_replaces(self):
        readout, streamed = self.make_readout(), self.make_readout()
        states = torch.randn(3, 4, 3, dtype=torch.float64)
        mask = torch.tensor([[True, True, False, False], [True, False, False, False],
                             [True, True, True, True]])
        readout.fit(states, mask=mask)
        streamed.update(states[:1], mask=mask[:1]).update(states[1:], mask=mask[1:])
        self.assertEqual(readout.n_samples, 7)
        self.assertEqual(streamed.n_samples, 7)
        torch.testing.assert_close(readout.weight, streamed.weight)
        streamed.fit(states[:1], mask=mask[:1])
        self.assertEqual(streamed.n_samples, 2)

    def test_effective_weight_is_cached_and_invalidated_on_updates(self):
        readout = self.make_readout().fit(torch.randn(2, 3, dtype=torch.float64))
        cached = readout.weight
        self.assertEqual(cached.data_ptr(), readout.weight.data_ptr())
        self.assertEqual(cached.data_ptr(), readout.dictionary_weight.data_ptr())
        readout(torch.randn(2, 3, dtype=torch.float64))
        self.assertEqual(cached.data_ptr(), readout.weight.data_ptr())
        readout.update(torch.randn(5, 3, dtype=torch.float64) + 2)
        updated = readout.weight
        self.assertNotEqual(cached.data_ptr(), updated.data_ptr())
        expected = ((readout.last_projection.weight * readout.mean_slope)
                    @ readout.first_projection.weight)
        torch.testing.assert_close(updated, expected)
        with self.assertRaisesRegex(ValueError, "at least one"):
            readout.fit(torch.ones(2, 3), mask=torch.tensor([False, False]))
        self.assertEqual(updated.data_ptr(), readout.weight.data_ptr())
        readout.fit(torch.randn(1, 3, dtype=torch.float64))
        self.assertNotEqual(updated.data_ptr(), readout.weight.data_ptr())

    def test_fit_does_not_mutate_native_modes_parameters_or_gradients(self):
        readout = self.make_readout()
        modules = (readout.norm, readout.first_projection, readout.last_projection)
        before = []
        for module in modules:
            module.train()
            for parameter in module.parameters():
                parameter.grad = torch.full_like(parameter, 7)
                before.append((parameter, parameter.detach().clone(), parameter.grad.clone()))
        raw = torch.randn(2, 3, 3, dtype=torch.float64, requires_grad=True)
        with torch.inference_mode():
            readout.fit(readout.norm(raw))
        self.assertIsNone(raw.grad)
        for module in modules:
            self.assertTrue(module.training)
        for parameter, value, gradient in before:
            torch.testing.assert_close(parameter, value)
            torch.testing.assert_close(parameter.grad, gradient)

    def test_save_load_checks_binding_native_identity_and_weight_consistency(self):
        readout = self.make_readout().fit(torch.randn(3, 4, 3, dtype=torch.float64))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "visual.pt"
            readout.save(path)
            loaded = self.make_readout().load(path)
            torch.testing.assert_close(readout.weight, loaded.weight)
            self.assertEqual(loaded.n_samples, 12)
            different_binding = self.make_readout(binding={**BINDING, "model_id": "different"})
            with self.assertRaisesRegex(ValueError, "binding"):
                different_binding.load(path)
            different_native = self.make_readout()
            with torch.no_grad():
                different_native.last_projection.bias.add_(1)
            with self.assertRaisesRegex(ValueError, "fingerprint"):
                different_native.load(path)
            artifact = torch.load(path, weights_only=True)
            artifact["weight"] += 1
            torch.save(artifact, path)
            with self.assertRaisesRegex(ValueError, "effective weight"):
                self.make_readout().load(path)

    def test_affine_control_is_exact_at_calibration_mean(self):
        readout = self.make_readout()
        readout.norm = nn.Identity()
        states = torch.randn(3, 4, 3, dtype=torch.float64)
        normalized = readout.norm(states)
        readout.fit(normalized)
        mean = normalized.mean((0, 1))
        torch.testing.assert_close(readout.affine(mean), readout.native(mean))

    def test_invalid_fit_preserves_existing_calibration(self):
        readout = self.make_readout().fit(torch.randn(2, 3, dtype=torch.float64))
        weight = readout.weight.clone()
        with self.assertRaisesRegex(ValueError, "at least one"):
            readout.fit(torch.ones(2, 3), mask=torch.tensor([False, False]))
        self.assertEqual(readout.n_samples, 2)
        torch.testing.assert_close(readout.weight, weight)
        with torch.no_grad():
            readout.first_projection.weight.add_(1)
        with self.assertRaisesRegex(ValueError, "changed"):
            readout.update(torch.randn(2, 3))


class ModalityIndexTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def test_nonnegative_reconstruction_and_known_modality_mass(self):
        text = torch.tensor([[2., 0., 0.]], dtype=torch.float64)
        visual = torch.tensor([[0., 3., 0.], [0., 0., 4.]], dtype=torch.float64)
        values = torch.tensor([[1., 3., 2.]], dtype=torch.float64)
        result = modality_index(values, text, visual, k=3)
        self.assertEqual(result.method, "mj_modality_index")
        torch.testing.assert_close(result.tensors["reconstruction"], values)
        torch.testing.assert_close(result.tensors["raw_mi"], torch.tensor([5 / 6], dtype=torch.float64))
        self.assertTrue((result.tensors["coefficients"] >= 0).all())
        torch.testing.assert_close(result.tensors["excess_mi"],
                                   result.tensors["raw_mi"] - result.tensors["rotated_mi"])
        self.assertIn("not evidence", result.metadata["interpretation"])

    def test_positive_atom_rescaling_does_not_move_mi(self):
        text, visual = torch.eye(4, dtype=torch.float64)[:2], torch.eye(4, dtype=torch.float64)[2:]
        values = torch.tensor([[1., 2., 3., 4.], [3., 1., 4., 2.]], dtype=torch.float64)
        first = modality_index(values, text, visual, k=4)
        second = modality_index(values, text * torch.tensor([[.02], [73.]]),
                                visual * torch.tensor([[7.], [.004]]), k=4)
        for name in ("raw_mi", "rotated_mi", "excess_mi", "reconstruction", "rotated_reconstruction"):
            torch.testing.assert_close(first.tensors[name], second.tensors[name])

    def test_rotation_preserves_visual_norm_and_gram(self):
        torch.manual_seed(31)
        visual = torch.randn(7, 4, dtype=torch.float64)
        result = modality_index(torch.randn(2, 4, dtype=torch.float64), torch.randn(9, 4), visual, k=2)
        rotation = result.tensors["rotation"]
        rotated = visual @ rotation
        torch.testing.assert_close(rotation @ rotation.T, torch.eye(4, dtype=torch.float64))
        torch.testing.assert_close(rotated @ rotated.T, visual @ visual.T)
        torch.testing.assert_close(rotated.norm(dim=-1), visual.norm(dim=-1))

    def test_zero_mass_is_nan_with_explicit_validity(self):
        result = modality_index(torch.zeros(2, 3), torch.eye(3), torch.eye(3), k=2)
        for name in ("raw_mi", "rotated_mi", "excess_mi"):
            self.assertTrue(torch.isnan(result.tensors[name]).all())
        for name in ("valid", "rotated_valid", "excess_valid"):
            self.assertFalse(result.tensors[name].any())
        self.assertTrue((result.tensors["atom_ids"] == -1).all())

    def test_size_matching_is_extra_deterministic_control(self):
        torch.manual_seed(9)
        values, text, visual = torch.randn(2, 3), torch.randn(11, 3), torch.randn(4, 3)
        original = modality_index(values, text, visual, k=2, seed=7)
        matched = modality_index(values, text, visual, k=2, seed=7, size_match=True)
        repeated = modality_index(values, text, visual, k=2, seed=7, size_match=True)
        torch.testing.assert_close(original.tensors["raw_mi"], matched.tensors["raw_mi"])
        self.assertEqual(matched.tensors["size_matched_text_ids"].numel(), 4)
        self.assertEqual(matched.tensors["size_matched_text_ids"].unique().numel(), 4)
        torch.testing.assert_close(matched.tensors["size_matched_text_ids"], repeated.tensors["size_matched_text_ids"])
        torch.testing.assert_close(matched.tensors["size_matched_raw_mi"], repeated.tensors["size_matched_raw_mi"], equal_nan=True)

    def test_signed_coefficients_and_mismatched_coordinates_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            modality_index(torch.ones(2, 3), torch.eye(3), torch.eye(3), signed=True)
        with self.assertRaisesRegex(ValueError, "same hidden"):
            modality_index(torch.ones(2, 3), torch.ones(3, 2), torch.eye(3))
        with self.assertRaisesRegex(ValueError, "at least as many"):
            modality_index(torch.ones(2, 3), torch.ones(2, 3), torch.ones(3, 3), size_match=True)


if __name__ == "__main__":
    unittest.main()

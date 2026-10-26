"""The public package name preserves APIs, computation and serialized types."""
import importlib
import os
import pickle
import pkgutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch

import vlmprobe
import vlm_probing

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
from tiny_model import TinyModel


class PackageImportTests(unittest.TestCase):
    def test_public_exports_and_public_nested_modules_share_identity(self):
        self.assertEqual(vlmprobe.__version__, vlm_probing.__version__)
        self.assertEqual(vlmprobe.__all__, vlm_probing.__all__)
        for name in vlm_probing.__all__:
            with self.subTest(export=name):
                self.assertIs(getattr(vlmprobe, name), getattr(vlm_probing, name))
        modules = []
        for top in pkgutil.iter_modules(vlm_probing.__path__, "vlm_probing."):
            if top.name == "vlm_probing._vendor":
                continue  # private implementation, not a public alias contract
            modules.append(top)
            if top.ispkg:
                package = importlib.import_module(top.name)
                modules.extend(pkgutil.walk_packages(package.__path__, top.name + "."))
        for info in modules:
            alias = "vlmprobe." + info.name.removeprefix("vlm_probing.")
            with self.subTest(module=alias):
                canonical = importlib.import_module(info.name)
                self.assertIs(importlib.import_module(alias), canonical)
                self.assertEqual(canonical.__name__, info.name)
        from vlmprobe import visualization
        from vlmprobe.core.types import ProbeResult
        from vlmprobe.training.jacobian import fit_jacobian_lenses
        from vlmprobe.causal.transformer_eap_ig import TransformerEAPIG
        self.assertIs(visualization, vlm_probing.visualization)
        self.assertIs(ProbeResult, vlm_probing.ProbeResult)
        self.assertIs(fit_jacobian_lenses, vlm_probing.training.fit_jacobian_lenses)
        self.assertIs(TransformerEAPIG,
                      vlm_probing.causal.transformer_eap_ig.TransformerEAPIG)

    def test_import_orders_do_not_require_optional_dependencies(self):
        script = '''
import importlib
import importlib.abc
import sys
blocked = {'numpy', 'matplotlib', 'PIL', 'transformers', 'torchvision', 'accelerate',
           'sae_lens', 'vit_prisma', 'overcomplete', 'transformer_lens', 'einops',
           'safetensors', 'huggingface_hub', 'yaml', 'requests', 'tqdm', 'sklearn'}
class MissingOptionalPackages(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in blocked:
            raise ModuleNotFoundError(fullname)
sys.meta_path.insert(0, MissingOptionalPackages())
first = sys.argv[1]
second = 'vlm_probing' if first == 'vlmprobe' else 'vlmprobe'
one = importlib.import_module(first)
two = importlib.import_module(second)
assert one.Prober is two.Prober
assert one.core.types.ProbeResult is two.ProbeResult
assert one.adapters.auto._FACTORIES is two.adapters.auto._FACTORIES
assert callable(one.visualization.plot_lens_heatmap)
assert not blocked.intersection(sys.modules)
'''
        environment = os.environ.copy()
        if "PYTHONPATH" in environment:
            environment["PYTHONPATH"] = os.pathsep.join(
                str(Path(path or ".").resolve())
                for path in environment["PYTHONPATH"].split(os.pathsep)
            )
        with tempfile.TemporaryDirectory() as directory:
            for first in ("vlmprobe", "vlm_probing"):
                with self.subTest(first=first):
                    completed = subprocess.run(
                        [sys.executable, "-c", script, first], cwd=directory,
                        env=environment, capture_output=True, text=True,
                        timeout=60,
                    )
                    self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_registration_is_visible_through_both_names(self):
        from vlmprobe.adapters.auto import _FACTORIES as new_registry
        from vlm_probing.adapters.auto import _FACTORIES as old_registry
        self.assertIs(new_registry, old_registry)

        class RegisteredModel(TinyModel):
            probing_adapter = None

        called = []

        def factory(model):
            called.append(model)
            return TinyModel.probing_adapter(model)

        vlmprobe.register_adapter(RegisteredModel, factory)
        try:
            model = RegisteredModel()
            self.assertIs(vlm_probing.Prober(model).model, model)
            self.assertIs(vlmprobe.Prober(model).model, model)
            self.assertEqual(called, [model, model])
            with self.assertRaisesRegex(ValueError, "already registered"):
                vlm_probing.register_adapter(RegisteredModel, factory)
        finally:
            old_registry.pop(RegisteredModel, None)

    def test_pickle_keeps_the_canonical_type_path(self):
        result = vlmprobe.ProbeResult("import-test", {"values": torch.arange(3)})
        serialized = pickle.dumps(result)
        self.assertIn(b"vlm_probing.core.types", serialized)
        loaded = pickle.loads(serialized)
        self.assertIs(type(loaded), vlmprobe.ProbeResult)
        self.assertIs(type(loaded), vlm_probing.ProbeResult)
        torch.testing.assert_close(loaded.tensors["values"], result.tensors["values"])
        self.assertIs(pickle.loads(pickle.dumps(vlmprobe.Prober)), vlm_probing.Prober)

    def test_three_family_flows_equal_existing_api_and_restore_the_model(self):
        torch.manual_seed(91)
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            model = TinyModel().eval()
            probe = vlmprobe.Prober(model)
            original = vlm_probing.Prober(model)
            inputs = {"input_ids": torch.tensor([[1, 2, 3], [4, 5, 6]]),
                      "image_tokens": torch.randn(2, 2, 6)}
            changed = {**inputs, "image_tokens": torch.zeros_like(inputs["image_tokens"])}
            with torch.no_grad():
                baseline = model(**inputs)["logits"].clone()
            hooks_before = [(tuple(m._forward_hooks), tuple(m._forward_pre_hooks))
                            for m in model.modules()]
            calls = (
                ("lens", lambda p: p.lens.logit(layers=[0, 1]).run(inputs)),
                ("attention", lambda p: p.attention.profile(layers=[0, 1]).run(inputs)),
                ("causal", lambda p: p.causal.patch(layers=[0, 1], tokens="visual").run(
                    changed, source=inputs, metric=vlmprobe.TokenMargin(4, 5))),
            )
            for family, run in calls:
                with self.subTest(family=family):
                    result, expected = run(probe), run(original)
                    self.assertIs(type(result), vlm_probing.ProbeResult)
                    self.assertEqual(result.method, expected.method)
                    self.assertEqual(set(result.tensors), set(expected.tensors))
                    for name in result.tensors:
                        torch.testing.assert_close(result.tensors[name], expected.tensors[name])
                    if family == "lens":
                        torch.testing.assert_close(result.tensors["logits"][-1], baseline[:, -1])
                    elif family == "causal":
                        self.assertTrue(result.tensors["effect"].abs().max() > 0)
            with torch.no_grad():
                torch.testing.assert_close(model(**inputs)["logits"], baseline)
            self.assertEqual(hooks_before,
                             [(tuple(m._forward_hooks), tuple(m._forward_pre_hooks))
                              for m in model.modules()])
            self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))
        finally:
            torch.set_num_threads(previous_threads)


if __name__ == "__main__":
    unittest.main()

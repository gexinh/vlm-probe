"""Notebook setup prepares Colab safely and preserves executed workflows."""
from contextlib import redirect_stdout
import copy
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tools.prepare_notebook_setup import (
    GUIDE_CELL_ID, SETUP_CELL_ID, SETUP_SOURCE, prepare_notebook, prepare_paths,
)


class NotebookPreparationTests(unittest.TestCase):
    def notebook(self):
        return {
            "nbformat": 4, "nbformat_minor": 5,
            "metadata": {"vlmprobe": {"source_sha256": "reviewed-workflow"}},
            "cells": [
                {"cell_type": "markdown", "id": "intro", "metadata": {},
                 "source": ["# Existing tutorial\n"]},
                {"cell_type": "code", "id": "analysis", "metadata": {"custom": 7},
                 "source": ["result = probe.run(inputs)\n"], "execution_count": 3,
                 "outputs": [{"output_type": "stream", "name": "stdout", "text": ["measured\n"]}]},
            ],
        }

    def test_insert_visible_setup_without_changing_workflow_or_claiming_execution(self):
        original = self.notebook()
        before = copy.deepcopy(original)
        prepared = prepare_notebook(original)
        self.assertEqual(original, before)
        self.assertEqual([c["id"] for c in prepared["cells"]],
                         ["intro", GUIDE_CELL_ID, SETUP_CELL_ID, "analysis"])
        self.assertEqual(prepared["cells"][0], before["cells"][0])
        self.assertEqual(prepared["cells"][-1], before["cells"][-1])
        self.assertEqual(prepared["metadata"]["vlmprobe"], before["metadata"]["vlmprobe"])
        self.assertEqual(prepared["cells"][2]["outputs"], [])
        self.assertIsNone(prepared["cells"][2]["execution_count"])
        self.assertNotIn("prepare_notebook_setup", SETUP_SOURCE)
        compile(SETUP_SOURCE, "<visible setup cell>", "exec")

    def test_reapplication_is_identical_and_preserves_real_setup_outputs(self):
        prepared = prepare_notebook(self.notebook())
        code = prepared["cells"][2]
        code["execution_count"] = 1
        code["outputs"] = [{"output_type": "stream", "name": "stdout", "text": ["actual setup\n"]}]
        self.assertEqual(prepare_notebook(prepared), prepared)

    def test_updating_managed_code_clears_only_its_old_execution_record(self):
        prepared = prepare_notebook(self.notebook())
        prepared["cells"][2].update(source=["# previous setup\n"], execution_count=1,
                                    outputs=[{"output_type": "stream", "text": ["old\n"]}])
        updated = prepare_notebook(prepared)
        self.assertEqual(updated["cells"][2]["source"], SETUP_SOURCE.splitlines(keepends=True))
        self.assertEqual(updated["cells"][2]["outputs"], [])
        self.assertEqual(updated["cells"][-1], prepared["cells"][-1])

    def test_conflicting_managed_ids_are_rejected(self):
        for conflict in ("duplicate", "wrong type"):
            with self.subTest(conflict=conflict):
                prepared = prepare_notebook(self.notebook())
                if conflict == "duplicate":
                    prepared["cells"].append(copy.deepcopy(prepared["cells"][2]))
                else:
                    prepared["cells"][2]["cell_type"] = "markdown"
                before = copy.deepcopy(prepared)
                with self.assertRaisesRegex(ValueError, "conflicting managed cell ID"):
                    prepare_notebook(prepared)
                self.assertEqual(prepared, before)

    def test_check_does_not_write_and_repeat_write_is_noop(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "demo.ipynb"
            path.write_text(json.dumps(self.notebook()))
            path.chmod(0o644)
            before, mtime = path.read_bytes(), path.stat().st_mtime_ns
            self.assertEqual(prepare_paths([path]), [path])
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(path.stat().st_mtime_ns, mtime)
            self.assertEqual(prepare_paths([path], write=True), [path])
            updated, mtime = path.read_bytes(), path.stat().st_mtime_ns
            self.assertEqual(path.stat().st_mode & 0o777, 0o644)
            self.assertEqual(prepare_paths([path], write=True), [])
            self.assertEqual(path.read_bytes(), updated)
            self.assertEqual(path.stat().st_mtime_ns, mtime)
            self.assertEqual(list(Path(directory).glob("*.tmp")), [])


class ColabSetupExecutionTests(unittest.TestCase):
    MODULES = ("vlmprobe", "vlm_probing", "torch", "transformers", "numpy",
               "PIL", "safetensors", "huggingface_hub", "google.colab")

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.checkout = self.root / "vlm-probe"
        self.original_cwd = Path.cwd()
        self.original_path = list(sys.path)
        self.calls = []
        self.reads = []
        self.sha = "1" * 40
        self.git_root = self.checkout
        self.git_dirty = ""

    def tearDown(self):
        os.chdir(self.original_cwd)
        sys.path[:] = self.original_path

    def make_project(self, *, name="vlmprobe"):
        self.checkout.mkdir(exist_ok=True)
        for relative in (".git", "demos", "src/vlmprobe", "src/vlm_probing"):
            (self.checkout / relative).mkdir(parents=True, exist_ok=True)
        (self.checkout / "pyproject.toml").write_text(
            f'[project]\nname = "{name}"\n[project.optional-dependencies]\nevaluation = []\n'
        )
        (self.checkout / "src/vlmprobe/__init__.py").write_text("from vlm_probing import *\n")
        (self.checkout / "src/vlm_probing/__init__.py").write_text('__version__ = "1.1.0"\n')
        (self.checkout / "src/vlm_probing/prober.py").write_text("# project marker\n")

    def check_call(self, arguments, **kwargs):
        self.calls.append(list(arguments))
        self.assertFalse(kwargs.get("shell", False))
        if arguments[:2] == ["git", "clone"]:
            self.assertEqual(Path(arguments[-1]), self.checkout)
            self.make_project()
        return 0

    def check_output(self, arguments, **kwargs):
        self.reads.append(list(arguments))
        self.assertFalse(kwargs.get("shell", False))
        self.assertEqual(arguments[:3], ["git", "-C", str(self.checkout)])
        command = arguments[3:]
        if command == ["rev-parse", "--show-toplevel"]:
            return str(self.git_root) + "\n"
        if command == ["rev-parse", "HEAD"]:
            return self.sha + "\n"
        if command == ["status", "--porcelain"]:
            return self.git_dirty
        raise AssertionError(f"Unexpected Git read: {arguments}")

    def execute(self, *, colab=True, module_detection=False, env=None, loaded=None,
                namespace=None):
        namespace = {} if namespace is None else namespace
        environment = {"VLM_PROBING_COLAB_DIR": str(self.checkout)}
        if colab and not module_detection:
            environment["COLAB_RELEASE_TAG"] = "test"
        environment.update(env or {})
        versions = {"vlmprobe": "1.1.0", "torch": "2.6.0"}
        output = io.StringIO()
        with patch.dict(os.environ, environment, clear=True), patch.dict(sys.modules):
            for name in self.MODULES:
                sys.modules.pop(name, None)
            if module_detection:
                sys.modules["google.colab"] = SimpleNamespace()
            sys.modules.update(loaded or {})
            with patch("subprocess.check_call", side_effect=self.check_call), \
                    patch("subprocess.check_output", side_effect=self.check_output), \
                    patch("importlib.metadata.version", side_effect=lambda name: versions.get(name, "1.0")), \
                    redirect_stdout(output):
                exec(compile(SETUP_SOURCE, "<notebook setup>", "exec"), namespace)
        return namespace, output.getvalue()

    def pip_calls(self):
        return [arguments for arguments in self.calls if arguments[1:4] == ["-m", "pip", "install"]]

    def test_local_runtime_has_no_subprocess_directory_or_import_path_side_effect(self):
        before = Path.cwd(), list(sys.path)
        with patch("os.chdir") as chdir:
            _, output = self.execute(colab=False)
        self.assertEqual(self.calls, [])
        self.assertEqual(self.reads, [])
        self.assertFalse(self.checkout.exists())
        chdir.assert_not_called()
        self.assertEqual((Path.cwd(), sys.path), before)
        self.assertIn("Local runtime", output)

    def test_fresh_colab_clone_uses_argv_ref_and_local_extras(self):
        _, output = self.execute(env={
            "VLM_PROBING_REPOSITORY": "gexinh/vlm-probe",
            "VLM_PROBING_REF": "a" * 40,
        })
        self.assertEqual(self.calls[0], [
            "git", "clone", "--depth=1", "--no-checkout",
            "https://github.com/gexinh/vlm-probe.git", str(self.checkout),
        ])
        self.assertEqual(self.calls[1][-4:], ["fetch", "--depth=1", "origin", "a" * 40])
        self.assertEqual(self.calls[2][-3:], ["checkout", "--detach", "FETCH_HEAD"])
        self.assertEqual(self.pip_calls()[0], [
            sys.executable, "-m", "pip", "install", "-e",
            str(self.checkout) + "[transformers,visualization,sae,evaluation]",
        ])
        self.assertEqual(Path.cwd(), self.checkout)
        self.assertIn(str(self.checkout / "src"), sys.path)
        self.assertIn(self.sha, output)
        self.assertNotIn("Restart", output)

    def test_loaded_colab_module_also_enables_setup(self):
        self.execute(module_detection=True)
        self.assertEqual(len(self.pip_calls()), 1)

    def test_existing_project_and_dirty_files_are_reused_without_git_mutation(self):
        self.make_project()
        marker = self.checkout / "my-notes.txt"
        marker.write_text("keep this exactly\n")
        self.git_dirty = "?? my-notes.txt\n"
        _, output = self.execute(env={"VLM_PROBING_REF": "another-branch"})
        self.assertEqual(marker.read_text(), "keep this exactly\n")
        self.assertEqual(self.calls, self.pip_calls())
        self.assertEqual(len(self.pip_calls()), 1)
        self.assertIn("repository/ref settings apply only to a new checkout", output)
        self.assertIn("Local checkout changes are preserved", output)
        self.assertIn(self.sha, output)

    def test_existing_unrelated_directory_is_not_changed(self):
        self.checkout.mkdir()
        marker = self.checkout / "valuable.txt"
        marker.write_bytes(b"never replace")
        before = sorted(p.name for p in self.checkout.iterdir())
        with self.assertRaisesRegex(RuntimeError, "not a vlmprobe checkout"):
            self.execute()
        self.assertEqual(sorted(p.name for p in self.checkout.iterdir()), before)
        self.assertEqual(marker.read_bytes(), b"never replace")
        self.assertEqual(self.calls, [])
        self.assertEqual(self.reads, [])
        self.assertEqual(Path.cwd(), self.original_cwd)

    def test_marker_files_do_not_override_another_project_identity(self):
        self.make_project(name="another-project")
        before = (self.checkout / "pyproject.toml").read_bytes()
        with self.assertRaisesRegex(RuntimeError, "different project"):
            self.execute()
        self.assertEqual(self.calls, [])
        self.assertEqual(self.reads, [])
        self.assertEqual((self.checkout / "pyproject.toml").read_bytes(), before)
        self.assertEqual(Path.cwd(), self.original_cwd)

    def test_directory_inside_another_git_project_is_not_installed(self):
        self.make_project()
        self.git_root = self.root
        with self.assertRaisesRegex(RuntimeError, "inside another Git project"):
            self.execute()
        self.assertEqual(self.calls, [])
        self.assertEqual(Path.cwd(), self.original_cwd)

    def test_repeated_setup_skips_clone_fetch_checkout_and_pip(self):
        namespace, _ = self.execute()
        count = len(self.calls)
        _, output = self.execute(namespace=namespace)
        self.assertEqual(len(self.calls), count)
        self.assertEqual(len(self.pip_calls()), 1)
        self.assertIn("already prepared the same checkout", output)
        self.assertNotIn("Restart", output)

    def test_old_imported_package_requires_restart_after_installation(self):
        self.make_project()
        loaded = {"vlmprobe": SimpleNamespace(__version__="0.9.0", __file__="/old/vlmprobe/__init__.py")}
        with self.assertRaisesRegex(RuntimeError, "Restart the Colab runtime"):
            self.execute(loaded=loaded)
        self.assertEqual(len(self.pip_calls()), 1)

    def test_current_source_import_does_not_request_unnecessary_restart(self):
        self.make_project()
        loaded = {"vlmprobe": SimpleNamespace(
            __version__="1.1.0", __file__=str(self.checkout / "src/vlmprobe/__init__.py")
        )}
        _, output = self.execute(loaded=loaded)
        self.assertIn("Setup complete", output)
        self.assertNotIn("Restart", output)

    def test_updated_loaded_dependency_also_requires_restart(self):
        self.make_project()
        loaded = {"torch": SimpleNamespace(__version__="2.0.0", __file__="/installed/torch/__init__.py")}
        with self.assertRaisesRegex(RuntimeError, "torch 2.0.0 is loaded"):
            self.execute(loaded=loaded)
        self.assertEqual(len(self.pip_calls()), 1)

    def test_changed_checkout_after_import_requires_restart_even_at_same_version(self):
        namespace, _ = self.execute()
        self.sha = "2" * 40
        loaded = {"vlmprobe": SimpleNamespace(
            __version__="1.1.0", __file__=str(self.checkout / "src/vlmprobe/__init__.py")
        )}
        with self.assertRaisesRegex(RuntimeError, "checkout or installation changed"):
            self.execute(namespace=namespace, loaded=loaded)
        self.assertEqual(len(self.pip_calls()), 2)

    def test_invalid_repository_or_option_like_ref_fails_before_filesystem_changes(self):
        for env in ({"VLM_PROBING_REPOSITORY": "../another"},
                    {"VLM_PROBING_REF": "--upload-pack=bad"},
                    {"VLM_PROBING_REF": "two words"}):
            with self.subTest(env=env):
                with self.assertRaises(ValueError):
                    self.execute(env=env)
                self.assertEqual(self.calls, [])
                self.assertEqual(self.reads, [])
                self.assertFalse(self.checkout.exists())


if __name__ == "__main__":
    unittest.main()

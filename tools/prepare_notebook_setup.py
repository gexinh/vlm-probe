#!/usr/bin/env python3
"""Add a visible, self-contained Colab setup cell to notebook tutorials.

The cell performs environment preparation only. Original analysis cells and
their stored outputs remain unchanged. Run with --write to update notebooks;
the default reports notebooks needing preparation without writing to disk.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import stat
import tempfile


ROOT = Path(__file__).resolve().parents[1]
SETUP_CELL_ID = "vlmprobe-runtime-setup"
GUIDE_CELL_ID = "vlmprobe-runtime-setup-guide"
SETUP_VERSION = 1
GUIDE_SOURCE = """## Runtime setup

Run the setup cell first. In Google Colab it obtains a Git checkout and installs
the notebook dependencies; local execution keeps the existing environment.
Choose a GPU runtime when required by the walkthrough.

Optional environment settings: VLM_PROBING_REPOSITORY (default
ubc-tea/vlm-probe), VLM_PROBING_REF (default main), and
VLM_PROBING_COLAB_DIR (default /content/vlm-probe). The repository and ref
select a new checkout. An existing vlmprobe checkout is reused at its current
commit without discarding local changes. The cell prints that commit.

All input construction, calibration, probing and plotting remain visible in
the analysis cells below. Restart the runtime only if the setup cell reports
that an already imported package or checkout has changed.
"""

SETUP_SOURCE = r'''#@title Runtime setup (Colab only; expand to inspect)
# Environment setup only; the analysis workflow is visible in the cells below.
import os as _nb_setup_os
import sys as _nb_setup_sys

_nb_setup_in_colab = (
    "google.colab" in _nb_setup_sys.modules
    or bool(_nb_setup_os.environ.get("COLAB_RELEASE_TAG"))
)
if _nb_setup_in_colab:
    import hashlib as _nb_setup_hashlib
    import importlib as _nb_setup_importlib
    import importlib.metadata as _nb_setup_metadata
    from pathlib import Path as _NBSetupPath
    import re as _nb_setup_re
    import subprocess as _nb_setup_subprocess

    _nb_setup_repo = _nb_setup_os.environ.get(
        "VLM_PROBING_REPOSITORY", "ubc-tea/vlm-probe"
    )
    _nb_setup_ref = _nb_setup_os.environ.get("VLM_PROBING_REF", "main")
    if not _nb_setup_re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9-]*/[A-Za-z0-9][A-Za-z0-9_.-]*",
        _nb_setup_repo,
    ):
        raise ValueError("VLM_PROBING_REPOSITORY must be a GitHub owner/repository.")
    if not _nb_setup_ref or _nb_setup_ref.startswith("-") or any(
        character.isspace() for character in _nb_setup_ref
    ):
        raise ValueError("VLM_PROBING_REF must be a nonempty Git ref or commit SHA.")
    _nb_setup_checkout = _NBSetupPath(
        _nb_setup_os.environ.get("VLM_PROBING_COLAB_DIR", "/content/vlm-probe")
    ).expanduser().resolve()
    _nb_setup_existing = _nb_setup_checkout.exists()
    if not _nb_setup_existing:
        _nb_setup_checkout.parent.mkdir(parents=True, exist_ok=True)
        _nb_setup_subprocess.check_call([
            "git", "clone", "--depth=1", "--no-checkout",
            "https://github.com/" + _nb_setup_repo + ".git",
            str(_nb_setup_checkout),
        ])
        _nb_setup_subprocess.check_call([
            "git", "-C", str(_nb_setup_checkout), "fetch",
            "--depth=1", "origin", _nb_setup_ref,
        ])
        _nb_setup_subprocess.check_call([
            "git", "-C", str(_nb_setup_checkout),
            "checkout", "--detach", "FETCH_HEAD",
        ])

    # Validate before changing directories or installing anything. In
    # particular, never reset, clean or overwrite an existing user directory.
    _nb_setup_pyproject = _nb_setup_checkout / "pyproject.toml"
    _nb_setup_init = _nb_setup_checkout / "src/vlm_probing/__init__.py"
    if (
        not _nb_setup_checkout.is_dir()
        or not _nb_setup_pyproject.is_file()
        or not _nb_setup_init.is_file()
        or not (_nb_setup_checkout / "src/vlmprobe/__init__.py").is_file()
        or not (_nb_setup_checkout / "src/vlm_probing/prober.py").is_file()
        or not (_nb_setup_checkout / "demos").is_dir()
    ):
        raise RuntimeError(
            "The setup directory is not a vlmprobe checkout. It was left "
            "untouched. Set VLM_PROBING_COLAB_DIR to a new directory."
        )
    _nb_setup_project_section = _nb_setup_re.search(
        r"(?ms)^\[project\][^\n]*\n(.*?)(?=^\[|\Z)",
        _nb_setup_pyproject.read_text(),
    )
    if not _nb_setup_project_section or not _nb_setup_re.search(
        r"""(?m)^name\s*=\s*["']vlmprobe["']\s*$""",
        _nb_setup_project_section.group(1),
    ):
        raise RuntimeError(
            "The existing directory declares a different project; it was "
            "left untouched. Choose another VLM_PROBING_COLAB_DIR."
        )

    def _nb_setup_git(*arguments):
        return _nb_setup_subprocess.check_output(
            ["git", "-C", str(_nb_setup_checkout), *arguments], text=True
        ).strip()

    try:
        _nb_setup_git_root = _NBSetupPath(
            _nb_setup_git("rev-parse", "--show-toplevel")
        ).resolve()
        _nb_setup_sha = _nb_setup_git("rev-parse", "HEAD")
        _nb_setup_dirty = bool(_nb_setup_git("status", "--porcelain"))
    except _nb_setup_subprocess.CalledProcessError as _nb_setup_error:
        raise RuntimeError(
            "The existing vlmprobe directory is not a readable Git checkout; "
            "it was not reset or replaced. Choose another setup directory."
        ) from _nb_setup_error
    if _nb_setup_git_root != _nb_setup_checkout:
        raise RuntimeError(
            "The setup directory is inside another Git project, not the "
            "vlmprobe checkout root. Choose another setup directory."
        )
    if _nb_setup_existing:
        print(
            "Reusing the existing vlmprobe checkout; repository/ref settings "
            "apply only to a new checkout."
        )
    print("vlmprobe checkout:", _nb_setup_checkout, "@", _nb_setup_sha)
    if _nb_setup_dirty:
        print("Local checkout changes are preserved.")

    _nb_setup_distributions = {
        "vlmprobe": "vlmprobe",
        "vlm_probing": "vlmprobe",
        "torch": "torch",
        "transformers": "transformers",
        "numpy": "numpy",
        "PIL": "Pillow",
        "safetensors": "safetensors",
        "huggingface_hub": "huggingface-hub",
    }
    _nb_setup_loaded = {
        name: (
            getattr(_nb_setup_sys.modules[name], "__version__", None),
            getattr(_nb_setup_sys.modules[name], "__file__", None),
        )
        for name in _nb_setup_distributions
        if name in _nb_setup_sys.modules
    }
    _nb_setup_extras = "transformers,visualization,sae,evaluation"
    _nb_setup_signature = (
        str(_nb_setup_checkout), _nb_setup_sha, _nb_setup_sys.executable,
        _nb_setup_hashlib.sha256(_nb_setup_pyproject.read_bytes()).hexdigest(),
        _nb_setup_extras,
    )
    _nb_setup_previous = globals().get("_vlmprobe_colab_setup_state", {})
    if not isinstance(_nb_setup_previous, dict):
        _nb_setup_previous = {}
    _nb_setup_os.chdir(_nb_setup_checkout)
    if _nb_setup_previous.get("signature") != _nb_setup_signature:
        _nb_setup_subprocess.check_call([
            _nb_setup_sys.executable, "-m", "pip", "install", "-e",
            str(_nb_setup_checkout) + "[" + _nb_setup_extras + "]",
        ])
        _vlmprobe_colab_setup_state = {"signature": _nb_setup_signature}
    else:
        print("This runtime already prepared the same checkout and dependencies.")
    _nb_setup_source_path = str(_nb_setup_checkout / "src")
    if _nb_setup_source_path not in _nb_setup_sys.path:
        _nb_setup_sys.path.insert(0, _nb_setup_source_path)
    _nb_setup_importlib.invalidate_caches()

    _nb_setup_restart = []
    if (
        _nb_setup_previous.get("signature")
        and _nb_setup_previous["signature"] != _nb_setup_signature
        and any(name in _nb_setup_loaded for name in ("vlmprobe", "vlm_probing"))
    ):
        _nb_setup_restart.append(
            "the checkout or installation changed after vlmprobe was imported"
        )
    for _nb_setup_name, (_nb_setup_loaded_version, _nb_setup_origin) in _nb_setup_loaded.items():
        if _nb_setup_name in ("vlmprobe", "vlm_probing") and _nb_setup_origin:
            if not _NBSetupPath(_nb_setup_origin).resolve().is_relative_to(
                _nb_setup_checkout / "src"
            ):
                _nb_setup_restart.append(_nb_setup_name + " was imported from another checkout")
        if _nb_setup_loaded_version is not None:
            try:
                _nb_setup_installed_version = _nb_setup_metadata.version(
                    _nb_setup_distributions[_nb_setup_name]
                )
            except _nb_setup_metadata.PackageNotFoundError:
                continue
            if str(_nb_setup_loaded_version) != _nb_setup_installed_version:
                _nb_setup_restart.append(
                    _nb_setup_name + " " + str(_nb_setup_loaded_version)
                    + " is loaded; installed version is " + _nb_setup_installed_version
                )
    if _nb_setup_restart:
        raise RuntimeError(
            "Setup completed, but this runtime already imported an older or "
            "different package. Restart the Colab runtime, rerun this setup "
            "cell, and then run the analysis cells. Details: "
            + "; ".join(_nb_setup_restart)
        )
    print("Setup complete. Run the analysis cells below.")
else:
    print("Local runtime: using the existing checkout and environment.")
'''


def _lines(value: str) -> list[str]:
    return value.splitlines(keepends=True)


def _code_cell() -> dict:
    return {
        "cell_type": "code",
        "id": SETUP_CELL_ID,
        "metadata": {"tags": ["environment-setup"],
                     "jupyter": {"source_hidden": True},
                     "colab": {"cellView": "form"}},
        "source": _lines(SETUP_SOURCE),
        "execution_count": None,
        "outputs": [],
    }


def _guide_cell() -> dict:
    return {
        "cell_type": "markdown",
        "id": GUIDE_CELL_ID,
        "metadata": {},
        "source": _lines(GUIDE_SOURCE),
    }


def prepare_notebook(notebook: dict) -> dict:
    """Return a copy with one managed guide/setup pair and intact workflow."""
    result = copy.deepcopy(notebook)
    cells = result["cells"]
    for identifier, cell_type in ((GUIDE_CELL_ID, "markdown"), (SETUP_CELL_ID, "code")):
        matches = [cell for cell in cells if cell.get("id") == identifier]
        if len(matches) > 1 or (matches and matches[0].get("cell_type") != cell_type):
            raise ValueError(f"Notebook contains conflicting managed cell ID {identifier!r}")
    original_code = next((cell for cell in cells if cell.get("id") == SETUP_CELL_ID), None)
    code = _code_cell()
    if original_code and original_code.get("source") == code["source"]:
        # The same environment cell can retain its real execution record.
        code = original_code
    cells[:] = [cell for cell in cells if cell.get("id") not in {GUIDE_CELL_ID, SETUP_CELL_ID}]
    index = 1 if cells and cells[0].get("cell_type") == "markdown" else 0
    cells[index:index] = [_guide_cell(), code]
    result.setdefault("metadata", {})["vlmprobe_setup"] = {
        "version": SETUP_VERSION,
        "cell_id": SETUP_CELL_ID,
        "source_sha256": hashlib.sha256(SETUP_SOURCE.encode()).hexdigest(),
        "scope": "environment bootstrap only; original workflow cells preserved",
    }
    return result


def prepare_paths(paths: list[Path], *, write: bool = False) -> list[Path]:
    """List changed notebooks; write only with explicit opt-in."""
    changed = []
    for path in paths:
        if path.suffix != ".ipynb":
            raise ValueError(f"Expected a notebook path: {path}")
        notebook = json.loads(path.read_text())
        prepared = prepare_notebook(notebook)
        if prepared == notebook:
            continue
        changed.append(path)
        if write:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=path.parent, prefix="." + path.stem + "-",
                suffix=".tmp", delete=False,
            ) as stream:
                stream.write(json.dumps(prepared, ensure_ascii=False, indent=1) + "\n")
                temporary = Path(stream.name)
            os.chmod(temporary, stat.S_IMODE(path.stat().st_mode))
            temporary.replace(path)
    return changed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notebooks", nargs="*", type=Path, help="Default: all demos/*.ipynb")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--write", action="store_true", help="Insert/update the managed cells")
    group.add_argument("--check", action="store_true", help="Report pending updates without writing")
    args = parser.parse_args()
    paths = args.notebooks or sorted((ROOT / "demos").glob("*.ipynb"))
    changed = prepare_paths(paths, write=args.write)
    for path in changed:
        print(("Prepared: " if args.write else "Needs setup: ") + str(path))
    if not args.write and changed:
        raise SystemExit(1)
    print(f"{len(paths)} notebooks checked; {len(changed)} {'updated' if args.write else 'pending'}.")


if __name__ == "__main__":
    main()

# Installation and package releases

[Home](../README.md) / [Documentation](README.md)

The library, installable distribution, and recommended Python import are
**`vlmprobe`**. The laboratory repository is **`ubc-tea/vlm-probe`**, with a
pre-release repository at **`gexinh/vlm-probe`**. Both
share one PyPI project.

Version **1.1.0** promotes the reviewed MJ-Lens and library-feedback updates
from the `1.1.0.dev2` development snapshot. The release changes version and
publication metadata; numerical methods and their defaults remain the same.

## Install and import

Install the published package from [PyPI](https://pypi.org/project/vlmprobe/):

```bash
python -m pip install "vlmprobe[transformers,visualization]"
```

```python
import vlmprobe
from vlmprobe import Prober, TokenMargin, visualization as viz

probe = Prober(model, processor)  # caller-owned model and native processor
result = probe.lens.logit(tokens="last_prompt").run(inputs)
```

Native model inputs, adapter checks, metrics, fitting, and visualization retain
the [common API](API.md). Importing the base package requires PyTorch; plotting
and Hugging Face dependencies are loaded only when those features are used.

| Extra | Adds |
| --- | --- |
| `visualization` | Matplotlib, NumPy, and Pillow for built-in plots. |
| `transformers` | Supported Transformers 5.3.x, Accelerate, Torchvision, and Pillow. |
| `gemma4` | Gemma 4 runtime: Transformers 5.5.4, Accelerate, Torchvision, and Pillow; use in a separate environment instead of `transformers`. |
| `notebooks` | Jupyter and the plotting/tokenizer tools used by demos. |
| `sae` | PyTorch ≥2.6, NumPy, Safetensors and Hugging Face Hub for sparse dictionaries and safe historical checkpoint loading. |
| `evaluation` | NumPy and scikit-learn for optional sparse diagnostic probes; reconstruction, sparsity and localization use PyTorch. |

For tensor methods and custom PyTorch adapters, the base installation is
`python -m pip install vlmprobe`.

For the latest development code, install from GitHub without cloning the
repository yourself:

```bash
python -m pip install "vlmprobe[transformers,visualization] @ git+https://github.com/ubc-tea/vlm-probe.git"
```

For a pinned experiment, append a commit or release tag to the Git URL, using
[pip's VCS syntax](https://pip.pypa.io/en/stable/topics/vcs-support/).

## Compatibility

`vlm_probing` remains an import alias to the same implementation. Both import
names share the exact classes, submodules, adapter registry, and serialized
types; there are no separate copies of the algorithms. Advanced imports work
under either spelling, for example `vlmprobe.adapters` or
`vlmprobe.training.jacobian`.

The old distribution was called `vlm-probing`. If it is already installed,
remove that distribution before installing `vlmprobe`, because both releases
contain the compatibility namespace:

```bash
python -m pip uninstall vlm-probing
```

Existing source imports and saved results do not need to be rewritten.

## Build a release

From the repository root:

```bash
python -m pip install build twine
python -m build
python -m twine check --strict dist/*
```

`dist/` contains a wheel and source distribution. The wheel installs the
library and third-party notices. Model weights, datasets, report/slides,
experiment archives, and notebook outputs are excluded. Demos and their input
images remain available in the GitHub repository.

The version has one source: `vlm_probing.__version__`. Setuptools reads that
constant to create distribution metadata; `vlmprobe.__version__` exposes the
same value. The source distribution is sufficient to rebuild the wheel.

## Publish a new version

The repository includes a GitHub Actions workflow named **Build and publish
Python package** (`.github/workflows/package.yml`). Pushes and pull requests
only build and test the package on Python 3.10 and 3.12. Publishing is a manual
workflow action, using PyPI's token-free trusted publishing.

`pip install vlmprobe` downloads a versioned distribution uploaded to PyPI; it
does not fetch either repository's current `main`. GitHub pushes alone do not
update that installation. Both mirrors build the package, but only
`gexinh/vlm-probe` can execute the PyPI publish job. The Trusted Publisher must
match the current repository name; update an older `vlm-prob` or `vlm-probing` entry before
publishing after the rename. Publish each version once. Repository links in package metadata
may point to the lab even while the publishing workflow runs in the personal
mirror. See [repository synchronization](REPOSITORY_SYNC.md).

The project uses a [trusted publisher](https://docs.pypi.org/trusted-publishers/)
with these values:

| Field | Value |
| --- | --- |
| PyPI project name | `vlmprobe` |
| GitHub owner | `gexinh` |
| Repository | `vlm-probe` |
| Workflow filename | `package.yml` |
| Environment | `pypi` |

The matching GitHub repository environment `pypi` allows deployment from
`main`. Before each new release, increment `vlm_probing.__version__` and push
the reviewed change. In **Actions → Build and publish Python package → Run
workflow**, choose `main` and set
`publish_target` to `pypi`. The workflow publishes only after both build/test
jobs succeed. Its OIDC permission belongs only to the publish job; no PyPI API
token is stored in the repository.

To test a release first, configure a trusted publisher on **TestPyPI**
with environment `testpypi`, then select `testpypi` in the workflow. TestPyPI
accounts and projects are separate from PyPI. An index refuses overwriting an
already published version; increment the version for later releases.

## License metadata

Original VLMProbe code uses the [PolyForm Noncommercial License 1.0.0](../LICENSE).
Bundled third-party components retain their original MIT terms; see
[licensing notices](../NOTICE.md). The distribution records the combined SPDX
expression `PolyForm-Noncommercial-1.0.0 AND MIT`, with all license files included
in source archives and wheels. This metadata does not relicense model weights
or datasets.

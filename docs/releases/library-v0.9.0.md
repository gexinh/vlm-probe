# VLMProbe 0.9.0: install and import as `vlmprobe`

## Goal

Let users install the library with pip and start probing a caller-owned model
through `import vlmprobe`, without cloning the repository manually or setting
`PYTHONPATH`.

## Changes from 0.8.0

- Rename the distribution to `vlmprobe` and add the matching public import.
  Keep `vlm_probing` and all its submodule/class identities compatible.
- Preserve one algorithm implementation, shared adapter registration, and old
  pickle paths. The new namespace does not duplicate execution code.
- Keep PyTorch as the sole mandatory runtime dependency. Existing optional
  model, visualization, and notebook extras remain separate.
- Add package metadata, a single version source, a lean source-distribution
  manifest, and automated installed-wheel checks.
- Add manual trusted-publisher workflows for PyPI/TestPyPI. GitHub pip installs
  work before PyPI publication; plain `pip install vlmprobe` requires the first
  successful PyPI release.
- Update the README, current API/model/method documentation, and CPU example
  to use the new import. Link the independent technical report repository.

Older experiment versions, redundant algorithm copies, reports/slides,
datasets, checkpoints, and output archives are not added to the distribution.
The previous release note remains as historical release documentation.

## Expected experimental results

The package/import change does not alter probing computations or model
configuration. Given the same model, inputs, metric, and method settings, both
import names must return identical lens, attention, and causal tensors.
Model hooks and gradients must be restored after execution. No new model
inference, lens training, or dataset-level result is introduced.

## Validation

- All 5 namespace/identity tests and 22 existing Prober tests passed. Checks
  cover both import orders without optional dependencies, shared registration,
  serialized types, and equal lens/attention/causal tensors on actual forwards.
- All 5 namespace tests also passed against the installed wheel outside the
  repository, with `PYTHONPATH` unset; both import paths resolve to that
  environment's installed files and report version 0.9.0.
- The CPU quickstart completed all configured methods, including fitting and
  artifact save/load. No pretrained weights or dataset downloads were needed.
- Wheel and source-distribution builds passed strict Twine metadata checks.
  The wheel includes all 72 source modules and vendored notices; the archives
  exclude generated/experiment folders. A wheel rebuilt from the unpacked
  source distribution has identical file contents and metadata.
- 66 Python documentation blocks parsed successfully; workflow YAML and
  whitespace checks passed.

## PyPI publication

Version **0.9.0** is published at
[PyPI](https://pypi.org/project/vlmprobe/0.9.0/), built from commit
`b837be9aaaeb9efbbbd9c4458c944d99d54eab26`. The
[publication workflow](https://github.com/gexinh/vlm-probing/actions/runs/38105633245)
passed the Python 3.10 and 3.12 build/installed-wheel checks and the trusted
publisher upload. The GitHub `pypi` environment restricts deployment to `main`.

Installation with `pip install vlmprobe==0.9.0` was verified against the public
PyPI index in a new virtual environment reusing an existing PyTorch install.
Both import names resolve to the installed package, retain shared class and
adapter identities, and report version 0.9.0. All five namespace tests passed
against that PyPI installation, including actual forwards through all three
probing families.
The `transformers,visualization` extras also resolved successfully against
PyPI in a pip dry run. Both published archives have the same SHA256 digests
as the successful workflow's Python 3.12 build artifacts.

README and project-page installation instructions now lead with the published
PyPI package; GitHub installation remains available for development versions.

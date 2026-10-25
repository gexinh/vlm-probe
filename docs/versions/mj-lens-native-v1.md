# Native MJ-Lens integration, v1

## Goal and baseline

Add the modality-split Jacobian Lens from *Does the Workspace See? Testing a
Multimodal Global Workspace in Vision-Language Models*, OpenReview
[`Xrp8mNku6R`](https://openreview.net/forum?id=Xrp8mNku6R). The reviewed input is
the uploaded 55-page PDF, SHA-256
`11c7ab4483ba6326ea83b25bd047c6198f39355032ffd608dca0da8b73c0f780`.

This development version starts from `be21ea55d8e10a07d885d6a3b241a0b624e9c8e3`
(library 1.0.2), on `feat/mj-lens`. Its package version is `1.1.0.dev0`; it is
separate from the pending 1.0.2 PyPI publication. No report experiments or
existing estimator artifacts are replaced.

## Differences from the previous version

- `probe.lens.mj(...)` binds a source modality, output channel and calibration
  context. Understanding models expose text-to-text and visual-to-text;
  native Janus additionally exposes text-to-visual and visual-to-visual.
- Eq. (2) is implemented as a mean over admissible causal pairs within each
  prompt, followed by an equal prompt mean. The paper does not publish a finite
  normalizer implementation; this is the explicit interpretation of its written
  expectation. It is distinct from the existing Jacobian Lens's sum over
  targets and is stored under a different estimator/artifact identity.
- Native RMS decoder contracts declare paper-style block **input** indices and
  the actual final-normalization input. The existing residual-output APIs and
  indices keep their previous meanings.
- Dataset calibration shares VJPs across selected layers, accumulates odd/even
  prompt estimates, checks resume identities and restores model mode and
  parameter flags. Empty modality pairs fail explicitly.
- Native Transformers-format Janus supports ordinary image understanding and
  explicit teacher-forced code sequences. Text and visual IDs remain separate;
  a visual encoder is never treated as a visual output codebook.
- Visual-head calibration averages pointwise GELU derivatives and constructs
  the bias-free effective readout of Eq. (3). The same fitted head can be shared
  across contexts. Native and affine heads remain separate fidelity controls.
  Effective weights are cached and invalidated by refitting or native changes.
- Gain-aware lens dictionaries support a nonnegative sparse decomposition.
  Raw, rotated-control and excess modality indices are returned together,
  including optional size-matched dictionaries. The paper leaves its exact
  solver unspecified; the nonnegative OMP/projected-NNLS solver is this
  library's documented implementation choice. Zero mass is undefined/NaN.
- Codebook plots require explicit source grids and a native decoder. A single
  text readout is not tiled into an image. Native VQ display conversion is
  explicit, and rendering is not presented as fresh generation or grounding.
- The method guide includes parameter tables, complete pretrained loading and
  calibration examples, a standalone offline script and an executed notebook.
  README retains the existing method families and adds one matching MJ button.

## Expected experimental results

On controlled causal fixtures, each modality operator should equal its analytic
pair-conditioned derivative. Unequal batches and resumed calibration should
produce the same estimate as one joint dataset. Native final-normalization
readouts, visual-head derivatives and saved/reloaded outputs should agree with
their declared contracts; mismatched modalities, contexts, heads or revisions
should fail rather than silently reuse an incompatible artifact.

The random tiny Janus example should produce four finite operators, distinct
source/target pair counts, visual code grids and controlled modality-index
results. Its heatmaps and VQ colors are mechanics demonstrations, not learned
concepts. Raw MI may be dominated by the rotated control; excess MI may be
negative. Readability alone does not establish either semantic faithfulness or
causal control of generation.

## Validation and limits

- Complete source suite: **379 tests passed**, including 50 MJ-specific tests.
- Installed-wheel regressions: **55 tests passed**. All 95 installed Python
  files match the reviewed source, and the complete tutorial also executes
  against that installed wheel. Wheel/sdist builds and strict metadata checks
  passed.
- Offline end-to-end script and notebook execute all four operators on four
  calibration images and one independent test image; no cached probe results
  or external lens packages are used.
- Native architecture checks cover Janus pixels, teacher-forced codes, final
  readout/gradients and VQ decoding, plus existing RMS VLM families. These use
  small random native models under Transformers 5.3.0.
- The notebook contains all input construction, calibration and plotting code;
  all 12 code cells executed without errors and display four live figures.
- The implementation was independently checked against the paper's formulas,
  tap conventions, normalization, dictionary controls and interpretation limits.
- Static documentation checks passed for 21 method guides, 13 model guides,
  36 Python blocks, 20 scripts, ten notebooks and 95 factory parameters.

The paper's pretrained-model benchmarks, causal-ablation findings, layerwise
faithfulness bands and generation-writing studies are not reproduced here.
Pretrained example paths are provided but are not claims of new measured
accuracy. No compatible public author MJ artifacts were identified from the
paper, so loading currently uses this library's calibrated save/load format.
Exact full-width VJPs, vocabulary dictionaries and orthogonal rotation can be
expensive on large models; budget layer/channel/data selections explicitly.

No model weights, paper PDF, copied upstream toolkit, generated training cache
or obsolete alternate implementation is included in the source tree. The
existing Jacobian Lens remains intact. Source changes remain on this local
development branch until reviewed; no release, PyPI, Hugging Face or report
publication is performed by this integration.

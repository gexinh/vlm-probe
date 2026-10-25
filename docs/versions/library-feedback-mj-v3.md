# Library feedback audit and MJ-Lens publication, v3

## Goal and baseline

Audit the six findings in `vlmprobe_LIBRARY_ISSUES_EN.pdf` against the current
library, reproduce applicable symptoms, repair remaining defects, and publish
the pretrained MJ-Lens walkthrough with consistent documentation and websites.
The feedback PDF SHA-256 is
`a04a894e5fb0d7344aefb07d21cec70a0f57095add755e273439389e95a8659f`.
It names commit `93e8f3e0` and package 1.0.1; its statements are not automatically
current defects. The previous hotfix is already in the reviewed source.

This development version is `1.1.0.dev2`. It builds on the
[pretrained MJ-Lens v2](mj-lens-pretrained-v2.md) working tree and native MJ
integration `f45879c6cddb007d9a6bb47b726a65b6a1bd96eb`. The current personal
remote baseline is `be21ea55d8e10a07d885d6a3b241a0b624e9c8e3`; the lab baseline
`df01f5f7220570f30a00b90660c2a8f7242cea2e` differs from it only in README links.

## Findings and solutions

| Feedback | Current finding | Action |
| --- | --- | --- |
| 1. Causal propagation positional prior | Early-position concentration is reproducible. Existing construction warnings and applicability metadata already disclose the architecture boundary. Complete content independence and universal immunity of particular alternatives are not supported. | Preserve each method's formulas. Add an explicit mass-distribution diagnostic; document matched controls and undefined ratios. |
| 2A. Short calibration prompts | The retained-source/target guard remains intentional. Current errors already identify failed rows, valid/retained counts, selector, exclusions and remedies. | Retain the existing actionable failure; do not silently drop prompts or change default exclusions. |
| 2B. Repeated single-batch `fit` | Current public calls already warn that the previous Jacobian is replaced; `fit_jacobian_lenses` accumulates corpus batches. | Preserve that contract. Repair two remaining corpus resume/serialization defects found during this audit. |
| 3. Pixel shape used as visual grid | Native `visual_grid` already handles audited single-image, packed-Qwen and InternVL tile-local paths. SmolVLM automatic grid inference explicitly rejects an unavailable mapping. | Verify native processors and preserve explicit capability boundaries instead of guessing a grid. |
| 4. Attention sink hides display structure | Percentile display limits already exist and leave measured values unchanged. | Verify default, signed, shared-scale and clipping behavior; retain min/max as the default. |
| 5. Qwen text adapter gap | Exact native Qwen2/Qwen3 text contracts and an informative unsupported-class error already exist. | Re-run parity, gradient, readout and capability checks; no duplicate adapter implementation. |
| 6. Preprocessing removes small subjects | Native preprocessing can reduce an annotated object below one pixel. The read-only preprocessing report already measures audited geometry and bbox resolution. | Preserve native preprocessing. Explain that geometric retention or an exact-color heuristic does not establish semantic subject preservation. |

### Position-concentration evidence

On a fixed analytical fixture with 48 tokens, six layers, one final query and
47 selected visual keys, uniform causal transitions yield a first/last-third
mass ratio of 16.546 for rollout and 37.338 for TAM. Uniform bidirectional
transitions give 1.0. Changing allowed edge weights under the same causal mask
reduces the ratios to 1.635 and 3.012. Gradient-weighted propagation fixtures
also retain the asymmetry; target conditioning alone does not remove it.

A separate cached pretrained Qwen2.5-VL-3B probe uses original, uniform-gray and
seeded-noise versions of one source image. All three inputs have 99 total
tokens, 54 visual tokens in a 6×9 grid, the same last-prompt query and 36 layers.
With float32 execution, `highest` matmul precision and a 224² pixel cap,
rollout top/bottom mass ratios are 7.248, 1.584 and 6.100. The maps also change.
This supports a real concentration risk with retained content dependence; it is
not a reproduction of the feedback's 90-image medians or a universal model
ranking. A two-point token-count comparison also does not establish a general
scaling law.

`vlmprobe.attention.attention_position_diagnostics` accepts explicitly selected,
finite nonnegative visual scores `[..., N]` in native order. It returns raw
sequence-third masses and ratios, with optional row-third and native tile-group
statistics. Unequal tile sizes have separate mass and mean-per-token ratios.
It does not infer image geometry, subtract a floor, normalize a heatmap or
correct the original methods. Zero/zero remains undefined; positive/zero is
infinite. Concentration by itself is not proof of a causal-mask effect or of
localization. Independent review also exercises numerical range, leading
dimensions, signed-input rejection and save/load behavior.

The diagnostic accumulates raw masses in FP64 and computes each ratio with a
common numerical scale for the two compared bands. This prevents a large but
finite uniform input from becoming an undefined ratio through sum overflow,
without modifying input scores or returning normalized masses. Tests include
large FP32/FP64 values, tiny scales and subnormal values, as well as true zero
mass and one-sided support.

### Remaining Jacobian defects repaired

- **Truncated resume input:** a completed checkpoint could be marked complete
  again when passed fewer batches, including an exhausted iterator. Resume now
  requires visiting the entire recorded prefix and rejects a shorter iterable
  before replacing existing artifacts. Equal-prompt weighting and Jacobian
  mathematics are unchanged.
- **Tensor selector identity:** valid boolean token masks were stored directly
  in settings, causing final JSON serialization to fail after expensive VJPs;
  equivalent masks could also fail resume comparison. Settings now store a
  JSON-safe selector identity with values, shape and dtype. Equivalent cloned
  masks resume, changed masks reject, and old Tensor-bearing checkpoints can
  recover their completed calculations. Reports expose resumed/new batch and
  prompt counts.

### Native geometry checks

Actual cached processor/config probes on a 512×256 input produce:

| Model | Native pixel tensor | Verified visual geometry |
| --- | --- | --- |
| ViT-B/16 | `[1,3,224,224]` | 14×14, 196 tokens, excluding CLS |
| LLaVA-1.5 | `[1,3,336,336]` | 24×24, 576 tokens |
| InternVL3-1B | `[3,3,448,448]` | Three tile-local 16×16 grids, 768 tokens |
| Qwen2.5-VL-3B | `[648,1176]` | Native merged 9×18 grid, 162 tokens |
| SmolVLM/Idefics3 | `[1,9,3,512,512]` | Automatic mapping explicitly unavailable; no invented grid |

The native LLaVA CLIP path with caller-applied square padding reduces a 4×4
source bbox in a 2048×256 image to 0.65625×0.65625 output pixels. Its geometric
retained-area fraction is still 1.0. Thus geometric containment does not mean
that the model receives enough resolution to recognize the subject. The
diagnostic reports caller padding separately and does not alter supplied pixels.

## Differences from v2

- Add the explicit attention concentration tool and corpus calibration fixes
  above; reuse existing repairs rather than implementing old defects twice.
- Retain the real pretrained MJ notebook and its independent calibration,
  native/plain controls, failed examples and signed vocabulary readouts.
- Update API, calibration, model/method and visualization documentation. Replace
  stale random-demo descriptions with the measured pretrained workflow.
- Add MJ-Lens to the existing Lens branch of the editable SVG overview and
  regenerate matching PNG/PDF exports without changing the four-family layout.
- Update the project page's Lens card, notebook navigation, Janus/model count
  and measured MJ preview. Each repository's Pages build resolves its own
  navigation; the preview is copied from the verified experiment artifact.
- Keep both README bodies identical, using stable documentation links. Only
  the lab README receives a trailing **Pre-release version** Beta button to
  `gexinh/vlm-probe`, with no new section. The personal README omits that block.
- Preserve explicit promotion: personal `main` may carry a beta first; the lab
  updates only when the reviewed synchronization command runs. Validate both
  histories, reject unmerged changes and verify identical non-README trees.

## Expected experimental and integration results

- Causal and bidirectional fixtures should distinguish structural concentration
  while allowing content-dependent changes. Diagnostics must preserve raw
  scores, declared geometry and undefined zero-mass cases.
- Failed/shortened calibration resumes must leave existing artifacts intact;
  valid selectors must serialize and resume without changing the estimator.
- Existing grid, Qwen, clipping and preprocessing contracts must keep passing.
- The pretrained MJ example should retain its measured outputs and interpretation
  boundaries. These fixes do not imply improved MJ semantic accuracy.
- Both built sites must resolve their local assets and repository navigation,
  include the MJ guide/notebook/preview, and fit desktop and mobile viewports.
- Published Git trees must differ only in the managed README footer. Rendering
  and repeat synchronization must be idempotent; no force-push, automatic beta
  promotion or PyPI publication is part of this update.

The source distribution excludes downloaded models, calibration tensors,
private feedback PDF, temporary experiments, site builds and obsolete random
MJ preview images. Existing report contents are preserved.

## Validation

- **396 source tests pass**, with no skipped tests.
- **111 installed-wheel tests pass**, with no skipped tests, including MJ,
  Jacobian, native geometry/Qwen, plotting, positional diagnostics and import
  contracts. All 106 installed runtime files (97 Python files) match the frozen
  source; the verification environment does not fall back to source imports.
- Wheel and sdist `1.1.0.dev2` build and pass strict Twine checks.
- Static documentation validation passes: 21 method guides, 13 model guides,
  37 Python blocks, 20 example scripts, 10 notebooks and 95 factory parameters.
- Both repository-specific Pages builds pass desktop (1440 px) and mobile
  (390 px) browser checks: no horizontal overflow, broken images or JavaScript
  errors; ten notebook links and the MJ guide resolve to the intended repository.
- The measured MJ notebook retains its nine executed code cells, four figures
  and zero execution errors. The standalone source and notebook code agree.
- Eighteen temporary bare-repository tests verify the footer-only README
  difference, migration, preserved histories, explicit beta promotion,
  synchronization idempotence and divergence/dirty-tree safeguards.

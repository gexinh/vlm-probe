# MJ-Lens pretrained report-case walkthrough, v2

## Goal and baseline

Replace the public random-model MJ-Lens illustration with a complete experiment
on a real pretrained VLM and the technical report's original held-out examples.
Readers should be able to inspect the raw input, calibration protocol, native
answer, readout, controls and visualization without replaying saved probe data.

This version starts from `f45879c6cddb007d9a6bb47b726a65b6a1bd96eb`
(`feat/mj-lens`, `1.1.0.dev0`). Development is isolated on
`feat/mj-lens-pretrained-demo`; the package version is `1.1.0.dev1`.

## Differences from v1

- Replace the random Janus public notebook and its three preview plots with a
  native Qwen2.5-VL-3B-Instruct workflow. Random native-model fixtures remain in
  unit tests for numerical contracts, rather than being presented as results.
- Reuse all eight fixed report cases: two each from GQA, MMVP, VSR and
  VisOnlyQA. Preserve the original questions, answer instructions and images;
  include raw-file checksums and dataset provenance. Draw only supplied boxes.
- Estimate separate visual-to-text and text-to-text MJ operators on 32
  independent COCO image/caption pairs, with no showcase-image overlap. The
  `train` label is an internal partition of official COCO val2014.
- Pin the actual backbone revision
  `66285546d2b821cf421d4f5eb2576359d3770cd3`. Use frozen float32 weights,
  eager attention, seed 0, TF32 matmul precision `high`, block inputs
  8/16/24/35, 64 output dimensions per exact VJP call, 224² calibration pixels,
  448² evaluation pixels, and at most 24 caption tokens plus EOS.
- Keep every execution step visible in `demos/10_mj_lens.ipynb` and
  `examples/tutorials/lens_mj.py`. Empty output directories calibrate anew;
  matching checkpoints resume explicitly. Held-out forwards, generation,
  readouts, measurements and figures are always recomputed.
- Add reusable spatial vocabulary-readout and text-state comparison plots to
  `vlmprobe.visualization`, with explicit geometry, signed scores, layer IDs,
  native/plain controls and input validation.
- Expose decoded vocabulary tokens, reference-versus-foil margins, answer
  ranks, native-output KL and native top-20 overlap. A reliable-looking heatmap
  or stable calibration operator cannot substitute for these controls.

## Issues found through the real-model run

1. **Teacher-forcing metadata alignment.** Appending caption IDs to processor
   output left `mm_token_type_ids` at the original length under Transformers
   5.3.0, causing an indexing error. The tutorial now processes the complete
   native prompt/caption/EOS sequence and verifies prefix tokenization and all
   sequence-aligned metadata. This fixes input preparation without changing MJ
   mathematics.
2. **Question-swap numerical control.** The first control shortened the question
   suffix, changing matrix shapes under TF32. Its maximum visual-logit change
   was 0.07256 (0.00189 relative to the original maximum magnitude). The final
   control keeps question lengths, image tokens, image pixels and native grids
   identical, varying only later question words. It reports the measured
   difference rather than treating floating-point variation as future-text
   influence.
3. **Resume timing interpretation.** Calibration reports now identify elapsed
   time as belonging to the current call and separate resumed/new prompt and
   batch counts. A complete resume must not be reported as a fresh-fit runtime.

## Expected experimental results

The frozen model should run on every raw report input. MJ and plain readouts
must use the same declared block-input states and vocabulary; save/load must
preserve logits. Calibration must exclude held-out images, and resumed
accumulators must retain their identity and settings checks.

Visual-to-text maps describe answer-token readouts of decoder image states.
Those states precede the question in this native causal template. They are not
question-conditioned attention or causal grounding. Text-to-text readouts at
the last prompt state can use the complete image/question input. Neither
readout is required to outperform the native model or plain readout. Qwen's
text-output experiment makes no image-generation or modality-index claim.

## Measured results and scope

The initial full run completed both 32-image calibrations and all eight held-out
cases in approximately 14.3 minutes on one otherwise idle A100 80 GB. Each
channel's calibration input loop took approximately 419 seconds. A subsequent
notebook execution resumed those completed, identity-checked accumulators and
recomputed the held-out analysis; it did not load cached showcase results.

Split-half operator cosines range from 0.929 to 0.988 for V→T and 0.932 to 0.999
for T→T. Nevertheless, intermediate text readouts often predict unrelated
tokens and have near-zero native top-20 overlap. Stability is not semantic
validity.

On these eight fixed cases, the native model answers five correctly. At block
input 35, MJ's top token matches four reference answers, compared with five for
the plain same-state readout. MJ reverses the native model's correct yellow
shirt answer. Mean native top-20 overlap at this layer is 0.531 for MJ versus
0.406 for plain, while mean KL is worse for MJ (0.446 versus 0.268). The two
fidelity measurements therefore do not support an overall improvement claim.
Saved/reloaded text transports reproduce the measured logits exactly.

The length-matched later-question swap also changes visual readout logits by
exactly zero in the final execution. This supports the declared causal-order
interpretation under the matched-input control; it does not establish semantic
validity of the visual readouts.

These are results for one pinned pretrained checkpoint, one caption-calibration
protocol and eight fixed report cases. They are neither dataset-level accuracy
estimates nor a reproduction of the paper's benchmark or causal experiments.

## Validation

- All **383 source tests** and **59 installed-wheel regression tests** pass.
- Wheel and sdist builds pass strict Twine checks. All 96 installed Python
  files match the reviewed runtime source, with no source-import fallback.
- The final notebook executes all **9 code cells**, renders **4 figures** and
  contains **0 error outputs**. Its source matches the standalone tutorial.
- Static validation passes for 21 method guides, 13 model guides, 36 Python
  blocks, 20 example scripts, 10 notebooks and 95 documented factory parameters.
- `docs/assets/tutorials/mj-report-execution.json` and `demos/validation.json`
  record the pinned model, calibration identities, measured results, figure
  hashes, source hash and notebook execution scope.

No backbone weights, calibration tensors, downloaded calibration image cache,
paper PDF or obsolete random preview is added to the public source tree.

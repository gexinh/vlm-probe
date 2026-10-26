# Region decoding and pretrained SAE tutorials, v4

## Goal and baseline

Make the MJ-Lens walkthrough answer the intended question: which vocabulary
tokens are read from the image region inside an annotated bounding box at
different decoder layers? Add a live pretrained visual-SAE tutorial and expose
all notebooks in a prominent homepage table.

This local development version starts from reviewed release source
`a04ba1cbeefcf6e73216b5b1017fdb940c30c85c` (1.1.0), on
`feat/region-mj-sae-tutorials`. Its package version is `1.1.1.dev0`.
Publication to GitHub, Hugging Face and PyPI is not part of this iteration.

## Differences from the previous version

- Replace the main MJ answer-versus-foil heatmap with original photographs,
  native visual grids, target boxes, numbered overlapping patches and layerwise
  full-vocabulary Top-k readouts. Keep margins, text-state readouts and fidelity
  plots as supplementary analyses.
- Select every patch with positive area overlap with the processed bbox.
  Report each patch's original token IDs, decoded strings/reprs, logits and
  probabilities. Compute the region summary as an intersection-area-weighted
  mixture of full per-patch softmax distributions. Do not average hidden states,
  restrict the vocabulary to answer choices or hide unrelated/special tokens.
- Reference answers contribute only post-hoc ranks and probabilities. The
  decoder receives the original image/question, without drawn boxes or grids.
- Add a pretrained DINO/Prisma ReLU SAE notebook using real author weights,
  independent feature-selection images and three fixed held-out report images.
  No SAE or backbone weights are trained. Raw image processing, checkpoint
  loading, feature selection, evaluation and plotting remain visible.
- Port the necessary native DINO residual adapter and reusable SAE evaluation
  modules from the report worktree into the current library, without importing
  SAELens, ViT-Prisma or Overcomplete at runtime. Keep optional scikit-learn
  dependencies lazy and separate from the base PyTorch installation.
- Preserve the uncast backbone `native_activation` in SAE results. Reconstruction
  evaluation uses that reference when dictionary precision differs from the
  backbone; the existing dictionary-dtype `activation` field retains its meaning.
- Add an independent **Notebook Tutorials** table to the README, with notebook
  links, Colab launch buttons and compute requirements. Add a visible Colab setup
  cell that prepares the source checkout and dependencies while leaving local
  runs and pre-existing user directories safe.
- Keep existing MJ calibration artifacts and previous experiment outputs
  intact. Use a new output directory for this iteration and remove obsolete
  display-only code where it is no longer needed.

## Expected experimental results

The same bbox should select the same native patches at every chosen layer.
Top-k tokens should match the full vocabulary scores exactly, including
whitespace, punctuation, special tokens and semantically unrelated outputs.
Reference labels must not alter the patch selection or candidates. Four
selected, previously calibrated MJ layers are the default display; they are not
described as all 36 layers of the checkpoint.

MJ readouts should expose layer-dependent vocabulary distributions on identical
regions while retaining their modality and tap conventions. The tutorial does
not add ordinary Jacobian Lens as a VLM comparison. Useful words are an empirical outcome rather than a guarantee;
the tutorial must show failed or uninformative readouts honestly.

The SAE walkthrough should reproduce live sparse feature activity, reconstruction
and localization measurements from pretrained weights on the raw report images.
Feature IDs do not acquire concept names merely by being selected. Spatial
association, reconstruction quality and intervention fidelity are separate
measurements, and three examples are not a dataset benchmark.

The new notebook must execute end to end without replaying cached showcase
results or training new weights. Existing workflow measurements are retained;
new setup-cell verification is recorded separately rather than claiming that
every older experiment was rerun. Colab badges are launch links, not a claim
that hosted Colab execution has been tested before publication.

## Measured results and verification

Both new workflows were executed from original photographs with the current
source, including the environment setup cell. Each completed 10 code cells with
zero errors. MJ produced seven inline figures; pretrained SAE produced two.
Nine older workflows and their stored outputs are byte-for-byte unchanged as
cell objects after excluding the two newly inserted setup/guide cells. Their
historical execution records are retained separately from setup validation.

- **MJ region decoding:** Qwen2.5-VL-3B, independently calibrated on 32 COCO
  image/caption pairs, with fresh inference on eight report cases. The two GQA
  boxes select 12 and 30 native patches at all four displayed layers. Early and
  middle readouts include ` yellow`, `黄色`, `黑色` and ` black`, alongside other
  tokens; block input 35 includes unrelated tokens, which remain visible.
  Exact capitalized QA-token rank is one vocabulary-ID diagnostic, not a
  synonym-pooled concept score. Resumed calibration records zero new fitting
  batches; the reported notebook runtime is not the cost of fitting from scratch.
- **Frozen visual SAE:** DINO-B/16, Prisma block-11 ReLU, 768→49,152 dimensions.
  Thirty-two official COCO train2017 images select features and references;
  three val2017 report photographs remain held out by ID and SHA-256. Training
  steps: zero. Source/native formula maximum absolute error is `9.54e-7`;
  no-op and artifact reload changes are zero. FVU on 588 patch positions is
  `0.00591635`, with mean L0 `1085.19`. All three selected map maxima fall
  inside the annotated regions; three examples are a walkthrough, not a benchmark.
  Final-layer patch deletion leaves CLS unchanged and changes pooled patch
  representations. These representation effects are not VQA answer effects.
- **Regression checks:** full source pytest suite: **651 passed, 2 opt-in skipped**,
  with 425 subtests passed. The source unittest suite also passed 422 tests.
  Optional dependencies remain lazy; imports are tested while blocking plotting,
  Transformers, scikit-learn and the three upstream SAE toolkits.
- **Documentation:** 11 notebook setup cells are synchronized; static tutorial
  validation passes. Local project-page checks at desktop and mobile widths
  find no broken images, horizontal overflow or JavaScript errors. These are
  local checks; hosted Colab was not executed and new remote links await publication.
- **Packaging:** the `1.1.1.dev0` wheel and source archive build and pass strict
  Twine metadata checks. The installed wheel matches all 106 Python source files;
  installed API, importer, region visualization and evaluation regressions pass.

## Additional fixes established by the live walkthrough

Historical Prisma files may envelope tensors with a pickled config class.
The importer maps only that known class name to an inert data container through
PyTorch's restricted loader, verifies embedded versus sidecar inference fields,
and rejects unknown classes, malicious reducers and mismatched configurations.
It never uses `weights_only=False` or imports the official toolkit. This path
requires PyTorch ≥2.6, now declared by the `sae` extra; the base minimum stays 2.2.

An existing GPT-2 automatic-pretrained-loader failure was reproduced on the
unchanged 1.1.0 baseline in an isolated source tree. `Prober` installs attention
support taps, but the strict reference model had no taps, so the loader rejected
its own supported model. Both sides now receive the same audited instrumentation
before complete structure comparison. Additional or replaced tap modules still
fail validation. Tests also retain weight, tokenizer and checksum mismatch checks.

The old documentation test assumed a removed Markdown methods table. It now
checks the current SAE button section for all method destinations. No reader-facing
methods layout was reverted.

## Artifacts and publication state

- Main notebooks: `demos/10_mj_lens.ipynb`, `demos/11_sae_pretrained.ipynb`.
- Full live code: `examples/tutorials/lens_mj.py`, `examples/tutorials/sae_pretrained.py`.
- Public execution scope: `docs/assets/tutorials/notebook-tutorials-validation.json`
  and `demos/validation.json`.
- Output root: `/bigdata/gexin/vlm_probing/outputs/tutorials_region_sae_v4_20261025`.
  `outputs/mj_bbox_readouts` and `outputs/sae_pretrained` are ignored local links
  to the new run artifacts. Prior MJ calibration and report outputs remain intact.

This work remains on the local development branch. GitHub, Hugging Face and PyPI
were not updated in this iteration.

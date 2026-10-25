# Live walkthroughs

Each notebook starts with raw text or images, loads a model, prepares native
inputs and prompts, calls `vlmprobe`, inspects the returned tensors and draws
figures with `vlmprobe.visualization`. Calibrated lens notebooks also show the
independent training/validation split and fitting code. No notebook substitutes saved showcase
probe tensors or pre-rendered figures for live method execution.

## Start here

Clone the repository to obtain the notebooks and attributed raw input samples:

```bash
git clone https://github.com/ubc-tea/vlm-probe.git
cd vlm-probe
python -m pip install -e ".[transformers,visualization,notebooks]"
jupyter lab demos
```

Open a notebook and run its cells in order. The first run downloads its model
if necessary. Most notebooks select CUDA when available; GPT-2 also runs on CPU.
Most VLM walkthroughs use SmolVLM-500M and one image. MJ-Lens uses
pretrained Qwen2.5-VL-3B, independent COCO caption calibration and the report's
eight fixed GQA/MMVP/VSR/VisOnlyQA examples.
Integration, exact Jacobians and dense Attention Lens fitting cost more than
ordinary readouts; the relevant cells state their cost and any explicit opt-in.

| Notebook | Default input/model | Visible workflow |
| --- | --- | --- |
| [Text lenses](lens_comparison_demo.ipynb) | IOI text; pretrained GPT-2 | Tokenization → Logit → independent Tuned calibration → readouts and plots; complete optional Attention/Jacobian fitting and Patchscope. |
| [VLM lenses](vlm_lens_demo.ipynb) | GQA image/question; SmolVLM-500M | Image/chat processing → Logit and EmbedLens → independent COCO-image calibration → Tuned and Patchscope → plots. |
| [MJ-Lens](10_mj_lens.ipynb) | Eight report image/questions; pretrained Qwen2.5-VL-3B | Verify raw inputs → download 32 independent COCO image/captions → native teacher forcing → V→T and T→T calibration → fresh readouts, spatial vocabulary maps and native/plain fidelity controls. |
| [ViT attention](attention_comparison_demo.ipynb) | Raw image; pretrained ViT-B/16 | Processed image/grid → fixed class score → attention maps including DTD/LRP → overlays. |
| [VLM attention](vlm_attention_comparison_demo.ipynb) | GQA image/question; SmolVLM-500M | Chat input → teacher-forced full-answer score → eight attention analyses → aligned image overlays. |
| [Activation and attribution patching](activation_patching_demo.ipynb) | Published IOI pair; GPT-2 | Explicit base/donor tokenization → fixed answer margin → actual versus estimated replacement effects → curves. |
| [Attention Knockout](attention_knockout_demo.ipynb) | GQA image/question; SmolVLM-500M | Input and complete-answer score → image/text route blocking across layers → measured score changes and plot. |
| [Path Patching](path_patching_demo.ipynb) | Published IOI pair; GPT-2 | Base/donor inputs → selected sender/receiver heads → controlled replay, identity check and path plots. |
| [Residual Steering / VSV](steering_vsv_demo.ipynb) | COCO snowboarding photo; SmolVLM-500M | Image/text-only pair → per-image direction → live generation at several strengths → captions and score curves. |
| [EAP / EAP-IG](eap_ig_demo.ipynb) | Published IOI pair; GPT-2 | Token alignment and margin → explicit transformer-edge attribution → actual retained-circuit evaluation → plots. |

The short calibration splits teach the training protocol; they do not produce
research-quality decoders. Keep showcase inputs out of fitting, and use the
[training guide](../docs/TRAINING.md) to scale with independent document/image
splits and immutable artifact identities. The default text-lens path skips the
expensive dense Attention Lens and exact Jacobian sections explicitly; their
complete training code remains visible and runs when enabled.

## MJ-Lens report walkthrough

[MJ-Lens](10_mj_lens.ipynb) uses the original questions and images recorded in
[the report input manifest](assets/mj_report_samples.json). GQA target boxes
are included where the source provides them. The six additional MMVP, VSR and
VisOnlyQA images are unchanged raw inputs under
[images/mj_report](assets/images/mj_report/SOURCE_NOTICES.md).

The [calibration manifest](assets/mj_calibration_coco.json) lists 32 independent
COCO val2014 image/caption records and their download URLs and byte hashes.
Its internal `train` partition is not official COCO train2014. Raw calibration
images download to `VLM_PROBING_DATA_CACHE`, or
`~/.cache/vlmprobe/coco_calibration`; model weights and fitted probe tensors are
not embedded in the notebook. Showcase images are excluded by hash and COCO ID.

Exact Jacobian calibration is more expensive than a forward-only lens. The
visible configuration exposes the calibration count, source layers and VJP
batch size through `VLM_PROBING_MJ_CALIBRATION`, `VLM_PROBING_MJ_LAYERS` and
`VLM_PROBING_MJ_DIM_BATCH`. A different calibration corpus or set of layers
requires a distinct output directory/artifact identity; VJP batch size controls
memory and work scheduling without changing the estimator. The
notebook shows its run configuration before fitting and computes new showcase
results after calibration or an explicitly recorded calibration resume.

Its spatial `V→T` maps read image-token states in the decoder. In the native
image-before-question template those states cannot see the later question;
only the vocabulary contrast is chosen from the question. The `T→T` last-prompt
readout is question-conditioned. Qwen has no visual output codebook, so this
walkthrough does not label its maps as V→V image reconstructions or MI results.

## Change a checkpoint or input

Notebook configuration uses the same optional environment names:

```bash
export VLM_PROBING_DEVICE="cuda:0"
export VLM_PROBING_MODEL="HuggingFaceTB/SmolVLM-500M-Instruct"
```

Set the model override only for a notebook using that model family; unset it
before opening a GPT-2 or ViT notebook. `VLM_PROBING_OUTPUT` selects the output
directory. Each notebook exposes its raw input and prompt in code. The
[model guides](../docs/models/README.md#use-any-supported-method) explain how to
replace model-specific loading/preprocessing while retaining the method API.
Do not reuse another checkpoint's tokenizer, fitted lens, visual grid or layer IDs.

The [method pages](../docs/methods/README.md) give full API/default/constraint
tables and standalone scripts under [examples/tutorials](../examples/tutorials).
Use these when you want one method rather than a whole family comparison.

## Execution records and archived measurements

[validation.json](validation.json) records live walkthrough verification;
[the tutorial preview directory](../docs/assets/tutorials) contains figures
produced by the example code and their provenance. Stored notebook outputs are
for reading on GitHub; running the notebook computes fresh results.

The existing `snapshots/` directory retains historical report measurements with
source identities. These are reference archives, not a prerequisite, fallback
or source of computed results in the tutorials. Existing input/data helpers
support historical experiments; the walkthroughs show their essential input
and target construction directly.

## Data and provenance

| Source | Included samples and role |
| --- | --- |
| [GQA](https://cs.stanford.edu/people/dorarad/gqa/download.html), [Cross-modal Information Flow author subset](https://github.com/FightingFighting/cross-modal-information-flow-in-MLLM/tree/main/datasets) | Question 07302654 / image 2341576 / yellow shirt; question 19225230 / image 2342134 / black shirt. Original images, object boxes, and question metadata are bundled. |
| [COCO2014](https://cocodataset.org/#download), [VISTA POPE records](https://github.com/LzVv123456/VISTA/tree/main/pope_coco) | Images `COCO_val2014_000000310196.jpg` and `COCO_val2014_000000210789.jpg`. Image-specific source URLs and hashes are in `assets/coco_samples.json`. |
| [MMVP](https://huggingface.co/datasets/MMVP/MMVP) | Original evaluation images 1 and 2 and their open/closed-wing questions; the source imagefolder calls its split `train`. |
| [VSR zero-shot](https://huggingface.co/datasets/cambridgeltl/vsr_zeroshot) | Official test rows 0 and 1, their COCO photographs and spatial propositions. No bounding boxes are inferred. |
| [VisOnlyQA Eval Real](https://huggingface.co/datasets/ryokamoi/VisOnlyQA_Eval_Real) | Row 0 from `geometry__triangle` and `charts__extraction`, retaining official no-reasoning prompts and exact source image hashes. |
| COCO val2014 MJ calibration | 32 image/caption records from an independent internal calibration partition. The lightweight manifest includes download URLs, caption IDs, hashes and explicit showcase-overlap checks. |
| [Transformer Explainability author samples](https://github.com/hila-chefer/Transformer-Explainability/tree/main/samples) | `catdog.png` and `el2.png`, plus their exact processed 224×224 crops. These are illustrative author samples, not a claimed held-out ImageNet split. |
| [Author IOI pairs](https://github.com/hannamw/eap-ig-faithfulness/tree/main/data/ioi) | Published GPT-2 CSV rows 826 and 475, preserving the author clean/corrupted prompts and BOS convention in the EAP-IG experiment. |
| [VRUBench paper examples](https://arxiv.org/abs/2604.15294) | Appendix A two-step and Figure 7(c) four-step examples with attributed prompt text and paired changes. These are not a download of the full official benchmark. Release availability in the provenance file is a historical collection-time check. |
| [WikiText-2](https://huggingface.co/datasets/Salesforce/wikitext), BookCorpus | Historical report calibration sources; raw corpora are not bundled. The live text tutorial defines a separate small teaching split directly in code. |
| COCO2014 train / validation | Historical LLaVA calibration used 1,024 training and 256 validation images. The live VLM tutorial explicitly uses two separate bundled COCO images for a short training demonstration and a separate GQA showcase. |

[assets/manifest.json](assets/manifest.json) records the hashes of bundled
measurements/images and the original archive hashes. Machine-local cache paths
are redacted in portable snapshots; measured numbers are unchanged. Source
HTTP headers are omitted. The ViT snapshots point to image files rather than
embedding duplicate base64 PNGs.

Third-party photos and data retain their original rights; bundling does not
change their terms. See [image source notices](assets/images/SOURCE_NOTICES.md).
Method papers and implemented scope are indexed in
[References](../docs/REFERENCES.md). The repository includes attributed input examples and historical measurement
archives; model weights, fitted checkpoints and full datasets are not bundled.

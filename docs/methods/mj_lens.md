# Multimodal J-Lens

[Home](../../README.md) / [Methods](README.md) / Multimodal J-Lens

Read intermediate decoder states through separate text-to-text,
visual-to-text, text-to-visual and visual-to-visual Jacobian transports.
Visual outputs require a model with a discrete image-generation codebook.

## Principle

MJ-Lens separates the source position's modality from the output channel.
For a prompt $p$, let $A_p^{m,n}$ contain the valid source/target pairs
$(t,t')$ with source modality $m$, output modality $n$, and $t' \geq t$.
The calibrated operator is

$$
J_\ell^{m\to n}
= \frac{1}{P}\sum_{p=1}^{P}
\frac{1}{|A_p^{m,n}|}
\sum_{(t,t')\in A_p^{m,n}}
\frac{\partial h_{\mathrm{final},t'}}{\partial h_{\ell,t}}.
$$

Each prompt receives equal weight after averaging its admissible pairs.
This differs from the existing [Jacobian Lens](jacobian_lens.md), whose
estimator sums current/future targets before averaging sources. The two
artifacts are distinct and cannot be substituted for one another.

For MJ-Lens, layer $\ell$ means the **input to decoder block** $\ell$;
the target is the input to the actual final RMSNorm. Other lens guides use
post-block residual indices. Read `probe.describe()["methods"]["lens.mj"]`
to select the MJ block-input sites for a model.

An understanding model supplies $J^{T\to T}$ and $J^{V\to T}$, decoded by
its actual final normalization and text vocabulary head. A unified model
with a discrete image head additionally supplies $J^{T\to V}$ and
$J^{V\to V}$. A visual encoder or an image-input marker does not by itself
provide an image-generation alphabet.

For the supported two-layer GELU image head,

$$
\mathrm{head}(x)=W_2\,\mathrm{GELU}(W_1x+b_1)+b_2,
\qquad
W_{\mathrm{eff}}=W_2\,\mathrm{diag}\!\left(
\frac{1}{N}\sum_{t=1}^{N}\mathrm{GELU}'(W_1x_t+b_1)
\right)W_1.
$$

Calibration averages the derivative at sampled normalized visual states;
it does not differentiate once at their mean. The default visual readout
is the paper's bias-free $W_{\mathrm{eff}}\,\mathrm{norm}(Jh)$.
`native(...)` retains the full nonlinear head, and `affine(...)` adds a
calibration-mean offset for fidelity controls. Neither changes the default
MJ readout. Calibrate one visual readout and reuse it across output contexts.

**Analysis question:** How do modality-specific transport and the choice of
output alphabet change what an intermediate decoder state can reveal?

## API and parameters

The following signatures describe arguments. The complete scripts below
construct the models, inputs and methods before calling them.

```text
method = probe.lens.mj(
    *, layers=None, source="visual", target="text",
    visual_readout=None, context=None, binding=None,
)
method = method.fit(inputs, *, dim_batch=32, skip_first=0, exclude_last=False)
method = method.fit_batches(
    input_batches, *, directory=None, dim_batch=32, resume=False,
    checkpoint_every=8, skip_first=0, exclude_last=False,
)
result = method.run(inputs, *, tokens=None)
dictionary = method.dictionary(layer, *, token_ids=None)
method.save(directory)
method = method.load(directory)
```

| Parameter | Default | Meaning and constraints |
| --- | --- | --- |
| `layers` | `None` | All declared MJ block-input indices. An integer or unique nonempty list selects a subset. |
| `source` | `"visual"` | `"visual"` or `"text"`: modality of states being read and of calibration source positions. |
| `target` | `"text"` | `"text"` or `"visual"`: actual output channel. Visual targets require a declared image-generation head and calibrated `visual_readout`. |
| `visual_readout` | `None` | `LinearizedVisualReadout` built from this model's exact final norm and native image-head modules. Required for visual output. |
| `context` | `None` | Defaults to `"text_output"` for a text target and `"image_output"` for a visual target. `MJBatch.context` must match. Context names identify separate calibration protocols. |
| `binding` | `None` | Identity mapping. Saving/resuming requires explicit `model_revision`, `tokenizer_id` and `calibration_id`; visual artifacts also require `codebook_id`. Use immutable revisions/data hashes. The runner binds the source site, output channel and readout. |
| `inputs` / `input_batches` | Required | Native dictionaries, `ProbeInputs`, or `MJBatch` objects. A corpus iterable must be deterministic and re-iterable for resume. |
| `dim_batch` | `32` | Positive number of exact output-basis VJPs grouped per backward call. Changes memory/work scheduling, not an approximation rank. |
| `skip_first` | `0` | Nonnegative number of initial valid expanded positions excluded from calibration. |
| `exclude_last` | `False` | Exclude the last valid position from both source and target selections. |
| `directory` | `None` | Save resumable accumulation, prompt counts, split-half diagnostics and fitted artifacts here. Use a new directory for a new experiment. |
| `resume` | `False` | Resume only matching layer/modality/context/identity/exclusion settings. Inputs must be the same ordered corpus. |
| `checkpoint_every` | `8` | Positive number of input batches between resumable checkpoints. |
| `tokens` | `None` | By default read every valid source-modality position. A common selector or boolean mask can restrict them; selecting another modality raises an error. |
| `layer` | Required for `dictionary` | One fitted MJ block-input index. |
| `token_ids` | `None` | Optional nonempty integer subset of text vocabulary IDs or visual codebook IDs for dictionary construction. These are separate ID spaces. |

`fit(...)` replaces a previous estimate. Use `fit_batches(...)` to average
a dataset rather than calling `fit` repeatedly. Every prompt must contain an
admissible causal source/target pair after masking and exclusions. Short,
empty or disjoint selections fail explicitly. The model remains frozen;
this is derivative estimation, not SAE training or backbone fine-tuning.
Exact calibration still scales with hidden width and can be expensive for
large models. It must run outside `torch.inference_mode()`.

### Declaring output positions

```text
batch = MJBatch(inputs, target_masks={}, context="text_output")
```

| Parameter | Meaning |
| --- | --- |
| `inputs` | Native model inputs or `ProbeInputs` with expanded decoder coordinates. |
| `target_masks` | Optional `{"text": mask, "visual": mask}` with boolean `[batch, expanded_sequence]` tensors. Masks may not select padding. Text defaults to the layout's text positions; visual targets must always be explicit. |
| `context` | Nonempty protocol name matching the method's context. Non-text contexts require an explicit `MJBatch`. |

For teacher-forced Janus image generation,
`prepare_janus_generation(probe, prompt_inputs, codes)` builds this batch.
`prompt_inputs` must contain unpadded text prompt IDs and an optional all-one
attention mask. `codes` is a long `[batch, visual_positions]` tensor with at
least two actual visual code IDs. A state carrying code $i$ predicts code
$i+1$; the last code has no supplied successor and is excluded from targets.
The last text prompt state is not relabeled as a visual position.

### Visual readout calibration

```text
readout = fit_visual_readout(probe, image_output_batches, *, binding)
# Equivalently, with explicitly supplied native modules:
readout = LinearizedVisualReadout(norm, first_projection, activation, last_projection, *, binding=None)
readout = readout.fit(normalized_visual_states, *, mask=None)
readout = readout.update(normalized_visual_states, *, mask=None)
linear_logits = readout(raw_hidden)
native_logits = readout.native(raw_hidden)
affine_logits = readout.affine(raw_hidden)
readout.save(path)
readout = readout.load(path)
```

| Parameter or property | Meaning |
| --- | --- |
| `image_output_batches` | `MJBatch` objects with context `"image_output"` and actual visual output masks; the helper captures and normalizes final visual states. |
| `norm` | This model's exact final normalization. |
| `first_projection` / `last_projection` | Native `nn.Linear` image-head projections, including their original biases for calibration/native controls. |
| `activation` | Native `nn.GELU` (`"none"` or `"tanh"`) or the audited Transformers `GELUActivation`. Other nonlinear heads require their own implementation. |
| `binding` | Nonempty `model_id`, `readout_id`, `calibration_id`; any additional identity fields are retained and checked. |
| `normalized_visual_states` | Already normalized `[..., hidden]` states. `fit` resets; `update` accumulates positions with their actual sample counts. |
| `mask` | Optional boolean mask matching every axis except hidden; must select at least one position. |
| `raw_hidden` | Unnormalized hidden states. The callable and fidelity controls apply `norm` once. |
| `weight` / `dictionary_weight` | Effective `[visual_codebook, hidden]` matrix. |
| `mean_slope` / `n_samples` | Calibrated per-channel mean derivative / number of selected visual positions. |
| `path` | A calibration-only artifact; native model parameters are not serialized. Loading checks binding, dimensions, activation and exact native parameter fingerprint. |

## Complete pretrained VLM example: visual-to-text

Install `python -m pip install "vlmprobe[transformers,visualization]"` from a
version containing MJ-Lens. The native adapter targets Transformers 5.3.
This example downloads a pretrained Qwen model, draws its own input images,
calibrates on two independent image/prompt pairs, then reads a held-out image.
The tiny calibration set demonstrates usage, not benchmark quality.

```python
import hashlib
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from PIL import Image, ImageDraw
from transformers import AutoModelForImageTextToText, AutoProcessor
from vlmprobe import Prober
from vlmprobe.visualization import plot_lens_heatmap

MODEL_ID = os.environ.get("VLM_PROBING_MODEL", "Qwen/Qwen2.5-VL-3B-Instruct")
DEVICE = os.environ.get("VLM_PROBING_DEVICE", "cuda:0" if torch.cuda.is_available() else "cpu")
DTYPE = torch.bfloat16 if DEVICE.startswith("cuda") else torch.float32
OUT = Path("outputs/mj-qwen")
OUT.mkdir(parents=True, exist_ok=True)
torch.manual_seed(7)
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID, dtype=DTYPE, attn_implementation="eager",
).to(DEVICE).eval()
processor = AutoProcessor.from_pretrained(
    MODEL_ID, min_pixels=56 * 56, max_pixels=112 * 112,
)
probe = Prober(model, processor)
print(probe.describe()["methods"]["lens.mj"])

def circle_image(color, offset=0):
    image = Image.new("RGB", (112, 112), "white")
    ImageDraw.Draw(image).ellipse((28 + offset, 28, 84 + offset, 84), fill=color)
    return image

def prepare(image, question):
    messages = [{"role": "user", "content": [
        {"type": "image"}, {"type": "text", "text": question},
    ]}]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = probe.prepare(prompt=prompt, image=image, device=DEVICE)
    for key, value in inputs.kwargs.items():
        if isinstance(value, torch.Tensor) and value.is_floating_point():
            inputs.kwargs[key] = value.to(DTYPE)
    return inputs

calibration_cases = [("red", -4, "Describe the colored object."),
                     ("green", 4, "What color is the circle?")]
calibration_inputs = [prepare(circle_image(color, offset), question)
                      for color, offset, question in calibration_cases]
heldout = prepare(circle_image("blue"), "What color is the circle? Answer with one color.")
revision = str(getattr(model.config, "_commit_hash", None) or "local-checkpoint")
corpus_hash = hashlib.sha256(json.dumps(calibration_cases).encode()).hexdigest()
binding = {
    "model_id": MODEL_ID, "model_revision": revision,
    "tokenizer_id": processor.tokenizer.name_or_path + "@" + revision,
    "calibration_id": "drawn-circles-v1:" + corpus_hash,
}
layers = [probe.describe()["methods"]["lens.mj"]["sites"][-1]]
method = probe.lens.mj(layers=layers, source="visual", target="text", binding=binding)
method.fit_batches(calibration_inputs, directory=OUT / "calibration", dim_batch=16)
result = method.run(heldout)
print("positions:", result.tensors["positions"])
print("logits:", tuple(result.tensors["logits"].shape))
print("split-half:", method.calibration_report["split_half_cosine"])

# One subword's vocabulary rank, not a whole-word or concept correctness score.
answer_ids = processor.tokenizer.encode("blue", add_special_tokens=False)
target_id = answer_ids[0]
print("Displayed subword:", repr(processor.tokenizer.decode([target_id])))
ax = plot_lens_heatmap(result, target_token_id=target_id,
                       title="MJ visual-to-text: held-out circle, last block input")
ax.figure.savefig(OUT / "mj-text.png", dpi=160, bbox_inches="tight")
plt.show()
result.save(OUT / "heldout-result")

# Loading uses this library's fitted artifact; it performs no new calibration.
restored = probe.lens.mj(layers=layers, source="visual", target="text", binding=binding)
restored.load(OUT / "calibration")
torch.testing.assert_close(restored.run(heldout).tensors["logits"], result.tensors["logits"])
```

Select `source="text"` to calibrate the corresponding text-to-text transport.
Qwen2.5-VL and Qwen3-VL provide understanding-side readouts, not Janus-style
visual output codebooks. Change their native loading/input preparation as
described in the [model guides](../models/README.md); the MJ calls remain the same.
For reproducible releases, pin model/processor revisions and use a checksum
for local checkpoint files instead of the illustrative fallback identity.

## Unified model example: Janus visual output

This complete example uses the native Transformers checkpoint conversion
`deepseek-community/Janus-Pro-1B`. The original `deepseek-ai` repository's
custom `MultiModalityCausalLM` class is not this native adapter contract.
The example samples actual codebook outputs, fits the head on visual output
positions, then calibrates visual-to-visual transport. Generation is complete;
only a 16-code prefix is used for this small calibration demonstration.

```python
import hashlib
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from transformers import AutoProcessor, JanusForConditionalGeneration
from vlmprobe import Prober, fit_visual_readout, decode_mj_codes
from vlmprobe.adapters import prepare_janus_generation

MODEL_ID = "deepseek-community/Janus-Pro-1B"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE.startswith("cuda") else torch.float32
OUT = Path("outputs/mj-janus")
OUT.mkdir(parents=True, exist_ok=True)
model = JanusForConditionalGeneration.from_pretrained(
    MODEL_ID, dtype=DTYPE, attn_implementation="eager",
).to(DEVICE).eval()
processor = AutoProcessor.from_pretrained(MODEL_ID)
probe = Prober(model, processor)

def generate_replay(description):
    messages = [{"role": "user", "content": [{"type": "text", "text": description}]}]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    native = processor(text=prompt, generation_mode="image", return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        codes = model.generate(**native, generation_mode="image", do_sample=False,
                               guidance_scale=5.0, return_dict_in_generate=False)
    prompt_inputs = {"input_ids": native["input_ids"],
                     "attention_mask": native["attention_mask"]}
    return prompt_inputs, codes

descriptions = ["A red square on a white background.", "A green oval on a white background."]
training_pairs = [generate_replay(description) for description in descriptions]
training_batches = [prepare_janus_generation(probe, prompt, codes[:, :16])
                    for prompt, codes in training_pairs]
revision = str(getattr(model.config, "_commit_hash", None) or "local-checkpoint")
data_hash = hashlib.sha256("\n".join(descriptions).encode()).hexdigest()
binding = {
    "model_id": MODEL_ID, "model_revision": revision,
    "tokenizer_id": processor.tokenizer.name_or_path + "@" + revision,
    "codebook_id": MODEL_ID + "@" + revision + ":vqmodel",
    "calibration_id": "generated-shapes-v1:" + data_hash,
}
readout_binding = {"model_id": MODEL_ID + "@" + revision,
                   "readout_id": "model.generation_head",
                   "calibration_id": binding["calibration_id"]}
readout = fit_visual_readout(probe, training_batches, binding=readout_binding)
readout.save(OUT / "visual-readout.pt")
layers = [probe.describe()["methods"]["lens.mj"]["sites"][-1]]
method = probe.lens.mj(layers=layers, source="visual", target="visual",
                       context="image_output", visual_readout=readout, binding=binding)
method.fit_batches(training_batches, directory=OUT / "transport", dim_batch=16)

test_prompt, test_codes = generate_replay("A blue circle on a white background.")
heldout = prepare_janus_generation(probe, test_prompt, test_codes)
result = method.run(heldout)
side = model.config.vq_config.num_patches
images, grid_codes = decode_mj_codes(result, grid=(side, side),
                                    decoder=probe.spec.mj.decode_visual_codes)
print("code IDs:", tuple(grid_codes.shape), "decoded images:", tuple(images.shape))
# The native Janus decoder returns channel-last images in [-1, 1].
plt.imshow(((images[0, 0].float().cpu() + 1) / 2).clamp(0, 1))
plt.axis("off")
plt.savefig(OUT / "mj-visual.png", dpi=160, bbox_inches="tight")
plt.show()
```

Use a larger, independent rollout corpus and held-out comparisons with
`readout.native(...)` before interpreting a visual reconstruction. The visual
head and Jacobian are separate approximations. A complete code grid must come
from distinct visual source positions; `decode_mj_codes` rejects text-source
results and mismatched grids. It never tiles one text state to create an image.

## Dictionaries and controlled modality index

For final RMSNorm gain $g$, the text and visual dictionaries are the row atoms
of $U_T=W_U\,\mathrm{diag}(g)J^{m\to T}$ and
$U_V=W_{\mathrm{eff}}\,\mathrm{diag}(g)J^{m\to V}$. Both use the same source
layer, modality and hidden coordinates. Dictionary construction keeps the
learned gain; actual logits additionally keep the input-dependent RMS division.
The gain-only dictionary formula is not a generic LayerNorm substitution.

The sparse nonnegative decomposition and raw modality index are

$$
h\approx\sum_i\alpha_i u_i,\qquad \alpha_i\geq0,\quad
\|\alpha\|_0\leq K,\qquad
\mathrm{MI}(h)=\frac{\sum_{i\in V}\alpha_i\|u_i\|}
{\sum_i\alpha_i\|u_i\|}.
$$

```text
result = modality_index(
    activations, text_dictionary, visual_dictionary,
    *, k=25, seed=0, size_match=False, signed=False,
)
```

| Parameter | Default | Meaning |
| --- | --- | --- |
| `activations` | Required | Floating `[..., hidden]` source activations. |
| `text_dictionary` / `visual_dictionary` | Required | Floating `[atoms, hidden]` row dictionaries expressed in exactly those source coordinates. |
| `k` | `25` | Positive maximum number of selected atoms. |
| `seed` | `0` | Local deterministic seed for the orthogonal visual rotation and optional text subset; does not reset global RNG state. |
| `size_match` | `False` | Additionally subsample text atoms to the visual atom count. Requires at least as many text atoms; returns separate controls without replacing primary results. |
| `signed` | `False` | Must remain `False`; signed coefficients do not implement this index. Negative activation/atom coordinates remain valid. |

Import the helper with `from vlmprobe.lenses import modality_index`. Construct
the dictionaries with the two fitted methods' `.dictionary(layer)` calls.
The implementation normalizes atoms, selects nonnegative OMP supports, and
performs 64 projected-gradient NNLS iterations per support. Coefficients in
the result are converted back to original atom units. This finite solver is
an explicit library choice; the paper does not specify an exact solver.

Always report `excess_mi = raw_mi - rotated_mi`. The visual control applies a
random orthogonal rotation that preserves its row norms and Gram matrix.
`raw_mi` alone measures dictionary allocation and is not evidence of visual
semantic content. Zero reconstruction mass gives NaN with validity masks.
Also inspect the size-matched result and repeat the analysis with independently
calibrated odd/even prompt operators. The calibration report's split-half cosine
is a numerical diagnostic; it is not the paper's split-half MI/context comparison.
Large dictionaries and many positions can require substantial memory; select
an explicit modest set of source positions before decomposition.

## Outputs and visualization

`method.run(...)` returns `ProbeResult("mj_lens", ...)` with
`logits[L,N,V]`, `transported[L,N,D]`, `positions[N,2]` and layer/modality/context
metadata. `positions` uses `[batch, expanded_decoder_position]`; image tokens
are expanded model coordinates. Visual output adds `code_ids[L,N]`, whose
IDs belong to the visual codebook and cannot be passed to the text tokenizer.

`modality_index(...)` returns raw/rotated/excess MI, validity masks,
reconstructions, nonnegative coefficients, atom IDs and visual-atom masks.
Inactive support entries have ID `-1`. Size-matched atom IDs index its reduced
union; `size_matched_text_ids` maps text atoms back to the full dictionary.

The [offline executable tutorial](../../examples/tutorials/lens_mj.py) and
[notebook](../../demos/10_mj_lens.ipynb) walk through model construction,
input preparation, calibration, held-out inference, artifact loading and
visualization without reading cached probe results. Its small models are
analytic integration examples, not pretrained-model findings.

![MJ text readout from the executed tutorial](../assets/tutorials/lens-mj-text.png)

![MJ visual codebook readout from the executed tutorial](../assets/tutorials/lens-mj-visual.png)

The Qwen and Janus loaders above provide pretrained execution paths; they do
not establish pretrained calibration quality. No external pretrained MJ
artifact is bundled or downloaded by `.load(...)`: it loads this library's
identity-bound calibration files. Neither a vocabulary readout, reconstruction
nor modality index establishes a causal concept or a global-workspace claim.

[Paper](https://openreview.net/forum?id=Xrp8mNku6R) ·
[Tensor implementation](../../src/vlm_probing/lenses/mj.py) ·
[Visual readout and controls](../../src/vlm_probing/lenses/mj_readout.py) ·
[Dataset calibration](../../src/vlm_probing/training/mj.py)

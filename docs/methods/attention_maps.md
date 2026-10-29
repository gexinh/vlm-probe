# Target-conditioned attention maps

[Home](../../README.md) / [Methods](README.md) / Attention maps

Explain a fixed model score using attention probabilities, gradients, and information propagated across layers. No probe training is required. Each method below computes a different approximation; the same target, input and geometry make their differences visible.

## API and every parameter

`probe` is a `Prober` bound to a loaded model; `inputs` contains prepared model inputs. The blocks below are API signatures: `*` marks keyword-only parameters, not an argument to copy. The [complete example](#complete-example-raw-image-to-five-fresh-maps) defines the model, inputs, and all other variables before running the method.

```text
grad_cam = probe.attention.grad_cam(*, layers=None, queries=None, keys=None, normalize=False)
grad_cam_result = grad_cam.run(inputs, *, metric=None)

attattr = probe.attention.attribution(
    *, layers=None, queries=None, keys=None, steps=20, quadrature="right",
)
attattr_result = attattr.run(inputs, *, metric=None)

tam = probe.attention.tam(
    *, layers=None, queries=None, keys=None, steps=20,
    quadrature="right", input_key="pixel_values", start_layer=0,
    residual_normalize=True,
)
tam_result = tam.run(inputs, *, metric=None, baseline=None)

beyond = probe.attention.beyond_intuition(
    *, variant="head", layers=None, queries=None, keys=None, steps=20,
    quadrature="right", input_key="pixel_values", start_layer=0,
)
beyond_result = beyond.run(inputs, *, metric=None, baseline=None)
```

| Parameter | Default | Methods and meaning / constraints |
| --- | --- | --- |
| `layers` | `None` | All four: available attention layers. Grad-CAM/ATTATTR accept any unique nonempty supported subset. TAM/Beyond require `[0,...,n]` in forward order; gaps are rejected. |
| `queries` | `None` | All four: CLS `[0]` for classification, `"last_prompt"` for language/VLMs. Explicit token selector changes query rows to explain. |
| `keys` | `None` | All four: `"visual"` for classification, `"all"` for language/VLMs. Unselected map entries are zero; original probabilities are not renormalized over this selection. |
| `normalize` | `False` | Grad-CAM only: per-query min-max normalize the displayed CAM. Raw scores are preserved by default. A constant/zero map remains zero. |
| `steps` | `20` | ATTATTR/TAM/Beyond: positive integer integration sample count. More samples cost more forwards; check convergence. Endpoint-inclusive sampling requires at least 2. |
| `quadrature` | `"right"` | ATTATTR/TAM/Beyond: `"right"`, `"left"` Riemann samples, or `"endpoints"` equally weighted samples including 0 and 1 (not trapezoidal integration). |
| `input_key` | `"pixel_values"` | TAM/Beyond only: name of the floating native input tensor to interpolate. Integer token IDs cannot be interpolated. All other input fields remain fixed. |
| `start_layer` | `0` | TAM/Beyond only: zero-based inclusive start within the selected consecutive stack; `0 <= start_layer < len(layers)`. Captured preceding layers remain available but are skipped in propagation. |
| `residual_normalize` | `True` | TAM only: use half-normalized residual transitions. `False` matches the official code's transition scale rather than the paper convention. |
| `variant` | `"head"` | Beyond only: `"head"` weights heads using attention gradients; `"token"` uses projected value / pre-LN input norms and needs both captures plus a no-bias value projection contract. |
| `inputs` | required | Prepared native model kwargs or `ProbeInputs`. All processor geometry/mask fields must be kept. |
| `metric` | `None`; required | Fixed differentiable scalar or `[B]` target, e.g. `ClassScore`, `TokenMargin`, `SequenceLogProb`. Do not change targets during integration. |
| `baseline` | `None` | TAM/Beyond only: floating tensor matching `inputs[input_key]` shape/device/dtype. `None` means zero in processed-input coordinates, not a black raw image. Explicit baselines on Grad-CAM/ATTATTR are rejected. |

`inputs` is a native model-kwargs dictionary or `ProbeInputs`; pass every processor field, not just token IDs. Layer indices are zero-based model indices (not plot row numbers). A selection is an integer or a nonempty list of unique supported indices. Token selectors accept `"all"`, `"visual"`, `"text"`, `"last_prompt"`, `"prediction"`, nonnegative expanded-sequence positions, or a boolean `[B,T]` mask. Padding cannot be selected. For VLMs these positions include expanded image slots, not just tokenizer text positions.

Results are detached CPU `ProbeResult` tensors by default; `result.save(path)` stores tensors and JSON-safe run metadata. `B/H/T/V/L` denote batch/head/token/vocabulary/selected-layer axes. Check `probe.describe()["methods"][method_name]` for the actual checkpoint. Eager attention at the probed component and no KV cache are required for these model-bound attention methods.

## Architectural applicability

TAM and Beyond Intuition propagate information through an attention stack.
Applying their formulas to causal decoder attention is a transfer from vision
classification: causal masks can favor early sequence positions when
transitions are composed. Maps still depend on the input's attention weights
and the selected target; this caveat does not imply content-independent maps.
Neither target conditioning nor a different propagation rule guarantees removal
of positional bias. Compare matched inputs and account for sequence order
before interpreting a peak as object evidence.

Constructing TAM/Beyond on an adapter with `causal_attention=True` emits a
`CausalAttentionWarning`; `result.metadata["applicability"]` records the
architecture caveat and confirms that the propagation formula is unchanged.
Native language adapters declare causal attention, native ViT declares
bidirectional attention, and custom adapters remain unspecified unless they
explicitly set the field. Bidirectional attention also requires independent
faithfulness checks. Grad-CAM and ATTATTR do not compose the same transition
stack and receive no propagation warning; this is not a claim that they are
free of other attention-explanation limitations.

The [positional concentration diagnostic](attention_rollout.md#inspect-positional-concentration)
also accepts explicitly selected nonnegative TAM and Beyond Intuition scores.
A gradient-conditioned method can still inherit sequence-position effects;
check matched controls rather than treating any variant as automatically immune.

## Memory and numerical precision

`queries` and `keys` select entries after full attention and the required
gradients have been computed. For example, `queries="last_prompt", keys="visual"`
does not make an answer-to-image probe a sparse attention computation. Returned
maps retain complete native token axes, with unselected entries masked to zero.
Dense attention allocation grows as `B * H * T * T`; attention, gradients and
intermediate tensors can therefore dominate memory for long multimodal inputs.
For Qwen3.5 decoder probes, the [model guide](../models/qwen3_5.md) uses decoder
eager attention with vision SDPA to avoid unnecessary eager vision allocation.

Integration samples run sequentially. Reducing `steps` mainly reduces runtime;
it does not eliminate the dense attention and gradient tensors required by
each sample. Shorten inputs or reduce image resolution to reduce dense
per-sample allocation. Analyzing fewer layers can reduce retained results,
but does not make the underlying attention computation sparse.

Attention profiles, Rollout, gradient products and relevance propagation use
at least FP32 for FP16/BF16 inputs, while preserving FP64. ATTATTR and
input-path methods also accumulate and average gradients in that analysis
precision. Input paths are constructed in the wider dtype with explicit
baseline/input endpoints; tensors injected into the model keep their native
dtype. Analysis maps and statistics may therefore be wider than native
attention tensors. Wider buffers require more memory. This avoids additional
rounding in the analysis; it does not replace convergence checks, recover
gradients already lost inside the model, or change the backbone's precision.

## Complete example: raw image to five fresh maps

This standalone script loads a pretrained classifier, obtains its fixed top-class target, runs every map method, inspects the results, saves them, and plots using `vlmprobe.visualization`. A GPU is recommended for the integration methods; the code also supports CPU. The short 8-sample example is a walkthrough, not a paper-scale reproduction.

```python
from pathlib import Path
import os
from io import BytesIO
from urllib.request import urlopen
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, ViTForImageClassification
from vlmprobe import Prober, ClassScore
from vlmprobe import visualization as viz
import matplotlib.pyplot as plt

# Install: pip install "vlmprobe[transformers,visualization]"
# Set VLM_PROBING_MODEL to a local directory to avoid downloading this checkpoint.
model_id = os.environ.get("VLM_PROBING_MODEL", "google/vit-base-patch16-224")
device = os.environ.get("VLM_PROBING_DEVICE", "cuda:0" if torch.cuda.is_available() else "cpu")
model = ViTForImageClassification.from_pretrained(model_id, attn_implementation="eager").to(device).eval()
processor = AutoImageProcessor.from_pretrained(os.environ.get("VLM_PROBING_PROCESSOR", model_id))
probe = Prober(model, processor)

# A raw COCO image, not a saved attention map or a previous model result.
image_path = os.environ.get("VLM_PROBING_IMAGE")
image = (Image.open(image_path) if image_path else Image.open(BytesIO(urlopen(
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/coco_sample.png"
).read()))).convert("RGB")
inputs = {k: v.to(device) for k, v in processor(images=image, return_tensors="pt").items()}
with torch.no_grad():
    logits = model(**inputs).logits
    target_id = int(logits[0].argmax())  # Choose once; keep fixed during attribution.
metric = ClassScore(target_id, kind="logit")
print("Fixed target:", target_id, model.config.id2label.get(target_id))

# Plot the actual processed image; its crop/resize aligns with the patch tokens.
mean = torch.tensor(processor.image_mean)[:, None, None]
std = torch.tensor(processor.image_std)[:, None, None]
rgb = (inputs["pixel_values"][0].detach().cpu() * std + mean).clamp(0, 1)
processed_image = Image.fromarray((rgb.permute(1, 2, 0).numpy() * 255).astype("uint8"))
patch = model.config.patch_size
grid_shape = tuple(int(n // patch) for n in inputs["pixel_values"].shape[-2:])
visual_positions = list(range(1, 1 + grid_shape[0] * grid_shape[1]))  # CLS is 0.
last = probe.describe()["methods"]["attention.profile"]["sites"][-1]
output_dir = Path(os.environ.get("VLM_PROBING_OUTPUT", "outputs/attention-tutorial"))
output_dir.mkdir(parents=True, exist_ok=True)
steps = int(os.environ.get("VLM_PROBING_STEPS", "8"))
# 8 samples keep the walkthrough short; use 20+ and check convergence for analysis.
results = {
    "Grad-CAM": probe.attention.grad_cam(layers=[last], queries=[0], keys="visual").run(inputs, metric=metric),
    "ATTATTR": probe.attention.attribution(layers=[last], queries=[0], keys="visual", steps=steps).run(inputs, metric=metric),
    "TAM": probe.attention.tam(queries=[0], keys="visual", steps=steps, residual_normalize=True).run(inputs, metric=metric),
    "Beyond-H": probe.attention.beyond_intuition(variant="head", queries=[0], keys="visual", steps=steps).run(inputs, metric=metric),
    "Beyond-C": probe.attention.beyond_intuition(variant="token", queries=[0], keys="visual", steps=steps).run(inputs, metric=metric),
}
for name, result in results.items():
    print(name, {key: tuple(value.shape) for key, value in result.tensors.items()})
    result.save(output_dir / name.lower())
fig, axes = viz.plot_attention_comparison(
    processed_image, results, grid_shape=grid_shape,
    visual_positions=visual_positions, query_positions=[0], columns=3,
    signed={name: name == "ATTATTR" for name in results},
    shared_scale=False, image_alpha=0.25, alpha=1.0,
)
fig.savefig(output_dir / "attention-maps.png", dpi=160, bbox_inches="tight")
plt.show()
```

![Fresh attention map comparison](../assets/tutorials/attention-maps.png)

The [execution manifest](../assets/tutorials/attention-execution.json) records the actual checkpoint, image, target and integration setting. `image_alpha` fades the original image while the measured heatmap stays opaque. Signed ATTATTR uses its own diverging scale; unrelated method units are not forced onto one shared colorbar.

## Switching to a VLM

Keep the factories and parameters above. Replace classifier loading/preprocessing with its [model guide](../models/README.md), construct a fixed `SequenceLogProb` or `TokenMargin` target, and use expanded visual/query positions for plotting. The [VLM walkthrough](../../demos/vlm_attention_comparison_demo.ipynb) shows that entire process, including answer alignment, a nonzero gray-image input baseline and a verified single-tile grid. Multi-crop layouts must be split into their actual grids; do not guess a square grid from token count.

## What each method computes

| Method | Model-bound factory | Principle |
| --- | --- | --- |
| Attention Grad-CAM | `probe.attention.grad_cam(...)` | For each head/query, average score gradients over selected keys that the model's attention mask actually permits; weight attention by those head weights, average heads, then rectify. |
| Hao ATTATTR | `probe.attention.attribution(...)` | Integrate score gradients while replacing all heads of one layer by `alpha * original_attention`; multiply the integral by original attention. |
| TAM | `probe.attention.tam(...)` | Propagate the original final attention backwards through earlier residual attention transitions; multiply by positive integrated final-attention gradients along an input path. |
| Beyond Intuition / Ours-H | `probe.attention.beyond_intuition(variant="head", ...)` | Build forward residual perception maps using head importance derived from `abs(A.T @ gradient)`; apply input-path final-attention feedback. |
| Beyond Intuition / Ours-C | `probe.attention.beyond_intuition(variant="token", ...)` | Build perception maps using projected-value norm / pre-LN input norm as source-token weights; apply the same feedback. |

ATTATTR integrates in **attention probability space** with a fixed input.
Its zero baseline prevents that layer from attending anywhere. Probabilities
are deliberately not renormalized during integration. Each selected layer
gets an independent run, so an attribution sweep across 12 layers with 20
steps requires 240 interventions.

Grad-CAM gradient pooling intersects the chosen keys with each query's native
attention support, including causal and local-window restrictions. An early
decoder query therefore excludes inaccessible future keys from both its
gradient sum and its averaging count. The support is taken from the model's
actual mask, rather than inferred from whether a probability happens to be
zero. A query with no supported selected keys has zero head weights and a zero
CAM. Invalid pairs also remain zero during optional display normalization.
Bidirectional vision attention retains the usual selected-patch-key pooling.
Supported native adapters provide the support mask automatically. Custom
adapters must declare restricted support through
[`ModelSpec.attention_supports`](../ADAPTERS.md#optional-capabilities); without
it, Grad-CAM assumes unrestricted support within the selected keys.

TAM and Beyond Intuition integrate in **model input space**. Their default
path scales processed `pixel_values` from zero to the original tensor while
keeping other inputs and the output target fixed. Zero normalized pixels do
not necessarily correspond to a black raw image. A caller-supplied baseline
can be passed to `.run(inputs, metric=metric, baseline=baseline_pixels)`;
the baseline must match the processed tensor's shape, device, and dtype.
The original papers' input-path feedback is an integrated attention gradient,
without multiplying it by attention as ATTATTR does.

These methods need no probe training or pretrained probe checkpoint. The
model weights remain caller-owned; attribution obtains activation gradients
without fitting parameters. Eager attention is required, and the adapter
must expose consumed probabilities rather than detached diagnostic outputs.

## Returned tensors

`baseline` applies to input-path methods only. `normalize` is optional
per-query min-max display normalization for Grad-CAM; raw values are the
default. `start_layer` is zero-based inclusive. For the token approximation,
the adapter must additionally capture block inputs and project **unweighted
value vectors** through the output weights with no output bias. Attention
head outputs after `A @ V` are not a substitute.

| Result | Fields and shapes |
| --- | --- |
| Grad-CAM | `cam[L,B,Q,K]`; `head_weights[L,B,H,Q]` |
| ATTATTR | signed `attribution[L,B,H,Q,K]`; `head_attribution[L,B,H]` is the sum over selected query/key pairs; `token_attribution[L,B,Q,K]` is the sum over heads |
| TAM | `state[B,T,T]`, `feedback[B,T,T]`, `relevance[B,T,T]`; selected `map[B,T,T]` |
| Beyond Intuition | `perception[B,T,T]`, `feedback[B,T,T]`, `relevance[B,T,T]`; selected `map[B,T,T]`; `layer_perception[P,B,T,T]`; head `head_weights[P,B,H]` or token `token_weights[P,B,T]` |
| Shared | `query_mask[B,T]`, `key_mask[B,T]`, `baseline_score[B]`; JSON-safe model, metric, layer and integration metadata |

`L` is the number of individually analyzed layers; `P` is the number of
propagated layers at or after `start_layer`; `B` is batch size; `H` is head
count; `T/Q/K` are complete token axes. ATTATTR's head sum is a convenience
aggregation, not the original paper's maximum-based head-pruning importance
metric. All model-bound results default to detached CPU tensors.

## Original settings and this library's defaults

| Method | Original paper/code | Library behavior |
| --- | --- | --- |
| Grad-CAM | The official ViT baseline uses last-layer CLS-to-patch attention, averages gradients over patch keys, and min-max normalizes the display. | Specify the same query/key selections; `normalize=True` reproduces the display normalization. It is an attention-map adaptation of feature Grad-CAM. |
| ATTATTR | Paper Equation 4: right Riemann endpoints, 20 steps. Official code: left endpoints, default 64 steps. | `quadrature="right", steps=20` follows the paper; `quadrature="left", steps=64` matches the code sampling. |
| TAM | Paper: half-normalized residual transitions, right Riemann integration. Official code: unnormalized residual transitions and uniformly averaged endpoint-inclusive input samples. | Model-bound default follows the paper. The tensor kernel's `residual_normalize=False` matches the code scale; it differs by a global power of two, leaving an independently normalized heatmap unchanged. |
| Beyond Intuition | Official generator: input-path integration includes both endpoints, default 20 samples; effectively one-based `start_layer`. | Library indices are zero-based; `start_layer=k-1` matches official `k`. Head weights are normalized independently per example. A zero-gradient head group gets zero weights instead of dividing by zero. |

`quadrature="endpoints"` uses uniform samples including both `0` and `1`
(at least two samples) for the official TAM/Beyond Intuition sampling rule.
This is an equally weighted sample average, not a trapezoidal quadrature.

The standalone tensor classes accept an already averaged
`integrated_gradients[B,H,Q,K]`. They do not infer or run the integration
path themselves. Grad-CAM uses ordinary gradients from the original forward;
the head-wise Beyond Intuition approximation also needs the full original
attention-gradient stack. Standalone `AttentionGradCAM.run` accepts an optional
`support_mask`: a boolean `[B or 1,H or 1,Q,K]` tensor on the attention device,
where `True` denotes an accessible pair. It is intersected with `query_mask` and
`key_mask` for pooling and display. `support_mask=None` means unrestricted
support within those selections, preserving the previous selected-key rule;
support is never inferred from `attention > 0`.

The [source index](../REFERENCES.md#attention-methods) maps each rule to its
original paper and author code.

## Modality and scope

Hao's original study uses text/BERT. TAM originally uses image classification
with ViT. Beyond Intuition evaluates BERT, ViT and CLIP; an autoregressive
VLM decoder demonstration is a method transfer. Native ViT, supported eager
Llama/LLaVA attention, and custom adapters can use these kernels when their
contracts are met. Hybrid linear-attention layers cannot be skipped as though
they were ordinary softmax attention. Check `probe.describe()` before choosing
methods or layers.

Full Chefer CVPR 2021 DTD/LRP has a separate original-rule ViT backend:

```python
lrp = probe.attention.dtd_lrp(method="transformer_attribution")
result = lrp.run(inputs["pixel_values"], target=class_id)
patch_heatmap = result.tensors["patch_relevance"]
```

The existing `probe.attention.relevance(...)` implements Chefer's ICCV 2021
generic positive-gradient self-attention recurrence. It is a different
algorithm and must not be labeled as full DTD/LRP. Beyond Intuition is cited to the published TMLR 2023 paper; its repository
also retains an earlier 2022 manuscript date.

## References

Executed comparisons: [native ViT walkthrough](../../demos/attention_comparison_demo.ipynb)
and [SmolVLM/GQA walkthrough](../../demos/vlm_attention_comparison_demo.ipynb).
Both retain the actual inputs, fixed targets and method configurations.

- [Grad-CAM paper](https://arxiv.org/abs/1610.02391), ICCV 2017;
  [official attention-map baseline](https://github.com/hila-chefer/Transformer-Explainability/blob/c3e578f76b954e8528afeaaee26de3f07e3fe559/baselines/ViT/ViT_explanation_generator.py).
- [Self-Attention Attribution paper](https://arxiv.org/abs/2004.11207), AAAI 2021;
  [official ATTATTR code](https://github.com/YRdddream/attattr/blob/f5a7396bee476504e22424b8f11a5ad4e7e55a00/examples/generate_attrscore.py).
- [Transition Attention Maps paper](https://openreview.net/forum?id=TT-cf6QSDaQ),
  NeurIPS 2021 XAI4Debugging workshop;
  [official code](https://github.com/XianrenYty/Transition_Attention_Maps/blob/8329327179f6dcf76df29e03ae519cd594595678/baselines/ViT/ViT_explanation_generator.py).
- [Beyond Intuition paper](https://openreview.net/forum?id=rm0zIzlhcX);
  [official generator](https://github.com/jiaminchen-1031/transformerinterp/blob/a76854abfcfd66a4f74109225ecfc40e07b5a3ef/ViT/baselines/ViT/ViT_explanation_generator.py)
  and [projection definition](https://github.com/jiaminchen-1031/transformerinterp/blob/a76854abfcfd66a4f74109225ecfc40e07b5a3ef/ViT/baselines/ViT/ViT_new.py).
- [Chefer DTD/LRP paper](https://arxiv.org/abs/2012.09838), CVPR 2021;
  [official repository](https://github.com/hila-chefer/Transformer-Explainability).

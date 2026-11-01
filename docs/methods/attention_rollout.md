# Attention Rollout

[Home](../../README.md) / [Methods](README.md) / Attention Rollout

Compose residual attention transitions across layers to summarize indirect token interactions.

## API and parameters

`probe` is a `Prober` bound to a loaded model; `inputs` contains prepared model inputs. The blocks below are API signatures: `*` marks keyword-only parameters, not an argument to copy. The [complete example](#complete-example) defines the model, inputs, and all other variables before running the method.

```text
method = probe.attention.rollout(*, layers=None, head_reduction="mean", residual=True)
result = method.run(inputs, *, metric=None)
```

| Parameter | Default | Meaning and constraints |
| --- | --- | --- |
| `layers` | `None` | Consecutive attention layers `[0, ..., n]` in ascending order. The default uses the complete available prefix; gaps or a selection starting after 0 are rejected. |
| `head_reduction` | `"mean"` | Fuse heads with `"mean"`, `"max"`, or `"min"` before propagation. |
| `residual` | `True` | Add an identity transition at each layer, then row-normalize. `False` uses only attention transitions. |
| `inputs` | `required` | Prepared native model kwargs or `ProbeInputs` matching the bound model. |
| `metric` | `None` | Must be omitted or `None`. A non-`None` metric raises `ValueError` before the model runs: Rollout is target-independent. |

`inputs` is a native model-kwargs dictionary or `ProbeInputs`; pass every processor field, not just token IDs. Layer indices are zero-based model indices (not plot row numbers). A selection is an integer or a nonempty list of unique supported indices. Token selectors accept `"all"`, `"visual"`, `"text"`, `"last_prompt"`, `"prediction"`, nonnegative expanded-sequence positions, or a boolean `[B,T]` mask. Padding cannot be selected. For VLMs these positions include expanded image slots, not just tokenizer text positions.

Results are detached CPU `ProbeResult` tensors by default; `result.save(path)` stores tensors and JSON-safe run metadata. `B/H/T/V/L` denote batch/head/token/vocabulary/selected-layer axes. Check `probe.describe()["methods"][method_name]` for the actual checkpoint. Eager attention at the probed component and no KV cache are required for these model-bound attention methods.

Rollout computes and composes complete dense attention matrices. Choosing a
subset of rows or image patches for plotting does not reduce those allocations;
memory grows quadratically with token count. Results record
`metadata["metric"] = None`. Use [attention relevance](attention_relevance.md)
or a [target-conditioned attention map](attention_maps.md) when an answer target
is part of the analysis.

## Causal decoder applicability

Rollout was developed to summarize attention flow; composing causally masked
transitions can favor early sequence positions because later positions can
access them. This may appear on a random-weight decoder as well. It does not
mean the result is always independent of image content: actual attention
weights still determine the transitions. The original paper explicitly discusses
this decoder limitation.

For adapters declaring `ModelSpec.causal_attention=True`, constructing the
method emits `CausalAttentionWarning`. Results record the declaration, caveat
and unchanged-formula status in `metadata["applicability"]`. Native HF language
adapters declare `True`, ViT declares `False`, and custom adapters default to
unknown (`None`). The library preserves the original rollout calculation;
head fusion, residual transitions and defaults are unchanged. A bright early
image patch should therefore not be taken as an object-localization result
without controls for sequence position. Target conditioning or a different
propagation rule is not a guarantee that this bias disappears.

Adapters declaring `ModelSpec.last_stream_kv_layer` stop propagation at that
layer, because later layers reuse stored K/V rather than reading the current
stream. See [Gemma 4](../models/gemma4.md#residual-interventions-and-propagation).

## Inspect positional concentration

After the complete example below, inspect the same selected query and visual
keys with the explicit diagnostic tool:

```python
from vlmprobe.attention import attention_position_diagnostics

# ViT example below: query 0 is CLS; visual_positions excludes CLS.
visual_scores = result.tensors["rollout"][0, 0, visual_positions]
diagnostic = attention_position_diagnostics(visual_scores, grid_shape=grid_shape)
print(diagnostic.tensors["top_bottom_ratio"])
diagnostic.save(output_dir / "position-diagnostic")
```

For a causal VLM, select its intended native query and actual visual keys
instead. The tool accepts nonnegative finite scores `[..., N]` in native visual
order and preserves leading dimensions. `grid_shape=(H,W)` must describe one
actual row-major image grid. For explicitly identified native tile groups,
pass `tile_sizes=[...]` instead; tile mass and mean-per-token ratios are returned
separately so unequal group sizes remain visible. It never guesses spatial
layout from a packed tensor or treats concatenated tiles as a single image.

The tool measures raw mass in equal first/last sequence thirds and, when given
a grid, equal top/bottom row thirds. It performs no floor subtraction,
normalization or correction. Zero over zero is undefined (`NaN`); positive mass
over zero is `inf`. Empty thirds are also undefined. A ratio far from one is a
concentration measurement, not sufficient evidence of mask-induced bias or
content independence. Compare matched images and mask/weight controls before
making either claim. Signed attribution inputs require the caller to choose
and declare an evidence transform first.

## Complete example

The example loads pretrained weights, reads a raw image, computes a fresh result, and calls the built-in visualization API. No probe training is required. Run outside `torch.inference_mode()` for gradient methods.

```python
from pathlib import Path
import os
from io import BytesIO
from urllib.request import urlopen
import torch
from PIL import Image
from transformers import AutoImageProcessor, ViTForImageClassification
from vlmprobe import Prober
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
result = probe.attention.rollout(layers=list(range(last + 1)), head_reduction="mean", residual=True).run(inputs)
print("Final map:", result.tensors["rollout"].shape)
ax = viz.plot_attention_overlay(
    processed_image, result, tensor="rollout", grid_shape=grid_shape,
    visual_positions=visual_positions, query_positions=[0],
    signed=False, image_alpha=0.25, alpha=1.0,
    title="Attention Rollout",
)
ax.figure.savefig(output_dir / "attention-rollout.png", dpi=160, bbox_inches="tight")
result.save(output_dir / "attention-rollout")
plt.show()
```

![Executed tutorial result](../assets/tutorials/attention-rollout.png)

The preview was generated by this tutorial, not loaded as its input. See the [execution manifest](../assets/tutorials/attention-execution.json) for the checkpoint and settings.

## Outputs and interpretation

`rollout[B,T,T]` is the final transition map; `layer_rollouts[L,B,T,T]` retains every prefix. For each layer, fuse heads, optionally add identity, row-normalize, then left-multiply the preceding rollout. Padding is excluded. This omits values, MLPs and non-attention visual injection edges; interpret it as descriptive attention flow, not causal influence.

## Support and references

Use the same factory calls on another supported checkpoint after changing only model loading and preprocessing in its [model guide](../models/README.md). Hybrid Qwen3.5 supports individual full-attention layers, but rollout/relevance/TAM/Beyond Intuition cannot skip its intervening linear-attention blocks. An all-zero measured map is valid; an unavailable method is a capability error, not a zero map.

[Quantifying Attention Flow in Transformers](https://aclanthology.org/2020.acl-main.385/) · [Implementation](../../src/vlm_probing/attention/rollout.py)

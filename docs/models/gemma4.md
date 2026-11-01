# Gemma 4

[Home](../../README.md) / [Models](README.md) / Gemma 4

Native classes: `Gemma4ForConditionalGeneration` (image/text) and
`Gemma4ForCausalLM` (text). Example checkpoint:
[`google/gemma-4-E2B-it`](https://huggingface.co/google/gemma-4-E2B-it).

## Install from this checkout

Use a separate environment and select the Gemma 4 extra:

```bash
python -m pip install -e ".[gemma4,visualization]"
```

This pins **Transformers 5.5.4**. Choose `gemma4` instead of the
`transformers` extra, which targets 5.3.x for the other native adapters.
Gemma 4 has an independent version gate; installing its runtime does not
make the existing 5.3.x adapters supported on 5.5.4. Version 5.5.0 is not
accepted because its K/V sharing depends on a decode cache, whereas public
probes require uncached full-sequence execution. Newer versions require
separate validation before this gate is expanded.

## Load and probe

This example requires access to the checkpoint on Hugging Face and enough
GPU memory to load it. Accept the model's access terms and authenticate in
your environment before downloading its weights. Replace `example.jpg`
with your image. Native preprocessing expands the image placeholders;
pass all processor fields through to the probe.

```python
import torch
from PIL import Image
from transformers import AutoProcessor, Gemma4ForConditionalGeneration
from vlmprobe import Prober

model_id = "google/gemma-4-E2B-it"
device, dtype = "cuda:0", torch.bfloat16
model = Gemma4ForConditionalGeneration.from_pretrained(
    model_id, dtype=dtype, device_map={"": device}, attn_implementation="eager",
).eval()
processor = AutoProcessor.from_pretrained(model_id)
probe = Prober(model, processor)

messages = [{"role": "user", "content": [
    {"type": "image"},
    {"type": "text", "text": "What is in this image?"},
]}]
prompt = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
)
inputs = probe.prepare(
    prompt=prompt, image=Image.open("example.jpg").convert("RGB"), device=device,
)
for key, value in inputs.kwargs.items():
    if isinstance(value, torch.Tensor) and value.is_floating_point():
        inputs.kwargs[key] = value.to(dtype=dtype)

print(probe.describe()["methods"])
last = max(probe.spec.residuals)
result = probe.lens.logit(layers=[0, last], tokens="last_prompt").run(inputs)
top_id = result.tensors["logits"][-1, 0].argmax().item()
print(processor.tokenizer.decode([top_id]))
result.save("outputs/gemma4/lens")

profile = probe.attention.profile(layers=[last], queries="last_prompt").run(inputs)
print(profile.tensors["group_mass"])
```

For causal interventions, explicitly choose the answer metric. A single-token
margin measures the specified vocabulary contrast, not complete-answer
accuracy; use `SequenceLogProb` for a teacher-forced multi-token answer.

```python
from vlmprobe import TokenMargin

# Illustrative next-token contrast. Inspect the tokenizer's exact subwords.
positive = processor.tokenizer.encode(" cat", add_special_tokens=False)[0]
negative = processor.tokenizer.encode(" dog", add_special_tokens=False)[0]
effect = probe.causal.knockout(
    layers=[last], queries="last_prompt", keys="visual",
).run(inputs, metric=TokenMargin(positive, negative))
print(effect.tensors["baseline_score"], effect.tensors["effect"])
```

For text-only prompts on a multimodal checkpoint such as E2B, keep
`Gemma4ForConditionalGeneration` and omit the image; the automatic layout
contains no visual tokens. Do not load E2B with `Gemma4ForCausalLM`: its text
weights are stored under different names, so every weight is newly initialized
with only a load report and no exception. Use that class only for a checkpoint
whose weights match it, and check `output_loading_info=True`.

## Capabilities and boundaries

| Operation | Gemma 4 contract |
| --- | --- |
| Logit, Tuned, Jacobian, Embed and Patchscope lenses | All residual/embedding sites; calibrated lenses need fitting |
| Attention Lens | Calibrated projected-head readout before post-attention RMSNorm |
| Attention profile, Grad-CAM, ATTATTR | All decoder layers with eager attention and, where needed, an explicit target |
| Rollout, relevance, TAM, head-weighted Beyond Intuition | Eager attention up to the [last layer that reads its own K/V](#residual-interventions-and-propagation); TAM/Beyond Intuition integrate processed image inputs |
| Residual patching, attribution patching, steering and SAE binding | Sites capture complete block outputs, including PLE and layer scaling; [these edits have architecture limits](#residual-interventions-and-propagation) |
| Knockout, temperature and probability reweighting | Actual eager scores/probabilities, including K/V-sharing layers |
| Joint steering generation | Uncached full-prefix greedy replay, one unpadded example |
| IOI path patching and fixed-scale head logit attribution | Not declared: post-attention normalization couples heads, and K/V may be shared |
| MJ-Lens, token-weighted Beyond Intuition, edge-level EAP-IG | Not declared; require additional architecture contracts |
| Audio/video and automatic `visual_grid` | Not supported in this adapter |

Gemma 4 uses scaled word embeddings, per-layer inputs (PLE), alternating
local/global attention, optional shared K/V, and possibly distinct global
head widths or shared K/V projections. Residuals are captured after all
block branches, at the next block's input or the final norm's input. The
readout applies native RMSNorm, the LM head and configured output soft-capping.

Lens logits at every layer pass through the same native soft cap. On E2B
(cap 30) raw maxima at the sampled layers between 0 and 33 were 58-98, and about
9 at the final layer, so intermediate logits saturate near the cap, margins
compress and BF16 values can tie at the top. Rank ties are a readout property, not evidence
that those tokens are equivalent. For uncapped diagnostics, apply
`decoder.norm` and `lm_head` to a captured residual directly.

Head projections sum to the bias-free `o_proj` output **before** the block's
post-attention RMSNorm. They are useful inputs to a calibrated Attention
Lens; they are not exact additive writes to the residual stream. No fixed
linear head-to-logit decomposition is exposed.

Image selection uses the native image token ID, intersected with valid
sequence positions. Audio/video tensors and soft-token placeholders are rejected
rather than classified as text. Keep `image_position_ids` and
`mm_token_type_ids` for paired experiments; automatic alignment checks them.
Supply explicit grids to plotting functions: image patch coordinates and
pooling are not yet exposed through `visual_grid`.

Checkpoints using `use_bidirectional_attention="vision"` require
`mm_token_type_ids` with images: without them native evaluation silently uses
causal masks, so the adapter rejects those inputs. The value `"all"` is rejected
because native 5.5.4 masks stay causal under it.

Local windows and optional bidirectional vision masks change which pairs
are allowed. Grad-CAM uses structural support captured from the native
mask, independently of probability magnitude. Rollout and related maps
describe this computation; they are not proof of spatial faithfulness or
of a causal effect. Residual/lens methods also work with SDPA, which does
not expose attention matrices or score/probability interventions.

## Residual interventions and propagation

Two routes bypass the residual stream, so it does not determine all downstream
computation. `describe()["notes"]`, `ResidualInterventionWarning` and
`result.metadata["applicability"]` declare the limits for each configuration.

- **Per-layer inputs.** PLE is computed from the merged input embeddings, so
  image features reach every block outside the residual stream. Paired patching
  keeps the receiver's per-layer inputs.
- **Shared K/V.** Layers after the sharing point reuse K/V stored by earlier
  layers instead of reading the current residual stream. A residual edit at or
  after the last layer that reads its own K/V cannot reach any other position.

| Checkpoint config | PLE | Shared K/V layers | Last layer reading its own K/V |
| --- | --- | --- | --- |
| E2B | Yes | 20 of 35 | 14 |
| E4B | Yes | 18 of 42 | 23 |
| 26B-A4B, 31B | No | None | Every layer |

Measured once on E2B with BF16 weights on CPU, patching a blue-square run with a
red-circle source (margin -32.4 and +30.0), scored at the last position. BF16
values vary slightly by device (61.44 instead of 61.31 on a GPU); the exact zeros
do not.

| Layer | Restore all positions | + per-layer inputs | + stored K/V | Default `tokens="visual"` effect |
| --- | --- | --- | --- | --- |
| 0 | Margin -31.6 | Logits match source | Margin -31.6 | 1.25 |
| 9 | Margin -25.3 | Logits match source | Margin -25.3 | 6.31 |
| 13 | Not measured | Not measured | Not measured | 61.31 |
| 14 | Margin -32.1 | Margin -32.1 | Logits match source | 0 (exact) |
| 24 | Margin -25.7 | Margin -25.7 | Logits match source | 0 (exact) |

The two extra columns are hook experiments, not public options. Read a layer
sweep accordingly: the jump at 13 and the zeros from 14 follow from which
stream later K/V are computed from. They do not show that late layers ignore the
image, because attention there still reads visual K/V stored at layers 13 and 14.
To measure that, use attention interventions such as knockout, which act on
every layer. Steering, SAE edits and Patchscope at these layers are limited by
the same K/V route.

Rollout, relevance, TAM and Beyond Intuition compose each layer's attention onto
the current stream. Shared layers read a different stream, so these methods stop
at the last layer reading its own K/V and reject later layers. By default they
therefore cover layers 0-14 on E2B, and `result.metadata["layers"]` records the
stack. Single-layer methods and attention interventions cover all layers.

## Validation

Run the offline checks without downloading model weights:

```bash
python examples/gemma4_smoke.py
python -m unittest discover -s tests -p test_gemma4_adapter.py -v
```

Tests instantiate real tiny vision towers, spatial pooling, projectors and
decoders with random weights. They exercise PLE, shared K/V, different
local/global head dimensions, alternative K=V attention, padding, soft-capping
and bidirectional visual support. They check FP32/BF16 native logit and
gradient parity, final readout equality, layout alignment, self-patching,
nonzero causal effects, frozen parameter gradients, hook cleanup and state
restoration after exceptions. They also check that residual restoration fails
exactly when a bypass is declared and that edits past the K/V boundary have an
exact zero effect at other positions. Separate CI jobs retain Transformers 5.3.

This is **architecture validation**, plus the E2B measurements above and
native logit parity on those E2B weights with BF16 on one device. Pretrained
accuracy, other checkpoints, MoE configurations (accepted, with tiny-model logit
parity only), audio/video, quantization, multi-device/offloaded models and
compiled execution have not been validated by this contribution.

Upstream references: [Gemma 4 documentation](https://huggingface.co/docs/transformers/model_doc/gemma4)
and the pinned [Transformers 5.5.4 implementation](https://github.com/huggingface/transformers/blob/v5.5.4/src/transformers/models/gemma4/modeling_gemma4.py).

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

For a text checkpoint, load `Gemma4ForCausalLM` and its tokenizer and pass
the complete tokenizer dictionary. The same residual methods work; the
automatic layout contains no visual tokens.

## Capabilities and boundaries

| Operation | Gemma 4 contract |
| --- | --- |
| Logit, Tuned, Jacobian, Embed and Patchscope lenses | All residual/embedding sites; calibrated lenses need fitting |
| Attention Lens | Calibrated projected-head readout before post-attention RMSNorm |
| Attention profile, rollout and relevance | All decoder layers with eager attention; propagation describes the native masks |
| Grad-CAM, ATTATTR, TAM, head-weighted Beyond Intuition | Eager attention with an explicit target; TAM/Beyond Intuition integrate processed image inputs |
| Residual patching, attribution patching, steering and SAE binding | Complete block outputs, including PLE and layer scaling |
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

Local windows and optional bidirectional vision masks change which pairs
are allowed. Grad-CAM uses structural support captured from the native
mask, independently of probability magnitude. Rollout and related maps
describe this computation; they are not proof of spatial faithfulness or
of a causal effect. Residual/lens methods also work with SDPA, which does
not expose attention matrices or score/probability interventions.

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
restoration after exceptions. Separate CI jobs retain Transformers 5.3.

This is **architecture validation**. Pretrained checkpoint execution,
pretrained accuracy, MoE configurations, audio/video, quantization,
multi-device/offloaded models and compiled execution have not been
validated by this contribution. The loading example is not a claim that
those pretrained weights were tested.

Upstream references: [Gemma 4 documentation](https://huggingface.co/docs/transformers/model_doc/gemma4)
and the pinned [Transformers 5.5.4 implementation](https://github.com/huggingface/transformers/blob/v5.5.4/src/transformers/models/gemma4/modeling_gemma4.py).

# Model Support

[Home](../../README.md) / [Documentation](../README.md)

Automatic adapters target **Transformers 5.3.x**, tested with **5.3.0**. The nine
VLM families below use native Transformers classes. Each guide provides a full
loading and image-input example. [Llama](llama.md), [GPT-2](gpt2.md), and [Qwen2/Qwen3 text](qwen_text.md) provide text controls;
[ViT](vit.md) provides a pure-vision classification control.

## Use any supported method

**Model-specific code loads the model and prepares its inputs. Method calls use
the same public API across adapters.** A short call in a model guide
is an example, not an exhaustive list of supported methods. Use the
[method guides](../methods/README.md) for each complete workflow, including
parameters, required calibration or reference inputs, outputs and plotting.

After the loading/input block in any model guide, inspect the actual binding:

```python
capabilities = probe.describe()["methods"]
for name, item in capabilities.items():
    if item["available"]:
        print(name, "supported layers:", item["sites"])
    else:
        print(name, "unavailable:", item["missing"])

# Select an operation that this actual model can provide.
if capabilities["lens.logit"]["available"]:
    layers = capabilities["lens.logit"]["sites"]
    result = probe.lens.logit(layers=layers[::4] or layers[:1]).run(inputs)
else:
    print("This model has no vocabulary readout; choose a supported attention method.")
```

| What changes when switching models? | What stays the same? |
| --- | --- |
| Model/processor class, checkpoint, chat template and native image preprocessing | `Prober(model, processor)` and method factory/run structure |
| Available layers, head counts, image grids and token layout | Explicit `layers`, token selectors, `ProbeResult` and plotting interface |
| The tokenizer and answer tokenization | Score construction rules; use complete-answer scoring when labels have several pieces |
| Fitted lens artifact identity | Independent training/validation and save/load conventions |

The architecture can limit a method even though its call syntax is shared:
Qwen3.5 has a hybrid attention stack; ViT has class logits rather than a language
vocabulary; original EAP-IG uses an audited GPT-2 edge graph. A calibrated lens
also needs weights for the exact checkpoint and layer. Inspect the capability
matrix below and the model-specific notes before running an expensive sweep.

The method guides include complete examples. To reuse them with your prepared
model, keep the loading/preparation block from this guide and then apply the
method's configuration, target/reference construction and plotting steps.
Do not copy a different model's tokenizer, hard-coded layer IDs or fitted artifacts.

| Family | Example size | Validation in this release |
| --- | --- | --- |
| [Qwen3.5](qwen3_5.md) | 4B | Tiny architecture + eight pretrained report cases |
| [Qwen3-VL](qwen3_vl.md) | 2B | Tiny architecture + eight pretrained report cases |
| [Qwen2.5-VL](qwen2_5_vl.md) | 3B | Tiny architecture + local pretrained checkpoint |
| [Qwen2-VL](qwen2_vl.md) | 2B | Tiny architecture with actual vision path |
| [SmolVLM](smolvlm.md) | 256M / 500M | Tiny architecture + pretrained cases; 500M Tuned calibration |
| [InternVL](internvl.md) | 1B | Tiny architecture + eight pretrained report cases |
| [LLaVA](llava.md) | 7B | Tiny architecture + pretrained GQA/COCO demos and shared report cases |
| [Pixtral](pixtral.md) | 12B | Native Mistral/Pixtral architecture tests + eight pretrained cases; seven attention methods |
| [Janus](janus.md) | Pro-1B / Pro-7B, native format | Tiny native pixel/text, teacher-forced codebook, MJ transport, head and VQ decode contracts; no new pretrained accuracy claim |
| [Qwen2 / Qwen3 text](qwen_text.md) | 0.5B / 0.6B | Native tiny text architecture tests; no new pretrained accuracy claim |
| [ViT](vit.md) | ViT-B/16 | Tiny classifier + actual pretrained author-checkpoint map/deletion controls |

Tiny tests instantiate real vision towers, projectors, and decoders with random
weights. They check all available lens methods, gradients, padding, visual token
coordinates, readout equality, head projection, self-patching, and nontrivial
interventions. This is architecture validation; it does not establish pretrained
accuracy on the checkpoints whose weights were not run. The
[technical report](https://github.com/gexinh/vlm-probing-report/blob/main/technical-report.pdf) adds actual execution on five
pretrained VLMs, eight fixed cases each, with explicit capability boundaries.
All eight official processors were also exercised on image+chat inputs.
[Architecture checks](../../examples/hf_model_matrix.py) use small native models.

## Capability matrix

[MJ-Lens](../methods/mj_lens.md) has its own declared block-input sites and final
RMSNorm target, visible under `lens.mj` in `describe()`. Native Qwen, Llama,
Idefics3, InternVL, LLaVA and Janus RMS decoders expose text-output transport;
Janus also declares its discrete visual output head. An image encoder alone
does not provide that second output channel. GPT-2's affine LayerNorm and
undeclared custom/Pixtral contracts are not automatically treated as MJ RMS
readouts. The table below describes the existing residual-output methods.

| Methods | Qwen3.5-4B | Other seven VLM families |
| --- | --- | --- |
| Logit, Tuned, Jacobian, Embed, Patchscope | All residual/embedding sites | All residual/embedding sites |
| Attention Lens | Standard-attention layers | All language attention layers |
| Attention profile, head logit attribution | Standard-attention layers | All language attention layers |
| Attention rollout, relevance | Unavailable across the hybrid stack | Consecutive language attention layers from 0 |
| Activation patching, attribution patching, steering | All residual layers | All residual layers |
| IOI-style path patching | Unavailable: linear-attention outputs cannot all be frozen | Head → Q/K/V or final residual, every language attention layer |
| Attention knockout, temperature | Full-attention layers only | Native eager Llama, Mistral, Qwen2, Qwen2-VL, Qwen2.5-VL and Qwen3-VL score taps |
| Attention probability reweighting / ATTATTR | Full-attention layers only | Consumed editable probabilities in the same audited eager decoders |
| EAP-IG | Requires explicit edge sites | Requires explicit edge sites; original input-path graph available for GPT-2 text |

Availability describes the adapter contract. Tuned, Attention, and Jacobian
Lens still require checkpoint-specific calibration or a compatible artifact;
the table does not imply that fitted decoders are supplied for every model.
Grad-CAM can inspect individual full-attention layers, including Qwen3.5.
TAM and Beyond Intuition require the same consecutive attention stack as
Rollout, so the hybrid Qwen3.5 stack is outside their current scope.

Attention profiles, rollout, and relevance require `attn_implementation="eager"`.
Logit/residual probes and projected-head lenses also work with SDPA. Fused
backends do not expose probability matrices. `probe.describe()` reports the
actual sites available on the supplied instance. Keep the attention backend
fixed after constructing a Prober; construct a new one after changing it.
Path patching also works with SDPA because it edits Q/K/V projections and head
messages directly. For GQA, K/V receiver indices name physical shared KV heads.
Path controls and GPT-2 were also checked on random-weight native architectures.
The [path notebook](../../demos/path_patching_demo.ipynb) includes measured pretrained examples.

The [ViT vision-control adapter](vit.md) supports class-conditioned Grad-CAM,
ATTATTR, TAM, Beyond Intuition and genuine Chefer DTD/LRP with the same classifier
weights. DTD is scoped to this explicit ViT backend. VLM use of the other
attribution maps is a documented method transfer, not evidence of full VLM DTD.

## Input and execution contract

Pass the complete processor output, including grid and mask metadata. Probing
uses uncached, full-sequence forwards. All positions refer to the expanded
language sequence, including visual slots and padding. Multiple images within a
prompt, video, quantization, sharded/offloaded models, and compiled models require
additional validation; this release tests unquantized image+text models on CPU
and single-GPU bf16 execution for pretrained VLMs. GPT-2 EAP-IG uses float32.

A residual site is measured after a language block and any intervening visual
injection, before the next block or final readout norm. Image tokens are selected
using explicit architecture token IDs. No fixed-length visual-token guess is
used. For custom layouts and architectures, see [adapters](../ADAPTERS.md).

Legacy Transformers 4.57.x remains supported for Llama, Qwen2.5-VL, and GPT-2.
The expanded model set, including Qwen2/Qwen3 text-only classes, uses the 5.3.x extra.

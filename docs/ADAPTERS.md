# Model adapters

[Home](../README.md) / [Documentation](README.md)

The adapter owns model-specific execution. Algorithm classes remain independent
of architecture paths. A public adapter supplies `model`, `model_id`, `readout`,
`spec`, and `run`; `TorchModelAdapter` implements scoped module hooks and is the
usual starting point.

## An explicit custom adapter

This complete binding uses the included `TinyModel`. A real model needs its own
verified module paths and sequence layout provider.

```python
import torch
from examples.tiny_model import TinyModel
from vlmprobe import (
    HookPoint, ModelReadout, ModelSpec, Prober, TokenLayout, TorchModelAdapter,
)

model = TinyModel().eval()

def layout(inputs):
    ids = inputs["input_ids"]
    n_visual = inputs["image_tokens"].shape[1]
    image_ids = torch.zeros(ids.shape[0], n_visual, dtype=torch.long, device=ids.device)
    expanded_ids = torch.cat([image_ids, ids], dim=1)
    valid = torch.ones_like(expanded_ids, dtype=torch.bool)
    visual = torch.zeros_like(valid)
    visual[:, :n_visual] = True
    return TokenLayout(valid, visual, token_ids=expanded_ids)

adapter = TorchModelAdapter(
    model,
    sites={
        "layer0": HookPoint("layers.0"),
        "layer1": HookPoint("layers.1"),
        "embedding_space": HookPoint("layers.0", kind="input", selector=0),
    },
    model_id="my-checkpoint-revision",
    readout=ModelReadout(model.lm_head, norm=model.norm),
    spec=ModelSpec(
        residuals={0: "layer0", 1: "layer1"},
        embeddings="embedding_space",
        input_embeddings=model.embedding.weight,
        layout=layout,
    ),
)
probe = Prober(model, adapter=adapter)
print(probe.describe())
```

This adapter supports residual lenses/interventions and EmbedLens. It deliberately
does not claim attention or head access. The full demonstration contract is in
[`TinyModel.probing_adapter`](../examples/tiny_model.py).

## Hook and sequence contracts

- `HookPoint(module, kind="output", selector=None)` captures a tensor output.
  An integer/string selector selects a tuple item/dictionary value.
- `kind="input"` needs a selector: an integer positional argument index or a
  keyword name. Confirm which calling convention the real model uses.
- `tensor_slice=(start, stop)` selects a last-dimension interval after the
  container selector. Disjoint slices support GPT-2's packed Q/K/V projection;
  overlapping aliases are rejected, and edits preserve the remaining slices.
- Each requested site must execute exactly once per forward. Repeated/shared
  sites and missing sites raise errors. Hooks are removed on success and failure.
- An intervention must preserve shape, dtype, and device. It must replace the
  tensor actually consumed by downstream computation.
- The adapter runs in evaluation mode and restores every module's prior training
  flag. Gradient captures retain the real graph; downstream in-place mutations
  are rejected. It does not update parameters or their accumulated gradients.
- Full-sequence, uncached inputs are required. HF adapter defaults supply
  `use_cache=False` and `return_dict=True`; custom models declare their own
  `forward_defaults`. Explicit caches are rejected.
- `ModelSpec.layout(kwargs)` returns masks in expanded sequence coordinates. The
  layout is validated against actual captured/output sequence lengths. For a
  particular batch, `ProbeInputs(kwargs, layout=...)` overrides the provider.
- `valid` is required; `visual` is required for visual/text selection. With no
  `prompt` mask, all valid positions are considered prompt. `token_ids` enables
  strict alignment and teacher-forced scoring. The library does not guess where
  a model inserts visual tokens.

## Optional capabilities

| ModelSpec field | Contract |
| --- | --- |
| `residuals` | `{layer: site}`, residuals after a block and any visual injection, before the next block/final norm |
| `attentions` | `{layer: site}`, actual probabilities `[B,H,Q,K]`; observational unless declared editable |
| `attention_scores` | `{layer: site}`, editable masked logits before softmax |
| `causal_attention` | Explicit `True` for causally masked attention, `False` for bidirectional attention, or `None` when unspecified. Propagation warnings use this declaration, not output shape. |
| `last_stream_kv_layer` | Last layer whose attention reads K/V from the residual stream, or `None` when every layer does. Propagation stops there, and residual edits from that layer warn that they cannot reach other positions. |
| `residual_bypass` | `True` when later blocks consume state outside the residual stream, so replacing one residual does not restore a paired run. Patching and attribution then warn. |
| `attention_supports` | Optional `{layer: site}` for layers in `attentions`; each site captures a boolean `[B or 1,H or 1,Q,K]` tensor on the attention device, with `True` for structurally permitted pairs |
| `editable_attention` | Layer indices whose probability site is consumed by `A @ V` after the hook |
| `heads` | `{layer: site}`, projected heads `[B,T,H,D_residual]`, or raw inputs for `project_heads` |
| `project_heads(layer, tensor)` | Convert a head capture to post-output-projection residual contributions |
| `embeddings` | Site aligned to the expanded sequence in input embedding space |
| `input_embeddings` | Input embedding table `[vocabulary, residual_dimension]` |
| `linear_readout(full_residual)` | Return `(weight[V,D], inverse_norm_scale[B,T,1], center_bool)` |
| `edges` | `{edge_name: site}` for independently replaceable actual graph messages |
| `path_heads` | `{layer: HeadSite(site, heads, head_dim)}`; editable outputs at every decoder layer, `[B,T,H*D]` or `[B,T,H,D]` |
| `path_qkv` | `{layer: {"q"/"k"/"v": HeadSite(...)}}`; editable receiver inputs, distinct from output freeze sites |
| `path_final` | Final residual site before the readout normalization |
| `alignment_keys` | Model-kwargs tensor keys compared before paired interventions |
| `forward_defaults` | Explicit default model kwargs merged before public execution |
| `output_kind` | `"language"` for `[B,T,V]`, `"classification"` for `[B,C]` with a separate patch layout |
| `attention_inputs / attention_values` | Pre-LN residual and actual V sites for Beyond Intuition token weighting |
| `project_values(layer, tensor)` | Concatenate/replicate V heads then apply output weights without output bias; this is not an `A @ V` head contribution |

Head projection must split the *query-head* axis correctly, including GQA. Exclude
output-projection biases from individual head contributions. Fixed-scale logit
attribution uses the full residual's denominator; fold norm gain into the readout
weight and handle biases separately. LayerNorm needs centering; RMSNorm does not.

Supported native HF and ViT adapters supply `attention_supports` automatically.
For a custom attention adapter, declare support sites when causal, padding or
local-window masks restrict which keys each query can access. Batch and head
dimensions may broadcast from size 1; query and key dimensions must match the
captured probabilities. Capture the model's actual allowed pairs independently
of probability magnitude: a permitted probability can be numerically zero, so
`attention > 0` is not a valid support mask. Grad-CAM intersects this support
with the user's query/key selections before pooling gradients. If a custom
adapter omits a layer's support site, Grad-CAM retains unrestricted support
within the selected keys; it does not infer unknown causal or window masks.
See the [attention-map guide](methods/attention_maps.md#what-each-method-computes)
for the tensor-kernel `support_mask` contract.

A returned attention diagnostic is not necessarily editable. Audited native
Transformers 5.3 eager Llama, Mistral, Qwen2, Qwen3, Qwen2-VL, Qwen2.5-VL, Qwen3-VL,
Qwen3.5 full-attention blocks, and ViT classifiers expose masked logits and the
probabilities actually consumed by `A @ V` through parameterless taps.
Their per-instance attention calls use an audited eager interface; the backend
is restored before decoder mask construction and after exceptions. Global
Transformers classes and model weights are unchanged. Unrecognized attention
classes require explicit score sites. See [native score taps](../src/vlm_probing/adapters/hf_attention.py).

For EAP-IG, every edge must be an independently replaceable message at a consumer.
A list of residual nodes does not establish an edge-level circuit. Joint replay
must actually use all supplied messages; identical aliases are rejected by the
hook adapter. The activation-space implementation stacks equally shaped edge
tensors. The original input-path algorithm uses `graph="transformer"`, with a
scoped native GPT-2 provider that splits Q/K/V residual inputs independently,
retains the native computation, and actually replaces excluded edge messages.

IOI path patching requires `path_heads` to cover all decoder layers. Partial
coverage cannot preserve its freeze rule. Pre-output-projection messages are
valid output controls when the projection is fixed and linear; the projection
and shared bias recompute normally. Receiver Q/K/V is cached upstream of this
freeze site. In GQA, Q uses query-head counts and K/V use physical KV-head counts.
Q/K hooks precede positional rotation; Qwen3 and Qwen3-VL hooks follow per-head
normalization. For custom adapters, declare the tensor actually consumed by
attention, and preserve that hook-stage convention in experimental metadata.

## Causal attention propagation

Native language adapters declare `causal_attention=True`; ViT declares `False`.
Custom adapters default to `None`, which means unknown rather than bidirectional.
Set this field only after checking the actual attention masks. Rollout, relevance,
TAM and Beyond Intuition warn when constructed for a declared causal stack and
retain the declaration and caveat in `result.metadata["applicability"]`.

A causal mask allows later positions to read earlier ones, which can favor early
sequence positions when transitions are composed. Attention weights and target
gradients still affect the result; the map is not necessarily independent of
image content. The original propagation formulas remain unchanged. Use these
maps as descriptions under the stated mask, with matched-input and sequence-order
controls where appropriate; no alternate propagation method is guaranteed to
remove positional bias. Bidirectional attention alone also does not establish
spatial faithfulness or a causal effect. See the
[rollout paper's decoder discussion](https://aclanthology.org/2020.acl-main.385/).

## Registering an architecture

```python
from vlmprobe import register_adapter

register_adapter(MyModelClass, make_adapter)
probe = Prober(my_model)  # make_adapter(my_model)
```

Registration matches the exact Python class, so unknown subclasses are not silently
assumed compatible. Alternatively implement `model.probing_adapter()` returning
a configured adapter. Registration takes precedence over that protocol and the
built-in HF adapters. `replace=True` explicitly replaces an existing registration.

Built-in HF adapters use explicit contracts for eight VLM families, Llama, GPT-2, Qwen2/Qwen3 text,
and the native ViT classifier.
See the [model matrix](models/README.md) for classes, versions, examples, and
validation. Qwen3.5 linear-attention blocks have residual sites but no standard
softmax attention/head sites. Qwen3-VL captures residuals after DeepStack visual
injection. Unknown wrapper classes and unaudited composite text backbones are
rejected. Use an explicit adapter for other contracts and update integration
tests before registering another architecture.

## Integration checks

Gemma 4 uses a separate native contract in
[`adapters/gemma4.py`](../src/vlm_probing/adapters/gemma4.py), pinned to
Transformers 5.5.4. It captures complete post-block residuals, including PLE
and layer scaling; its readout includes native final logit soft-capping.
Its head projections precede post-attention RMSNorm, so the adapter does not
declare additive residual head attribution or IOI path controls. PLE and shared
K/V bypass the residual stream, which it declares through `residual_bypass` and
`last_stream_kv_layer`. Audio/video inputs are rejected. See the [Gemma 4 guide](models/gemma4.md) for supported
methods, runtime setup and validation scope.

Before declaring a model supported, verify final-layer readout equals model logits,
self-patching leaves outputs unchanged, nontrivial edits change downstream scores,
restoring a complete residual from a source run reproduces that source unless
`residual_bypass` is declared,
layout masks respect padding/visual expansion, and head projections sum to the
bias-free attention output. Check hook cleanup and model-state restoration after
exceptions. Gradient probes need graph-preserving sites even when model parameters
are frozen. Multi-device and quantized execution require separate validation.

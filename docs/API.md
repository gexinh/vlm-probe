# Common API

[Home](../README.md) / [Documentation](README.md)

Model-specific setup lives in the [model guides](models/README.md). Individual
factory parameters and examples live in the [method guides](methods/README.md).

## Bind and inspect

```python
from vlmprobe import Prober

probe = Prober(model, processor=processor, adapter="auto", result_device="cpu")
support = probe.describe()
print(support["methods"]["lens.logit"])
# {"available": True, "sites": [...], "requires": ..., "missing": None}
```

`model` is a caller-owned model instance. Automatic selection uses exact known
classes, registered factories, or a model's `probing_adapter()` protocol.
For another architecture, pass an explicit [adapter](ADAPTERS.md).
`describe()` runs no inference and exposes method sites and missing capabilities.

## Prepare inputs

```python
inputs = probe.prepare(prompt=formatted_prompt, image=pil_image, device="cuda:0")
result = probe.lens.logit(layers=[0]).run(inputs)

# Existing processor output works directly.
native_inputs = processor(text=formatted_prompt, images=pil_image, return_tensors="pt")
native_inputs = {k: v.to("cuda:0") for k, v in native_inputs.items()}
result = probe.lens.logit(layers=[0]).run(native_inputs)
```

`prepare(*, prompt, image=None, layout=None, device=None, **kwargs)` forwards
processor arguments and returns `ProbeInputs`. Apply the model's chat template
first, as shown on each model page. The optional layout uses expanded sequence
coordinates. Native dictionary values need not all be tensors; `prepare` moves
tensor values only. Probing requires full-sequence inputs and rejects KV caches,
`use_cache=True`, and nonzero `logits_to_keep`.

For spatial plots, inspect [input geometry and preprocessing](INPUT_GEOMETRY.md).
The `visual_grid` helper uses audited model-specific grid metadata and checks
token counts. `preprocessing_report` is an opt-in diagnostic; it never changes
the processor output or claims that geometric retention proves subject visibility.

## Select layers and tokens

MJ-Lens uses an explicit **decoder block-input** convention; ordinary lens
indices refer to the residual sites below. Its source/target modalities and
final-normalization target are declared by the adapter. See the
[MJ-Lens guide](methods/mj_lens.md) before reusing another lens's layer IDs.

Factory configuration is reusable: `method = probe.lens.logit(...)`, followed
by `method.run(batch)` for each batch. Layers are zero-based. For native HF
adapters, residual layer `i` is the next block's input (or final norm input),
including any visual injection after block `i`.

| Selector | Meaning |
| --- | --- |
| `layers=None` | All available sites for that method; propagation uses the contiguous prefix from layer 0. |
| `layers=3` or `layers=[3, 7]` | Explicit unique layer indices, preserving order. |
| `tokens="all"` | All valid positions. |
| `tokens="visual"` / `"text"` | Positions selected by the adapter's visual mask or its valid complement. |
| `tokens="last_prompt"` | Last prompt position, or last valid position if no prompt mask exists. |
| `tokens="prediction"` | Last original prompt position plus valid generated positions outside `layout.prompt`; used for generation-step steering. |
| `tokens=4` / `[4, 5]` | Expanded sequence positions, valid in every batch example. |
| `tokens=boolean_mask` | Explicit `[batch, expanded_sequence]` selection, excluding padding. |

`queries` and `keys` use the same selector conventions. Residual interventions
at multiple layers are separate experiments by default. Steering and knockout
accept `joint=True` to edit all selected live sites together. Rollout and
relevance compose consecutive attention layers.

For attention maps, query/key selectors are applied after dense attention and
the required gradients are computed. Returned maps keep full native token
axes, with unselected entries masked to zero; the selectors are not a sparse
execution or memory-saving mode. Attention tensors scale quadratically with
sequence length. Integration samples run sequentially, so reducing `steps`
mainly changes runtime rather than the dense per-sample memory requirement.
See [attention memory and precision](methods/attention_maps.md#memory-and-numerical-precision).

### Joint interventions and generation

```python
window = list(range(8, 17))  # nine layers, center 12
effect = probe.causal.knockout(
    layers=window, queries=question_mask, keys=visual_mask, joint=True,
).run(inputs, metric=metric)

vsv = probe.causal.vsv(image_inputs, same_text_without_image)
steering = vsv.configure(strength=0.17, tokens="prediction")
caption = steering.generate(image_inputs, direction=vsv.directions,
                            max_new_tokens=144, do_sample=False)
```

Joint `.run` returns one score per batch item, rather than a stack of separate
layer effects. `.forward(..., capture=[site, ...])` executes only the live
intervention forward and returns a `ForwardTrace`. Joint steering also provides
uncached greedy `.generate` for one unpadded example. Image placeholders must
already be expanded. `prediction` fixes the original prompt boundary and
reapplies edits to all prior generation positions in each full-prefix replay.
Teacher-forced analyses must set `TokenLayout.prompt` to that same boundary.
Outputs include `generated_ids`, `fullsequence_ids`, and `decoded_text` metadata.

Original EAP-IG is configured with `probe.causal.eap_ig(graph="transformer", steps=5)`;
the native GPT-2 provider supports input-embedding integration and actual
complement-edge circuit evaluation. `graph="activation"` retains the explicitly
declared edge-message interpolation variant. See the [method guide](methods/eap_ig.md).

## Metrics and explicit layouts

For a ViT vision-classification control, use `ClassScore(class_id)` on the
native `[B,C]` logits. It supports `kind="logit"` or `"probability"` and keeps
patch coordinates separate from the output-class axis. See [ViT](models/vit.md)
and the [attention-map interfaces](methods/attention_maps.md).

For single-token candidate answers:

```python
from vlmprobe import TokenMargin

positive = processor.tokenizer.encode(" red", add_special_tokens=False)
negative = processor.tokenizer.encode(" blue", add_special_tokens=False)
assert len(positive) == len(negative) == 1
metric = TokenMargin(positive[0], negative[0], position="last_prompt")
result = probe.causal.patch(layers=[0]).run(corrupt, source=clean, metric=metric)
```

The score is positive-token logit minus negative-token logit. It needs exactly
one prediction position per example. For teacher-forced multi-token answers,
supply the full prompt+answer sequence and an answer mask:

```python
from vlmprobe import ProbeInputs, TokenLayout, SequenceLogProb

inputs = ProbeInputs(model_kwargs, TokenLayout(
    valid=valid_mask, visual=visual_mask, prompt=prompt_mask, token_ids=expanded_ids,
))
metric = SequenceLogProb(answer_mask, reduction="sum")
```

Masks are boolean `[B,T]`; token IDs are long `[B,T]`, aligned with actual model
logits. Sequence scoring shifts logits by one and excludes visual/padding
targets. Each answer token needs a preceding prediction position; EOS is scored
only when selected. Score separate complete inputs for different candidate answers.

Custom metrics may be `callable(logits) -> scalar_or_batch_tensor` or objects
with `score(logits, layout)`. Preserve the graph for gradient methods.

Attention profiles, Rollout and head logit attribution are target-independent.
Their common `.run(inputs, metric=None)` signature accepts only an omitted or
`None` metric; a non-`None` value raises `ValueError` before any model forward.
These results record `metadata["metric"] = None`. Relevance, Grad-CAM, ATTATTR,
TAM, Beyond Intuition and attention interventions require an explicit target
metric and record the target they evaluate.

Grad-CAM pools gradients only over selected keys permitted by the model's
actual query-specific attention mask, including causal and local-window
restrictions. Attention propagation, gradient products, integration, causal
differences, metric differences and SAE analysis statistics use at least FP32
for FP16/BF16 inputs and preserve FP64. Model forward/backward tensors and
replayed interventions retain their native dtype; returned analysis scores
may therefore have a wider dtype than model activations. Input and activation
integration select their declared endpoints directly, avoiding cancellation
when a small source value is reconstructed from a large difference. These
numerical and masking conventions are described in the
[attention-map guide](methods/attention_maps.md#what-each-method-computes).
See [numerical precision](NUMERICAL_PRECISION.md) for return dtypes, endpoint
handling, SAE editing and the limits of wider analysis.

## Aligned interventions

`alignment="strict"` checks valid/visual/prompt masks, expanded token IDs, and
adapter-declared grid/position metadata. Source and receiver use identical
coordinates. For intentionally changed text with verified positional alignment,
`alignment="position"` relaxes only token-ID equality. Residual patching does not
remap positions. Path patching can map selected donor sender positions onto
selected base sender positions within an otherwise structurally aligned batch.

### Controlled paths

```python
path = probe.causal.path(
    senders=[(0, 0)], receivers=[(1, 0, "v")],
    sender_tokens="visual", receiver_tokens="visual",
)
result = path.run(corrupt, donor=clean, metric=metric)
```

This executes base, donor, controlled, and receiver-only forwards. It freezes
every attention-head output in the controlled run, then replaces only the chosen
sender positions with donor values. MLPs and normalization recompute; receiver
Q/K/V is cached independently of its frozen head output. `receivers="residual"`
selects the final residual before readout normalization. Multiple endpoints form
one joint intervention with `.run()`. Use `.sweep()` to measure each sender
independently through the configured receiver set:

```python
senders = [(layer, head) for layer, point in sorted(probe.spec.path_heads.items())
           for head in range(point.heads)]
result = probe.causal.path(senders=senders, receivers="residual").sweep(
    clean, donor=corrupt, metric=metric, alignment="position",
)
print(result.tensors["effect"].shape)  # [independent paths, batch]
```

The sweep shares the unmodified base/donor caches, while running both controlled
and receiver-only forwards for every sender. It executes `2 + 2N` forwards for
`N` paths and preserves the same intervention rules as individual `.run()` calls.
An optional `progress(completed, total, sender)` callback reports each completed
path. The [notebook](../demos/path_patching_demo.ipynb) adds dataset preparation,
per-example normalization, plotting, and saved experiment metadata.

`donor_tokens` defaults to `sender_tokens`; explicit masks can select different
role positions in each example, with equal selected counts per pair. Receiver
injection always uses base coordinates. K/V indices name physical KV heads in
GQA. See the [method guide](methods/path_patching.md) for scope and model support.

## Fitted-lens artifacts

Tuned, Attention, Jacobian and MJ-Lens need calibration or `load` before `run`.
The model's parameters and accumulated gradients are preserved.

```python
binding = {
    "model_id": "checkpoint-and-revision",
    "tokenizer_id": "tokenizer-and-revision",
    "readout_id": "final-norm-and-lm-head-v1",
    "calibration_id": "dataset-split-and-preprocessing-v1",
}
lens = probe.lens.tuned(layers=[0], binding=binding)
lens.fit(calibration_inputs, steps=100, lr=1e-3)
lens.save("artifacts/tuned")
restored = probe.lens.tuned(layers=[0], binding=binding).load("artifacts/tuned")
result = restored.run(evaluation_inputs)
```

`fit` accepts one batch; `fit` and `load` return the bound object. Tuned/Attention
retain weights across repeated fits but restart optimizers; their losses are
available in `.losses[layer]`. Jacobian fitting replaces the transport estimate.
For Tuned/Attention dataset calibration, use `.fit_batches(train_inputs, validation_inputs,
directory, ...)`: it caches inputs once, keeps Adam across minibatches, validates
each epoch and saves resumable optimizer state. Inspect `.training_history[layer]`.
Jacobian Lens instead uses `vlmprobe.training.fit_jacobian_lenses`; MJ-Lens uses
its own modality-conditioned `method.fit_batches(input_batches, ...)`. Neither
derivative estimator uses Adam. MJ calibration records causal-pair counts,
odd/even split-half diagnostics and resume counts. Its
[complete guide](methods/mj_lens.md) explains block-input indices and
text-versus-visual output heads.
The [training guide](TRAINING.md) covers this interface, disk-backed readouts,
the validated author-checkpoint importer and grouped Jacobian estimation.
Artifact saving requires explicit identities and writes one file per layer.
Loading validates the binding and dimensions. These files use this library's
format, not a third-party lens checkpoint format.

To reproduce the released Attention Lens initialization, new head decoders can
use `head = model.get_output_embeddings()` followed by
`.fit(calibration_inputs, initial_unembedding=head.weight,
initial_bias=getattr(head, "bias", None), ...)`. The weight shape is `[V,D_residual]`;
the optional bias is `[V]`. No final LayerNorm is inserted into head decoders.
Supplying these options to an already fitted/loaded lens raises an error,
preventing an accidental reset. See the [training guide](TRAINING.md).

## Results and persistence

[MJ-Lens](methods/mj_lens.md) uses `probe.lens.mj(layers=..., source="visual",
target="text", binding=...)`. Its source layers are **block inputs**, with L0
at the embedding stream. `.fit(inputs)` replaces one-batch estimates;
`.fit_batches(input_batches, directory=..., resume=...)` accumulates exact
causal-pair means with odd/even prompt diagnostics. It estimates derivatives
without an optimizer. These coordinates and normalization differ from the
original `probe.lens.jacobian()` and their artifacts are not interchangeable.

Janus visual outputs additionally require a shared calibrated
`fit_visual_readout(...)` and explicit `MJBatch` output-position masks/context.
`.save(directory)` and `.load(directory)` validate model revision, source and
target channel, source/final sites, readout, tokenizer/codebook and calibration
identities. `decode_mj_codes(...)` requires a real codebook grid and native
decoder, and only accepts many visual-source positions. See the
[complete walkthrough](../examples/tutorials/lens_mj.py).

`ProbeResult` contains `.method`, `.tensors`, and `.metadata`.
Outputs are detached and moved to `result_device` (CPU by default).

| Method | Main outputs |
| --- | --- |
| Logit, Tuned, Jacobian | `logits[L,N,V]`, `positions[N,2]`; Jacobian also `transported[L,N,D]` |
| MJ-Lens | `logits[L,N,A]`, `transported[L,N,D]`, `positions[N,2]`; visual output adds `code_ids[L,N]`. `A` is the selected output alphabet. |
| Embed | `token_ids[1,N,K]`, `similarities[1,N,K]`, norms and positions |
| Attention Lens | `head_logits[L,N,H,V]`, summed logits and positions |
| Patchscope | Target `logits[L,B,T_target,V]` |
| Exact layer interventions | Baseline/intervention scores and effect: `[L,B]` or `[L]` |
| Joint steering / knockout | Baseline/intervention scores and effect: `[B]` or scalar |
| Path patching | Baseline/donor/intervention scores and effect: `[B]` or scalar; joint endpoint set |
| Path patching `.sweep()` | Shared baseline/donor `[B]`; intervention/effect `[N,B]`; `senders[N,2]`. Scalar metrics omit the batch axis. |
| Attribution patching | `token_scores[L,B,T]`, batch-summed `estimated_effect[L]` |
| EAP-IG, activation graph | `edge_scores[E]`, scalar endpoints/effect/completeness error |
| EAP-IG, transformer graph | `edge_scores[E]`, `eap_scores[E]`, endpoint scores, edge indices; actual recovery curves from `.evaluate` |
| Attention profile | `entropy[L,B,H,Q]`, `group_mass[L,B,H,Q,G]` |
| Head logit attribution | `head_logits[L,B,H,Q,V]`, `summed_logits[L,B,Q,V]` |
| Rollout / relevance | Final `[B,T,T]` and intermediate `[L,B,T,T]` matrices |
| Attention Grad-CAM | `cam[L,B,Q,K]`, `head_weights[L,B,H,Q]`, selected query/key masks |
| Attention Attribution | Signed `attribution[L,B,H,Q,K]`, summed head/token scores |
| TAM / Beyond Intuition | Final `map[B,T,T]`, perception/state and integrated feedback |
| Chefer DTD | Class logits/targets, real LRP CAMs/gradients and spatial `patch_relevance[B,H_patch,W_patch]` |

`N` packs selected positions in batch/token order. Layer indices, configuration,
and provenance are recorded in metadata.

```python
result.save("outputs/probe")
tensors = torch.load("outputs/probe/tensors.pt", weights_only=True)
```

The accompanying `metadata.json` records the operation. Models and processors
are not serialized. Execution is scoped to one model forward at a time; concurrent
or nested execution on the same adapter is unsupported.


## Sparse autoencoders

Configure `probe.sae.topk(layer=..., tokens=...)`,
fit on frozen-model train/validation inputs with `method.fit(...)`, then inspect
`method.run(...)` and `method.profile(...)`, or measure `method.reconstruct()`
and `method.edit(...)`. `probe.sae.load(...)` restores the same native interface.
The [SAE API](methods/sae/API.md) lists every dictionary/training option and result shape;
the [complete tutorial](methods/sae/examples/README.md) includes image inputs, training,
native artifacts, model cards and visualization.

For author weights, follow the [pretrained visual SAE walkthrough](methods/sae/examples/pretrained.md).
`vlmprobe.evaluation.sae.evaluate(...)` reports reconstruction, sparsity and
optional native-output fidelity/interventions; `localization(...)` evaluates
aligned patch maps. The [evaluation guide](methods/sae/evaluation/README.md)
defines inputs, metrics and controls. Evaluation leaves SAE and backbone weights frozen.

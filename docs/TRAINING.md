# Lens Calibration

[Home](../README.md) / [Documentation](README.md)

Tuned and Attention Lens fit small decoders while keeping the supplied foundation
model frozen. Jacobian Lens estimates a transport matrix from activation
derivatives. Logit Lens, EmbedLens, and an identity Patchscope need no training.

| Method | Calibration | Demonstrated artifacts |
| --- | --- | --- |
| Tuned Lens | One affine residual translator per selected layer, minimizing `KL(model || lens)` | Verified author GPT-2 checkpoint; independently fitted SmolVLM-500M and LLaVA-1.5-7B translators. |
| Attention Lens | Per-head vocabulary decoders, jointly fitted through summed logits | GPT-2 heads at layers 7 and 10, calibrated on 4,096 BookCorpus contexts and validated on 512 separate WikiText-2 contexts. |
| Jacobian Lens | Mean downstream residual Jacobian; no optimizer | All 12 GPT-2 residual layers, calibrated on 256 independent WikiText-2 contexts. |
| [MJ-Lens](methods/mj_lens.md) | Modality-pair causal expectation and optional generation-head mean slopes; no optimizer | Offline tiny native Janus walkthrough fits all four operators and retains odd/even prompt estimates. These fixtures do not establish pretrained accuracy. |

VLM Tuned Lens calibration used 1,024 COCO training images and 256 separate
validation images, with eight caption prediction positions per image. Showcase
GQA inputs were not fitted. These are demonstration-scale calibrations, not
author checkpoints for those VLMs. Fitted artifacts are checkpoint-specific and
are not included in the repository.

## Prepare independent input splits

Split by image or document before tokenization. Keep calibration, held-out
validation, and showcase examples separate; tokens from one image are not
independent examples. Use the model's native chat/image preprocessing and retain
expanded token masks for every batch.

```python
# Generic construction; supply disjoint records from your own dataset.
def prepare_record(record):
    formatted = processor.apply_chat_template(
        record["messages"], tokenize=False, add_generation_prompt=True,
    )
    return probe.prepare(
        prompt=formatted, image=record.get("image"), device="cuda:0",
    )

train_inputs = [prepare_record(record) for record in training_records]
validation_inputs = [prepare_record(record) for record in held_out_records]
showcase_inputs = prepare_record(showcase_record)
```

This example calibrates prompt-end positions. To fit caption positions as in
the COCO calibration, append the caption, provide `TokenLayout.prompt` at the
original prompt boundary, and pass an explicit caption prediction-position
mask. [The API](API.md#metrics-and-explicit-layouts) explains shifted
teacher-forced scoring and layouts. [Data references](REFERENCES.md#datasets-and-evaluation-inputs)
identify the datasets; [demo manifests](../demos/README.md) record the exact
checkpoint and artifact identities used for saved outputs.

## Fit and restore a bound lens

`train_inputs` and `validation_inputs` are separate iterables of native input
dictionaries or `ProbeInputs` on the model's device.

```python
binding = {
    "model_id": "checkpoint@immutable-revision",
    "tokenizer_id": "tokenizer@immutable-revision",
    "readout_id": "native-final-norm-and-head",
    "calibration_id": "dataset-manifest-sha256-and-split",
}
method = probe.lens.tuned(layers=[7, 15], tokens="last_prompt", binding=binding)
method.fit_batches(
    train_inputs, validation_inputs, "artifacts/tuned",
    batch_size=64, epochs=12, lr=3e-4, patience=3, seed=0,
)
print(method.training_history[7]["best_validation_kl"])
result = method.run(showcase_inputs)

restored = probe.lens.tuned(layers=[7, 15], binding=binding).load("artifacts/tuned")
```

The workflow captures selected activations and teacher logits once into CPU
memory, keeps Adam across minibatches, validates each epoch, and saves the best
decoder and resumable optimizer state. The initial decoder is also a selection
candidate. `patience` counts epochs without an improvement of at least
`min_delta` (default `1e-4`). Repeating the same call with matching inputs,
binding, seed, and optimizer settings resumes training; use a fresh directory
to restart a stopped experiment.

For Attention Lens, create a new `probe.lens.attention(...)` and initialize it
from the native unembedding, following the released author code:

```python
head = model.get_output_embeddings()
method = probe.lens.attention(layers=[7], binding=binding)
method.fit_batches(
    train_inputs, validation_inputs, "artifacts/attention",
    initial_unembedding=head.weight,
    initial_bias=getattr(head, "bias", None),
    batch_size=64, epochs=12, lr=3e-4, patience=3,
)
```

These heads are projected residual-space contributions. The decoders apply no
independent final LayerNorm. Dense storage costs approximately
`heads × residual_width × vocabulary` parameters per layer, before optimizer
state; fitting every head on a large VLM can be expensive.

## Use disk-backed teacher caches

For larger calibrations, store selected activations and teacher logits in
separate deterministic train/validation caches. DataLoaders should yield
`activations`, `teacher_logits`, and an optional boolean `mask`.

```python
from vlmprobe.training import train_distribution_lens

history = train_distribution_lens(
    kernel, cached_training_loader, cached_validation_loader,
    "artifacts/layer_7", epochs=12, lr=3e-4, patience=3,
    provenance={"cache_manifest_sha256": cache_hash},
)
```

This is the same persistent trainer used by `fit_batches`; dataset loading,
preprocessing, and cache construction remain caller-owned. The convenience
`fit(inputs)` accepts one batch, retains fitted weights on repeated calls, and
restarts its optimizer. It is useful for an API smoke test, not a replacement
for independent dataset calibration.

For image-indexed shard caches, `vlm_probing.training.validate_cache(cache_root,
data_root, layers)` checks extraction completion, the SHA-256 of
`data_root/samples.json`, disjoint train/validation image IDs, and aligned
activation/teacher positions. It expects `config.json`, `status.json`, and
`train_*.pt` / `validation_*.pt` shards in the cache directory; it does not
construct a dataset or cache.

## Import the author Tuned Lens

`vlm_probing.training.import_author_tuned_lens` imports the compatible released
GPT-2 lens. It verifies architecture, the expected base safetensors SHA, and
every loaded base-model parameter. The expected hash must come from the pinned
author-compatible base revision.

Author hidden-state index 0 names embeddings; library residual layer 0 is after
block 0. The importer maps translator `i+1` to residual layer `i` and keeps a
native identity readout at the final block. See the
[author repository](https://github.com/AlignmentResearch/tuned-lens) and
[text lens demo](../demos/lens_comparison_demo.ipynb). Importing a third-party
checkpoint is separate from this library's normal artifact `.load(...)`.

## Estimate Jacobian Lens

```python
from vlmprobe.training import fit_jacobian_lenses

method, report = fit_jacobian_lenses(
    probe, calibration_input_batches,
    layers=[3, 7, 11], binding=binding, directory="artifacts/jacobian",
    dim_batch=32, skip_first=16, exclude_last=True, tokens="text",
)
result = method.run(independent_showcase_inputs)
```

The estimator groups exact basis-vector VJPs, accumulates prompt-weighted
Jacobians across deterministic batches, and resumes from checkpoints. It does
not use random projections or train an auxiliary classifier. The defaults
(`skip_first=16, exclude_last=True`) require at least **18 valid expanded token
positions per prompt** with unrestricted masks; padding does not count. Source
selectors and target masks must intersect the retained positions as well.
For example, `tokens="last_prompt"` under an all-prompt layout conflicts with
`exclude_last=True`. Short-context studies can explicitly set `skip_first=0,
exclude_last=False`; record this change to the calibration convention.

If any prompt has no remaining source or target, calibration rejects the whole
batch before derivative calculations and reports failing row indices, token
counts and exclusion settings. The dataset estimator also reports the batch
index; neither path silently drops rows. Repeated `method.fit(inputs)` replaces
the previous one-batch Jacobian and emits `RuntimeWarning`. Use
`fit_jacobian_lenses` above to accumulate across input batches; the generic
`method.fit_batches` is only for Tuned and Attention Lens.

Do not fit under `torch.inference_mode()`. Large hidden widths make exact
Jacobians costly; calibration on one checkpoint is not portable to another.

Artifact saving requires explicit checkpoint, tokenizer, readout, calibration,
and site identities. Loading validates these bindings and dimensions. A missing
artifact means that the method must be calibrated for that model, not that the
underlying residual layer cannot support the lens.

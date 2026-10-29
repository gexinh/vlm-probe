# SAE API

[SAE hub](README.md) · [Complete runnable example](examples/README.md) ·
[Pretrained loading without fitting](loading.md)

## Use existing weights

The snippets in this reference assume `probe` is a `Prober` bound to your loaded
backbone and processor, and `inputs` is a prepared input batch. The linked
[complete loading example](loading.md) constructs all three from scratch.

```python
# Native dictionary saved by this library; no fitting is needed.
method = probe.sae.load("outputs/dictionary", device="cpu")
result = method.run(inputs)

# Published GPT-2 layer-0 TopK recipe; downloads and converts the author weights.
# The already-loaded backbone/tokenizer must match this recipe exactly.
method = probe.sae.from_pretrained(
    "sae-6783018355728933dae6d812", cache_dir="outputs/sae_cache", device="cpu",
)
result = method.run(inputs)
```

The [complete inference-only example](loading.md) includes the matching
backbone, tokenizer, fresh input and visualization. `from_pretrained` accepts
only implemented conversion recipes. A catalog record alone does not imply
automatic source-format or backbone compatibility.

| Loading parameter | Default | Meaning |
| --- | --- | --- |
| `card_id` | Required | Published model ID with an implemented automatic loading recipe. |
| `cache_dir` | `None` | Source checkpoint download cache and converted native-artifact cache; defaults to the Hugging Face cache root. |
| `device` | `"cpu"` | Device for the loaded SAE dictionary; backbone placement remains caller-owned. |

## Configure and fit a new dictionary

Here `probe`, `train_inputs` and `validation_inputs` are constructed in a
method's complete example, such as [ReLU SAE](relu.md#complete-live-example).

```python
from vlmprobe.sae import SAETrainingConfig

method = probe.sae.topk(layer=0, tokens="visual", d_sae=1024, k=32)
history = method.fit(
    train_inputs, validation_inputs, output_dir="outputs/fit",
    training=SAETrainingConfig(max_steps=200, batch_size=128),
    preprocessing_id="my-independent-training-split-v1",
)
method.save("outputs/dictionary", model_card=None, registry_dir=None)
loaded = probe.sae.load("outputs/dictionary", tokens=None, binding=None,
                        registry_dir=None, artifact_dir=None, device="cpu")
result = loaded.run(validation_inputs[0])
```

Every named factory is also available through
`probe.sae.method("topk", layer=0, tokens="visual", d_sae=1024, k=32)`.

`.fit(...)` trains the SAE encoder/decoder and any method-specific learned
parameters on activations from the frozen backbone. It does not fine-tune that
backbone. Loading a trained dictionary and calling `.run(...)` performs
inference without optimizer updates. A pretrained backbone and a pretrained SAE
are separate sets of weights; one does not supply the other. See the
[inference-only guide](loading.md) for complete loading examples.

`layer` is one zero-based residual layer from `describe()`. `tokens` accepts the
common token selector (`all`, `visual`, `text`, `last_prompt`, positions or a
boolean mask). A saved batch-specific selector needs matching coordinates at
inference; it is not a dataset-independent image region description.
`d_sae` defaults to four times the inferred activation width;
`expansion_factor=4` controls that default. An omitted `k` is capped at the
inferred dictionary width; an explicitly invalid `k` raises.
`train_inputs` and `validation_inputs` are finite native model input batches or
ProbeInputs. Actual model forwards capture detached CPU float32 activations.
Each split has a configurable `max_cache_bytes` payload cap. It is not a hard
total-RAM limit: concatenation temporarily retains both chunks and their joined
tensor, plus the other split; the trainer also uses in-memory tensors. `train_ids` and `validation_ids`, when provided, must
be disjoint source groups. Exact repeated complete batches are independently
rejected; this does not infer patient/slide overlap for you.

Fit hashes loaded model tensors and processor configuration, records the exact
adapter hook and token selection, and derives a calibration fingerprint from
inputs/activations/source IDs. `preprocessing_id` must describe your actual
external transforms; its default explicitly records that they are unverified.
Load verifies actual model tensors, architecture/configuration, numerical-stack
versions and processor/tokenizer behavior when fingerprints are present.
The frozen backbone is assumed unchanged between method calls; verification
is performed at fit/load, rather than rehashing a large backbone on every run. Source artifacts
without automatic fingerprints require a caller-supplied verified binding;
width matching alone is rejected as an automatic compatibility claim.

Loading creates an inference method. Calling `.fit()` on that loaded object
is rejected: it would otherwise silently discard weights or recipe settings.
To resume training, recreate the original factory/configuration and call fit
with its matching output directory; warm-start fine-tuning is not yet provided.
Resaving modified dictionary weights invalidates inherited numerical/semantic
evidence and marks earlier training metrics as stale.

Torch and explicit loader/sampler RNG states are preserved during collection.
Arbitrary Python/NumPy transforms and persistent worker processes need their own
deterministic data or caching protocol.

## Dictionary hyperparameters

The model-bound factory forwards keyword arguments to `SAEConfig` after inferring
`d_in`. Methods share these fields, but only relevant algorithm options affect
the selected method. Explicit recipe defaults and mathematical scope appear in
the [method inventory](ALGORITHMS.md).

| Parameter | Default | Meaning |
| --- | --- | --- |
| `architecture` | `required` | Native algorithm name; see the method inventory. |
| `d_in` | `required` | Captured activation width; inferred by the model-bound fit workflow. |
| `d_sae` | `required` | Dictionary feature count; fixed for the artifact. |
| `k` | `32` | Positive per-vector TopK count, or batch budget as selected below; pursuit defaults to this many iterations. |
| `l1_coefficient` | `0.01` | Coefficient for L1 sparsity in ReLU/Gated or explicit JumpReLU L1 recipe. |
| `seed` | `0` | Local dictionary initialization seed; does not reset caller RNG. |
| `normalization` | `None` | Optional named normalization override: rms, none, layer_norm, constant_norm_rescale. |
| `input_normalization` | `'scalar'` | Internal coordinate recipe: scalar, none, layer_norm or constant_norm. |
| `normalization_eps` | `1e-05` | Positive stability constant for per-input normalization. |
| `normalization_std_correction` | `0` | Standard-deviation correction: 0 (population) or 1 (sample). |
| `apply_decoder_bias_to_input` | `True` | Subtract decoder bias before encoding normalized inputs. |
| `apply_b_dec_to_input` | `None` | Optional source-format alias of apply_decoder_bias_to_input. |
| `use_decoder_bias` | `True` | Whether reconstruction includes the learned decoder bias. |
| `normalize_decoder` | `True` | Apply per-atom decoder normalization after optimizer steps. |
| `rescale_acts_by_decoder_norm` | `False` | Explicit source recipe that rescales feature values using decoder norms. |
| `auxk_coefficient` | `0.0` | Optional dead-feature auxiliary reconstruction weight; 0 disables it. |
| `auxk_k` | `None` | Number of dead features considered by AuxK; None uses the documented method default. |
| `dead_feature_steps` | `100` | Number of training steps without firing before a feature is dead for AuxK. |
| `jumprelu_threshold` | `0.1` | Positive starting threshold for each JumpReLU feature. |
| `jumprelu_bandwidth` | `0.05` | Rectangle surrogate width for threshold derivatives. |
| `l0_coefficient` | `0.01` | Weight on JumpReLU L0 surrogate sparsity. |
| `jumprelu_sparsity` | `'l0'` | Explicit JumpReLU sparsity recipe: l0 or l1. |
| `jumprelu_ste_to_input` | `False` | Whether threshold surrogate gradients also reach encoder preactivations. |
| `batch_budget` | `'average'` | average: k times number of vectors; global: k across the entire minibatch. |
| `threshold_ema` | `0.99` | EMA decay for BatchTopK inference thresholds. |
| `threshold_init` | `'first_batch'` | Initialize inference threshold from first_batch or zero_ema. |
| `matryoshka_widths` | `()` | Increasing prefix widths ending at d_sae; empty selects half/full widths. |
| `matryoshka_weights` | `()` | Positive weights for each nested reconstruction loss; empty means equal weights. |
| `dictionary` | `'learned'` | learned or archetypal; archetypal method names select the latter. |
| `archetypal_candidates` | `128` | Maximum number of training-only anchor vectors. |
| `archetypal_delta` | `1.0` | Bound on the archetypal dictionary relaxation component. |
| `pursuit_iterations` | `None` | Maximum greedy support additions; None uses k. |
| `residual_tolerance` | `0.01` | Residual stopping tolerance in pursuit. |
| `stop_on_duplicate_support` | `True` | Stop matching pursuit when no new support is added. |
| `omp_nnls_iterations` | `32` | Projected-gradient refit steps for nonnegative OMP; finite approximation, not exact NNLS. |
| `quantization_levels` | `4` | Number of learned scalar code levels; QSAE is experimental. |
| `hard_quantization` | `False` | Use discrete quantization rather than the soft code path. |

## Training parameters

`SAETrainingConfig` is shared by all algorithms. The `normalization` field here
chooses whether the training-only RMS scale is fitted for scalar-normalized
models (`rms`) or disabled (`none`); per-input dictionary normalization takes
precedence. These are not different source-specific trainers.

| Parameter | Default | Meaning |
| --- | --- | --- |
| `epochs` | `10` | Maximum epochs unless a step/vector budget is chosen. |
| `lr` | `0.001` | Adam learning rate. |
| `batch_size` | `256` | Number of activation vectors per optimizer minibatch. |
| `patience` | `3` | Stop after this many epochs without min_delta validation-loss improvement. |
| `min_delta` | `0.0001` | Minimum validation-loss improvement for best-model selection. |
| `seed` | `0` | Local dictionary initialization seed; does not reset caller RNG. |
| `normalization` | `'rms'` | Fit train-only scalar RMS (rms) or disable it (none); per-input model normalization takes precedence. |
| `max_steps` | `None` | Optional exact optimizer-step budget; mutually exclusive with max_vectors. |
| `max_vectors` | `None` | Optional exact cumulative training-vector budget; mutually exclusive with max_steps. |
| `device` | `'cpu'` | Dictionary training device (cpu or cuda); the caller owns backbone placement. |
| `resume` | `True` | Resume matching optimizer/RNG/cursor/method state from output_dir. |
| `grad_clip` | `5.0` | Positive maximum gradient norm applied before Adam updates. |
| `progress_every` | `20` | Progress/checkpoint write interval in optimizer steps. |

## Probe operations and outputs

| Call | Parameters | Tensors |
| --- | --- | --- |
| `method.run(inputs)` | Native input batch | `features[N,F]`, `activation[N,D]`, `reconstruction[N,D]`, `reconstruction_error[N,D]`, `l0[N]`, `positions[N,2]` |
| `method.profile(top_k=10).run(inputs)` | Positive top count per feature; ranks absolute values and retains signs | Feature mean/variance/firing frequency/min/max; `top_activation_values[K,F]`, `top_positions[K,F,2]`. Mean and variance use at least FP32 and preserve FP64. |
| `method.reconstruct().run(inputs, metric=None)` | Optional task metric | Baseline/edited logits or representation outputs; optional baseline/edited score and score change |
| `method.edit(feature_ids=[...], scale=0., mode="error_preserving").run(inputs, metric=None)` | Unique valid IDs; finite scaling; error_preserving or reconstruction | Same output/score comparison as above, with explicit intervention metadata |

`N` packs selected batch/token positions, `F=d_sae`, `D=d_in`; the two position
columns are batch and native expanded-token index. Signed features count as
active when nonzero. Profiles rank absolute activation, retaining negative
values. Representation outputs remain vectors; they are not fabricated logits.

Default feature editing uses `x + decode(z_edited) - decode(z)` with the same
per-input normalization context reused for both decodes. `scale=1` returns an
exact no-op, including native bf16 precision. Reconstruction is deliberately a
separate control. Generic low-level operations remain available as
`probe.sae.features/profile/reconstruct/edit(layer=..., sae=..., binding=...)`.

The decoded difference and its addition to the original residual use at least
FP32 and preserve FP64; only the final replacement is cast to the backbone's
dtype. Profile mean/variance and loss statistics also use wider analysis
precision. See [numerical precision](../../NUMERICAL_PRECISION.md).

## Lower-level training, artifacts and discovery

```python
from vlmprobe.sae import (SAEConfig, SparseAutoencoder, SAETrainingConfig,
                         fit_sae, list_methods, list_models, get_model_card,
                         load_artifact, save_artifact, import_checkpoint)
```

`fit_sae(model, train_vectors, validation_vectors, output_dir, training=..., provenance=...)`
uses the same engine as model-bound fitting. Vectors must be finite floating
`[N,D]` in the bound native coordinate system. Legacy `train_sae(...)` is only a
compatibility adapter to this engine.

`save_artifact(model, path, model_card=None, registry_dir=None)` writes either
config.json/weights.pt/model_card.json or the same schema in a safe `.pt`
envelope. `load_artifact(path, expected_binding=None, device="cpu")` validates
checksums, types, exact state, config and optional binding. Inference weights
and optimizer resume state are separate. Browse `list_models(...)` and `get_model_card(id, ...)`;
see [model cards](models/README.md) for local registration, published pretrained
cards and explicit source checkpoint conversion.


### Model discovery parameters

```text
list_models(*, modality=None, method=None, status=None, query=None,
            artifact_kind="sae", registry_dir=None, include_builtin=True, limit=None)
get_model_card(card_id, *, registry_dir=None)
```

| Parameter | Meaning |
| --- | --- |
| `modality` | Filter language, vision or multimodal resources. Unreported source metadata remains unspecified. |
| `method` | Filter an algorithm name such as relu, topk or jumprelu. |
| `query` | Case-insensitive search over model, hook, release and other card metadata. |
| `artifact_kind` | `"sae"` selects dictionary weights; `"backbone"` selects activation-extraction models; `None` includes both. |
| `registry_dir` | Optional directory containing your saved artifact/model cards. |
| `include_builtin` | Include the packaged author checkpoint catalog; use False to browse only local runs. |
| `status` | Optional artifact metadata filter, for example published or trained. |
| `limit` | Optional positive maximum result count. |

Cards provide source model/hook/configuration, dimensions when reported, direct
checkpoint links, and a link to our model information page. `load_model(...)`
restores a registered native artifact; source weights use the explicit
checkpoint conversion interface before model-bound probing.

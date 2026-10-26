# SAE evaluation

[Home](../../../README.md) / [SAE](../README.md) / Evaluation

`vlmprobe.evaluation` evaluates frozen SAE dictionaries on identified native
activations or full model inputs. Its reports separate reconstruction, sparsity,
task/output fidelity, feature interventions and spatial localization. It does not
train SAEs or depend on external SAE evaluation toolkits.

Start with the [complete pretrained visual notebook](../../../../demos/11_sae_pretrained.ipynb)
and [checkpoint/input guide](../examples/pretrained.md). They include raw images,
author preprocessing, source tensor conversion, independent calibration,
held-out extraction, controls and library plots.

## Public calls and parameters

```python
from vlmprobe import evaluation

# `method` is a loaded model-bound SAE, for example probe.sae.load(...).
config = evaluation.SAEEvaluationConfig(chunk_size=128)
report = evaluation.sae.evaluate(
    method,
    [evaluation.EvaluationBatch(inputs, ids=["image-001"])],
    config=config,
    reference_mean=calibration_mean,
    provenance={"split": "held out"},
)
print(report.summary)
report.save("outputs/sae_evaluation")
```

This API fragment assumes already-prepared `inputs`, a loaded `method`, and a
native-width `calibration_mean`; the linked notebook defines all three end to end.

| Argument | Meaning |
| --- | --- |
| `method` | A fitted/loaded `probe.sae` method bound to the exact backbone, hook, preprocessing and token selection |
| `batches` | Iterable or callable returning `EvaluationBatch(inputs, ids, group_ids=None, labels=None, metric=None)`; IDs identify original examples |
| `reference_mean` | Optional native activation-width vector, usually computed on independent calibration inputs; fixes the global FVU denominator |
| `provenance` | Caller-supplied split/input/target metadata saved alongside the method binding, source identity, native-value digest and configuration |

`EvaluationBatch.metric` is required for fidelity/interventions and must return one
finite score per original example. A plain callable accepts the native output
tensor; a metric object may implement `score(output, layout)`. Representation
adapters require a vector metric, not a language/classification logit metric.
Repeated views can share `group_ids`; example IDs remain unique in model evaluation.

| `SAEEvaluationConfig` field | Default | Meaning |
| --- | --- | --- |
| `metrics` | `("reconstruction", "sparsity")` | Add `"fidelity"` or `"intervention"` when full inputs and an explicit metric are available |
| `chunk_size` | `1024` | Positions per statistics/encode chunk; reduces temporary memory for activation-cache evaluation |
| `feature_ids` | `()` | Explicit unique dictionary IDs for intervention; required if intervention is requested |
| `intervention_scales` | `(0., .5, 1., 2.)` | Multipliers for selected feature contributions; zero deletes and one is the identity control |
| `random_control_repeats` | `3` | Number of count-matched random groups when explicit groups are absent |
| `random_feature_ids` | `()` | Caller-frozen, disjoint, count-matched control groups; can be frequency matched on calibration statistics |
| `seed` | `0` | Seed for selecting fallback count-matched controls |

Full-input evaluation still uses `method.run(...)`, which materializes that model
batch's features. Use small model batches for wide dictionaries. `chunk_size` is
not a substitute for controlling model-batch size.

## Reconstruction and sparsity

`evaluation.sae.evaluate_activations(...)` accepts an encode/decode dictionary or a
loaded method and replayable `ActivationBatch` records:

```python
cached_batches = [evaluation.ActivationBatch(
    native_activations,  # [example, token, hidden] or [position, hidden]
    ids=example_ids,
    mask=valid_patch_mask,  # optional bool mask matching all leading dimensions
)]
report = evaluation.sae.evaluate_activations(
    method, cached_batches, reference_mean=calibration_mean,
    config=evaluation.SAEEvaluationConfig(chunk_size=128),
)
```

The evaluator records native-space MSE, global FVU, normalized MSE, average L0,
fraction of inactive dictionary features, and per-feature firing counts, mean,
variance, minimum and maximum. Nonzero signed activations count as active.
Metrics compare reconstructed values to the original backbone activation precision,
including when the dictionary runs at lower precision. Per-position source
normalization is restored before reconstruction metrics.

FVU is total squared reconstruction error divided by total squared deviation from
the supplied reference mean. Without `reference_mean`, the exact global evaluation
mean is used. It is not an average of per-batch ratios. A zero denominator produces
`null` and `fvu_defined=False`, not a fabricated zero. Partition-independent native
activation and sample digests help verify matched comparisons.

`evaluation.sae.compare(...)` and `compare_activations(...)` accept a mapping from
names to methods and reject different model/hook/preprocessing/token bindings or
changed input digests. Iterables must be replayable, or supplied as a fresh iterator
factory. Numerical dimensions alone do not make checkpoints comparable.

## Fidelity and feature interventions

```python
targets = (123, 456)
controls = evaluation.sae.matched_random_features(
    calibration_report.feature_statistics, targets, repeats=3, seed=17,
)
config = evaluation.SAEEvaluationConfig(
    metrics=("reconstruction", "sparsity", "fidelity", "intervention"),
    feature_ids=targets,
    intervention_scales=(0., .5, 1.),
    random_feature_ids=controls,
)
report = evaluation.sae.evaluate(
    method, [evaluation.EvaluationBatch(inputs, ids=example_ids, metric=metric)],
    reference_mean=calibration_mean, config=config,
)
```

The pipeline measures full residual reconstruction, exact identity, selected feature
scaling/deletion, and random-feature controls. Error-preserving edits add only the
decoded feature difference to the original residual. They preserve the native
reconstruction error. Scores are stored as `baseline_score`, `edited_score` and
`score_change` under an operation prefix; score change is **edited minus baseline**.

Choose features and frequency-matched controls on independent calibration data,
then freeze them before evaluation. `matched_random_features` samples among the
32 closest available firing frequencies, excludes targets, and returns disjoint
groups across repeats. Without explicit groups, random controls match feature
count only; the report records this distinction. A random control is not a proof
that a chosen feature has a unique semantic or causal role.

`evaluation.sae.language_fidelity(...)` additionally evaluates explicit
`LanguageFidelityBatch` next-token labels and boolean score masks in the exact
expanded native-logit coordinates. It reports next-token NLL and teacher KL;
callers exclude padding, image placeholders and unscored positions. Do not use
text-token coordinates before visual-token expansion as a substitute.

## Spatial localization

```python
localization = evaluation.sae.localization(
    feature_maps,   # [example, grid_height, grid_width]
    target_masks,   # [example, image_height, image_width], boolean / 0-1
    ids=example_ids, sign="positive",
    provenance={"geometry": "exact native image resize/crop"},
)
```

The caller aligns masks to the actual processed image. Maps are bilinearly resized
to mask resolution without min-max normalization. `sign` selects positive,
negative or absolute feature evidence. Returned metrics include coverage,
point-at-1 across all examples, point-at-1 conditional on activation, and mass
inside boxes conditional on activation. A zero map is inactive, has point-at-1=0,
and has undefined mass (`null`). Every target mask must be nonempty. Bounding-box
membership is observational and is not automatically a segmentation or causal map.

## Optional probes and visualization

`evaluation.sae.pca_features(...)` fits PCA on training features only.
`evaluation.sae.sparse_probe(...)` trains a sparse linear diagnostic probe, selecting
its regularization on validation inputs and measuring the held-out split once.
These tools fit the **diagnostic probe**, not the frozen SAE. Group IDs must be
disjoint across train/validation/test. Install their optional dependencies with
`pip install -e '.[evaluation]'`; NumPy/scikit-learn imports are lazy.

```python
from vlmprobe.visualization import plot_sae_evaluation, plot_sae_comparison

plot_sae_evaluation(report, top_k=20)  # firing frequency by dictionary ID
plot_sae_comparison(matched_reports, x="l0", y="fvu")
```

`EvaluationReport.save(directory)` writes strict `summary.json`, per-example CSV
and per-feature CSV. Undefined metrics serialize as `null`. The report retains
configuration, source/binding metadata, controls, split identity and native-value
digests. It does not serialize raw images, complete activations or model weights.

[SAE methods](../README.md) · [Pretrained walkthrough](../examples/pretrained.md) ·
[Model cards](../models/README.md) · [Training](../training/README.md)

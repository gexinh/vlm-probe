# EAP-IG

[Home](../../README.md) / [Methods](README.md) / EAP-IG

Rank residual-message edges using input-path integrated gradients, then test proposed circuits with real complement-edge interventions.

## API and parameters

`probe` is a `Prober` bound to a loaded model; `inputs` contains prepared model inputs. The blocks below are API signatures: `*` marks keyword-only parameters, not an argument to copy. The [complete example](#complete-example-input-to-output) defines the model, inputs, and all other variables before running the method.

The two configurations below select their graph explicitly. The factory's
default is `graph="activation"`; use `graph="transformer"` for the GPT-2 path.

```text
# Original input-embedding-path algorithm on the native GPT-2 graph.
method = probe.causal.eap_ig(
    *, edges=None, steps=32, graph="transformer", quadrature="left",
)
attribution = method.run(
    inputs, *, donor, metric, alignment="position", progress=None,
)

# Separate activation-space variant; requires adapter-declared edge captures.
activation_method = probe.causal.eap_ig(
    *, edges=None, steps=32, graph="activation", quadrature="left",
)
activation_result = activation_method.run(
    inputs, *, source, metric, alignment="strict",
)
```

| Parameter | Default | Meaning and constraints |
| --- | --- | --- |
| `graph` | `"activation"` | Explicitly use `"transformer"` for original input-embedding-path EAP-IG on native GPT-2. `"activation"` uses a separate internal-message interpolation variant. |
| `edges` | `None` | Native transformer graph supplies all real edge names; passing `edges` there is rejected. Activation graph defaults to all declared `spec.edges`, requiring a nonempty unique supported list. |
| `steps` | `32` (factory) | Positive integer: native graph uses this many gradient samples; activation-space variant uses this many intervals (`steps + 1` endpoints). Example uses 5 native samples; check convergence for an experiment. |
| `quadrature` | `"left"` | Native graph: `"left"` uses 0..(m−1)/m, matching author code; `"right"` uses 1/m..1, matching the paper equation. Activation variant uses its own fixed trapezoidal rule; this option applies to the native graph. |
| `inputs` | required | Clean native model inputs for the transformer graph; receiver inputs for the activation variant. |
| `donor` | required (transformer) | Structurally aligned corrupted reference inputs. |
| `source` | required (activation) | Aligned endpoint source inputs; separate name prevents confusing the two algorithms. |
| `metric` | required | Differentiable output task score, scalar or `[B]`. Native graph defines loss as negative batch-mean task score. |
| `alignment` | `"position"` (transformer), `"strict"` (activation) | Token-ID equality can be relaxed, never structural validity. Verify role positions yourself. |
| `progress` | `None` (transformer `run`) | Optional `(completed, total, alpha)` callback after each integration gradient sample. |

A metric is a `TokenMargin`, `SequenceLogProb`, or a callable `metric(logits)` returning a scalar or `[B]`; it is evaluated on full-sequence logits. `TokenMargin(positive, negative, position="last_prompt")` compares two vocabulary IDs at one selected prediction position. For multi-token answers, explicitly append answer tokens, preserve the original prompt boundary, and use `SequenceLogProb(answer_mask, reduction="sum")` or `"mean"`. First-subword margins are not complete-answer probabilities.

### Native graph computation

Cache clean and corrupted residual-message source outputs. Hold their difference `delta_u = corrupt_u - clean_u` fixed. Differentiate loss with respect to each independently editable destination input along `x_corrupt + alpha * (x_clean - x_corrupt)`. EAP uses the clean endpoint gradient; EAP-IG averages those gradients across input-embedding interpolation points. Edge scores sum over valid sequence positions and residual dimensions.

The audited native graph currently supports standard **text-only `GPT2LMHeadModel`**. It separates each head's pre-LayerNorm Q/K/V destination input, MLP input, and final readout input. Standard GPT-2 has 32,491 independently replaceable edges. Cross-attention, reordered/upcast attention, and VLM graphs are not currently claimed. A positive score estimates increased loss when an edge is corrupted. Scores are not probabilities, and their sum has no input-feature IG completeness guarantee.

### Circuit evaluation and exact edge checks

```text
evaluation = method.evaluate(
    inputs, *, donor, metric, attribution, budgets, alignment="position",
    rankings=("eap", "eap_ig"), strategy="greedy", epsilon=1e-6, progress=None,
)
exact_edges = method.patch_edges(
    inputs, *, donor, metric, excluded_edges, alignment="position",
)
```

| Parameter | Default | Meaning and constraints |
| --- | --- | --- |
| `donor / metric` | required | Same corrupted-reference and differentiable task-score contracts as native `run`. |
| `attribution` | required | Native graph `ProbeResult` with matching edge names, used for ranking. |
| `budgets` | required | Nonempty list of integer edge counts between 0 and the full graph edge count, inclusive. |
| `rankings` | `("eap", "eap_ig")` | Which score rankings to evaluate; one or both names. |
| `strategy` | `"greedy"` | Backward connected greedy selection from readout using absolute scores, with pruning. `"topk"` is the explicit alternative. |
| `epsilon` | `1e-6` | Finite positive threshold for the absolute clean-minus-reference margin used to mark normalization eligibility. |
| `alignment` | `"position"` | Same native input-pair structural checks. |
| `progress` | `None` | Optional `(ranking, requested_budget, actual_retained_count)` callback after each circuit forward. |
| `excluded_edges` | required (`patch_edges`) | Nonempty list of graph edge indices. Each edge is corrupted independently; this is not a joint edge knockout. |

Excluded edges take cached corrupted source outputs; retained edges use source values from the **current intervened forward**, not clean caches. Every excluded edge in the graph is actually replaced. Requested budget and actual retained counts may differ after connectivity pruning.

`run()` returns `edge_scores[E]`, `eap_scores[E]`, `edge_indices[E,2]`, baseline/donor scores, and path losses. `evaluate()` returns `circuit_score[R,K,B]`, `faithfulness[R,K,B]`, `eligible[B]`, `requested_edges[K]`, `retained_edges[R,K]`, retained masks, and all/none-retained endpoint errors. Scalar metrics omit `B`. Faithfulness is `(circuit_score - donor_score)/(baseline_score - donor_score)`. Ineligible zero-denominator placeholders must be masked. Negative recovery and recovery above one are valid.

The activation variant requires independently replaceable `ModelSpec.edges`, interpolates selected internal messages jointly with trapezoidal quadrature, and returns activation-path `edge_scores`, endpoint effects and a completeness error for that separate path. It is not the original input-embedding EAP-IG. Both graph variants compute source differences, path arithmetic, gradient contraction and scores in at least FP32 for FP16/BF16 messages and preserve FP64. Replayed model inputs retain their native dtype. The activation path selects its first receiver and final masked-source tensors directly; it does not reconstruct those endpoints by subtracting and adding potentially cancelling values. Endpoint score differences also use the wider analysis precision.

Install `pip install "vlmprobe[transformers,visualization]"`. The following block is complete: it loads GPT-2, constructs an author-published IOI prompt pair, checks candidate tokenization, runs the methods, saves results, and draws the built-in plots. First use downloads GPT-2 (about 0.5 GB); CPU works and CUDA is faster. No method training is needed. `VLM_PROBING_MODEL` can point to an existing local checkpoint.

## Complete example: input to output

```python
import os
from pathlib import Path
import torch
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, GPT2LMHeadModel
from vlmprobe import Prober, TokenMargin
from vlmprobe.visualization import plot_causal_heatmap, plot_intervention_curves

# First run downloads GPT-2 (~0.5 GB). Use an existing local checkpoint if desired.
model_id = os.environ.get("VLM_PROBING_MODEL", "openai-community/gpt2")
device = os.environ.get("VLM_PROBING_DEVICE", "cuda:0" if torch.cuda.is_available() else "cpu")
model = GPT2LMHeadModel.from_pretrained(model_id, attn_implementation="eager").to(device).eval()
tokenizer = AutoTokenizer.from_pretrained(model_id)
probe = Prober(model, processor=tokenizer)
print({name: info for name, info in probe.describe()["methods"].items()
       if name.startswith("causal.")})
output_dir = Path(os.environ.get("VLM_PROBING_OUTPUT", "outputs/tutorials/causal"))
output_dir.mkdir(parents=True, exist_ok=True)
# An explicit author-published IOI pair (row 826); no result snapshots are used.
# Source: hannamw/eap-ig-faithfulness, data/ioi/gpt2.csv; see docs/REFERENCES.md.
clean_prompt = "When Amy and Laura got a snack at the house, Laura decided to give it to"
reference_prompt = "When Amy and Laura got a snack at the house, Nicholas decided to give it to"

def prepare_text(text):
    # The authors prepend GPT-2's BOS token. Keep the convention explicit.
    ids = [tokenizer.bos_token_id, *tokenizer.encode(text, add_special_tokens=False)]
    input_ids = torch.tensor([ids], device=device)
    return {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)}

inputs = prepare_text(clean_prompt)
reference = prepare_text(reference_prompt)
assert inputs["input_ids"].shape == reference["input_ids"].shape, "Pair token lengths must match"
positive_ids = tokenizer.encode(" Amy", add_special_tokens=False)
negative_ids = tokenizer.encode(" Laura", add_special_tokens=False)
assert len(positive_ids) == len(negative_ids) == 1, "TokenMargin requires one-token candidates"
metric = TokenMargin(positive=positive_ids[0], negative=negative_ids[0])
print("Clean:", clean_prompt)
print("Reference:", reference_prompt)
print("Candidates:", tokenizer.decode(positive_ids), tokenizer.decode(negative_ids))
# Check the role alignment, not just the equal length.
for position, (a, b) in enumerate(zip(inputs["input_ids"][0], reference["input_ids"][0])):
    print(position, repr(tokenizer.decode([a.item()])), repr(tokenizer.decode([b.item()])))
# Choose the original input-embedding-path algorithm explicitly.
method = probe.causal.eap_ig(graph="transformer", steps=5, quadrature="left")
attribution = method.run(inputs, donor=reference, metric=metric, alignment="position")
print("Graph edges:", len(method.graph.edge_names))
print("EAP-IG scores:", attribution.tensors["edge_scores"].shape)
print("EAP scores:", attribution.tensors["eap_scores"].shape)
print("Clean / reference margins:", attribution.tensors["baseline_score"],
      attribution.tensors["donor_score"])
attribution.save(output_dir / "eap_ig")
# Every plotted circuit value comes from a fresh forward intervention.
# Use separate attribution/evaluation pairs for a generalization experiment.
budgets = [0, 100, 1000, len(method.graph.edge_names)]
evaluation = method.evaluate(
    inputs, donor=reference, metric=metric, attribution=attribution,
    budgets=budgets, rankings=("eap", "eap_ig"), strategy="greedy",
    alignment="position", epsilon=1e-6,
)
print("Eligible nonzero denominator:", evaluation.tensors["eligible"])
print("All-retained endpoint error:", evaluation.tensors["all_retained_error"])
print("None-retained endpoint error:", evaluation.tensors["none_retained_error"])
evaluation.save(output_dir / "eap_ig_faithfulness")
# Corrupt three high-scoring edges individually and compare to the estimate.
edge_ids = attribution.tensors["edge_scores"].abs().topk(3).indices.tolist()
exact_edges = method.patch_edges(inputs, donor=reference, metric=metric,
                                 excluded_edges=edge_ids, alignment="position")
for row, edge in enumerate(edge_ids):
    print(method.graph.edge_names[edge],
          "estimated loss increase:", float(attribution.tensors["edge_scores"][edge]),
          "actual loss increase:", float(exact_edges.tensors["loss_increase"][row, 0]))
from vlmprobe.visualization import plot_connections

fig, axes = plt.subplots(1, 2, figsize=(16, 4.8), constrained_layout=True,
                         gridspec_kw={"width_ratios": [1, 1.4]})
curves = {}
for index, name in enumerate(evaluation.metadata["rankings"]):
    # Never plot a zero placeholder from an ineligible denominator as a failure.
    values = evaluation.tensors["faithfulness"][index, :, 0].clone()
    if not bool(evaluation.tensors["eligible"][0]):
        values[:] = float("nan")
    curves[name.upper().replace("_", "-")] = {
        "x": evaluation.tensors["retained_edges"][index], "y": values,
    }
if bool(evaluation.tensors["eligible"][0]):
    plot_intervention_curves(
        curves, ax=axes[0], baseline=1.0, xlabel="Actual retained edges",
        ylabel="Recovered clean-vs-reference margin", title="Circuit intervention, not a score sum",
    )
    axes[0].set_xscale("symlog", linthresh=100)
    axes[0].set_xlim(0, len(method.graph.edge_names))
else:
    axes[0].text(.5, .5, "Clean/reference margin too similar to normalize", ha="center")
    axes[0].set_axis_off()
# The graph draws only supplied high-scoring edges; it does not discover roles.
indices = attribution.tensors["edge_scores"].abs().topk(8).indices.tolist()
edges = []
for index in indices:
    u, v = method.graph.edge_indices[index].tolist()
    edges.append((method.graph.source_names[u], method.graph.destination_names[v],
                  float(attribution.tensors["edge_scores"][index])))
nodes = {node for source, receiver, _ in edges for node in (source, receiver)}
def layer_coordinate(node):
    if node == "input": return -1
    if node == "readout": return model.config.n_layer
    return int(node.split(".")[0][1:])
plot_connections(edges, node_layers={node: layer_coordinate(node) for node in nodes},
                 top_k=8, ax=axes[1], title="Candidate edges ranked by EAP-IG",
                 value_label="Estimated corrupted-edge loss increase")
for label in axes[1].texts:
    label.set_fontsize(7)
fig.savefig(output_dir / "causal-eap-ig.png", dpi=150, bbox_inches="tight")
plt.show()
```
### Cost and evidence

With `m` samples, native attribution uses `3 + m` forwards and `1 + m` backwards. Circuit evaluation uses `4 + R*K` forwards; checking `n` edges uses `2+n`. No model training occurs. Do not disable gradients during attribution. This small example selects and tests on the same one-example input; use held-out prompt pairs for a generalization study. Increase integration resolution only after establishing the input/graph contract and endpoint checks.

[Live notebook](../../demos/eap_ig_demo.ipynb) · [Standalone script](../../examples/tutorials/causal_eap_ig.py) · [Paper](https://arxiv.org/abs/2403.17806) · [Native graph implementation](../../src/vlm_probing/causal/transformer_eap_ig.py)

## Example visualization

This preview was generated by running the code on **pretrained GPT-2** with the explicit IOI pair below. It is an example-level result, not a full-dataset reproduction. The notebook reruns inference and does not load the preview.

![Circuit-recovery measurements and high-scoring estimated edges](../assets/tutorials/causal-eap-ig.png)

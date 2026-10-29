# Attribution Patching

[Home](../../README.md) / [Methods](README.md) / Attribution Patching

Estimate a residual replacement effect from the receiver gradient and the source-minus-receiver activation difference. Use measured activation patching to validate important estimates.

## API and parameters

`probe` is a `Prober` bound to a loaded model; `inputs` contains prepared model inputs. The blocks below are API signatures: `*` marks keyword-only parameters, not an argument to copy. The [complete example](#complete-example-input-to-output) defines the model, inputs, and all other variables before running the method.

```text
method = probe.causal.attribute(*, layers=None, tokens="visual")
result = method.run(inputs, *, source, metric, alignment="strict")
```

| Parameter | Default | Meaning and constraints |
| --- | --- | --- |
| `layers` | `None` | All available residual layers; pass one integer or a nonempty list of unique available layer IDs. Check `probe.describe()`. |
| `tokens` | `"visual"` | Positions to replace/attribute. Use expanded token selectors described below. |
| `inputs` | required | Receiver/base input: native model keyword dictionary or `ProbeInputs`. Full sequence, no KV cache. |
| `source` | required | Aligned donor input providing replacement residuals. It need not be the clean condition; define the intervention direction explicitly. |
| `metric` | required | Output score to compare. See metric contract below. |
| `alignment` | `"strict"` | `"strict"` checks token IDs, masks, and declared alignment metadata. `"position"` permits different token IDs, retaining all structural checks. Neither establishes semantic correspondence automatically. |

Selections use zero-based **expanded model coordinates**, including image tokens. Common token selectors are `"all"`, `"visual"`, `"text"`, `"last_prompt"`, `"prediction"`, an integer/list of positions, or a boolean `[B,T]` mask. `"last_prompt"` selects one final context token; `"prediction"` additionally selects generated positions when `TokenLayout.prompt` preserves the original context. `"visual"`/`"text"` need a visual-token layout. For text-only examples, explicitly choose `tokens="all"` or `"last_prompt"` rather than the visual-token default.

A metric is a `TokenMargin`, `SequenceLogProb`, or a callable `metric(logits)` returning a scalar or `[B]`; it is evaluated on full-sequence logits. `TokenMargin(positive, negative, position="last_prompt")` compares two vocabulary IDs at one selected prediction position. For multi-token answers, explicitly append answer tokens, preserve the original prompt boundary, and use `SequenceLogProb(answer_mask, reduction="sum")` or `"mean"`. First-subword margins are not complete-answer probabilities.

### Computation and returned tensors

For selected residual `h` and source `h_source`, differentiate the batch-summed metric with respect to the receiver residual and compute `(h_source - h) * gradient`. Sum hidden dimensions for `token_scores[L,B,T]`, then selected batch/token positions for `estimated_effect[L]`. Masked positions have zero score. The difference, products and sums use at least FP32 for FP16/BF16 residuals and preserve FP64; the gradient itself is taken in the model's native dtype. The estimate is **batch-summed**; compare against `exact.effect.sum(-1)` for a batched exact experiment, rather than one arbitrary batch row.

The source forward provides the activation difference; the receiver replay requires gradients. Do not wrap this call in `torch.no_grad()` or inference mode. Frozen model weights are allowed: the method differentiates retained activations without training model parameters. Large changes and nonlinear interactions can make the estimate differ in magnitude or sign from actual replacements.

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
layers = sorted(probe.spec.residuals)
# Both methods replace/estimate the same reference-minus-clean change.
# This is corruption of the clean input, not restoration of the reference.
exact = probe.causal.patch(layers=layers, tokens="last_prompt").run(
    inputs, source=reference, metric=metric, alignment="position",
)
estimated = probe.causal.attribute(layers=layers, tokens="last_prompt").run(
    inputs, source=reference, metric=metric, alignment="position",
)
print("Exact effect [layer,batch]:", exact.tensors["effect"])
print("Estimated effect [layer]:", estimated.tensors["estimated_effect"])
# A separate all-position attribution scan supplies the token heatmap.
token_map = probe.causal.attribute(layers=layers, tokens="all").run(
    inputs, source=reference, metric=metric, alignment="position",
)
print("Token scores [layer,batch,token]:", token_map.tensors["token_scores"].shape)
exact.save(output_dir / "activation_patching")
estimated.save(output_dir / "attribution_patching")
token_map.save(output_dir / "attribution_patching_all_positions")
fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), constrained_layout=True)
plot_intervention_curves(
    {"Measured patch": exact.tensors["effect"][:, 0],
     "First-order estimate": estimated.tensors["estimated_effect"]},
    x=layers, ax=axes[0], xlabel="Decoder layer", ylabel="Change in Amy-minus-Laura logit margin",
    title="The approximation can differ from the measured effect",
)
labels = ["<BOS>" if i == 0 else tokenizer.decode([token])
          for i, token in enumerate(inputs["input_ids"][0].tolist())]
plot_causal_heatmap(
    token_map, tensor="token_scores", column_labels=labels, ax=axes[1],
    title="Separate attribution scan across all positions",
    value_label="Estimated margin change",
)
axes[1].tick_params(axis="x", labelsize=7)
fig.savefig(output_dir / "causal-patching.png", dpi=150, bbox_inches="tight")
plt.show()
```
### Reading the visualization

The left curve compares two quantities in the same score units and intervention direction. The right panel is a separate all-position attribution scan; it does not share the final-prompt-token scope of the line comparison and does not measure individual token replacements. For batch size greater than one, use `exact.tensors["effect"].sum(-1)` for the left curve and choose `batch_index` explicitly for the token map.

The public factory is shared across supported residual adapters, including VLMs. Source/receiver alignment and model-specific image preparation follow [activation patching](activation_patching.md); callable metrics must remain differentiable.

[Live notebook](../../demos/activation_patching_demo.ipynb) · [Standalone script](../../examples/tutorials/causal_activation_patching.py) · [Reference](../REFERENCES.md#causal-methods) · [Implementation](../../src/vlm_probing/causal/attribution_patching.py)

## Example visualization

This preview was generated by running the code on **pretrained GPT-2** with the explicit IOI pair below. It is an example-level result, not a full-dataset reproduction. The notebook reruns inference and does not load the preview.

![Comparison of attribution estimates with actual replacements](../assets/tutorials/causal-patching.png)

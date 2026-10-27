<p align="center">
  <img src="https://raw.githubusercontent.com/ubc-tea/vlm-probe/main/docs/assets/vlm-probing-icon.svg" width="112" alt="VLMProbe icon">
</p>

<h1 align="center">VLMProbe</h1>

<p align="center"><strong>Exploring Model Internals Beyond the Answer</strong></p>

<p align="center">
  <img src="https://raw.githubusercontent.com/ubc-tea/vlm-probe/main/docs/assets/library-overview-sae.svg" width="100%" alt="Model inputs, adapter configuration, lens/attention/causal/SAE tools, and structured results">
</p>

<p align="center">
  <a href="https://ubc-tea.github.io/vlm-probe/"><img src="https://img.shields.io/badge/Project%20Page-live-3266e5?style=flat-square" alt="VLMProbe project page"></a>
  <a href="https://huggingface.co/spaces/gexinh/vlm-probe"><img src="https://img.shields.io/badge/Hugging%20Face-Space-ffd21e?style=flat-square&amp;logo=huggingface&amp;logoColor=white" alt="VLMProbe on Hugging Face"></a>
  <a href="https://arxiv.org/abs/2610.33072"><img src="https://img.shields.io/badge/arXiv-2610.33072-b31b1b?style=flat-square&amp;logo=arxiv" alt="arXiv paper 2610.33072"></a>
  <a href="https://pypi.org/project/vlmprobe/"><img src="https://img.shields.io/pypi/v/vlmprobe?style=flat-square&amp;logo=pypi&amp;logoColor=white" alt="Published PyPI version"></a>
  <a href="pyproject.toml"><img src="https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-3776ab?style=flat-square&amp;logo=python&amp;logoColor=white" alt="Python 3.10, 3.11 and 3.12"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-PolyForm%20Noncommercial%201.0.0-22863a?style=flat-square" alt="PolyForm Noncommercial License 1.0.0"></a>
</p>

<p align="center">
  <a href="docs/README.md">Documentation</a> ·
  <a href="#notebook-tutorials">Notebook Tutorials</a> ·
  <a href="docs/VISUALIZATION.md">Visualization</a> ·
  <a href="https://arxiv.org/abs/2610.33072">Paper</a> ·
  <a href="docs/assets/vlmprobe-technical-report.pdf">Technical report</a> ·
  <a href="#citation">Citation</a> ·
  <a href="docs/REFERENCES.md">Papers and datasets</a>
</p>

VLMProbe brings lens readouts, attention explanations, causal interventions,
and sparse autoencoders into one model-bound interface. It supports vision-language
models, language models, and a ViT image-classification control. Bind a loaded
model once, select layers and token groups, and inspect how its internal
computation relates to a chosen answer.

- **Lens:** read vocabulary predictions from residual states, attention heads,
  or projected visual embeddings.
- **Attention:** compare attention patterns and target-conditioned spatial
  attributions on the same input.
- **Causal:** replace activations, isolate paths, block attention routes, or steer
  residual states and measure the resulting change.
- **SAE:** train or load sparse dictionaries, inspect feature activity, and
  measure feature interventions with reconstruction controls.

Explicit adapters handle hook locations, expanded visual-token coordinates,
and model readouts. Results retain tensors and experiment metadata; shared
visualization functions turn them into heatmaps, image overlays, curves, and
connection graphs. Capability checks distinguish an unsupported operation from
a measured zero effect.

<p align="center">
  <a href="https://drive.google.com/file/d/1vELcJ0T67VzG6_x4r79uQmDQRQUELXP2/view?usp=sharing"><img src="https://raw.githubusercontent.com/ubc-tea/vlm-probe/main/docs/assets/vlmprobe-demo.gif" width="960" alt="VLMProbe video demo — animated preview; click to watch the full video with narration"></a>
</p>
<p align="center"><a href="https://drive.google.com/file/d/1vELcJ0T67VzG6_x4r79uQmDQRQUELXP2/view?usp=sharing">Watch the full video with narration ↗</a></p>

## Installation

Python 3.10+ is required. Install the current source version from GitHub:

```bash
python -m pip install "vlmprobe[sae,transformers,visualization] @ git+https://github.com/ubc-tea/vlm-probe.git@main"
```

The published package is also available on [PyPI](https://pypi.org/project/vlmprobe/):

```bash
python -m pip install --upgrade "vlmprobe[sae,transformers,visualization]"
```

For tensor methods and custom PyTorch adapters, install the base package:

```bash
python -m pip install vlmprobe
```

The Python import name is **`vlmprobe`**. Existing `vlm_probing` imports remain
compatible, including adapter registration and saved Python objects.

```python
import vlmprobe

probe = vlmprobe.Prober(model, processor)  # use your loaded model
result = probe.lens.logit(tokens="last_prompt").run(inputs)
```

The base package requires only PyTorch. The `transformers` extra targets
Transformers 5.3.x and includes Accelerate, Torchvision, and Pillow;
`visualization` provides the plotting dependencies. Add `notebooks` when using
Jupyter. Model weights and complete datasets are downloaded separately.

To run all notebook tutorials from a checkout, use
`python -m pip install -e ".[sae,evaluation,transformers,visualization,notebooks]"`.
The [package guide](docs/PACKAGING.md) covers wheels and release publication.

## Quick Start

This example loads **Qwen3-VL-2B-Instruct** on one CUDA GPU. Replace
`example.jpg` with your image, or use an image from [the bundled demos](demos/README.md).

```python
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor
from vlmprobe import Prober

model_id = "Qwen/Qwen3-VL-2B-Instruct"
model = AutoModelForImageTextToText.from_pretrained(
    model_id, dtype=torch.bfloat16, device_map={"": "cuda:0"},
    attn_implementation="eager",
)
processor = AutoProcessor.from_pretrained(model_id)
probe = Prober(model, processor)

messages = [{"role": "user", "content": [
    {"type": "image"},
    {"type": "text", "text": "What is in this image?"},
]}]
prompt = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True,
)
inputs = probe.prepare(
    prompt=prompt, image=Image.open("example.jpg").convert("RGB"), device="cuda:0",
)

print(probe.describe()["methods"]["lens.logit"])
result = probe.lens.logit(layers=[0, 13, 27], tokens="last_prompt").run(inputs)
token_id = result.tensors["logits"][-1, 0].argmax().item()
print(processor.tokenizer.decode([token_id]))
result.save("outputs/logit_lens")

# Read projected image tokens in the input embedding space.
visual = probe.lens.embed(tokens="visual").run(inputs, top_k=5)

# Observe the attention actually used by the language decoder.
attention = probe.attention.profile(
    layers=[27], queries="last_prompt", include_attention=True,
).run(inputs)
```

Use a model you have already loaded, and pass its native processor dictionary
directly to `.run(...)` when convenient. Layer indices are zero-based; tokens
refer to the expanded sequence, including image positions. The
[model guides](docs/models/README.md) explain each architecture's input contract.

```python
from vlmprobe import visualization as viz

answer_id = processor.tokenizer.encode(" yellow", add_special_tokens=False)[0]
ax = viz.plot_lens_heatmap(result, target_token_id=answer_id)
ax.figure.savefig("outputs/lens.png", bbox_inches="tight")
```

This plots one vocabulary token's rank across selected layers and positions.
A complete answer can contain several tokens; choose the intended subword
explicitly, or use complete-answer metrics for interventions.

| Common operation | Purpose |
| --- | --- |
| `Prober(model, processor=None)` | Bind a caller-owned model and resolve its adapter. |
| `probe.describe()` | Inspect available methods, sites, and missing capabilities. |
| `probe.prepare(prompt=..., image=..., device=...)` | Convert formatted text and images into native inputs. |
| `probe.lens / probe.attention / probe.causal / probe.sae` | Configure a model-bound analysis method. |
| `method.run(inputs, ...)` | Execute a probe and return `ProbeResult`. |
| `method.fit_batches(...)` / `method.load(...)` | Calibrate or restore Tuned, Attention and MJ-Lens; Jacobian Lens has a separate corpus estimator. |
| `result.save(directory)` | Save CPU tensors and JSON metadata. |

[The API guide](docs/API.md) covers selectors, complete-answer metrics,
paired-input alignment, and result shapes. [Visualization](docs/VISUALIZATION.md)
describes reusable plotting functions. The [demos](demos/README.md) walk through
input preparation, live model calls, calibration when needed, and plotting.
Every [method guide](docs/methods/README.md) includes parameter defaults and a complete input-to-output example.
A [small CPU example](examples/prober_quickstart.py)
exercises the API without downloading pretrained weights.

## Model Support

Nine VLM families have explicit adapters. Text-only GPT-2/Llama/Qwen and a ViT
classifier provide complementary controls. Availability is method-specific;
see the [capability matrix](docs/models/README.md).

| Family | Example checkpoint | Guide |
| --- | --- | --- |
| <a href="docs/models/qwen3_vl.md"><img src="docs/assets/model-buttons/qwen3-vl.svg" alt="Qwen3-VL" width="172" height="32"></a> | Qwen3-VL-2B-Instruct | [Usage](docs/models/qwen3_vl.md) |
| <a href="docs/models/qwen2_5_vl.md"><img src="docs/assets/model-buttons/qwen2-5-vl.svg" alt="Qwen2.5-VL" width="172" height="32"></a> | Qwen2.5-VL-3B-Instruct | [Usage](docs/models/qwen2_5_vl.md) |
| <a href="docs/models/qwen2_vl.md"><img src="docs/assets/model-buttons/qwen2-vl.svg" alt="Qwen2-VL" width="172" height="32"></a> | Qwen2-VL-2B-Instruct | [Usage](docs/models/qwen2_vl.md) |
| <a href="docs/models/qwen3_5.md"><img src="docs/assets/model-buttons/qwen3-5.svg" alt="Qwen3.5" width="172" height="32"></a> | Qwen3.5-4B; hybrid attention restrictions | [Usage](docs/models/qwen3_5.md) |
| <a href="docs/models/smolvlm.md"><img src="docs/assets/model-buttons/smolvlm.svg" alt="SmolVLM" width="172" height="32"></a> | SmolVLM-500M-Instruct | [Usage](docs/models/smolvlm.md) |
| <a href="docs/models/internvl.md"><img src="docs/assets/model-buttons/internvl.svg" alt="InternVL" width="172" height="32"></a> | InternVL3-1B-hf | [Usage](docs/models/internvl.md) |
| <a href="docs/models/llava.md"><img src="docs/assets/model-buttons/llava.svg" alt="LLaVA" width="172" height="32"></a> | llava-1.5-7b-hf | [Usage](docs/models/llava.md) |
| <a href="docs/models/pixtral.md"><img src="docs/assets/model-buttons/pixtral.svg" alt="Pixtral" width="172" height="32"></a> | Pixtral-12B-2409 | [Usage](docs/models/pixtral.md) |
| <a href="docs/models/janus.md"><img src="docs/assets/model-buttons/janus.svg" alt="Janus" width="172" height="32"></a> | Janus-Pro-1B / 7B, native Transformers format | [Usage](docs/models/janus.md) |
| <a href="docs/models/README.md"><img src="docs/assets/model-buttons/language-models.svg" alt="Language models" width="172" height="32"></a> | GPT-2 / Llama / Qwen2 / Qwen3 | [GPT-2](docs/models/gpt2.md) · [Llama](docs/models/llama.md) · [Qwen](docs/models/qwen_text.md) |
| <a href="docs/models/vit.md"><img src="docs/assets/model-buttons/vision-classifier.svg" alt="Vision classifier" width="172" height="32"></a> | ViT-B/16 | [Usage](docs/models/vit.md) |

## Methods

Each guide explains the implemented computation, parameters, outputs, and
relationship to its original paper. Reusable primitives are labeled separately
from paper methods.

### Lens

<p>
  <a href="docs/methods/logit_lens.md"><img src="docs/assets/method-buttons/logit-lens.svg" alt="Logit Lens" width="200" height="44"></a>
  <a href="docs/methods/tuned_lens.md"><img src="docs/assets/method-buttons/tuned-lens.svg" alt="Tuned Lens" width="200" height="44"></a>
  <a href="docs/methods/attention_lens.md"><img src="docs/assets/method-buttons/attention-lens.svg" alt="Attention Lens" width="200" height="44"></a>
  <a href="docs/methods/jacobian_lens.md"><img src="docs/assets/method-buttons/jacobian-lens.svg" alt="Jacobian Lens" width="200" height="44"></a>
  <a href="docs/methods/mj_lens.md"><img src="docs/assets/method-buttons/mj-lens.svg" alt="MJ-Lens" width="200" height="44"></a>
  <a href="docs/methods/embed_lens.md"><img src="docs/assets/method-buttons/embed-lens.svg" alt="EmbedLens" width="200" height="44"></a>
  <a href="docs/methods/patchscope.md"><img src="docs/assets/method-buttons/patchscopes.svg" alt="Patchscopes" width="200" height="44"></a>
</p>

### Attention

<p>
  <a href="docs/methods/attention_rollout.md"><img src="docs/assets/method-buttons/rollout.svg" alt="Rollout" width="200" height="44"></a>
  <a href="docs/methods/attention_maps.md"><img src="docs/assets/method-buttons/grad-cam.svg" alt="Grad-CAM" width="200" height="44"></a>
  <a href="docs/methods/attention_maps.md"><img src="docs/assets/method-buttons/attattr.svg" alt="ATTATTR" width="200" height="44"></a>
  <a href="docs/methods/attention_maps.md"><img src="docs/assets/method-buttons/tam.svg" alt="TAM" width="200" height="44"></a>
  <a href="docs/methods/attention_maps.md"><img src="docs/assets/method-buttons/beyond-intuition.svg" alt="Beyond Intuition" width="200" height="44"></a>
  <a href="docs/methods/attention_relevance.md"><img src="docs/assets/method-buttons/generic-relevance.svg" alt="Generic relevance" width="200" height="44"></a>
  <a href="docs/methods/chefer_lrp.md"><img src="docs/assets/method-buttons/vit-dtd-lrp.svg" alt="ViT DTD / LRP" width="200" height="44"></a>
</p>

### Causal

<p>
  <a href="docs/methods/activation_patching.md"><img src="docs/assets/method-buttons/activation-patching.svg" alt="Activation patching" width="200" height="44"></a>
  <a href="docs/methods/path_patching.md"><img src="docs/assets/method-buttons/path-patching.svg" alt="IOI-style path patching" width="200" height="44"></a>
  <a href="docs/methods/attribution_patching.md"><img src="docs/assets/method-buttons/attribution-patching.svg" alt="Attribution patching" width="200" height="44"></a>
  <a href="docs/methods/attention_knockout.md"><img src="docs/assets/method-buttons/attention-knockout.svg" alt="Attention Knockout" width="200" height="44"></a>
  <a href="docs/methods/eap_ig.md"><img src="docs/assets/method-buttons/eap-ig.svg" alt="EAP / EAP-IG" width="200" height="44"></a>
  <a href="docs/methods/steering.md"><img src="docs/assets/method-buttons/steering-vsv.svg" alt="Residual steering / VSV" width="200" height="44"></a>
</p>

### SAE

<p>
  <a href="docs/methods/sae/relu.md"><img src="docs/assets/method-buttons/relu.svg" alt="ReLU" width="200" height="44"></a>
  <a href="docs/methods/sae/topk.md"><img src="docs/assets/method-buttons/topk.svg" alt="TopK" width="200" height="44"></a>
  <a href="docs/methods/sae/gated.md"><img src="docs/assets/method-buttons/gated.svg" alt="Gated" width="200" height="44"></a>
  <a href="docs/methods/sae/jumprelu.md"><img src="docs/assets/method-buttons/jumprelu.svg" alt="JumpReLU" width="200" height="44"></a>
  <a href="docs/methods/sae/abstopk.md"><img src="docs/assets/method-buttons/abstopk.svg" alt="AbsTopK" width="200" height="44"></a>
  <a href="docs/methods/sae/batchtopk.md"><img src="docs/assets/method-buttons/batchtopk.svg" alt="BatchTopK" width="200" height="44"></a>
  <a href="docs/methods/sae/matryoshka_batchtopk.md"><img src="docs/assets/method-buttons/matryoshka-batchtopk.svg" alt="Matryoshka BatchTopK" width="200" height="44"></a>
  <a href="docs/methods/sae/archetypal_topk.md"><img src="docs/assets/method-buttons/archetypal-topk.svg" alt="Archetypal TopK" width="200" height="44"></a>
  <a href="docs/methods/sae/archetypal_jumprelu.md"><img src="docs/assets/method-buttons/archetypal-jumprelu.svg" alt="Archetypal JumpReLU" width="200" height="44"></a>
  <a href="docs/methods/sae/matching_pursuit.md"><img src="docs/assets/method-buttons/matching-pursuit.svg" alt="Matching Pursuit" width="200" height="44"></a>
  <a href="docs/methods/sae/signed_matching_pursuit.md"><img src="docs/assets/method-buttons/signed-mp.svg" alt="Signed MP" width="200" height="44"></a>
  <a href="docs/methods/sae/orthogonal_matching_pursuit.md"><img src="docs/assets/method-buttons/nonnegative-omp.svg" alt="Nonnegative OMP" width="200" height="44"></a>
  <a href="docs/methods/sae/signed_orthogonal_matching_pursuit.md"><img src="docs/assets/method-buttons/signed-omp.svg" alt="Signed OMP" width="200" height="44"></a>
  <a href="docs/methods/sae/qsae.md"><img src="docs/assets/method-buttons/qsae.svg" alt="QSAE" width="200" height="44"></a>
</p>

### Supporting tools

<p>
  <a href="docs/methods/attention_profile.md"><img src="docs/assets/method-buttons/attention-profile.svg" alt="Attention profile" width="200" height="44"></a>
  <a href="docs/methods/head_logit_attribution.md"><img src="docs/assets/method-buttons/head-logit-attribution.svg" alt="Head logit attribution" width="200" height="44"></a>
  <a href="docs/methods/attention_reweight.md"><img src="docs/assets/method-buttons/probability-reweighting.svg" alt="Probability reweighting" width="200" height="44"></a>
  <a href="docs/methods/attention_temperature.md"><img src="docs/assets/method-buttons/temperature.svg" alt="Temperature" width="200" height="44"></a>
</p>

## Notebook Tutorials

Start with a complete input-to-result walkthrough. Each notebook shows native
preprocessing, method calls, output inspection and visualization; calibration
steps are included when the method needs them.

| Feature | Notebook | Compute |
| --- | --- | --- |
| Text lenses: Logit, Tuned, Attention and Jacobian | [Open notebook](demos/lens_comparison_demo.ipynb) · [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gexinh/vlm-probe/blob/c50b3ed3d289d298f6ffbf1852ea34cd83ce25f6/demos/lens_comparison_demo.ipynb) | CPU or GPU; costly optional fits are labeled. |
| VLM lenses: image inputs, readouts and calibration | [Open notebook](demos/vlm_lens_demo.ipynb) · [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gexinh/vlm-probe/blob/c50b3ed3d289d298f6ffbf1852ea34cd83ce25f6/demos/vlm_lens_demo.ipynb) | GPU recommended. |
| MJ-Lens: bbox-region vocabulary across decoder layers | [Open notebook](demos/10_mj_lens.ipynb) · [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gexinh/vlm-probe/blob/c50b3ed3d289d298f6ffbf1852ea34cd83ce25f6/demos/10_mj_lens.ipynb) | 24 GB+ CUDA recommended; independent image/caption calibration. |
| ViT attention and attribution maps | [Open notebook](demos/attention_comparison_demo.ipynb) · [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gexinh/vlm-probe/blob/c50b3ed3d289d298f6ffbf1852ea34cd83ce25f6/demos/attention_comparison_demo.ipynb) | CPU or GPU. |
| VLM attention: shared image and answer target | [Open notebook](demos/vlm_attention_comparison_demo.ipynb) · [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gexinh/vlm-probe/blob/c50b3ed3d289d298f6ffbf1852ea34cd83ce25f6/demos/vlm_attention_comparison_demo.ipynb) | GPU recommended. |
| Activation and attribution patching | [Open notebook](demos/activation_patching_demo.ipynb) · [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gexinh/vlm-probe/blob/c50b3ed3d289d298f6ffbf1852ea34cd83ce25f6/demos/activation_patching_demo.ipynb) | CPU or GPU. |
| Attention Knockout | [Open notebook](demos/attention_knockout_demo.ipynb) · [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gexinh/vlm-probe/blob/c50b3ed3d289d298f6ffbf1852ea34cd83ce25f6/demos/attention_knockout_demo.ipynb) | GPU recommended. |
| Path patching | [Open notebook](demos/path_patching_demo.ipynb) · [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gexinh/vlm-probe/blob/c50b3ed3d289d298f6ffbf1852ea34cd83ce25f6/demos/path_patching_demo.ipynb) | CPU or GPU. |
| Residual steering / VSV | [Open notebook](demos/steering_vsv_demo.ipynb) · [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gexinh/vlm-probe/blob/c50b3ed3d289d298f6ffbf1852ea34cd83ce25f6/demos/steering_vsv_demo.ipynb) | GPU recommended. |
| EAP / EAP-IG | [Open notebook](demos/eap_ig_demo.ipynb) · [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gexinh/vlm-probe/blob/c50b3ed3d289d298f6ffbf1852ea34cd83ce25f6/demos/eap_ig_demo.ipynb) | CPU or GPU. |
| Pretrained visual SAE: features, reconstruction and evaluation | [Open notebook](demos/11_sae_pretrained.ipynb) · [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gexinh/vlm-probe/blob/c50b3ed3d289d298f6ffbf1852ea34cd83ce25f6/demos/11_sae_pretrained.ipynb) | 8 GB+ GPU recommended; no SAE training. |

The [tutorial index](demos/README.md) lists checkpoints, input provenance and
execution requirements. Colab buttons open a verified notebook revision with
a pinned runtime source. Select a runtime with enough memory for the stated
model and calibration workload.

## Package Architecture

```text
src/vlmprobe/    # public import name; shares the implementation below
src/vlm_probing/
  prober.py       # shared model-bound entry point
  api/           # method factories, capture and replay
  adapters/      # explicit architecture and token-layout contracts
  lenses/        # BaseLens and concrete readout methods
  attention/     # BaseAttention and concrete attention methods
  causal/        # BaseCausal and concrete interventions/estimators
  sae/           # native dictionaries, one trainer, artifacts and model cards
  training/      # calibration, checkpoint import and resume
  visualization/ # shared heatmaps, overlays, curves and graphs
  core/          # results and execution traces
demos/           # runnable notebooks, compact measurements and input images
docs/            # model, method, API and source guides
```

Family base classes keep algorithm implementations small; adapters let them
share model execution. [Architecture](docs/ARCHITECTURE.md) and
[adapter contracts](docs/ADAPTERS.md) describe these boundaries.

## References

The [paper and dataset index](docs/REFERENCES.md) links the original method
papers, author implementations, and datasets used for demonstrations and lens
calibration. The
[technical report](docs/assets/vlmprobe-technical-report.pdf)
compares real readouts, explanations, and interventions on shared cases across
five pretrained VLMs, with language and vision controls.

## Credits

SAE integration draws on [SAELens](https://github.com/decoderesearch/SAELens),
[ViT-Prisma](https://github.com/Prisma-Multimodal/ViT-Prisma) and
[Overcomplete](https://github.com/KempnerInstitute/overcomplete).
Our native algorithms and checkpoint converters retain source attribution and
MIT license notices; see [SAE provenance](docs/methods/sae/SOURCES.md).

## License

Original VLMProbe code is licensed under the [PolyForm Noncommercial License
1.0.0](LICENSE). Third-party components retain their original terms; see
[licensing notices](NOTICE.md).

## Citation

If you use VLMProbe in your research, please cite the [paper](https://arxiv.org/abs/2610.33072):

```bibtex
@misc{huang2026vlmprobeopensourcelibraryvlm,
  title={VLMProbe: An Open-source Library for VLM Probing and Interpretation},
  author={Gexin Huang and Haisheng Fu and Yanting Yang and Chen Zhao and Ruinan Jin and Jiaqi Liu and Xiaoxiao Li},
  year={2026},
  eprint={2610.33072},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2610.33072},
}
```

<!-- vlmprobe:prerelease-link:start -->
Pre-release version: [![Beta](https://img.shields.io/badge/Beta-gexinh%2Fvlm--probe-6f42c1?style=flat-square&logo=github)](https://github.com/gexinh/vlm-probe)
<!-- vlmprobe:prerelease-link:end -->

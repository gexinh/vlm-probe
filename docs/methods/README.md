# Method Guides

[Home](../../README.md) / [Documentation](../README.md)

Each guide presents the API and parameter defaults first, followed by a complete
example: imports, model loading, input/prompt preparation, calibration or target
construction when needed, method execution, result inspection and visualization.
Examples can be copied without first running a hidden shared setup cell.
Availability depends on the model adapter;
check `probe.describe()` or the [model matrix](../models/README.md).

Residual and head lenses support language-model and VLM representations;
EmbedLens uses the input-embedding space; its visual-analysis mode needs projected
visual tokens, while `tokens="all"` also permits text-only inputs. Spatial attention maps
can analyze vision classifiers or decoder visual tokens when the adapter
exposes the required sites. Original EAP-IG currently uses a text-only GPT-2
graph. These modality boundaries are documented on each method page.

## Lens

| Method | Public factory | What it does |
| --- | --- | --- |
| [Logit Lens](logit_lens.md) | `probe.lens.logit(...)` | Decode intermediate residuals with the model's final normalization and vocabulary head. |
| [Tuned Lens](tuned_lens.md) | `probe.lens.tuned(...)` | Fit one affine translator per residual layer before applying the model's frozen readout. |
| [Attention Lens](attention_lens.md) | `probe.lens.attention(...)` | Train separate vocabulary decoders for individual attention heads. |
| [Jacobian Lens](jacobian_lens.md) | `probe.lens.jacobian(...)` | Estimate a layer-to-final-residual Jacobian, transport representations, and decode them. |
| [MJ-Lens](mj_lens.md) | `probe.lens.mj(...)` | Separate source modalities and output channels, with optional discrete visual-codebook readout and controlled modality index. |
| [EmbedLens](embed_lens.md) | `probe.lens.embed(...)` | Find vocabulary embeddings nearest to projected visual tokens. |
| [Patchscope](patchscope.md) | `probe.lens.patchscope(...)` | Insert a source representation into a target prompt and inspect its output logits. |

## Attention

| Method | Public factory | What it does |
| --- | --- | --- |
| [Attention Profile](attention_profile.md) | `probe.attention.profile(...)` | Measure attention entropy, concentration, and mass assigned to token groups. |
| [Head Logit Attribution](head_logit_attribution.md) | `probe.attention.head_logits(...)` | Project head contributions to vocabulary logits with a fixed final normalization scale. |
| [Attention Rollout](attention_rollout.md) | `probe.attention.rollout(...)` | Compose normalized self-attention transitions across consecutive layers. |
| [Attention Relevance](attention_relevance.md) | `probe.attention.relevance(...)` | Propagate positive gradient-weighted self-attention for a selected output score. |
| [Attention Reweighting](attention_reweight.md) | `probe.attention.reweight(...)` | Rescale attention on selected key tokens and measure the output change. |
| [Attention Temperature](attention_temperature.md) | `probe.attention.temperature(...)` | Change the sharpness of attention at selected query rows. |
| [Attention Grad-CAM](attention_maps.md) | `probe.attention.grad_cam(...)` | Target-conditioned CAM using attention heads as channels. |
| [Self-Attention Attribution](attention_maps.md) | `probe.attention.attribution(...)` | Integrate gradients along a single layer's zero-to-original probability path. |
| [TAM](attention_maps.md) | `probe.attention.tam(...)` | Combine backward Markov transitions with input-path final-attention feedback. |
| [Beyond Intuition](attention_maps.md) | `probe.attention.beyond_intuition(...)` | Headwise or tokenwise perception with integrated reasoning feedback. |
| [Chefer DTD/LRP](chefer_lrp.md) | `probe.attention.dtd_lrp(...)` | Copy native ViT weights into the pinned author's actual relevance-propagation backend. |

Profile, head logit attribution, reweighting, and temperature are reusable
statistics/readout/intervention tools. They do not reproduce VAR, OPERA, or PAI.
The [source index](../REFERENCES.md) identifies the paper-backed attribution
algorithms and their implemented scope.

## Causal

| Method | Public factory | What it does |
| --- | --- | --- |
| [Activation Patching](activation_patching.md) | `probe.causal.patch(...)` | Replace selected receiver residuals with source residuals and measure the causal effect. |
| [Path Patching](path_patching.md) | `probe.causal.path(...)` | Freeze other head outputs, transfer sender outputs, then replay only recomputed receiver Q/K/V or final residual. |
| [Attribution Patching](attribution_patching.md) | `probe.causal.attribute(...)` | Approximate a patch effect using activation differences and a receiver-side gradient. |
| [Attention Knockout](attention_knockout.md) | `probe.causal.knockout(...)` | Block selected query-to-key attention paths before softmax. |
| [EAP / EAP-IG](eap_ig.md) | `probe.causal.eap_ig(graph="transformer", ...)` | Attribute independently replaceable GPT-2 edges along an input-embedding path, then evaluate retained circuits with actual forwards. |
| [Residual Steering](steering.md) | `probe.causal.steer(...)` | Add a supplied direction to selected residuals and measure the output change. |
| [Visual Steering Vector (VSV)](steering.md#training-free-per-image-vsv-constructor) | `probe.causal.vsv(positive, negative, ...)` | Build per-image residual directions from image and text-only inputs; configure a joint steering run. |

## Sparse Autoencoders (SAE)

Learn sparse feature dictionaries from frozen-model activations, inspect their
activity, and measure feature interventions with reconstruction controls.

| Method | Public factory | What it does |
| --- | --- | --- |
| [ReLU](sae/relu.md) | `probe.sae.relu(...)` | Nonnegative features with L1 sparsity. |
| [TopK](sae/topk.md) | `probe.sae.topk(...)` | Per-vector positive TopK selection. |
| [Gated](sae/gated.md) | `probe.sae.gated(...)` | Separate feature selection from activation magnitude. |
| [JumpReLU](sae/jumprelu.md) | `probe.sae.jumprelu(...)` | Learn feature thresholds with a surrogate-gradient sparsity objective. |
| [AbsTopK](sae/abstopk.md) | `probe.sae.abstopk(...)` | Keep the largest absolute activations with their signs. |
| [BatchTopK](sae/batchtopk.md) | `probe.sae.batchtopk(...)` | Allocate a sparsity budget across a training minibatch. |
| [Matryoshka BatchTopK](sae/matryoshka_batchtopk.md) | `probe.sae.matryoshka_batchtopk(...)` | Train nested dictionary prefixes with shared sparse codes. |
| [Archetypal TopK](sae/archetypal_topk.md) | `probe.sae.archetypal_topk(...)` | Constrain TopK dictionary atoms with training-data anchors. |
| [Archetypal JumpReLU](sae/archetypal_jumprelu.md) | `probe.sae.archetypal_jumprelu(...)` | Combine learned thresholds and an archetypal dictionary. |
| [Matching Pursuit](sae/matching_pursuit.md) | `probe.sae.matching_pursuit(...)` | Greedily encode residuals with nonnegative coefficients. |
| [Signed MP](sae/signed_matching_pursuit.md) | `probe.sae.signed_matching_pursuit(...)` | Greedily encode residuals with signed coefficients. |
| [Nonnegative OMP](sae/orthogonal_matching_pursuit.md) | `probe.sae.orthogonal_matching_pursuit(...)` | Refit nonnegative coefficients jointly on selected support. |
| [Signed OMP](sae/signed_orthogonal_matching_pursuit.md) | `probe.sae.signed_orthogonal_matching_pursuit(...)` | Refit signed coefficients by least squares. |
| [QSAE](sae/qsae.md) | `probe.sae.qsae(...)` | Use learned scalar quantization levels; an experimental recipe. |

[Model cards](sae/models/README.md) · [Training](sae/training/README.md) · [Shared API](sae/API.md)

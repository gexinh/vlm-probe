# Papers and Datasets

[Home](../README.md) / [Documentation](README.md)

This index separates method sources, reusable primitives, and the datasets used
by demonstrations or lens calibration. A paper reference identifies the
computation being implemented; the scope column records what this library
actually exposes. One- or two-case demos do not reproduce dataset-level claims.

## Lens methods

| Method | Original work and author implementation | Implemented scope |
| --- | --- | --- |
| [Logit Lens](methods/logit_lens.md) | nostalgebraist (2020), [Interpreting GPT: the logit lens](https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens) | Native final normalization and vocabulary readout on intermediate residuals. |
| [Tuned Lens](methods/tuned_lens.md) | Belrose et al. (2023), [Eliciting Latent Predictions from Transformers with the Tuned Lens](https://arxiv.org/abs/2303.08112). [Code](https://github.com/AlignmentResearch/tuned-lens) | Per-layer affine translation fitted to the frozen model's output distribution; validated GPT-2 author-checkpoint import. |
| [Attention Lens](methods/attention_lens.md) | Sakarvadia et al., BlackboxNLP 2023, [Attention Lens: A Tool for Mechanistically Interpreting the Attention Head Information Retrieval Mechanism](https://arxiv.org/abs/2310.16270). [Code](https://github.com/msakarvadia/AttentionLens) | Learned vocabulary decoders on projected head contributions, trained jointly through their summed logits following the released code. |
| [Jacobian Lens](methods/jacobian_lens.md) | Gurnee et al., Transformer Circuits 2026, [Verbalizable Representations Form a Global Workspace in Language Models](https://transformer-circuits.pub/2026/workspace/index.html). [Code](https://github.com/anthropics/jacobian-lens) | Mean downstream residual Jacobian and native readout. Calibration is implemented; the paper's full J-space experiments are outside this release. |
| [MJ-Lens](methods/mj_lens.md) | Anonymous ICLR 2027 submission, [Does the Workspace See? Testing a Multimodal Global Workspace in Vision-Language Models](https://openreview.net/forum?id=Xrp8mNku6R), Eq. 2–5 and Appendices C, D, H.3. | Independent implementation: causal modality-pair expectation, frozen-model calibration, Janus image-head linearization and controlled dictionary decomposition. The decomposition solver is a documented library choice; paper-scale causal/generation experiments are not reproduced. |
| [EmbedLens](methods/embed_lens.md) | Fan et al., CVPR 2026, [What Do Visual Tokens Really Encode? Uncovering Sparsity and Redundancy in Multimodal Large Language Models](https://arxiv.org/abs/2603.00510). [Code](https://github.com/EIT-NLP/EmbedLens) | Cosine neighbors of projected visual tokens in the language input-embedding table. No automatic sink/dead/alive classification or pruning benchmark. |
| [Patchscopes](methods/patchscope.md) | Ghandeharioun et al., ICML 2024, [Patchscopes: A Unifying Framework for Inspecting Hidden Representations of Language Models](https://proceedings.mlr.press/v235/ghandeharioun24a.html). [Code](https://github.com/PAIR-code/interpretability/tree/master/patchscopes) | Residual transfer into a configured target prompt, optional mapping, and teacher-forced target logits. |

## Attention methods

| Method | Original work and author implementation | Implemented scope |
| --- | --- | --- |
| [Rollout](methods/attention_rollout.md) | Abnar and Zuidema, ACL 2020, [Quantifying Attention Flow in Transformers](https://aclanthology.org/2020.acl-main.385/). [Code](https://github.com/samiraabnar/attention_flow) | Residual-aware row-normalized self-attention transitions. Attention flow/max-flow is not implemented. |
| [Attention Grad-CAM](methods/attention_maps.md) | Selvaraju et al., ICCV 2017, [Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization](https://arxiv.org/abs/1610.02391). [Author ViT attention adaptation](https://github.com/hila-chefer/Transformer-Explainability) | Attention heads serve as channels. This is an attention-map adaptation, not the original CNN feature-map algorithm. |
| [ATTATTR](methods/attention_maps.md) | Hao et al., AAAI 2021, [Self-Attention Attribution: Interpreting Information Interactions Inside Transformer](https://arxiv.org/abs/2004.11207). [Code](https://github.com/YRdddream/attattr) | Independent per-layer probability-space integrated gradients, keeping the model input fixed. |
| [TAM](methods/attention_maps.md) | Yuan et al., NeurIPS 2021 XAI4Debugging workshop, [Explaining Information Flow Inside Vision Transformers Using Markov Chain](https://xai4debugging.github.io/files/papers/explaining_information_flow_in.pdf). [Code](https://github.com/XianrenYty/Transition_Attention_Maps) | Backward residual transitions with input-path final-attention gradient feedback. |
| [Beyond Intuition](methods/attention_maps.md) | Chen et al., TMLR 2023, [Beyond Intuition: Rethinking Token Attributions Inside Transformers](https://openreview.net/forum?id=rm0zIzlhcX). [Code](https://github.com/jiaminchen-1031/transformerinterp) | Headwise and tokenwise perception variants with integrated input-path reasoning feedback. |
| [Generic relevance](methods/attention_relevance.md) | Chefer et al., ICCV 2021, [Generic Attention-model Explainability for Interpreting Bi-Modal and Encoder-Decoder Transformers](https://arxiv.org/abs/2103.15679). [Code](https://github.com/hila-chefer/Transformer-MM-Explainability) | Positive-gradient self-attention recurrence. Full encoder-decoder/cross-modal relevance propagation is not claimed. |
| [DTD/LRP](methods/chefer_lrp.md) | Chefer et al., CVPR 2021, [Transformer Interpretability Beyond Attention Visualization](https://arxiv.org/abs/2012.09838). [Code](https://github.com/hila-chefer/Transformer-Explainability) | Genuine author relevance-propagation backend for eligible ViT classifiers. It is separate from generic relevance and unavailable on native VLM decoders. |

The original ATTATTR study uses text/BERT; TAM uses image classification.
Decoder-side VLM examples are explicitly method transfers. Individual guides
record paper-versus-code integration rules and required hook sites.

## Causal methods

| Method | Original work and author implementation | Implemented scope |
| --- | --- | --- |
| [Activation patching](methods/activation_patching.md) | A general intervention; Meng et al., NeurIPS 2022, [Locating and Editing Factual Associations in GPT](https://arxiv.org/abs/2202.05262) is a causal-tracing application. [ROME code](https://github.com/kmeng01/rome) | Aligned residual replacement and measured output-score changes. No ROME editing algorithm or complete causal-tracing benchmark. |
| [IOI-style path patching](methods/path_patching.md) | Wang et al., ICLR 2023, [Interpretability in the Wild: A Circuit for Indirect Object Identification in GPT-2 Small](https://arxiv.org/abs/2211.00593), Appendix B. [Pinned author code](https://github.com/redwoodresearch/Easy-Transformer/blob/ea15315dd24481e9e2ac5c3ef335d82907a1dc34/easy_transformer/ioi_utils.py#L985-L1133) | Four actual forwards: base/reference caches, controlled head-output freeze, then receiver-only replay. It tests chosen paths; it does not automatically assign functional head names or discover the entire IOI circuit. |
| [Attribution patching](methods/attribution_patching.md) | Nanda (2023), [Attribution Patching: Activation Patching at Industrial Scale](https://www.neelnanda.io/mechanistic-interpretability/attribution-patching) | Receiver-side gradient times source-minus-receiver activation; a local approximation to replacement effects. |
| [EAP / EAP-IG](methods/eap_ig.md) | Hanna et al., COLM 2024, [Have Faith in Faithfulness: Going Beyond Circuit Overlap When Finding Model Mechanisms](https://arxiv.org/abs/2403.17806). [Code](https://github.com/hannamw/EAP-IG) | Original input-embedding interpolation on an audited GPT-2 residual-message graph, with EAP scores and actual circuit recovery. Caller-defined activation-space interpolation is a separately labeled variant. |
| [Attention Knockout](methods/attention_knockout.md) | Zhang et al., CVPR 2025, [Cross-modal Information Flow in Multimodal Large Language Models](https://arxiv.org/abs/2411.18620). [Code and GQA subset](https://github.com/FightingFighting/cross-modal-information-flow-in-MLLM) | Editable pre-softmax masking of selected key-to-query routes, including joint layer windows. |
| [Residual steering / VSV](methods/steering.md) | Li et al., ICML 2025, [The Hidden Life of Tokens: Reducing Hallucination of Large Vision-Language Models via Visual Information Steering](https://proceedings.mlr.press/v267/li25ca.html). [VISTA code](https://github.com/LzVv123456/VISTA) | Generic residual-direction injection plus per-image contrastive directions following the paper's residual formula. SLA is not implemented; demos are VSV-only, not complete VISTA. |

Path patching follows the paper's head-output freeze rule. The released IOI code
instead freezes Q/K/V after caching receiver inputs; the distinction is recorded
in [the method guide](methods/path_patching.md). EAP-IG offers left endpoint
sampling matching the released code and right endpoints matching paper Eq. 3.

## Supporting primitives

[Attention profile](methods/attention_profile.md) provides entropy,
concentration, and token-group mass. Related visual-attention applications
include [VAR](https://arxiv.org/abs/2503.03321) and
[OPERA](https://arxiv.org/abs/2311.17911); neither full decoding algorithm is
implemented here.

[Head logit attribution](methods/head_logit_attribution.md) uses a fixed
normalization scale and vocabulary projection, with background in
[A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html).
It is not a trained Attention Lens.

[Probability reweighting](methods/attention_reweight.md) and
[attention temperature](methods/attention_temperature.md) are generic
interventions with no unique originating paper. Reweighting post-softmax
probabilities is not [PAI](https://arxiv.org/abs/2407.21771), whose attention
component edits pre-softmax scores and whose full algorithm adds contrastive
decoding.

## Datasets and evaluation inputs

| Dataset | Citation and official source | Role in this project |
| --- | --- | --- |
| GQA | Hudson and Manning, CVPR 2019, [GQA: A New Dataset for Real-World Visual Reasoning and Compositional Question Answering](https://arxiv.org/abs/1902.09506). [Images, questions and scene graphs](https://cs.stanford.edu/people/dorarad/gqa/download.html); [Knockout authors' subset](https://github.com/FightingFighting/cross-modal-information-flow-in-MLLM/tree/main/datasets) | Shared attribute-question cases, target bounding boxes, lens/attention/causal demos. |
| COCO 2014 | Lin et al., ECCV 2014, [Microsoft COCO: Common Objects in Context](https://arxiv.org/abs/1405.0312). [Official data](https://cocodataset.org/#download); [VISTA evaluation records](https://github.com/LzVv123456/VISTA/tree/main/pope_coco) | Open-ended VSV examples and separate train/validation image-caption splits for VLM Tuned Lens calibration. |
| MMVP | Tong et al., CVPR 2024, [Eyes Wide Shut? Exploring the Visual Shortcomings of Multimodal LLMs](https://arxiv.org/abs/2401.06209). [Author repository](https://github.com/tsb0601/MMVP); [official dataset](https://huggingface.co/datasets/MMVP/MMVP) | Paired-image visual perception and cross-model controlled-path cases. |
| VSR | Liu et al., TACL 2023, [Visual Spatial Reasoning](https://arxiv.org/abs/2205.00363). [Author repository](https://github.com/cambridgeltl/visual-spatial-reasoning) | Shared spatial-relation cases in the technical report. |
| VisOnlyQA | Kamoi et al., COLM 2025, [VisOnlyQA: Large Vision Language Models Still Struggle with Visual Perception of Geometric Information](https://arxiv.org/abs/2412.00947). [Author repository](https://github.com/ryokamoi/VisOnlyQA) | Geometric-perception cases in the technical report. |
| IOI | Wang et al., ICLR 2023, [Interpretability in the Wild](https://arxiv.org/abs/2211.00593). [EAP-IG authors' GPT-2 pairs](https://github.com/hannamw/eap-ig-faithfulness/tree/main/data/ioi) | Text-only four-lens controls and original EAP/EAP-IG circuit evaluation. IOI is a synthetic task, not a natural-image dataset. |
| BookCorpus | Zhu et al., ICCV 2015, [Aligning Books and Movies: Towards Story-like Visual Explanations by Watching Movies and Reading Books](https://arxiv.org/abs/1506.06724) | Independent GPT-2 Attention Lens calibration contexts. The dataset variant and split belong to the calibration manifest. |
| WikiText-2 | Merity et al., ICLR 2017, [Pointer Sentinel Mixture Models](https://arxiv.org/abs/1609.07843). [Dataset card](https://huggingface.co/datasets/Salesforce/wikitext) | Held-out text validation and independent Jacobian calibration contexts. |
| ViT author example images | [Chefer author samples](https://github.com/hila-chefer/Transformer-Explainability/tree/main/samples) | Cat/dog and elephant/zebra illustration inputs; these are author sample images, not a claimed ImageNet evaluation split. |
| VRUBench examples | Yang et al., ACL 2026, [How Do LLMs and VLMs Understand Viewpoint Rotation Without Vision? An Interpretability Study](https://arxiv.org/abs/2604.15294) | Earlier paired-text path-patching examples. The demo records the published example source rather than claiming a full VRUBench download. |

Only the small input images and compact measured snapshots needed by the demos
are included in the repository. [Demo provenance](../demos/README.md) and
the image notices identify sample IDs and sources. Complete datasets, model
weights, and fitted probe artifacts remain external. The
[training guide](TRAINING.md) records calibration sizes and split requirements.


## Sparse autoencoders

The [SAE method inventory](methods/sae/ALGORITHMS.md) maps each native algorithm and
explicit recipe to its primary paper/source. Sparse dictionary methods include ReLU
and Gated dictionaries, positive/signed TopK, learned JumpReLU thresholds,
batch/nested sparsity budgets, archetypal dictionary constraints, pursuit and
experimental quantized codes. [Source provenance](methods/sae/SOURCES.md) records pinned
SAELens, ViT-Prisma and Overcomplete revisions, preserved licenses, and scope.
[Model-card workflows](methods/sae/models/README.md) preserve training provenance and
require end-to-end validation for published pretrained cards. The CPU tutorial uses procedural images and a
random tiny ViT, not a claimed benchmark dataset.

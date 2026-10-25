# Documentation

[Home](../README.md)

Start with a model guide, prepare its native inputs, and select a method from the
available families. The same result format feeds the plotting functions and saved
demo workflows.

- [Model guides and capability matrix](models/README.md): loading, native inputs, supported sites, and validation scope.
- [Method guides](methods/README.md): API and parameter defaults, complete input-to-output examples, visualizations, and paper mappings.
- [Common API](API.md): token selectors, complete-answer metrics, alignment, fitting, and persistence.
- [Repository synchronization](REPOSITORY_SYNC.md): laboratory and personal mirrors, project pages, report updates, and one PyPI publisher.
- [Installation and package releases](PACKAGING.md): pip installation, compatible imports, wheels, and PyPI publishing.
- [Visualization](VISUALIZATION.md): reusable lens heatmaps, image overlays, causal grids, curves, and graphs.
- [Input geometry](INPUT_GEOMETRY.md): native visual grids, token counts, preprocessing summaries, and coordinate limits.
- [Lens calibration](TRAINING.md): training and validation splits, author checkpoints, persistent optimization, and resumable Jacobian estimation.
- [MJ-Lens](methods/mj_lens.md): pretrained VLM report-case walkthrough, modality-split calibration, native/plain controls and optional Janus visual-codebook readout.
- [SAE platform](methods/sae/README.md): native methods, one training engine, model cards, checkpoint imports and visualization.
- [SAE tutorial](methods/sae/examples/README.md): a complete live image-to-feature workflow, including training and plots.
- [Package architecture](ARCHITECTURE.md): family base classes, adapters, execution, and results.
- [Custom adapters](ADAPTERS.md): connect another architecture through an explicit contract.
- [Papers and datasets](REFERENCES.md): original algorithms, author code, and data sources.
- [Demos](../demos/README.md): live input-to-result notebooks, including calibration where required and built-in visualization.
- [Technical report](assets/vlmprobe-technical-report.pdf): shared-case comparisons across five pretrained VLMs, plus language and vision controls.

## Choose a demo

| Question | Notebook |
| --- | --- |
| How do residual and head readouts differ? | [Text lens comparison](../demos/lens_comparison_demo.ipynb) |
| What do VLM layers and image embeddings decode to? | [VLM lens comparison](../demos/vlm_lens_demo.ipynb) |
| How do modality-split readouts behave on the report's image/questions? | [Pretrained MJ-Lens](../demos/10_mj_lens.ipynb) |
| How do spatial attribution rules differ on one image? | [ViT attention comparison](../demos/attention_comparison_demo.ipynb) · [VLM attention comparison](../demos/vlm_attention_comparison_demo.ipynb) |
| Which residual positions support an answer? | [Activation and attribution patching](../demos/activation_patching_demo.ipynb) |
| Which attention route supports an answer? | [Attention Knockout](../demos/attention_knockout_demo.ipynb) |
| Which controlled head path changes a selected score? | [Path patching](../demos/path_patching_demo.ipynb) |
| Do selected edges recover an IOI answer margin? | [EAP / EAP-IG](../demos/eap_ig_demo.ipynb) |
| How does residual steering change a caption? | [Steering / VSV](../demos/steering_vsv_demo.ipynb) |

Notebook code prepares inputs and runs the library before plotting results.
Each notebook states its default checkpoint, compute requirements, calibration
scope and any optional expensive section. Historical measurement snapshots are
kept as reference data; the tutorials do not use them as computed results.

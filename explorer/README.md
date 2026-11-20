# Probe Explorer

Probe Explorer compares recorded VLMProbe measurements in a browser. Use the
family and experiment selectors, then choose up to four method panels for that
experiment. The bundled examples load without Python packages, model weights,
or a network connection.

## Start locally

From the repository root, run

```bash
python3 -m http.server 8772 --bind 127.0.0.1
```

Open [Probe Explorer](http://127.0.0.1:8772/explorer/). The project-page build
also includes the Explorer at `explorer/`.

## Read the views

| Family | Included experiments |
| --- | --- |
| Attention | Two LLaVA/GQA cases with eight archived methods. Switch among measured layers, toggle image display and patch grids, and inspect the same patch across aligned maps. Independent color scales preserve each method’s measurement units. |
| Lens | Compare Logit and Tuned Lens rank and answer scores on the same case. Inspect EmbedLens lexical similarity and fragments or Patchscope scores in the target prompt. |
| Causal | Compare answer-score curves and token-level matrices for activation patching and its gradient approximation under the same intervention. |
| Sparse autoencoders | Inspect the published pretrained DINO feature overlays, held-out localization, feature deletion and reconstruction controls. A second experiment compares 14 SAE recipe runs on the same small random ViT setup. |

The DINO notebook includes the rendered maps and evaluation measurements.
Patch-level hover uses numerical arrays when the experiment provides them.
Expand **Experiment settings and source** for the protocol and archived source.

## Bring your own result

Save results with `result.save(directory)`. In the Python environment where
VLMProbe is installed, export one or more results from the same model and input.

```bash
python explorer/export_result.py outputs/logit outputs/tuned \
  --title "LLaVA lens comparison" --target-token-id 612 --output lens.json
```

Load the JSON with **Import results**. Imported results stay in the current
browser session. The Explorer reads the JSON and any embedded image locally.

For attention or SAE image overlays, pass the exact processed image and the
native grid. Supply expanded visual positions in row-major order. The query,
when needed, also uses expanded token coordinates. Use `--help` for selectors
such as batch index, head, target token ID, and SAE feature IDs. The exporter
can also be imported from a notebook in the repository via
`explorer.export_result.build_bundle`.

Exported visual grids retain every selected patch. For long output sequences,
dense browser matrices show the final 64 lens positions or the first 128
attention or SAE position columns. The saved Python result retains the full
tensors. Supply the target metric when running an SAE intervention to export
its effect scores.

## Regenerate the examples

```bash
python3 explorer/build_examples.py
```

The generator reads measured outputs already tracked under
`demos/snapshots/`, `demos/`, and `docs/assets/tutorials/`.

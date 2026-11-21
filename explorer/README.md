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
| Attention | Two LLaVA/GQA cases with eight archived methods and an annotated reference image. Compare the share of absolute signal in the target with the fraction of the input grid covered by the annotation. Inspect aligned patches, change measured layers, or open the original boxed image at full size. |
| Lens | Compare complete-answer probability and answer-token ranks at matched layers. Inspect EmbedLens lexical similarity and fragments or Patchscope scores in the target prompt. |
| Causal | See the measured activation patching curve and gradient estimate on the same axis, including a layer where their signs differ. Inspect the underlying token-level effects in the individual method panels. |
| Sparse autoencoders | See held-out DINO feature localization, published feature overlays, and feature deletion against its reconstruction control. Rank 14 SAE recipe runs on the same small random ViT by reconstruction error or sparsity. |

**Signal focus** highlights the highest-magnitude patches covering at least
the selected fraction of each map’s absolute signal. It keeps tied patches
at the cutoff and rings a unique peak. Gold shows positive signal and
blue shows negative signal. The solid mint box shows the original
annotation and the dashed outline shows the target patches counted in
the score. Overlay and heatmap views retain independent color scales in
their measurement units.

Attention leaders are scoped to the selected maps, input, annotated region
and layer. The uniform baseline is the share of native patches assigned
to the region by the archived benchmark. The panel updates when you change
methods or layers.

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

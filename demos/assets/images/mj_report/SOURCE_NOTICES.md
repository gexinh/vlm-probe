# Raw MJ-Lens report inputs

These six files are the unchanged raw inputs used by the technical report. They
contain no MJ-Lens results. Exact byte hashes, source revisions, original sample
identifiers and user questions are recorded in
[mj_report_samples.json](../../mj_report_samples.json). The two GQA photographs
are already present in the parent image directory.

| Included files | Original source |
| --- | --- |
| `mmvp_1.jpg`, `mmvp_2.jpg` | [MMVP](https://huggingface.co/datasets/MMVP/MMVP/tree/37eafecab8a3940c50c2ade5b36de69dbc99a8cf), released evaluation image IDs 1 and 2. Its imagefolder split is named `train`. |
| `vsr_000000017697.jpg`, `vsr_000000199940.jpg` | [VSR zero-shot test](https://huggingface.co/datasets/cambridgeltl/vsr_zeroshot/tree/148b3777ceef4a1bfe46377614980836db7d12f2), rows 0 and 1; source photographs are COCO train2017 images with the same IDs. |
| `visonly_geometry_triangle.jpg`, `visonly_charts_extraction.png` | [VisOnlyQA Eval Real](https://huggingface.co/datasets/ryokamoi/VisOnlyQA_Eval_Real/tree/39edb70649cb903eb0db945113e5bd407b3c4791), row 0 of `geometry__triangle` and `charts__extraction`; upstream image sources are MathVista and ChartQA. |

Images and annotations retain their source rights and notices. The library
license does not relicense these third-party inputs. No image-specific license
is inferred from the dataset name. Source and redistribution details should be
read from the linked datasets and their original image sources.

Only the GQA records supply target boxes. No boxes have been invented for
MMVP, VSR or VisOnlyQA.

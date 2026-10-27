# VLMProbe 1.0.2

## Goal

Release the current reviewed library as 1.0.2 on GitHub and PyPI. This release
includes the development changes accumulated since the published 1.0.1 package.

## Changes since 1.0.1

- Add MJ-Lens with corpus calibration, visual-token vocabulary readouts and a
  complete pretrained VLM notebook using report images and target regions.
- Add reusable SAE evaluation pipelines and a pretrained vision-SAE notebook
  covering activation collection, reconstruction, sparse features and spatial
  visualization without training a new dictionary.
- Include model-adapter, calibration, geometry, attention and checkpoint-loading
  fixes developed alongside the method and notebook audits.
- Improve method guides, model support, notebook navigation and project pages;
  add the paper citation, animated video preview and Hugging Face button.
- Include the PolyForm Noncommercial License 1.0.0 for original project code
  and preserve bundled third-party MIT notices.

The canonical package version changes from the internal development label
`1.1.1.dev0` to `1.0.2`, following the public 1.0.1 release. The version label
does not change numerical implementations or experimental protocols. Earlier
development notes retain their original version labels as historical records.

## Installation

```bash
python -m pip install --upgrade "vlmprobe[sae,evaluation,transformers,visualization]==1.0.2"
```

Both `import vlmprobe` and the compatible `import vlm_probing` expose version
`1.0.2`. Notebooks, documentation, model cards and the report are available in
the GitHub repository; model weights and datasets are downloaded separately.

## Expected results and validation

The source distribution and wheel must report 1.0.2, include the same runtime
source as the release commit, and contain all project and third-party license
notices. Validation covers source regressions, strict distribution metadata
checks, installed-wheel imports and repository synchronization. GitHub Actions
builds and checks both Python 3.10 and 3.12 before PyPI publication.

The release source passed 654 tests and 456 subtests, with three optional tests
skipped. The local wheel and source archive passed strict Twine checks; all
115 packaged source/resource files and seven license/notice files matched the
checkout. Five independent installed-wheel import checks passed. The README
badge layout was checked at desktop and mobile widths.

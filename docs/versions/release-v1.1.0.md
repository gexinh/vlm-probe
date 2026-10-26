# Release 1.1.0

## Goal

Publish the reviewed library to PyPI and synchronize the Hugging Face Space's
source, documentation, pretrained MJ-Lens walkthrough, overview and report to
the same release. Promote the validated development snapshot to a normal
release so a standard `pip install --upgrade vlmprobe` can obtain it.

## Differences from the previous version

- Promote `1.1.0.dev2` at `8ac9cca7bb5ba8d30c3dc43f636ffaca6ebbba14` to `1.1.0`.
  Numerical implementations, model bindings, calibration defaults and measured
  MJ outputs are unchanged.
- Compared with the previously published PyPI 1.0.1, include MJ-Lens and native
  Janus contracts, the complete pretrained VLM walkthrough, Qwen text adapters,
  corpus calibration fixes, explicit position-concentration diagnostics,
  audited native geometry and percentile display controls.
- Use the canonical `gexinh/vlm-probe` URL for pre-release source metadata.
- Synchronize both GitHub repositories with the established lab-only beta
  footer, and export only tracked source plus generated static-site files to
  Hugging Face. Preserve its Space configuration and update source provenance.
- Keep model weights, downloaded calibration images, fitted transports, private
  feedback PDFs and temporary build/verification directories out of the release.

## Expected results

PyPI distribution metadata and both `vlmprobe.__version__` and
`vlm_probing.__version__` should report 1.1.0. Installing the released wheel
should expose MJ-Lens, the revised Jacobian estimator and attention diagnostics
with the same runtime source as the reviewed snapshot except its version line.

The Hugging Face static app should show MJ-Lens in the Lens methods and main
overview, link the complete pretrained notebook, and serve the matching
measured preview and unchanged latest technical-report PDF. Its source files
must match the recorded GitHub source commit, with only documented
Space-specific deployment files differing.

## Validation scope

The reviewed development source passed 396 source tests and 111 independent
installed-wheel tests, with all runtime hashes checked. The release adds a new
distribution build, strict metadata checks and installation tests for 1.1.0.
Publication is verified against PyPI-hosted files and the live Hugging Face app;
a started upload or successful local build alone is not treated as publication.

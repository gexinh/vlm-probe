# Personal pre-release publication and reciprocal links, v1

## Goal and baseline

Publish the reviewed MJ bbox readouts, pretrained SAE notebook and clearer
tutorial navigation to the personal repository for inspection. Keep the lab
stable version separate until an explicit promotion. Start from local tutorial
revision `e2613bf` (`feat/region-mj-sae-tutorials`, `1.1.1.dev0`).

## Changes from the previous version

- Add a trailing **Stable version** button to the personal README. Keep the lab's
  trailing **Pre-release version** button. No standalone section is added.
- Render project installation and documentation URLs for the current repository;
  compare normalized bodies when promoting reviewed changes. Preserve unrelated
  README text and reject malformed or modified managed blocks.
- Preserve historical footerless and beta-footer source histories during a
  reviewed migration, without force pushes or automatic lab promotion.
- Pin the Colab notebook and runtime source to reviewed immutable revisions.
  This prevents a personal beta tutorial from installing the old lab code or a
  later incompatible personal `main`. Keep explicit repository/ref overrides.
- Retain existing experiment implementations, raw data, calibrated operators and
  scientific outputs. Only launch/setup metadata changes where required.

## Expected results

The personal README opens the lab stable repository from its final button; the
lab beta button opens the personal repository. Repeated README rendering is
idempotent, both newline styles remain valid, and unknown collaborator edits
are rejected rather than overwritten. Tutorial launch paths must include the
new SAE notebook and use the native APIs from the verified source revision.

Scientific readouts should remain unchanged under the same configuration.
The runtime setup is a local no-op; Colab obtains and installs the pinned source.
No SAE training, PyPI upload or Hugging Face synchronization is part of this
publication. The lab code remains at its current stable commit for review.

## Verification and publication

- Full local suite: **655 passed, 2 opt-in skipped**, with 456 subtests passed.
- Reciprocal footer/synchronization contracts: 21 tests passed; notebook setup:
  19 tests passed. All remote-write fixtures use temporary local bare repositories.
- Both MJ and SAE notebooks completed all 10 code cells with zero errors and
  seven/two inline figures after the launch-binding update. Existing nine
  workflow cells and scientific outputs are preserved.
- Colab notebook launch revision is
  `gexinh/vlm-probe@c50b3ed3d289d298f6ffbf1852ea34cd83ce25f6`.
- Runtime source is pinned to
  `gexinh/vlm-probe@e2613bf7ef55f1889aa9f95d2004d1ab6770c1b1`.
  Hosted Colab itself is not exercised by these local checks.
- Publication target is personal `main`; the lab code remains at
  `8b3ccd1d29bfde7e01247f34f051f95d61047e53` until explicit promotion.
  Final remote identity and workflow results are verified after pushing.

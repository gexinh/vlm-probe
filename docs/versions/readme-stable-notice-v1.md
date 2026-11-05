# README stable-version notice v1

## Goal

Promote the reviewed personal-library changes to the laboratory repository and
make the stable version easy to find from the personal README.

## Differences from the previous version

- Move the personal repository's stable-version link from the footer to a short
  centered notice between the badge row and documentation navigation. Its button
  opens `ubc-tea/vlm-probe`.
- Keep the notice out of the laboratory README. The shared documentation and
  library files remain identical after repository-specific README rendering.
- Extend the synchronization utility to recognize the new notice and migrate
  historical footer layouts while preserving contributor edits and Git history.
- Include the previously reviewed Gemma 4 adapter, native attention parity fixes,
  mixed-mask capability declarations, tests and guides in the laboratory main.

The runtime algorithms and package version are unchanged by the README update.
This promotion does not create a release or update PyPI.

## Expected behavior

Readers of the personal README can open the stable laboratory repository from
the first screen. The laboratory README has no redundant link directing readers
back to its own stable version.

Synchronization should preserve both repository histories, reject unreviewed
divergence, retain all shared files, and change only the intended managed README
sections and repository URLs. Repeated rendering and synchronization should be
idempotent. Existing Git tags, published releases and the `gexinh` branch remain
unchanged.

## Validation

Validation covers README rendering and its visual layout, managed-block
validation, migration from old footers, disposable-repository synchronization,
the real remote histories and equality of all shared tracked files. The Gemma
runtime changes were previously validated at `a8c57bb`; the laboratory's CI
checks the promoted package and model contracts after synchronization.

The 27 disposable-repository synchronization tests passed, together with 21
independent checks of notice placement, newline handling, shared-content
preservation and rejection of malformed or edited blocks. The personal header
was rendered in Chrome and checked visually. Removing the repository-specific
blocks produces the same README body before and after this update.

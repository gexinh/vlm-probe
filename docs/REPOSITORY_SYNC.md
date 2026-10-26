# Laboratory and personal repositories

The laboratory repository is [ubc-tea/vlm-probe](https://github.com/ubc-tea/vlm-probe).
The pre-release repository is [gexinh/vlm-probe](https://github.com/gexinh/vlm-probe).
The lab repository carries reviewed updates; the personal repository can carry
beta or pre-release development first. After a promotion, their library, version,
demos, input images and report are identical. README content is shared; project
URLs point to the repository being viewed. The personal README ends with a small
**Stable version** button pointing to the lab repository. The lab README ends
with a **Pre-release version** button pointing to the personal repository.
Neither button adds a separate README section.

Issues, pull requests, GitHub Releases and repository settings are separate
GitHub objects; pushing source does not duplicate them or publish a PyPI release.

## Synchronize an update

In a new checkout, configure these aliases once, preserving its existing `origin`:

```bash
git remote add personal git@github.com:gexinh/vlm-probe.git
git remote add lab git@github.com:ubc-tea/vlm-probe.git
```

If an alias already exists, inspect it with `git remote -v` rather than adding it
again. From the repository root on `main`, inspect or publish with:

```bash
python tools/sync_repositories.py
python tools/sync_repositories.py --push
```

Use the personal checkout as the shared editing source. The tool fetches both
branches and checks for unmerged remote changes before publishing. It renders
README project URLs and the repository-specific trailing link block, preserves
both histories, and verifies that only README.md differs between the promoted trees. It also checks
that stripping the managed footers and normalizing project URLs gives the exact
shared README body. It never force-pushes or deletes refs. If someone edited
either GitHub copy, merge that remote's changes locally, review them, commit,
and retry.
Publishing requires a clean working tree. Release tags can be pushed to both
remotes separately after reviewing the release.

### Publish a beta without promoting it

Push the reviewed development commit to the personal repository only:

```bash
git push personal main
```

This leaves the lab repository unchanged. Use a development or pre-release
package version when appropriate. Once that version has been reviewed, run the
explicit synchronization command above to promote the reviewed `main` tree.
There is no automatic mirroring from personal pushes into the lab repository.

Colab buttons open an immutable, verified notebook revision. Its setup cell
loads the pinned runtime source recorded in the tutorial, so a personal beta
never silently installs an older lab `main`. `VLM_PROBING_REPOSITORY` and
`VLM_PROBING_REF` remain explicit overrides. Existing user checkouts are reused
without resetting their contents. Refresh and verify both pins when tutorial
code requires a different runtime source.

### README rendering and migration

The publishing helper exposes `release_readme(text, repository)` in
`tools/sync_repositories.py`. It renders project URLs for the selected repository
and adds its single trailing counterpart button. `shared_readme_body(text)` removes the managed
footer and normalizes project URLs for content comparison. Repeated rendering
is a no-op. The personal block uses `<!-- vlmprobe:stable-link:start -->` /
`<!-- vlmprobe:stable-link:end -->`; the lab block retains
`<!-- vlmprobe:prerelease-link:start -->` /
`<!-- vlmprobe:prerelease-link:end -->`. Mixed, duplicate, malformed, modified
or nontrailing managed blocks cause an error rather than deleting text.

When migrating an existing checkout, render the personal form once, review the
README diff, and commit it before publishing. The command accepts the exact
historical footerless personal body, lab beta footer and lab URL-only renderings
when preserving old commits. Other
remote README edits still require explicit merging and review; they are never
overwritten just because they affect README.md. The old `render_readme(...)`
URL renderer remains available for validating those historical commits.

Synchronization is explicit, not a background service. Two remote writes cannot
be atomic: a network failure or concurrent update can leave one ahead. Fetch,
inspect, and rerun after resolving the cause. No force-pushes or deleted refs are
used.

## One page per repository

- Lab: https://ubc-tea.github.io/vlm-probe/
- Personal: https://gexinh.github.io/vlm-probe/

The same Pages workflow renders repository links from `GITHUB_REPOSITORY` at
build time. Each deployed website links to its own code and documentation.
Recorded experimental measurements retain their original provenance. Both
websites include the same report PDF under `assets/vlmprobe-technical-report.pdf`.

A repository administrator must enable **Settings → Pages → Build and deployment
→ Source: GitHub Actions** in each repository. After enabling it, run
**Actions → Deploy project page → Run workflow** on `main`.

For a local build:

```bash
python tools/build_project_page.py --repository ubc-tea/vlm-probe --output /tmp/vlmprobe-lab-site
python tools/build_project_page.py --repository gexinh/vlm-probe --output /tmp/vlmprobe-personal-site
```

## One report and one PyPI package

The current reviewed PDF is [docs/assets/vlmprobe-technical-report.pdf](assets/vlmprobe-technical-report.pdf).
Its source commit and checksum are recorded in [report provenance](assets/report-provenance.json).
The standalone [LaTeX repository](https://github.com/gexinh/vlm-probing-report)
remains the report authoring source. Update and compile it first, then replace
the tracked PDF and provenance record. Local report sources and historical
experiments under `docs/report/` remain excluded from this library repository.

`pip install vlmprobe` downloads the published distribution from PyPI. It does
not read either GitHub `main` branch. One PyPI project suffices for both mirrors;
the release workflow remains bound to **gexinh/vlm-probe**, `package.yml`,
environment `pypi`. Only that repository can run the publish job. Ordinary pushes
build and test but do not publish. Before publishing a new release, increment
the package version and synchronize the reviewed source. A published PyPI file
cannot be replaced with a different file under the same version/filename.

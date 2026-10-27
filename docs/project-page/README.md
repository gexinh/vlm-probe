# VLMProbe project page

A responsive research project page for four method families, explicit model
adapters and 11 notebook tutorials. It includes the framework diagram, links to
the arXiv paper and video demo, and a replay of a recorded Attention Knockout
experiment. The browser replay does not run new model inference.

## Build and preview

From the repository root:

```bash
python tools/build_project_page.py --repository ubc-tea/vlm-probe --output /tmp/vlmprobe-site
python -m http.server 8765 --directory /tmp/vlmprobe-site
```

Open http://localhost:8765/. The build copies the current report, framework and
measured MJ-Lens preview, and generates the searchable SAE catalog. Use HTTP
because the experiment and catalog fetch local JSON.

## Edit

- Page text and resource links: `docs/project-page/index.html`.
- Styling and responsive layout: `docs/project-page/styles.css`.
- Browser interaction: `docs/project-page/app.js`.
- Archived experiment and its provenance: `docs/project-page/assets/experiment.json`.
- README badges, citation and video entry: the root `README.md`.

Keep Python imports, package names and repository paths as `vlmprobe` or
`vlm-probe`; visible project headings use **VLMProbe**.

## GitHub Pages

- Laboratory: https://ubc-tea.github.io/vlm-probe/
- Personal: https://gexinh.github.io/vlm-probe/

The shared workflow renders navigation for the repository being deployed.
Enable **Settings → Pages → Build and deployment → Source: GitHub Actions**,
then run **Actions → Deploy project page → Run workflow** on `main` if needed.
Subsequent changes to the watched page sources deploy automatically.

See [repository synchronization](../REPOSITORY_SYNC.md) for promotion and the
repository-specific Stable / Pre-release README buttons.

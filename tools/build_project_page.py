#!/usr/bin/env python3
"""Build either repository's Pages site without changing the shared source.

Usage::

    python tools/build_project_page.py --repository ubc-tea/vlm-probe --output /tmp/vlmprobe-site

GitHub Actions supplies GITHUB_REPOSITORY; local builds default to the personal
repository. Archived experiment data is copied byte for byte, including its
original provenance URL. Only HTML navigation links are repository-specific.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
from pathlib import Path
import re
import runpy
import shutil


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs" / "project-page"
REPORT = ROOT / "docs" / "assets" / "vlmprobe-technical-report.pdf"
OVERVIEW = ROOT / "docs" / "assets" / "library-overview-sae.svg"
MJ_PREVIEW = ROOT / "docs/assets/tutorials/lens-mj-report-region-tokens.png"
REPORT_LINK = "assets/vlmprobe-technical-report.pdf"
REPOSITORIES = ("gexinh/vlm-probe", "gexinh/vlm-prob", "gexinh/vlm-probing", "ubc-tea/vlm-probe")
CATALOG_RESOURCE = ROOT / "src/vlm_probing/sae/resources/published_model_cards.json.gz"
BUILD_MARKER = ".vlmprobe-site-build"
HREF = re.compile(r"(?P<prefix>\bhref\s*=\s*)(?P<quote>[\"'])(?P<url>.*?)(?P=quote)")


def repository_name(value: str) -> str:
    """Accept a GitHub owner/repository, never an arbitrary URL or path."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9-]*/[A-Za-z0-9_.-]+", value):
        raise argparse.ArgumentTypeError("repository must have the form owner/repository")
    if value.split("/", 1)[1] in {".", ".."}:
        raise argparse.ArgumentTypeError("repository name cannot be '.' or '..'")
    return value


def rewrite_links(html: str, repository: str) -> str:
    """Resolve project navigation for this mirror; leave other citations alone."""
    target = f"https://github.com/{repository}"

    def replace(match: re.Match[str]) -> str:
        url = match["url"]
        for name in REPOSITORIES:
            base = f"https://github.com/{name}"
            if url != base and not url.startswith(base + "/"):
                continue
            suffix = url[len(base):]
            filename = suffix.rsplit("/", 1)[-1]
            if suffix.startswith("/releases/") and filename in {
                "vlm-probing-technical-report.pdf",
                "vlmprobe-technical-report.pdf",
            }:
                url = REPORT_LINK
            else:
                url = target + suffix
            break
        return f'{match["prefix"]}{match["quote"]}{url}{match["quote"]}'

    return HREF.sub(replace, html)


def build(repository: str, output: Path, *, ref: str | None = None) -> Path:
    output = output.resolve()
    if ROOT.is_relative_to(output) or SOURCE.is_relative_to(output) or output.is_relative_to(SOURCE):
        raise ValueError("output must not overwrite the repository or project-page source")
    if REPORT.is_relative_to(output):
        raise ValueError("output must not contain the source report")
    for name in ("index.html", "styles.css", "app.js", "assets/experiment.json"):
        if not (SOURCE / name).is_file():
            raise FileNotFoundError(f"missing project-page source: {SOURCE / name}")
    if not REPORT.is_file():
        raise FileNotFoundError(f"missing latest technical report: {REPORT}")
    if not OVERVIEW.is_file():
        raise FileNotFoundError(f"missing current library overview: {OVERVIEW}")
    if not MJ_PREVIEW.is_file():
        raise FileNotFoundError(f"missing measured MJ-Lens preview: {MJ_PREVIEW}")
    with REPORT.open("rb") as stream:
        if stream.read(5) != b"%PDF-":
            raise ValueError(f"technical report is not a PDF: {REPORT}")
    if output.exists():
        if not output.is_dir():
            raise ValueError(f"output is not a directory: {output}")
        if any(output.iterdir()) and not (output / BUILD_MARKER).is_file():
            raise ValueError(f"refusing to replace a non-build directory: {output}")
        shutil.rmtree(output)
    output.mkdir(parents=True)
    html = rewrite_links((SOURCE / "index.html").read_text(encoding="utf-8"), repository)
    (output / "index.html").write_text(html, encoding="utf-8")
    for name in ("styles.css", "app.js"):
        shutil.copy2(SOURCE / name, output / name)
    shutil.copytree(SOURCE / "assets", output / "assets")
    shutil.copy2(REPORT, output / REPORT_LINK)
    shutil.copy2(OVERVIEW, output / "assets" / OVERVIEW.name)
    shutil.copy2(MJ_PREVIEW, output / "assets/lens-mj-report-region-tokens.png")
    # Generate directly in the build tree: each mirror gets its own navigation,
    # and building a site must not rewrite the checked-in documentation.
    payload = json.loads(gzip.decompress(CATALOG_RESOURCE.read_bytes()))
    catalog = runpy.run_path(str(ROOT / "tools/build_sae_model_pages.py"))
    owner, name = repository.split("/", 1)
    catalog["build_web"](
        payload["models"], payload["counts"], repository=repository,
        ref=ref or os.environ.get("GITHUB_REF_NAME", "main"),
        site_url=f"https://{owner}.github.io/{name}/sae-models/",
        output=output / "sae-models",
    )
    (output / ".nojekyll").touch()
    (output / BUILD_MARKER).write_text(repository + "\n", encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repository",
        type=repository_name,
        default=os.environ.get("GITHUB_REPOSITORY", REPOSITORIES[0]),
        help="GitHub owner/repository (default: GITHUB_REPOSITORY or gexinh/vlm-probe)",
    )
    parser.add_argument("--ref", help="Git ref for model-card source links (default: GITHUB_REF_NAME or main)")
    parser.add_argument("--output", type=Path, default=ROOT / "_site")
    args = parser.parse_args()
    output = build(args.repository, args.output, ref=args.ref)
    print(f"Built {args.repository} project page at {output}")


if __name__ == "__main__":
    main()

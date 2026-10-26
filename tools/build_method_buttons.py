#!/usr/bin/env python3
"""Build the README's local, individually linked method buttons."""
from __future__ import annotations

import re
from html import escape
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "docs/assets/method-buttons"

# Family, fill, border, icon colour, icon geometry, (label, asset name, guide).
FAMILIES = (
    ("Lens", "#effaf8", "#b8dfd9", "#087d79",
     '<circle cx="18" cy="17" r="5"/><path d="m22 21 4 4M18 14v6m-3-3h6"/>', (
        ("Logit Lens", "logit-lens", "logit_lens.md"),
        ("Tuned Lens", "tuned-lens", "tuned_lens.md"),
        ("Attention Lens", "attention-lens", "attention_lens.md"),
        ("Jacobian Lens", "jacobian-lens", "jacobian_lens.md"),
        ("MJ-Lens", "mj-lens", "mj_lens.md"),
        ("EmbedLens", "embed-lens", "embed_lens.md"),
        ("Patchscopes", "patchscopes", "patchscope.md"),
    )),
    ("Attention", "#f1f5ff", "#c8d7f5", "#3266d5",
     '<path d="M10 18s3-6 9-6 9 6 9 6-3 6-9 6-9-6-9-6Z"/><circle cx="19" cy="18" r="2.5"/>', (
        ("Rollout", "rollout", "attention_rollout.md"),
        ("Grad-CAM", "grad-cam", "attention_maps.md"),
        ("ATTATTR", "attattr", "attention_maps.md"),
        ("TAM", "tam", "attention_maps.md"),
        ("Beyond Intuition", "beyond-intuition", "attention_maps.md"),
        ("Generic relevance", "generic-relevance", "attention_relevance.md"),
        ("ViT DTD / LRP", "vit-dtd-lrp", "chefer_lrp.md"),
    )),
    ("Causal", "#f7f3ff", "#dacdf3", "#7950c5",
     '<circle cx="12" cy="18" r="2.5"/><circle cx="25" cy="12" r="2.5"/><circle cx="25" cy="24" r="2.5"/><path d="m14.5 17 8-4m-8 6 8 4"/>', (
        ("Activation patching", "activation-patching", "activation_patching.md"),
        ("IOI-style path patching", "path-patching", "path_patching.md"),
        ("Attribution patching", "attribution-patching", "attribution_patching.md"),
        ("Attention Knockout", "attention-knockout", "attention_knockout.md"),
        ("EAP / EAP-IG", "eap-ig", "eap_ig.md"),
        ("Residual steering / VSV", "steering-vsv", "steering.md"),
    )),
    ("SAE", "#fff9ed", "#ebd6a9", "#9b6612",
     '<rect x="11" y="11" width="5" height="5" rx="1"/><rect x="21" y="11" width="5" height="5" rx="1"/><rect x="11" y="21" width="5" height="5" rx="1"/><path d="M21 21h5v5h-5z" fill="currentColor"/>', (
        ("ReLU", "relu", "sae/relu.md"),
        ("TopK", "topk", "sae/topk.md"),
        ("Gated", "gated", "sae/gated.md"),
        ("JumpReLU", "jumprelu", "sae/jumprelu.md"),
        ("AbsTopK", "abstopk", "sae/abstopk.md"),
        ("BatchTopK", "batchtopk", "sae/batchtopk.md"),
        ("Matryoshka BatchTopK", "matryoshka-batchtopk", "sae/matryoshka_batchtopk.md"),
        ("Archetypal TopK", "archetypal-topk", "sae/archetypal_topk.md"),
        ("Archetypal JumpReLU", "archetypal-jumprelu", "sae/archetypal_jumprelu.md"),
        ("Matching Pursuit", "matching-pursuit", "sae/matching_pursuit.md"),
        ("Signed MP", "signed-mp", "sae/signed_matching_pursuit.md"),
        ("Nonnegative OMP", "nonnegative-omp", "sae/orthogonal_matching_pursuit.md"),
        ("Signed OMP", "signed-omp", "sae/signed_orthogonal_matching_pursuit.md"),
        ("QSAE", "qsae", "sae/qsae.md"),
    )),
    ("Supporting tools", "#f5f7fa", "#d8dfe8", "#596b80",
     '<path d="M11 13h16M11 23h16"/><circle cx="16" cy="13" r="2.5" fill="#f5f7fa"/><circle cx="22" cy="23" r="2.5" fill="#f5f7fa"/>', (
        ("Attention profile", "attention-profile", "attention_profile.md"),
        ("Head logit attribution", "head-logit-attribution", "head_logit_attribution.md"),
        ("Probability reweighting", "probability-reweighting", "attention_reweight.md"),
        ("Temperature", "temperature", "attention_temperature.md"),
    )),
)


def button_svg(label: str, fill: str, border: str, colour: str, icon: str) -> str:
    """Keep spacing inside the SVG because GitHub strips inline README CSS."""
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="210" height="46" viewBox="0 0 210 46" role="img" aria-label="{escape(label, quote=True)}">
  <title>{escape(label)}</title>
  <rect x="0.5" y="0.5" width="199" height="35" rx="7" fill="{fill}" stroke="{border}"/>
  <g color="{colour}" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" fill="none">{icon}</g>
  <text x="36" y="18" dominant-baseline="central" fill="#24364b" font-family="Arial, Helvetica, sans-serif" font-size="13.5" font-weight="500">{escape(label)}</text>
</svg>
'''


def main() -> None:
    readme = ROOT / "README.md"
    text = readme.read_text()
    before, rest = text.split("## Methods\n", 1)
    next_section = re.search(r"(?m)^## [^\n]+(?:\n|$)", rest)
    if next_section is None:
        raise ValueError("README Methods section must have a following level-two heading.")
    after = rest[next_section.start():]
    ASSETS.mkdir(parents=True, exist_ok=True)
    lines = []
    expected_assets = set()
    for family, fill, border, colour, icon, methods in FAMILIES:
        lines.extend([f"### {family}", "", "<p>"])
        for label, name, guide in methods:
            path = f"docs/methods/{guide}"
            if not (ROOT / path).is_file():
                raise FileNotFoundError(path)
            filename = name + ".svg"
            expected_assets.add(filename)
            (ASSETS / filename).write_text(button_svg(label, fill, border, colour, icon))
            lines.append(
                f'  <a href="{path}"><img src="docs/assets/method-buttons/{filename}" '
                f'alt="{escape(label, quote=True)}" width="200" height="44"></a>'
            )
        lines.extend(["</p>", ""])
    # Remove obsolete generated buttons after a method is renamed or removed.
    for path in ASSETS.glob("*.svg"):
        if path.name not in expected_assets:
            path.unlink()
    intro = ("## Methods\n\n"
             "Each guide explains the implemented computation, parameters, outputs, and\n"
             "relationship to its original paper. Reusable primitives are labeled separately\n"
             "from paper methods.\n\n")
    readme.write_text(before + intro + "\n".join(lines) + "\n" + after)
    print(f"Built {len(expected_assets)} method buttons across {len(FAMILIES)} groups.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build compact model-family navigation labels for the README support table."""
from __future__ import annotations

from html import escape
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "docs/assets/model-buttons"

# Original line icons, drawn in the same visual language as the method buttons.
QWEN_ICON = '<path d="m18 8 2.4 5.6L26 16l-5.6 2.4L18 24l-2.4-5.6L10 16l5.6-2.4Z"/><path d="m24 7 .8 1.8L27 10l-2.2.8L24 13l-.8-2.2L21 10l2.2-1.2Z"/>'
FAMILIES = (
    ("Qwen3-VL", "qwen3-vl", "qwen3_vl.md", "#f5f1ff", "#dfd4f5", "#7954bd", QWEN_ICON),
    ("Qwen2.5-VL", "qwen2-5-vl", "qwen2_5_vl.md", "#f5f1ff", "#dfd4f5", "#7954bd", QWEN_ICON),
    ("Qwen2-VL", "qwen2-vl", "qwen2_vl.md", "#f5f1ff", "#dfd4f5", "#7954bd", QWEN_ICON),
    ("Qwen3.5", "qwen3-5", "qwen3_5.md", "#f5f1ff", "#dfd4f5", "#7954bd", QWEN_ICON),
    ("SmolVLM", "smolvlm", "smolvlm.md", "#effaf8", "#c0e2da", "#168275",
     '<path d="M11 9h15v11h-8l-5 4v-4h-2Z"/><circle cx="15" cy="14.5" r=".7"/><circle cx="18.5" cy="14.5" r=".7"/><circle cx="22" cy="14.5" r=".7"/>'),
    ("InternVL", "internvl", "internvl.md", "#eff6ff", "#cbdcf3", "#3b76b5",
     '<ellipse cx="18" cy="16" rx="9" ry="4.5" transform="rotate(-35 18 16)"/><ellipse cx="18" cy="16" rx="4.5" ry="9" transform="rotate(-35 18 16)"/><circle cx="18" cy="16" r="1.6"/>'),
    ("LLaVA", "llava", "llava.md", "#eff8f5", "#c8e0d7", "#267e67",
     '<path d="M11 9h15v11H11Z"/><circle cx="21.5" cy="12.5" r="1.3"/><path d="m12 18 4-4 4 4 2-2 3 3M15 23h7"/>'),
    ("Pixtral", "pixtral", "pixtral.md", "#fff6ed", "#f1d6b8", "#b97622",
     '<rect x="11" y="9" width="5" height="5" rx="1"/><rect x="20" y="9" width="5" height="5" rx="1"/><rect x="11" y="18" width="5" height="5" rx="1"/><path d="M20 18h5v5h-5z" fill="#b97622"/>'),
    ("Janus", "janus", "janus.md", "#f1f3ff", "#d2d8f2", "#626eb1",
     '<circle cx="14.5" cy="16" r="6"/><circle cx="21.5" cy="16" r="6"/><path d="M18 11v10"/>'),
    ("Language models", "language-models", "README.md", "#f4f6fa", "#d7dfe9", "#63758c",
     '<path d="M11 8h14v16H11Z"/><path d="M14 12h8M14 16h8M14 20h5"/>'),
    ("Vision classifier", "vision-classifier", "vit.md", "#f0f7fc", "#cadfea", "#3f7c9d",
     '<rect x="10" y="8" width="17" height="16" rx="2"/><path d="M16 8v16M21 8v16M10 13h17M10 19h17"/>'),
)


def button_svg(label: str, fill: str, border: str, colour: str, icon: str) -> str:
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="172" height="32" viewBox="0 0 172 32" role="img" aria-label="{escape(label, quote=True)}">
  <title>{escape(label)}</title>
  <rect x=".5" y=".5" width="171" height="31" rx="6" fill="{fill}" stroke="{border}"/>
  <g stroke="{colour}" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" fill="none">{icon}</g>
  <text x="36" y="16" dominant-baseline="central" fill="#293d55" font-family="Arial, Helvetica, sans-serif" font-size="13" font-weight="600">{escape(label)}</text>
</svg>
'''


def main() -> None:
    readme = ROOT / "README.md"
    text = readme.read_text()
    before, remainder = text.split("## Model Support\n", 1)
    section, after = remainder.split("\n## Methods\n", 1)
    ASSETS.mkdir(parents=True, exist_ok=True)
    expected = set()
    for label, name, guide, fill, border, colour, icon in FAMILIES:
        destination = f"docs/models/{guide}"
        if not (ROOT / destination).is_file():
            raise FileNotFoundError(destination)
        filename = name + ".svg"
        expected.add(filename)
        (ASSETS / filename).write_text(button_svg(label, fill, border, colour, icon))
        family_cell = (f'<a href="{destination}"><img src="docs/assets/model-buttons/{filename}" '
                       f'alt="{escape(label, quote=True)}" width="172" height="32"></a>')
        # Replace only the first cell. Existing checkpoints and guide columns stay intact.
        pattern = rf'(?m)^\| (?:{re.escape(label)}|<a\b[^\n]*?alt="{re.escape(label)}"[^\n]*?</a>) \|'
        section, count = re.subn(pattern, lambda _: "| " + family_cell + " |", section)
        if count != 1:
            raise ValueError(f"expected exactly one support-table row for {label}")
    for path in ASSETS.glob("*.svg"):
        if path.name not in expected:
            path.unlink()
    readme.write_text(before + "## Model Support\n" + section + "\n## Methods\n" + after)
    print(f"Built {len(expected)} model-family buttons; other table columns preserved.")


if __name__ == "__main__":
    main()

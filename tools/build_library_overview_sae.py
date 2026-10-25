#!/usr/bin/env python3
"""Build the four-family overview without changing the approved source figure.

The input, configure and output panels inherit the editable SVG drawing from
``docs/assets/library-protocol.svg``. The new Tools panel keeps the original
palette and adds a fourth sparse-autoencoder branch. All miniature plots are
schematic, not experiment results.

Run from the repository root::

    python tools/build_library_overview_sae.py --export

SVG generation uses only the standard library. PNG and vector PDF export use
Inkscape when ``--export`` is supplied.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import os
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "docs/assets/library-protocol.svg"
OUTPUT = ROOT / "docs/assets/library-overview-sae.svg"
NS = "http://www.w3.org/2000/svg"
ET.register_namespace("", NS)


def element(tag: str, **attributes: object) -> ET.Element:
    return ET.Element(f"{{{NS}}}{tag}", {k.replace("_", "-"): str(v) for k, v in attributes.items()})


def add(parent: ET.Element, tag: str, **attributes: object) -> ET.Element:
    child = element(tag, **attributes)
    parent.append(child)
    return child


def text(parent: ET.Element, x: float, y: float, value: str, **attributes: object) -> ET.Element:
    child = add(parent, "text", x=x, y=y, **attributes)
    child.text = value
    return child


def by_id(root: ET.Element, identifier: str) -> ET.Element:
    return next(node for node in root.iter() if node.get("id") == identifier)


def find_text(parent: ET.Element, value: str) -> ET.Element:
    return next(node for node in parent.iter(f"{{{NS}}}text") if node.text == value)


def grid(parent: ET.Element, x: int, y: int, values: list[list[str]], cell_w: int, cell_h: int) -> None:
    for row, colors in enumerate(values):
        for column, color in enumerate(colors):
            add(parent, "rect", x=x + column * cell_w, y=y + row * cell_h,
                width=cell_w - 1.5, height=cell_h - 1.5, fill=color)


def miniature(parent: ET.Element, family: str, y: int) -> None:
    x, top = 1046, y + 49
    if family == "lens":
        add(parent, "rect", x=x, y=top, width=126, height=77, rx=4,
            fill="#FFFFFF", stroke="#DCE5F1")
        grid(parent, x + 3, top + 3, [
            ["#D6F2ED", "#B0E6DE", "#D6F2ED", "#E7F7F4"],
            ["#90DBCF", "#59C8B9", "#B0E6DE", "#D6F2ED"],
            ["#59C8B9", "#159F99", "#90DBCF", "#B0E6DE"],
            ["#B0E6DE", "#087D7A", "#59C8B9", "#90DBCF"],
            ["#D6F2ED", "#087D7A", "#B0E6DE", "#D6F2ED"],
        ], 30, 14)
        caption = "Layer–token map"
    elif family == "attention":
        add(parent, "rect", x=x, y=top, width=126, height=77, rx=4,
            fill="#EAF2FF", stroke="#DCE5F1")
        add(parent, "path", d=f"M{x+1} {top+56}L{x+33} {top+28}L{x+60} {top+56}L{x+86} {top+19}L{x+125} {top+59}V{top+76}H{x+1}Z",
            fill="#C4D6EE")
        for row, opacities in enumerate(((.25, .55, .25), (.55, .95, .55), (.25, .55, .25))):
            for column, opacity in enumerate(opacities):
                add(parent, "rect", x=x + 52 + column * 18, y=top + 13 + row * 18,
                    width=16, height=16, fill="#4C8DF6", opacity=opacity)
        caption = "Attribution map"
    elif family == "causal":
        baseline = top + 30
        add(parent, "line", x1=x + 5, y1=baseline, x2=x + 124, y2=baseline,
            stroke="#7183A0", stroke_width=1.5)
        add(parent, "line", x1=x + 5, y1=top + 8, x2=x + 5, y2=top + 75,
            stroke="#7183A0", stroke_width=1.5)
        for index, (height, color) in enumerate(((13, "#B29AE9"), (21, "#9E7FE2"),
                                                (40, "#8259E8"), (-16, "#4C8DF6"), (8, "#C7B6F0"))):
            add(parent, "rect", x=x + 16 + index * 23, y=baseline + min(height, 0),
                width=14, height=abs(height), rx=1.5, fill=color)
        caption = "Score changes"
    else:
        add(parent, "rect", x=x, y=top, width=126, height=77, rx=4,
            fill="#FFFFFF", stroke="#E8DEBD")
        grid(parent, x + 4, top + 5, [
            ["#F5F1E8", "#F5F1E8", "#EDC16A", "#F5F1E8", "#F5F1E8", "#F5F1E8"],
            ["#F5F1E8", "#F5F1E8", "#F5F1E8", "#F5F1E8", "#F5F1E8", "#C9901D"],
            ["#F5F1E8", "#D49B26", "#F5F1E8", "#F5F1E8", "#F5F1E8", "#F5F1E8"],
            ["#F5F1E8", "#F5F1E8", "#F5F1E8", "#E4B44F", "#F5F1E8", "#F5F1E8"],
        ], 20, 17)
        caption = "Sparse features"
    text(parent, x + 63, y + 145, caption, **{"class": "small", "text-anchor": "middle", "style": "font-size:17px"})


def tools(root: ET.Element) -> None:
    for name in ("lens-tools", "attention-tools", "causal-tools"):
        root.remove(by_id(root, name))
    families = (
        ("lens", "Lens", "Readout · optional calibrated decoder", "#17BEBB", "#127D79", "#F7FCFB", "#B9E5DE",
         ("Logit · Tuned · Jacobian", "Attention Lens · MJ-Lens", "EmbedLens · Patchscope")),
        ("attention", "Attention", "Observation · target-score attribution", "#4C8DF6", "#326EBF", "#F7FAFF", "#CADDF9",
         ("Rollout · Grad-CAM · ATTATTR", "TAM · Beyond Intuition", "DTD/LRP")),
        ("causal", "Causal", "Intervention · attribution estimates", "#8259E8", "#6A42BB", "#FAF8FF", "#D9CFF5",
         ("Activation / path patching", "Knockout · Steering", "Attribution / EAP-IG estimates")),
        ("sae", "SAE", "Learned dictionary · feature analysis", "#D49920", "#946A16", "#FFFCF5", "#EEDBAA",
         ("ReLU · TopK · Gated · JumpReLU", "BatchTopK · Matryoshka · Pursuit", "Encode · Reconstruct · Edit")),
    )
    for index, (family, name, role, accent, role_color, fill, stroke, methods) in enumerate(families):
        y = 163 + index * 172
        group = add(root, "g", id=f"{family}-tools")
        add(group, "rect", x=636, y=y, width=560, height=156, rx=14,
            fill=fill, stroke=stroke, stroke_width=1.8)
        add(group, "rect", x=636, y=y + 18, width=5, height=120, rx=2, fill=accent)
        text(group, 657, y + 30, name, **{"class": "heading"})
        text(group, 657, y + 54, role, **{"class": "role", "style": f"fill:{role_color}"})
        for offset, method in zip((86, 111, 136), methods):
            text(group, 657, y + offset, method, **{"class": "body"})
        miniature(group, family, y)


def build() -> Path:
    root = ET.fromstring(BASELINE.read_text())
    root.set("height", "754")
    root.set("viewBox", "0 100 1600 754")
    by_id(root, "canvas").set("height", "854")
    by_id(root, "figure-title").text = "vlmprobe: lens, attention, causal and SAE analysis"
    by_id(root, "figure-description").text = (
        "Four-stage library overview. Native language, vision-language and vision inputs "
        "are bound through Prober and a model adapter. Layer and head selection, token "
        "groups, target scores and optional training or loading configure four tool "
        "families: Lens (including MJ-Lens), Attention, Causal and SAE. Results include readouts, attribution "
        "maps, measured interventions and sparse features. Structured experiment "
        "artifacts retain analysis results, figures, fitted lenses and SAE dictionaries."
    )
    by_id(root, "draft-metadata").text = (
        "New four-family overview; approved three-family baseline preserved separately. "
        "All four miniature plots are schematic and contain no experimental measurements. "
        "Generated by tools/build_library_overview_sae.py; editable SVG text and shapes."
    )

    connectors = by_id(root, "pipeline-connectors")
    connectors.clear()
    connectors.set("id", "pipeline-connectors")
    connectors.set("class", "flow")
    for identifier, path, arrow in (
        ("input-to-configure", "M252 499H290", True),
        ("configure-to-branch", "M586 499H611", False),
        ("tool-branch-trunk", "M611 241V757", False),
        ("output-trunk", "M1218 241V757", False),
        ("trunk-to-output", "M1218 499H1232", True),
    ):
        add(connectors, "path", id=identifier, d=path, **({"marker-end": "url(#flow-arrow)"} if arrow else {}))
    for family, center in zip(("lens", "attention", "causal", "sae"), (241, 413, 585, 757)):
        add(connectors, "path", id=f"branch-to-{family}", d=f"M611 {center}H626", marker_end="url(#flow-arrow)")
        add(connectors, "path", id=f"{family}-to-output-trunk", d=f"M1196 {center}H1218")

    for identifier in ("input-stage", "configure-stage", "output-stage"):
        panel = by_id(root, identifier)
        next(child for child in panel if child.tag.endswith("rect")).set("height", "672")

    inputs = by_id(root, "input-stage")
    by_id(root, "illustrative-image").set("transform", "translate(0 29)")
    find_text(inputs, "Image and/or text").set("y", "513")
    find_text(inputs, "Question / prompt").set("y", "543")
    paired = by_id(root, "optional-paired-input")
    paired.set("transform", "translate(0 67)")
    next(child for child in paired if child.tag.endswith("rect")).set("height", "166")
    find_text(paired, "Reference input").set("y", "665")
    find_text(paired, "Reference input").set("style", "font-size:18px")
    list(paired)[4].set("y", "651")
    find_text(inputs, "Native model inputs").set("y", "812")

    configure = by_id(root, "configure-stage")
    find_text(configure, "Available sites").text = "Model components"
    fitted = by_id(root, "optional-lens-calibration")
    fitted.clear()
    fitted.set("id", "optional-lens-sae-training")
    add(fitted, "rect", x=320, y=572, width=246, height=226, rx=8, **{"class": "optional"})
    text(fitted, 333, 603, "Optional fit / load", **{"class": "label"})
    text(fitted, 333, 638, "Calibrated lenses", **{"class": "body"})
    text(fitted, 333, 670, "SAE dictionaries", **{"class": "body"})
    text(fitted, 333, 717, "Train + held-out inputs", **{"class": "small"})
    text(fitted, 333, 752, "Save / load an artifact", **{"class": "small"})
    find_text(configure, "Check capabilities first").set("y", "823")

    tools(root)

    output = by_id(root, "output-stage")
    panel = deepcopy(list(output)[0])
    output.clear()
    output.set("id", "output-stage")
    output.append(panel)
    text(output, 1262, 198, "Structured outputs", **{"class": "heading"})
    for heading, y, lines in (
        ("Readouts", 248, ("Layer / head token scores",)),
        ("Attribution", 327, ("Maps and effect estimates",)),
        ("Interventions", 406, ("Changes in answer scores", "Changes in generated text")),
        ("Sparse features", 513, ("Feature activity and maps", "Reconstruction and sparsity")),
    ):
        text(output, 1262, y, heading, **{"class": "label"})
        for index, line in enumerate(lines):
            text(output, 1262, y + 29 + index * 28, line, **{"class": "body"})
    add(output, "line", x1=1262, y1=597, x2=1548, y2=597, **{"class": "separator"})
    text(output, 1262, 631, "Experiment archive", **{"class": "label"})
    for y, item in zip((666, 694, 722, 750, 778, 806), (
        "Input examples", "Model and method settings", "Analysis results", "Figures and plots",
        "Example notebooks", "Lenses and SAE dictionaries",
    )):
        add(output, "circle", cx=1267, cy=y - 6, r=2.6, fill="#53627A")
        text(output, 1280, y, item, **{"class": "body", "style": "font-size:19px"})

    # Keep the original drawing order: every card covers the connectors beneath it.
    output_index = list(root).index(output)
    for family in ("lens", "attention", "causal", "sae"):
        group = by_id(root, f"{family}-tools")
        root.remove(group)
        root.insert(output_index, group)
        output_index += 1
    for index, node in enumerate(root.iter(f"{{{NS}}}text"), start=1):
        if not node.get("id"):
            node.set("id", f"overview-text-{index}")
    ET.indent(root, space="  ")
    OUTPUT.write_text(ET.tostring(root, encoding="unicode") + "\n")
    return OUTPUT


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", action="store_true", help="Also export 2× PNG and vector PDF with Inkscape.")
    args = parser.parse_args()
    svg = build()
    print(svg)
    if args.export:
        for suffix, options in (("png", ["--export-width=3200"]), ("pdf", ["--export-text-to-path"])):
            destination = svg.with_suffix(f".{suffix}")
            subprocess.run(["inkscape", str(svg), f"--export-type={suffix}",
                            f"--export-filename={destination}", *options], check=True,
                           env={**os.environ, "DISPLAY": ""})
            print(destination)


if __name__ == "__main__":
    main()

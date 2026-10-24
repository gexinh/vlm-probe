#!/usr/bin/env python3
"""Check tutorial syntax, local links, API coverage and live notebook structure.

This does not execute models. Run the linked example scripts or notebooks to
verify numerical behavior; record those runs separately from these checks.
"""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
import re
from urllib.parse import unquote, urlsplit


ROOT = Path(__file__).resolve().parents[1]
FENCES = re.compile(r"^```([^\n]*)\n(.*?)^```", re.M | re.S)
LINKS = re.compile(r"!?\[[^\]]*\]\(([^\s)]+)(?:\s+\"[^\"]*\")?\)")
ROUTES = {
    "lenses": {
        "logit_lens": ["logit"], "tuned_lens": ["tuned"],
        "attention_lens": ["attention"], "jacobian_lens": ["jacobian"],
        "mj_lens": ["mj"],
        "embed_lens": ["embed"], "patchscope": ["patchscope"],
    },
    "attention": {
        "attention_profile": ["profile"], "attention_rollout": ["rollout"],
        "attention_relevance": ["relevance"], "attention_maps": [
            "grad_cam", "attribution", "tam", "beyond_intuition"],
        "attention_reweight": ["reweight"], "attention_temperature": ["temperature"],
        "head_logit_attribution": ["head_logits"], "chefer_lrp": ["dtd_lrp"],
    },
    "causal": {
        "activation_patching": ["patch"], "attribution_patching": ["attribute"],
        "path_patching": ["path"], "attention_knockout": ["knockout"],
        "eap_ig": ["eap_ig"], "steering": ["steer", "vsv"],
    },
}


def check() -> dict:
    errors = []
    counts = {"method_guides": 0, "model_guides": 0, "python_blocks": 0,
              "example_scripts": 0, "notebooks": 0, "factory_parameters": 0}
    for folder in ("methods", "models"):
        for path in sorted((ROOT / "docs" / folder).glob("*.md")):
            if path.name == "README.md":
                continue
            counts["method_guides" if folder == "methods" else "model_guides"] += 1
            content = path.read_text()
            for index, (language, source) in enumerate(FENCES.findall(content)):
                if language.strip() == "python":
                    counts["python_blocks"] += 1
                    try:
                        ast.parse(source)
                    except SyntaxError as exc:
                        errors.append(f"{path.relative_to(ROOT)} Python block {index}: {exc}")
            for destination in LINKS.findall(content):
                parsed = urlsplit(destination)
                if parsed.scheme or parsed.netloc or not parsed.path:
                    continue
                target = path.parent / unquote(parsed.path)
                if not target.exists():
                    errors.append(f"{path.relative_to(ROOT)}: missing local link {destination}")
    for module, pages in ROUTES.items():
        tree = ast.parse((ROOT / "src" / "vlm_probing" / "api" / f"{module}.py").read_text())
        family = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                      and node.name.endswith("Methods"))
        factories = {node.name: node for node in family.body if isinstance(node, ast.FunctionDef)}
        for page, routes in pages.items():
            content = (ROOT / "docs" / "methods" / f"{page}.md").read_text()
            for route in routes:
                args = factories[route].args
                for arg in args.posonlyargs + args.args + args.kwonlyargs:
                    if arg.arg == "self":
                        continue
                    counts["factory_parameters"] += 1
                    if not re.search(rf"\b{re.escape(arg.arg)}\b", content):
                        errors.append(f"{page}.md: missing {route} parameter {arg.arg}")
    for path in sorted((ROOT / "examples" / "tutorials").glob("*.py")):
        counts["example_scripts"] += 1
        try:
            ast.parse(path.read_text())
        except SyntaxError as exc:
            errors.append(f"{path.relative_to(ROOT)}: {exc}")
    for path in sorted((ROOT / "demos").glob("*.ipynb")):
        counts["notebooks"] += 1
        notebook = json.loads(path.read_text())
        code = []
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] != "code":
                continue
            source = "".join(cell["source"])
            # Installation magics are valid notebook syntax, not Python AST.
            clean = "\n".join(line for line in source.splitlines()
                              if not line.lstrip().startswith(("%", "!")))
            try:
                ast.parse(clean)
            except SyntaxError as exc:
                errors.append(f"{path.name} cell {index}: {exc}")
            if any(output.get("output_type") == "error" for output in cell.get("outputs", [])):
                errors.append(f"{path.name} cell {index}: stored execution error")
            code.append(source)
        source = "\n".join(code)
        for forbidden in ("load_snapshot", "from demos.common import", "show_figure("):
            if forbidden in source:
                errors.append(f"{path.name}: replay helper {forbidden}")
        if "Prober(" not in source:
            errors.append(f"{path.name}: missing visible model binding")
        if ".run(" not in source and ".generate(" not in source:
            errors.append(f"{path.name}: missing visible probe execution")
        if not re.search(r"(?:viz\.|visualization\.|plot_)", source):
            errors.append(f"{path.name}: missing visible visualization call")
    return {"scope": "static documentation checks; no model execution", **counts,
            "success": not errors, "errors": errors}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Optional JSON verification record")
    args = parser.parse_args()
    result = check()
    message = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(message + "\n")
    print(message)
    raise SystemExit(0 if result["success"] else 1)


if __name__ == "__main__":
    main()

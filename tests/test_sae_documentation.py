"""Reader-facing method navigation, complete examples and relocated links."""
import ast
from pathlib import Path
import re
from urllib.parse import unquote, urlsplit

from vlmprobe.sae import list_methods

ROOT = Path(__file__).resolve().parents[1]
SAE = ROOT / "docs/methods/sae"


def without_code(text):
    return re.sub(r"```[^\n]*\n.*?```", "", text, flags=re.S)


def test_readme_links_to_individual_methods_without_a_separate_preview():
    text = (ROOT / "README.md").read_text()
    section = re.search(r"^### SAE\n(.*?)(?=^### |^## |\Z)", text, re.M | re.S)
    assert section, "SAE methods must remain visible in the Methods section"
    destinations = set(re.findall(r'href="([^"\s]+)"', section.group(1)))
    destinations.update(re.findall(r"\]\(([^\s)]+)\)", section.group(1)))
    assert "## SAE preview" not in text
    assert "fourth" not in text.lower()
    assert not (ROOT / "docs/sae").exists()
    for method in list_methods():
        destination = f"docs/methods/sae/{method['name']}.md"
        assert destination in destinations
        assert (ROOT / destination).is_file()


def test_every_method_has_a_complete_example_and_shared_resources():
    for method in list_methods():
        page = SAE / f"{method['name']}.md"
        text = page.read_text()
        assert re.search(r"^## .*API", text, flags=re.M), page
        assert "models/README.md" in text and "training/README.md" in text, page
        signature = re.search(r"```text\s*\n(.*?)```", text, flags=re.S).group(1)
        assert signature.startswith(f"method = probe.sae.{method['name']}("), page
        assert "history = method.fit(" in signature, page
        assert "**Usage sequence.**" in text, page
        blocks = re.findall(r"```python\s*\n(.*?)```", text, flags=re.S)
        assert blocks, page
        complete = [block for block in blocks if "Prober(" in block and ".fit(" in block and ".run(" in block]
        assert complete, f"{page}: requires one complete model/input/fit/run example"
        for block in blocks:
            ast.parse(block, filename=str(page))


def test_relative_document_links_and_example_figures_resolve():
    files = [ROOT / path for path in ("README.md", "docs/README.md", "docs/API.md",
             "docs/REFERENCES.md", "docs/methods/README.md", "docs/ARCHITECTURE.md")]
    files.extend(SAE.rglob("*.md"))
    errors = []
    for page in files:
        for destination in re.findall(r"\[[^\]]*\]\(([^\s)]+)\)", without_code(page.read_text())):
            url = urlsplit(destination.strip("<>"))
            if url.scheme or url.netloc or not url.path or url.path.startswith("/"):
                continue
            target = (page.parent / unquote(url.path)).resolve()
            if not target.exists():
                errors.append(f"{page.relative_to(ROOT)} -> {destination}")
    assert not errors, "Broken documentation links:\n" + "\n".join(errors[:50])

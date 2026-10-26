#!/usr/bin/env python3
"""Promote one reviewed library tree with repository-specific README footers."""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import tempfile
from pathlib import Path
from urllib.parse import urlsplit


PERSONAL_REPOSITORY = "gexinh/vlm-probe"
PREVIOUS_PERSONAL_REPOSITORY = "gexinh/vlm-prob"
LEGACY_PERSONAL_REPOSITORY = "gexinh/vlm-probing"
LAB_REPOSITORY = "ubc-tea/vlm-probe"
REPOSITORIES = (PERSONAL_REPOSITORY, PREVIOUS_PERSONAL_REPOSITORY,
                LEGACY_PERSONAL_REPOSITORY, LAB_REPOSITORY)
PRERELEASE_START = "<!-- vlmprobe:prerelease-link:start -->"
PRERELEASE_END = "<!-- vlmprobe:prerelease-link:end -->"
PRERELEASE_LINE = (
    "Pre-release version: "
    "[![Beta](https://img.shields.io/badge/Beta-gexinh%2Fvlm--probe-6f42c1?style=flat-square&logo=github)]"
    "(https://github.com/gexinh/vlm-probe)"
)
STABLE_START = "<!-- vlmprobe:stable-link:start -->"
STABLE_END = "<!-- vlmprobe:stable-link:end -->"
STABLE_LINE = (
    "Stable version: "
    "[![Stable](https://img.shields.io/badge/Stable-ubc--tea%2Fvlm--probe-2ea44f?style=flat-square&logo=github)]"
    "(https://github.com/ubc-tea/vlm-probe)"
)


def _without_release_footer(text: str) -> str:
    """Remove one exact managed trailing block without discarding user edits."""
    policies = ((PRERELEASE_START, PRERELEASE_LINE, PRERELEASE_END),
                (STABLE_START, STABLE_LINE, STABLE_END))
    present = [(start, line, end) for start, line, end in policies
               if start in text or end in text]
    marker_prefixes = re.findall(r"<!--\s*vlmprobe:(?:prerelease|stable)-link\b", text)
    if not present:
        if marker_prefixes:
            raise ValueError("README contains malformed managed release footer markers.")
        return text
    if len(present) != 1 or len(marker_prefixes) != 2:
        raise ValueError("README must have exactly one complete managed release footer.")
    start, line, end = present[0]
    if text.count(start) != 1 or text.count(end) != 1:
        raise ValueError("README must have exactly one complete managed release footer.")
    if text.index(end) < text.index(start):
        raise ValueError("README managed release footer markers must be in start/end order.")
    before, rest = text.split(start, 1)
    content, after = rest.split(end, 1)
    if before and not before.endswith(("\n", "\r")):
        raise ValueError("Managed release footer must start on its own line.")
    if after.strip():
        raise ValueError("Managed release footer must be at the end; preserve and review following text.")
    if content.replace("\r\n", "\n") != "\n" + line + "\n":
        raise ValueError("README contains unexpected managed release footer content; review it before replacing.")
    return before


def shared_readme_body(text: str) -> str:
    """Return the exact common body, normalized to the stable project URLs."""
    newline = "\r\n" if "\r\n" in text else "\n"
    return render_readme(_without_release_footer(text), LAB_REPOSITORY).rstrip("\r\n") + newline


def release_readme(text: str, repository: str) -> str:
    """Render the shared README for this repository and its one trailing link.

    This is the current publishing policy. ``render_readme`` remains the
    historical URL renderer, used to validate old mirror-only commits safely.
    Only a single managed block at the end may be removed; malformed markers or
    text following that block fail rather than silently deleting user content.
    """
    if repository not in REPOSITORIES:
        raise ValueError("Unknown repository for README rendering.")
    newline = "\r\n" if "\r\n" in text else "\n"
    # Source installs and documentation links must follow the selected release
    # repository. Strip the footer first so its opposite-repository destination
    # is kept separate from those own-project links.
    body = render_readme(_without_release_footer(text), repository).rstrip("\r\n") + newline
    footer = ((PRERELEASE_START, PRERELEASE_LINE, PRERELEASE_END)
              if repository == LAB_REPOSITORY else (STABLE_START, STABLE_LINE, STABLE_END))
    return body + newline + newline.join((*footer, ""))


def render_readme(text: str, repository: str) -> str:
    """Retarget this project's GitHub, raw-file and Pages URLs only.

    The previous personal names ``gexinh/vlm-prob`` and ``gexinh/vlm-probing``
    are accepted aliases, including when validating historical mirror commits.
    The standalone ``vlm-probing-report`` repository and third-party links
    remain unchanged. No badges, links or explanatory text are added.
    """
    return _render_readme(text, repository, retarget_pinned=True)


def _render_readme(text: str, repository: str, *, retarget_pinned: bool) -> str:
    if repository not in REPOSITORIES:
        raise ValueError("Unknown repository for README rendering.")
    owner, name = repository.split("/")

    def replace(match):
        url = urlsplit(match.group())
        for source in REPOSITORIES:
            source_owner, source_name = source.split("/")
            if url.netloc == "github.com":
                prefix, replacement = "/" + source, "/" + repository
                matches = (url.path in (prefix, prefix + ".git")
                           or url.path.startswith(prefix + "/")
                           or (retarget_pinned and url.path.startswith(prefix + ".git@")))
            elif url.netloc == "raw.githubusercontent.com":
                prefix, replacement = "/" + source, "/" + repository
                matches = url.path.startswith(prefix + "/")
            elif url.netloc == source_owner + ".github.io":
                prefix, replacement = "/" + source_name, "/" + name
                matches = url.path == prefix or url.path.startswith(prefix + "/")
                if matches:
                    url = url._replace(netloc=owner + ".github.io")
            else:
                continue
            if matches:
                return url._replace(path=replacement + url.path[len(prefix):]).geturl()
        return match.group()

    return re.sub(r"https?://[^\s<>\"'()]+", replace, text)


def git(repo: Path, *args: str, check: bool = True, env=None, input=None) -> subprocess.CompletedProcess:
    # In particular, status must not refresh/write the user's index merely to
    # perform the clean-tree check. Explicit temporary-index writes still work.
    return subprocess.run(["git", "--no-optional-locks", "-C", str(repo), *args], check=check, text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env, input=input)


def readme_at(repo: Path, commit: str) -> str:
    # Binary subprocess output avoids silently changing a CRLF README to LF.
    return subprocess.check_output(["git", "-C", str(repo), "show", f"{commit}:README.md"]).decode("utf-8")


def is_ancestor(repo: Path, ancestor: str, descendant: str) -> bool:
    result = git(repo, "merge-base", "--is-ancestor", ancestor, descendant, check=False)
    if result.returncode not in (0, 1):
        result.check_returncode()
    return result.returncode == 0


def changed_paths(repo: Path, before: str, after: str) -> set[str]:
    return set(git(repo, "diff", "--name-only", "-z", before, after).stdout.split("\0")) - {""}


def lab_tree(repo: Path, target: str) -> str:
    """Create the derived tree without touching the checkout or its index."""
    entry = git(repo, "ls-tree", target, "--", "README.md").stdout.split()
    if not entry or entry[0] not in ("100644", "100755") or entry[1] != "blob":
        raise ValueError("The reviewed commit must contain a regular README.md. Nothing was pushed.")
    rendered = release_readme(readme_at(repo, target), LAB_REPOSITORY)
    blob = git(repo, "hash-object", "-w", "--stdin", input=rendered).stdout.strip()
    with tempfile.TemporaryDirectory(prefix="vlmprobe-sync-index-") as directory:
        environment = {**os.environ, "GIT_INDEX_FILE": str(Path(directory) / "index")}
        git(repo, "read-tree", target, env=environment)
        git(repo, "update-index", "--cacheinfo", f"{entry[0]},{blob},README.md", env=environment)
        tree = git(repo, "write-tree", env=environment).stdout.strip()
    if changed_paths(repo, target, tree) - {"README.md"}:
        raise ValueError("Derived lab tree changes files other than README.md. Nothing was pushed.")
    return tree


def synchronize(repo: Path, remotes: tuple[str, str], branch: str, push: bool) -> str:
    git(repo, "check-ref-format", "--branch", branch)
    if remotes[0] == remotes[1]:
        raise ValueError("Choose two different remote names.")
    current = git(repo, "symbolic-ref", "--quiet", "--short", "HEAD").stdout.strip()
    if current != branch:
        raise ValueError(f"Check out {branch} before synchronizing it (current: {current}).")
    if push and git(repo, "status", "--porcelain").stdout.strip():
        raise ValueError("Commit or set aside local changes before publishing.")
    target = git(repo, "rev-parse", f"refs/heads/{branch}").stdout.strip()
    readme = readme_at(repo, target)
    if readme != release_readme(readme, PERSONAL_REPOSITORY):
        raise ValueError("Local README.md must use the personal repository URLs and the "
                         "managed personal stable footer; render it with release_readme(text, "
                         "PERSONAL_REPOSITORY) before committing. Nothing was pushed.")
    for remote in remotes:
        git(repo, "remote", "get-url", remote)
        git(repo, "fetch", remote, f"refs/heads/{branch}:refs/remotes/{remote}/{branch}")
    heads = {remote: git(repo, "rev-parse", f"refs/remotes/{remote}/{branch}").stdout.strip()
             for remote in remotes}
    personal, lab = remotes
    if not is_ancestor(repo, heads[personal], target):
        raise ValueError(f"{personal}/{branch} contains changes absent from local {branch}. "
                         "Merge and review those changes first. Nothing was pushed.")
    if not is_ancestor(repo, heads[lab], target):
        base = git(repo, "merge-base", heads[lab], target).stdout.strip()
        if changed_paths(repo, base, heads[lab]) - {"README.md"}:
            raise ValueError(f"{lab}/{branch} has library changes absent locally. "
                             "Merge and review those changes first. Nothing was pushed.")
        base_readme = readme_at(repo, base)
        # Previous sync versions left pinned pip VCS URLs on the personal
        # repository. Accept that exact historical rendering, not arbitrary
        # README edits, while always producing fully retargeted URLs now.
        historical_body = _without_release_footer(base_readme).rstrip("\r\n")
        newline = "\r\n" if "\r\n" in base_readme else "\n"
        historical_body += newline
        historical_renderings = {release_readme(base_readme, LAB_REPOSITORY),
                                 render_readme(historical_body, LAB_REPOSITORY),
                                 _render_readme(historical_body, LAB_REPOSITORY, retarget_pinned=False)}
        if readme_at(repo, heads[lab]) not in historical_renderings:
            raise ValueError(f"{lab}/{branch} has unexpected README edits. "
                             "Review them before synchronizing. Nothing was pushed.")
    # Both remote histories are approved before constructing or publishing a commit.
    tree = lab_tree(repo, target)
    already_current = (git(repo, "rev-parse", heads[lab] + "^{tree}").stdout.strip() == tree
                       and is_ancestor(repo, target, heads[lab]))
    print(f"Personal target: {target}")
    print(f"Lab tree: {tree}" + (" (already synchronized)" if already_current else " (README adapted)"))
    print("Shared README body and library match; personal links to stable, lab links to pre-release.")
    if not push:
        print("Check only. Add --push to update both remotes.")
        return target
    if already_current:
        lab_target = heads[lab]
    else:
        # Retain the old lab history and the reviewed personal commit. Neither
        # the working-tree files nor the user's index are checked out or changed.
        parents = [target] if is_ancestor(repo, heads[lab], target) else [heads[lab], target]
        parent_args = [arg for parent in parents for arg in ("-p", parent)]
        lab_target = git(repo, "commit-tree", tree, *parent_args, "-m",
                         "Promote reviewed library; adapt repository README footer").stdout.strip()
    targets = {personal: target, lab: lab_target}
    for remote, destination in targets.items():
        if heads[remote] != destination:
            # A concurrent update is rejected by Git; never force or delete refs.
            git(repo, "push", remote, f"{destination}:refs/heads/{branch}")
    mismatches = []
    for remote, destination in targets.items():
        line = git(repo, "ls-remote", remote, f"refs/heads/{branch}").stdout.strip()
        if not line or line.split()[0] != destination:
            mismatches.append(remote)
    if mismatches:
        raise ValueError("Remote verification failed for " + ", ".join(mismatches) +
                         ". Inspect both remotes before retrying.")
    if changed_paths(repo, target, lab_target) - {"README.md"}:
        raise ValueError("Remote library trees differ outside README.md; inspect both remotes.")
    print(f"Verified personal: {target}\nVerified lab: {lab_target}")
    print("All tracked paths except README.md are identical.")
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd(), help="Local Git checkout (default: current directory)")
    parser.add_argument("--personal", default="personal", help="Personal remote name")
    parser.add_argument("--lab", default="lab", help="Laboratory remote name")
    parser.add_argument("--branch", default="main")
    parser.add_argument("--push", action="store_true", help="Publish after checking both histories; default is check-only")
    args = parser.parse_args()
    try:
        synchronize(args.repo, (args.personal, args.lab), args.branch, args.push)
    except (ValueError, subprocess.CalledProcessError) as error:
        detail = (error.stderr or str(error)) if isinstance(error, subprocess.CalledProcessError) else str(error)
        if isinstance(detail, bytes):
            detail = detail.decode("utf-8", errors="replace")
        detail = detail.strip()
        parser.exit(1, detail + "\n")


if __name__ == "__main__":
    main()

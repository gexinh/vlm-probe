"""Exercise README rendering and actual remote writes in disposable Git repos."""
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

from tools.sync_repositories import (
    LAB_REPOSITORY, PERSONAL_REPOSITORY, PRERELEASE_END, PRERELEASE_LINE, PRERELEASE_START,
    STABLE_END, STABLE_LINE, STABLE_START, changed_paths, is_ancestor, readme_at,
    release_readme, render_readme, shared_readme_body, synchronize,
)


LEGACY_README = '''# vlmprobe
[Project page](https://gexinh.github.io/vlm-probe/)
[Code](https://github.com/gexinh/vlm-probe)
![Icon](https://raw.githubusercontent.com/gexinh/vlm-probe/main/icon.svg)
Install: git+https://github.com/gexinh/vlm-probe.git
[Report source](https://github.com/gexinh/vlm-probing-report)
[External](https://github.com/another/library)
'''

PERSONAL_README = release_readme(LEGACY_README, PERSONAL_REPOSITORY)


class RepositorySyncTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.repo = self.root / "checkout"
        self.run_git(self.root, "init", "--initial-branch=main", str(self.repo))
        self.configure_identity(self.repo)
        (self.repo / "README.md").write_text(PERSONAL_README)
        (self.repo / "version.py").write_text('__version__ = "0.3.0"\n')
        (self.repo / "report.pdf").write_bytes(b"%PDF-test\x00\xff\n")
        self.commit("base")
        for remote in ("personal", "lab"):
            bare = self.root / remote
            self.run_git(self.root, "init", "--bare", "--initial-branch=main", str(bare))
            self.run_git(self.repo, "remote", "add", remote, str(bare))
            self.run_git(self.repo, "push", remote, "main")

    @staticmethod
    def run_git(repo, *args):
        return subprocess.check_output(["git", "-C", str(repo), *args], stderr=subprocess.PIPE, text=True).strip()

    def configure_identity(self, repo):
        self.run_git(repo, "config", "user.name", "Sync test")
        self.run_git(repo, "config", "user.email", "sync@example.invalid")

    def commit(self, text):
        (self.repo / "file").write_text(text)
        self.run_git(self.repo, "add", "--all")
        self.run_git(self.repo, "commit", "-m", text)
        return self.run_git(self.repo, "rev-parse", "HEAD")

    def head(self, remote):
        return self.run_git(self.repo, "ls-remote", remote, "refs/heads/main").split()[0]

    def assert_matching_trees(self, personal, lab, personal_readme=PERSONAL_README):
        self.assertEqual(changed_paths(self.repo, personal, lab), {"README.md"})
        self.assertEqual(readme_at(self.repo, personal), personal_readme)
        expected = release_readme(personal_readme, LAB_REPOSITORY)
        self.assertEqual(readme_at(self.repo, lab), expected)
        self.assertEqual(expected.count("https://ubc-tea.github.io/vlm-probe/"), 1)
        self.assertNotIn("https://gexinh.github.io/vlm-prob", expected)
        self.assertIn("https://github.com/gexinh/vlm-probing-report", expected)
        self.assertEqual(release_readme(expected, PERSONAL_REPOSITORY), personal_readme)
        self.assertEqual(expected.count(PRERELEASE_START), 1)
        self.assertEqual(expected.count(PRERELEASE_LINE), 1)
        self.assertNotIn(PRERELEASE_START, personal_readme)
        self.assertEqual(personal_readme.count(STABLE_START), 1)
        self.assertEqual(personal_readme.count(STABLE_LINE), 1)
        self.assertNotIn(STABLE_START, expected)
        self.assertEqual(shared_readme_body(expected), shared_readme_body(personal_readme))

    def test_rendering_retargets_only_own_urls_and_preserves_readme_structure(self):
        expected = (LEGACY_README.replace("https://gexinh.github.io/vlm-probe/",
                                           "https://ubc-tea.github.io/vlm-probe/")
                    .replace("https://github.com/gexinh/vlm-probe)",
                             "https://github.com/ubc-tea/vlm-probe)")
                    .replace("https://github.com/gexinh/vlm-probe.git",
                             "https://github.com/ubc-tea/vlm-probe.git")
                    .replace("https://raw.githubusercontent.com/gexinh/vlm-probe/",
                             "https://raw.githubusercontent.com/ubc-tea/vlm-probe/"))
        self.assertEqual(render_readme(LEGACY_README, "ubc-tea/vlm-probe"), expected)
        self.assertEqual(render_readme(expected, "ubc-tea/vlm-probe"), expected)
        self.assertEqual(render_readme(expected, "gexinh/vlm-probe"), LEGACY_README)
        crlf = LEGACY_README.replace("\n", "\r\n")
        self.assertEqual(render_readme(crlf, "ubc-tea/vlm-probe"), expected.replace("\n", "\r\n"))

    def test_legacy_personal_alias_renders_to_current_repo_and_lab(self):
        for alias in ("vlm-prob", "vlm-probing"):
            with self.subTest(alias=alias):
                legacy = (LEGACY_README
                          .replace("gexinh.github.io/vlm-probe/", f"gexinh.github.io/{alias}/")
                          .replace("github.com/gexinh/vlm-probe)", f"github.com/gexinh/{alias})")
                          .replace("github.com/gexinh/vlm-probe.git", f"github.com/gexinh/{alias}.git")
                          .replace("raw.githubusercontent.com/gexinh/vlm-probe/",
                                   f"raw.githubusercontent.com/gexinh/{alias}/"))
                self.assertEqual(render_readme(legacy, "gexinh/vlm-probe"), LEGACY_README)
                self.assertEqual(render_readme(LEGACY_README, f"gexinh/{alias}"), legacy)
                self.assertEqual(render_readme(legacy, "ubc-tea/vlm-probe"),
                                 render_readme(LEGACY_README, "ubc-tea/vlm-probe"))
                self.assertIn("https://github.com/gexinh/vlm-probing-report", legacy)
                crlf = legacy.replace("\n", "\r\n")
                self.assertEqual(render_readme(crlf, "gexinh/vlm-probe"),
                                 LEGACY_README.replace("\n", "\r\n"))

    def test_renaming_keeps_paths_queries_fragments_and_other_repo_names(self):
        original = '''[Method](https://github.com/gexinh/vlm-probing/blob/main/docs/API.md?plain=1#sae)
[Page](https://gexinh.github.io/vlm-probing/sae-models/?method=topk#model)
[Report](https://github.com/gexinh/vlm-probing-report/blob/main/main.tex)
[Other](https://github.com/gexinh/vlm-prob-other)
'''
        expected = '''[Method](https://github.com/gexinh/vlm-probe/blob/main/docs/API.md?plain=1#sae)
[Page](https://gexinh.github.io/vlm-probe/sae-models/?method=topk#model)
[Report](https://github.com/gexinh/vlm-probing-report/blob/main/main.tex)
[Other](https://github.com/gexinh/vlm-prob-other)
'''
        self.assertEqual(render_readme(original, "gexinh/vlm-probe"), expected)

    def test_pinned_vcs_installs_retarget_repository_without_changing_refs(self):
        for source in ("vlm-prob", "vlm-probing", "vlm-probe"):
            for ref in ("main", "v1.0.1", "feature/example"):
                with self.subTest(source=source, ref=ref):
                    original = (f'Install: git+https://github.com/gexinh/{source}.git@{ref}'
                                '#subdirectory=library\n'
                                'Other: git+https://github.com/another/library.git@main\n'
                                'Report: git+https://github.com/gexinh/vlm-probing-report.git@main\n')
                    for target in ("gexinh/vlm-probe", "ubc-tea/vlm-probe"):
                        expected = original.replace(f"gexinh/{source}.git@", f"{target}.git@")
                        self.assertEqual(render_readme(original, target), expected)

    def test_release_readme_has_identical_body_and_idempotent_repository_footers(self):
        for newline in ("\n", "\r\n"):
            original = LEGACY_README.replace("\n", newline)
            personal = release_readme(original, PERSONAL_REPOSITORY)
            lab = release_readme(original, LAB_REPOSITORY)
            body = render_readme(original, LAB_REPOSITORY)
            self.assertEqual(shared_readme_body(personal), body)
            self.assertEqual(shared_readme_body(lab), body)
            self.assertNotIn(PRERELEASE_START, personal)
            self.assertNotIn(STABLE_START, lab)
            self.assertEqual(personal, render_readme(original, PERSONAL_REPOSITORY) + newline + newline.join(
                (STABLE_START, STABLE_LINE, STABLE_END, "")))
            self.assertEqual(lab, body + newline + newline.join(
                (PRERELEASE_START, PRERELEASE_LINE, PRERELEASE_END, "")))
            self.assertEqual(release_readme(lab, PERSONAL_REPOSITORY), personal)
            self.assertEqual(release_readme(lab, LAB_REPOSITORY), lab)
            self.assertEqual(release_readme(personal, PERSONAL_REPOSITORY), personal)
            self.assertEqual(release_readme(personal, LAB_REPOSITORY), lab)
            self.assertIn("(https://github.com/ubc-tea/vlm-probe)", personal)
            self.assertIn("(https://github.com/gexinh/vlm-probe)", lab)
            self.assertNotIn("\n##", lab[len(body):])
            self.assertNotIn("\n##", personal[len(body):])

    def test_release_readme_refuses_malformed_or_nontrailing_managed_blocks(self):
        body = shared_readme_body(PERSONAL_README)
        for start, line, end in ((STABLE_START, STABLE_LINE, STABLE_END),
                                 (PRERELEASE_START, PRERELEASE_LINE, PRERELEASE_END)):
            block = "\n".join((start, line, end, ""))
            for text in (body + start, body + end, body + end + start,
                         body + block + block, body + block + "Collaborator text\n",
                         body.rstrip() + block,
                         body + block.replace(line, line + "\nCollaborator text"),
                         body + block.replace(line, "Unknown link"),
                         body + block.replace("link:start -->", "link:start --")):
                for renderer in (lambda value: release_readme(value, PERSONAL_REPOSITORY),
                                 shared_readme_body):
                    with self.subTest(text=text, renderer=renderer), self.assertRaisesRegex(ValueError, "release footer"):
                        renderer(text)
        with self.assertRaisesRegex(ValueError, "exactly one complete"):
            release_readme(PERSONAL_README + "\n" + "\n".join(
                (PRERELEASE_START, PRERELEASE_LINE, PRERELEASE_END, "")), LAB_REPOSITORY)

    def test_footerless_readmes_and_old_lab_footers_migrate_without_body_edits(self):
        body = shared_readme_body(LEGACY_README)
        old_lab = body + "\n" + "\n".join(
            (PRERELEASE_START, PRERELEASE_LINE, PRERELEASE_END, ""))
        personal = release_readme(body, PERSONAL_REPOSITORY)
        self.assertEqual(release_readme(old_lab, PERSONAL_REPOSITORY), personal)
        self.assertEqual(release_readme(body, LAB_REPOSITORY), old_lab)
        self.assertEqual(release_readme(old_lab, LAB_REPOSITORY), old_lab)
        self.assertEqual(shared_readme_body(old_lab), body)

    def test_legacy_local_readme_requires_explicit_reviewed_migration(self):
        previous_personal, previous_lab = self.head("personal"), self.head("lab")
        (self.repo / "README.md").write_text(LEGACY_README)
        self.commit("legacy local URLs")
        with self.assertRaisesRegex(ValueError, "personal repository URLs.*Nothing was pushed"):
            synchronize(self.repo, ("personal", "lab"), "main", True)
        self.assertEqual(self.head("personal"), previous_personal)
        self.assertEqual(self.head("lab"), previous_lab)

    def test_footerless_local_readme_requires_explicit_reviewed_migration(self):
        previous_personal, previous_lab = self.head("personal"), self.head("lab")
        (self.repo / "README.md").write_text(shared_readme_body(PERSONAL_README))
        self.commit("old shared body without personal footer")
        with self.assertRaisesRegex(ValueError, "managed personal stable footer.*Nothing was pushed"):
            synchronize(self.repo, ("personal", "lab"), "main", True)
        self.assertEqual(self.head("personal"), previous_personal)
        self.assertEqual(self.head("lab"), previous_lab)

    def test_sync_migrates_footerless_personal_and_old_lab_beta_footer(self):
        body = shared_readme_body(LEGACY_README)
        (self.repo / "README.md").write_text(body)
        old_base = self.commit("historical personal without footer")
        self.run_git(self.repo, "push", "personal", "main")
        self.run_git(self.repo, "push", "lab", "main")
        (self.repo / "README.md").write_text(release_readme(body, LAB_REPOSITORY))
        self.run_git(self.repo, "add", "README.md")
        self.run_git(self.repo, "commit", "-m", "historical lab beta footer")
        old_lab = self.run_git(self.repo, "rev-parse", "HEAD")
        self.run_git(self.repo, "push", "lab", "main")
        self.run_git(self.repo, "reset", "--hard", old_base)
        (self.repo / "README.md").write_text(release_readme(body, PERSONAL_REPOSITORY))
        target = self.commit("reviewed personal stable footer")
        synchronize(self.repo, ("personal", "lab"), "main", True)
        self.assert_matching_trees(target, self.head("lab"))
        self.assertTrue(is_ancestor(self.repo, old_lab, self.head("lab")))

    def test_sync_after_repo_rename_accepts_historical_lab_readme(self):
        self.assert_historical_rename_sync("gexinh/vlm-probing")

    def test_sync_after_short_alias_rename_accepts_historical_lab_readme(self):
        self.assert_historical_rename_sync("gexinh/vlm-prob", pinned_install=True)

    def test_sync_migrates_existing_repository_urls_to_managed_footers(self):
        self.assert_historical_rename_sync(PERSONAL_REPOSITORY)

    def assert_historical_rename_sync(self, alias, *, pinned_install=False):
        # The lab's existing README was rendered from the previous repository
        # name; its history must still be retained when personal links change.
        personal_readme = LEGACY_README
        if pinned_install:
            personal_readme += "Pinned: git+https://github.com/gexinh/vlm-probe.git@v1.0.1\n"
        legacy = render_readme(personal_readme, alias)
        (self.repo / "README.md").write_text(legacy)
        old_base = self.commit("old personal name")
        self.run_git(self.repo, "push", "personal", "main")
        self.run_git(self.repo, "push", "lab", "main")
        old_lab_readme = render_readme(legacy, "ubc-tea/vlm-probe")
        if pinned_install:
            # The former renderer did not recognize a .git@ref URL, so the
            # historical lab README retained the personal pinned install.
            old_lab_readme = old_lab_readme.replace("ubc-tea/vlm-probe.git@", f"{alias}.git@")
        (self.repo / "README.md").write_text(old_lab_readme)
        self.run_git(self.repo, "add", "README.md")
        self.run_git(self.repo, "commit", "-m", "historical lab README")
        old_lab = self.run_git(self.repo, "rev-parse", "HEAD")
        self.run_git(self.repo, "push", "lab", "main")
        self.run_git(self.repo, "reset", "--hard", old_base)
        personal_readme = release_readme(personal_readme, PERSONAL_REPOSITORY)
        (self.repo / "README.md").write_text(personal_readme)
        target = self.commit("new shared README policy")
        synchronize(self.repo, ("personal", "lab"), "main", True)
        new_lab = self.head("lab")
        self.assert_matching_trees(target, new_lab, personal_readme)
        self.assertTrue(is_ancestor(self.repo, old_lab, new_lab))
        self.assertTrue(is_ancestor(self.repo, target, new_lab))
        synchronize(self.repo, ("personal", "lab"), "main", True)
        self.assertEqual(self.head("personal"), target)
        self.assertEqual(self.head("lab"), new_lab)

    def test_check_does_not_push_and_push_only_changes_lab_readme(self):
        previous = self.head("personal")
        target = self.commit("next")
        # A status check normally refreshes cached timestamps in the index.
        # The synchronizer must leave even that metadata untouched.
        path = self.repo / "file"
        stat = path.stat()
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 2_000_000_000))
        index_before = (self.repo / ".git" / "index").read_bytes()
        synchronize(self.repo, ("personal", "lab"), "main", False)
        self.assertEqual(self.head("personal"), previous)
        self.assertEqual(self.head("lab"), previous)
        synchronize(self.repo, ("personal", "lab"), "main", True)
        self.assertEqual(self.head("personal"), target)
        self.assert_matching_trees(target, self.head("lab"))
        self.assertTrue(is_ancestor(self.repo, target, self.head("lab")))
        self.assertEqual(self.run_git(self.repo, "rev-parse", "HEAD"), target)
        self.assertEqual((self.repo / "README.md").read_text(), PERSONAL_README)
        self.assertEqual((self.repo / "file").read_text(), "next")
        self.assertEqual((self.repo / ".git" / "index").read_bytes(), index_before)
        self.assertEqual(self.run_git(self.repo, "status", "--porcelain"), "")

    def test_repeated_updates_preserve_lab_history_and_unchanged_sync_is_noop(self):
        self.commit("first")
        synchronize(self.repo, ("personal", "lab"), "main", True)
        old_lab = self.head("lab")
        updated_readme = release_readme(shared_readme_body(PERSONAL_README) +
                                       "\nUpdated usage documentation\n", PERSONAL_REPOSITORY)
        (self.repo / "README.md").write_text(updated_readme)
        target = self.commit("second")
        synchronize(self.repo, ("personal", "lab"), "main", True)
        new_lab = self.head("lab")
        self.assertTrue(is_ancestor(self.repo, old_lab, new_lab))
        self.assertTrue(is_ancestor(self.repo, target, new_lab))
        self.assert_matching_trees(target, new_lab, updated_readme)
        parents = self.run_git(self.repo, "show", "-s", "--format=%P", new_lab).split()
        self.assertEqual(parents, [old_lab, target])
        synchronize(self.repo, ("personal", "lab"), "main", True)
        self.assertEqual(self.head("personal"), target)
        self.assertEqual(self.head("lab"), new_lab)

    def test_personal_beta_push_waits_for_explicit_lab_promotion(self):
        self.commit("reviewed")
        synchronize(self.repo, ("personal", "lab"), "main", True)
        stable = self.head("lab")
        beta = self.commit("personal beta")
        self.run_git(self.repo, "push", "personal", "main")
        self.assertEqual(self.head("personal"), beta)
        self.assertEqual(self.head("lab"), stable)
        synchronize(self.repo, ("personal", "lab"), "main", False)
        self.assertEqual(self.head("lab"), stable)
        synchronize(self.repo, ("personal", "lab"), "main", True)
        promoted = self.head("lab")
        self.assertTrue(is_ancestor(self.repo, stable, promoted))
        self.assertTrue(is_ancestor(self.repo, beta, promoted))
        self.assert_matching_trees(beta, promoted)

    def test_lab_footer_cannot_be_published_to_personal_readme(self):
        self.commit("first")
        synchronize(self.repo, ("personal", "lab"), "main", True)
        personal, lab = self.head("personal"), self.head("lab")
        self.run_git(self.repo, "reset", "--hard", lab)
        self.commit("edit-from-lab-checkout")
        with self.assertRaisesRegex(ValueError, "must use the personal repository URLs and the"):
            synchronize(self.repo, ("personal", "lab"), "main", True)
        self.assertEqual(self.head("personal"), personal)
        self.assertEqual(self.head("lab"), lab)

    def test_lab_library_divergence_prevents_writes_to_both_remotes(self):
        previous = self.head("personal")
        lab_only = self.commit("lab-only")
        self.run_git(self.repo, "push", "lab", "main")
        self.run_git(self.repo, "reset", "--hard", previous)
        self.commit("local-only")
        with self.assertRaisesRegex(ValueError, "Nothing was pushed"):
            synchronize(self.repo, ("personal", "lab"), "main", True)
        self.assertEqual(self.head("personal"), previous)
        self.assertEqual(self.head("lab"), lab_only)

    def test_personal_divergence_prevents_writes_to_both_remotes(self):
        previous = self.head("lab")
        personal_only = self.commit("personal-only")
        self.run_git(self.repo, "push", "personal", "main")
        self.run_git(self.repo, "reset", "--hard", previous)
        self.commit("local-only")
        with self.assertRaisesRegex(ValueError, "Nothing was pushed"):
            synchronize(self.repo, ("personal", "lab"), "main", True)
        self.assertEqual(self.head("personal"), personal_only)
        self.assertEqual(self.head("lab"), previous)

    def test_unexpected_lab_readme_edits_are_not_silently_overwritten(self):
        self.commit("first")
        synchronize(self.repo, ("personal", "lab"), "main", True)
        previous_personal = self.head("personal")
        other = self.root / "collaborator"
        self.run_git(self.root, "clone", str(self.root / "lab"), str(other))
        self.configure_identity(other)
        with (other / "README.md").open("a") as stream:
            stream.write("\nImportant collaborator edit\n")
        self.run_git(other, "add", "README.md")
        self.run_git(other, "commit", "-m", "Edit README")
        self.run_git(other, "push", "origin", "main")
        edited_lab = self.head("lab")
        self.commit("second")
        with self.assertRaisesRegex(ValueError, "unexpected README edits.*Nothing was pushed"):
            synchronize(self.repo, ("personal", "lab"), "main", True)
        self.assertEqual(self.head("personal"), previous_personal)
        self.assertEqual(self.head("lab"), edited_lab)

    def test_dirty_tree_is_not_published(self):
        previous = self.head("lab")
        (self.repo / "file").write_text("unreviewed")
        with self.assertRaisesRegex(ValueError, "local changes"):
            synchronize(self.repo, ("personal", "lab"), "main", True)
        self.assertEqual(self.head("lab"), previous)


if __name__ == "__main__":
    unittest.main()

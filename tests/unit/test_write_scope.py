"""写作用域公共逻辑：非空产物验收、新鲜度与写时路径校验。"""

from __future__ import annotations

import os

from so_agent.runtime.write_scope import (
    entry_satisfied,
    missing_scope_entries,
    normalize_entry,
    path_allowed,
)


class TestNormalizeEntry:
    def test_strips_dot_and_backslashes(self):
        assert normalize_entry(".\\out\\a.txt") == "out/a.txt"

    def test_rejects_escape_and_absolute(self):
        assert normalize_entry("../a.txt") is None
        assert normalize_entry("/etc/passwd") is None
        assert normalize_entry("C:\\Windows\\a.txt") is None
        assert normalize_entry("   ") is None


class TestEntrySatisfied:
    def test_empty_file_does_not_count(self, tmp_path):
        (tmp_path / "a.txt").write_text("", encoding="utf-8")
        assert entry_satisfied(tmp_path, "a.txt") is False

    def test_non_empty_file_counts(self, tmp_path):
        (tmp_path / "a.txt").write_text("x", encoding="utf-8")
        assert entry_satisfied(tmp_path, "a.txt") is True

    def test_directory_requires_non_empty_file(self, tmp_path):
        (tmp_path / "out").mkdir()
        assert entry_satisfied(tmp_path, "out") is False
        (tmp_path / "out" / "a.txt").write_text("x", encoding="utf-8")
        assert entry_satisfied(tmp_path, "out") is True

    def test_glob_requires_a_match(self, tmp_path):
        (tmp_path / "out").mkdir()
        assert entry_satisfied(tmp_path, "out/*.csv") is False
        (tmp_path / "out" / "a.csv").write_text("x", encoding="utf-8")
        assert entry_satisfied(tmp_path, "out/*.csv") is True

    def test_since_rejects_old_artifacts(self, tmp_path):
        target = tmp_path / "a.txt"
        target.write_text("x", encoding="utf-8")
        os.utime(target, (1_000_000, 1_000_000))
        assert entry_satisfied(tmp_path, "a.txt") is True
        assert entry_satisfied(tmp_path, "a.txt", since=2_000_000) is False


class TestMissingScopeEntries:
    def test_preserves_order_and_deduplicates(self, tmp_path):
        (tmp_path / "a.txt").write_text("x", encoding="utf-8")
        assert missing_scope_entries(tmp_path, ["b.txt", "a.txt", "b.txt"]) == [
            "b.txt"
        ]


class TestPathAllowed:
    def test_empty_scope_allows_everything(self):
        assert path_allowed([], "any/path.txt") is True

    def test_exact_directory_and_glob(self):
        scope = ["report.md", "out", "logs/*.json"]
        assert path_allowed(scope, "report.md") is True
        assert path_allowed(scope, "out/deep/a.csv") is True
        assert path_allowed(scope, "logs/run.json") is True
        assert path_allowed(scope, "out2/a.csv") is False
        assert path_allowed(scope, "logs/run.txt") is False

    def test_escape_is_never_allowed(self):
        assert path_allowed(["out"], "../outside.txt") is False
        assert path_allowed(["*"], "C:\\absolute.txt") is False

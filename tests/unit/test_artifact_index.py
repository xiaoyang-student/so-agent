"""沙箱产物索引：只列浅层文件元数据，不读取内容。"""

from __future__ import annotations

from pathlib import Path

from so_agent.runtime.artifact_index import build_artifact_index


def test_empty_directory_returns_empty_text(tmp_path: Path):
    assert build_artifact_index(tmp_path) == ""


def test_lists_shallow_files_sorted_without_content(tmp_path: Path):
    (tmp_path / "b.csv").write_text("secret,data\n1,2\n", encoding="utf-8")
    (tmp_path / "a.txt").write_text("hello", encoding="utf-8")
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "inner.txt").write_text("inner", encoding="utf-8")

    index = build_artifact_index(tmp_path)

    assert "当前沙箱已有文件" in index
    assert "- a.txt (5 bytes" in index
    assert "- b.csv (" in index
    assert index.index("a.txt") < index.index("b.csv")
    assert "inner.txt" not in index
    assert "secret,data" not in index


def test_max_entries_marks_remaining(tmp_path: Path):
    for name in ("a.txt", "b.txt", "c.txt"):
        (tmp_path / name).write_text("x", encoding="utf-8")

    index = build_artifact_index(tmp_path, max_entries=2)

    assert "- a.txt" in index
    assert "- b.txt" in index
    assert "- c.txt" not in index
    assert "其余 1 个文件未列出" in index


def test_missing_directory_returns_empty_text(tmp_path: Path):
    assert build_artifact_index(tmp_path / "missing") == ""

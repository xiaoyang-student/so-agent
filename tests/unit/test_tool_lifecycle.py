"""生成工具生命周期：弃用、归档、pin 与活跃保护。"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone

from so_agent.models import GeneratedToolManifest
from so_agent.runtime.tool_lifecycle import sweep_generated_tools

NOW = datetime(2026, 11, 2, 12, 0, tzinfo=timezone.utc)


def _manifest(name: str, **overrides) -> GeneratedToolManifest:
    data = {
        "tool_id": f"tool-{name}",
        "tool_name": name,
        "entry_file": f"{name}.py",
        "created_by_task": "task-1",
        "review_status": "approved",
        "verified": True,
        "safety_status": "passed",
        "created_at": (NOW - timedelta(days=90)).isoformat(),
        "last_used_at": (NOW - timedelta(days=60)).isoformat(),
    }
    data.update(overrides)
    return GeneratedToolManifest(**data)


def _write_store(tmp_path, manifests):
    tools_dir = tmp_path / "generated"
    tools_dir.mkdir(parents=True)
    for manifest in manifests:
        (tools_dir / manifest.entry_file).write_text("def run():\n    return 1\n")
    (tools_dir / "registry.json").write_text(
        json.dumps([item.model_dump(mode="json") for item in manifests]),
        encoding="utf-8",
    )
    return tools_dir


def _load(tools_dir):
    raw = json.loads((tools_dir / "registry.json").read_text(encoding="utf-8"))
    return {item["tool_name"]: item for item in raw}


def test_unused_approved_tool_is_deprecated_then_archived(tmp_path):
    tools_dir = _write_store(tmp_path, [_manifest("old_tool")])
    report = sweep_generated_tools(
        tools_dir=tools_dir,
        archive_dir=tmp_path / "archive",
        now=NOW,
        approved_ttl_days=30,
        deprecation_grace_days=7,
    )
    assert report.deprecated == ["old_tool"]
    assert "deprecated_at" in _load(tools_dir)["old_tool"]

    later = NOW + timedelta(days=8)
    report2 = sweep_generated_tools(
        tools_dir=tools_dir,
        archive_dir=tmp_path / "archive",
        now=later,
        approved_ttl_days=30,
        deprecation_grace_days=7,
    )
    assert report2.archived == ["old_tool"]
    assert _load(tools_dir) == {}
    assert not (tools_dir / "old_tool.py").exists()
    assert (tmp_path / "archive" / "index.json").is_file()


def test_pinned_and_active_tools_are_protected(tmp_path):
    tools_dir = _write_store(
        tmp_path,
        [
            _manifest("pinned_tool", pinned=True),
            _manifest("active_tool"),
        ],
    )
    report = sweep_generated_tools(
        tools_dir=tools_dir,
        archive_dir=tmp_path / "archive",
        now=NOW,
        approved_ttl_days=30,
        active_tools={"active_tool"},
    )
    assert report.deprecated == []
    assert report.archived == []


def test_old_candidate_is_archived(tmp_path):
    candidate = _manifest(
        "candidate_tool",
        review_status="pending",
        verified=False,
        safety_status="unchecked",
        created_at=(NOW - timedelta(days=30)).isoformat(),
        last_used_at="",
    )
    tools_dir = _write_store(tmp_path, [candidate])
    report = sweep_generated_tools(
        tools_dir=tools_dir,
        archive_dir=tmp_path / "archive",
        now=NOW,
        candidate_ttl_days=7,
    )
    assert report.archived == ["candidate_tool"]
    assert _load(tools_dir) == {}


def test_archive_retention_purges_old_files(tmp_path):
    tools_dir = _write_store(
        tmp_path,
        [
            _manifest(
                "dead_tool",
                review_status="pending",
                verified=False,
                created_at=(NOW - timedelta(days=30)).isoformat(),
            )
        ],
    )
    archive_dir = tmp_path / "archive"
    sweep_generated_tools(
        tools_dir=tools_dir,
        archive_dir=archive_dir,
        now=NOW,
        candidate_ttl_days=7,
    )
    archived_dirs = [item for item in archive_dir.iterdir() if item.is_dir()]
    assert len(archived_dirs) == 1

    old = NOW - timedelta(days=200)
    for item in archived_dirs:
        stamp = old.timestamp()
        os.utime(item, (stamp, stamp))
    report = sweep_generated_tools(
        tools_dir=tools_dir,
        archive_dir=archive_dir,
        now=NOW,
        archive_retention_days=90,
    )
    assert report.purged
    assert not archived_dirs[0].exists()
    assert (archive_dir / "index.json").is_file()


def test_missing_registry_is_noop(tmp_path):
    report = sweep_generated_tools(
        tools_dir=tmp_path / "generated",
        archive_dir=tmp_path / "archive",
        now=NOW,
    )
    assert report.swept is False
    assert report.errors == []

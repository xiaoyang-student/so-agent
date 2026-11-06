"""旧沙箱工具库迁移到主控侧 ToolStore。"""

from __future__ import annotations

import json
from pathlib import Path

from so_agent.models import GeneratedToolManifest
from so_agent.runtime.tool_store_migration import migrate_legacy_generated_tools


def _write_legacy(
    legacy: Path,
    name: str,
    *,
    source: str = "def run(**kwargs):\n    return {}\n",
    **overrides,
) -> GeneratedToolManifest:
    legacy.mkdir(parents=True, exist_ok=True)
    (legacy / f"{name}.py").write_text(source, encoding="utf-8")
    (legacy / f"{name}.mcp_schema.json").write_text(
        json.dumps(
            {"name": name, "description": "legacy", "inputSchema": {"type": "object"}}
        ),
        encoding="utf-8",
    )
    data = {
        "tool_id": f"tool-{name}",
        "tool_name": name,
        "entry_file": f"{name}.py",
        "created_by_task": "old-task",
        "review_status": "approved",
        "verified": True,
        "safety_status": "passed",
    }
    data.update(overrides)
    manifest = GeneratedToolManifest(**data)
    registry_path = legacy / "registry.json"
    entries = (
        json.loads(registry_path.read_text(encoding="utf-8"))
        if registry_path.is_file()
        else []
    )
    entries.append(manifest.model_dump(mode="json"))
    registry_path.write_text(
        json.dumps(entries, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def test_migrates_sources_manifest_and_computes_hash(tmp_path):
    legacy = tmp_path / "sandbox" / "generated_tools"
    target = tmp_path / "tools" / "generated"
    _write_legacy(legacy, "old_calc")

    report = migrate_legacy_generated_tools(legacy, target)

    assert report.migrated == ["old_calc"]
    assert (target / "old_calc.py").is_file()
    assert (target / "old_calc.mcp_schema.json").is_file()
    entries = json.loads((target / "registry.json").read_text(encoding="utf-8"))
    assert entries[0]["tool_name"] == "old_calc"
    assert len(entries[0]["content_hash"]) == 64
    # 旧目录保留（只复制不移动）
    assert (legacy / "old_calc.py").is_file()


def test_migration_is_idempotent(tmp_path):
    legacy = tmp_path / "legacy"
    target = tmp_path / "target"
    _write_legacy(legacy, "old_calc")

    first = migrate_legacy_generated_tools(legacy, target)
    second = migrate_legacy_generated_tools(legacy, target)

    assert first.migrated == ["old_calc"]
    assert second.migrated == []
    assert second.skipped == ["old_calc"]
    entries = json.loads((target / "registry.json").read_text(encoding="utf-8"))
    assert len(entries) == 1


def test_existing_target_tool_wins(tmp_path):
    legacy = tmp_path / "legacy"
    target = tmp_path / "target"
    _write_legacy(legacy, "shared_tool", source="def run():\n    return 'old'\n")
    _write_legacy(target, "shared_tool", source="def run():\n    return 'new'\n")

    report = migrate_legacy_generated_tools(legacy, target)

    assert report.migrated == []
    assert report.skipped == ["shared_tool"]
    assert "'new'" in (target / "shared_tool.py").read_text(encoding="utf-8")


def test_hash_mismatch_is_rejected(tmp_path):
    legacy = tmp_path / "legacy"
    target = tmp_path / "target"
    _write_legacy(legacy, "tampered", content_hash="0" * 64)

    report = migrate_legacy_generated_tools(legacy, target)

    assert report.migrated == []
    assert any("哈希不匹配" in error for error in report.errors)
    assert not (target / "tampered.py").exists()


def test_same_source_and_target_is_noop(tmp_path):
    legacy = tmp_path / "same"
    _write_legacy(legacy, "old_calc")

    report = migrate_legacy_generated_tools(legacy, legacy)

    assert report.migrated == []
    assert report.errors == []

"""工具清单并发写：合并写不丢条目、使用统计跨实例精确累加。"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from so_agent.models import GeneratedToolManifest
from so_agent.runtime.tool_registry import ToolRegistry
from so_agent.runtime.tool_store import (
    ToolStoreLockError,
    registry_write_lock,
)


def _manifest(tools_dir: Path, name: str, **overrides) -> GeneratedToolManifest:
    tools_dir.mkdir(parents=True, exist_ok=True)
    (tools_dir / f"{name}.py").write_text(
        "def run(**kwargs):\n    return {}\n", encoding="utf-8"
    )
    data = {
        "tool_id": f"tool-{name}",
        "tool_name": name,
        "entry_file": f"{name}.py",
        "created_by_task": "task-1",
        "review_status": "approved",
        "verified": True,
        "safety_status": "passed",
    }
    data.update(overrides)
    return GeneratedToolManifest(**data)


def _registry_names(path: Path) -> set[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {item["tool_name"] for item in payload}


def test_save_merges_entries_written_by_other_instance(tmp_path):
    tools_dir = tmp_path / "generated"
    registry_path = tools_dir / "registry.json"
    first = ToolRegistry(generated_tools_dir=tools_dir)
    second = ToolRegistry(generated_tools_dir=tools_dir)
    first.register_generated_tool(_manifest(tools_dir, "tool_a"))
    second.register_generated_tool(_manifest(tools_dir, "tool_b"))

    first.save_generated_registry(registry_path)
    second.save_generated_registry(registry_path)

    assert _registry_names(registry_path) == {"tool_a", "tool_b"}


def test_record_tool_use_accumulates_across_stale_instances(tmp_path):
    tools_dir = tmp_path / "generated"
    registry_path = tools_dir / "registry.json"
    writer = ToolRegistry(generated_tools_dir=tools_dir)
    writer.register_generated_tool(_manifest(tools_dir, "calc"))
    writer.save_generated_registry(registry_path)

    stale = ToolRegistry(generated_tools_dir=tools_dir)
    stale.load_generated_registry(registry_path)
    assert stale.get_manifest("calc").use_count == 0

    writer.record_tool_use("calc", True)
    stale.record_tool_use("calc", False)

    reloaded = ToolRegistry(generated_tools_dir=tools_dir)
    reloaded.load_generated_registry(registry_path)
    manifest = reloaded.get_manifest("calc")
    assert manifest.use_count == 2
    assert manifest.failure_count == 1
    assert manifest.last_used_at


def test_registry_write_lock_is_exclusive(tmp_path):
    registry_path = tmp_path / "registry.json"
    with registry_write_lock(registry_path):
        with pytest.raises(ToolStoreLockError):
            with registry_write_lock(registry_path, timeout=0.1):
                pass


def test_registry_write_lock_takes_over_stale_lock(tmp_path):
    registry_path = tmp_path / "registry.json"
    lock_path = registry_path.with_name(f"{registry_path.name}.lock")
    lock_path.write_text("stale", encoding="ascii")
    old = time.time() - 3600
    os.utime(lock_path, (old, old))

    with registry_write_lock(registry_path, timeout=1.0):
        assert lock_path.is_file()
    assert not lock_path.is_file()


def _registry_with_tool(
    tmp_path: Path, *, threshold: int
) -> tuple[ToolRegistry, Path]:
    tools_dir = tmp_path / "generated"
    registry = ToolRegistry(
        generated_tools_dir=tools_dir,
        failure_quarantine_threshold=threshold,
    )
    registry.register_generated_tool(_manifest(tools_dir, "calc"))
    registry.save_generated_registry(tools_dir / "registry.json")
    return registry, tools_dir / "registry.json"


def test_consecutive_failures_quarantine_and_block_safety_clearance(tmp_path):
    registry, registry_path = _registry_with_tool(tmp_path, threshold=2)

    registry.record_tool_use("calc", False)
    assert registry.get_manifest("calc").safety_status == "passed"

    registry.record_tool_use("calc", False)
    manifest = registry.get_manifest("calc")
    assert manifest.safety_status == "quarantined"
    assert manifest.consecutive_failures == 2
    assert registry.is_safety_cleared("calc") is False

    reloaded = ToolRegistry(generated_tools_dir=tmp_path / "generated")
    reloaded.load_generated_registry(registry_path)
    assert reloaded.get_manifest("calc").safety_status == "quarantined"


def test_success_resets_consecutive_failure_counter(tmp_path):
    registry, _ = _registry_with_tool(tmp_path, threshold=2)

    registry.record_tool_use("calc", False)
    registry.record_tool_use("calc", True)
    registry.record_tool_use("calc", False)

    manifest = registry.get_manifest("calc")
    assert manifest.consecutive_failures == 1
    assert manifest.safety_status == "passed"


def test_clear_quarantine_restores_tool_and_persists(tmp_path):
    registry, registry_path = _registry_with_tool(tmp_path, threshold=1)
    registry.record_tool_use("calc", False)
    assert registry.get_manifest("calc").safety_status == "quarantined"

    assert registry.clear_quarantine("calc") is True
    assert registry.get_manifest("calc").safety_status == "passed"
    assert registry.get_manifest("calc").consecutive_failures == 0

    reloaded = ToolRegistry(generated_tools_dir=tmp_path / "generated")
    reloaded.load_generated_registry(registry_path)
    assert reloaded.get_manifest("calc").safety_status == "passed"
    assert reloaded.is_safety_cleared("calc") is True
    assert registry.clear_quarantine("calc") is False

"""工具注册表：使用统计、安全准入判定与外部 MCP 注册。"""

from __future__ import annotations

import json

import pytest

from so_agent.mcp.protocol import MCPToolSchema
from so_agent.mcp.server import MCPToolServer
from so_agent.models import GeneratedToolManifest
from so_agent.runtime.tool_registry import ToolRegistry, ToolRegistryError


def _manifest(tmp_path, **overrides) -> GeneratedToolManifest:
    tools_dir = tmp_path / "generated"
    tools_dir.mkdir(exist_ok=True)
    (tools_dir / "calc.py").write_text("def run(a, b):\n    return a + b\n")
    data = {
        "tool_id": "tool-calc",
        "tool_name": "calc",
        "entry_file": "calc.py",
        "created_by_task": "task-1",
        "review_status": "approved",
        "verified": True,
        "safety_status": "passed",
    }
    data.update(overrides)
    return GeneratedToolManifest(**data)


def test_generated_tool_requires_safety_and_verification(tmp_path):
    registry = ToolRegistry(generated_tools_dir=tmp_path / "generated")
    registry.register_generated_tool(_manifest(tmp_path))
    assert registry.is_approved("calc") is True
    assert registry.is_safety_cleared("calc") is True

    unchecked = registry.get_manifest("calc").model_copy(
        update={"safety_status": "unchecked"}
    )
    registry.register_generated_tool(unchecked)
    assert registry.is_approved("calc") is True
    assert registry.is_safety_cleared("calc") is False


def test_usage_statistics_are_persisted(tmp_path):
    tools_dir = tmp_path / "generated"
    registry = ToolRegistry(generated_tools_dir=tools_dir)
    registry.register_generated_tool(_manifest(tmp_path))

    registry.record_tool_use("calc", True)
    registry.record_tool_use("calc", False)

    reloaded = json.loads((tools_dir / "registry.json").read_text(encoding="utf-8"))
    entry = reloaded[0]
    assert entry["use_count"] == 2
    assert entry["failure_count"] == 1
    assert entry["last_used_at"]


def test_external_tool_registration_and_removal(tmp_path):
    registry = ToolRegistry(generated_tools_dir=tmp_path / "generated")
    schema = MCPToolSchema(
        name="srv.echo", description="echo", inputSchema={"type": "object"}
    )
    server = MCPToolServer(name="srv.echo")
    registry.register_external_tool(
        "srv.echo", "echo", schema, server, server_name="srv"
    )
    assert registry.get_tool("srv.echo")["source"] == "external_mcp"
    assert registry.is_approved("srv.echo") is True
    assert registry.is_safety_cleared("srv.echo") is True
    assert registry.list_external_tools() == ["srv.echo"]
    assert registry.unregister_external_tool("srv.echo") is True
    assert registry.get_tool("srv.echo") is None


def test_external_tool_cannot_shadow_builtin(tmp_path):
    registry = ToolRegistry(generated_tools_dir=tmp_path / "generated")
    schema = MCPToolSchema(name="code_agent", description="x")
    server = MCPToolServer(name="code_agent")
    with pytest.raises(ToolRegistryError):
        registry.register_external_tool(
            "code_agent", "x", schema, server, server_name="srv"
        )

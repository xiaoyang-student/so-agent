"""单元测试：生成工具落盘位置与源码哈希完整性。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from agents.tool_context import ToolContext

from so_agent.config import Settings
from so_agent.context import ProjectContext
from so_agent.integrity import sha256_file
from so_agent.mcp.adapter import generated_tool_to_mcp_server
from so_agent.mcp.protocol import METHOD_TOOLS_CALL, MCPRequest
from so_agent.models import GeneratedToolManifest
from so_agent.runtime.tool_registry import ToolRegistry, ToolRegistryError
from so_agent.tool_packages.code_agent.agent import build_code_tools

TOOL_SOURCE = "def run(value):\n    return {'double': value * 2}\n"


async def _write_candidate(context: ProjectContext, tool_store: Path) -> dict:
    tools = build_code_tools(
        context,
        settings=Settings(_env_file=None, tool_store_dir=str(tool_store)),
    )
    arguments = json.dumps(
        {
            "tool_name": "demo_tool",
            "code": TOOL_SOURCE,
            "manifest_data": json.dumps(
                {"description": "demo", "created_by_task": "task-hash"}
            ),
        }
    )
    ctx = ToolContext(
        context=None,
        tool_name="write_generated_tool",
        tool_call_id="call-1",
        tool_arguments=arguments,
    )
    return json.loads(await tools["write_generated_tool"].on_invoke_tool(ctx, arguments))


def _manifest(entry_file: str, content_hash: str) -> GeneratedToolManifest:
    return GeneratedToolManifest(
        tool_id="tool-1",
        tool_name="demo_tool",
        entry_file=entry_file,
        created_by_task="task-hash",
        review_status="approved",
        content_hash=content_hash,
    )


class TestToolStoreDefault:
    async def test_candidate_written_into_tool_store(self, tmp_path):
        context = ProjectContext(sandbox_dir=tmp_path / "sandbox", project_name="hash")
        tool_store = tmp_path / "tools"
        result = await _write_candidate(context, tool_store)
        assert result["status"] == "ok"

        generated_dir = tool_store / "generated"
        assert (generated_dir / "demo_tool.py").is_file()
        assert (generated_dir / "demo_tool.mcp_schema.json").is_file()
        assert not (context.sandbox_dir / "generated_tools").exists()

        registry_payload = json.loads(
            (generated_dir / "registry.json").read_text(encoding="utf-8")
        )
        assert registry_payload[0]["content_hash"] == sha256_file(
            generated_dir / "demo_tool.py"
        )


class TestHashIntegrity:
    def test_tampered_file_is_rejected_at_registration(self, tmp_path):
        generated_dir = tmp_path / "gen"
        generated_dir.mkdir()
        entry = generated_dir / "demo_tool.py"
        entry.write_text(TOOL_SOURCE, encoding="utf-8")
        manifest = _manifest("demo_tool.py", sha256_file(entry))

        registry = ToolRegistry(generated_tools_dir=generated_dir)
        registry.register_generated_tool(manifest)

        entry.write_text("def run(v):\n    return 'tampered'\n", encoding="utf-8")
        with pytest.raises(ToolRegistryError, match="篡改"):
            registry.register_generated_tool(manifest)

    def test_missing_file_is_rejected_at_registration(self, tmp_path):
        generated_dir = tmp_path / "gen"
        generated_dir.mkdir()
        manifest = _manifest("demo_tool.py", "0" * 64)
        registry = ToolRegistry(generated_tools_dir=generated_dir)
        with pytest.raises(ToolRegistryError, match="缺失"):
            registry.register_generated_tool(manifest)

    async def test_tampered_file_is_rejected_at_execution(self, tmp_path):
        generated_dir = tmp_path / "gen"
        generated_dir.mkdir()
        entry = generated_dir / "demo_tool.py"
        entry.write_text(TOOL_SOURCE, encoding="utf-8")
        manifest = _manifest("demo_tool.py", sha256_file(entry))

        server = generated_tool_to_mcp_server(
            entry, manifest, generated_tools_root=generated_dir
        )
        entry.write_text("def run(v):\n    return 'tampered'\n", encoding="utf-8")

        response = await server.handle_request(
            MCPRequest(
                method=METHOD_TOOLS_CALL,
                params={"name": "demo_tool", "arguments": {"value": 3}},
                id="call-1",
            )
        )
        assert response.error is not None
        assert "篡改" in response.error.message

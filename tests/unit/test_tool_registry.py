"""单元测试：工具注册表 ToolRegistry。

覆盖 leader 补充要求：
- 主管能看到所有工具（三个固定入口 + 动态生成工具）；
- 动态生成工具必须评审通过（review_status=approved）后才可放行使用；
- 生成工具不得与主管专属固定入口（code_agent/subagent_creator/review_agent）重名。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from so_agent.mcp.protocol import MCPToolSchema
from so_agent.mcp.server import MCPServer, MCPToolServer
from so_agent.models import GeneratedToolManifest
from so_agent.runtime.tool_registry import (
    DEFAULT_BUILTIN_AGENTS,
    ToolRegistry,
    ToolRegistryError,
)

BUILTIN_NAMES = ("code_agent", "subagent_creator", "review_agent")
# 预置可分配工具（随注册表默认登记）
PRESET_NAMES = ("read_file", "write_file", "run_in_sandbox")
# 主管专属工具（assignable=False，严禁分配）
EXCLUSIVE_NAMES = ("code_agent", "subagent_creator")


def make_manifest(name: str, status: str = "pending") -> GeneratedToolManifest:
    return GeneratedToolManifest(
        tool_id=f"tool-{name}",
        tool_name=name,
        entry_file=f"{name}.py",
        description=f"{name} tool",
        created_by_task="t1",
        review_status=status,  # type: ignore[arg-type]
    )


class TestBuiltinRegistration:
    def test_default_builtin_agents_has_three_entries(self):
        assert set(DEFAULT_BUILTIN_AGENTS) == set(BUILTIN_NAMES)

    def test_registry_initializes_with_three_builtins(self):
        registry = ToolRegistry()
        assert registry.list_tool_names() == sorted(
            set(BUILTIN_NAMES) | set(PRESET_NAMES)
        )
        schemas = registry.list_tools()
        assert [schema.name for schema in schemas] == registry.list_tool_names()
        assert all(isinstance(schema, MCPToolSchema) for schema in schemas)

    def test_builtin_entries_approved(self):
        registry = ToolRegistry()
        for name in BUILTIN_NAMES:
            assert registry.is_approved(name) is True
            entry = registry.get_tool(name)
            assert entry is not None
            assert entry["source"] == "builtin"
            assert entry["approved"] is True
            assert entry["agent_ref"]

    def test_get_tool_returns_copy(self):
        registry = ToolRegistry()
        entry = registry.get_tool("code_agent")
        entry["name"] = "mutated"
        assert registry.get_tool("code_agent")["name"] == "code_agent"

    def test_get_tool_missing_returns_none(self):
        registry = ToolRegistry()
        assert registry.get_tool("no_such_tool") is None

    def test_custom_builtin_agents_override(self):
        registry = ToolRegistry(
            builtin_agents={"my_entry": ("desc", "pkg.ref")},
            include_assignable_presets=False,
        )
        assert registry.list_tool_names() == ["my_entry"]
        assert registry.is_approved("my_entry") is True

    def test_register_builtin_empty_name_rejected(self):
        registry = ToolRegistry()
        with pytest.raises(ToolRegistryError):
            registry.register_builtin_tool("", "desc", "pkg.ref")

    def test_register_builtin_empty_ref_rejected(self):
        registry = ToolRegistry()
        with pytest.raises(ToolRegistryError):
            registry.register_builtin_tool("x", "desc", "")

    def test_register_builtin_overrides_existing(self):
        registry = ToolRegistry()
        registry.register_builtin_tool("code_agent", "新描述", "pkg.new_ref")
        assert registry.get_tool("code_agent")["description"] == "新描述"


class TestGeneratedToolRegistration:
    def test_register_pending_generated(self):
        registry = ToolRegistry()
        registry.register_generated_tool(make_manifest("report_writer", "pending"))
        entry = registry.get_tool("report_writer")
        assert entry is not None
        assert entry["source"] == "generated"
        assert entry["approved"] is False
        assert entry["review_status"] == "pending"

    def test_pending_not_available(self):
        registry = ToolRegistry()
        registry.register_generated_tool(make_manifest("report_writer", "pending"))
        assert registry.is_approved("report_writer") is False
        assert "report_writer" not in registry.list_approved_tools()

    def test_rejected_not_available(self):
        registry = ToolRegistry()
        registry.register_generated_tool(make_manifest("report_writer", "rejected"))
        assert registry.is_approved("report_writer") is False

    def test_pending_then_approved_available(self):
        registry = ToolRegistry()
        registry.register_generated_tool(make_manifest("report_writer", "pending"))
        registry.register_generated_tool(make_manifest("report_writer", "approved"))
        assert registry.is_approved("report_writer") is True
        assert "report_writer" in registry.list_approved_tools()

    def test_approved_then_rejected_revokes_availability(self):
        registry = ToolRegistry()
        registry.register_generated_tool(make_manifest("report_writer", "approved"))
        registry.register_generated_tool(make_manifest("report_writer", "rejected"))
        assert registry.is_approved("report_writer") is False

    def test_builtin_name_conflict_rejected(self):
        """生成工具不得冒名主管专属固定入口。"""
        registry = ToolRegistry()
        for name in BUILTIN_NAMES:
            with pytest.raises(ToolRegistryError):
                registry.register_generated_tool(make_manifest(name, "approved"))

    def test_wrong_manifest_type_rejected(self):
        registry = ToolRegistry()
        with pytest.raises(ToolRegistryError):
            registry.register_generated_tool({"tool_name": "x"})  # type: ignore[arg-type]

    def test_unknown_tool_not_approved(self):
        registry = ToolRegistry()
        assert registry.is_approved("ghost_tool") is False

    def test_get_manifest(self):
        registry = ToolRegistry()
        manifest = make_manifest("report_writer", "pending")
        registry.register_generated_tool(manifest)
        assert registry.get_manifest("report_writer") is manifest
        assert registry.get_manifest("code_agent") is None
        assert registry.get_manifest("ghost") is None


class TestSupervisorFullVisibility:
    """主管能看到全部工具：固定入口 + 全部动态生成工具。"""

    def test_list_tools_includes_builtin_and_generated(self):
        registry = ToolRegistry()
        registry.register_generated_tool(make_manifest("tool_a", "pending"))
        registry.register_generated_tool(make_manifest("tool_b", "approved"))
        names = registry.list_tool_names()
        assert set(names) == set(BUILTIN_NAMES) | set(PRESET_NAMES) | {"tool_a", "tool_b"}
        assert names == sorted(names)
        # list_tools 返回 MCP 暴露契约，顺序与名称清单一致
        assert [schema.name for schema in registry.list_tools()] == names

    def test_list_approved_excludes_pending(self):
        registry = ToolRegistry()
        registry.register_generated_tool(make_manifest("tool_a", "pending"))
        registry.register_generated_tool(make_manifest("tool_b", "approved"))
        approved = registry.list_approved_tools()
        assert set(approved) == set(BUILTIN_NAMES) | set(PRESET_NAMES) | {"tool_b"}
        assert "tool_a" not in approved


class TestMCPSurface:
    """MCP 契约面：每个工具附带 MCPToolSchema，装配阶段可挂载 MCPServer。"""

    def test_builtin_and_preset_schemas_exposed(self):
        registry = ToolRegistry()
        for name in (*BUILTIN_NAMES, *PRESET_NAMES):
            schema = registry.get_mcp_schema(name)
            assert schema is not None
            assert schema.name == name
            assert isinstance(schema.inputSchema, dict)
        # 预置工具带真实参数契约（read_file 需要 path）
        read_schema = registry.get_mcp_schema("read_file")
        assert read_schema is not None
        assert "path" in read_schema.inputSchema.get("properties", {})

    def test_get_mcp_schema_unknown_returns_none(self):
        assert ToolRegistry().get_mcp_schema("ghost") is None

    def test_attach_mcp_server_and_retrieve(self):
        registry = ToolRegistry()
        server = MCPServer(name="code_agent")
        schema = MCPToolSchema(name="code_agent", description="desc")
        server.register_tool(schema, lambda arguments: "ok")
        registry.attach_mcp("code_agent", schema, server)
        assert registry.get_mcp_server("code_agent") is server

    def test_attach_mcp_unknown_tool_rejected(self):
        registry = ToolRegistry()
        with pytest.raises(ToolRegistryError):
            registry.attach_mcp("ghost", MCPToolSchema(name="ghost"))

    def test_attach_mcp_invalid_server_rejected(self):
        registry = ToolRegistry()
        with pytest.raises(ToolRegistryError):
            registry.attach_mcp(
                "code_agent",
                MCPToolSchema(name="code_agent"),
                server="not-a-server",  # type: ignore[arg-type]
            )

    def test_generated_registration_auto_creates_mcp_server(self, tmp_path):
        registry = ToolRegistry(generated_tools_dir=tmp_path)
        manifest = GeneratedToolManifest(
            tool_id="gt-1",
            tool_name="csv_report",
            entry_file="csv_report.py",
            description="渲染 CSV 报表",
            created_by_task="t1",
            review_status="approved",
            mcp_schema={
                "name": "csv_report",
                "description": "MCP 描述优先",
                "inputSchema": {
                    "type": "object",
                    "properties": {"rows": {"type": "array"}},
                    "required": ["rows"],
                },
            },
        )
        registry.register_generated_tool(manifest)
        schema = registry.get_mcp_schema("csv_report")
        assert schema is not None
        assert schema.description == "MCP 描述优先"
        assert schema.inputSchema["required"] == ["rows"]
        server = registry.get_mcp_server("csv_report")
        assert isinstance(server, MCPToolServer)
        assert server.name == "csv_report"
        assert server.has_tool("csv_report") is True


class TestAssignableTools:
    """可分配性：主管专属工具严禁分配，其余工具进入分配候选。"""

    def test_exclusive_tools_not_assignable(self):
        registry = ToolRegistry()
        for name in EXCLUSIVE_NAMES:
            assert registry.is_assignable(name) is False
        # review_agent 非专属可分配；预置工具可分配
        assert registry.is_assignable("review_agent") is True
        for name in PRESET_NAMES:
            assert registry.is_assignable(name) is True

    def test_unknown_tool_not_assignable(self):
        assert ToolRegistry().is_assignable("ghost") is False

    def test_generated_tool_assignable(self):
        registry = ToolRegistry()
        registry.register_generated_tool(make_manifest("tool_a", "pending"))
        assert registry.is_assignable("tool_a") is True

    def test_list_assignable_includes_presets_and_review(self):
        registry = ToolRegistry()
        names = [item["name"] for item in registry.list_assignable_tools()]
        assert "code_agent" not in names
        assert "subagent_creator" not in names
        assert "review_agent" in names
        for name in PRESET_NAMES:
            assert name in names

    def test_list_assignable_approved_only_filters_pending(self):
        registry = ToolRegistry()
        registry.register_generated_tool(make_manifest("tool_a", "pending"))
        registry.register_generated_tool(make_manifest("tool_b", "approved"))
        all_names = [item["name"] for item in registry.list_assignable_tools()]
        assert "tool_a" in all_names
        assert "tool_b" in all_names
        approved_names = [
            item["name"] for item in registry.list_assignable_tools(approved_only=True)
        ]
        assert "tool_b" in approved_names
        assert "tool_a" not in approved_names


class TestLoadGeneratedRegistry:
    def test_missing_file_raises(self, tmp_path):
        registry = ToolRegistry()
        with pytest.raises(ToolRegistryError):
            registry.load_generated_registry(tmp_path / "missing.json")

    def test_invalid_json_raises(self, tmp_path):
        path = tmp_path / "registry.json"
        path.write_text("{not-json", encoding="utf-8")
        with pytest.raises(ToolRegistryError):
            ToolRegistry().load_generated_registry(path)

    def test_top_level_not_list_raises(self, tmp_path):
        path = tmp_path / "registry.json"
        path.write_text(json.dumps({"tool_name": "x"}), encoding="utf-8")
        with pytest.raises(ToolRegistryError):
            ToolRegistry().load_generated_registry(path)

    def test_invalid_element_raises(self, tmp_path):
        path = tmp_path / "registry.json"
        path.write_text(json.dumps([{"tool_name": "x"}]), encoding="utf-8")
        with pytest.raises(ToolRegistryError):
            ToolRegistry().load_generated_registry(path)

    def test_load_success_returns_count(self, tmp_path):
        path = tmp_path / "registry.json"
        path.write_text(
            json.dumps(
                [
                    make_manifest("tool_a", "approved").model_dump(mode="json"),
                    make_manifest("tool_b", "pending").model_dump(mode="json"),
                ]
            ),
            encoding="utf-8",
        )
        registry = ToolRegistry()
        loaded = registry.load_generated_registry(path)
        assert loaded == 2
        assert registry.is_approved("tool_a") is True
        assert registry.is_approved("tool_b") is False

    def test_init_with_registry_path_loads_immediately(self, tmp_path):
        path = tmp_path / "registry.json"
        path.write_text(
            json.dumps([make_manifest("tool_a", "approved").model_dump(mode="json")]),
            encoding="utf-8",
        )
        registry = ToolRegistry(generated_registry_path=path)
        assert registry.is_approved("tool_a") is True


class TestSaveGeneratedRegistry:
    def test_save_and_roundtrip(self, tmp_path):
        registry = ToolRegistry()
        registry.register_generated_tool(make_manifest("z_tool", "approved"))
        registry.register_generated_tool(make_manifest("a_tool", "pending"))
        path = tmp_path / "out" / "registry.json"
        registry.save_generated_registry(path)
        assert path.is_file()

        payload = json.loads(path.read_text(encoding="utf-8"))
        assert isinstance(payload, list)
        assert [item["tool_name"] for item in payload] == ["a_tool", "z_tool"]

        reloaded = ToolRegistry()
        assert reloaded.load_generated_registry(path) == 2
        assert reloaded.is_approved("z_tool") is True
        assert reloaded.is_approved("a_tool") is False

    def test_save_creates_parent_dirs(self, tmp_path):
        registry = ToolRegistry()
        registry.register_generated_tool(make_manifest("tool_a", "pending"))
        path = tmp_path / "deep" / "nested" / "registry.json"
        registry.save_generated_registry(path)
        assert path.is_file()

    def test_save_empty_registry_writes_empty_list(self, tmp_path):
        path = tmp_path / "registry.json"
        ToolRegistry().save_generated_registry(path)
        assert json.loads(path.read_text(encoding="utf-8")) == []

    def test_saved_file_is_utf8_with_newline(self, tmp_path):
        registry = ToolRegistry()
        registry.register_generated_tool(make_manifest("工具名_a", "pending"))
        path = tmp_path / "registry.json"
        registry.save_generated_registry(path)
        text = path.read_text(encoding="utf-8")
        assert text.endswith("\n")
        assert "工具名_a" in text

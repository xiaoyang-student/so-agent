"""单元测试：MCP Server / Client / 适配器。

覆盖：
- MCPServer 注册、发现与请求路由（tools/list、tools/call 全路径）；
- MCPClient 连接管理、工具发现与调用（含错误路径）；
- agent_to_mcp_server / mcp_server_to_agent_tool / generated_tool_to_mcp_server
  三个适配器的契约转换与调用桥接。
"""

from __future__ import annotations

import json

import pytest
from agents.tool_context import ToolContext
from pydantic import BaseModel

import so_agent.mcp.adapter as adapter_mod
from so_agent.mcp.adapter import (
    MCPAdapterError,
    agent_to_mcp_server,
    generated_tool_to_mcp_server,
    mcp_server_to_agent_tool,
)
from so_agent.mcp.client import MCPClient, MCPClientError
from so_agent.mcp.protocol import (
    INTERNAL_ERROR,
    INVALID_PARAMS,
    METHOD_NOT_FOUND,
    METHOD_TOOLS_CALL,
    METHOD_TOOLS_LIST,
    MCPProtocolError,
    MCPRequest,
    MCPResponse,
    MCPToolSchema,
    extract_result_text,
)
from so_agent.mcp.server import MCPServer, MCPToolServer
from so_agent.models import GeneratedToolManifest


def make_ctx(tool_name: str) -> ToolContext:
    return ToolContext(
        context=None,
        tool_name=tool_name,
        tool_call_id=f"call-{tool_name}-1",
        tool_arguments="{}",
    )


class TestMCPServer:
    def test_empty_name_rejected(self):
        with pytest.raises(MCPProtocolError):
            MCPServer(name="   ")

    def test_register_and_discover(self):
        server = MCPServer(name="srv")
        schema = MCPToolSchema(name="t1", description="d")
        server.register_tool(schema, lambda arguments: "ok")
        assert server.has_tool("t1") is True
        assert server.has_tool("missing") is False
        assert server.get_tool_schema("t1") is schema
        assert server.get_tool_schema("missing") is None
        assert server.list_tools() == [schema]

    def test_register_rejects_invalid_inputs(self):
        server = MCPServer(name="srv")
        with pytest.raises(MCPProtocolError):
            server.register_tool("not-schema", lambda arguments: "ok")  # type: ignore[arg-type]
        with pytest.raises(MCPProtocolError):
            server.register_tool(MCPToolSchema(name="t"), "not-callable")  # type: ignore[arg-type]

    def test_reregister_overrides_same_name(self):
        server = MCPServer(name="srv")
        server.register_tool(MCPToolSchema(name="t", description="v1"), lambda a: "1")
        server.register_tool(MCPToolSchema(name="t", description="v2"), lambda a: "2")
        assert len(server.list_tools()) == 1
        schema = server.get_tool_schema("t")
        assert schema is not None and schema.description == "v2"

    async def test_tools_list_route(self):
        server = MCPServer(name="srv")
        schema = MCPToolSchema(name="t1", description="d")
        server.register_tool(schema, lambda arguments: "ok")
        response = await server.handle_request(
            MCPRequest(method=METHOD_TOOLS_LIST, id="L1")
        )
        assert response.ok is True
        assert response.id == "L1"
        assert response.result == {"tools": [schema.model_dump(mode="json")]}

    async def test_unknown_method_rejected(self):
        server = MCPServer(name="srv")
        response = await server.handle_request(MCPRequest(method="tools/unknown", id=1))
        assert response.error is not None
        assert response.error.code == METHOD_NOT_FOUND

    async def test_request_type_checked(self):
        server = MCPServer(name="srv")
        with pytest.raises(MCPProtocolError):
            await server.handle_request({"method": METHOD_TOOLS_LIST})  # type: ignore[arg-type]

    async def test_call_sync_handler(self):
        server = MCPServer(name="srv")
        server.register_tool(
            MCPToolSchema(name="echo"), lambda arguments: {"got": arguments}
        )
        response = await server.handle_request(
            MCPRequest(
                method=METHOD_TOOLS_CALL,
                params={"name": "echo", "arguments": {"x": 1}},
                id=1,
            )
        )
        assert response.ok is True
        assert json.loads(extract_result_text(response)) == {"got": {"x": 1}}

    async def test_call_async_handler(self):
        server = MCPServer(name="srv")

        async def handler(arguments):
            return "async-ok"

        server.register_tool(MCPToolSchema(name="a"), handler)
        response = await server.handle_request(
            MCPRequest(method=METHOD_TOOLS_CALL, params={"name": "a"}, id=1)
        )
        assert extract_result_text(response) == "async-ok"

    async def test_call_missing_name_rejected(self):
        server = MCPServer(name="srv")
        response = await server.handle_request(
            MCPRequest(method=METHOD_TOOLS_CALL, params={}, id=1)
        )
        assert response.error is not None
        assert response.error.code == INVALID_PARAMS

    async def test_call_unknown_tool_rejected(self):
        server = MCPServer(name="srv")
        response = await server.handle_request(
            MCPRequest(method=METHOD_TOOLS_CALL, params={"name": "ghost"}, id=1)
        )
        assert response.error is not None
        assert response.error.code == METHOD_NOT_FOUND

    async def test_call_arguments_not_dict_rejected(self):
        server = MCPServer(name="srv")
        server.register_tool(MCPToolSchema(name="t"), lambda arguments: "ok")
        response = await server.handle_request(
            MCPRequest(
                method=METHOD_TOOLS_CALL,
                params={"name": "t", "arguments": [1, 2]},
                id=1,
            )
        )
        assert response.error is not None
        assert response.error.code == INVALID_PARAMS

    async def test_call_value_error_maps_to_invalid_params(self):
        server = MCPServer(name="srv")

        def handler(arguments):
            raise ValueError("参数不符合契约")

        server.register_tool(MCPToolSchema(name="t"), handler)
        response = await server.handle_request(
            MCPRequest(method=METHOD_TOOLS_CALL, params={"name": "t"}, id=1)
        )
        assert response.error is not None
        assert response.error.code == INVALID_PARAMS

    async def test_call_other_exception_maps_to_internal_error(self):
        server = MCPServer(name="srv")

        def handler(arguments):
            raise RuntimeError("boom")

        server.register_tool(MCPToolSchema(name="t"), handler)
        response = await server.handle_request(
            MCPRequest(method=METHOD_TOOLS_CALL, params={"name": "t"}, id=1)
        )
        assert response.error is not None
        assert response.error.code == INTERNAL_ERROR

    async def test_call_without_arguments_passes_empty_dict(self):
        server = MCPServer(name="srv")
        received: list[dict] = []
        server.register_tool(
            MCPToolSchema(name="t"),
            lambda arguments: received.append(arguments) or "ok",
        )
        response = await server.handle_request(
            MCPRequest(method=METHOD_TOOLS_CALL, params={"name": "t"}, id=1)
        )
        assert response.ok is True
        assert received == [{}]

    async def test_call_basemodel_result_stringified(self):
        class Result(BaseModel):
            value: int

        server = MCPServer(name="srv")
        server.register_tool(MCPToolSchema(name="t"), lambda arguments: Result(value=7))
        response = await server.handle_request(
            MCPRequest(method=METHOD_TOOLS_CALL, params={"name": "t"}, id=1)
        )
        assert json.loads(extract_result_text(response)) == {"value": 7}

    async def test_call_mcpresponse_passthrough_with_id_override(self):
        server = MCPServer(name="srv")
        server.register_tool(
            MCPToolSchema(name="t"), lambda arguments: MCPResponse(result="直接")
        )
        response = await server.handle_request(
            MCPRequest(method=METHOD_TOOLS_CALL, params={"name": "t"}, id=7)
        )
        assert response.id == 7
        assert response.result == "直接"


class TestMCPClient:
    def test_connect_type_checked(self):
        with pytest.raises(MCPClientError):
            MCPClient().connect("not-a-server")  # type: ignore[arg-type]

    def test_connect_list_get_disconnect(self):
        client = MCPClient()
        s1 = MCPServer(name="a")
        s2 = MCPServer(name="b")
        client.connect(s1)
        client.connect(s2)
        assert client.list_servers() == ["a", "b"]
        assert client.get_server("a") is s1
        assert client.disconnect("a") is True
        assert client.disconnect("a") is False
        assert client.get_server("a") is None

    def test_list_tools_aggregation_sorted_by_server(self):
        client = MCPClient()
        beta = MCPServer(name="beta")
        beta.register_tool(MCPToolSchema(name="t1"), lambda arguments: "1")
        alpha = MCPServer(name="alpha")
        alpha.register_tool(MCPToolSchema(name="t2"), lambda arguments: "2")
        client.connect(beta)
        client.connect(alpha)
        assert [schema.name for schema in client.list_tools()] == ["t2", "t1"]
        assert [schema.name for schema in client.list_tools("beta")] == ["t1"]

    def test_list_tools_unknown_server_rejected(self):
        with pytest.raises(MCPClientError):
            MCPClient().list_tools("missing")

    async def test_call_tool_success(self):
        client = MCPClient()
        server = MCPServer(name="srv")
        server.register_tool(
            MCPToolSchema(name="echo"),
            lambda arguments: {"ok": arguments.get("x")},
        )
        client.connect(server)
        response = await client.call_tool("srv", "echo", {"x": 9})
        assert response.ok is True
        assert json.loads(extract_result_text(response)) == {"ok": 9}

    async def test_call_tool_unknown_server_rejected(self):
        with pytest.raises(MCPClientError):
            await MCPClient().call_tool("srv", "t", {})

    async def test_call_tool_unknown_tool_rejected(self):
        client = MCPClient()
        client.connect(MCPServer(name="srv"))
        response = await client.call_tool("srv", "ghost", {})
        assert response.error is not None
        assert response.error.code == METHOD_NOT_FOUND

    def test_call_tool_sync(self):
        client = MCPClient()
        server = MCPServer(name="srv")
        server.register_tool(MCPToolSchema(name="t"), lambda arguments: "sync-ok")
        client.connect(server)
        response = client.call_tool_sync("srv", "t", {})
        assert extract_result_text(response) == "sync-ok"

    def test_call_tool_sync_unknown_server_rejected(self):
        with pytest.raises(MCPClientError):
            MCPClient().call_tool_sync("srv", "t", {})


class DemoInput(BaseModel):
    value: int


class DemoOutput(BaseModel):
    doubled: int


class TestAgentToMCPServer:
    def _install_fake_runner(self, monkeypatch, captured: list):
        class FakeRunResult:
            def __init__(self, final_output):
                self.final_output = final_output

        class FakeRunner:
            @classmethod
            async def run(cls, agent, input=None, *, max_turns=None, **kwargs):
                captured.append({"agent": agent, "input": input, "max_turns": max_turns})
                payload = DemoInput.model_validate_json(input)
                return FakeRunResult(DemoOutput(doubled=payload.value * 2))

        monkeypatch.setattr(adapter_mod, "Runner", FakeRunner)

    def test_invalid_models_rejected(self):
        with pytest.raises(MCPAdapterError):
            agent_to_mcp_server(object(), "demo", "d", dict, DemoOutput)  # type: ignore[arg-type]
        with pytest.raises(MCPAdapterError):
            agent_to_mcp_server(object(), "demo", "d", DemoInput, "not-model")  # type: ignore[arg-type]

    def test_schema_comes_from_input_model(self):
        server = agent_to_mcp_server(object(), "demo", "示例工具", DemoInput, DemoOutput)
        assert isinstance(server, MCPToolServer)
        assert server.name == "demo"
        schema = server.get_tool_schema("demo")
        assert schema is not None
        assert schema.description == "示例工具"
        assert schema.inputSchema == DemoInput.model_json_schema()

    async def test_call_runs_agent_and_returns_json_text(self, monkeypatch):
        captured: list = []
        self._install_fake_runner(monkeypatch, captured)
        agent = object()
        server = agent_to_mcp_server(
            agent, "demo", "d", DemoInput, DemoOutput, max_turns=5
        )
        response = await server.handle_request(
            MCPRequest(
                method=METHOD_TOOLS_CALL,
                params={"name": "demo", "arguments": {"value": 21}},
                id=1,
            )
        )
        assert response.ok is True
        assert json.loads(extract_result_text(response)) == {"doubled": 42}
        assert captured[0]["agent"] is agent
        assert captured[0]["max_turns"] == 5
        assert json.loads(captured[0]["input"]) == {"value": 21}

    async def test_invalid_arguments_map_to_invalid_params(self, monkeypatch):
        self._install_fake_runner(monkeypatch, [])
        server = agent_to_mcp_server(object(), "demo", "d", DemoInput, DemoOutput)
        response = await server.handle_request(
            MCPRequest(
                method=METHOD_TOOLS_CALL,
                params={"name": "demo", "arguments": {"value": "not-int"}},
                id=2,
            )
        )
        assert response.error is not None
        assert response.error.code == INVALID_PARAMS


class TestMCPServerToAgentTool:
    def _server_with_echo(self) -> MCPServer:
        server = MCPToolServer(name="srv")
        schema = MCPToolSchema(
            name="echo",
            description="回显文本",
            inputSchema={
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
        )

        async def handler(arguments):
            return {"echo": arguments["text"]}

        server.register_tool(schema, handler)
        return server

    def test_unregistered_tool_rejected(self):
        with pytest.raises(MCPAdapterError):
            mcp_server_to_agent_tool(self._server_with_echo(), "missing")

    def test_function_tool_contract_from_schema(self):
        tool = mcp_server_to_agent_tool(self._server_with_echo(), "echo")
        assert tool.name == "echo"
        assert tool.description == "回显文本"
        assert tool.params_json_schema["properties"]["text"]["type"] == "string"

    async def test_invoke_bridges_to_mcp_call(self):
        tool = mcp_server_to_agent_tool(self._server_with_echo(), "echo")
        raw = await tool.on_invoke_tool(
            make_ctx("echo"), json.dumps({"text": "你好"}, ensure_ascii=False)
        )
        assert json.loads(raw) == {"echo": "你好"}

    async def test_invoke_invalid_json_returns_error_json(self):
        tool = mcp_server_to_agent_tool(self._server_with_echo(), "echo")
        raw = await tool.on_invoke_tool(make_ctx("echo"), "{not-json")
        payload = json.loads(raw)
        assert payload["status"] == "error"

    async def test_invoke_tool_error_returns_error_json(self):
        server = MCPToolServer(name="srv")

        def bad_handler(arguments):
            raise ValueError("坏参数")

        server.register_tool(MCPToolSchema(name="bad"), bad_handler)
        tool = mcp_server_to_agent_tool(server, "bad")
        raw = await tool.on_invoke_tool(make_ctx("bad"), "{}")
        payload = json.loads(raw)
        assert payload["status"] == "error"
        assert str(INVALID_PARAMS) in payload["error"]


class TestGeneratedToolToMCPServer:
    def _manifest(self, name: str = "csv_report") -> GeneratedToolManifest:
        return GeneratedToolManifest(
            tool_id="gt-1",
            tool_name=name,
            entry_file=f"{name}.py",
            description="统计行数",
            created_by_task="t1",
            review_status="approved",  # type: ignore[arg-type]
            input_schema={
                "type": "object",
                "properties": {"rows": {"type": "array"}},
                "required": ["rows"],
            },
        )

    def test_invalid_manifest_rejected(self):
        with pytest.raises(MCPAdapterError):
            generated_tool_to_mcp_server("x.py", {"tool_name": "x"})  # type: ignore[arg-type]

    def test_call_loaded_source(self, tmp_path):
        source = tmp_path / "csv_report.py"
        source.write_text(
            "def run(rows):\n"
            "    return {'count': len(rows)}\n",
            encoding="utf-8",
        )
        server = generated_tool_to_mcp_server(source, self._manifest())
        assert server.name == "csv_report"
        response = server.handle_request_sync(
            MCPRequest(
                method=METHOD_TOOLS_CALL,
                params={"name": "csv_report", "arguments": {"rows": [1, 2, 3]}},
                id=1,
            )
        )
        assert response.ok is True
        assert json.loads(extract_result_text(response)) == {"count": 3}

    def test_missing_entry_maps_to_internal_error(self, tmp_path):
        source = tmp_path / "empty_tool.py"
        source.write_text("VALUE = 1\n", encoding="utf-8")
        server = generated_tool_to_mcp_server(source, self._manifest("empty_tool"))
        response = server.handle_request_sync(
            MCPRequest(
                method=METHOD_TOOLS_CALL,
                params={"name": "empty_tool", "arguments": {}},
                id=1,
            )
        )
        assert response.error is not None
        assert response.error.code == INTERNAL_ERROR

    def test_missing_file_maps_to_internal_error(self, tmp_path):
        server = generated_tool_to_mcp_server(
            tmp_path / "ghost.py", self._manifest("ghost")
        )
        response = server.handle_request_sync(
            MCPRequest(
                method=METHOD_TOOLS_CALL,
                params={"name": "ghost", "arguments": {}},
                id=1,
            )
        )
        assert response.error is not None
        assert response.error.code == INTERNAL_ERROR


class TestAgentPackagesMCPSurface:
    """三个固定 Agent 工具包按 MCP 规范暴露（get_mcp_schema）。"""

    def test_three_fixed_agents_expose_mcp_schema(self):
        from so_agent.tool_packages.code_agent import agent as code_agent_mod
        from so_agent.tool_packages.review_agent import agent as review_agent_mod
        from so_agent.tool_packages.subagent_creator import agent as creator_mod

        for module, expected_name in (
            (code_agent_mod, "code_agent"),
            (creator_mod, "subagent_creator"),
            (review_agent_mod, "review_agent"),
        ):
            schema = module.get_mcp_schema()
            assert isinstance(schema, MCPToolSchema)
            assert schema.name == expected_name
            assert schema.description
            assert schema.inputSchema.get("type") == "object"
            assert schema.inputSchema.get("properties")

"""外部 MCP 安全接入测试：默认关闭、白名单、资源上限与本地假服务器。"""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from pydantic import ValidationError

from so_agent.mcp.client import MCPClient
from so_agent.mcp.external import (
    ExternalMCPClient,
    ExternalMCPConfig,
    ExternalMCPError,
    ExternalMCPPolicyError,
    ExternalMCPServer,
    ExternalMCPServerConfig,
    ExternalMCPTransportError,
    discover_external_servers,
    load_external_mcp_config,
)
from so_agent.mcp.protocol import extract_result_text


class _FakeMCPHandler(BaseHTTPRequestHandler):
    """本地假 MCP 服务器：按路径与工具名返回不同行为。"""

    def log_message(self, format, *args):  # noqa: A002
        return

    def _respond(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: int, payload: dict) -> None:
        self._respond(
            status,
            json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            "application/json",
        )

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length") or 0)
        request = json.loads(self.rfile.read(length) or b"{}")
        request_id = request.get("id")
        if self.path == "/redirect":
            self._respond(302, b"", "application/json")
            return
        if self.path == "/invalid":
            self._respond(200, b"not-json", "application/json")
            return
        if self.path == "/wrong-type":
            payload = json.dumps(
                {"jsonrpc": "2.0", "id": request_id, "result": {}}
            ).encode("utf-8")
            self._respond(200, payload, "text/plain")
            return

        method = request.get("method")
        if method == "tools/list":
            self._json(
                200,
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {
                        "tools": [
                            {
                                "name": "echo",
                                "description": "回显参数",
                                "inputSchema": {"type": "object"},
                            },
                            {
                                "name": "secret",
                                "description": "未授权工具",
                                "inputSchema": {"type": "object"},
                            },
                        ]
                    },
                },
            )
            return
        if method == "tools/call":
            name = (request.get("params") or {}).get("name")
            if name == "slow":
                time.sleep(0.5)
            if name == "boom":
                self._json(500, {"jsonrpc": "2.0", "id": request_id, "error": {}})
                return
            if name == "huge":
                text = "x" * 5000
            else:
                arguments = (request.get("params") or {}).get("arguments") or {}
                text = "echo:" + json.dumps(arguments, ensure_ascii=False)
            self._json(
                200,
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {
                        "content": [{"type": "text", "text": text}],
                        "isError": False,
                    },
                },
            )
            return
        self._json(
            200,
            {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {"code": -32601, "message": "method not found"},
            },
        )


@pytest.fixture()
def fake_mcp_base_url():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _FakeMCPHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def _config(url: str, **overrides) -> ExternalMCPServerConfig:
    data = {
        "name": "market",
        "url": url,
        "allowed_hosts": ["127.0.0.1"],
        "allowed_tools": ["echo", "huge", "slow", "boom"],
        "allow_insecure_http": True,
        "allow_private_network": True,
        "timeout_seconds": 3.0,
    }
    data.update(overrides)
    return ExternalMCPServerConfig(**data)


def test_config_is_disabled_by_default(fake_mcp_base_url):
    config = ExternalMCPConfig()
    assert config.enabled is False
    assert config.servers == []


async def test_disabled_config_never_connects(fake_mcp_base_url):
    config = ExternalMCPConfig(
        enabled=False,
        servers=[_config(fake_mcp_base_url + "/rpc")],
    )
    assert await discover_external_servers(config) == []


def test_policy_rejects_insecure_and_unlisted_hosts(fake_mcp_base_url):
    with pytest.raises(ValidationError, match="https"):
        _config(fake_mcp_base_url + "/rpc", allow_insecure_http=False)
    with pytest.raises(ValidationError, match="不在 allowed_hosts"):
        _config(
            fake_mcp_base_url + "/rpc",
            allowed_hosts=["mcp.example.com"],
        )
    with pytest.raises(ValidationError, match="allowed_tools"):
        _config(fake_mcp_base_url + "/rpc", allowed_tools=["  "])
    with pytest.raises(ValidationError, match="通配符"):
        _config(
            fake_mcp_base_url + "/rpc",
            allowed_hosts=["127.0.0.1", "*.example.com"],
        )


def test_loader_defaults_to_file_and_policy_errors(tmp_path):
    good = tmp_path / "external_mcp.yaml"
    good.write_text(
        "enabled: true\n"
        "servers:\n"
        "  - name: market\n"
        "    url: https://mcp.example.com/rpc\n"
        "    allowed_hosts: [mcp.example.com]\n"
        "    allowed_tools: [search]\n",
        encoding="utf-8",
    )
    config = load_external_mcp_config(good)
    assert config.enabled is True
    assert config.servers[0].name == "market"
    assert load_external_mcp_config(good, enabled=False).enabled is False

    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "enabled: true\n"
        "servers:\n"
        "  - name: market\n"
        "    url: http://mcp.example.com/rpc\n"
        "    allowed_hosts: [mcp.example.com]\n"
        "    allowed_tools: [search]\n",
        encoding="utf-8",
    )
    with pytest.raises(ExternalMCPPolicyError, match="https"):
        load_external_mcp_config(bad)


async def test_client_filters_tools_and_rejects_unlisted(fake_mcp_base_url):
    client = ExternalMCPClient(_config(fake_mcp_base_url + "/rpc"))

    tools = await client.list_tools()
    assert [tool.name for tool in tools] == ["echo"]

    response = await client.call_tool("echo", {"message": "你好"})
    assert response.ok
    assert "echo:" in extract_result_text(response)

    with pytest.raises(ExternalMCPError, match="未列入"):
        await client.call_tool("secret", {})


async def test_proxy_server_integrates_with_mcp_client(fake_mcp_base_url):
    proxy = ExternalMCPServer(
        ExternalMCPClient(_config(fake_mcp_base_url + "/rpc"))
    )
    schemas = await proxy.discover()
    assert [schema.name for schema in schemas] == ["echo"]

    client = MCPClient()
    client.connect(proxy)
    assert [tool.name for tool in client.list_tools("market")] == ["echo"]

    response = await client.call_tool("market", "echo", {"value": 7})
    assert response.ok
    assert "echo:" in extract_result_text(response)

    denied = await client.call_tool("market", "secret", {})
    assert denied.ok is False
    assert "工具不存在" in denied.error.message


async def test_transport_limits_timeout_and_protocol_errors(fake_mcp_base_url):
    small = ExternalMCPClient(
        _config(
            fake_mcp_base_url + "/rpc",
            max_response_bytes=1024,
        )
    )
    with pytest.raises(ExternalMCPTransportError, match="超过上限"):
        await small.call_tool("huge", {})

    slow = ExternalMCPClient(
        _config(fake_mcp_base_url + "/rpc", timeout_seconds=0.1)
    )
    with pytest.raises(ExternalMCPTransportError, match="请求失败"):
        await slow.call_tool("slow", {})

    boom = ExternalMCPClient(_config(fake_mcp_base_url + "/rpc"))
    with pytest.raises(ExternalMCPTransportError, match="HTTP 500"):
        await boom.call_tool("boom", {})

    redirect = ExternalMCPClient(_config(fake_mcp_base_url + "/redirect"))
    with pytest.raises(ExternalMCPTransportError, match="不跟随重定向"):
        await redirect.list_tools()

    invalid = ExternalMCPClient(_config(fake_mcp_base_url + "/invalid"))
    with pytest.raises(ExternalMCPError, match="不是合法 JSON"):
        await invalid.list_tools()

    wrong_type = ExternalMCPClient(_config(fake_mcp_base_url + "/wrong-type"))
    with pytest.raises(ExternalMCPError, match="内容类型"):
        await wrong_type.list_tools()


async def test_request_oversize_is_rejected_before_network(fake_mcp_base_url):
    client = ExternalMCPClient(
        _config(fake_mcp_base_url + "/rpc", max_request_bytes=1024)
    )
    with pytest.raises(ExternalMCPError, match="请求超过上限"):
        await client.call_tool("echo", {"payload": "y" * 2000})


async def test_private_network_requires_explicit_opt_in(fake_mcp_base_url):
    client = ExternalMCPClient(
        _config(
            fake_mcp_base_url + "/rpc",
            allow_private_network=False,
        )
    )
    with pytest.raises(ExternalMCPPolicyError, match="非公网地址"):
        await client.list_tools()

"""外部 MCP 目录：扫描校验、注册、移除与签名。"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from so_agent.mcp.client import MCPClient
from so_agent.mcp.directory import (
    MCPDirectoryManager,
    directory_signature,
    scan_mcp_directory,
)
from so_agent.runtime.tool_registry import ToolRegistry


class _FakeMCPHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):  # noqa: A002
        return

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length") or 0)
        request = json.loads(self.rfile.read(length) or b"{}")
        request_id = request.get("id")
        method = request.get("method")
        if method == "tools/list":
            payload = {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "tools": [
                        {
                            "name": "echo",
                            "description": "回显",
                            "inputSchema": {"type": "object"},
                        },
                        {
                            "name": "secret",
                            "description": "未授权",
                            "inputSchema": {"type": "object"},
                        },
                    ]
                },
            }
        elif method == "tools/call":
            name = (request.get("params") or {}).get("name")
            arguments = (request.get("params") or {}).get("arguments") or {}
            payload = {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps(
                                {"tool": name, "args": arguments}, ensure_ascii=False
                            ),
                        }
                    ],
                    "isError": False,
                },
            }
        else:
            payload = {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {"code": -32601, "message": "method not found"},
            }
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture()
def fake_server():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _FakeMCPHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/rpc"
    finally:
        server.shutdown()
        server.server_close()


def _write_server(
    root,
    name: str,
    url: str,
    *,
    enabled: bool = True,
    allowed_tools: list[str] | None = None,
    declared_name: str | None = None,
) -> None:
    directory = root / name
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "server.yaml").write_text(
        "\n".join(
            [
                f"name: {declared_name or name}",
                f"enabled: {'true' if enabled else 'false'}",
                "transport: https_jsonrpc",
                f"url: {url}",
                "allowed_hosts: [127.0.0.1]",
                "allowed_tools: [echo]"
                if allowed_tools is None
                else f"allowed_tools: [{', '.join(allowed_tools)}]",
                "timeout_seconds: 5",
                "max_response_bytes: 100000",
                "allow_insecure_http: true",
                "allow_private_network: true",
            ]
        ),
        encoding="utf-8",
    )


def test_scan_validates_entries(tmp_path, fake_server):
    _write_server(tmp_path, "srv", fake_server, enabled=False)
    _write_server(tmp_path, "bad", fake_server, declared_name="other")
    entries = scan_mcp_directory(tmp_path)
    assert [entry.name for entry in entries] == ["bad", "srv"]
    by_name = {entry.name: entry for entry in entries}
    assert by_name["srv"].enabled is False
    assert by_name["srv"].config is not None
    assert by_name["bad"].error


def test_scan_supports_https_alias_and_stdio_manifest(tmp_path):
    https_dir = tmp_path / "remote"
    https_dir.mkdir()
    (https_dir / "server.yaml").write_text(
        "\n".join(
            [
                "name: remote",
                "transport: https",
                "url: https://example.com/rpc",
                "allowed_hosts: [example.com]",
                "allowed_tools: [echo]",
            ]
        ),
        encoding="utf-8",
    )
    stdio_dir = tmp_path / "local"
    stdio_dir.mkdir()
    (stdio_dir / "server.yaml").write_text(
        "\n".join(
            [
                "name: local",
                "transport: stdio",
                "allow_stdio: true",
                "command: [python, server.py]",
                "stdio_allow_commands: [python]",
                "allowed_tools: [echo]",
            ]
        ),
        encoding="utf-8",
    )

    by_name = {entry.name: entry for entry in scan_mcp_directory(tmp_path)}

    assert by_name["remote"].error == ""
    assert by_name["remote"].config.transport == "https"
    assert by_name["local"].error == ""
    assert by_name["local"].config.transport == "stdio"


def test_directory_disabled_server_is_skipped(tmp_path, fake_server):
    _write_server(tmp_path, "srv", fake_server, enabled=False)
    registry = ToolRegistry(generated_tools_dir=tmp_path / "generated")
    manager = MCPDirectoryManager(tmp_path)

    import asyncio

    report = asyncio.run(manager.refresh(registry))
    assert report.registered == []
    assert any("未启用" in item for item in report.skipped)
    assert registry.list_external_tools() == []


async def test_enabled_server_registers_whitelisted_tools(tmp_path, fake_server):
    _write_server(tmp_path, "srv", fake_server)
    registry = ToolRegistry(generated_tools_dir=tmp_path / "generated")
    manager = MCPDirectoryManager(tmp_path)

    report = await manager.refresh(registry)
    assert report.registered == ["srv.echo"]
    assert registry.get_tool("srv.echo")["source"] == "external_mcp"
    assert registry.is_approved("srv.echo") is True
    assert registry.is_assignable("srv.echo") is True
    assert registry.get_tool("srv.secret") is None

    client = MCPClient()
    client.connect(registry.get_mcp_server("srv.echo"))
    response = await client.call_tool("srv.echo", "srv.echo", {"x": 1})
    assert response.error is None
    text = response.result["content"][0]["text"]
    assert json.loads(text) == {"tool": "echo", "args": {"x": 1}}


async def test_removed_server_tools_are_unregistered(tmp_path, fake_server):
    _write_server(tmp_path, "srv", fake_server)
    registry = ToolRegistry(generated_tools_dir=tmp_path / "generated")
    manager = MCPDirectoryManager(tmp_path)
    await manager.refresh(registry)
    assert registry.get_tool("srv.echo") is not None

    import shutil

    shutil.rmtree(tmp_path / "srv")
    report = await manager.refresh(registry, force=True)
    assert report.registered == []
    assert registry.get_tool("srv.echo") is None


def test_signature_changes_with_manifest(tmp_path, fake_server):
    _write_server(tmp_path, "srv", fake_server, enabled=False)
    first = directory_signature(tmp_path)
    _write_server(tmp_path, "srv", fake_server, enabled=True)
    assert directory_signature(tmp_path) != first

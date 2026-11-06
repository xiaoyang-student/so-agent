"""外部 MCP：环境变量认证头与显式启用的 stdio 传输。"""

from __future__ import annotations

import asyncio
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from pydantic import ValidationError

from so_agent.mcp.external import (
    ExternalMCPClient,
    ExternalMCPError,
    ExternalMCPPolicyError,
    ExternalMCPServerConfig,
)

_STDIO_SERVER = '''
import json
import sys


def reply(payload):
    sys.stdout.write(json.dumps(payload) + "\\n")
    sys.stdout.flush()


for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    request = json.loads(line)
    method = request.get("method")
    request_id = request.get("id")
    if method == "initialize":
        reply({
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "serverInfo": {"name": "fake-stdio"},
            },
        })
    elif method == "tools/list":
        reply({
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "tools": [
                    {"name": "echo", "description": "e", "inputSchema": {"type": "object"}},
                    {"name": "secret", "description": "s", "inputSchema": {"type": "object"}},
                ]
            },
        })
    elif method == "tools/call":
        params = request.get("params") or {}
        reply({
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "content": [
                    {
                        "type": "text",
                        "text": "echo:" + json.dumps(params.get("arguments") or {}),
                    }
                ],
                "isError": False,
            },
        })
'''


def _stdio_config(script: Path, **overrides) -> ExternalMCPServerConfig:
    data = {
        "name": "local_stdio",
        "transport": "stdio",
        "allow_stdio": True,
        "command": [sys.executable, str(script)],
        "stdio_allow_commands": [sys.executable, Path(sys.executable).name],
        "allowed_tools": ["echo"],
        "timeout_seconds": 10.0,
    }
    data.update(overrides)
    return ExternalMCPServerConfig(**data)


def test_stdio_requires_explicit_opt_in_and_allowlist(tmp_path):
    script = tmp_path / "server.py"
    script.write_text(_STDIO_SERVER, encoding="utf-8")

    with pytest.raises(ValidationError, match="allow_stdio"):
        _stdio_config(script, allow_stdio=False)
    with pytest.raises(ValidationError, match="stdio_allow_commands"):
        _stdio_config(script, stdio_allow_commands=[])
    with pytest.raises(ValidationError, match="允许清单"):
        _stdio_config(script, command=[sys.executable, "other.py"], stdio_allow_commands=["nope"])
    with pytest.raises(ValidationError, match="不使用 url"):
        _stdio_config(script, url="https://example.com/rpc")


async def test_stdio_transport_handshake_and_tool_filtering(tmp_path):
    script = tmp_path / "server.py"
    script.write_text(_STDIO_SERVER, encoding="utf-8")
    client = ExternalMCPClient(_stdio_config(script))

    try:
        tools = await client.list_tools()
        assert [tool.name for tool in tools] == ["echo"]

        response = await client.call_tool("echo", {"value": 1})
        assert response.error is None
        text = response.result["content"][0]["text"]
        assert json.loads(text.split(":", 1)[1]) == {"value": 1}

        with pytest.raises(ExternalMCPError, match="allow_tools"):
            await client.call_tool("secret", {})
    finally:
        await client.aclose()


async def test_stderr_flood_does_not_block_server(tmp_path):
    # stderr 管道写满会让子进程阻塞在 write 上，旧实现表现为读取超时
    script = tmp_path / "server.py"
    flood = (
        "import sys\n"
        "for _ in range(2000):\n"
        "    sys.stderr.write('x' * 200 + '\\n')\n"
        "sys.stderr.flush()\n"
    )
    script.write_text(flood + _STDIO_SERVER, encoding="utf-8")
    client = ExternalMCPClient(_stdio_config(script))

    try:
        tools = await client.list_tools()
    finally:
        await client.aclose()
    assert [tool.name for tool in tools] == ["echo"]


async def test_stderr_tail_is_reported_when_server_dies(tmp_path):
    script = tmp_path / "server.py"
    script.write_text(
        "import sys, time\n"
        "sys.stderr.write('boom: dependency missing\\n')\n"
        "sys.stderr.flush()\n"
        "time.sleep(0.5)\n"
        "sys.exit(1)\n",
        encoding="utf-8",
    )
    client = ExternalMCPClient(_stdio_config(script))

    try:
        with pytest.raises(ExternalMCPError, match="boom: dependency missing"):
            await client.list_tools()
    finally:
        await client.aclose()


async def test_aclose_terminates_grandchild_process(tmp_path):
    marker = tmp_path / "grandchild.txt"
    child_code = (
        "import time, pathlib; time.sleep(1.5); "
        f"pathlib.Path({str(marker)!r}).write_text('alive', encoding='utf-8')"
    )
    script = tmp_path / "server.py"
    script.write_text(
        "import subprocess, sys\n"
        f"subprocess.Popen([sys.executable, '-c', {child_code!r}])\n"
        + _STDIO_SERVER,
        encoding="utf-8",
    )
    client = ExternalMCPClient(_stdio_config(script))

    await client.list_tools()
    await client.aclose()
    await asyncio.sleep(2.5)

    assert not marker.exists()


class _AuthHandler(BaseHTTPRequestHandler):
    seen: list[dict] = []

    def log_message(self, format, *args):  # noqa: A002
        return

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length") or 0)
        request = json.loads(self.rfile.read(length) or b"{}")
        type(self).seen.append(dict(self.headers))
        payload = json.dumps(
            {
                "jsonrpc": "2.0",
                "id": request.get("id"),
                "result": {
                    "tools": [
                        {
                            "name": "echo",
                            "description": "e",
                            "inputSchema": {"type": "object"},
                        }
                    ]
                },
            }
        ).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


@pytest.fixture()
def auth_server():
    _AuthHandler.seen = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), _AuthHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def _https_config(url: str, **overrides) -> ExternalMCPServerConfig:
    data = {
        "name": "authed",
        "url": url + "/rpc",
        "allowed_hosts": ["127.0.0.1"],
        "allowed_tools": ["echo"],
        "allow_insecure_http": True,
        "allow_private_network": True,
        "timeout_seconds": 3.0,
        "auth_headers": {"Authorization": "TEST_MCP_TOKEN"},
        "auth_header_prefix": "Bearer ",
    }
    data.update(overrides)
    return ExternalMCPServerConfig(**data)


async def test_auth_header_is_read_from_environment(auth_server, monkeypatch):
    monkeypatch.setenv("TEST_MCP_TOKEN", "secret-value")
    client = ExternalMCPClient(_https_config(auth_server))

    tools = await client.list_tools()

    assert [tool.name for tool in tools] == ["echo"]
    assert _AuthHandler.seen[0]["Authorization"] == "Bearer secret-value"


async def test_missing_auth_env_fails_closed(auth_server, monkeypatch):
    monkeypatch.delenv("TEST_MCP_TOKEN", raising=False)
    client = ExternalMCPClient(_https_config(auth_server))

    with pytest.raises(ExternalMCPPolicyError, match="TEST_MCP_TOKEN"):
        await client.list_tools()
    assert _AuthHandler.seen == []


def test_auth_config_rejects_literal_secret():
    with pytest.raises(ValidationError, match="环境变量名"):
        ExternalMCPServerConfig(
            name="authed",
            url="https://example.com/rpc",
            allowed_hosts=["example.com"],
            allowed_tools=["echo"],
            auth_headers={"Authorization": "Bearer abc def"},
        )

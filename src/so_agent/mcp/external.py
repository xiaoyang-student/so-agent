"""外部 MCP 安全接入（HTTPS JSON-RPC，默认关闭）。

本模块只把外部 MCP Server 当作不可信远程服务处理，安全边界固定为：

- 默认关闭：``ExternalMCPConfig.enabled`` 为 False 时不建立任何连接；
- 域名白名单：每个服务器必须声明精确 ``allowed_hosts``，URL 主机必须命中；
- 工具白名单：``tools/list`` 结果按 ``allowed_tools`` 过滤，未列出的工具
  既不可见也不可调用（默认拒绝）；
- HTTPS：默认只允许 https；http 必须显式 ``allow_insecure_http=True``；
- 内网隔离：默认拒绝解析到私有/回环/链路本地/保留地址的主机，需显式
  ``allow_private_network=True`` 才放行（内网部署场景）；
- 资源上限：请求体 / 响应体字节上限、连接与读取超时；
- 不跟随重定向，避免白名单主机把请求引向其他域名；
- 传输层可替换（``MCPTransport``），当前提供标准库 HTTPS 实现，
  SSE / 会话协商等流式传输可在不改客户端契约的前提下扩展。

外部 MCP 服务器返回的 JSON-RPC 响应会被完整校验（id 回传、结构、
内容类型与大小）；任何异常都以 ``ExternalMCPError`` 失败关闭，不做
静默降级。
"""

from __future__ import annotations

import asyncio
import ipaddress
import json
import os
import socket
import urllib.error
import urllib.parse
import urllib.request
import uuid
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, Protocol, runtime_checkable

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from so_agent.mcp.protocol import (
    INTERNAL_ERROR,
    INVALID_PARAMS,
    METHOD_TOOLS_CALL,
    METHOD_TOOLS_LIST,
    MCPError,
    MCPRequest,
    MCPResponse,
    MCPToolSchema,
    extract_result_text,
)
from so_agent.mcp.server import MCPServer
from so_agent.runtime.process_tree import terminate_process_tree

DEFAULT_CONFIG_FILENAME = "external_mcp.yaml"
_STDERR_TAIL_LINES = 20
_JSON_CONTENT_TYPES: frozenset[str] = frozenset(
    {"application/json", "application/json-rpc"}
)


class ExternalMCPError(Exception):
    """外部 MCP 接入领域错误（策略、传输或协议失败）。"""


class ExternalMCPPolicyError(ExternalMCPError):
    """外部 MCP 配置违反安全策略。"""


class ExternalMCPTransportError(ExternalMCPError):
    """外部 MCP 传输失败（网络、超时、大小或状态码）。"""


@dataclass(frozen=True)
class TransportResponse:
    """传输层原始响应（尚未解析 JSON）。"""

    status_code: int
    content_type: str
    body: bytes


@runtime_checkable
class MCPTransport(Protocol):
    """外部 MCP 传输契约（便于替换为 SSE/流式实现）。"""

    async def post(
        self,
        url: str,
        *,
        body: bytes,
        headers: Mapping[str, str],
        timeout: float,
        max_response_bytes: int,
    ) -> TransportResponse:
        """发送一次 POST 请求并返回有界响应。"""
        ...


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    """拒绝跟随重定向：3xx 交给客户端按失败处理。"""

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: D102
        return None


class HttpsJSONRPCTransport:
    """标准库 HTTPS JSON-RPC 传输（线程池执行，不阻塞事件循环）。"""

    async def post(
        self,
        url: str,
        *,
        body: bytes,
        headers: Mapping[str, str],
        timeout: float,
        max_response_bytes: int,
    ) -> TransportResponse:
        return await asyncio.to_thread(
            self._post_sync,
            url,
            body,
            dict(headers),
            timeout,
            max_response_bytes,
        )

    @staticmethod
    def _post_sync(
        url: str,
        body: bytes,
        headers: dict[str, str],
        timeout: float,
        max_response_bytes: int,
    ) -> TransportResponse:
        request_headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            **headers,
        }
        request = urllib.request.Request(
            url, data=body, headers=request_headers, method="POST"
        )
        opener = urllib.request.build_opener(_NoRedirectHandler)
        try:
            with opener.open(request, timeout=timeout) as response:
                status = int(getattr(response, "status", 200))
                content_type = response.headers.get("Content-Type", "") or ""
                raw = response.read(max_response_bytes + 1)
        except urllib.error.HTTPError as exc:
            raw = exc.read(max_response_bytes + 1)
            if len(raw) > max_response_bytes:
                raise ExternalMCPTransportError(
                    f"外部 MCP 响应超过上限（{max_response_bytes} 字节）"
                ) from exc
            return TransportResponse(
                status_code=int(exc.code),
                content_type=exc.headers.get("Content-Type", "") or "",
                body=raw,
            )
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise ExternalMCPTransportError(
                f"外部 MCP 请求失败：{type(exc).__name__}: {exc}"
            ) from exc
        if len(raw) > max_response_bytes:
            raise ExternalMCPTransportError(
                f"外部 MCP 响应超过上限（{max_response_bytes} 字节）"
            )
        return TransportResponse(
            status_code=status, content_type=content_type, body=raw
        )


class StdioJSONRPCTransport:
    """本地 MCP 子进程传输（换行分隔 JSON-RPC）。

    仅在配置显式 ``allow_stdio=True`` 时由客户端选用。首次请求前完成
    ``initialize`` / ``notifications/initialized`` 握手；子进程环境变量
    默认只保留运行必需项，额外变量必须经 ``stdio_env_passthrough`` 显式
    透传，避免把进程密钥整体泄漏给外部 Server。
    """

    _PROTOCOL_VERSION = "2024-11-05"
    _BASE_ENV_KEYS = (
        "PATH",
        "HOME",
        "LANG",
        "LC_ALL",
        "TMPDIR",
        "SYSTEMROOT",
        "SystemRoot",
        "TEMP",
        "TMP",
        "USERPROFILE",
        "COMSPEC",
        "PATHEXT",
    )

    def __init__(
        self,
        command: list[str],
        *,
        env_passthrough: list[str] | tuple[str, ...] = (),
        cwd: str | Path | None = None,
    ) -> None:
        self._command = list(command)
        self._env_passthrough = tuple(env_passthrough)
        self._cwd = str(cwd) if cwd is not None else None
        self._process: asyncio.subprocess.Process | None = None
        self._initialized = False
        self._lock = asyncio.Lock()
        self._stderr_tail: deque[str] = deque(maxlen=_STDERR_TAIL_LINES)
        self._stderr_task: asyncio.Task[None] | None = None

    def _environment(self) -> dict[str, str]:
        env = {
            key: os.environ[key]
            for key in self._BASE_ENV_KEYS
            if key in os.environ
        }
        for key in self._env_passthrough:
            if key in os.environ:
                env[key] = os.environ[key]
        return env

    async def _ensure_process(self, max_response_bytes: int) -> None:
        if self._process is not None and self._process.returncode is None:
            return
        await self._cancel_stderr_reader()
        spawn_kwargs: dict[str, Any] = {}
        if os.name != "nt":
            # 独立进程组：关闭时才能按组终止 MCP 服务器派生的子进程
            spawn_kwargs["start_new_session"] = True
        try:
            self._process = await asyncio.create_subprocess_exec(
                *self._command,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self._cwd,
                env=self._environment(),
                limit=max(64 * 1024, max_response_bytes + 1024),
                **spawn_kwargs,
            )
        except (OSError, ValueError) as exc:
            raise ExternalMCPTransportError(
                f"stdio MCP 子进程启动失败：{type(exc).__name__}: {exc}"
            ) from exc
        self._initialized = False
        self._stderr_tail.clear()
        # 必须持续排空 stderr：管道写满会让子进程阻塞，表现为响应超时
        self._stderr_task = asyncio.create_task(
            self._drain_stderr(self._process)
        )

    async def _drain_stderr(self, process: asyncio.subprocess.Process) -> None:
        """读取并保留 stderr 尾部若干行（不写日志，避免泄漏子进程输出）。"""
        stream = process.stderr
        if stream is None:
            return
        try:
            while True:
                line = await stream.readline()
                if not line:
                    return
                self._stderr_tail.append(
                    line.decode("utf-8", errors="replace").rstrip()[:500]
                )
        except asyncio.CancelledError:
            raise
        except (ValueError, OSError):
            return

    async def _cancel_stderr_reader(self) -> None:
        task = self._stderr_task
        self._stderr_task = None
        if task is None or task.done():
            return
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass

    def _stderr_hint(self) -> str:
        tail = [line for line in list(self._stderr_tail)[-3:] if line.strip()]
        return f"；stderr: {' | '.join(tail)}" if tail else ""

    async def _send(self, payload: dict[str, Any]) -> None:
        process = self._process
        if process is None or process.stdin is None:
            raise ExternalMCPTransportError("stdio MCP 子进程不可用")
        line = json.dumps(payload, ensure_ascii=False).encode("utf-8") + b"\n"
        try:
            process.stdin.write(line)
            await process.stdin.drain()
        except (BrokenPipeError, ConnectionResetError, OSError) as exc:
            raise ExternalMCPTransportError(
                f"stdio MCP 写入失败：{exc}"
            ) from exc

    async def _read_matching(
        self,
        request_id: Any,
        timeout: float,
        max_response_bytes: int,
    ) -> bytes:
        process = self._process
        if process is None or process.stdout is None:
            raise ExternalMCPTransportError("stdio MCP 子进程不可用")
        deadline = asyncio.get_running_loop().time() + max(0.1, timeout)
        for _ in range(100):
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                raise ExternalMCPTransportError(
                    "stdio MCP 读取响应超时" + self._stderr_hint()
                )
            try:
                raw = await asyncio.wait_for(
                    process.stdout.readline(), timeout=remaining
                )
            except asyncio.TimeoutError as exc:
                raise ExternalMCPTransportError(
                    "stdio MCP 读取响应超时" + self._stderr_hint()
                ) from exc
            except (ValueError, OSError) as exc:
                raise ExternalMCPTransportError(
                    f"stdio MCP 响应超过上限或读取失败：{exc}"
                ) from exc
            if not raw:
                raise ExternalMCPTransportError(
                    "stdio MCP 子进程已关闭输出" + self._stderr_hint()
                )
            if len(raw) > max_response_bytes + 1:
                raise ExternalMCPTransportError(
                    f"stdio MCP 响应超过上限（{max_response_bytes} 字节）"
                )
            try:
                payload = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ExternalMCPError("stdio MCP 输出不是合法 JSON") from exc
            if isinstance(payload, dict) and payload.get("id") == request_id:
                return json.dumps(payload, ensure_ascii=False).encode("utf-8")
        raise ExternalMCPError("stdio MCP 未返回匹配的响应 id")

    async def _handshake(self, timeout: float, max_response_bytes: int) -> None:
        init_id = f"stdio-init-{uuid.uuid4().hex[:12]}"
        await self._send(
            {
                "jsonrpc": "2.0",
                "id": init_id,
                "method": "initialize",
                "params": {
                    "protocolVersion": self._PROTOCOL_VERSION,
                    "capabilities": {},
                    "clientInfo": {"name": "so-agent", "version": "1.0.0"},
                },
            }
        )
        payload = json.loads(
            (
                await self._read_matching(
                    init_id, timeout, max_response_bytes
                )
            ).decode("utf-8")
        )
        if payload.get("error") is not None:
            raise ExternalMCPError(
                f"stdio MCP 初始化失败：{payload.get('error')}"
            )
        await self._send(
            {"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}}
        )
        self._initialized = True

    async def post(
        self,
        url: str,
        *,
        body: bytes,
        headers: Mapping[str, str],
        timeout: float,
        max_response_bytes: int,
    ) -> TransportResponse:
        try:
            payload = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ExternalMCPError("stdio MCP 请求不是合法 JSON") from exc
        async with self._lock:
            await self._ensure_process(max_response_bytes)
            if not self._initialized:
                await self._handshake(timeout, max_response_bytes)
            await self._send(payload)
            raw = await self._read_matching(
                payload.get("id"), timeout, max_response_bytes
            )
        return TransportResponse(
            status_code=200, content_type="application/json", body=raw
        )

    async def aclose(self) -> None:
        """关闭子进程及其派生进程（幂等）。"""
        process = self._process
        self._process = None
        self._initialized = False
        await self._cancel_stderr_reader()
        if process is None:
            return
        # 先杀进程树再关 stdin：优雅退出会让父进程先消失，Windows 的
        # taskkill /T 将找不到仍存活的孙进程
        await terminate_process_tree(process, wait_timeout=3)
        if process.stdin is not None:
            try:
                process.stdin.close()
            except OSError:
                pass


class ExternalMCPServerConfig(BaseModel):
    """单个外部 MCP 服务器的安全策略与资源上限。

    ``transport`` 支持 ``https``（默认，推荐）与 ``stdio``（本地子进程，
    必须显式 ``allow_stdio=True`` 且命令命中 ``stdio_allow_commands``）。
    认证头只保存环境变量名，真实值运行时从进程环境读取，避免密钥落盘。
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_.-]+$")
    transport: Literal["https", "stdio"] = "https"
    url: str = ""
    allowed_hosts: list[str] = Field(default_factory=list)
    allowed_tools: list[str] = Field(min_length=1)
    timeout_seconds: float = Field(default=30.0, gt=0, le=120)
    max_response_bytes: int = Field(default=1_000_000, ge=1024, le=20 * 1024 * 1024)
    max_request_bytes: int = Field(default=1_000_000, ge=1024, le=20 * 1024 * 1024)
    allow_insecure_http: bool = False
    allow_private_network: bool = False
    # 认证头：header 名 → 环境变量名（值不写入配置文件）
    auth_headers: dict[str, str] = Field(default_factory=dict)
    auth_header_prefix: str = Field(default="", max_length=64)
    # stdio 传输：显式启用 + 命令允许清单 + 透传的环境变量名
    allow_stdio: bool = False
    command: list[str] = Field(default_factory=list, max_length=16)
    stdio_allow_commands: list[str] = Field(default_factory=list)
    stdio_env_passthrough: list[str] = Field(default_factory=list, max_length=32)

    @model_validator(mode="after")
    def _validate_security_policy(self) -> "ExternalMCPServerConfig":
        tools = {tool.strip() for tool in self.allowed_tools if tool.strip()}
        if not tools:
            raise ValueError("allowed_tools 不能为空（默认拒绝全部工具）")
        self._validate_auth_headers()
        if self.transport == "stdio":
            self._validate_stdio_policy()
            return self
        self._validate_https_policy()
        return self

    def _validate_auth_headers(self) -> None:
        for header, env_name in self.auth_headers.items():
            if not header.strip() or any(ch.isspace() for ch in header):
                raise ValueError(f"认证头名称非法：{header!r}")
            if not env_name.strip() or not env_name.replace("_", "").isalnum():
                raise ValueError(
                    f"认证头 {header!r} 必须映射到环境变量名，而不是字面密钥"
                )

    def _validate_https_policy(self) -> None:
        parts = urllib.parse.urlsplit(self.url.strip())
        if parts.scheme not in {"https", "http"}:
            raise ValueError("外部 MCP URL 必须使用 https（或显式允许的 http）")
        if parts.scheme == "http" and not self.allow_insecure_http:
            raise ValueError(
                "外部 MCP 默认要求 https；确需 http 必须显式 allow_insecure_http=True"
            )
        if not parts.hostname:
            raise ValueError("外部 MCP URL 缺少主机名")
        if parts.username or parts.password:
            raise ValueError("外部 MCP URL 不得内嵌用户名或密码")
        if parts.fragment:
            raise ValueError("外部 MCP URL 不得包含 fragment")

        hosts = {
            host.strip().casefold().rstrip(".")
            for host in self.allowed_hosts
            if host and host.strip()
        }
        if not hosts:
            raise ValueError("allowed_hosts 不能为空")
        for host in hosts:
            if any(marker in host for marker in ("*", "/", "\\")):
                raise ValueError("allowed_hosts 只支持精确主机名，不支持通配符")
        hostname = parts.hostname.casefold().rstrip(".")
        if hostname not in hosts:
            raise ValueError(
                f"URL 主机 {hostname!r} 不在 allowed_hosts 白名单：{sorted(hosts)}"
            )

    def _validate_stdio_policy(self) -> None:
        if not self.allow_stdio:
            raise ValueError(
                "stdio 传输默认禁止；确需本地子进程必须显式 allow_stdio=True"
            )
        if self.url:
            raise ValueError("stdio 传输不使用 url")
        if not self.command:
            raise ValueError("stdio 传输必须提供 command")
        allowed = {
            item.strip() for item in self.stdio_allow_commands if item.strip()
        }
        if not allowed:
            raise ValueError("stdio 传输必须配置 stdio_allow_commands 允许清单")
        executable = self.command[0]
        if executable not in allowed and Path(executable).name not in allowed:
            raise ValueError(
                f"stdio 命令 {executable!r} 不在 stdio_allow_commands 允许清单"
            )

    @property
    def hostname(self) -> str:
        if self.transport != "https":
            return ""
        return (urllib.parse.urlsplit(self.url).hostname or "").casefold().rstrip(".")

    @property
    def allowed_tool_set(self) -> frozenset[str]:
        return frozenset(
            tool.strip() for tool in self.allowed_tools if tool.strip()
        )


class ExternalMCPConfig(BaseModel):
    """外部 MCP 配置文件契约（默认关闭）。"""

    model_config = ConfigDict(frozen=True, extra="forbid")

    enabled: bool = False
    servers: list[ExternalMCPServerConfig] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_servers(self) -> "ExternalMCPConfig":
        if self.enabled and not self.servers:
            raise ValueError("external_mcp.enabled=True 时至少配置一个 server")
        names = [server.name for server in self.servers]
        if len(set(names)) != len(names):
            raise ValueError("外部 MCP server 名称必须唯一")
        return self


def load_external_mcp_config(
    path: str | Path,
    *,
    enabled: bool | None = None,
) -> ExternalMCPConfig:
    """从 YAML 文件加载外部 MCP 配置。

    Args:
        path: 配置文件路径（``DEFAULT_CONFIG_FILENAME`` 的 YAML 结构）。
        enabled: 显式覆盖文件中的 enabled（None 表示以文件为准）。

    Raises:
        ExternalMCPPolicyError: 文件不可读、YAML 非法或违反安全策略时。
    """
    config_path = Path(path).expanduser()
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ExternalMCPPolicyError(f"外部 MCP 配置不可读：{exc}") from exc
    except yaml.YAMLError as exc:
        raise ExternalMCPPolicyError(f"外部 MCP 配置 YAML 非法：{exc}") from exc
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ExternalMCPPolicyError("外部 MCP 配置顶层必须是对象")
    data = dict(raw)
    if enabled is not None:
        data["enabled"] = enabled
    try:
        return ExternalMCPConfig.model_validate(data)
    except ValidationError as exc:
        raise ExternalMCPPolicyError(f"外部 MCP 配置违反安全策略：{exc}") from exc


class ExternalMCPClient:
    """受策略约束的外部 MCP 客户端。"""

    def __init__(
        self,
        config: ExternalMCPServerConfig,
        *,
        transport: MCPTransport | None = None,
    ) -> None:
        self._config = config
        if transport is not None:
            self._transport: MCPTransport = transport
        elif config.transport == "stdio":
            self._transport = StdioJSONRPCTransport(
                config.command,
                env_passthrough=config.stdio_env_passthrough,
            )
        else:
            self._transport = HttpsJSONRPCTransport()

    @property
    def config(self) -> ExternalMCPServerConfig:
        return self._config

    @property
    def allowed_tools(self) -> frozenset[str]:
        return self._config.allowed_tool_set

    def _request_headers(self) -> dict[str, str]:
        """解析认证头；环境变量缺失时失败关闭，绝不发送空凭据。"""
        headers = {"X-So-Agent-Server": self._config.name}
        for header, env_name in self._config.auth_headers.items():
            value = os.environ.get(env_name, "").strip()
            if not value:
                raise ExternalMCPPolicyError(
                    f"外部 MCP 认证环境变量未设置：{env_name}"
                )
            headers[header] = f"{self._config.auth_header_prefix}{value}"
        return headers

    async def aclose(self) -> None:
        """释放传输层资源（stdio 子进程等）；未提供时为空操作。"""
        close = getattr(self._transport, "aclose", None)
        if callable(close):
            await close()

    async def list_tools(self) -> list[MCPToolSchema]:
        """发现工具并按白名单过滤（默认拒绝）。"""
        response = await self._rpc(METHOD_TOOLS_LIST, {})
        if response.error is not None:
            raise ExternalMCPError(
                f"外部 MCP 工具发现失败：{response.error.message}"
            )
        result = response.result
        tools_raw = result.get("tools") if isinstance(result, dict) else result
        if not isinstance(tools_raw, list):
            raise ExternalMCPError("外部 MCP tools/list 结果缺少 tools 数组")
        tools: list[MCPToolSchema] = []
        for item in tools_raw:
            if not isinstance(item, dict):
                raise ExternalMCPError("外部 MCP 工具描述必须是 JSON 对象")
            try:
                schema = MCPToolSchema.model_validate(item)
            except ValidationError as exc:
                raise ExternalMCPError(
                    f"外部 MCP 工具描述非法：{exc}"
                ) from exc
            if schema.name in self.allowed_tools:
                tools.append(schema)
        return tools

    async def call_tool(
        self, tool_name: str, arguments: dict[str, Any] | None = None
    ) -> MCPResponse:
        """调用白名单内工具；未授权工具直接拒绝，不发起网络请求。"""
        if tool_name not in self.allowed_tools:
            raise ExternalMCPError(
                f"工具 {tool_name!r} 未列入 allow_tools 白名单（默认拒绝）"
            )
        if arguments is not None and not isinstance(arguments, dict):
            raise ExternalMCPError("工具参数必须是 JSON 对象")
        return await self._rpc(
            METHOD_TOOLS_CALL,
            {"name": tool_name, "arguments": dict(arguments or {})},
        )

    async def _rpc(self, method: str, params: dict[str, Any]) -> MCPResponse:
        request = MCPRequest(
            method=method,
            params=params,
            id=f"ext-{uuid.uuid4().hex[:12]}",
        )
        body = json.dumps(
            request.model_dump(mode="json"), ensure_ascii=False
        ).encode("utf-8")
        if len(body) > self._config.max_request_bytes:
            raise ExternalMCPError(
                f"外部 MCP 请求超过上限（{self._config.max_request_bytes} 字节）"
            )
        headers = self._request_headers()
        if self._config.transport == "https":
            await self._validate_target()
        response = await self._transport.post(
            self._config.url,
            body=body,
            headers=headers,
            timeout=self._config.timeout_seconds,
            max_response_bytes=self._config.max_response_bytes,
        )
        if not 200 <= response.status_code < 300:
            hint = "（不跟随重定向）" if 300 <= response.status_code < 400 else ""
            raise ExternalMCPTransportError(
                f"外部 MCP 返回 HTTP {response.status_code}{hint}"
            )
        content_type = response.content_type.split(";", 1)[0].strip().lower()
        if content_type not in _JSON_CONTENT_TYPES:
            raise ExternalMCPError(
                f"外部 MCP 响应内容类型不受支持：{content_type or '缺失'}"
            )
        if len(response.body) > self._config.max_response_bytes:
            raise ExternalMCPTransportError(
                f"外部 MCP 响应超过上限（{self._config.max_response_bytes} 字节）"
            )
        try:
            payload = json.loads(response.body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ExternalMCPError("外部 MCP 响应不是合法 JSON") from exc
        if not isinstance(payload, dict):
            raise ExternalMCPError("外部 MCP 响应必须是 JSON-RPC 对象")
        try:
            mcp_response = MCPResponse.model_validate(payload)
        except ValidationError as exc:
            raise ExternalMCPError(f"外部 MCP 响应结构非法：{exc}") from exc
        if mcp_response.id != request.id:
            raise ExternalMCPError(
                f"外部 MCP 响应 id 不匹配：期望 {request.id!r}，"
                f"实际 {mcp_response.id!r}"
            )
        if (
            mcp_response.error is None
            and mcp_response.result is None
        ):
            raise ExternalMCPError("外部 MCP 响应缺少 result 与 error")
        return mcp_response

    async def _validate_target(self) -> None:
        """请求前复核 URL 与解析地址（防止 DNS 指向内网）。"""
        parts = urllib.parse.urlsplit(self._config.url)
        if parts.scheme == "https":
            pass
        elif parts.scheme == "http" and self._config.allow_insecure_http:
            pass
        else:
            raise ExternalMCPPolicyError("外部 MCP URL 协议不允许")
        host = (parts.hostname or "").casefold().rstrip(".")
        allowed_hosts = {
            item.strip().casefold().rstrip(".")
            for item in self._config.allowed_hosts
        }
        if host not in allowed_hosts:
            raise ExternalMCPPolicyError(f"外部 MCP 主机不在白名单：{host!r}")
        if self._config.allow_private_network:
            return
        port = parts.port or (443 if parts.scheme == "https" else 80)
        try:
            infos = await asyncio.to_thread(
                socket.getaddrinfo, host, port, 0, socket.SOCK_STREAM
            )
        except OSError as exc:
            raise ExternalMCPTransportError(
                f"外部 MCP 主机解析失败：{host}：{exc}"
            ) from exc
        for info in infos:
            address = str(info[4][0]).split("%", 1)[0]
            try:
                ip = ipaddress.ip_address(address)
            except ValueError as exc:
                raise ExternalMCPTransportError(
                    f"外部 MCP 主机解析结果非法：{address!r}"
                ) from exc
            if not ip.is_global:
                raise ExternalMCPPolicyError(
                    f"外部 MCP 主机 {host} 解析到非公网地址 {ip}；"
                    "如确为内网服务需显式 allow_private_network=True"
                )


class ExternalMCPServer(MCPServer):
    """把外部 MCP 客户端包装为进程内 MCPServer（可接入 MCPClient）。"""

    def __init__(self, client: ExternalMCPClient) -> None:
        super().__init__(client.config.name)
        self._client = client

    @property
    def client(self) -> ExternalMCPClient:
        return self._client

    async def aclose(self) -> None:
        await self._client.aclose()

    async def discover(self) -> list[MCPToolSchema]:
        """拉取外部工具并按白名单注册；可重复调用刷新。"""
        schemas = await self._client.list_tools()
        self._tools.clear()
        for schema in schemas:
            self.register_tool(schema, self._handler_for(schema.name))
        return list(schemas)

    def _handler_for(self, tool_name: str):
        async def _invoke(arguments: dict[str, Any]) -> str:
            response = await self._client.call_tool(tool_name, arguments)
            if response.error is not None:
                message = f"外部工具 {tool_name!r} 失败：{response.error.message}"
                if response.error.code == INVALID_PARAMS:
                    raise ValueError(message)
                raise RuntimeError(message)
            return extract_result_text(response)

        return _invoke


async def discover_external_servers(
    config: ExternalMCPConfig,
    *,
    transport: MCPTransport | None = None,
) -> list[ExternalMCPServer]:
    """按配置建立外部 MCP 连接并完成工具发现（默认关闭时返回空列表）。"""
    if not config.enabled:
        return []
    servers: list[ExternalMCPServer] = []
    try:
        for server_config in config.servers:
            server = ExternalMCPServer(
                ExternalMCPClient(server_config, transport=transport)
            )
            await server.discover()
            servers.append(server)
    except Exception:
        for server in servers:
            await server.aclose()
        raise
    return servers


def external_mcp_error_response(
    exc: ExternalMCPError,
    *,
    request_id: str | int | None = None,
) -> MCPResponse:
    """把接入错误映射为标准 MCP 错误响应。"""
    code = INVALID_PARAMS if isinstance(exc, ExternalMCPPolicyError) else INTERNAL_ERROR
    return MCPResponse(id=request_id, error=MCPError(code=code, message=str(exc)))


__all__ = [
    "DEFAULT_CONFIG_FILENAME",
    "ExternalMCPClient",
    "ExternalMCPConfig",
    "ExternalMCPError",
    "ExternalMCPPolicyError",
    "ExternalMCPServer",
    "ExternalMCPServerConfig",
    "ExternalMCPTransportError",
    "HttpsJSONRPCTransport",
    "MCPTransport",
    "StdioJSONRPCTransport",
    "TransportResponse",
    "discover_external_servers",
    "external_mcp_error_response",
    "load_external_mcp_config",
]

"""外部 MCP Server 目录：扫描、签名、注册与可选轮询监听。

约定目录：``{tool_store}/mcp_servers/<server>/server.yaml``。清单字段与
``ExternalMCPServerConfig`` 一致，另加 ``enabled``（默认 false）与
``transport``（``https_jsonrpc``/``https`` 或显式启用的 ``stdio``）。复制
新 Server 后，任务开始时按目录签名判断是否需要重新扫描；也可显式强制
刷新。

安全边界：默认关闭、工具白名单、HTTPS 精确主机白名单、不跟随重定向、
拒绝内网地址，全部复用 ``so_agent.mcp.external`` 的既有策略；目录里
的单条配置非法时只跳过该条，不影响其他 Server。
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from so_agent.mcp.external import (
    ExternalMCPClient,
    ExternalMCPServer,
    ExternalMCPServerConfig,
)
from so_agent.mcp.protocol import INVALID_PARAMS, MCPToolSchema, extract_result_text
from so_agent.mcp.server import MCPToolServer

logger = logging.getLogger("so_agent.mcp.directory")

MANIFEST_FILENAME = "server.yaml"
_MAX_MANIFEST_BYTES = 64_000
_SUPPORTED_TRANSPORTS = frozenset({"https_jsonrpc", "https", "stdio"})


@dataclass
class MCPDirectoryEntry:
    """单个目录条目的扫描结果。"""

    directory: Path
    name: str
    enabled: bool = False
    config: ExternalMCPServerConfig | None = None
    error: str = ""


@dataclass
class MCPDirectoryReport:
    """一次目录刷新/扫描的结果。"""

    server_dir: Path
    scanned: int = 0
    registered: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    changed: bool = False


def scan_mcp_directory(server_dir: str | Path) -> list[MCPDirectoryEntry]:
    """扫描目录并校验每个 Server 清单（不做网络发现）。"""
    root = Path(server_dir)
    entries: list[MCPDirectoryEntry] = []
    if not root.is_dir():
        return entries
    for child in sorted(root.iterdir(), key=lambda item: item.name):
        if not child.is_dir() or child.name.startswith("."):
            continue
        manifest = child / MANIFEST_FILENAME
        if not manifest.is_file():
            continue
        entry = MCPDirectoryEntry(directory=child, name=child.name)
        try:
            if manifest.stat().st_size > _MAX_MANIFEST_BYTES:
                entry.error = f"清单超过大小上限 {_MAX_MANIFEST_BYTES} 字节"
                entries.append(entry)
                continue
            raw = yaml.safe_load(manifest.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                entry.error = "清单必须是 YAML 对象"
                entries.append(entry)
                continue
            enabled = bool(raw.pop("enabled", False))
            transport = str(raw.pop("transport", "https_jsonrpc")).strip().lower()
            if transport not in _SUPPORTED_TRANSPORTS:
                entry.error = f"不支持的 transport：{transport}"
                entries.append(entry)
                continue
            declared_name = str(raw.get("name") or child.name).strip()
            if declared_name != child.name:
                entry.error = (
                    f"清单 name={declared_name!r} 与目录名 {child.name!r} 不一致"
                )
                entries.append(entry)
                continue
            raw["name"] = child.name
            raw["transport"] = (
                "stdio" if transport == "stdio" else "https"
            )
            entry.config = ExternalMCPServerConfig(**raw)
            entry.name = declared_name
            entry.enabled = enabled
        except (OSError, yaml.YAMLError, ValidationError, TypeError) as exc:
            entry.error = f"清单非法：{exc}"
        entries.append(entry)
    return entries


def directory_signature(server_dir: str | Path) -> str:
    """计算目录轻量签名（仅清单文件的相对路径/大小/mtime）。"""
    root = Path(server_dir)
    digest = hashlib.sha256()
    if not root.is_dir():
        return ""
    for manifest in sorted(root.glob(f"*/{MANIFEST_FILENAME}")):
        try:
            stat = manifest.stat()
        except OSError:
            continue
        digest.update(
            f"{manifest.relative_to(root).as_posix()}\0{stat.st_size}\0{stat.st_mtime_ns}\n".encode(
                "utf-8"
            )
        )
    return digest.hexdigest()


class MCPDirectoryManager:
    """把目录中的外部 MCP Server 注册进工具注册表（按签名增量刷新）。"""

    def __init__(
        self, server_dir: str | Path, *, external_timeout_grace: float = 5.0
    ) -> None:
        self._server_dir = Path(server_dir)
        self._signature = ""
        self._loaded_tools: set[str] = set()
        self._report = MCPDirectoryReport(server_dir=self._server_dir)
        self._timeout_grace = external_timeout_grace

    @property
    def server_dir(self) -> Path:
        return self._server_dir

    @property
    def report(self) -> MCPDirectoryReport:
        return self._report

    async def refresh(
        self, registry: Any, *, force: bool = False
    ) -> MCPDirectoryReport:
        """按需刷新目录；签名未变化且非强制时直接复用上次结果。"""
        signature = directory_signature(self._server_dir)
        if not force and signature == self._signature and self._signature:
            return self._report

        report = MCPDirectoryReport(server_dir=self._server_dir, changed=True)
        entries = scan_mcp_directory(self._server_dir)
        report.scanned = len(entries)

        for tool_name in sorted(self._loaded_tools):
            registry.unregister_external_tool(tool_name)
        self._loaded_tools.clear()

        for entry in entries:
            if entry.error:
                report.errors.append(f"{entry.name}: {entry.error}")
                continue
            if not entry.enabled or entry.config is None:
                report.skipped.append(f"{entry.name}（未启用）")
                continue
            server = ExternalMCPServer(ExternalMCPClient(entry.config))
            try:
                schemas = await asyncio.wait_for(
                    server.discover(),
                    timeout=float(entry.config.timeout_seconds) + self._timeout_grace,
                )
            except Exception as exc:  # 单个 Server 失败不影响其他条目
                report.errors.append(f"{entry.name}: 工具发现失败：{exc}")
                logger.warning("外部 MCP Server %s 发现失败：%s", entry.name, exc)
                continue
            for schema in schemas:
                full_name = f"{entry.name}.{schema.name}"
                existing = registry.get_tool(full_name)
                if existing is not None and existing.get("source") != "external_mcp":
                    report.errors.append(
                        f"{full_name}: 与已有工具重名（来源 {existing.get('source')}）"
                    )
                    continue
                proxy = self._build_proxy(server, entry.name, full_name, schema)
                try:
                    registry.register_external_tool(
                        full_name,
                        schema.description or f"外部 MCP 工具（{entry.name}）",
                        schema,
                        proxy,
                        server_name=entry.name,
                    )
                except Exception as exc:
                    report.errors.append(f"{full_name}: 注册失败：{exc}")
                    continue
                self._loaded_tools.add(full_name)
                report.registered.append(full_name)

        self._signature = signature
        self._report = report
        return report

    @staticmethod
    def _build_proxy(
        server: ExternalMCPServer,
        server_name: str,
        full_name: str,
        schema: MCPToolSchema,
    ) -> MCPToolServer:
        proxy = MCPToolServer(name=full_name)

        async def _handler(arguments: dict[str, Any]) -> str:
            response = await server.client.call_tool(schema.name, arguments)
            if response.error is not None:
                message = (
                    f"外部 MCP 工具 {server_name}.{schema.name} 失败："
                    f"{response.error.message}"
                )
                if response.error.code == INVALID_PARAMS:
                    raise ValueError(message)
                raise RuntimeError(message)
            return extract_result_text(response)

        proxy.register_tool(
            schema.model_copy(update={"name": full_name}), _handler
        )
        return proxy


class MCPDirectoryWatcher:
    """可选的轮询监听：只做扫描与变更告警，不直接注册工具。

    工具注册仍由任务开始时的刷新完成，避免目录变化影响运行中的任务。
    """

    def __init__(
        self,
        server_dir: str | Path,
        *,
        interval_seconds: float = 5.0,
        on_change: Any | None = None,
    ) -> None:
        self._server_dir = Path(server_dir)
        self._interval = max(1.0, float(interval_seconds))
        self._on_change = on_change
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._signature = directory_signature(self._server_dir)

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name="mcp-directory-watcher", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=self._interval + 1.0)
        self._thread = None

    def _run(self) -> None:
        while not self._stop.wait(self._interval):
            signature = directory_signature(self._server_dir)
            if signature == self._signature:
                continue
            self._signature = signature
            entries = scan_mcp_directory(self._server_dir)
            enabled = sum(1 for entry in entries if entry.enabled and not entry.error)
            errors = [f"{entry.name}:{entry.error}" for entry in entries if entry.error]
            logger.info(
                "检测到 MCP 目录变更：共 %s 个条目，%s 个启用，错误 %s",
                len(entries),
                enabled,
                errors or "无",
            )
            if callable(self._on_change):
                try:
                    self._on_change(entries)
                except Exception:
                    logger.exception("MCP 目录变更回调失败")


__all__ = [
    "MCPDirectoryEntry",
    "MCPDirectoryManager",
    "MCPDirectoryReport",
    "MCPDirectoryWatcher",
    "MANIFEST_FILENAME",
    "directory_signature",
    "scan_mcp_directory",
]

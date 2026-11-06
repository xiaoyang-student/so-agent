"""主控侧工具存储（ToolStore）路径解析与沙箱交付。

工具存储与执行环境解耦：生成工具源码、清单、外部 MCP 目录都属于
主控侧持久存储，不随沙箱生命周期变化；沙箱只接收只读执行副本。
未来云沙箱只需实现可选的 ``stage_tool(path)`` 异步方法，返回执行侧
路径（例如上传后的对象路径），准入、审核、生命周期逻辑无需改动。
"""

from __future__ import annotations

import contextlib
import inspect
import os
import random
import time
from pathlib import Path
from typing import Any, Iterator

from so_agent.config import Settings, get_settings

DEFAULT_TOOL_STORE_DIR = "tools"
GENERATED_SUBDIR = "generated"
MCP_SERVERS_SUBDIR = "mcp_servers"
ARCHIVE_SUBDIR = "archive"

# 写锁最长等待与陈旧锁判定（秒）。清单读改写都是毫秒级操作，
# 超过陈旧阈值仍存在的锁视为崩溃残留，允许接管。
_LOCK_TIMEOUT_SECONDS = 10.0
_LOCK_STALE_SECONDS = 60.0


class ToolStoreLockError(RuntimeError):
    """工具库写锁获取失败（并发写超过等待上限）。"""


def resolve_tool_store_dir(settings: Settings | None = None) -> Path:
    """返回工具存储根目录（相对路径以当前工作目录为基）。"""
    resolved = settings or get_settings()
    raw = Path(getattr(resolved, "tool_store_dir", DEFAULT_TOOL_STORE_DIR) or DEFAULT_TOOL_STORE_DIR)
    return raw.expanduser().resolve() if raw.is_absolute() else (Path.cwd() / raw).resolve()


def resolve_generated_tools_dir(settings: Settings | None = None) -> Path:
    """返回生成工具源码目录：``{tool_store}/generated``。"""
    return resolve_tool_store_dir(settings) / GENERATED_SUBDIR


def resolve_generated_registry_path(settings: Settings | None = None) -> Path:
    """返回生成工具清单：``{tool_store}/generated/registry.json``。"""
    return resolve_generated_tools_dir(settings) / "registry.json"


def resolve_mcp_servers_dir(settings: Settings | None = None) -> Path:
    """返回外部 MCP Server 声明目录（可被 ``mcp_servers_dir`` 覆盖）。"""
    resolved = settings or get_settings()
    configured = str(getattr(resolved, "mcp_servers_dir", "") or "").strip()
    if configured:
        raw = Path(configured).expanduser()
        return raw.resolve() if raw.is_absolute() else (Path.cwd() / raw).resolve()
    return resolve_tool_store_dir(resolved) / MCP_SERVERS_SUBDIR


def resolve_archive_dir(settings: Settings | None = None) -> Path:
    """返回过期工具归档目录：``{tool_store}/archive``。"""
    return resolve_tool_store_dir(settings) / ARCHIVE_SUBDIR


def ensure_generated_store(settings: Settings | None = None) -> Path:
    """确保生成工具目录存在并返回。"""
    directory = resolve_generated_tools_dir(settings)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def atomic_write_text(path: str | Path, text: str) -> None:
    """原子写入文本（同目录临时文件 + ``os.replace``）。"""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        temp.write_text(text, encoding="utf-8")
        os.replace(temp, target)
    finally:
        try:
            temp.unlink(missing_ok=True)
        except OSError:
            pass


@contextlib.contextmanager
def registry_write_lock(
    registry_path: str | Path,
    *,
    timeout: float = _LOCK_TIMEOUT_SECONDS,
) -> Iterator[None]:
    """跨进程互斥锁（``registry.json.lock``），串行化清单读改写。

    采用 ``O_CREAT|O_EXCL`` 创建锁文件，Windows 与 POSIX 行为一致；
    进程崩溃遗留的锁超过陈旧阈值后可被接管。锁只保护清单文件，
    不保护源码文件移动（归档与执行仍以哈希校验兜底）。

    Raises:
        ToolStoreLockError: 等待超过 ``timeout`` 秒仍无法获得锁。
    """
    target = Path(registry_path)
    lock_path = target.with_name(f"{target.name}.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + max(0.1, float(timeout))
    handle: int | None = None
    while True:
        try:
            handle = os.open(
                lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600
            )
            os.write(handle, f"{os.getpid()} {time.time():.6f}".encode("ascii"))
            break
        except FileExistsError:
            try:
                if time.time() - lock_path.stat().st_mtime > _LOCK_STALE_SECONDS:
                    lock_path.unlink(missing_ok=True)
                    continue
            except OSError:
                pass
            if time.monotonic() >= deadline:
                raise ToolStoreLockError(
                    f"等待工具清单写锁超时（{timeout:g}s）：{lock_path}"
                ) from None
            time.sleep(0.02 + random.random() * 0.02)
    try:
        yield
    finally:
        if handle is not None:
            try:
                os.close(handle)
            except OSError:
                pass
        try:
            lock_path.unlink(missing_ok=True)
        except OSError:
            pass


async def stage_tool_for_sandbox(
    sandbox: Any, tool_path: str | Path
) -> Path:
    """把工具源码交付到沙箱执行侧。

    默认直接返回本地路径（local 验证沙箱）。云沙箱后端实现可选的
    ``stage_tool(path)`` 后，返回其上传后的执行侧路径；异常按调用方
    的错误处理路径向上抛出，绝不静默回退到本地执行。
    """
    path = Path(tool_path)
    stage = getattr(sandbox, "stage_tool", None)
    if not callable(stage):
        return path
    staged = stage(path)
    if inspect.isawaitable(staged):
        staged = await staged
    return Path(staged)

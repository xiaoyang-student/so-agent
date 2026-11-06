"""子进程树终止：沙箱执行与外部 MCP 传输共用。

POSIX 依赖调用方以 ``start_new_session=True`` 启动子进程，才能按进程组
终止；Windows 使用 ``taskkill /T`` 递归结束子进程。
"""

from __future__ import annotations

import asyncio
import os
import signal


async def terminate_process_tree(
    process: asyncio.subprocess.Process | None,
    *,
    wait_timeout: float | None = None,
) -> None:
    """终止子进程及其派生进程；进程已退出时为空操作。"""
    if process is None or process.returncode is not None:
        return
    if os.name == "nt":
        try:
            killer = await asyncio.create_subprocess_exec(
                "taskkill",
                "/PID",
                str(process.pid),
                "/T",
                "/F",
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await asyncio.wait_for(killer.wait(), timeout=5)
        except (OSError, asyncio.TimeoutError):
            pass
        if process.returncode is None:
            try:
                process.kill()
            except ProcessLookupError:
                pass
    else:
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            try:
                process.kill()
            except ProcessLookupError:
                pass
    if wait_timeout is not None:
        try:
            await asyncio.wait_for(process.wait(), timeout=wait_timeout)
        except (asyncio.TimeoutError, OSError):
            pass


__all__ = ["terminate_process_tree"]

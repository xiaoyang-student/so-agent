"""沙箱后端共享的路径与输出解码工具。"""

from __future__ import annotations

import re
from pathlib import Path


class SandboxError(Exception):
    """沙箱领域错误：非法标识符、路径越界、空代码、后端不可用等。"""


class SandboxBackendError(SandboxError):
    """沙箱后端解析 / 构造错误（未知后端、后端不可用等）。"""


class SandboxPathError(SandboxError):
    """沙箱标识符或路径越界。"""


def resolve_run_dir(
    root: Path,
    task_id: str,
    run_id: str,
    id_pattern: re.Pattern[str],
) -> Path:
    """校验标识符并返回位于沙箱根目录内的运行目录。"""
    for value, label in ((task_id, "task_id"), (run_id, "run_id")):
        if not isinstance(value, str) or not id_pattern.match(value):
            raise SandboxPathError(
                f"{label} 非法：{value!r}；只允许字母/数字/下划线/连字符，长度 1-64"
            )
    resolved = (root / f"task_{task_id}" / f"run_{run_id}").resolve()
    if not resolved.is_relative_to(root):
        raise SandboxPathError(
            f"运行目录越界：{resolved} 不在沙箱根目录 {root} 之内"
        )
    return resolved


def decode_output(data: bytes, limit: int) -> tuple[str, bool]:
    """按字节上限截断并解码 UTF-8 输出。"""
    limit = int(limit)
    truncated = limit > 0 and len(data) > limit
    if truncated:
        data = data[:limit]
    text = data.decode("utf-8", errors="replace")
    if truncated:
        text += f"\n...[输出超过 sandbox_output_limit（{limit} 字节），已截断]"
    return text, truncated

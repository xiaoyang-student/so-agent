"""子任务写作用域：声明路径归一化、产出验收与写时校验。

验收以文件系统事实为准：声明为文件的条目必须存在且非空；声明为目录的
条目必须包含至少一个文件；通配符条目必须至少匹配一个文件。可选的
``since`` 时间戳用于要求产出发生在本次尝试开始之后，避免上一轮残留
文件冒充本次结果。
"""

from __future__ import annotations

import fnmatch
import re
from pathlib import Path

GLOB_CHARS: tuple[str, ...] = ("*", "?", "[")

# 文件系统时间戳分辨率与调度抖动容忍度（秒）
_SINCE_TOLERANCE = 2.0

_ABSOLUTE_PATH_PATTERN = re.compile(r"^[A-Za-z]:")


def normalize_entry(entry: str) -> str | None:
    """规范化作用域条目；空条目、绝对路径、含 ``..`` 的路径返回 None。"""
    raw = str(entry).strip().replace("\\", "/")
    while raw.startswith("./"):
        raw = raw[2:]
    if not raw or raw.startswith("/") or _ABSOLUTE_PATH_PATTERN.match(raw):
        return None
    if any(part == ".." for part in raw.split("/")):
        return None
    return raw


def _fresh_enough(target: Path, since: float | None) -> bool:
    if since is None:
        return True
    try:
        return target.stat().st_mtime >= since - _SINCE_TOLERANCE
    except OSError:
        return False


def _usable_file(target: Path, since: float | None) -> bool:
    try:
        return target.is_file() and target.stat().st_size > 0 and _fresh_enough(
            target, since
        )
    except OSError:
        return False


def entry_satisfied(root: Path, entry: str, *, since: float | None = None) -> bool:
    """判断单个作用域条目是否已由本次尝试产出（非空且足够新）。"""
    normalized = normalize_entry(entry)
    if normalized is None:
        return False
    if any(char in normalized for char in GLOB_CHARS):
        try:
            return any(
                _usable_file(match, since) for match in root.glob(normalized)
            )
        except (ValueError, OSError):
            return False
    target = root / normalized
    try:
        target.resolve().relative_to(root.resolve())
    except (ValueError, OSError):
        return False
    if target.is_file():
        return _usable_file(target, since)
    if target.is_dir():
        try:
            return any(
                _usable_file(item, since) for item in target.rglob("*")
            )
        except OSError:
            return False
    return False


def missing_scope_entries(
    root: Path, scope: list[str], *, since: float | None = None
) -> list[str]:
    """返回未满足的作用域条目（保持声明顺序并去重）。"""
    missing: list[str] = []
    for entry in scope:
        if entry in missing:
            continue
        if not entry_satisfied(root, entry, since=since):
            missing.append(entry)
    return missing


def path_allowed(scope: list[str], relative_path: str) -> bool:
    """写时校验：路径是否落在本子任务声明的写入范围内。

    空作用域表示调用方未声明限制（保持既有行为）；条目本身可以是文件、
    目录前缀或通配符。
    """
    if not scope:
        return True
    normalized = normalize_entry(relative_path)
    if normalized is None:
        return False
    for entry in scope:
        candidate = normalize_entry(entry)
        if candidate is None:
            continue
        if any(char in candidate for char in GLOB_CHARS):
            if fnmatch.fnmatch(normalized, candidate):
                return True
            continue
        if normalized == candidate or normalized.startswith(candidate + "/"):
            return True
    return False


__all__ = [
    "GLOB_CHARS",
    "entry_satisfied",
    "missing_scope_entries",
    "normalize_entry",
    "path_allowed",
]

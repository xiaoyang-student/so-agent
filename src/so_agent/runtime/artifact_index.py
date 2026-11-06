"""确定性沙箱产物索引：只列浅层文件元数据，不读取文件内容。

子 Agent 在不知道依赖产物名称时容易靠猜路径，索引用于把"沙箱里现在
有什么"以确定性文本注入提示词/输入。索引只包含沙箱根目录下的常规
文件（不递归、不跟随目录），默认按相对路径排序，最多 ``max_entries`` 条。

``includes`` 用于把索引收敛到与本子任务相关的文件（写入范围 / 缓存
脚本）；未匹配的空位由最近修改的产物补齐，避免索引随沙箱文件增长而
无限放大每轮上下文。
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_MAX_ENTRIES: int = 80


def build_artifact_index(
    root: Path,
    *,
    max_entries: int = DEFAULT_MAX_ENTRIES,
    includes: Iterable[str] | None = None,
) -> str:
    """返回沙箱浅层文件索引文本；无文件或目录不可读时返回空字符串。"""
    if max_entries <= 0:
        return ""
    try:
        root = Path(root).expanduser().resolve()
        entries = [item for item in root.iterdir() if item.is_file()]
    except OSError:
        return ""
    if not entries:
        return ""

    patterns = [str(item) for item in includes or () if str(item).strip()]
    if patterns:
        from so_agent.runtime.write_scope import path_allowed

        matched = [item for item in entries if path_allowed(patterns, item.name)]
        if len(matched) > max_entries:
            matched.sort(key=_mtime, reverse=True)
            matched = matched[:max_entries]
        elif len(matched) < max_entries:
            rest = [item for item in entries if item not in matched]
            rest.sort(key=_mtime, reverse=True)
            matched.extend(rest[: max_entries - len(matched)])
        entries = matched
    entries.sort(key=lambda item: item.name.casefold())

    lines = ["当前沙箱已有文件（仅浅层文件名与元数据，不含内容）："]
    for item in entries[:max_entries]:
        try:
            stat = item.stat()
        except OSError:
            continue
        modified = datetime.fromtimestamp(
            stat.st_mtime, tz=timezone.utc
        ).strftime("%Y-%m-%dT%H:%M:%SZ")
        lines.append(f"- {item.name} ({stat.st_size} bytes, {modified})")
    if len(entries) > max_entries:
        lines.append(f"…（其余 {len(entries) - max_entries} 个文件未列出）")
    return "\n".join(lines)


def _mtime(item: Path) -> float:
    try:
        return item.stat().st_mtime
    except OSError:
        return 0.0


__all__ = ["DEFAULT_MAX_ENTRIES", "build_artifact_index"]

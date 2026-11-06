"""确定性上下文压缩：按字符预算保留头尾，明确标记省略范围。

不调用模型、不做语义摘要，保证同一输入得到同一输出；用于把超长的
执行结果与重规划载荷控制在预算内，同时保留开头（结论/契约）和结尾
（验证/错误）这两段通常最关键的信息。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Iterable

# 压缩不得改写的关键字段：编号、契约枚举、路径与验收规则一旦被截断，
# 下游解析/路由/文件校验都会失真。仅保护标量键与短列表（如
# write_scope / artifacts），不保护 instructions / results 等长文本。
DEFAULT_PROTECTED_KEYS: frozenset[str] = frozenset(
    {
        "task_id",
        "subtask_id",
        "agent_id",
        "grant_id",
        "call_id",
        "plan_version",
        "phase",
        "action",
        "action_performed",
        "success",
        "passed",
        "status",
        "error",
        "file_path",
        "write_scope",
        "artifacts",
        "acceptance_criteria",
        "required_fixes",
        "verification",
        "tool_name",
        "allowed_tools",
        "depends_on",
        "role",
    }
)


@dataclass
class CompressionReport:
    """一次压缩的统计结果。"""

    original_chars: int = 0
    compressed_chars: int = 0
    paths: list[str] = field(default_factory=list)

    @property
    def changed(self) -> bool:
        return self.compressed_chars < self.original_chars


def compress_text(text: str, max_chars: int) -> str:
    """头尾保留压缩；``max_chars <= 0`` 或未超限时原样返回。"""
    if not isinstance(text, str) or max_chars <= 0 or len(text) <= max_chars:
        return text
    omitted = len(text) - max_chars
    marker = f"\n…[上下文压缩：省略 {omitted} 字符]…\n"
    if max_chars <= len(marker) + 2:
        return text[:max_chars]
    budget = max_chars - len(marker)
    head = int(budget * 0.6)
    tail = budget - head
    return text[:head] + marker + (text[-tail:] if tail else "")


def _walk_compress(
    value: Any,
    *,
    per_string_chars: int,
    path: str,
    paths: list[str],
    protected: frozenset[str],
) -> Any:
    if isinstance(value, str):
        compressed = compress_text(value, per_string_chars)
        if compressed != value:
            paths.append(path or "$")
        return compressed
    if isinstance(value, list):
        return [
            _walk_compress(
                item,
                per_string_chars=per_string_chars,
                path=f"{path}[{index}]",
                paths=paths,
                protected=protected,
            )
            for index, item in enumerate(value)
        ]
    if isinstance(value, dict):
        return {
            key: (
                item
                if key in protected
                else _walk_compress(
                    item,
                    per_string_chars=per_string_chars,
                    path=f"{path}.{key}" if path else str(key),
                    paths=paths,
                    protected=protected,
                )
            )
            for key, item in value.items()
        }
    return value


def _longest_string_path(
    value: Any,
    path: str = "$",
    *,
    protected: frozenset[str] = frozenset(),
) -> tuple[str, int]:
    best_path = ""
    best_len = 0
    if isinstance(value, str):
        return path, len(value)
    if isinstance(value, list):
        for index, item in enumerate(value):
            candidate_path, length = _longest_string_path(
                item, f"{path}[{index}]", protected=protected
            )
            if length > best_len:
                best_path, best_len = candidate_path, length
        return best_path, best_len
    if isinstance(value, dict):
        for key, item in value.items():
            if key in protected:
                continue
            candidate_path, length = _longest_string_path(
                item, f"{path}.{key}", protected=protected
            )
            if length > best_len:
                best_path, best_len = candidate_path, length
        return best_path, best_len
    return best_path, best_len


def _set_path(value: Any, path: str, replacement: Any) -> Any:
    if path == "$":
        return replacement
    tokens: list[Any] = []
    for part in path[2:].split("."):
        if not part:
            continue
        name, _, indexes = part.partition("[")
        tokens.append(name)
        if indexes:
            tokens.extend(
                int(item)
                for item in indexes.rstrip("]").split("][")
                if item
            )
    current = value
    for token in tokens[:-1]:
        current = current[token]
    current[tokens[-1]] = replacement
    return value


def compress_payload(
    payload: Any,
    *,
    max_total_chars: int = 0,
    per_string_chars: int = 0,
    protected_keys: Iterable[str] | None = None,
) -> tuple[Any, CompressionReport]:
    """压缩 JSON 兼容载荷。

    先按单字符串上限头尾压缩，再在总量仍超限时反复压缩最长的字符串，
    直到满足总量或无法继续缩短。输入不被修改。``protected_keys`` 中的
    字段名（含其子树）不参与任何压缩。
    """
    protected = frozenset(protected_keys or ())
    try:
        original = len(json.dumps(payload, ensure_ascii=False))
    except (TypeError, ValueError):
        original = 0
    paths: list[str] = []
    compressed = _walk_compress(
        payload,
        per_string_chars=per_string_chars,
        path="$",
        paths=paths,
        protected=protected,
    )
    after = len(json.dumps(compressed, ensure_ascii=False))

    attempts = 0
    while max_total_chars > 0 and after > max_total_chars and attempts < 50:
        attempts += 1
        path, length = _longest_string_path(compressed, protected=protected)
        if not path or length <= 200:
            break
        target = max(200, length // 2)
        longest = _get_path(compressed, path)
        shorter = compress_text(longest, target)
        if shorter == longest:
            break
        compressed = _set_path(compressed, path, shorter)
        paths.append(path)
        after = len(json.dumps(compressed, ensure_ascii=False))

    unique_paths = list(dict.fromkeys(paths))
    return compressed, CompressionReport(
        original_chars=original,
        compressed_chars=after,
        paths=unique_paths,
    )


def _get_path(value: Any, path: str) -> Any:
    if path == "$":
        return value
    tokens: list[Any] = []
    for part in path[2:].split("."):
        if not part:
            continue
        name, _, indexes = part.partition("[")
        tokens.append(name)
        if indexes:
            tokens.extend(
                int(item)
                for item in indexes.rstrip("]").split("][")
                if item
            )
    current = value
    for token in tokens:
        current = current[token]
    return current


__all__ = [
    "CompressionReport",
    "DEFAULT_PROTECTED_KEYS",
    "compress_payload",
    "compress_text",
]

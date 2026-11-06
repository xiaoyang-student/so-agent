"""确定性模型输入压缩：轮间只追加，超限时在边界折叠一次。

code_agent 的 token 主要消耗在“每轮重发历史”而不是输出：旧轮次的
代码参数、read_file 全文与执行 stdout 会随对话累积并被反复计费。
本模块在模型调用前做纯结构化替换，不调用模型、不读取文件。

两段式策略（``max_chars > 0`` 时）：

- 未超过 ``max_chars`` 时原样放行，历史逐轮只追加，prompt 前缀稳定，
  供应商的自动前缀缓存可命中；
- 超过上限时在边界处压缩一次：把边界之前的工具输出替换为回执、
  旧代码参数替换为长度标记，并插入一条确定性摘要；边界之前的产物
  冻结为固定列表，之后每轮复用。

若每轮都把“最近几条输出”窗口向后滑，前缀每轮都会变化，缓存命中
率恒为 0；因此边界必须由确定性的超限事件触发，而不是轮次滑动。
折叠后还要留出 ``_COMPACT_HYSTERESIS_RATIO`` 的回差：回执化的读取
结果仍占篇幅，若不设回差会在每轮重复折叠、前缀反复变化。

``max_chars <= 0`` 时保留旧的逐轮回执行为（关闭上限即关闭边界压缩）。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

_WRITE_TOOLS = frozenset({"write_file", "write_and_run", "write_generated_tool"})
_EXEC_TOOLS = frozenset({"run_in_sandbox", "write_and_run"})
_READ_TOOLS = frozenset({"read_file"})
_CODE_ARG_KEYS = ("content", "code", "manifest_data")
_MIN_PRUNE_ARG_CHARS = 300
_RECEIPT_TAIL_CHARS = 400
_COMPACT_TARGET_RATIO = 0.65
_COMPACT_HYSTERESIS_RATIO = 0.25
_COMPACT_SUMMARY_MAX_CHARS = 900
_MAX_TRACKED_RUNS = 16


@dataclass
class PruneStats:
    """一次裁剪的确定性统计（只含计数与字符数，不含载荷内容）。"""

    before_chars: int = 0
    after_chars: int = 0
    pruned_calls: int = 0
    pruned_outputs: int = 0
    dropped_items: int = 0
    compacted: bool = False
    compactions: int = 0

    @property
    def saved_chars(self) -> int:
        return max(0, self.before_chars - self.after_chars)


@dataclass
class CompactState:
    """一次 Agent 运行内的压缩边界（跨模型调用复用）。"""

    frozen_source_count: int = 0
    frozen_marker: str = ""
    frozen_items: list[Any] = field(default_factory=list)
    compactions: int = 0
    last_after_chars: int = 0


def _dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


def _text_size(value: Any) -> int:
    return len(value) if isinstance(value, str) else len(_dump(value))


def _parse(text: Any) -> Any:
    if isinstance(text, (dict, list)):
        return text
    if not isinstance(text, str):
        return None
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None


def _short(text: Any, limit: int = 300) -> str:
    value = text if isinstance(text, str) else _dump(text)
    if len(value) <= limit:
        return value
    return value[:limit].rstrip() + "…"


def _tail(text: Any, limit: int = _RECEIPT_TAIL_CHARS) -> str:
    value = text if isinstance(text, str) else _dump(text)
    if len(value) <= limit:
        return value
    return "…" + value[-limit:]


def _receipt_output(name: str, raw: Any) -> str:
    """把一条旧工具输出压缩为回执 JSON 文本。"""
    payload = _parse(raw)
    if not isinstance(payload, dict):
        return _dump({"status": "ok", "note": _short(raw, 200)})
    if payload.get("status") == "error":
        return _dump(
            {
                "status": "error",
                "error": _short(payload.get("error"), 300),
            }
        )
    metadata = payload.get("metadata")
    exec_result = payload.get("execution_result")
    if isinstance(exec_result, dict):
        metadata = exec_result.get("metadata") or metadata
    nested = exec_result if isinstance(exec_result, dict) else payload
    if name in _EXEC_TOOLS or "success" in nested:
        receipt: dict[str, Any] = {
            "status": "ok",
            "success": nested.get("success"),
            "exit_code": (metadata or {}).get("exit_code")
            if isinstance(metadata, dict)
            else None,
            "run_dir": (metadata or {}).get("run_dir")
            if isinstance(metadata, dict)
            else None,
            "output_tail": _tail(nested.get("output")),
            "note": "完整 stdout/stderr 已落盘，需要时按 run_dir 读取",
        }
        return _dump({k: v for k, v in receipt.items() if v is not None})
    if name in _WRITE_TOOLS:
        receipt = {
            "status": "ok",
            "path": payload.get("path") or payload.get("code_path"),
            "created": payload.get("created"),
            "bytes": payload.get("bytes"),
            "total_bytes": payload.get("total_bytes"),
            "note": "写入回执；文件内容与 diff 已省略，可用 read_file 重读",
        }
        return _dump({k: v for k, v in receipt.items() if v is not None})
    return _dump({"status": "ok", "note": _short(raw, 200)})


def _shrink_arguments(name: str, arguments: Any) -> tuple[Any, bool]:
    """替换旧工具调用中的大代码参数，保留路径等控制字段。"""
    data = _parse(arguments)
    if not isinstance(data, dict):
        return arguments, False
    changed = False
    label = (
        data.get("path")
        or data.get("tool_name")
        or data.get("task_id")
        or "沙箱"
    )
    for key in _CODE_ARG_KEYS:
        value = data.get(key)
        if isinstance(value, str) and len(value) > _MIN_PRUNE_ARG_CHARS:
            data[key] = f"[已省略 {len(value)} 字符：内容已写入 {label}]"
            changed = True
    return (_dump(data), True) if changed else (arguments, False)


def prune_model_input(
    items: list[Any],
    *,
    max_chars: int = 0,
    keep_last_outputs: int = 2,
) -> tuple[list[Any], PruneStats]:
    """返回裁剪后的输入列表与统计；不修改入参。"""
    original = [_copy_item(item) for item in items]
    pruned = [_copy_item(item) for item in items]
    stats = PruneStats(before_chars=sum(_text_size(i) for i in original))

    call_names: dict[str, str] = {}
    for item in pruned:
        if isinstance(item, dict) and item.get("type") == "function_call":
            call_id = item.get("call_id")
            if isinstance(call_id, str):
                call_names[call_id] = str(item.get("name") or "")

    output_indices = [
        index
        for index, item in enumerate(pruned)
        if isinstance(item, dict) and item.get("type") == "function_call_output"
    ]
    protected = set(output_indices[-max(1, keep_last_outputs):])
    # 保护最近一次执行输出（而非最早一次），保证模型能看到真实 stdout 尾部
    for index in reversed(output_indices):
        call_id = pruned[index].get("call_id")
        if call_names.get(call_id) in _EXEC_TOOLS:
            protected.add(index)
            break
    last_call_index = max(
        (
            index
            for index, item in enumerate(pruned)
            if isinstance(item, dict) and item.get("type") == "function_call"
        ),
        default=-1,
    )

    for index, item in enumerate(pruned):
        if not isinstance(item, dict):
            continue
        kind = item.get("type")
        if kind == "function_call" and index != last_call_index:
            name = str(item.get("name") or "")
            if name in _WRITE_TOOLS or name == "run_in_sandbox":
                arguments, changed = _shrink_arguments(name, item.get("arguments"))
                if changed:
                    item["arguments"] = arguments
                    stats.pruned_calls += 1
        elif kind == "function_call_output" and index not in protected:
            call_id = item.get("call_id")
            name = call_names.get(call_id, "")
            # 回执化读取结果会让模型丢失工作记忆并反复回读同一文件，
            # 反而放大轮次与 token；读取窗口本身已限制单次大小。
            if name in _READ_TOOLS:
                continue
            item["output"] = _receipt_output(name, item.get("output"))
            stats.pruned_outputs += 1

    limit = int(max_chars or 0)
    if limit > 0 and sum(_text_size(i) for i in pruned) > limit:
        pairs: dict[str, dict[str, int]] = {}
        for index, item in enumerate(pruned):
            if not isinstance(item, dict):
                continue
            kind = item.get("type")
            call_id = item.get("call_id")
            if not isinstance(call_id, str):
                continue
            if kind == "function_call":
                pairs.setdefault(call_id, {})["call"] = index
            elif kind == "function_call_output":
                pairs.setdefault(call_id, {})["output"] = index
        protected_indices = set(protected) | {0, last_call_index}
        protected_calls = {
            pruned[index].get("call_id")
            for index in protected_indices
            if 0 <= index < len(pruned)
            and isinstance(pruned[index], dict)
            and pruned[index].get("call_id")
        }
        dropped: set[int] = set()
        for index, item in enumerate(pruned):
            remaining = sum(
                _text_size(value)
                for position, value in enumerate(pruned)
                if position not in dropped
            )
            if remaining <= limit:
                break
            if index in protected_indices or index < 1 or index >= len(pruned) - 2:
                continue
            if not isinstance(item, dict):
                continue
            kind = item.get("type")
            call_id = item.get("call_id")
            if kind in {"function_call", "function_call_output"} and call_id:
                if call_id in protected_calls:
                    continue
                pair = pairs.get(call_id, {})
                targets = {
                    position
                    for position in (pair.get("call"), pair.get("output"))
                    if isinstance(position, int)
                }
                if targets & protected_indices:
                    continue
                dropped |= targets
            # reasoning 项超限也不丢：DeepSeek 思考模式下 SDK 需要把
            # 同一回合的 reasoning_content 随助手消息回传，缺失会让
            # 后续工具调用请求直接 400（invalid_request_error）
            elif kind == "message":
                dropped.add(index)
        if dropped:
            stats.dropped_items = len(dropped)
            pruned = [
                item for index, item in enumerate(pruned) if index not in dropped
            ]

    stats.after_chars = sum(_text_size(i) for i in pruned)
    return pruned, stats


def _copy_item(item: Any) -> Any:
    if isinstance(item, dict):
        return dict(item)
    return item


def _stable_marker(item: Any) -> str:
    payload = json.dumps(item, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _call_names(items: list[Any]) -> dict[str, str]:
    names: dict[str, str] = {}
    for item in items:
        if isinstance(item, dict) and item.get("type") == "function_call":
            call_id = item.get("call_id")
            if isinstance(call_id, str):
                names[call_id] = str(item.get("name") or "")
    return names


def _boundary_cut(
    items: list[Any],
    call_names: dict[str, str],
    start: int,
    keep_last_outputs: int,
) -> int:
    """返回新的压缩边界：边界之前折叠，之后保持原文。"""
    output_indices = [
        index
        for index, item in enumerate(items)
        if isinstance(item, dict) and item.get("type") == "function_call_output"
    ]
    if not output_indices:
        return len(items)
    recent = max(1, keep_last_outputs)
    protected = set(output_indices[-recent:])
    # 最新执行输出只在靠近边界时保护：更早的 stdout 已落盘且有回执
    # 尾部，若永久保护会把边界钉死在原地，使 tail 无限增长。
    guard = set(output_indices[-(recent + 1):])
    for index in reversed(output_indices):
        call_id = items[index].get("call_id")
        if call_names.get(call_id) in _EXEC_TOOLS:
            if index in guard:
                protected.add(index)
            break
    return max(start, min(protected))


def _compact_prefix(
    items: list[Any],
    call_names: dict[str, str],
) -> tuple[list[Any], int, int]:
    """把边界之前的调用参数与工具输出确定性替换为回执。"""
    out: list[Any] = []
    pruned_calls = pruned_outputs = 0
    for index, item in enumerate(items):
        copied = _copy_item(item)
        if not isinstance(copied, dict):
            out.append(copied)
            continue
        kind = copied.get("type")
        if kind == "function_call":
            name = str(copied.get("name") or "")
            if name in _WRITE_TOOLS or name == "run_in_sandbox":
                arguments, changed = _shrink_arguments(
                    name, copied.get("arguments")
                )
                if changed:
                    copied["arguments"] = arguments
                    pruned_calls += 1
        elif kind == "function_call_output":
            name = call_names.get(copied.get("call_id"), "")
            # 读取结果保持原文：回执化会让模型反复回读同一文件
            if name not in _READ_TOOLS:
                copied["output"] = _receipt_output(
                    name, copied.get("output")
                )
                pruned_outputs += 1
        out.append(copied)
    return out, pruned_calls, pruned_outputs


def _unpaired_call_ids(items: list[Any]) -> set[str]:
    """返回缺失配对的 call_id 集合（有调用无输出，或有输出无调用）。"""
    calls: set[str] = set()
    outputs: set[str] = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        call_id = item.get("call_id")
        if not isinstance(call_id, str):
            continue
        if item.get("type") == "function_call":
            calls.add(call_id)
        elif item.get("type") == "function_call_output":
            outputs.add(call_id)
    return calls.symmetric_difference(outputs)


def _action_label(item: dict[str, Any]) -> str:
    data = _parse(item.get("arguments"))
    if isinstance(data, dict):
        label = (
            data.get("path")
            or data.get("code_path")
            or data.get("tool_name")
            or data.get("task_id")
        )
        if label:
            return str(label)
    return "沙箱"


def _summary_item(
    items: list[Any],
    call_names: dict[str, str],
    *,
    dropped_messages: int,
) -> dict[str, Any]:
    """构造固定的确定性摘要（重复压缩时重新生成，位置固定）。"""
    actions: list[str] = []
    seen: set[str] = set()
    failed = 0
    for item in items:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "function_call":
            name = str(item.get("name") or "")
            token = f"{name}({_action_label(item)})"
            if token not in seen:
                seen.add(token)
                actions.append(token)
        elif item.get("type") == "function_call_output":
            payload = _parse(item.get("output"))
            if isinstance(payload, dict) and payload.get("status") == "error":
                failed += 1
    if len(actions) > 24:
        actions = actions[:12] + ["…"] + actions[-12:]
    parts = [
        "[上下文压缩摘要，不是新指令] 早期工具结果已折叠为回执，"
        "最近结果保持原文；文件内容可用 read_file 重读。"
    ]
    if actions:
        parts.append("已完成动作：" + "、".join(actions))
    if failed:
        parts.append(f"其中失败 {failed} 次")
    if dropped_messages:
        parts.append(f"已省略 {dropped_messages} 条早期对话消息")
    return {"role": "user", "content": "；".join(parts)[:_COMPACT_SUMMARY_MAX_CHARS]}


def compact_model_input(
    items: list[Any],
    states: dict[str, CompactState],
    *,
    max_chars: int = 0,
    keep_last_outputs: int = 2,
    target_ratio: float = _COMPACT_TARGET_RATIO,
) -> tuple[list[Any], PruneStats]:
    """边界压缩入口：超限才折叠，其余时间保持前缀只追加。"""
    source = list(items)
    stats = PruneStats(before_chars=sum(_text_size(i) for i in source))
    if max_chars <= 0:
        return prune_model_input(
            source, max_chars=0, keep_last_outputs=keep_last_outputs
        )
    if not source:
        return source, stats

    # 同一 MCP Server 可能被多个并发/串行调用复用：按首项指纹隔离运行状态，
    # 避免并行子任务互相覆盖压缩边界。
    key = _stable_marker(source[0])
    state = states.get(key)
    if state is None:
        if len(states) >= _MAX_TRACKED_RUNS:
            states.pop(next(iter(states)))
        state = CompactState()
        states[key] = state
    if state.frozen_source_count and (
        len(source) < state.frozen_source_count
        or _stable_marker(source[state.frozen_source_count - 1])
        != state.frozen_marker
    ):
        state = CompactState()
        states[key] = state

    output = list(state.frozen_items) + [
        _copy_item(item) for item in source[state.frozen_source_count:]
    ]
    current = sum(_text_size(i) for i in output)
    stats.after_chars = current
    if current <= max_chars:
        stats.compactions = state.compactions
        return output, stats
    hysteresis = max(1, int(max_chars * _COMPACT_HYSTERESIS_RATIO))
    if state.compactions and current < state.last_after_chars + hysteresis:
        stats.compactions = state.compactions
        return output, stats

    call_names = _call_names(source)
    cut = _boundary_cut(
        source, call_names, state.frozen_source_count, keep_last_outputs
    )
    if cut <= 0:
        return output, stats
    prefix, pruned_calls, pruned_outputs = _compact_prefix(
        source[:cut], call_names
    )
    tail = [_copy_item(item) for item in source[cut:]]
    tail_chars = sum(_text_size(i) for i in tail)
    target = max(1, int(max_chars * target_ratio))

    dropped = 0
    index = 1  # 首项是任务载荷，任何情况下都保留
    while (
        sum(_text_size(i) for i in prefix) + tail_chars > target
        and index < len(prefix)
    ):
        item = prefix[index]
        # reasoning 项不能丢：DeepSeek 思考模式要求同回合回传
        if isinstance(item, dict) and item.get("type") == "message":
            del prefix[index]
            dropped += 1
            continue
        index += 1

    summary = _summary_item(
        source[:cut], call_names, dropped_messages=dropped
    )
    # 回执本身仍有篇幅：消息丢完后若仍超目标，从最旧开始整对丢弃。
    # 整对删除保证不会留下无输出的调用或无调用的输出。输出可能因并行
    # 执行乱序落在边界之后（tail），这类调用必须保留，否则 tail 里的
    # 输出会变成孤立的 tool 消息，Chat Completions 直接 400。
    if sum(_text_size(i) for i in prefix) + tail_chars > target:
        tail_call_ids = {
            item.get("call_id")
            for item in tail
            if isinstance(item, dict)
            and item.get("type") == "function_call_output"
            and isinstance(item.get("call_id"), str)
        }
        order: list[str] = []
        for item in prefix[1:]:
            if isinstance(item, dict) and item.get("type") == "function_call":
                call_id = item.get("call_id")
                if (
                    isinstance(call_id, str)
                    and call_id not in order
                    and call_id not in tail_call_ids
                ):
                    order.append(call_id)
        drop_ids: set[str] = set()
        remaining = sum(_text_size(i) for i in prefix) + tail_chars
        sizes_by_call: dict[str, int] = {}
        for item in prefix:
            if isinstance(item, dict):
                call_id = item.get("call_id")
                if isinstance(call_id, str):
                    sizes_by_call[call_id] = (
                        sizes_by_call.get(call_id, 0) + _text_size(item)
                    )
        for call_id in order:
            if remaining <= target:
                break
            drop_ids.add(call_id)
            remaining -= sizes_by_call.get(call_id, 0)
        if drop_ids:
            prefix = [
                item
                for item in prefix
                if not (
                    isinstance(item, dict)
                    and item.get("call_id") in drop_ids
                )
            ]
            dropped = cut - len(prefix)

    # 结构完整性兜底：调用与输出必须成对，否则 Chat Completions 会因
    # 孤立 tool 消息直接 400（即使折衷未达目标也只能保留配对）。
    unpaired = _unpaired_call_ids(prefix + tail)
    if unpaired:
        prefix = [
            item
            for item in prefix
            if not (
                isinstance(item, dict) and item.get("call_id") in unpaired
            )
        ]
        tail = [
            item
            for item in tail
            if not (
                isinstance(item, dict) and item.get("call_id") in unpaired
            )
        ]
        dropped = cut - len(prefix)
    output = prefix[:1] + [summary] + prefix[1:] + tail
    state.frozen_items = list(prefix[:1] + [summary] + prefix[1:])
    state.frozen_source_count = cut
    state.frozen_marker = _stable_marker(source[cut - 1])
    state.compactions += 1

    stats.pruned_calls = pruned_calls
    stats.pruned_outputs = pruned_outputs
    stats.dropped_items = dropped
    stats.compacted = True
    stats.compactions = state.compactions
    stats.after_chars = sum(_text_size(i) for i in output)
    state.last_after_chars = stats.after_chars
    return output, stats


__all__ = [
    "CompactState",
    "PruneStats",
    "compact_model_input",
    "prune_model_input",
]

"""运行指标聚合与 Prometheus 文本导出。

指标只来自确定性的运行时事实（任务状态、事件计数、token 累计、工具
清单与 MCP 目录），不调用模型、不读取业务产出内容。
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Iterable

_EVENT_COUNTERS = {
    "tool_calls": "tool_called",
    "tool_denials": "tool_denied",
    "agent_created": "agent_created",
    "replans": "replan",
    "human_escalations": "human_escalation",
    "guidance_applied": "human_guidance_applied",
    "checkpoints_saved": "checkpoint_saved",
    "context_compressed": "context_compressed",
    "notifications_sent": "human_escalation_notified",
    "notifications_failed": "human_escalation_notify_failed",
}


def aggregate_events(event_logs: Iterable[list[dict[str, Any]]]) -> dict[str, Any]:
    """汇总多个任务的事件日志（只统计数据，不保留载荷）。"""
    counts = Counter()
    total = 0
    for log in event_logs:
        for record in log:
            event_type = str(record.get("event_type") or "")
            if not event_type:
                continue
            counts[event_type] += 1
            total += 1
    summary = {key: counts.get(event_type, 0) for key, event_type in _EVENT_COUNTERS.items()}
    summary["total"] = total
    summary["by_type"] = dict(sorted(counts.items()))
    return summary


def merge_event_summaries(*summaries: dict[str, Any]) -> dict[str, Any]:
    """合并多个 ``aggregate_events`` 结果（用于重启后补齐已持久化任务）。"""
    counts: Counter = Counter()
    total = 0
    for summary in summaries:
        for event_type, count in (summary.get("by_type") or {}).items():
            counts[str(event_type)] += int(count or 0)
        total += int(summary.get("total") or 0)
    merged = {
        key: counts.get(event_type, 0)
        for key, event_type in _EVENT_COUNTERS.items()
    }
    merged["total"] = total
    merged["by_type"] = dict(sorted(counts.items()))
    return merged


def _escape_label(value: Any) -> str:
    return str(value).replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')


def _line(name: str, value: Any, labels: dict[str, Any] | None = None) -> str:
    if labels:
        rendered = ",".join(
            f'{key}="{_escape_label(item)}"' for key, item in sorted(labels.items())
        )
        return f"{name}{{{rendered}}} {int(value)}"
    return f"{name} {int(value)}"


def render_prometheus(metrics: dict[str, Any]) -> str:
    """把指标快照渲染为 Prometheus 文本格式。"""
    tasks = metrics.get("tasks", {})
    tokens = metrics.get("tokens", {})
    events = metrics.get("events", {})
    tools = metrics.get("tools", {})
    mcp = metrics.get("mcp", {})
    lines: list[str] = [
        "# HELP so_agent_tasks_total 控制台任务总数",
        "# TYPE so_agent_tasks_total gauge",
        _line("so_agent_tasks_total", tasks.get("total", 0)),
        _line("so_agent_tasks_active", tasks.get("active", 0)),
        _line("so_agent_tasks_human_required", tasks.get("human_required", 0)),
    ]
    for status, count in sorted((tasks.get("by_status") or {}).items()):
        lines.append(_line("so_agent_tasks_status", count, {"status": status}))
    lines.extend(
        [
            "# HELP so_agent_tokens_total 全部任务 token 累计",
            "# TYPE so_agent_tokens_total counter",
            _line("so_agent_tokens_total", tokens.get("total_tokens", 0)),
            "# HELP so_agent_events_total 全部任务事件累计",
            "# TYPE so_agent_events_total counter",
            _line("so_agent_events_total", events.get("total", 0)),
        ]
    )
    for event_type, count in sorted((events.get("by_type") or {}).items()):
        lines.append(
            _line("so_agent_events_type", count, {"event_type": event_type})
        )
    lines.extend(
        [
            "# HELP so_agent_generated_tools_total 生成工具总数",
            "# TYPE so_agent_generated_tools_total gauge",
            _line("so_agent_generated_tools_total", tools.get("generated_total", 0)),
            _line("so_agent_generated_tools_quarantined", tools.get("quarantined", 0)),
            _line("so_agent_generated_tools_verified", tools.get("verified", 0)),
            _line("so_agent_generated_tools_use_total", tools.get("use_count", 0)),
            _line(
                "so_agent_generated_tools_failure_total",
                tools.get("failure_count", 0),
            ),
        ]
    )
    for status, count in sorted((tools.get("by_review_status") or {}).items()):
        lines.append(
            _line("so_agent_generated_tools_review", count, {"status": status})
        )
    for status, count in sorted((tools.get("by_safety_status") or {}).items()):
        lines.append(
            _line("so_agent_generated_tools_safety", count, {"status": status})
        )
    lines.extend(
        [
            "# HELP so_agent_mcp_servers 外部 MCP Server 数量",
            "# TYPE so_agent_mcp_servers gauge",
            _line("so_agent_mcp_servers", mcp.get("servers", 0)),
            _line("so_agent_mcp_servers_enabled", mcp.get("enabled", 0)),
            _line("so_agent_mcp_servers_errors", mcp.get("errors", 0)),
        ]
    )
    return "\n".join(lines) + "\n"


__all__ = ["aggregate_events", "merge_event_summaries", "render_prometheus"]

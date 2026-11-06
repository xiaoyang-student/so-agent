"""复杂任务基准探针：真实 LLM、全量 token、分步耗时与稳定性证据。

与 ``real_llm_smoke.py`` 的区别：本脚本面向“测量”而非“冒烟”。它通过
OpenAI Agents SDK 的 TracingProcessor 捕获进程内**每一次**模型生成
（含主管、评审、code_agent、动态子 Agent 的嵌套调用）的 usage 与起止
时间，再与本项目事件日志里的工具调用、Agent 生命周期、重试记录对齐，
输出可复核的 JSON 证据。

用法：
    python scripts/bench_probe.py --task sales --runs 1
    python scripts/bench_probe.py --task sales --runs 2 --timeout 900

证据写入 ``sandbox/bench-<时间戳>/evidence.json``；同目录另有
``spans.json``（原始 span 明细）、``events.log``（本项目事件日志）与
任务产物。脚本不打印、不落盘任何密钥。
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import math
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from agents.tracing import (  # noqa: E402
    AgentSpanData,
    FunctionSpanData,
    GenerationSpanData,
    ResponseSpanData,
    TracingProcessor,
    set_trace_processors,
    set_tracing_disabled,
)

from so_agent.main import _load_secrets, _resolve_settings  # noqa: E402

SALES_TASK = {
    "objective": (
        "在沙箱目录内完成一份销售数据分析交付物：\n"
        "1. 生成 sales.csv：至少 36 行，列为 date,region,product,quantity,unit_price；"
        "日期覆盖 2026-01 至 2026-06，3 个地区（华东/华南/华北），3 个产品（A/B/C），"
        "quantity 为正整数，unit_price 为正数；\n"
        "2. 编写 analyze_sales.py：读取 sales.csv，按地区和产品分别汇总销售额"
        "（quantity*unit_price），计算总销售额与销售额最高的产品，输出 summary.json"
        "（字段：total_sales、by_region、by_product、top_product），并在标准输出打印汇总表；\n"
        "3. 运行 analyze_sales.py 生成 summary.json；\n"
        "4. 根据 summary.json 编写 report.md：包含总销售额、地区排名、产品排名、"
        "至少一条基于数据的建议、数据口径说明；\n"
        "5. 使用 read_file 回读 report.md 与 summary.json，核对关键数字一致。"
    ),
    "constraints": "只允许在沙箱目录内读写；不访问网络；report.md 与 summary.json 的数字必须与 CSV 计算结果一致",
    "acceptance_criteria": (
        "sales.csv 不少于 36 行且列齐全；summary.json 包含 total_sales/by_region/by_product/top_product，"
        "且总额可由 CSV 复算一致；report.md 非空且包含总销售额数字"
    ),
}

# 校验用短任务（仅验证探针本身能抓到嵌套调用的 usage）
PROBE_TASK = {
    "objective": "在沙箱目录创建 probe-ok.md，写入一行 PROBE_OK，然后使用 read_file 回读确认。",
    "constraints": "只允许在沙箱目录内读写",
    "acceptance_criteria": "probe-ok.md 存在且内容包含 PROBE_OK",
}

INCIDENT_LOG = """2026-03-01T09:30:05Z GET  /api/orders     200 137ms
2026-03-01T09:30:35Z GET  /api/cart       200 194ms
2026-03-01T09:31:05Z POST /api/checkout   200 257ms
2026-03-01T09:31:35Z GET  /api/search     200 115ms
2026-03-01T09:32:05Z GET  /api/orders     200 168ms
2026-03-01T09:32:35Z GET  /api/cart       200 269ms
2026-03-01T09:33:05Z GET  /api/user       200 270ms
2026-03-01T09:33:35Z POST /api/checkout   200 246ms
2026-03-01T09:34:05Z GET  /api/orders     200 157ms
2026-03-01T09:34:35Z GET  /api/cart       200 183ms
2026-03-01T09:35:05Z GET  /api/user       200 114ms
2026-03-01T09:35:35Z POST /api/checkout   200 193ms
2026-03-01T09:36:05Z GET  /api/orders     400 110ms
2026-03-01T09:36:35Z GET  /api/cart       200 147ms
2026-03-01T09:37:05Z GET  /api/user       200 272ms
2026-03-01T09:37:35Z POST /api/checkout   200 224ms
2026-03-01T09:38:05Z POST /api/orders     200 245ms
2026-03-01T09:38:35Z GET  /api/cart       200 133ms
2026-03-01T09:39:05Z GET  /api/user       200 164ms
2026-03-01T09:39:35Z GET  /api/search     200 309ms
2026-03-01T09:40:05Z POST /api/checkout   500 3670ms
2026-03-01T09:40:35Z POST /api/checkout   500 3120ms
2026-03-01T09:41:05Z POST /api/checkout   500 2680ms
2026-03-01T09:41:35Z POST /api/checkout   500 2410ms
2026-03-01T09:42:05Z POST /api/checkout   500 2260ms
2026-03-01T09:42:35Z POST /api/checkout   500 2040ms
2026-03-01T09:43:05Z POST /api/checkout   500 1810ms
2026-03-01T09:43:35Z POST /api/checkout   500 1680ms
2026-03-01T09:44:05Z POST /api/checkout   500 1520ms
2026-03-01T09:44:35Z GET  /api/cart       200 286ms
2026-03-01T09:45:05Z GET  /api/user       200 255ms
2026-03-01T09:45:35Z GET  /api/search     200 247ms
2026-03-01T09:46:05Z GET  /api/orders     201 84ms
2026-03-01T09:46:35Z POST /api/checkout   200 115ms
2026-03-01T09:47:05Z GET  /api/user       200 189ms
2026-03-01T09:47:35Z GET  /api/search     400 201ms
2026-03-01T09:48:05Z POST /api/orders     200 167ms
2026-03-01T09:48:35Z GET  /api/cart       200 70ms
2026-03-01T09:49:05Z GET  /api/user       200 175ms
2026-03-01T09:49:35Z GET  /api/search     400 77ms
2026-03-01T09:50:05Z GET  /api/orders     200 233ms
2026-03-01T09:50:35Z POST /api/checkout   200 265ms
2026-03-01T09:51:05Z GET  /api/user       200 298ms
2026-03-01T09:51:35Z GET  /api/search     200 191ms
2026-03-01T09:52:05Z POST /api/orders     200 214ms
2026-03-01T09:52:35Z GET  /api/cart       200 154ms
2026-03-01T09:53:05Z GET  /api/user       200 190ms
2026-03-01T09:53:35Z GET  /api/search     200 232ms
2026-03-01T09:54:05Z GET  /api/orders     200 265ms
2026-03-01T09:54:35Z GET  /api/cart       200 197ms
2026-03-01T09:55:05Z GET  /api/user       200 252ms
2026-03-01T09:55:35Z GET  /api/search     400 143ms
2026-03-01T09:56:05Z GET  /api/orders     400 133ms
2026-03-01T09:56:35Z GET  /api/cart       200 174ms
2026-03-01T09:57:05Z GET  /api/user       200 269ms
2026-03-01T09:57:35Z GET  /api/search     400 98ms
2026-03-01T09:58:05Z POST /api/orders     201 79ms
2026-03-01T09:58:35Z GET  /api/cart       200 208ms
2026-03-01T09:59:05Z GET  /api/user       200 170ms
2026-03-01T09:59:35Z GET  /api/search     200 163ms
"""

INCIDENT_TASK = {
    "objective": (
        "在沙箱根目录分析一份访问日志，产出可复核的故障分析交付物：\n"
        "1. 读取沙箱根目录的 access.log（60 行，格式：ISO8601时间 方法 路径 状态码 延迟ms）；\n"
        "2. 编写 analyze_incident.py：解析 access.log，计算指标并写入 metrics.json，"
        "该文件只包含以下 6 个字段：total_requests（总请求数）、error_count（状态码>=500 的请求数）、"
        "error_rate（error_count/total_requests，保留 4 位小数）、"
        "top_error_endpoint（5xx 最多的路径，并列时取字典序最小）、"
        "worst_5min_window（5xx 最多的 5 分钟窗口，格式 HH:MM，取该窗口起始时刻，UTC；并列取最早）、"
        "p95_latency_ms（全部请求延迟的 95 百分位，nearest-rank 法：排序后取第 ceil(0.95*N) 个，1-based）；\n"
        "3. 运行 analyze_incident.py 生成 metrics.json；\n"
        "4. 编写 incident_report.md：包含故障时间线、top_error_endpoint 与 worst_5min_window、"
        "p95 延迟、至少 2 条根因假设并说明依据、至少 2 条可执行建议；\n"
        "5. 使用 read_file 回读 metrics.json 与 incident_report.md，核对数字与脚本输出一致。"
    ),
    "constraints": (
        "只允许在沙箱目录内读写；不访问网络；不得修改 access.log；"
        "metrics.json 的数字必须由脚本从 access.log 计算得出，不得手工填写"
    ),
    "acceptance_criteria": (
        "access.log 未被修改；analyze_incident.py 存在；metrics.json 的 6 个字段与 access.log 复算结果完全一致；"
        "incident_report.md 非空且包含 metrics.json 中的 p95 延迟、top_error_endpoint、worst_5min_window"
    ),
}

TASKS: dict[str, dict[str, str]] = {
    "sales": SALES_TASK,
    "probe": PROBE_TASK,
    "incident": INCIDENT_TASK,
}

# 无文件系统的渠道（原生 API / 网页）需要把输入直接放进提示词
RAW_APPENDIX: dict[str, str] = {
    "incident": "\n\n输入日志 access.log 全文（共 60 行）：\n" + INCIDENT_LOG,
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_ts(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _seconds_between(start: str | None, end: str | None) -> float | None:
    a, b = _parse_ts(start), _parse_ts(end)
    if a is None or b is None:
        return None
    return round(max(0.0, (b - a).total_seconds()), 3)


@dataclass
class _SpanRecord:
    span_id: str
    parent_id: str | None
    trace_id: str
    kind: str
    name: str | None
    started_at: str | None
    ended_at: str | None
    duration: float | None
    usage: dict[str, int] = field(default_factory=dict)
    model: str | None = None
    input_preview: str = ""
    output_preview: str = ""
    agent_path: list[str] = field(default_factory=list)


class UsageCollector(TracingProcessor):
    """同步收集所有 trace/span，结束时按最近 Agent 祖先归属。"""

    def __init__(self) -> None:
        self.traces: dict[str, dict[str, Any]] = {}
        self.spans: dict[str, _SpanRecord] = {}
        self._started: dict[str, str] = {}
        # span 开始时即登记父链元数据：父 span 可能仍在进行中、尚未进入
        # self.spans，但子生成的 Agent 归属仍需可解析。
        self._meta: dict[str, tuple[str, str | None, str | None]] = {}

    def on_trace_start(self, trace: Any) -> None:
        self.traces[trace.trace_id] = {
            "trace_id": trace.trace_id,
            "name": getattr(trace, "name", None),
            "started_at": _now_iso(),
        }

    def on_trace_end(self, trace: Any) -> None:
        record = self.traces.get(trace.trace_id)
        if record is not None:
            record["ended_at"] = _now_iso()

    def on_span_start(self, span: Any) -> None:
        try:
            self._started[span.span_id] = _now_iso()
            data = span.span_data
            self._meta[span.span_id] = (
                getattr(data, "type", type(data).__name__),
                getattr(span, "parent_id", None),
                getattr(data, "name", None),
            )
        except Exception:
            pass

    def on_span_end(self, span: Any) -> None:
        try:
            self._record(span)
        except Exception:
            return

    def _record(self, span: Any) -> None:
        data = span.span_data
        kind = getattr(data, "type", type(data).__name__)
        name: str | None = None
        usage: dict[str, int] = {}
        model: str | None = None
        input_preview = ""
        output_preview = ""

        if isinstance(data, AgentSpanData):
            name = getattr(data, "name", None)
        elif isinstance(data, FunctionSpanData):
            name = getattr(data, "name", None)
            input_preview = _preview(getattr(data, "input", None))
            output_preview = _preview(getattr(data, "output", None))
        elif isinstance(data, (GenerationSpanData, ResponseSpanData)):
            raw_usage = getattr(data, "usage", None) or {}
            if isinstance(raw_usage, dict):
                details = raw_usage.get("input_tokens_details") or {}
                usage = {
                    key: int(raw_usage.get(key) or 0)
                    for key in (
                        "requests",
                        "input_tokens",
                        "output_tokens",
                        "total_tokens",
                    )
                }
                usage["cached_tokens"] = int(
                    (details.get("cached_tokens") if isinstance(details, dict) else 0)
                    or raw_usage.get("cached_tokens")
                    or 0
                )
                usage["cache_write_tokens"] = int(
                    (
                        details.get("cache_write_tokens")
                        if isinstance(details, dict)
                        else 0
                    )
                    or raw_usage.get("cache_write_tokens")
                    or 0
                )
            model = getattr(data, "model", None)
        else:
            name = getattr(data, "type", None)

        started = getattr(span, "started_at", None) or self._started.get(span.span_id)
        ended = getattr(span, "ended_at", None) or _now_iso()
        parent_id = getattr(span, "parent_id", None)
        record = _SpanRecord(
            span_id=span.span_id,
            parent_id=parent_id,
            trace_id=span.trace_id,
            kind=kind,
            name=name,
            started_at=started,
            ended_at=ended,
            duration=_seconds_between(started, ended),
            usage=usage,
            model=model,
            input_preview=input_preview,
            output_preview=output_preview,
        )
        record.agent_path = self._agent_path(parent_id)
        self._meta[span.span_id] = (kind, parent_id, name or record.name)
        self.spans[span.span_id] = record

    def _agent_path(self, parent_id: str | None) -> list[str]:
        path: list[str] = []
        seen: set[str] = set()
        current = parent_id
        while current and current not in seen:
            seen.add(current)
            kind, next_parent, name = self._meta.get(
                current, ("", None, None)
            )
            if kind == "agent" and name:
                path.append(name)
            current = next_parent
        path.reverse()
        return path

    def shutdown(self) -> None:
        return None

    def force_flush(self) -> None:
        return None

    def summary(self) -> dict[str, Any]:
        generations: list[dict[str, Any]] = []
        by_agent: dict[str, dict[str, Any]] = {}
        totals = {
            "requests": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "cached_tokens": 0,
            "cache_write_tokens": 0,
            "generation_spans": 0,
        }
        for span in self.spans.values():
            if span.kind not in {"generation", "response"}:
                continue
            if not span.usage and span.kind == "response":
                continue
            agent = span.agent_path[-1] if span.agent_path else "(root)"
            bucket = by_agent.setdefault(
                agent,
                {
                    "requests": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                    "cached_tokens": 0,
                    "cache_write_tokens": 0,
                    "calls": 0,
                    "generation_seconds": 0.0,
                },
            )
            bucket["calls"] += 1
            bucket["generation_seconds"] = round(
                bucket["generation_seconds"] + (span.duration or 0.0), 3
            )
            for key in (
                "requests",
                "input_tokens",
                "output_tokens",
                "total_tokens",
                "cached_tokens",
                "cache_write_tokens",
            ):
                bucket[key] += span.usage.get(key, 0)
                totals[key] += span.usage.get(key, 0)
            totals["generation_spans"] += 1
            generations.append(
                {
                    "span_id": span.span_id,
                    "agent": agent,
                    "agent_path": span.agent_path,
                    "model": span.model,
                    "started_at": span.started_at,
                    "ended_at": span.ended_at,
                    "duration": span.duration,
                    "usage": span.usage,
                    "parent_span": span.parent_id,
                }
            )

        tool_spans: list[dict[str, Any]] = []
        for span in self.spans.values():
            if span.kind != "function":
                continue
            tool_spans.append(
                {
                    "span_id": span.span_id,
                    "name": span.name,
                    "agent_path": span.agent_path,
                    "started_at": span.started_at,
                    "ended_at": span.ended_at,
                    "duration": span.duration,
                    "input_preview": span.input_preview,
                    "output_preview": span.output_preview,
                    "parent_span": span.parent_id,
                }
            )

        agent_spans: list[dict[str, Any]] = []
        for span in self.spans.values():
            if span.kind != "agent":
                continue
            agent_spans.append(
                {
                    "span_id": span.span_id,
                    "name": span.name,
                    "started_at": span.started_at,
                    "ended_at": span.ended_at,
                    "duration": span.duration,
                    "parent_span": span.parent_id,
                }
            )

        generations.sort(key=lambda item: item["started_at"] or "")
        tool_spans.sort(key=lambda item: item["started_at"] or "")
        return {
            "totals": totals,
            "by_agent": by_agent,
            "generations": generations,
            "tool_spans": tool_spans,
            "agent_spans": agent_spans,
        }


def _preview(value: Any, limit: int = 400) -> str:
    if value is None:
        return ""
    text = value if isinstance(value, str) else json.dumps(
        value, ensure_ascii=False, default=str
    )
    return text[:limit]


def _read_events(log_path: Path) -> list[dict[str, Any]]:
    if not log_path.is_file():
        return []
    events: list[dict[str, Any]] = []
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return events


def _pair_tool_calls(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """按 call_id 把 invoke_started / invoke_completed / invoke_failed 配对。"""
    pending: dict[str, dict[str, Any]] = {}
    result: list[dict[str, Any]] = []
    for event in events:
        if event.get("event_type") != "tool_called":
            continue
        payload = event.get("payload") or {}
        call_id = str(payload.get("call_id") or "")
        action = payload.get("action")
        if action == "invoke_started":
            item = {
                "call_id": call_id,
                "tool": payload.get("tool_name"),
                "agent": payload.get("agent"),
                "subtask_id": payload.get("subtask_id"),
                "started_at": event.get("timestamp"),
                "arguments_preview": payload.get("arguments_preview"),
            }
            pending[call_id] = item
            result.append(item)
        else:
            item = pending.pop(call_id, None)
            if item is None:
                item = {
                    "call_id": call_id,
                    "tool": payload.get("tool_name"),
                    "agent": payload.get("agent"),
                    "subtask_id": payload.get("subtask_id"),
                    "started_at": None,
                    "arguments_preview": "",
                }
                result.append(item)
            item["ended_at"] = event.get("timestamp")
            item["status"] = "completed" if action == "invoke_completed" else "failed"
            item["result_ok"] = payload.get("result_ok")
            item["duration"] = payload.get("duration")
            item["output_preview"] = payload.get("output_preview")
            item["unsettled"] = payload.get("unsettled", False)
    for item in pending.values():
        item["status"] = "unsettled"
        item["result_ok"] = None
    for item in result:
        item.setdefault("status", "unsettled")
    return result


def _classify_events(
    events: list[dict[str, Any]],
) -> dict[str, Any]:
    counts: dict[str, int] = {}
    token_events: list[dict[str, Any]] = []
    retries: list[dict[str, Any]] = []
    lifecycle: list[dict[str, Any]] = []
    for event in events:
        kind = event.get("event_type") or "unknown"
        counts[kind] = counts.get(kind, 0) + 1
        payload = event.get("payload") or {}
        if kind == "token_usage":
            token_events.append(
                {
                    "timestamp": event.get("timestamp"),
                    "agent": payload.get("agent"),
                    "requests": payload.get("requests"),
                    "input_tokens": payload.get("input_tokens"),
                    "output_tokens": payload.get("output_tokens"),
                    "total_tokens": payload.get("total_tokens"),
                    "cached_tokens": payload.get("cached_tokens"),
                    "cache_write_tokens": payload.get("cache_write_tokens"),
                }
            )
        if kind in {
            "failure_classified",
            "recovery_decided",
            "retry_suppressed",
            "replacement_requested",
            "replacement_executing",
            "replacement_completed",
            "replacement_failed",
            "replan",
            "decomposition_issue",
        }:
            retries.append(
                {
                    "event_type": kind,
                    "timestamp": event.get("timestamp"),
                    "subtask_id": payload.get("subtask_id"),
                    "payload": payload,
                }
            )
        if kind in {
            "agent_created",
            "agent_released",
            "status_transition",
            "tool_granted",
            "tool_revoked",
        }:
            lifecycle.append(
                {
                    "event_type": kind,
                    "timestamp": event.get("timestamp"),
                    "agent_id": event.get("agent_id"),
                    "payload": payload,
                }
            )
    return {
        "counts": counts,
        "token_events": token_events,
        "retries": retries,
        "lifecycle": lifecycle,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_inventory(root: Path) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.name in {"events.log", "spans.json", "evidence.json"}:
            continue
        relative = str(path.relative_to(root)).replace("\\", "/")
        items.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": _sha256(path)[:16],
            }
        )
    return items


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _top_product_name(value: Any) -> str | None:
    """归一化 summary.top_product：接受字符串或含产品名的对象。

    任务规格只要求字段存在，未规定形状；模型可能写成
    ``{"product": "C", "sales": 1.0}``。两种形态均应视为正确。
    """
    if isinstance(value, dict):
        for key in ("product", "name", "top_product", "value"):
            candidate = value.get(key)
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip()
        return None
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _validate_sales(root: Path) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    csv_path = root / "sales.csv"
    summary_path = root / "summary.json"
    report_path = root / "report.md"
    script_path = root / "analyze_sales.py"

    checks["sales_csv_exists"] = csv_path.is_file()
    checks["analyze_script_exists"] = script_path.is_file()
    checks["summary_exists"] = summary_path.is_file()
    checks["report_exists"] = report_path.is_file() and report_path.stat().st_size > 0

    rows: list[dict[str, str]] = []
    if csv_path.is_file():
        with csv_path.open("r", encoding="utf-8", errors="replace", newline="") as fh:
            reader = csv.DictReader(fh)
            fields = set(reader.fieldnames or [])
            checks["csv_columns_ok"] = {
                "date",
                "region",
                "product",
                "quantity",
                "unit_price",
            }.issubset(fields)
            rows = list(reader)
    checks["csv_rows"] = len(rows)
    checks["csv_rows_ok"] = len(rows) >= 36

    computed: dict[str, Any] = {}
    if rows:
        total = 0.0
        by_region: dict[str, float] = {}
        by_product: dict[str, float] = {}
        valid = True
        for row in rows:
            quantity = _as_float(row.get("quantity"))
            price = _as_float(row.get("unit_price"))
            region = (row.get("region") or "").strip()
            product = (row.get("product") or "").strip()
            if quantity is None or price is None or not region or not product:
                valid = False
                continue
            amount = quantity * price
            total += amount
            by_region[region] = by_region.get(region, 0.0) + amount
            by_product[product] = by_product.get(product, 0.0) + amount
        computed = {
            "total_sales": round(total, 2),
            "by_region": {key: round(value, 2) for key, value in by_region.items()},
            "by_product": {
                key: round(value, 2) for key, value in by_product.items()
            },
            "top_product": max(by_product, key=by_product.get),
        }
    checks["csv_rows_valid"] = bool(computed)

    summary: dict[str, Any] = {}
    if summary_path.is_file():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            summary = {}
    checks["summary_fields_ok"] = all(
        key in summary
        for key in ("total_sales", "by_region", "by_product", "top_product")
    )

    if computed and summary:
        summary_total = _as_float(summary.get("total_sales"))
        checks["summary_total_matches"] = (
            summary_total is not None
            and abs(summary_total - computed["total_sales"]) <= 0.01
        )
        checks["summary_top_product_matches"] = _top_product_name(
            summary.get("top_product")
        ) == computed["top_product"]
        summary_regions = summary.get("by_region") or {}
        checks["summary_by_region_matches"] = isinstance(summary_regions, dict) and all(
            _as_float(summary_regions.get(region)) is not None
            and abs(_as_float(summary_regions.get(region)) - value) <= 0.01
            for region, value in computed["by_region"].items()
        )
    else:
        checks["summary_total_matches"] = False
        checks["summary_top_product_matches"] = False
        checks["summary_by_region_matches"] = False

    report_text = ""
    if report_path.is_file():
        report_text = report_path.read_text(encoding="utf-8", errors="replace")
    checks["report_has_total"] = False
    if computed and report_text:
        total = computed["total_sales"]
        variants = {
            f"{total:.2f}",
            f"{total:.1f}",
            f"{total:.0f}",
            f"{total:,.2f}",
            f"{total:,.0f}",
            str(total),
        }
        checks["report_has_total"] = any(item in report_text for item in variants)
    checks["report_mentions_regions"] = all(
        region in report_text for region in (computed.get("by_region") or {})
    )
    checks["report_chars"] = len(report_text)

    passed = all(
        checks.get(key)
        for key in (
            "sales_csv_exists",
            "analyze_script_exists",
            "summary_exists",
            "report_exists",
            "csv_columns_ok",
            "csv_rows_ok",
            "csv_rows_valid",
            "summary_fields_ok",
            "summary_total_matches",
            "summary_top_product_matches",
            "report_has_total",
        )
    )
    return {
        "checklist": checks,
        "computed": computed,
        "summary": summary,
        "passed": bool(passed),
    }


def _validate_probe(root: Path) -> dict[str, Any]:
    path = root / "probe-ok.md"
    exists = path.is_file()
    content = path.read_text(encoding="utf-8", errors="replace") if exists else ""
    return {
        "checklist": {"exists": exists, "has_marker": "PROBE_OK" in content},
        "passed": exists and "PROBE_OK" in content,
    }


def _incident_records(text: str) -> list[tuple[str, str, int, int]]:
    records: list[tuple[str, str, int, int]] = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            records.append(
                (parts[0], parts[2], int(parts[3]), int(parts[4].rstrip("ms")))
            )
        except ValueError:
            continue
    return records


def _incident_expected(text: str) -> dict[str, Any]:
    """从日志文本独立复算指标；不依赖被测 Agent 的任何中间产物。"""
    records = _incident_records(text)
    total = len(records)
    if not total:
        return {}
    errors = [item for item in records if item[2] >= 500]
    by_endpoint: dict[str, int] = {}
    for item in errors:
        by_endpoint[item[1]] = by_endpoint.get(item[1], 0) + 1
    top_endpoint = (
        min(by_endpoint, key=lambda name: (-by_endpoint[name], name))
        if by_endpoint
        else ""
    )
    by_window: dict[str, int] = {}
    for item in errors:
        minute = int(item[0][14:16])
        key = f"{item[0][11:13]}:{minute // 5 * 5:02d}"
        by_window[key] = by_window.get(key, 0) + 1
    worst_window = (
        min(by_window, key=lambda name: (-by_window[name], name))
        if by_window
        else ""
    )
    latencies = sorted(item[3] for item in records)
    rank = math.ceil(0.95 * total)
    return {
        "total_requests": total,
        "error_count": len(errors),
        "error_rate": round(len(errors) / total, 4),
        "top_error_endpoint": top_endpoint,
        "worst_5min_window": worst_window,
        "p95_latency_ms": latencies[rank - 1],
    }


def _as_int(value: Any) -> int | None:
    number = _as_float(value)
    if number is None or abs(number - round(number)) > 1e-6:
        return None
    return int(round(number))


def _validate_incident(root: Path) -> dict[str, Any]:
    expected = _incident_expected(INCIDENT_LOG)
    checks: dict[str, Any] = {}

    log_path = root / "access.log"
    log_text = (
        log_path.read_text(encoding="utf-8", errors="replace")
        if log_path.is_file()
        else ""
    )
    checks["access_log_intact"] = (
        log_text.replace("\r\n", "\n").strip() == INCIDENT_LOG.strip()
    )
    checks["analyze_script_exists"] = (root / "analyze_incident.py").is_file()
    metrics_path = root / "metrics.json"
    report_path = root / "incident_report.md"
    checks["metrics_exists"] = metrics_path.is_file()
    checks["report_exists"] = (
        report_path.is_file() and report_path.stat().st_size > 0
    )

    metrics: dict[str, Any] = {}
    if metrics_path.is_file():
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            metrics = {}
    checks["metrics_fields_ok"] = bool(expected) and all(
        key in metrics for key in expected
    )

    checks["total_requests_ok"] = _as_int(metrics.get("total_requests")) == expected.get(
        "total_requests"
    )
    checks["error_count_ok"] = _as_int(metrics.get("error_count")) == expected.get(
        "error_count"
    )
    rate = _as_float(metrics.get("error_rate"))
    checks["error_rate_ok"] = (
        rate is not None
        and abs(rate - float(expected.get("error_rate", -1))) <= 1e-4
    )
    checks["top_error_endpoint_ok"] = (
        str(metrics.get("top_error_endpoint") or "").strip()
        == expected.get("top_error_endpoint")
    )
    checks["worst_window_ok"] = (
        str(metrics.get("worst_5min_window") or "").strip()
        == expected.get("worst_5min_window")
    )
    p95 = _as_float(metrics.get("p95_latency_ms"))
    checks["p95_ok"] = (
        p95 is not None and abs(p95 - float(expected.get("p95_latency_ms", -1))) <= 0.5
    )

    report_text = ""
    if report_path.is_file():
        report_text = report_path.read_text(encoding="utf-8", errors="replace")
    checks["report_chars"] = len(report_text)
    p95_value = expected.get("p95_latency_ms")
    p95_variants = {str(p95_value), f"{p95_value:,}", f"{p95_value}ms", f"{p95_value:,}ms"}
    checks["report_mentions_p95"] = any(item in report_text for item in p95_variants)
    checks["report_mentions_endpoint"] = (
        expected.get("top_error_endpoint", "\0") in report_text
    )
    checks["report_mentions_window"] = (
        expected.get("worst_5min_window", "\0") in report_text
    )
    checks["report_has_cause"] = "根因" in report_text
    checks["report_has_recommendation"] = "建议" in report_text

    passed = all(
        checks.get(key)
        for key in (
            "access_log_intact",
            "analyze_script_exists",
            "metrics_exists",
            "report_exists",
            "metrics_fields_ok",
            "total_requests_ok",
            "error_count_ok",
            "error_rate_ok",
            "top_error_endpoint_ok",
            "worst_window_ok",
            "p95_ok",
            "report_mentions_p95",
            "report_mentions_endpoint",
            "report_mentions_window",
        )
    )
    return {
        "checklist": checks,
        "expected": expected,
        "metrics": metrics,
        "passed": bool(passed),
    }


VALIDATORS = {
    "sales": _validate_sales,
    "probe": _validate_probe,
    "incident": _validate_incident,
}


def _prepare_task(task_name: str, root: Path) -> None:
    """写入只读输入夹具；被 Agent 改动会在校验中体现为失败项。"""
    if task_name == "incident":
        with (root / "access.log").open("w", encoding="utf-8", newline="\n") as fh:
            fh.write(INCIDENT_LOG)


async def _run_once(
    task_name: str,
    run_dir: Path,
    timeout: float,
    collector: UsageCollector,
) -> dict[str, Any]:
    from so_agent.context import ProjectContext
    from so_agent.workflow import WorkflowEngine

    settings = _resolve_settings({})
    run_dir.mkdir(parents=True, exist_ok=True)
    _prepare_task(task_name, run_dir)
    task = TASKS[task_name]
    context = ProjectContext(
        sandbox_dir=run_dir,
        project_name=f"bench-{task_name}",
        config=settings,
    )
    engine = WorkflowEngine(context, settings=settings, log_file=run_dir / "events.log")

    started = time.perf_counter()
    started_at = _now_iso()
    error = None
    output: Any = None
    try:
        output = await asyncio.wait_for(
            engine.run(
                task["objective"], task["constraints"], task["acceptance_criteria"]
            ),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        error = f"timeout>{timeout}s"
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    elapsed = round(time.perf_counter() - started, 2)
    ended_at = _now_iso()

    events = _read_events(run_dir / "events.log")
    tool_calls = _pair_tool_calls(events)
    classified = _classify_events(events)
    validation = VALIDATORS[task_name](run_dir)
    artifacts = _artifact_inventory(run_dir)

    generation_usage = collector.summary()
    trace_spans = [dict(record.__dict__) for record in collector.spans.values()]
    return {
        "task": task_name,
        "started_at": started_at,
        "ended_at": ended_at,
        "elapsed_seconds": elapsed,
        "error": error,
        "status": getattr(getattr(output, "status", None), "value", None),
        "task_id": getattr(output, "task_id", None),
        "final_result": (getattr(output, "final_result", "") or "")[:2000],
        "plan_versions": [
            [spec.model_dump(mode="json") if hasattr(spec, "model_dump") else spec for spec in plan]
            for plan in (getattr(output, "plan_versions", None) or [])
        ],
        "agent_history": [
            record.model_dump(mode="json") if hasattr(record, "model_dump") else record
            for record in (getattr(output, "agent_history", None) or [])
        ],
        "tool_calls": tool_calls,
        "events": classified,
        "validation": validation,
        "artifacts": artifacts,
        "usage": generation_usage,
        "read_stats": _read_stats(trace_spans),
        "trace_spans": trace_spans,
    }


def _read_stats(spans: list[dict[str, Any]]) -> dict[str, Any]:
    """统计 read_file 的重复读取分布，用于判断折叠策略是否值得做。

    运行身份取最近的 agent 祖先 span：每次 code_agent 调用都是新上下文，
    跨运行的重复读取无法折叠，必须与运行内重复分开统计。
    """
    by_id = {span.get("span_id"): span for span in spans}

    def run_id(span: dict[str, Any]) -> str | None:
        current = by_id.get(span.get("parent_id"))
        seen: set[str] = set()
        while current is not None and current.get("span_id") not in seen:
            seen.add(current.get("span_id"))
            if current.get("kind") == "agent":
                return current.get("span_id")
            current = by_id.get(current.get("parent_id"))
        return None

    reads: list[dict[str, Any]] = []
    for span in spans:
        if span.get("name") != "read_file":
            continue
        try:
            arguments = json.loads(span.get("input_preview") or "{}")
        except (json.JSONDecodeError, TypeError):
            arguments = {}
        if not isinstance(arguments, dict):
            arguments = {}
        reads.append(
            {
                "run": run_id(span),
                "path": arguments.get("path"),
                "start_line": int(arguments.get("start_line") or 1),
                "max_lines": int(arguments.get("max_lines") or 0),
                "force": bool(arguments.get("force")),
                "collapsed": '"unchanged": true' in (span.get("output_preview") or ""),
            }
        )

    windows = Counter(
        (
            item["run"],
            item["path"],
            item["start_line"],
            item["max_lines"],
        )
        for item in reads
        if not item["force"]
    )
    repeated_windows = [
        {
            "run": key[0],
            "path": key[1],
            "start_line": key[2],
            "max_lines": key[3],
            "count": count,
        }
        for key, count in windows.items()
        if count > 1
    ]
    per_run_paths = Counter((item["run"], item["path"]) for item in reads)
    paths_by_run: dict[str, set[str]] = {}
    for item in reads:
        if item["path"] is not None:
            paths_by_run.setdefault(str(item["path"]), set()).add(str(item["run"]))
    return {
        "calls": len(reads),
        "agent_runs": len({item["run"] for item in reads}),
        "collapsed": sum(1 for item in reads if item["collapsed"]),
        "forced": sum(1 for item in reads if item["force"]),
        "in_run_repeat_paths": sum(
            1 for count in per_run_paths.values() if count > 1
        ),
        "in_run_repeat_windows": len(repeated_windows),
        "cross_run_repeat_paths": sorted(
            path for path, runs in paths_by_run.items() if len(runs) > 1
        ),
        "repeated_windows": repeated_windows,
    }


def _summarize_run(run: dict[str, Any]) -> dict[str, Any]:
    usage = run["usage"]
    tool_calls = run["tool_calls"]
    retries = run["events"]["retries"]
    agent_spans = usage["agent_spans"]
    slowest = sorted(
        (
            {
                "agent": item["agent"],
                "duration": item["duration"],
                "total_tokens": item["usage"].get("total_tokens", 0),
                "input_tokens": item["usage"].get("input_tokens", 0),
                "output_tokens": item["usage"].get("output_tokens", 0),
                "started_at": item["started_at"],
            }
            for item in usage["generations"]
        ),
        key=lambda item: item["duration"] or 0,
        reverse=True,
    )[:12]
    agent_span_total = sum(item["duration"] or 0 for item in agent_spans)
    return {
        "elapsed_seconds": run["elapsed_seconds"],
        "status": run["status"],
        "error": run["error"],
        "task_id": run["task_id"],
        "passed": run["validation"]["passed"],
        "total_tokens": usage["totals"]["total_tokens"],
        "input_tokens": usage["totals"]["input_tokens"],
        "output_tokens": usage["totals"]["output_tokens"],
        "cached_tokens": usage["totals"]["cached_tokens"],
        "cache_write_tokens": usage["totals"]["cache_write_tokens"],
        "cache_hit_ratio": round(
            usage["totals"]["cached_tokens"]
            / usage["totals"]["input_tokens"],
            4,
        )
        if usage["totals"]["input_tokens"]
        else 0.0,
        "model_calls": usage["totals"]["requests"] or usage["totals"]["generation_spans"],
        "generation_spans": usage["totals"]["generation_spans"],
        "by_agent": usage["by_agent"],
        "agent_span_seconds_total": round(agent_span_total, 3),
        "tool_calls_total": len(tool_calls),
        "tool_calls_failed": sum(1 for item in tool_calls if item.get("status") == "failed"),
        "tool_calls_unsettled": sum(
            1 for item in tool_calls if item.get("status") == "unsettled"
        ),
        "tool_seconds_total": round(
            sum(float(item.get("duration") or 0) for item in tool_calls), 3
        ),
        "retry_events": len(retries),
        "plan_steps_merged": run["events"]["counts"].get("plan_steps_merged", 0),
        "slowest_model_calls": slowest,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="复杂任务基准探针")
    parser.add_argument("--task", choices=sorted(TASKS), default="sales")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=1200.0)
    parser.add_argument(
        "--root",
        default=None,
        help="证据根目录（默认 sandbox/bench-<时间戳>）",
    )
    args = parser.parse_args(argv)

    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass

    _load_secrets()
    # 项目在导入时用环境变量关闭了 SDK 追踪；探针只在本进程内重新启用，
    # 并把处理器替换为本地收集器，不会向任何外部服务导出。
    set_tracing_disabled(False)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = (
        Path(args.root)
        if args.root
        else PROJECT_ROOT / "sandbox" / f"bench-{stamp}"
    )
    root.mkdir(parents=True, exist_ok=True)

    collectors: list[UsageCollector] = []

    completed_runs = 0
    for index in range(1, max(1, args.runs) + 1):
        collector = UsageCollector()
        collectors.append(collector)
        # 每次运行只挂本轮的收集器，避免上一轮 span 混入
        set_trace_processors([collector])
        run_dir = root / f"run-{index}"
        print(f"[run {index}] 开始：{args.task} -> {run_dir}", file=sys.stderr)
        try:
            run = asyncio.run(
                _run_once(args.task, run_dir, args.timeout, collector)
            )
        except Exception as exc:  # 进程级异常也写入证据，不静默
            print(f"[run {index}] 探针异常：{type(exc).__name__}: {exc}", file=sys.stderr)
            run = {
                "task": args.task,
                "elapsed_seconds": 0.0,
                "error": f"{type(exc).__name__}: {exc}",
                "status": "probe_error",
                "task_id": None,
                "final_result": "",
                "plan_versions": [],
                "agent_history": [],
                "tool_calls": [],
                "events": {"counts": {}, "token_events": [], "retries": [], "lifecycle": []},
                "validation": {"passed": False, "checklist": {}},
                "artifacts": [],
                "usage": collector.summary(),
            }
        summary = _summarize_run(run)
        run["summary"] = summary
        completed_runs += 1
        spans_path = run_dir / "spans.json"
        spans_path.write_text(
            json.dumps(
                {
                    "usage": run["usage"],
                    "trace_spans": run.get("trace_spans", []),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        run.pop("trace_spans", None)
        evidence_path = run_dir / "evidence.json"
        evidence_path.write_text(
            json.dumps(run, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        reads = run.get("read_stats") or {}
        print(
            f"[run {index}] {summary['status']} "
            f"{summary['elapsed_seconds']}s tokens={summary['total_tokens']} "
            f"tools={summary['tool_calls_total']} "
            f"failed={summary['tool_calls_failed']} passed={summary['passed']} "
            f"merged={summary['plan_steps_merged']} "
            f"agent_runs={reads.get('agent_runs', 0)} "
            f"reads={reads.get('calls', 0)} "
            f"dup_windows={reads.get('in_run_repeat_windows', 0)} "
            f"collapsed={reads.get('collapsed', 0)}",
            file=sys.stderr,
        )

    aggregate: dict[str, Any] = {
        "generated_at": _now_iso(),
        "task": args.task,
        "runs": [],
    }
    for index, collector in enumerate(collectors, start=1):
        run_evidence = root / f"run-{index}" / "evidence.json"
        if run_evidence.is_file():
            aggregate["runs"].append(
                json.loads(run_evidence.read_text(encoding="utf-8"))
            )
    if aggregate["runs"]:
        elapsed_values = [run["summary"]["elapsed_seconds"] for run in aggregate["runs"]]
        token_values = [run["summary"]["total_tokens"] for run in aggregate["runs"]]
        aggregate["summary"] = {
            "runs": len(aggregate["runs"]),
            "all_passed": all(run["summary"]["passed"] for run in aggregate["runs"]),
            "elapsed_seconds": elapsed_values,
            "total_tokens": token_values,
            "elapsed_avg": round(sum(elapsed_values) / len(elapsed_values), 3),
            "tokens_avg": round(sum(token_values) / len(token_values), 1),
            "total_retry_events": sum(
                run["summary"]["retry_events"] for run in aggregate["runs"]
            ),
            "failed_tool_calls": sum(
                run["summary"]["tool_calls_failed"] for run in aggregate["runs"]
            ),
        }
    aggregate_path = root / "evidence.json"
    aggregate_path.write_text(
        json.dumps(aggregate, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"证据已写入：{aggregate_path}", file=sys.stderr)
    return 0 if aggregate.get("summary", {}).get("all_passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())

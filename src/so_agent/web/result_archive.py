"""Web 结果沉淀：最近一次结果落盘 + 可选历史接口。

默认 ``keep_history=False``：每次新任务开始时清空上一次结果，只保留
``latest.md`` / ``latest.json``，与“Web 只是展示页面、不保留历史”的
使用方式一致。需要跨任务保留过程与问题时，打开
``web_keep_history`` 并传入 :class:`HistoryStore`，结果会追加到 JSONL；
调用方不依赖具体存储，便于以后替换为数据库或外部项目接口。
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any, Protocol


class HistoryStore(Protocol):
    """历史记录接口（预留）：按时间追加结果记录。"""

    def append(self, record: dict[str, Any]) -> None: ...

    def list_recent(self, limit: int = 20) -> list[dict[str, Any]]: ...


class JsonlHistoryStore:
    """JSONL 历史实现：写入简单、可追加、易于外部导入。"""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._lock = threading.Lock()
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: dict[str, Any]) -> None:
        line = json.dumps(record, ensure_ascii=False, default=str)
        with self._lock:
            with self._path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")

    def list_recent(self, limit: int = 20) -> list[dict[str, Any]]:
        count = max(1, int(limit))
        if not self._path.is_file():
            return []
        with self._lock:
            lines = self._path.read_text(encoding="utf-8").splitlines()
        records: list[dict[str, Any]] = []
        for line in reversed(lines[-count:]):
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return records


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _format_size(size: Any) -> str:
    try:
        value = int(size)
    except (TypeError, ValueError):
        return "-"
    if value < 1024:
        return f"{value} B"
    if value < 1024 * 1024:
        return f"{value / 1024:.1f} KiB"
    return f"{value / (1024 * 1024):.1f} MiB"


def render_markdown(payload: dict[str, Any]) -> str:
    """把结果载荷渲染为人类可读的 Markdown。"""
    lines: list[str] = ["# 任务结果", ""]
    meta = [
        ("任务 ID", payload.get("task_id")),
        ("状态", payload.get("status")),
        ("类型", payload.get("task_type")),
        ("创建时间", payload.get("created_at")),
        ("开始时间", payload.get("started_at")),
        ("结束时间", payload.get("finished_at")),
        ("Token 总量", payload.get("total_tokens")),
    ]
    for label, value in meta:
        if value not in (None, ""):
            lines.append(f"- {label}: {value}")
    by_agent = payload.get("tokens_by_agent") or {}
    if by_agent:
        detail = "、".join(f"{name} {tokens}" for name, tokens in by_agent.items())
        lines.append(f"- Token 分布: {detail}")
    lines.append("")

    sections = [
        ("目标", payload.get("objective")),
        ("约束", payload.get("constraints")),
        ("验收标准", payload.get("acceptance_criteria")),
        ("最终结果", payload.get("final_result")),
        ("评审结论", payload.get("review_summary")),
        ("交付提示", payload.get("notice")),
        ("返工前结果", payload.get("previous_result")),
        ("错误", payload.get("error")),
    ]
    for title, value in sections:
        if isinstance(value, str) and value.strip():
            lines.extend([f"## {title}", "", value.strip(), ""])

    plan = payload.get("plan") or []
    if plan:
        lines.extend(["## 计划", ""])
        for item in plan:
            title = item.get("title") or item.get("subtask_id") or "-"
            status = item.get("status") or ""
            suffix = f"（{status}）" if status else ""
            lines.append(f"- {title}{suffix}")
        lines.append("")

    audit = payload.get("audit")
    if isinstance(audit, dict) and audit:
        lines.extend(["## 审计", ""])
        lines.append(f"- 结论: {audit.get('verdict', '-')}")
        lines.append(f"- 事件数: {audit.get('event_count', '-')}")
        for finding in (audit.get("findings") or [])[:20]:
            if isinstance(finding, dict):
                lines.append(
                    f"- [{finding.get('severity', '?')}] "
                    f"{finding.get('message', '')}"
                )
        lines.append("")

    artifacts = payload.get("artifacts") or []
    if artifacts:
        lines.extend(["## 产物", ""])
        for item in artifacts[:MAX_MARKDOWN_ARTIFACTS]:
            path = item.get("path", "-")
            size = _format_size(item.get("size"))
            lines.append(f"- `{path}`（{size}）")
        if len(artifacts) > MAX_MARKDOWN_ARTIFACTS:
            lines.append(f"- …共 {len(artifacts)} 个文件")
        lines.append("")

    escalation = payload.get("human_escalation")
    if isinstance(escalation, dict) and escalation:
        lines.extend(["## 人工兜底", ""])
        for key in ("reason", "recommended_action", "question"):
            value = escalation.get(key)
            if value:
                lines.append(f"- {key}: {value}")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


MAX_MARKDOWN_ARTIFACTS = 100


class FileResultArchive:
    """文件结果存档：``latest.json`` + ``latest.md``（可选历史）。"""

    def __init__(
        self,
        root: str | Path,
        *,
        keep_history: bool = False,
        history: HistoryStore | None = None,
    ) -> None:
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)
        self._keep_history = bool(keep_history)
        self._history = history
        self._lock = threading.Lock()

    @property
    def root(self) -> Path:
        return self._root

    @property
    def latest_json_path(self) -> Path:
        return self._root / "latest.json"

    @property
    def latest_markdown_path(self) -> Path:
        return self._root / "latest.md"

    def begin_run(self) -> None:
        """新任务开始：未启用历史时移除上一次结果。"""
        if self._keep_history:
            return
        with self._lock:
            for path in (self.latest_json_path, self.latest_markdown_path):
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    pass

    def publish(self, task_id: str, payload: dict[str, Any]) -> dict[str, str]:
        """写入最近一次结果；启用历史时追加一条记录。"""
        markdown = render_markdown(payload)
        with self._lock:
            _atomic_write(
                self.latest_json_path,
                json.dumps(payload, ensure_ascii=False, indent=2, default=str)
                + "\n",
            )
            _atomic_write(self.latest_markdown_path, markdown)
        if self._keep_history and self._history is not None:
            self._history.append({"task_id": task_id, **payload})
        return {
            "json": str(self.latest_json_path),
            "markdown": str(self.latest_markdown_path),
        }

    def latest(self) -> dict[str, Any] | None:
        path = self.latest_json_path
        if not path.is_file():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    def latest_markdown(self) -> str | None:
        path = self.latest_markdown_path
        if not path.is_file():
            return None
        try:
            return path.read_text(encoding="utf-8")
        except OSError:
            return None


__all__ = [
    "FileResultArchive",
    "HistoryStore",
    "JsonlHistoryStore",
    "render_markdown",
]

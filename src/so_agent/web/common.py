"""Web 控制台共享常量、辅助函数与内存视图。

独立成模块以便 manager 与快照/旅程/时间线/产物等子模块共享同一份
类型与展示常量，避免拆分后出现重复定义或循环导入。
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from so_agent.context import ProjectContext
from so_agent.models import ClassificationResult, HumanGuidance, TaskType
from so_agent.orchestrator import OrchestratorOutput
from so_agent.runtime.logging_setup import get_logger

if TYPE_CHECKING:
    from so_agent.workflow import WorkflowEngine

logger = get_logger("web.manager")

TERMINAL_STATUSES: frozenset[str] = frozenset(
    {"completed", "failed", "cancelled", "human_required"}
)

MAX_OBJECTIVE_CHARS = 20_000
MAX_ARTIFACTS = 200
MAX_ARTIFACT_BYTES = 20 * 1024 * 1024
MAX_TASKS_IN_MEMORY = 100
_MAX_TIMELINE_EVENTS = 300
_MAX_PREVIEW_CHARS = 600
_TASK_ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")

_INTERNAL_DIR_NAMES: frozenset[str] = frozenset(
    {
        "generated_tools",
        "sandbox",
        "checkpoints",
        "__pycache__",
        ".git",
        "node_modules",
        ".pytest_cache",
    }
)
# 调度器与沙箱子进程创建的运行目录（task_<id> / run_<id>）不是业务产物
_INTERNAL_DIR_PREFIXES: tuple[str, ...] = ("task_", "run_")
_INTERNAL_FILE_NAMES: frozenset[str] = frozenset(
    {
        "task_store.db",
        "task_store.db-journal",
        "task_store.db-wal",
        "task_store.db-shm",
    }
)

# Windows 注册表中 .csv 等扩展名的类型映射不统一（如 application/vnd.ms-excel），
# 这里显式声明常见文本产物，保证内联预览可用
_TEXT_EXTENSIONS: dict[str, str] = {
    ".md": "text/markdown",
    ".markdown": "text/markdown",
    ".csv": "text/csv",
    ".tsv": "text/tab-separated-values",
    ".log": "text/plain",
    ".txt": "text/plain",
    ".yaml": "text/yaml",
    ".yml": "text/yaml",
    ".jsonl": "text/plain",
    ".py": "text/x-python",
    ".js": "text/javascript",
}

_ROUTE_LABELS: dict[TaskType, str] = {
    TaskType.DIRECT_ANSWER: "直接回答",
    TaskType.DEEP_REASONING: "深度推理",
    TaskType.PERSISTENT_TASK: "持久化任务",
    TaskType.FULL_WORKFLOW: "完整工作流",
}
_ROUTE_LABELS_EN: dict[TaskType, str] = {
    TaskType.DIRECT_ANSWER: "Direct answer",
    TaskType.DEEP_REASONING: "Deep reasoning",
    TaskType.PERSISTENT_TASK: "Persistent task",
    TaskType.FULL_WORKFLOW: "Full workflow",
}

_ROUTE_BRANCHES: tuple[dict[str, str], ...] = (
    {
        "type": TaskType.DIRECT_ANSWER.value,
        "label": "直接回答",
        "condition": "简单 + 短：无需工具，一次模型调用即可回答",
        "path": "DirectAnswerPath",
    },
    {
        "type": TaskType.DEEP_REASONING.value,
        "label": "深度推理",
        "condition": "复杂 + 短：需要深度推理，但不调用工具",
        "path": "DeepReasoningPath",
    },
    {
        "type": TaskType.PERSISTENT_TASK.value,
        "label": "持久化任务",
        "condition": "简单 + 长：需要持久状态、定时或周期调度",
        "path": "PersistentTaskPath",
    },
    {
        "type": TaskType.FULL_WORKFLOW.value,
        "label": "完整工作流",
        "condition": "复杂 + 长：多步骤协作、工具调用或产物生成",
        "path": "WorkflowEngine",
    },
)
_ROUTE_BRANCHES_EN: tuple[dict[str, str], ...] = (
    {
        "type": TaskType.DIRECT_ANSWER.value,
        "label": "Direct answer",
        "condition": "Simple + short: no tools needed, one model call is enough",
        "path": "DirectAnswerPath",
    },
    {
        "type": TaskType.DEEP_REASONING.value,
        "label": "Deep reasoning",
        "condition": "Complex + short: deep reasoning without tool calls",
        "path": "DeepReasoningPath",
    },
    {
        "type": TaskType.PERSISTENT_TASK.value,
        "label": "Persistent task",
        "condition": "Simple + long: needs durable state or scheduled runs",
        "path": "PersistentTaskPath",
    },
    {
        "type": TaskType.FULL_WORKFLOW.value,
        "label": "Full workflow",
        "condition": "Complex + long: multi-step collaboration, tools or artifacts",
        "path": "WorkflowEngine",
    },
)

_ROUTE_METHOD_LABELS: dict[str, str] = {
    "rule": "规则预判",
    "llm": "模型分类",
    "upgrade": "工具意图升级",
    "fallback": "失败降级",
    "disabled": "路由器禁用",
}
_ROUTE_METHOD_LABELS_EN: dict[str, str] = {
    "rule": "Rule pre-check",
    "llm": "Model classification",
    "upgrade": "Tool intent upgrade",
    "fallback": "Fallback",
    "disabled": "Router disabled",
}

_GENERATED_TOOL_ACTIONS: frozenset[str] = frozenset(
    {
        "gap_detected",
        "candidate_created",
        "tool_verified",
        "tool_verify_skipped",
        "registered",
        "gap_unresolved",
    }
)
_GENERATED_TOOL_STEP_LABELS: dict[str, str] = {
    "gap_detected": "发现缺口",
    "candidate_created": "生成候选",
    "tool_verified": "沙箱验收",
    "tool_verify_skipped": "跳过验收",
    "registered": "注册完成",
    "gap_unresolved": "补齐失败",
}
_GENERATED_TOOL_STEP_LABELS_EN: dict[str, str] = {
    "gap_detected": "Gap detected",
    "candidate_created": "Candidate created",
    "tool_verified": "Sandbox verified",
    "tool_verify_skipped": "Verification skipped",
    "registered": "Registered",
    "gap_unresolved": "Gap unresolved",
}

_STATUS_LABELS: dict[str, str] = {
    "queued": "排队中",
    "running": "执行中",
    "completed": "已完成",
    "completed_provisional": "已交付·评审中",
    "failed": "失败",
    "cancelled": "已取消",
    "human_required": "待人工",
    "scheduled": "已调度",
    "pending": "待执行",
    "reused": "复用结果",
    "archived": "历史版本",
}
_STATUS_LABELS_EN: dict[str, str] = {
    "queued": "Queued",
    "running": "Running",
    "completed": "Completed",
    "completed_provisional": "Delivered · reviewing",
    "failed": "Failed",
    "cancelled": "Cancelled",
    "human_required": "Human required",
    "scheduled": "Scheduled",
    "pending": "Pending",
    "reused": "Reused",
    "archived": "Archived",
}


def _route_labels(lang: str) -> dict[TaskType, str]:
    return _ROUTE_LABELS_EN if lang == "en" else _ROUTE_LABELS


def _status_label(lang: str, value: str) -> str:
    table = _STATUS_LABELS_EN if lang == "en" else _STATUS_LABELS
    return table.get(value, value)
_MAX_TOOL_CALLS = 200

_PHASE_RANK: dict[str, int] = {
    "received": 0,
    "planning": 1,
    "plan_review": 2,
    "replanning": 2,
    "executing": 3,
    "replacement_requested": 3,
    "replacement_executing": 3,
    "decomposition_issue": 3,
    "aggregating": 4,
    "result_review": 5,
    "completed_provisional": 5,
    "completed": 6,
    "failed": 6,
    "cancelled": 6,
    "human_required": 6,
}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _preview(text: str | None, limit: int = _MAX_PREVIEW_CHARS) -> str:
    value = (text or "").strip()
    if len(value) <= limit:
        return value
    return value[:limit] + "…"


def _split_items(raw: str) -> list[str]:
    """把约束/验收标准文本按换行或分号拆分为列表（仅用于展示）。"""
    if not raw.strip():
        return []
    parts: list[str] = []
    for line in raw.replace("；", ";").splitlines():
        for item in line.split(";"):
            cleaned = item.strip()
            if cleaned:
                parts.append(cleaned)
    return parts


@dataclass
class WebTask:
    """单个 Web 任务记录（内存视图，终态可由持久化记录还原）。"""

    task_id: str
    objective: str
    constraints: str
    acceptance_criteria: str
    sandbox_dir: Path
    context: ProjectContext
    status: str = "queued"
    created_at: str = field(default_factory=_utc_now_iso)
    started_at: str | None = None
    finished_at: str | None = None
    classification: ClassificationResult | None = None
    output: OrchestratorOutput | None = None
    error: str | None = None
    cancel_requested: bool = False
    engine: WorkflowEngine | None = None
    runner: asyncio.Task[None] | None = None
    last_guidance: HumanGuidance | None = None
    restored_snapshot: dict[str, dict[str, Any]] = field(default_factory=dict)
    restored_tokens: dict[str, Any] = field(default_factory=dict)
    restored: bool = False


class ArtifactError(Exception):
    """产物路径非法或不可读取。"""


class GuidanceStateError(Exception):
    """人工指导不可提交：任务不存在、不在人工兜底态或仍在运行。"""

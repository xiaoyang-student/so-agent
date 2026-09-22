"""事件记录器（EventLogger）。

职责与行为约定：
- 以统一结构记录编排过程中的全部事件（任务、Agent、授权、替换、通信、
  人工升级等），首版仅追加写入内存列表，接口稳定后可平滑替换为持久化存储；
- 事件 payload 强制 JSON 兼容：Pydantic 模型、datetime、set/tuple 等对象
  会被自动转换为 JSON 可序列化形态后再写入，不可序列化的内容会被拒绝；
- 支持按 ``task_id`` / ``agent_id`` / ``event_type`` 过滤查询；
- 支持订阅回调（subscribe），供审计、监控与人工升级材料组装使用。

全局约束：所有 Agent 之间的通信必须严格使用 JSON；本模块是事件留痕入口，
同样遵循该约束，确保任何事件都可被序列化、审计与回放。
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any, Callable, TypeAlias

from pydantic import BaseModel

from so_agent.context import ProjectContext

# ---------------------------------------------------------------------------
# 事件类型常量（全量清单；新增事件类型必须先在此登记）
# ---------------------------------------------------------------------------
EVENT_TASK_CREATED: str = "task_created"
EVENT_STATUS_TRANSITION: str = "status_transition"
EVENT_TASK_DISPATCHED: str = "task_dispatched"
EVENT_TOOL_GRANTED: str = "tool_granted"
EVENT_TOOL_REVOKED: str = "tool_revoked"
EVENT_TOOL_CALLED: str = "tool_called"
EVENT_AGENT_CREATED: str = "agent_created"
EVENT_AGENT_RELEASED: str = "agent_released"
EVENT_REPLACEMENT_REQUESTED: str = "replacement_requested"
EVENT_DECOMPOSITION_ISSUE: str = "decomposition_issue"
EVENT_REPLAN: str = "replan"
EVENT_HUMAN_ESCALATION: str = "human_escalation"
EVENT_MESSAGE_ROUTED: str = "message_routed"
EVENT_MESSAGE_REJECTED: str = "message_rejected"

ALL_EVENT_TYPES: frozenset[str] = frozenset(
    {
        EVENT_TASK_CREATED,
        EVENT_STATUS_TRANSITION,
        EVENT_TASK_DISPATCHED,
        EVENT_TOOL_GRANTED,
        EVENT_TOOL_REVOKED,
        EVENT_TOOL_CALLED,
        EVENT_AGENT_CREATED,
        EVENT_AGENT_RELEASED,
        EVENT_REPLACEMENT_REQUESTED,
        EVENT_DECOMPOSITION_ISSUE,
        EVENT_REPLAN,
        EVENT_HUMAN_ESCALATION,
        EVENT_MESSAGE_ROUTED,
        EVENT_MESSAGE_REJECTED,
    }
)

# 事件类型名与事件回调签名
EventType: TypeAlias = str
EventCallback: TypeAlias = Callable[[dict[str, Any]], None]


def _utc_now() -> datetime:
    """返回当前 UTC 时间（与 models 保持同一时间基准）。"""
    return datetime.now(timezone.utc)


def _json_safe(value: Any) -> Any:
    """将常见对象递归转换为 JSON 兼容形态。

    - Pydantic 模型 → ``model_dump(mode="json")`` 结果；
    - datetime → ISO 8601 字符串；
    - dict / list / tuple / set → 同构容器；
    - 其余值原样返回（是否合法由 ``json.dumps`` 最终校验）。
    """
    if isinstance(value, BaseModel):
        return _json_safe(value.model_dump(mode="json"))
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_safe(item) for item in value]
    return value


class EventLogger:
    """编排过程事件记录器（内存实现）。"""

    def __init__(
        self,
        *,
        context: ProjectContext | None = None,
        max_events: int | None = None,
    ) -> None:
        """初始化记录器。

        Args:
            context: 可选的共享上下文；提供时事件会同步追加到
                ``context.event_log``，便于编排器与人工升级材料统一取用。
            max_events: 内部列表保留的最大事件数（None 表示不限制）；
                仅作用于本对象，不影响 ``context.event_log``。
        """
        self._context = context
        self._max_events = max_events
        self._events: list[dict[str, Any]] = []
        self._subscribers: list[EventCallback] = []

    # ------------------------------------------------------------------
    # 记录
    # ------------------------------------------------------------------
    def log_event(
        self,
        event_type: str,
        payload: dict[str, Any] | BaseModel | None = None,
        timestamp: datetime | None = None,
    ) -> dict[str, Any]:
        """记录一条事件并返回事件记录（JSON 兼容 dict）。

        Args:
            event_type: 事件类型，必须是 ``ALL_EVENT_TYPES`` 中已登记的类型。
            payload: 事件内容；支持 dict / Pydantic 模型 / None。写入前统一
                转换为 JSON 兼容形态；顶层 ``task_id`` / ``agent_id`` 会被
                提取为索引字段，供过滤查询使用。
            timestamp: 事件时间；默认当前 UTC 时间；naive 时间按 UTC 处理。

        Raises:
            ValueError: 事件类型未登记，或 payload 无法序列化为 JSON 时。
            TypeError: payload 类型不受支持时。
        """
        if not isinstance(event_type, str) or event_type not in ALL_EVENT_TYPES:
            raise ValueError(
                f"未登记的事件类型：{event_type!r}；"
                f"合法事件类型见 ALL_EVENT_TYPES：{sorted(ALL_EVENT_TYPES)}"
            )

        if isinstance(payload, BaseModel):
            data = payload.model_dump(mode="json")
        elif payload is None:
            data = {}
        elif isinstance(payload, dict):
            data = payload
        else:
            raise TypeError(
                "payload 必须是 dict / Pydantic 模型 / None，"
                f"实际为 {type(payload).__name__}"
            )
        data = _json_safe(data)
        try:
            json.dumps(data, ensure_ascii=False)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"事件 payload 无法序列化为 JSON：{exc}") from exc

        ts = timestamp if timestamp is not None else _utc_now()
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)

        record: dict[str, Any] = {
            "event_id": uuid.uuid4().hex[:12],
            "event_type": event_type,
            "timestamp": ts.isoformat(),
            "task_id": data.get("task_id"),
            "agent_id": data.get("agent_id"),
            "payload": data,
        }

        self._events.append(record)
        if self._context is not None:
            self._context.event_log.append(record)
        if self._max_events is not None and len(self._events) > self._max_events:
            del self._events[: len(self._events) - self._max_events]

        for callback in list(self._subscribers):
            try:
                callback(record)
            except Exception:  # 订阅者异常不得影响主流程
                continue
        return record

    # ------------------------------------------------------------------
    # 查询与订阅
    # ------------------------------------------------------------------
    def get_events(
        self,
        task_id: str | None = None,
        agent_id: str | None = None,
        event_type: str | None = None,
    ) -> list[dict[str, Any]]:
        """按条件过滤事件（未提供的条件不参与过滤），返回浅拷贝列表。"""
        matched: list[dict[str, Any]] = []
        for record in self._events:
            if task_id is not None and record.get("task_id") != task_id:
                continue
            if agent_id is not None and record.get("agent_id") != agent_id:
                continue
            if event_type is not None and record.get("event_type") != event_type:
                continue
            matched.append(dict(record))
        return matched

    def subscribe(self, callback: EventCallback) -> None:
        """注册事件回调（重复注册自动忽略）；回调异常不影响主流程。"""
        if callback not in self._subscribers:
            self._subscribers.append(callback)

    def unsubscribe(self, callback: EventCallback) -> None:
        """移除已注册的事件回调（不存在时静默忽略）。"""
        if callback in self._subscribers:
            self._subscribers.remove(callback)

    def clear(self) -> None:
        """清空内部事件列表（不影响 ``context.event_log`` 中已同步记录）。"""
        self._events.clear()

    @property
    def events(self) -> list[dict[str, Any]]:
        """返回内部事件列表的浅拷贝。"""
        return list(self._events)

    def __len__(self) -> int:
        """返回当前已记录的事件数量。"""
        return len(self._events)

"""主管路由器（SupervisorRouter）。

职责与行为约定：
- 通信拓扑为“唯一主管 + 各子 Agent”的星形结构：只允许主管 Agent 与子
  Agent 之间的纵向通信；子 Agent 之间的横向通信一律拒绝；
- 全局约束：所有 Agent 之间的通信必须严格使用 JSON——route_message 会把
  消息统一规范化为 JSON 对象（str 解析 / dict / Pydantic 模型序列化）并
  经 Pydantic 校验；任何自由文本、非法 JSON、数组/标量消息一律拒绝；
- 防冒充：消息内若携带 ``sender_id`` / ``recipient_id`` 字段，必须与路由
  参数一致，否则拒绝转发；
- 所有路由成功与拒绝均记录事件（message_routed / message_rejected），
  供审计与人工升级材料组装。

说明：router 只负责通信边界校验与留痕，实际消息投递（工具调用）由编排器
完成；本对象仅供主管 Agent 与控制层使用。
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Iterable

from pydantic import BaseModel, TypeAdapter, ValidationError

from so_agent.runtime.events import (
    EVENT_MESSAGE_REJECTED,
    EVENT_MESSAGE_ROUTED,
    EventLogger,
)
from so_agent.runtime.registry import DEFAULT_SUPERVISOR_ID

if TYPE_CHECKING:  # 仅用于类型标注，避免运行期循环依赖
    from so_agent.runtime.registry import AgentRegistry

# 用于校验消息必须是 JSON 对象并经 Pydantic 解析
_JSON_OBJECT_ADAPTER: TypeAdapter[dict[str, Any]] = TypeAdapter(dict[str, Any])


class CommunicationError(Exception):
    """通信边界错误：横向通信、非法 JSON、消息身份不一致等。"""


def _to_jsonable(value: Any) -> Any:
    """将常见对象递归转换为 JSON 兼容形态（模型→dict、时间→ISO 字符串）。"""
    if isinstance(value, BaseModel):
        return _to_jsonable(value.model_dump(mode="json"))
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_to_jsonable(item) for item in value]
    return value


class SupervisorRouter:
    """主管路由器：纵向通信校验 + 消息 JSON 强校验 + 全量留痕。"""

    def __init__(
        self,
        *,
        supervisor_id: str = DEFAULT_SUPERVISOR_ID,
        registry: "AgentRegistry | None" = None,
        events: EventLogger | None = None,
        extra_recipients: Iterable[str] | None = None,
    ) -> None:
        """初始化路由器。

        Args:
            supervisor_id: 主管 Agent 标识（通信星形结构的中心）。
            registry: 可选 Agent 注册表；提供时通信参与方必须已登记
                （或在 extra_recipients 中），否则视为非法通信。
            events: 事件记录器；默认新建内存实例。
            extra_recipients: 预置合法通信方（如三个固定工具入口名），
                用于注册表检查豁免。
        """
        self._supervisor_id = supervisor_id
        self._registry = registry
        self._extra_recipients: set[str] = set(extra_recipients or ())
        self.events = events if events is not None else EventLogger()

    # ------------------------------------------------------------------
    # 通信边界校验
    # ------------------------------------------------------------------
    def _check_pair(self, sender_id: str, recipient_id: str) -> str | None:
        """校验通信关系；合法返回 None，非法返回拒绝原因。"""
        if (
            not isinstance(sender_id, str)
            or not sender_id.strip()
            or not isinstance(recipient_id, str)
            or not recipient_id.strip()
        ):
            return "发送方/接收方标识不能为空"
        if sender_id == recipient_id:
            return f"拒绝自环消息：{sender_id!r} 不能向自身发送消息"
        if sender_id != self._supervisor_id and recipient_id != self._supervisor_id:
            return (
                f"禁止子 Agent 之间的横向通信：{sender_id!r} → {recipient_id!r}；"
                f"一切跨 Agent 交互必须经由主管 Agent（{self._supervisor_id}）中转"
            )
        if self._registry is not None:
            for party in (sender_id, recipient_id):
                if party == self._supervisor_id or party in self._extra_recipients:
                    continue
                if self._registry.get(party) is None:
                    return (
                        f"通信参与方 {party!r} 未登记为 Agent"
                        "（既不在注册表也不是预置入口），拒绝通信"
                    )
        return None

    def validate_communication(self, sender_id: str, recipient_id: str) -> bool:
        """判断一条通信关系是否合法（合法的唯一形态：主管 ↔ 子 Agent）。"""
        return self._check_pair(sender_id, recipient_id) is None

    # ------------------------------------------------------------------
    # 消息路由
    # ------------------------------------------------------------------
    def route_message(
        self,
        sender_id: str,
        recipient_id: str,
        message: "str | dict[str, Any] | BaseModel",
    ) -> None:
        """校验并记录一条纵向通信消息。

        消息必须为合法 JSON 对象（可被 Pydantic 解析）：
        - str：按 JSON 文本解析，解析失败即拒绝；
        - dict：直接校验（嵌套 Pydantic 模型/datetime 会自动转换）；
        - Pydantic 模型：按 ``model_dump(mode="json")`` 序列化后校验。

        Args:
            sender_id: 发送方标识（主管或已登记子 Agent）。
            recipient_id: 接收方标识（主管或已登记子 Agent）。
            message: 消息内容（严格 JSON 对象语义）。

        Raises:
            CommunicationError: 通信关系非法、消息非 JSON / 非对象、
                或消息声明字段与路由参数不一致时。拒绝事件会写入事件日志
                （message_rejected）。
        """
        reason = self._check_pair(sender_id, recipient_id)
        if reason is not None:
            self.events.log_event(
                EVENT_MESSAGE_REJECTED,
                {
                    "sender_id": sender_id,
                    "recipient_id": recipient_id,
                    "agent_id": sender_id,
                    "reason": reason,
                },
            )
            raise CommunicationError(reason)

        try:
            normalized = self._normalize_message(sender_id, recipient_id, message)
        except CommunicationError as exc:
            self.events.log_event(
                EVENT_MESSAGE_REJECTED,
                {
                    "sender_id": sender_id,
                    "recipient_id": recipient_id,
                    "agent_id": sender_id,
                    "reason": str(exc),
                },
            )
            raise

        self.events.log_event(
            EVENT_MESSAGE_ROUTED,
            {
                "sender_id": sender_id,
                "recipient_id": recipient_id,
                "agent_id": sender_id,
                "message": normalized,
            },
        )

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------
    def _normalize_message(
        self,
        sender_id: str,
        recipient_id: str,
        message: "str | dict[str, Any] | BaseModel",
    ) -> dict[str, Any]:
        """将消息规范化为 JSON 对象（dict）并完成全部格式校验。

        Raises:
            CommunicationError: 消息类型非法、非法 JSON、非对象、
                无法被 Pydantic 解析、含不可序列化内容，或身份字段不一致时。
        """
        if isinstance(message, BaseModel):
            data = _to_jsonable(message.model_dump(mode="json"))
        elif isinstance(message, dict):
            data = _to_jsonable(message)
        elif isinstance(message, str):
            try:
                data = json.loads(message)
            except json.JSONDecodeError as exc:
                raise CommunicationError(
                    f"消息不是合法 JSON（拒绝自由文本/非法格式）：{exc}"
                ) from exc
        else:
            raise CommunicationError(
                f"消息类型非法（{type(message).__name__}）："
                "必须为 str（JSON 文本）/ dict / Pydantic 模型"
            )

        if not isinstance(data, dict):
            raise CommunicationError("消息必须是 JSON 对象（键值映射），拒绝数组/标量")

        try:
            data = _JSON_OBJECT_ADAPTER.validate_python(data)
        except ValidationError as exc:
            raise CommunicationError(f"消息无法被 Pydantic 解析：{exc}") from exc

        for field, expected in (("sender_id", sender_id), ("recipient_id", recipient_id)):
            actual = data.get(field)
            if actual is not None and actual != expected:
                raise CommunicationError(
                    f"消息声明的 {field}={actual!r} 与路由参数 {expected!r} 不一致，"
                    "拒绝转发（疑似身份冒充）"
                )
        data.setdefault("sender_id", sender_id)
        data.setdefault("recipient_id", recipient_id)

        try:
            json.dumps(data, ensure_ascii=False)
        except (TypeError, ValueError) as exc:
            raise CommunicationError(f"消息包含不可 JSON 序列化的内容：{exc}") from exc
        return data

    @property
    def supervisor_id(self) -> str:
        """返回主管 Agent 标识。"""
        return self._supervisor_id

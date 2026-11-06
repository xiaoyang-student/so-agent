"""人工兜底处理器（从 workflow_support 拆分，行为不变）。"""

from __future__ import annotations

from typing import Any

from so_agent.models import HumanEscalationRequest


class HumanEscalationHandler:
    """人工兜底处理器：记录请求，并在配置后投递外部通知。"""

    def __init__(self, notifier: Any = None) -> None:
        """初始化处理器；``notifier`` 具备 ``notify(request)`` 协程。"""
        self._requests: list[HumanEscalationRequest] = []
        self._notifier = notifier
        self._last_notification: Any = None

    async def escalate(self, request: HumanEscalationRequest) -> None:
        """记录升级请求；配置通知器时投递外部通知（失败不抛出）。

        Args:
            request: 人工升级材料包（原始任务、历代计划、失败历史与证据）。
        """
        self._requests.append(request)
        if self._notifier is None:
            return
        self._last_notification = await self._notifier.notify(request)

    async def check_resolution(self, task_id: str) -> bool:
        """返回 False：人工处理通过 ``resume_with_guidance`` 显式恢复。"""
        return False

    @property
    def pending_requests(self) -> list[HumanEscalationRequest]:
        """返回已受理的升级请求列表（浅拷贝，供审计与测试）。"""
        return list(self._requests)

    @property
    def last_notification(self) -> Any:
        """最近一次通知结果（``NotificationResult`` 或 None）。"""
        return self._last_notification


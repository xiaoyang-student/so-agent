"""Token 预算与熔断（默认启用，按任务总量与单 Agent 累计双阈值控制）。

``token_budget_enabled=False`` 时只做用量累加与快照，不拦截；启用后
按任务级（总 token）与子 Agent 级双阈值检查，超限写入
``token_budget_exceeded`` 审计事件，并由控制层调用 ``enforce`` 抛出
``TokenBudgetExceeded`` 终止当前流水线。默认上限按历史真实任务的最坏
观测值留出余量（复杂任务峰值约 57 万 tokens），部署方可按成本策略覆盖。

归属：``WorkflowEngine`` 在任务创建后生成一个 tracker 并挂到
``ProjectContext.token_budget``，控制层的模型调用与嵌套 Agent 的用量
回调（``make_usage_recorder``）都累加到同一 tracker，保证总量口径一致。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from so_agent.runtime.events import EVENT_TOKEN_BUDGET_EXCEEDED, EventLogger


@dataclass(frozen=True)
class TokenBudgetLimits:
    """预算阈值；``<= 0`` 表示该维度不限制。"""

    enabled: bool = False
    max_total_tokens: int = 0
    max_agent_tokens: int = 0

    @classmethod
    def from_settings(cls, settings: Any) -> "TokenBudgetLimits":
        """从全局配置构造（未知字段缺失时退回默认关闭）。"""
        return cls(
            enabled=bool(getattr(settings, "token_budget_enabled", False)),
            max_total_tokens=int(
                getattr(settings, "token_budget_max_total_tokens", 0) or 0
            ),
            max_agent_tokens=int(
                getattr(settings, "token_budget_max_agent_tokens", 0) or 0
            ),
        )

    @property
    def active(self) -> bool:
        """是否真正启用控制（开关打开且至少一个正阈值）。"""
        return self.enabled and (
            self.max_total_tokens > 0 or self.max_agent_tokens > 0
        )


@dataclass(frozen=True)
class BudgetViolation:
    """一次超限判定结果。"""

    scope: str  # "task" | "agent"
    limit: int
    observed: int
    agent_id: str | None = None
    message: str = ""


class TokenBudgetExceeded(BaseException):
    """启用预算控制后超限；由 ``WorkflowEngine.run`` 显式映射为失败终态。

    刻意继承 ``BaseException``（与 ``PipelineStoppedError`` 同策略）：
    协调器与调度器存在大量 ``except Exception`` 宽捕获（把模型调用失败
    降级为可重试结果），若继承 ``Exception``，预算熔断会被吞掉并触发
    重试 / 重规划，反而继续消耗预算。
    """

    def __init__(self, violation: BudgetViolation) -> None:
        self.violation = violation
        super().__init__(violation.message)


@dataclass
class _State:
    total_tokens: int = 0
    per_agent: dict[str, int] = field(default_factory=dict)
    per_subtask: dict[str, int] = field(default_factory=dict)
    violations: list[dict[str, Any]] = field(default_factory=list)
    reported: set[tuple[str, str]] = field(default_factory=set)


class TokenBudgetTracker:
    """任务级 token 用量累加器与阈值检查器（启用时拦截，关闭时只观察）。"""

    def __init__(
        self,
        limits: TokenBudgetLimits | None = None,
        *,
        events: EventLogger | None = None,
        task_id: str | None = None,
    ) -> None:
        self._limits = limits or TokenBudgetLimits()
        self._events = events
        self._task_id = task_id
        self._state = _State()

    # ------------------------------------------------------------------
    # 属性
    # ------------------------------------------------------------------
    @property
    def limits(self) -> TokenBudgetLimits:
        return self._limits

    @property
    def total_tokens(self) -> int:
        return self._state.total_tokens

    @property
    def remaining_tokens(self) -> int | None:
        """任务级剩余可用 token；未启用任务级上限时返回 None。"""
        if not self._limits.active or self._limits.max_total_tokens <= 0:
            return None
        return max(0, self._limits.max_total_tokens - self._state.total_tokens)

    def tokens_for(self, agent_id: str) -> int:
        return self._state.per_agent.get(agent_id, 0)

    def tokens_for_subtask(self, subtask_id: str) -> int:
        """返回某子任务的累计 token 用量（跨尝试累计，精确归因）。"""
        return self._state.per_subtask.get(subtask_id, 0)

    # ------------------------------------------------------------------
    # 累加与检查
    # ------------------------------------------------------------------
    def observe(self, agent_id: str, usage: Mapping[str, Any]) -> None:
        """累加一次用量并做边界检查；超限只记录事件，不抛异常。

        不抛异常是刻意的：该入口同时被嵌套 Agent 的用量回调复用，
        向上抛会破坏工具结果；真正的熔断由控制层在安全检查点调用
        ``enforce``。
        """
        tokens = int(usage.get("total_tokens") or 0)
        if tokens < 0:
            tokens = 0
        self._state.total_tokens += tokens
        key = agent_id or "unknown"
        self._state.per_agent[key] = self._state.per_agent.get(key, 0) + tokens
        violation = self.check(key)
        if violation is not None:
            self._report(violation)

    def observe_subtask(self, subtask_id: str, usage: Mapping[str, Any]) -> None:
        """把一次用量归因到子任务（与 ``observe`` 并存，仅累加不拦截）。

        子任务级硬预算由调度器在每次尝试前用 ``tokens_for_subtask``
        显式结算并决定是否继续；此处不抛异常，避免破坏工具返回值。
        """
        if not subtask_id:
            return
        tokens = int(usage.get("total_tokens") or 0)
        if tokens < 0:
            tokens = 0
        self._state.per_subtask[subtask_id] = (
            self._state.per_subtask.get(subtask_id, 0) + tokens
        )

    def check(self, agent_id: str | None = None) -> BudgetViolation | None:
        """返回首个超限判定（任务级优先），未启用或未超限返回 None。"""
        if not self._limits.active:
            return None
        total_limit = self._limits.max_total_tokens
        if total_limit > 0 and self._state.total_tokens > total_limit:
            return BudgetViolation(
                scope="task",
                limit=total_limit,
                observed=self._state.total_tokens,
                message=(
                    f"任务 token 用量 {self._state.total_tokens} 超过上限 "
                    f"{total_limit}，已触发预算熔断"
                ),
            )
        agent_limit = self._limits.max_agent_tokens
        if agent_limit <= 0:
            return None
        # 不指定 agent 时扫描全部已归属 Agent：嵌套工具的用量由回调
        # 直接累加，控制层只在顶层检查点统一熔断，不能只看当前标签。
        candidates = (
            [(agent_id, self._state.per_agent.get(agent_id, 0))]
            if agent_id
            else sorted(self._state.per_agent.items())
        )
        for key, observed in candidates:
            if observed > agent_limit:
                return BudgetViolation(
                    scope="agent",
                    agent_id=key,
                    limit=agent_limit,
                    observed=observed,
                    message=(
                        f"Agent {key!r} token 用量 {observed} 超过上限 "
                        f"{agent_limit}，已触发预算熔断"
                    ),
                )
        return None

    def enforce(self, agent_id: str | None = None) -> None:
        """控制层安全检查点：超限时记录事件并抛 ``TokenBudgetExceeded``。"""
        violation = self.check(agent_id)
        if violation is None:
            return
        self._report(violation)
        raise TokenBudgetExceeded(violation)

    def snapshot(self) -> dict[str, Any]:
        """返回当前用量与阈值快照（供审计报告使用）。"""
        limits = self._limits
        return {
            "enabled": limits.enabled,
            "active": limits.active,
            "limits": {
                "max_total_tokens": limits.max_total_tokens,
                "max_agent_tokens": limits.max_agent_tokens,
            },
            "total_tokens": self._state.total_tokens,
            "remaining_tokens": self.remaining_tokens,
            "per_agent": dict(self._state.per_agent),
            "per_subtask": dict(self._state.per_subtask),
            "violations": [dict(item) for item in self._state.violations],
        }

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------
    def _report(self, violation: BudgetViolation) -> None:
        """同一 (scope, agent) 只写一次审计事件，避免刷屏。"""
        key = (violation.scope, violation.agent_id or "")
        if key in self._state.reported:
            return
        self._state.reported.add(key)
        payload: dict[str, Any] = {
            "task_id": self._task_id,
            "agent_id": violation.agent_id or "task",
            "scope": violation.scope,
            "limit": violation.limit,
            "observed": violation.observed,
            "message": violation.message,
        }
        self._state.violations.append(dict(payload))
        if self._events is None:
            return
        try:
            self._events.log_event(EVENT_TOKEN_BUDGET_EXCEEDED, payload)
        except Exception:  # 审计写入失败不得影响熔断语义
            return

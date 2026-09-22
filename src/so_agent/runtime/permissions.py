"""工具授权网关（PermissionGateway）。

职责与行为约定：
- 授权只能由主管 Agent 签发：``issued_by`` 必须等于主管标识，任何来自
  子 Agent 的签发请求一律拒绝——授权不可由子 Agent 转授；
- 授权按任务签发（task_id 绑定）且不自动继承：每次工具调用都必须以
  ``verify_grant(agent_id, tool_name, task_id)`` 显式校验，只有“调用者、
  任务编号、工具名称、授权白名单、有效期”全部匹配才放行；
- 主管专属工具（``assignable=False``，如 code_agent / subagent_creator）
  严禁出现在任何授权白名单中：启用工具注册表时 ``issue_grant`` 直接拒绝；
- ``revoke_grant`` 撤销立即生效；``purge_expired`` 主动清理过期授权；
- 拒绝一切未授权调用：校验不通过返回 False，由调用方（沙箱/工具层）拦截。

全局约束：授权凭证内容为纯数据（Pydantic 模型），可作为 JSON 在 Agent
之间传递与审计，不携带任何可执行内容。
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from so_agent.config import Settings, get_settings
from so_agent.context import ProjectContext
from so_agent.models import ToolGrant
from so_agent.runtime.events import EVENT_TOOL_GRANTED, EVENT_TOOL_REVOKED, EventLogger
from so_agent.runtime.registry import DEFAULT_SUPERVISOR_ID
from so_agent.runtime.tool_registry import ToolRegistry


class PermissionGatewayError(Exception):
    """授权网关领域错误：授权不存在、参数非法等。"""


def _utc_now() -> datetime:
    """返回当前 UTC 时间（统一时间基准）。"""
    return datetime.now(timezone.utc)


def _ensure_aware(moment: datetime) -> datetime:
    """确保 datetime 为 UTC aware；naive 时间按 UTC 处理。"""
    return moment if moment.tzinfo is not None else moment.replace(tzinfo=timezone.utc)


class PermissionGateway:
    """工具授权网关（内存实现）。"""

    def __init__(
        self,
        *,
        context: ProjectContext | None = None,
        settings: Settings | None = None,
        events: EventLogger | None = None,
        supervisor_id: str = DEFAULT_SUPERVISOR_ID,
        tool_registry: ToolRegistry | None = None,
    ) -> None:
        """初始化网关。

        Args:
            context: 可选的共享上下文；提供时授权记录同步镜像到
                ``context.tool_grants``（同一对象引用，撤销状态自动同步）。
            settings: 可选配置（预留授权有效期策略扩展）。
            events: 事件记录器；默认使用与 context 关联的新实例。
            supervisor_id: 主管 Agent 标识，仅该标识可签发授权。
            tool_registry: 可选工具注册表；提供时签发前校验工具可分配性
                （主管专属工具与未登记工具一律拒绝），并支持
                ``get_assignable_tools`` 查询分配候选。
        """
        self._context = context
        self._settings = settings or (
            context.config if context is not None else get_settings()
        )
        self._supervisor_id = supervisor_id
        self._tool_registry = tool_registry
        self._events = events if events is not None else EventLogger(context=context)
        self._grants: dict[str, ToolGrant] = {}

    # ------------------------------------------------------------------
    # 签发与撤销
    # ------------------------------------------------------------------
    def issue_grant(
        self,
        task_id: str,
        agent_id: str,
        allowed_tools: list[str],
        issued_by: str,
        *,
        expires_at: datetime | None = None,
    ) -> ToolGrant:
        """签发工具授权凭证。

        Args:
            task_id: 授权绑定的任务编号（授权不跨任务生效）。
            agent_id: 被授权 Agent。
            allowed_tools: 工具白名单（自动去重、去空、保序）。
            issued_by: 签发者，必须为主管 Agent 标识。
            expires_at: 过期时间；None 表示不过期（naive 按 UTC 处理）。

        Raises:
            PermissionError: 签发者不是主管 Agent（子 Agent 不可转授）时；
                或白名单包含主管专属工具（assignable=False）/ 未登记工具
                （启用工具注册表时生效）时。
            PermissionGatewayError: task_id / agent_id 为空，或白名单为空时。
        """
        if issued_by != self._supervisor_id:
            raise PermissionError(
                f"仅主管 Agent（{self._supervisor_id}）可签发工具授权，"
                f"拒绝 {issued_by!r} 的签发/转授请求"
            )
        if not isinstance(task_id, str) or not task_id.strip():
            raise PermissionGatewayError("task_id 不能为空")
        if not isinstance(agent_id, str) or not agent_id.strip():
            raise PermissionGatewayError("agent_id 不能为空")

        cleaned = list(
            dict.fromkeys(
                tool.strip()
                for tool in allowed_tools
                if isinstance(tool, str) and tool.strip()
            )
        )
        if not cleaned:
            raise PermissionGatewayError("allowed_tools 不能为空（禁止签发空授权）")

        if self._tool_registry is not None:
            unknown: list[str] = []
            exclusive: list[str] = []
            for tool in cleaned:
                entry = self._tool_registry.get_tool(tool)
                if entry is None:
                    unknown.append(tool)
                elif not entry.get("assignable", False):
                    exclusive.append(tool)
            if unknown:
                raise PermissionError(
                    "拒绝签发授权：以下工具未在工具注册表中登记，"
                    "无法确认可分配性：" + "、".join(unknown)
                )
            if exclusive:
                raise PermissionError(
                    "拒绝签发授权：以下工具为主管专属工具（assignable=False），"
                    "严禁分配给子 Agent：" + "、".join(exclusive)
                )

        grant = ToolGrant(
            grant_id=f"grant-{uuid.uuid4().hex[:12]}",
            task_id=task_id,
            agent_id=agent_id,
            allowed_tools=cleaned,
            issued_by=issued_by,
            expires_at=_ensure_aware(expires_at) if expires_at is not None else None,
        )
        self._grants[grant.grant_id] = grant
        if self._context is not None:
            self._context.tool_grants[grant.grant_id] = grant

        self._events.log_event(
            EVENT_TOOL_GRANTED,
            {
                "task_id": task_id,
                "agent_id": agent_id,
                "grant": grant.model_dump(mode="json"),
            },
        )
        return grant

    def revoke_grant(self, grant_id: str) -> None:
        """撤销授权（立即生效；重复撤销为幂等操作）。

        Raises:
            PermissionGatewayError: grant_id 不存在时。
        """
        grant = self._grants.get(grant_id)
        if grant is None:
            raise PermissionGatewayError(f"授权不存在，无法撤销：{grant_id}")
        if grant.revoked:
            return
        grant.revoked = True
        self._events.log_event(
            EVENT_TOOL_REVOKED,
            {
                "task_id": grant.task_id,
                "agent_id": grant.agent_id,
                "grant": grant.model_dump(mode="json"),
                "reason": "revoked",
            },
        )

    def revoke_grants_for_agent(self, agent_id: str) -> list[str]:
        """批量撤销某 Agent 的全部有效授权（释放/替换 Agent 时回收权限）。

        Returns:
            被撤销的 grant_id 列表。
        """
        revoked: list[str] = []
        for grant in list(self._grants.values()):
            if grant.agent_id == agent_id and not grant.revoked:
                self.revoke_grant(grant.grant_id)
                revoked.append(grant.grant_id)
        return revoked

    def purge_expired(self, now: datetime | None = None) -> list[str]:
        """将已过期的授权标记为撤销并记录事件。

        Args:
            now: 参照时间；默认当前 UTC 时间（naive 按 UTC 处理）。

        Returns:
            本次被处理的 grant_id 列表。
        """
        moment = _ensure_aware(now) if now is not None else _utc_now()
        expired: list[str] = []
        for grant in self._grants.values():
            if grant.revoked or grant.expires_at is None:
                continue
            if _ensure_aware(grant.expires_at) <= moment:
                grant.revoked = True
                self._events.log_event(
                    EVENT_TOOL_REVOKED,
                    {
                        "task_id": grant.task_id,
                        "agent_id": grant.agent_id,
                        "grant": grant.model_dump(mode="json"),
                        "reason": "expired",
                    },
                )
                expired.append(grant.grant_id)
        return expired

    # ------------------------------------------------------------------
    # 校验与查询
    # ------------------------------------------------------------------
    def get_assignable_tools(self) -> list[str]:
        """返回当前可分配给子 Agent 的工具名称列表（按字典序）。

        无工具注册表时返回空列表（旧装配方式不提供分配候选）。
        """
        if self._tool_registry is None:
            return []
        return [
            str(item["name"]) for item in self._tool_registry.list_assignable_tools()
        ]

    def verify_grant(self, agent_id: str, tool_name: str, task_id: str) -> bool:
        """校验一次工具调用是否被授权。

        校验维度：调用者（agent_id）、任务编号（task_id）、工具名称
        （tool_name 属于 allowed_tools）、授权有效性（未撤销、未过期）。
        全部满足才返回 True；任一不满足立即拒绝（返回 False）。
        """
        if not all(
            isinstance(value, str) and value.strip()
            for value in (agent_id, tool_name, task_id)
        ):
            return False

        moment = _utc_now()
        for grant in self._grants.values():
            if grant.agent_id != agent_id or grant.task_id != task_id:
                continue
            if grant.revoked or tool_name not in grant.allowed_tools:
                continue
            if grant.expires_at is not None and _ensure_aware(grant.expires_at) <= moment:
                continue
            return True
        return False

    def get_grant(self, grant_id: str) -> ToolGrant | None:
        """按 grant_id 查询授权记录（含已撤销，供审计）。"""
        return self._grants.get(grant_id)

    @property
    def grants(self) -> dict[str, ToolGrant]:
        """返回授权登记表的浅拷贝。"""
        return dict(self._grants)

    @property
    def supervisor_id(self) -> str:
        """返回主管 Agent 标识。"""
        return self._supervisor_id

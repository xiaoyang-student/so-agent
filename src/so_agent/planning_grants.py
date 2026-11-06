"""工具授权候选与真实签发（PlanningCoordinator 的授权能力）。

从 planning_coordinator 拆分（行为不变）：评审用纯候选（proposal 前缀、
不注册网关）与评审通过后的真实签发相互分离。
"""

from __future__ import annotations

import uuid
from datetime import timedelta

from so_agent.models import ToolGrant, _utc_now
from so_agent.runtime.registry import DEFAULT_SUPERVISOR_ID


class PlanningGrantMixin:
    """工具授权候选与真实签发。"""

    def _propose_grants(self) -> list[ToolGrant]:
        """为当前计划生成工具授权候选（纯数据，不注册授权网关）。

        与 ``_issue_grants`` 的差别：候选仅作为两阶段计划评审中授权
        评审的输入（评审需要看到"预计授权工具"），不调用 ``issue_grant``、
        不写入网关登记表；grant_id 统一以 ``proposal-`` 前缀标识——
        即使候选凭证被传入工具包装器，``get_grant`` 也查无登记记录，
        三元组校验与 grant_id 绑定校验都会拦截。真实授权只在计划
        评审与授权评审均通过后由 ``_issue_grants`` 签发。本方法同时
        清空本版本的 ``plan_grants`` / ``plan_grant_map``（被拒计划不
        得残留可供工具包装器使用的有效签发记录）。

        动态工具（未注册 / 未批准）不进候选白名单（与 ``_issue_grants``
        同一资格过滤）：计划阶段仅在子任务 allowed_tools 中作"工具
        缺口提醒"，执行阶段创建成功后再补发真实授权。

        Returns:
            本版本新生成的 ToolGrant 候选列表。
        """
        task_id = self._task_request.task_id
        self._plan_grants = []
        self._plan_grant_map = {}
        proposals: list[ToolGrant] = []
        for spec in self._subtasks:
            eligible = [
                name
                for name in spec.allowed_tools
                if self._tool_registry.is_approved(name)
                and self._tool_registry.is_assignable(name)
            ]
            if not eligible:
                continue
            proposals.append(
                ToolGrant(
                    grant_id=f"proposal-{uuid.uuid4().hex[:12]}",
                    task_id=task_id,
                    agent_id=f"{task_id}:{spec.subtask_id}:grant-slot",
                    allowed_tools=eligible,
                    issued_by=DEFAULT_SUPERVISOR_ID,
                    expires_at=None,
                )
            )
        self._plan_grant_proposals = proposals
        return list(proposals)

    def _issue_grants(self) -> list[ToolGrant]:
        """为当前计划的子任务签发工具授权（无工具的子任务不签发空授权）。

        调用时机：两路计划评审（计划评审 + 授权评审）均通过后才真实
        签发（评审输入只看到 ``_propose_grants`` 生成的纯候选）；执行
        协调器在返工窗口重开（``_ensure_batch_grants_active``）与工具
        缺口解决（``_resolve_tool_gaps``）时经 deps 通道补发。

        只对"已注册且已批准"的可分配工具签发：工具缺口（未注册 / 未批准）
        不进白名单，待执行阶段缺口流程解决后由本方法补发（重新签发时
        覆盖本版本计划的映射，旧授权不主动撤销）。

        复用任务过滤（任务 35 / 31 协调）：复用任务的候选授权已随
        ``_propose_grants`` 参与最小化计划评审（评审可见性保留），
        但其结果已直接并入 ``self._results`` 且不再执行——本方法在
        计划评审通过后、真实签发前跳过它们，避免为已复用任务留下
        无用途的活跃 grant；真实授权只供实际 pending 任务使用。

        每份授权可携带 ``expires_at``（当前时间 + ``grant_ttl_seconds``）
        作为授权生命周期的第二道防线：即使执行协调器的批次后回收（主动
        撤销）因异常路径遗漏，授权也会在 TTL 后自动失效。TTL 与任务
        总时长解耦（任务 80）：``max_pipeline_timeout`` 默认关闭时授权
        仍有独立有效期；``grant_ttl_seconds<=0`` 表示不过期（仍由任务
        退出时统一撤销兜底）。

        授权在规划阶段按"子任务槽位"签发（agent_id 形如
        ``{task_id}:{subtask_id}:grant-slot`` 的挂载点标识）；实际执行
        Agent（原始/替代/动态）只能在该授权范围内获得工具注入，且授权
        不可由子 Agent 转授（issued_by 固定为主管标识）。

        Returns:
            本版本新签发的 ToolGrant 列表。
        """
        task_id = self._task_request.task_id
        self._plan_grants = []
        self._plan_grant_map = {}
        grant_ttl = float(
            getattr(self._settings, "grant_ttl_seconds", 0.0) or 0.0
        )
        expires_at = (
            _utc_now() + timedelta(seconds=grant_ttl)
            if grant_ttl > 0
            else None
        )
        for spec in self._subtasks:
            # 复用任务（结果已并入 _results 且带来源标记）不重新执行：
            # 不为其真实签发授权（候选已随 _propose_grants 参与计划评审，
            # 评审可见性保留）。注意失败待重试任务不在此列：其结果
            # metadata 无来源标记，返工窗口重开与工具缺口补发的签发
            # 不受影响。
            existing = self._results.get(spec.subtask_id)
            if existing is not None and existing.metadata.get("reused_from"):
                continue
            eligible = [
                name
                for name in spec.allowed_tools
                if self._tool_registry.is_approved(name)
                and self._tool_registry.is_assignable(name)
            ]
            if not eligible:
                continue
            grant = self._gateway.issue_grant(
                task_id=task_id,
                agent_id=f"{task_id}:{spec.subtask_id}:grant-slot",
                allowed_tools=eligible,
                issued_by=DEFAULT_SUPERVISOR_ID,
                expires_at=expires_at,
            )
            self._plan_grants.append(grant)
            self._plan_grant_map[spec.subtask_id] = grant
        self._grants.extend(self._plan_grants)
        return list(self._plan_grants)

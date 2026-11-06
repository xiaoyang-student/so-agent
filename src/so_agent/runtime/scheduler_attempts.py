"""子任务尝试执行与确定性失败恢复（SubtaskDAGScheduler 的执行能力）。

从 runtime.scheduler 拆分（行为不变）：单子任务执行（重试 / 契约修复 /
替换 / 分解问题上报）与策略驱动的恢复循环在此实现。调度器的停止门、
事件日志、注册表与会话状态统一经 ``self`` 访问。
"""

from __future__ import annotations

import asyncio
import uuid
from typing import Any, Awaitable, Callable, TypeAlias

from so_agent.models import (
    AgentRecord,
    ExecutionResult,
    ReplacementRequest,
    SubtaskSpec,
)
from so_agent.runtime.events import (
    EVENT_FAILURE_CLASSIFIED,
    EVENT_RECOVERY_DECIDED,
    EVENT_REPLACEMENT_COMPLETED,
    EVENT_REPLACEMENT_EXECUTING,
    EVENT_REPLACEMENT_FAILED,
    EVENT_REPLACEMENT_REQUESTED,
    EVENT_RETRY_SUPPRESSED,
    EVENT_TASK_DISPATCHED,
)
from so_agent.runtime.failure import (
    CONTRACT_FAILURE_TYPES,
    PROVIDER_FAILURE_TYPES,
    FailureType,
    RecoveryAction,
    RecoveryDecision,
    failure_fingerprint,
)
from so_agent.runtime.telemetry import async_span_scope, set_span_attributes

# 执行器签名：接收（子任务规格, 执行 Agent 记录），返回结构化执行结果
TaskExecutor: TypeAlias = Callable[[SubtaskSpec, AgentRecord], Awaitable[ExecutionResult]]

# 替代生命周期通知的失败分类（载荷 failure_category 字段取值）
_REPLACEMENT_FAILED_CATEGORY_CANCELLED: str = "cancelled"
_REPLACEMENT_FAILED_CATEGORY_REGISTRATION: str = "agent_registration_rejected"
_REPLACEMENT_FAILED_CATEGORY_ATTEMPTS: str = "attempts_exhausted"

# 终止通知摘要的错误文本截断上限（防御超长错误信息进入审计载荷）
_FAILURE_SUMMARY_MAX_CHARS: int = 200

# 外层显式总超时的清理宽限（秒）：仅在显式开启 supervisor_total_timeout
# 或 attempt_timeout 时启用，外层 wait_for = 基准 + 本宽限。宽限仅用于
# 让内层阶段空闲超时先行触发并完成取消/审计；默认不设外层总超时。
_ATTEMPT_TIMEOUT_CLEANUP_GRACE: float = 10.0

_ISSUE_RECOMMENDATION = (
    "任务分解可能存在问题：建议主 Agent 重新评估子任务粒度、执行角色分配"
    "与工具授权范围（allowed_tools），必要时拆分该子任务后重规划。"
)

def _task_holds_pending_cancel(task: "asyncio.Task[Any] | None") -> bool:
    """判断 task 是否持有未消费的取消请求（任务 59，Python >=3.10 兼容）。

    ``asyncio.Task.cancelling`` 自 Python 3.11 才存在，项目声明
    ``requires-python >= 3.10``，不得直接假设该 API 存在：

    - Python 3.11+：直接读取 ``task.cancelling() > 0``，保持任务 45
      的严格取消恢复语义不变；
    - Python 3.10（或任何缺少该属性的环境）：返回 False，绝不抛
      ``AttributeError``。依据：3.10 的 ``asyncio.wait_for``（旧实现）
      用内部 task 隔离执行器 coroutine（3.12 起才改为直接展开），外层
      取消在 wait_for 层已被消费或以 CancelledError re-raise，能携带
      业务异常走到本判断的路径与 3.11 上 ``cancelling() == 0`` 的路径
      同类，按业务异常降级为失败结果（不恢复取消）；
    - task 为 None（无运行上下文，如同步调用路径）：返回 False。
    """
    if task is None:
        return False
    cancelling = getattr(task, "cancelling", None)
    if cancelling is None:
        return False
    return cancelling() > 0


class AttemptExecutionMixin:
    """子任务执行：策略驱动恢复、替换与分解问题上报。"""

    async def _run_subtask_guarded(
        self,
        semaphore: asyncio.Semaphore,
        task_id: str,
        spec: SubtaskSpec,
    ) -> ExecutionResult:
        """信号量包裹的子任务执行（限制全局并发度）。"""
        async with semaphore:
            async with async_span_scope(
                "so_agent.subtask",
                {"task.id": task_id, "subtask.id": spec.subtask_id},
            ) as span:
                result = await self._execute_subtask(task_id, spec)
                set_span_attributes(
                    span,
                    {
                        "subtask.success": bool(result.success),
                        "subtask.error_chars": len(result.error or ""),
                    },
                )
                return result

    async def _execute_subtask(self, task_id: str, spec: SubtaskSpec) -> ExecutionResult:
        """执行单个子任务：原始 Agent 重试 → 替换 → 分解问题上报。"""
        max_attempts = max(
            1, min(int(spec.max_attempts), int(self._settings.max_agent_attempts))
        )
        # 子任务真实耗时包络（含原始与替代 Agent 的全部尝试）：到点后
        # 只拒绝新尝试、返回可审计失败，不取消进行中的调用（取消会
        # 绕过 Agent 收尾并泄漏注册表槽位）
        wall_limit = float(
            getattr(self._settings, "subtask_max_wall_seconds", 0.0) or 0.0
        )
        deadline = (
            asyncio.get_running_loop().time() + wall_limit
            if wall_limit > 0
            else None
        )

        # ---- 1) 原始 Agent ----
        primary = self._new_agent_record(
            task_id, spec, can_request_replacement=True, replacement_of=None
        )
        if not self._register_agent(primary):
            result = ExecutionResult(
                subtask_id=spec.subtask_id,
                success=False,
                error=(
                    "创建原始 Agent 失败：动态 Agent 数量已达上限"
                    f"（max_dynamic_agents={self._settings.max_dynamic_agents}）"
                ),
            )
            self._record_decomposition_issue(
                task_id,
                spec,
                failed_agent_ids=[primary.agent_id],
                failure_reasons=[result.error or ""],
                attempted_approaches=["请求注册原始子 Agent 被拒绝"],
                recommendation=_ISSUE_RECOMMENDATION,
            )
            return result
        self._log_agent_created(task_id, primary)

        primary_result, primary_failures, primary_decision = await self._run_attempts(
            task_id, spec, primary, max_attempts, deadline=deadline
        )
        if primary_result.success:
            self._finalize_agent(primary, status="completed")
            return primary_result
        self._finalize_agent(primary, status="failed")

        failure_reasons = [
            f"原始 Agent {primary.agent_id} 尝试 {primary.attempt_count} 次均失败"
        ] + primary_failures

        # ---- 2) 替换请求与替代 Agent ----
        # 恢复策略（任务 46 / 69）：仅当策略判定 REPLACE（语义能力 /
        # 实现类故障停止重试）时才考虑替换；provider 类、不可重试类
        # 与契约类（format）故障禁止替换（format 走受限修复 / 重规划
        # 路径，策略不会对其返回 REPLACE）。停止门（任务 45）：stopping
        # 或剩余预算耗尽时不发起新的替换（_gate_allows 为 False 时落入
        # 下方分解问题上报判定）
        replacement_wanted = (
            primary_decision is not None
            and primary_decision.action is RecoveryAction.REPLACE
        )
        if (
            replacement_wanted
            and self._gate_allows("replacement")
            and self._replacement_allowed(task_id, primary)
        ):
            request = ReplacementRequest(
                request_id=f"repl-{uuid.uuid4().hex[:12]}",
                task_id=task_id,
                subtask_id=spec.subtask_id,
                requester_agent_id=primary.agent_id,
                failure_summary=primary_result.error or "未提供错误信息",
                attempt_evidence=primary_failures
                + list(primary_result.evidence)
                + ([f"error: {primary_result.error}"] if primary_result.error else []),
                requested_tools=list(spec.allowed_tools),
                request_count=self._replacement_counts.get(task_id, 0) + 1,
            )
            self.replacement_requests.append(request)
            self._replacement_counts[task_id] = request.request_count
            primary.replacement_requested = True
            self._events.log_event(
                EVENT_REPLACEMENT_REQUESTED,
                {
                    "task_id": task_id,
                    "agent_id": primary.agent_id,
                    "subtask_id": spec.subtask_id,
                    "request": request.model_dump(mode="json"),
                },
            )
            # 真实时点①（任务 34）：请求已创建、替代 Agent 尚未创建——
            # 上报 replacement_requested（全局 FSM 迁移由接收端串行化，
            # 早于替代 Agent 的创建与执行事件）
            await self._notify_lifecycle(
                EVENT_REPLACEMENT_REQUESTED,
                {
                    "task_id": task_id,
                    "subtask_id": spec.subtask_id,
                    "request_id": request.request_id,
                    "requester_agent_id": primary.agent_id,
                    "request_count": request.request_count,
                    "attempt_count": primary.attempt_count,
                },
            )

            replacement = self._new_agent_record(
                task_id,
                spec,
                can_request_replacement=False,  # 替代 Agent 固定无替换权
                replacement_of=primary.agent_id,
            )
            if self._register_agent(replacement):
                self._log_agent_created(task_id, replacement)
                # 真实时点②（任务 34）：替代 Agent 已创建成功、尚未实际
                # 尝试——上报 replacement_executing（早于替代 Agent 的
                # 首个 task_dispatched 事件）
                await self._notify_lifecycle(
                    EVENT_REPLACEMENT_EXECUTING,
                    {
                        "task_id": task_id,
                        "subtask_id": spec.subtask_id,
                        "request_id": request.request_id,
                        "agent_id": replacement.agent_id,
                    },
                )
                # 替代 Agent 注入原始失败上下文，引导采用不同策略
                replacement_spec = spec.model_copy(deep=True)
                replacement_spec.instructions += (
                    f"\n\n[替代 Agent 上下文] 原始 Agent 已失败 "
                    f"{primary.attempt_count} 次。"
                    f"失败摘要：{primary_result.error}。"
                    f"请采用不同的方法完成任务。"
                )
                # 原始 Agent 即使失败也可能留下部分产出，注入给替代 Agent 参考
                if primary_result.output and len(primary_result.output) > 20:
                    replacement_spec.instructions += (
                        f"\n\n[前任产出参考] 前一位 Agent 的部分成果如下"
                        f"（方向可能正确但未完成/有缺陷），请在此基础上补充和修正，"
                        f"不需要从头重做：\n{primary_result.output[:3000]}\n"
                    )
                try:
                    replacement_result, replacement_failures, _ = (
                        await self._run_attempts(
                            task_id,
                            replacement_spec,
                            replacement,
                            max_attempts,
                            executor=self._replacement_executor,
                            deadline=deadline,
                        )
                    )
                except asyncio.CancelledError:
                    # 真实时点④（终止，任务 34）：替代执行被取消——先上报
                    # 终止通知再继续传播取消；不增加替代次数、无二级替代
                    await self._notify_lifecycle(
                        EVENT_REPLACEMENT_FAILED,
                        {
                            "task_id": task_id,
                            "subtask_id": spec.subtask_id,
                            "request_id": request.request_id,
                            "agent_id": replacement.agent_id,
                            "attempt_count": replacement.attempt_count,
                            "failure_category": (
                                _REPLACEMENT_FAILED_CATEGORY_CANCELLED
                            ),
                            "summary": "替代 Agent 执行被取消",
                        },
                    )
                    raise
                # 真实时点③（任务 34）：替代执行已返回——上报
                # replacement_completed / replacement_failed（含非敏感
                # 结果摘要，不携带业务产出）
                if replacement_result.success:
                    await self._notify_lifecycle(
                        EVENT_REPLACEMENT_COMPLETED,
                        {
                            "task_id": task_id,
                            "subtask_id": spec.subtask_id,
                            "request_id": request.request_id,
                            "agent_id": replacement.agent_id,
                            "attempt_count": replacement.attempt_count,
                            "success": True,
                            "summary": (
                                "替代 Agent 执行成功"
                                f"（尝试 {replacement.attempt_count} 次）"
                            ),
                        },
                    )
                    self._finalize_agent(replacement, status="completed")
                    return replacement_result
                await self._notify_lifecycle(
                    EVENT_REPLACEMENT_FAILED,
                    {
                        "task_id": task_id,
                        "subtask_id": spec.subtask_id,
                        "request_id": request.request_id,
                        "agent_id": replacement.agent_id,
                        "attempt_count": replacement.attempt_count,
                        "failure_category": _REPLACEMENT_FAILED_CATEGORY_ATTEMPTS,
                        "summary": self._failure_summary(replacement_result.error),
                    },
                )
                self._finalize_agent(replacement, status="failed")
                self._record_decomposition_issue(
                    task_id,
                    spec,
                    failed_agent_ids=[primary.agent_id, replacement.agent_id],
                    failure_reasons=failure_reasons
                    + [
                        f"替代 Agent {replacement.agent_id} 尝试 "
                        f"{replacement.attempt_count} 次仍失败"
                    ]
                    + replacement_failures,
                    attempted_approaches=[
                        f"原始 Agent 尝试 {primary.attempt_count} 次",
                        f"替代 Agent 尝试 {replacement.attempt_count} 次（can_request_replacement=False）",
                    ],
                    recommendation=_ISSUE_RECOMMENDATION,
                )
                return replacement_result

            # 替代 Agent 注册被拒绝（动态 Agent 上限）
            # 真实时点④（终止，任务 34）：替代 Agent 创建失败——上报终止
            # 通知后再上报分解问题（不增加替代次数、无二级替代）
            await self._notify_lifecycle(
                EVENT_REPLACEMENT_FAILED,
                {
                    "task_id": task_id,
                    "subtask_id": spec.subtask_id,
                    "request_id": request.request_id,
                    "agent_id": replacement.agent_id,
                    "attempt_count": 0,
                    "failure_category": _REPLACEMENT_FAILED_CATEGORY_REGISTRATION,
                    "summary": "替代 Agent 注册被拒绝（动态 Agent 数量已达上限）",
                },
            )
            self._record_decomposition_issue(
                task_id,
                spec,
                failed_agent_ids=[primary.agent_id],
                failure_reasons=failure_reasons
                + ["替代 Agent 注册被拒绝（动态 Agent 数量已达上限）"],
                attempted_approaches=[f"原始 Agent 尝试 {primary.attempt_count} 次"],
                recommendation=_ISSUE_RECOMMENDATION,
            )
            return primary_result

        # ---- 3) 不上报替换：策略禁替换或额度/资格不足，上报分解问题 ----
        # 停止门（任务 45）：stopping 或剩余预算耗尽时不新增分解问题
        #（避免触发下游重规划）；失败结果已落账，可审计
        if replacement_wanted:
            # 策略要求替换但被额度/资格拦下：保留既有机器可读措辞
            blocked_reason = "替换额度已用尽或该 Agent 不具备替换资格"
        elif (
            primary_decision is not None
            and primary_decision.failure_type in PROVIDER_FAILURE_TYPES
        ):
            blocked_reason = (
                f"provider 类故障禁止替换（{primary_decision.failure_type.value}）："
                f"{primary_decision.reason}"
            )
        elif (
            primary_decision is not None
            and primary_decision.failure_type in CONTRACT_FAILURE_TYPES
        ):
            # 契约类（format）禁替换（任务 69）：修复未成即转重规划
            blocked_reason = (
                f"输出契约失败禁止替换（{primary_decision.failure_type.value}）："
                f"{primary_decision.reason}"
            )
        elif primary_decision is not None:
            blocked_reason = (
                f"不可恢复类故障禁止替换（{primary_decision.failure_type.value}）："
                f"{primary_decision.reason}"
            )
        else:
            # 预算耗尽早退路径（无恢复判定）：仅可审计，不再发起新动作
            blocked_reason = "pipeline 剩余预算不足，停止恢复路径"
        if self._gate_allows("escalate_issue"):
            self._record_decomposition_issue(
                task_id,
                spec,
                failed_agent_ids=[primary.agent_id],
                failure_reasons=failure_reasons + [blocked_reason],
                attempted_approaches=[f"原始 Agent 尝试 {primary.attempt_count} 次"],
                recommendation=_ISSUE_RECOMMENDATION,
            )
        return primary_result

    async def _run_attempts(
        self,
        task_id: str,
        spec: SubtaskSpec,
        agent: AgentRecord,
        max_attempts: int,
        executor: TaskExecutor | None = None,
        deadline: float | None = None,
    ) -> tuple[ExecutionResult, list[str], RecoveryDecision | None]:
        """对单个 Agent 执行策略驱动的确定性恢复（任务 69 修订）。

        每次失败经 runtime.failure 做确定性分类与稳定指纹去重，由恢复
        策略决定：立即重试 / 退避重试 / 受限契约修复 / 终止（替换 /
        重规划 / 停止）。失败分类、恢复判定与同因抑制均写审计事件；
        核心不变量——同指纹连续失败达到阈值即抑制重试，unknown 异常
        保守不重试，provider 类禁止替换（由 _execute_subtask 接手）。

        契约修复与业务重试的边界（任务 69）：

        - 业务重试（RETRY）：消耗业务尝试额度（attempt 计数，上限
          ``max_attempts``），注入失败上下文与上次部分产出引导调整
          策略；
        - 契约修复（REPAIR_CONTRACT）：独立额度（``repair_used``，上限
          ``recovery_repair_contract_max_attempts``），不消耗业务尝试
          额度、不推进 attempt 计数；修复轮仅注入受限修复指令（基于
          上次已有产出仅修正输出契约，明确禁止重跑业务任务），
          dispatch 事件以 ``contract_repair=True`` 审计标注——绝不
          静默占用普通业务 attempt，也不重跑完整业务任务冒充修复；
        - format 类失败首次仅允许修复；同一格式失败再现（同因抑制）
          或修复额度耗尽即转重规划（REPLAN），绝不发起替换。

        循环收敛性：业务轮每轮 attempt+1 且 attempt ≥ max_attempts 时
        策略必返回终止动作（REPLACE / REPLAN）；修复轮每轮 repair_used+1
        且额度尽时 FORMAT 必转 REPLAN——两类额度均单调递增，必然终止。

        分类信任边界（任务 69）：执行器正常返回的结果按低信任自由
        文本分类（classify_result，仅可判定 format / tool / sandbox /
        quality）；控制层捕获的异常按高信任证据分类（classify_exception：
        异常类名 / HTTP 状态码）。network / rate_limit / timeout /
        permission / invalid_input / budget / internal 等系统级类型只能
        由后者决定，子 Agent 自述关键词不得伪装改变恢复方向。

        Returns:
            (最后一次 ExecutionResult, 每次失败摘要列表, 终止性恢复判定)。
            判定为 None 表示成功返回或预算耗尽早退（无恢复动作）。
        """
        run = executor or self._executor
        timeout = self._resolve_attempt_timeout()
        last: ExecutionResult | None = None
        failures: list[str] = []
        decision: RecoveryDecision | None = None
        scope = (task_id, spec.subtask_id, agent.agent_id)
        repair_used = 0
        attempt = 0
        contract_repair_round = False

        while True:
            # 停止门（任务 45）：stopping 时拒绝重试（穿透宽捕获）；
            # 剩余预算耗尽（remaining <= 0，永不返回负值）时不调用执行器、
            # 不发 dispatch 事件，返回可审计的预算失败结果（契约修复轮
            # 同样受门与预算约束，不享受豁免）
            self._gate_check("retry")
            gate_remaining = self._gate_remaining()
            if gate_remaining is not None and gate_remaining <= 0.0:
                failures.append(
                    f"pipeline 剩余预算不足（剩余 {gate_remaining:g}s），"
                    f"第 {attempt + 1} 次尝试未发起"
                )
                return (
                    ExecutionResult(
                        subtask_id=spec.subtask_id,
                        success=False,
                        error=(
                            f"pipeline 剩余预算不足（剩余 {gate_remaining:g}s），"
                            f"不再发起执行"
                        ),
                    ),
                    failures,
                    None,
                )
            # 子任务硬预算（token）：跨尝试累计；超限后不再发起任何新
            # 尝试（含契约修复轮），以不可重试预算失败收敛，由上层重规划
            violation = self._subtask_budget_violation(spec.subtask_id)
            if violation:
                return self._subtask_stop(
                    task_id=task_id,
                    spec=spec,
                    agent=agent,
                    failure_type=FailureType.BUDGET,
                    reason=violation,
                    failures=failures,
                    attempt=attempt,
                )
            # 子任务硬预算（真实耗时）：包络耗尽同样停止新尝试
            wall_remaining: float | None = None
            if deadline is not None:
                wall_remaining = deadline - asyncio.get_running_loop().time()
                if wall_remaining <= 0.0:
                    return self._subtask_stop(
                        task_id=task_id,
                        spec=spec,
                        agent=agent,
                        failure_type=FailureType.TIMEOUT,
                        reason=(
                            "子任务真实耗时已达上限"
                            f"（{float(self._settings.subtask_max_wall_seconds):g}s），"
                            "不再发起新尝试"
                        ),
                        failures=failures,
                        attempt=attempt,
                    )
            if not contract_repair_round:
                attempt += 1  # 业务尝试计数（契约修复轮不推进，任务 69）
            is_repair_round = contract_repair_round
            contract_repair_round = False
            # 单次尝试窗口不超过 pipeline 剩余预算：只在显式开启绝对
            # deadline 时截断；未开启时 attempt_timeout=None 表示不设
            # 外层总超时（由内层空闲窗口检测卡死）
            attempt_timeout = timeout
            if gate_remaining is not None:
                if attempt_timeout is None:
                    attempt_timeout = max(1.0, gate_remaining)
                else:
                    attempt_timeout = max(
                        1.0, min(attempt_timeout, gate_remaining)
                    )
            if wall_remaining is not None:
                wall_cap = max(0.5, wall_remaining)
                attempt_timeout = (
                    wall_cap
                    if attempt_timeout is None
                    else max(0.5, min(attempt_timeout, wall_cap))
                )

            agent.attempt_count = attempt
            agent.status = "running"
            self._events.log_event(
                EVENT_TASK_DISPATCHED,
                {
                    "task_id": task_id,
                    "subtask_id": spec.subtask_id,
                    "agent_id": agent.agent_id,
                    "attempt": attempt,
                    "max_attempts": max_attempts,
                    # 契约修复轮审计标注（任务 69）：同 attempt 序号下区分
                    # 业务尝试与受限修复轮；新增字段，事件结构向后兼容
                    "contract_repair": is_repair_round,
                },
            )
            # 非首轮执行注入提示；两类提示严格分离（任务 69）：
            # - 业务重试轮：失败上下文 + 上次部分产出，引导调整策略；
            # - 契约修复轮：受限修复指令——基于已有产出仅修正输出契约，
            #   明确禁止重跑完整业务任务（不混合业务重试提示）
            effective_spec = spec
            if failures:
                effective_spec = spec.model_copy(deep=True)
                partial = last.output if last and last.output else ""
                if is_repair_round:
                    repair_hint = (
                        "\n\n[契约修复] 上次输出未满足约定的输出契约 / 格式。"
                        "本次为受限格式修复：不重新执行业务任务，基于上次的"
                        "已有产出，仅把输出修正为严格符合契约要求的字段与"
                        "结构，不得省略、改写或包裹多余内容。\n"
                    )
                    if partial and len(partial) > 20:
                        repair_hint += (
                            f"[待修复产出] 上次尝试的产出如下，请在此基础上"
                            f"仅做格式修正后重新返回：\n{partial[:2000]}\n"
                        )
                    effective_spec.instructions += repair_hint
                else:
                    retry_hint = (
                        f"\n\n[重试提示] 第 {attempt - 1} 次尝试失败，"
                        f"原因：{failures[-1]}。\n"
                        f"请调整策略避免重复相同错误。\n"
                    )
                    if partial and len(partial) > 20:
                        retry_hint += (
                            f"[已有部分产出] 以下是上次尝试的部分成果，"
                            f"如果方向正确请在此基础上补充完善，"
                            f"不需要从头重做：\n{partial[:2000]}\n"
                        )
                    effective_spec.instructions += retry_hint

            exc: BaseException | None = None
            pinned_type: FailureType | None = None
            try:
                if attempt_timeout is None:
                    outcome = await run(effective_spec, agent)
                else:
                    outcome = await asyncio.wait_for(
                        run(effective_spec, agent), timeout=attempt_timeout
                    )
            except asyncio.TimeoutError as err:
                exc = err
                # 内层（阶段空闲窗口 / 显式总预算）超时先于外层 wait_for
                # 触发时，异常消息携带 phase 证据——保留类型与消息供
                # 审计与 "TimeoutError" 字样检索（test_workflow_timeout）
                detail = (
                    f"{type(err).__name__}: {err}" if str(err) else type(err).__name__
                )
                timeout_text = (
                    f">{attempt_timeout:g}s，已终止"
                    if attempt_timeout is not None
                    else "（内层空闲窗口触发）"
                )
                outcome = ExecutionResult(
                    subtask_id=spec.subtask_id,
                    success=False,
                    error=f"单次执行超时（{timeout_text}，{detail}）",
                    duration=float(attempt_timeout or 0.0),
                )
            except asyncio.CancelledError:
                raise
            except Exception as err:
                # 取消防御（任务 45，Python 3.12+ 语义）：wait_for 对
                # coroutine 直接展开，外层取消会直接注入执行器内部；若
                # 执行器捕获 CancelledError 后转换为普通异常（如清理期
                # 抛业务异常），这里必须恢复取消语义向上传播（本任务仍
                # 持有未消费的取消请求 cancelling() > 0），绝不能降级为
                # 失败结果触发重试（否则停止/取消后仍会发起新尝试）。
                # cancelling 读取经 _task_holds_pending_cancel 兼容封装
                # （任务 59）：Python 3.10 无 Task.cancelling 时不抛
                # AttributeError，语义见该函数 docstring。
                if _task_holds_pending_cancel(asyncio.current_task()):
                    raise asyncio.CancelledError from err
                exc = err
                outcome = ExecutionResult(
                    subtask_id=spec.subtask_id,
                    success=False,
                    error=f"执行异常：{type(err).__name__}: {err}",
                )

            if not isinstance(outcome, ExecutionResult):
                # 执行器契约违规（实现类缺陷）：固定为 INTERNAL（不可重试）
                pinned_type = FailureType.INTERNAL
                outcome = ExecutionResult(
                    subtask_id=spec.subtask_id,
                    success=False,
                    error=(
                        "执行器返回非法结果："
                        f"{type(outcome).__name__}（要求 ExecutionResult）"
                    ),
                )
            last = outcome
            if outcome.success:
                return outcome, failures, None

            agent.status = "waiting"
            error_text = outcome.error or ""
            failures.append(
                f"第 {attempt} 次尝试失败"
                + ("（契约修复轮）" if is_repair_round else "")
                + f"：{outcome.error or '未提供错误信息'}"
            )

            # 确定性失败恢复：分类 → 稳定指纹 → 同因抑制 → 策略判定。
            # 信任边界（任务 69）：异常路径（exc）是控制层高信任证据，
            # 可判定系统级类型（provider / permission / budget /
            # internal）；执行器正常返回的结果文本是 Agent 侧低信任
            # 自由文本，只能命中 format / tool / sandbox / quality
            if pinned_type is not None:
                failure_type = pinned_type
            elif exc is not None:
                failure_type = self._classifier.classify_exception(exc)
            else:
                failure_type = self._classifier.classify_result(outcome)
            fingerprint = failure_fingerprint(failure_type, error_text)
            occurrences = self._deduplicator.record(scope, fingerprint)
            suppressed = self._deduplicator.suppressed(occurrences)
            decision = self._policy.decide(
                failure_type=failure_type,
                fingerprint=fingerprint,
                occurrences=occurrences,
                suppressed=suppressed,
                attempt=attempt,
                max_attempts=max_attempts,
                repair_used=repair_used,
            )
            self._emit_failure_recovery_events(
                task_id=task_id,
                spec=spec,
                agent=agent,
                attempt=attempt,
                failure_type=failure_type,
                fingerprint=fingerprint,
                occurrences=occurrences,
                suppressed=suppressed,
                decision=decision,
                error_text=error_text,
            )
            # Execution 记录语义（models.Execution.failure_type /
            # recovery_action 的值域）：权威判定写回失败结果元数据，
            # 无条件覆盖执行器或 LLM 自述的同名字段（判定不受模型输出
            # 影响；自述字段仅供审计对照，不得成为权威值）
            outcome.metadata["failure_type"] = failure_type.value
            outcome.metadata["recovery_action"] = decision.action.value

            if decision.action is RecoveryAction.RETRY:
                if decision.backoff_seconds > 0.0:
                    await self._sleep_backoff(decision.backoff_seconds)
                continue
            if decision.action is RecoveryAction.REPAIR_CONTRACT:
                # 受限契约修复轮（任务 69）：独立额度计数（repair_used
                # 单调递增保证收敛），不占业务尝试额度（attempt 不推进），
                # 下一轮仅注入修复指令（不混合业务重试提示）
                repair_used += 1
                contract_repair_round = True
                continue
            break  # REPLACE / REPLAN / STOP：交由 _execute_subtask 接手

        assert last is not None  # max_attempts >= 1 保证至少执行一次
        return last, failures, decision

    def _emit_failure_recovery_events(
        self,
        *,
        task_id: str,
        spec: SubtaskSpec,
        agent: AgentRecord,
        attempt: int,
        failure_type: FailureType,
        fingerprint: str,
        occurrences: int,
        suppressed: bool,
        decision: RecoveryDecision,
        error_text: str,
    ) -> None:
        """写入失败恢复审计事件（分类 / 判定 / 抑制；仅控制信息）。"""
        prefix: dict[str, Any] = {
            "task_id": task_id,
            "agent_id": agent.agent_id,
            "subtask_id": spec.subtask_id,
            "attempt": attempt,
            "failure_type": failure_type.value,
            "fingerprint": fingerprint,
            "occurrences": occurrences,
        }
        self._events.log_event(
            EVENT_FAILURE_CLASSIFIED,
            {
                **prefix,
                "error_summary": error_text[:_FAILURE_SUMMARY_MAX_CHARS],
            },
        )
        self._events.log_event(
            EVENT_RECOVERY_DECIDED,
            {
                **prefix,
                "action": decision.action.value,
                "reason": decision.reason,
                "backoff_seconds": round(decision.backoff_seconds, 4),
            },
        )
        if suppressed:
            self._events.log_event(
                EVENT_RETRY_SUPPRESSED,
                {
                    **prefix,
                    "threshold": self._deduplicator.threshold,
                },
            )

    def _subtask_stop(
        self,
        *,
        task_id: str,
        spec: SubtaskSpec,
        agent: AgentRecord,
        failure_type: FailureType,
        reason: str,
        failures: list[str],
        attempt: int,
    ) -> tuple[ExecutionResult, list[str], RecoveryDecision]:
        """构造子任务硬预算终止：不可重试失败 + STOP 判定（含审计事件）。

        预算 / 真实耗时超限是资源性终止：重试只会继续消耗，因此判定
        固定为 STOP，交由上层重规划改变策略，而不是在同一执行器上重跑。
        """
        fingerprint = failure_fingerprint(failure_type, reason)
        decision = RecoveryDecision(
            action=RecoveryAction.STOP,
            failure_type=failure_type,
            fingerprint=fingerprint,
            occurrences=1,
            suppressed=False,
            reason=reason,
        )
        self._emit_failure_recovery_events(
            task_id=task_id,
            spec=spec,
            agent=agent,
            attempt=attempt,
            failure_type=failure_type,
            fingerprint=fingerprint,
            occurrences=1,
            suppressed=False,
            decision=decision,
            error_text=reason,
        )
        failures.append(f"子任务硬预算终止：{reason}")
        result = ExecutionResult(
            subtask_id=spec.subtask_id,
            success=False,
            error=reason,
            metadata={
                "failure_type": failure_type.value,
                "recovery_action": decision.action.value,
            },
        )
        return result, failures, decision

    async def _sleep_backoff(self, seconds: float) -> None:
        """退避休眠：受 pipeline 剩余预算截断（绝不睡过 deadline）；
        sleeper 可注入以获得确定性测试行为。"""
        delay = max(0.0, float(seconds))
        remaining = self._gate_remaining()
        if remaining is not None:
            delay = max(0.0, min(delay, remaining))
        await self._sleeper(delay)

    def _resolve_attempt_timeout(self) -> float | None:
        """单次执行的外层兜底超时；None 表示不设总超时。

        默认不设外层总超时（任务 80）：合法长调用由内层
        ``run_agent_streamed`` 的阶段空闲窗口检测断流/卡死，而不是被
        固定总时长终止。仅在显式配置 ``attempt_timeout`` 或
        ``supervisor_total_timeout>0`` 时启用外层 wait_for 硬兜底，
        并额外留出清理宽限，确保内层先触发取消与审计。

        Returns:
            正的超时秒数；无限制时返回 None。
        """
        if self._attempt_timeout is not None:
            value = float(self._attempt_timeout)
            return value if value > 0 else None
        base = float(self._settings.supervisor_total_timeout)
        if base <= 0:
            return None
        return base + _ATTEMPT_TIMEOUT_CLEANUP_GRACE

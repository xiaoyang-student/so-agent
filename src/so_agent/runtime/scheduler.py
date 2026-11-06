"""并发调度器（SubtaskDAGScheduler）。

职责与行为约定：
- 依据 SubtaskSpec.dependencies 构建依赖图（DAG），做完整性校验（缺失依赖、
  自依赖、重复 ID、循环依赖一律拒绝）后按拓扑序调度；
- 互不依赖的子任务并发执行，并发度由 ``max_concurrency`` 限制
  （asyncio.Semaphore 实现）；依赖失败的子任务直接标记跳过（success=False），
  不进入执行队列，并沿依赖边传播；
- 确定性失败恢复（任务 46）：每次失败经 runtime.failure 做确定性分类 /
  稳定指纹去重 / 策略判定——同指纹连续失败达到阈值即抑制重试（确定性
  去重，防无限重试），network / rate_limit 独立有限指数退避（休眠与
  抖动源可注入），失败分类 / 恢复判定 / 抑制均写审计事件
  （failure_classified / recovery_decided / retry_suppressed）；
- 单个子 Agent 的业务尝试上限为 ``max_agent_attempts`` 与
  ``SubtaskSpec.max_attempts`` 的较小者（attempt_count 跟踪），真实
  尝试次数由恢复策略决定（不超过上限）；单次执行超时按失败处理；
- replacement 仅用于语义能力 / 实现类故障（quality / tool /
  sandbox，任务 69 起 format 不在替换集合内）；provider 类故障
  （network / rate_limit / timeout）与不可重试类故障禁止替换，直接
  上报分解问题。format 类输出契约失败同样禁止替换（任务 69）：首次
  仅允许受限契约修复轮（不占业务尝试额度、不重跑完整业务任务），
  同一格式失败再现或修复额度耗尽即停止重试并转重规划，绝不创建
  Replacement Agent。原始子 Agent 停止重试且策略要求替换时，可发起
  一次 ReplacementRequest（受 ``max_replacement_requests`` 全局限额
  约束）；替代 Agent 的 ``can_request_replacement`` 固定为 False；
  替代 Agent 仍失败则生成 TaskDecompositionIssue，供编排器触发
  重规划；
- 分类信任边界（任务 69）：执行器正常返回的结果按低信任自由文本
  分类（classify_result，仅可判定 format / tool / sandbox /
  quality）；控制层捕获的异常按高信任证据分类（classify_exception：
  异常类名 / HTTP 状态码）。network / rate_limit / timeout /
  permission / invalid_input / budget / internal 等系统级类型只能
  由控制层证据决定，子 Agent 自述关键词不得伪装改变恢复方向；
- 替代生命周期（任务 34）：在真实时点经受控异步回调
  （``lifecycle_notifier``，由编排器注入执行协调器的接收端）上报
  replacement_requested（请求创建后、替代 Agent 创建前）/
  replacement_executing（替代 Agent 创建成功且实际尝试前）/
  replacement_completed | replacement_failed（替代执行返回后，含非
  敏感结果摘要与失败分类）；回调载荷仅含 subtask_id / request_id /
  agent_id 等控制信息，不携带任何业务输出；全局 FSM 迁移由接收端
  （执行协调器）以 asyncio.Lock 串行化，回调失败不影响调度；
- 全程事件留痕（task_dispatched / agent_created / agent_released /
  replacement_requested / decomposition_issue），经 EventLogger 写入。

能力分组：依赖图（scheduler_dag.DAGMixin）与尝试执行 / 确定性恢复
（scheduler_attempts.AttemptExecutionMixin）位于独立模块；本模块保留
调度入口、停止门、替代生命周期通知与 Agent 记录等编排状态。

调用示例::

    scheduler = SubtaskDAGScheduler(executor=my_executor, registry=registry)
    results = await scheduler.schedule(task_id, subtasks)

executor 签名为 ``async def (subtask, agent_record) -> ExecutionResult``；
调度期间可读取 ``scheduler.results`` 获取已完成子任务的（含依赖的）输出。
"""

from __future__ import annotations

import asyncio
import random
import uuid
from typing import Any, Awaitable, Callable, TypeAlias

from so_agent.config import Settings, get_settings
from so_agent.context import ProjectContext
from so_agent.models import (
    AgentRecord,
    ExecutionResult,
    ReplacementRequest,
    SubtaskSpec,
    TaskDecompositionIssue,
)
from so_agent.runtime.events import (
    EVENT_AGENT_CREATED,
    EVENT_AGENT_RELEASED,
    EVENT_DECOMPOSITION_ISSUE,
    EVENT_RUNTIME_BREAKER_TRIPPED,
    EventLogger,
)
from so_agent.runtime.parallelism import (
    RuntimeBreaker,
    collapse_ready,
    record_breaker_result,
)
from so_agent.runtime.registry import AgentRegistry, AgentRegistryError
from so_agent.runtime.failure import (
    FailureClassifier,
    FailureDeduplicator,
    RecoveryPolicy,
)
# _task_holds_pending_cancel / _ATTEMPT_TIMEOUT_CLEANUP_GRACE 为历史
# 导入路径的 re-export（测试直接引用），实现位于尝试执行模块。
from so_agent.runtime.scheduler_attempts import (
    _ATTEMPT_TIMEOUT_CLEANUP_GRACE,
    _FAILURE_SUMMARY_MAX_CHARS,
    AttemptExecutionMixin,
    TaskExecutor,
    _task_holds_pending_cancel,
)
from so_agent.runtime.scheduler_dag import DAGMixin, SchedulerError
from so_agent.state_machine import PipelineStoppedError

# 替代生命周期通知签名：（事件名, 控制信息载荷）→ None。事件名与
# runtime.events 中的替代事件类型常量一致；载荷仅含控制信息（子任务
# 编号、申请编号、Agent 编号、尝试次数、失败分类与非敏感摘要），
# 不携带任何 Agent 的业务产出。由编排器注入执行协调器的受控接收端。
ReplacementLifecycleNotifier: TypeAlias = Callable[[str, dict[str, Any]], Awaitable[None]]

# 子任务 role → AgentRecord.agent_type 映射
_ROLE_TYPE_MAP: dict[str, str] = {
    "code": "code_agent",
    "code_agent": "code_agent",
    "review": "review_agent",
    "review_agent": "review_agent",
    "creator": "subagent_creator",
    "subagent_creator": "subagent_creator",
}


class SubtaskDAGScheduler(AttemptExecutionMixin, DAGMixin):
    """子任务 DAG 调度器（asyncio 并发实现）。"""

    def __init__(
        self,
        executor: TaskExecutor,
        *,
        replacement_executor: TaskExecutor | None = None,
        context: ProjectContext | None = None,
        settings: Settings | None = None,
        registry: AgentRegistry | None = None,
        events: EventLogger | None = None,
        attempt_timeout: float | None = None,
        lifecycle_notifier: ReplacementLifecycleNotifier | None = None,
        gate: Any = None,
        budget_check: Callable[[], None] | None = None,
        sleeper: Callable[[float], Awaitable[None]] | None = None,
        jitter: Callable[[], float] | None = None,
    ) -> None:
        """初始化调度器。

        Args:
            executor: 子任务执行器（异步可调用，返回 ExecutionResult）。
            replacement_executor: 替代 Agent 的执行器；默认复用 executor。
            context: 可选共享上下文（配置与事件日志来源）。
            settings: 可选配置；优先级为 显式参数 > context.config > 全局配置。
            registry: 可选 Agent 注册表；提供时调度过程创建的 Agent 会被
                登记/释放，并受 ``max_dynamic_agents`` 治理；未提供时仅做
                本地记录。
            events: 事件记录器；默认使用与 context 关联的新实例。
            attempt_timeout: 单次执行的外层兜底超时（秒）；None 且
                ``supervisor_total_timeout<=0``（默认）表示不设外层总
                超时，由内层阶段空闲窗口检测卡死；显式正数时外层
                ``wait_for`` 比内层多出清理宽限，确保内层先触发取消与
                审计。
            lifecycle_notifier: 可选的替代生命周期异步回调（任务 34）；
                由编排器注入执行协调器的受控接收端，在真实时点接收
                replacement_requested / replacement_executing /
                replacement_completed | replacement_failed 通知。未提供时
                调度器照常运行（仅无生命周期上报）。也可在构造后经
                :meth:`set_lifecycle_notifier` 注入。
            gate: 可选的 pipeline 停止门（任务 45，鸭子类型避免循环
                导入）：需要提供 ``stopping`` / ``remaining()`` /
                ``check(action)`` 接口；dispatch / retry / replacement
                入口先经门检查，停止后拒绝一切新动作。也可在构造后
                经 :meth:`set_gate` 注入。
            budget_check: 可选的同步预算检查回调；每轮派发前调用，
                超限时抛出 ``TokenBudgetExceeded``（BaseException）终止
                调度。也可在构造后经 :meth:`set_budget_check` 注入。
            sleeper: 可注入的异步休眠函数（``async (seconds) -> None``）；
                默认 ``asyncio.sleep``；仅用于 network / rate_limit 退避，
                测试可注入确定性替身。
            jitter: 可注入的退避抖动采样函数（返回 0~1）；默认
                ``random.random``；测试可注入固定值以获得确定性延迟。
        """
        self._executor = executor
        self._replacement_executor = replacement_executor or executor
        self._context = context
        self._settings = settings or (
            context.config if context is not None else get_settings()
        )
        self._registry = registry
        self._events = events if events is not None else EventLogger(context=context)
        self._attempt_timeout = attempt_timeout
        self._lifecycle_notifier = lifecycle_notifier
        self._gate = gate
        self._budget_check = budget_check
        self._subtask_budget_check: Callable[[str], str | None] | None = None
        # 确定性失败恢复内核（任务 46）：分类器 / 同因去重器 / 策略矩阵；
        # 参数统一取自 Settings.recovery_*（环境变量可覆盖），判定不受
        # LLM 输出影响；退避的休眠与抖动经显式注入以便确定性测试
        self._classifier = FailureClassifier()
        self._deduplicator = FailureDeduplicator(
            threshold=int(self._settings.recovery_retry_suppress_threshold)
        )
        self._policy = RecoveryPolicy(
            repair_contract_max_attempts=int(
                self._settings.recovery_repair_contract_max_attempts
            ),
            backoff_base_seconds=float(
                self._settings.recovery_backoff_base_seconds
            ),
            backoff_cap_seconds=float(
                self._settings.recovery_backoff_cap_seconds
            ),
            backoff_jitter_ratio=float(
                self._settings.recovery_backoff_jitter_ratio
            ),
            jitter=jitter if jitter is not None else random.random,
        )
        self._sleeper = sleeper if sleeper is not None else asyncio.sleep

        # 调度过程状态（每次 schedule 调用刷新 results；请求与问题累计保留）
        self.results: dict[str, ExecutionResult] = {}
        self.replacement_requests: list[ReplacementRequest] = []
        self.decomposition_issues: list[TaskDecompositionIssue] = []
        self._replacement_counts: dict[str, int] = {}  # task_id → 已发起的替换请求数
        # 子任务级进展回调（执行协调器据此重置看门狗并落检查点）
        self._progress_notifier: Callable[[str, ExecutionResult], None] | None = None

    # ------------------------------------------------------------------
    # 替代生命周期通知（受控回调；任务 34）
    # ------------------------------------------------------------------
    def set_lifecycle_notifier(self, notifier: ReplacementLifecycleNotifier) -> None:
        """注入（或替换）替代生命周期通知回调（由编排器装配时调用）。"""
        self._lifecycle_notifier = notifier

    def set_gate(self, gate: Any) -> None:
        """注入（或替换）pipeline 停止门（由编排器装配时调用）。"""
        self._gate = gate

    def set_budget_check(self, check: Callable[[], None]) -> None:
        """注入（或替换）token 预算检查回调（由编排器装配时调用）。"""
        self._budget_check = check

    def set_progress_notifier(
        self, notifier: Callable[[str, ExecutionResult], None]
    ) -> None:
        """注入子任务级进展回调（由编排器装配时调用）。"""
        self._progress_notifier = notifier

    def set_subtask_budget_check(
        self, check: Callable[[str], str | None]
    ) -> None:
        """注入子任务硬预算检查回调（每次尝试前结算，返回违规原因）。

        回调返回非空字符串表示该子任务累计用量已超出硬预算，调度器
        立即以不可重试的预算失败收敛（不再发起新尝试），由上层转入
        重规划。回调异常不得中断调度（按无违规处理）。
        """
        self._subtask_budget_check = check

    def _subtask_budget_violation(self, subtask_id: str) -> str | None:
        """结算子任务累计用量；无回调或回调异常时视为未超限。"""
        check = self._subtask_budget_check
        if check is None:
            return None
        try:
            reason = check(subtask_id)
        except Exception:
            return None
        if not reason:
            return None
        return str(reason)

    def _notify_progress(self, subtask_id: str, result: ExecutionResult) -> None:
        """上报一次子任务进展；回调异常不得影响调度主流程。"""
        notifier = self._progress_notifier
        if notifier is None:
            return
        try:
            notifier(subtask_id, result)
        except Exception:
            return

    def _gate_check(self, action: str) -> None:
        """停止门检查点：已请求停止时抛出穿透宽捕获的停止信号。

        Raises:
            PipelineStoppedError: 停止门已请求停止时（BaseException）。
        """
        gate = self._gate
        if gate is not None:
            gate.check(action)

    def _gate_remaining(self) -> float | None:
        """返回 pipeline 剩余预算（秒，永不为负）；无门时为 None。"""
        gate = self._gate
        if gate is None:
            return None
        return gate.remaining()

    def _gate_allows(self, action: str) -> bool:
        """停止门是否允许新动作：stopping 或剩余预算耗尽均不允许。"""
        gate = self._gate
        if gate is None:
            return True
        if gate.stopping:
            return False
        remaining = gate.remaining()
        return remaining is None or remaining > 0.0

    async def _notify_lifecycle(self, event: str, payload: dict[str, Any]) -> None:
        """在真实时点上报替代生命周期通知（受控回调，失败不阻断调度）。

        回调由编排器注入（执行协调器接收端以 asyncio.Lock 串行化全局
        FSM 迁移与事件写入）；本方法只传递控制信息载荷，并在回调异常
        时静默返回（取消除外，必须继续传播）。
        """
        notifier = self._lifecycle_notifier
        if notifier is None:
            return
        try:
            await notifier(event, payload)
        except asyncio.CancelledError:
            raise
        except Exception:  # 通知失败不得影响调度主流程
            return

    # ------------------------------------------------------------------
    # 调度入口
    # ------------------------------------------------------------------
    async def schedule(
        self,
        task_id: str,
        subtasks: list[SubtaskSpec],
        *,
        concurrency: int | None = None,
        breaker: RuntimeBreaker | None = None,
        prior_results: dict[str, ExecutionResult] | None = None,
    ) -> dict[str, ExecutionResult]:
        """按拓扑序并发调度一个计划版本的全部子任务。

        Args:
            task_id: 任务编号。
            subtasks: 同一计划版本（plan_version）的子任务规格列表。
            concurrency: 本任务并发上限（来自规划阶段的确定性决策）；
                None 表示沿用 ``max_concurrency``。串行决策传入 1。
            breaker: 运行时熔断器（G4）；触发后不再追加新任务，空闲时
                只放行一个，余下子任务收敛为串行。
            prior_results: 批次外已定型结果（断点续跑 / 返工批次的
                上游依赖）；仅用于依赖判定，不重复执行、不计入返回值。

        Returns:
            ``{subtask_id: ExecutionResult}``，覆盖全部子任务：成功、失败、
            以及因依赖失败被跳过的记录（success=False）。

        Raises:
            SchedulerError: 子任务清单为空、ID 重复、依赖缺失或存在环时。
        """
        if not isinstance(task_id, str) or not task_id.strip():
            raise SchedulerError("task_id 不能为空")
        prior = dict(prior_results or {})
        self._validate_dag(subtasks, external_deps=set(prior))

        results: dict[str, ExecutionResult] = {}
        self.results = results
        pending: dict[str, SubtaskSpec] = {spec.subtask_id: spec for spec in subtasks}
        running: dict[asyncio.Task[ExecutionResult], str] = {}
        cap = (
            int(concurrency)
            if concurrency is not None
            else int(self._settings.max_concurrency)
        )
        semaphore = asyncio.Semaphore(max(1, cap))

        def dep_result(dep: str) -> ExecutionResult | None:
            if dep in results:
                return results[dep]
            return prior.get(dep)

        try:
            while pending or running:
                # 0) 停止门（任务 45）：pipeline 已停止时拒绝新的 dispatch
                #（PipelineStoppedError 为 BaseException，穿透宽捕获）
                self._gate_check("dispatch")
                # 预算检查：每轮派发前结算已完成子任务与嵌套 Agent 的用量
                if self._budget_check is not None:
                    self._budget_check()
                # 1) 依赖失败传播：依赖未成功的子任务直接跳过，不再执行
                for subtask_id in [
                    sid
                    for sid, spec in pending.items()
                    if any(
                        (current := dep_result(dep)) is not None
                        and not current.success
                        for dep in spec.dependencies
                    )
                ]:
                    spec = pending.pop(subtask_id)
                    failed_deps = [
                        dep
                        for dep in spec.dependencies
                        if (current := dep_result(dep)) is not None
                        and not current.success
                    ]
                    results[subtask_id] = ExecutionResult(
                        subtask_id=subtask_id,
                        success=False,
                        error=f"跳过：依赖子任务未成功 {failed_deps}",
                    )
                    self._notify_progress(subtask_id, results[subtask_id])

                # 2) 就绪子任务出队并发执行（并发度由信号量限制；
                # 熔断触发后由 collapse_ready 收敛为串行，不再铺开）
                ready = [
                    sid
                    for sid, spec in pending.items()
                    if all(
                        (current := dep_result(dep)) is not None
                        and current.success
                        for dep in spec.dependencies
                    )
                ]
                for subtask_id in collapse_ready(ready, running, breaker):
                    spec = pending.pop(subtask_id)
                    task = asyncio.create_task(
                        self._run_subtask_guarded(semaphore, task_id, spec)
                    )
                    running[task] = subtask_id

                # 3) 无运行中任务仍有剩余子任务：依赖状态异常，兕底标记失败
                if not running:
                    for subtask_id in list(pending):
                        spec = pending.pop(subtask_id)
                        results[subtask_id] = ExecutionResult(
                            subtask_id=subtask_id,
                            success=False,
                            error="无法调度：依赖关系异常（与依赖图校验结果不一致）",
                        )
                        self._notify_progress(subtask_id, results[subtask_id])
                    break

                # 4) 等待任一子任务完成后再推进
                done, _ = await asyncio.wait(
                    set(running.keys()), return_when=asyncio.FIRST_COMPLETED
                )
                for finished in done:
                    subtask_id = running.pop(finished)
                    try:
                        results[subtask_id] = finished.result()
                    except asyncio.CancelledError:
                        raise
                    except PipelineStoppedError:
                        # 停止信号（BaseException）：不落业务结果，向上
                        # 传播（finally 统一清理其余运行中任务）
                        raise
                    except Exception as exc:  # 执行器未处理异常：记为失败，不中断调度
                        results[subtask_id] = ExecutionResult(
                            subtask_id=subtask_id,
                            success=False,
                            error=f"调度异常：{type(exc).__name__}: {exc}",
                        )
                    self._record_breaker(
                        breaker, task_id, subtask_id, results[subtask_id]
                    )
                    self._notify_progress(subtask_id, results[subtask_id])
        finally:
            # 调度器明确拥有其创建的全部 asyncio tasks（任务 45）：无论
            # 正常返回、停止信号还是父协程取消，都取消并等待全部运行中
            # 任务收尾（孤儿任务清零；清理期真实异常落账，不吞业务异常）
            await self._shutdown_running(running, results)
        return results

    def _record_breaker(
        self,
        breaker: RuntimeBreaker | None,
        task_id: str,
        subtask_id: str,
        result: ExecutionResult,
    ) -> None:
        """把一次结果计入熔断器；首次触发时写审计事件（仅控制信息）。"""
        if breaker is None:
            return
        if not record_breaker_result(breaker, result):
            return
        self._events.log_event(
            EVENT_RUNTIME_BREAKER_TRIPPED,
            {
                "task_id": task_id,
                "subtask_id": subtask_id,
                **breaker.as_dict(),
                "action": "collapse_to_serial",
            },
        )

    async def _shutdown_running(
        self,
        running: dict[asyncio.Task[ExecutionResult], str],
        results: dict[str, ExecutionResult],
    ) -> None:
        """取消并等待全部运行中任务（调度器对 task 集合的所有权收口）。

        - 先统一 cancel，再等待全部任务真正结束（孤儿任务清零）；
        - 父协程已被取消时，清理阶段的重复取消按有界次数吞并，保证
          清理完成；清理自身不吞真实业务异常：执行器抛出的非取消
          异常落账为失败结果（可审计），取消 / 停止信号不落业务结果；
        - 清理期任务仍正常完成的 ExecutionResult 保留进 results（不
          丢弃已完成工作）。
        """
        if not running:
            return
        tasks = list(running.keys())
        for task in tasks:
            if not task.done():
                task.cancel()
        # 有界吞并清理期取消（父协程重复取消时 await asyncio.wait 会再
        # 次抛 CancelledError）：最多 3 轮，保证清理必然完成
        pending_wait = set(tasks)
        for _ in range(3):
            if not pending_wait:
                break
            try:
                done, pending_wait = await asyncio.wait(
                    pending_wait, return_when=asyncio.ALL_COMPLETED
                )
            except asyncio.CancelledError:
                continue
        for task in tasks:
            subtask_id = running[task]
            if task.cancelled():
                continue
            try:
                results[subtask_id] = task.result()
            except asyncio.CancelledError:
                continue
            except PipelineStoppedError:
                # 停止门拒绝的尝试：不落业务结果、不转重试
                continue
            except Exception as exc:
                # 清理期真实异常：落账为可审计失败，绝不吞掉
                results[subtask_id] = ExecutionResult(
                    subtask_id=subtask_id,
                    success=False,
                    error=f"清理期终止：{type(exc).__name__}: {exc}",
                )
        running.clear()

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------

    def _new_agent_record(
        self,
        task_id: str,
        spec: SubtaskSpec,
        *,
        can_request_replacement: bool,
        replacement_of: str | None,
    ) -> AgentRecord:
        """构造子任务执行用的 AgentRecord（原始或替代）。"""
        kind = "replacement" if replacement_of else "primary"
        role = spec.role.strip().lower() if isinstance(spec.role, str) else ""
        return AgentRecord(
            agent_id=f"{task_id}:{spec.subtask_id}:{kind}-{uuid.uuid4().hex[:8]}",
            agent_type=_ROLE_TYPE_MAP.get(role, "dynamic"),
            parent_task_id=task_id,
            plan_version=spec.plan_version,
            allowed_tools=list(spec.allowed_tools),
            attempt_count=0,
            can_request_replacement=can_request_replacement,
            replacement_of=replacement_of,
            status="created",
        )

    def _register_agent(self, record: AgentRecord) -> bool:
        """登记 Agent；未配置注册表时视为成功（仅本地记录）。"""
        if self._registry is None:
            return True
        try:
            self._registry.register(record)
        except AgentRegistryError:
            return False
        return True

    def _finalize_agent(self, record: AgentRecord, *, status: str) -> None:
        """收尾 Agent：更新状态、释放配额并记录事件。"""
        if self._registry is not None:
            try:
                self._registry.update_status(record.agent_id, status)  # type: ignore[arg-type]
                self._registry.release(record.agent_id)
            except AgentRegistryError:
                record.status = status  # type: ignore[assignment]
        else:
            record.status = status  # type: ignore[assignment]
        self._events.log_event(
            EVENT_AGENT_RELEASED,
            {
                "task_id": record.parent_task_id,
                "agent_id": record.agent_id,
                "status": status,
                "attempt_count": record.attempt_count,
            },
        )

    def _log_agent_created(self, task_id: str, record: AgentRecord) -> None:
        """记录 Agent 创建事件。"""
        self._events.log_event(
            EVENT_AGENT_CREATED,
            {
                "task_id": task_id,
                "agent_id": record.agent_id,
                "agent": record.model_dump(mode="json"),
            },
        )

    def _replacement_allowed(self, task_id: str, record: AgentRecord) -> bool:
        """判断原始 Agent 是否允许发起替换请求。"""
        if not record.can_request_replacement or record.replacement_requested:
            return False
        used = self._replacement_counts.get(task_id, 0)
        return used < int(self._settings.max_replacement_requests)

    @staticmethod
    def _failure_summary(error: str | None) -> str:
        """构造非敏感失败摘要（截断错误诊断文本，不含任何业务产出）。"""
        detail = (error or "未提供错误信息").strip()
        if len(detail) > _FAILURE_SUMMARY_MAX_CHARS:
            detail = detail[:_FAILURE_SUMMARY_MAX_CHARS] + "…"
        return f"替代 Agent 执行失败：{detail}"

    def _record_decomposition_issue(
        self,
        task_id: str,
        spec: SubtaskSpec,
        *,
        failed_agent_ids: list[str],
        failure_reasons: list[str],
        attempted_approaches: list[str],
        recommendation: str,
    ) -> TaskDecompositionIssue:
        """生成并记录 TaskDecompositionIssue（供编排器触发重规划）。"""
        issue = TaskDecompositionIssue(
            task_id=task_id,
            subtask_id=spec.subtask_id,
            plan_version=spec.plan_version,
            failed_agent_ids=list(failed_agent_ids),
            failure_reasons=[reason for reason in failure_reasons if reason],
            attempted_approaches=list(attempted_approaches),
            recommendation=recommendation,
        )
        self.decomposition_issues.append(issue)
        self._events.log_event(
            EVENT_DECOMPOSITION_ISSUE,
            {
                "task_id": task_id,
                "agent_id": failed_agent_ids[0] if failed_agent_ids else None,
                "issue": issue.model_dump(mode="json"),
            },
        )
        return issue

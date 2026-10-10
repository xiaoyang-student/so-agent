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
    EVENT_FAILURE_CLASSIFIED,
    EVENT_RECOVERY_DECIDED,
    EVENT_REPLACEMENT_COMPLETED,
    EVENT_REPLACEMENT_EXECUTING,
    EVENT_REPLACEMENT_FAILED,
    EVENT_REPLACEMENT_REQUESTED,
    EVENT_RETRY_SUPPRESSED,
    EVENT_TASK_DISPATCHED,
    EventLogger,
)
from so_agent.runtime.failure import (
    CONTRACT_FAILURE_TYPES,
    PROVIDER_FAILURE_TYPES,
    FailureClassifier,
    FailureDeduplicator,
    FailureType,
    RecoveryAction,
    RecoveryDecision,
    RecoveryPolicy,
    failure_fingerprint,
)
from so_agent.runtime.registry import AgentRegistry, AgentRegistryError
from so_agent.state_machine import PipelineStoppedError

# 执行器签名：接收（子任务规格, 执行 Agent 记录），返回结构化执行结果
TaskExecutor: TypeAlias = Callable[[SubtaskSpec, AgentRecord], Awaitable[ExecutionResult]]

# 替代生命周期通知签名：（事件名, 控制信息载荷）→ None。事件名与
# runtime.events 中的替代事件类型常量一致；载荷仅含控制信息（子任务
# 编号、申请编号、Agent 编号、尝试次数、失败分类与非敏感摘要），
# 不携带任何 Agent 的业务产出。由编排器注入执行协调器的受控接收端。
ReplacementLifecycleNotifier: TypeAlias = Callable[[str, dict[str, Any]], Awaitable[None]]

# 替代生命周期通知的失败分类（载荷 failure_category 字段取值）
_REPLACEMENT_FAILED_CATEGORY_CANCELLED: str = "cancelled"
_REPLACEMENT_FAILED_CATEGORY_REGISTRATION: str = "agent_registration_rejected"
_REPLACEMENT_FAILED_CATEGORY_ATTEMPTS: str = "attempts_exhausted"

# 终止通知摘要的错误文本截断上限（防御超长错误信息进入审计载荷）
_FAILURE_SUMMARY_MAX_CHARS: int = 200

# 外层硬兜底清理宽限（秒）：调度器对单次执行的 asyncio.wait_for 超时
# = supervisor_total_timeout + 本宽限。宽限仅用于让内层分阶段流式超时
# （任务 33）先行触发并完成取消/审计，外层随后兜底；不改变任何工具或
# 授权时限。主管总预算不短于嵌套子 Agent 工具窗口（配置校验器保底），
# 因此外层兜底同时覆盖嵌套子 Agent 的最长合法运行。
_ATTEMPT_TIMEOUT_CLEANUP_GRACE: float = 10.0


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

# 子任务 role → AgentRecord.agent_type 映射
_ROLE_TYPE_MAP: dict[str, str] = {
    "code": "code_agent",
    "code_agent": "code_agent",
    "review": "review_agent",
    "review_agent": "review_agent",
    "creator": "subagent_creator",
    "subagent_creator": "subagent_creator",
}

_ISSUE_RECOMMENDATION = (
    "任务分解可能存在问题：建议主 Agent 重新评估子任务粒度、执行角色分配"
    "与工具授权范围（allowed_tools），必要时拆分该子任务后重规划。"
)


class SchedulerError(Exception):
    """调度器领域错误：子任务清单为空、依赖非法、存在循环依赖等。"""


class SubtaskDAGScheduler:
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
            attempt_timeout: 单次执行的超时（秒）；默认
                ``supervisor_total_timeout + _ATTEMPT_TIMEOUT_CLEANUP_GRACE``：
                外层 wait_for 作为硬兜底，比内层主管调用的绝对总预算
                （不短于嵌套子 Agent 工具窗口）多出极小清理宽限，确保
                内层先触发取消与审计，外层不会提前杀死。
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

    # ------------------------------------------------------------------
    # 替代生命周期通知（受控回调；任务 34）
    # ------------------------------------------------------------------
    def set_lifecycle_notifier(self, notifier: ReplacementLifecycleNotifier) -> None:
        """注入（或替换）替代生命周期通知回调（由编排器装配时调用）。"""
        self._lifecycle_notifier = notifier

    def set_gate(self, gate: Any) -> None:
        """注入（或替换）pipeline 停止门（由编排器装配时调用）。"""
        self._gate = gate

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
    async def schedule(self, task_id: str, subtasks: list[SubtaskSpec]) -> dict[str, ExecutionResult]:
        """按拓扑序并发调度一个计划版本的全部子任务。

        Args:
            task_id: 任务编号。
            subtasks: 同一计划版本（plan_version）的子任务规格列表。

        Returns:
            ``{subtask_id: ExecutionResult}``，覆盖全部子任务：成功、失败、
            以及因依赖失败被跳过的记录（success=False）。

        Raises:
            SchedulerError: 子任务清单为空、ID 重复、依赖缺失或存在环时。
        """
        if not isinstance(task_id, str) or not task_id.strip():
            raise SchedulerError("task_id 不能为空")
        self._validate_dag(subtasks)

        results: dict[str, ExecutionResult] = {}
        self.results = results
        pending: dict[str, SubtaskSpec] = {spec.subtask_id: spec for spec in subtasks}
        running: dict[asyncio.Task[ExecutionResult], str] = {}
        semaphore = asyncio.Semaphore(max(1, int(self._settings.max_concurrency)))

        try:
            while pending or running:
                # 0) 停止门（任务 45）：pipeline 已停止时拒绝新的 dispatch
                #（PipelineStoppedError 为 BaseException，穿透宽捕获）
                self._gate_check("dispatch")
                # 1) 依赖失败传播：依赖未成功的子任务直接跳过，不再执行
                for subtask_id in [
                    sid
                    for sid, spec in pending.items()
                    if any(dep in results and not results[dep].success for dep in spec.dependencies)
                ]:
                    spec = pending.pop(subtask_id)
                    failed_deps = [
                        dep
                        for dep in spec.dependencies
                        if dep in results and not results[dep].success
                    ]
                    results[subtask_id] = ExecutionResult(
                        subtask_id=subtask_id,
                        success=False,
                        error=f"跳过：依赖子任务未成功 {failed_deps}",
                    )

                # 2) 就绪子任务出队并发执行（并发度由信号量限制）
                for subtask_id in [
                    sid
                    for sid, spec in pending.items()
                    if all(dep in results and results[dep].success for dep in spec.dependencies)
                ]:
                    spec = pending.pop(subtask_id)
                    task = asyncio.create_task(self._run_subtask_guarded(semaphore, task_id, spec))
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
        finally:
            # 调度器明确拥有其创建的全部 asyncio tasks（任务 45）：无论
            # 正常返回、停止信号还是父协程取消，都取消并等待全部运行中
            # 任务收尾（孤儿任务清零；清理期真实异常落账，不吞业务异常）
            await self._shutdown_running(running, results)
        return results

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
    # 依赖图
    # ------------------------------------------------------------------
    def topological_order(self, subtasks: list[SubtaskSpec]) -> list[str]:
        """计算子任务的拓扑序（Kahn 算法，保持输入相对顺序）。

        Raises:
            SchedulerError: 清单为空、ID 重复、自依赖、依赖缺失或存在环时。
        """
        if not subtasks:
            raise SchedulerError("子任务列表为空，无法调度")

        ids = [spec.subtask_id for spec in subtasks]
        if len(ids) != len(set(ids)):
            duplicated = sorted({sid for sid in ids if ids.count(sid) > 1})
            raise SchedulerError(f"存在重复的 subtask_id：{duplicated}")
        id_set = set(ids)

        indegree: dict[str, int] = {sid: 0 for sid in ids}
        dependents: dict[str, list[str]] = {sid: [] for sid in ids}
        for spec in subtasks:
            for dep in spec.dependencies:
                if dep == spec.subtask_id:
                    raise SchedulerError(f"子任务 {spec.subtask_id} 依赖自身，非法")
                if dep not in id_set:
                    raise SchedulerError(
                        f"子任务 {spec.subtask_id} 依赖不存在的子任务 {dep!r}"
                    )
                indegree[spec.subtask_id] += 1
                dependents[dep].append(spec.subtask_id)

        queue = [sid for sid in ids if indegree[sid] == 0]
        order: list[str] = []
        while queue:
            current = queue.pop(0)
            order.append(current)
            for nxt in dependents[current]:
                indegree[nxt] -= 1
                if indegree[nxt] == 0:
                    queue.append(nxt)

        if len(order) != len(ids):
            remaining = sorted(set(ids) - set(order))
            raise SchedulerError(f"子任务依赖图存在循环依赖：{remaining}")
        return order

    def _validate_dag(self, subtasks: list[SubtaskSpec]) -> None:
        """调度前完成依赖图全量校验（委托拓扑序计算）。"""
        self.topological_order(subtasks)

    # ------------------------------------------------------------------
    # 单个子任务执行（重试 + 替换）
    # ------------------------------------------------------------------
    async def _run_subtask_guarded(
        self,
        semaphore: asyncio.Semaphore,
        task_id: str,
        spec: SubtaskSpec,
    ) -> ExecutionResult:
        """信号量包裹的子任务执行（限制全局并发度）。"""
        async with semaphore:
            return await self._execute_subtask(task_id, spec)

    async def _execute_subtask(self, task_id: str, spec: SubtaskSpec) -> ExecutionResult:
        """执行单个子任务：原始 Agent 重试 → 替换 → 分解问题上报。"""
        max_attempts = max(
            1, min(int(spec.max_attempts), int(self._settings.max_agent_attempts))
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
            task_id, spec, primary, max_attempts
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
            if not contract_repair_round:
                attempt += 1  # 业务尝试计数（契约修复轮不推进，任务 69）
            is_repair_round = contract_repair_round
            contract_repair_round = False

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
                outcome = await asyncio.wait_for(
                    run(effective_spec, agent), timeout=timeout
                )
            except asyncio.TimeoutError as err:
                exc = err
                # 内层（主管总预算 / 阶段窗口）超时先于外层 wait_for 触发时，
                # 异常消息携带 phase 证据——保留类型与消息供审计与
                # "TimeoutError" 字样检索（test_workflow_timeout）
                detail = (
                    f"{type(err).__name__}: {err}" if str(err) else type(err).__name__
                )
                outcome = ExecutionResult(
                    subtask_id=spec.subtask_id,
                    success=False,
                    error=f"单次执行超时（>{timeout:g}s），已终止（{detail}）",
                    duration=timeout,
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

    async def _sleep_backoff(self, seconds: float) -> None:
        """退避休眠：受 pipeline 剩余预算截断（绝不睡过 deadline）；
        sleeper 可注入以获得确定性测试行为。"""
        delay = max(0.0, float(seconds))
        remaining = self._gate_remaining()
        if remaining is not None:
            delay = max(0.0, min(delay, remaining))
        await self._sleeper(delay)

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------
    def _resolve_attempt_timeout(self) -> float:
        """单次执行超时：显式配置优先，否则 supervisor_total_timeout + 宽限。

        外层 asyncio.wait_for 是硬兜底：内层 run_agent_streamed 以主管
        绝对总预算（supervisor_total_timeout，运行时不超过 pipeline 剩余
        预算）为 deadline 并负责取消/审计；主管总预算不短于嵌套子 Agent
        工具窗口（配置校验器保底），因此外层必须比内层多出极小宽限，
        避免提前杀死内部总预算。显式传入的 attempt_timeout 仍然优先
        （显式即显式）。
        """
        if self._attempt_timeout is not None:
            return float(self._attempt_timeout)
        return (
            float(self._settings.supervisor_total_timeout)
            + _ATTEMPT_TIMEOUT_CLEANUP_GRACE
        )

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

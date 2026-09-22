"""并发调度器（TaskScheduler）。

职责与行为约定：
- 依据 SubtaskSpec.dependencies 构建依赖图（DAG），做完整性校验（缺失依赖、
  自依赖、重复 ID、循环依赖一律拒绝）后按拓扑序调度；
- 互不依赖的子任务并发执行，并发度由 ``max_concurrency`` 限制
  （asyncio.Semaphore 实现）；依赖失败的子任务直接标记跳过（success=False），
  不进入执行队列，并沿依赖边传播；
- 单个子 Agent 最多执行 ``max_agent_attempts`` 次（attempt_count 跟踪，
  与 SubtaskSpec.max_attempts 取较小者），单次执行超时按失败处理；
- 原始子 Agent 用尽尝试次数后，可发起一次 ReplacementRequest（受
  ``max_replacement_requests`` 全局限额约束）；替代 Agent 的
  ``can_request_replacement`` 固定为 False；替代 Agent 仍失败则生成
  TaskDecompositionIssue，供编排器触发重规划；
- 全程事件留痕（task_dispatched / agent_created / agent_released /
  replacement_requested / decomposition_issue），经 EventLogger 写入。

调用示例::

    scheduler = TaskScheduler(executor=my_executor, registry=registry)
    results = await scheduler.schedule(task_id, subtasks)

executor 签名为 ``async def (subtask, agent_record) -> ExecutionResult``；
调度期间可读取 ``scheduler.results`` 获取已完成子任务的（含依赖的）输出。
"""

from __future__ import annotations

import asyncio
import uuid
from typing import Awaitable, Callable, TypeAlias

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
    EVENT_REPLACEMENT_REQUESTED,
    EVENT_TASK_DISPATCHED,
    EventLogger,
)
from so_agent.runtime.registry import AgentRegistry, AgentRegistryError

# 执行器签名：接收（子任务规格, 执行 Agent 记录），返回结构化执行结果
TaskExecutor: TypeAlias = Callable[[SubtaskSpec, AgentRecord], Awaitable[ExecutionResult]]

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


class TaskScheduler:
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
                ``model_timeout × max_model_turns``（单次 Agent 运行最多
                max_model_turns 个模型回合，每回合上限 model_timeout）。
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

        # 调度过程状态（每次 schedule 调用刷新 results；请求与问题累计保留）
        self.results: dict[str, ExecutionResult] = {}
        self.replacement_requests: list[ReplacementRequest] = []
        self.decomposition_issues: list[TaskDecompositionIssue] = []
        self._replacement_counts: dict[str, int] = {}  # task_id → 已发起的替换请求数

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

        while pending or running:
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

            # 3) 无运行中任务仍有剩余子任务：依赖状态异常，兜底标记失败
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
                except Exception as exc:  # 执行器未处理异常：记为失败，不中断调度
                    results[subtask_id] = ExecutionResult(
                        subtask_id=subtask_id,
                        success=False,
                        error=f"调度异常：{type(exc).__name__}: {exc}",
                    )
        return results

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

        primary_result, primary_failures = await self._run_attempts(
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
        if self._replacement_allowed(task_id, primary):
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

            replacement = self._new_agent_record(
                task_id,
                spec,
                can_request_replacement=False,  # 替代 Agent 固定无替换权
                replacement_of=primary.agent_id,
            )
            if self._register_agent(replacement):
                self._log_agent_created(task_id, replacement)
                replacement_result, replacement_failures = await self._run_attempts(
                    task_id, spec, replacement, max_attempts, executor=self._replacement_executor
                )
                if replacement_result.success:
                    self._finalize_agent(replacement, status="completed")
                    return replacement_result
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

        # ---- 3) 无替换资格/额度：直接上报任务分解问题 ----
        self._record_decomposition_issue(
            task_id,
            spec,
            failed_agent_ids=[primary.agent_id],
            failure_reasons=failure_reasons
            + ["替换额度已用尽或该 Agent 不具备替换资格"],
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
    ) -> tuple[ExecutionResult, list[str]]:
        """对单个 Agent 执行最多 max_attempts 次尝试（含超时处理）。

        Returns:
            (最后一次 ExecutionResult, 每次失败摘要列表)。
        """
        run = executor or self._executor
        timeout = self._resolve_attempt_timeout()
        last: ExecutionResult | None = None
        failures: list[str] = []

        for attempt in range(1, max_attempts + 1):
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
                },
            )
            try:
                outcome = await asyncio.wait_for(run(spec, agent), timeout=timeout)
            except asyncio.TimeoutError:
                outcome = ExecutionResult(
                    subtask_id=spec.subtask_id,
                    success=False,
                    error=f"单次执行超时（>{timeout:g}s），已终止",
                    duration=timeout,
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                outcome = ExecutionResult(
                    subtask_id=spec.subtask_id,
                    success=False,
                    error=f"执行异常：{type(exc).__name__}: {exc}",
                )

            if not isinstance(outcome, ExecutionResult):
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
                return outcome, failures

            agent.status = "waiting"
            failures.append(f"第 {attempt} 次尝试失败：{outcome.error or '未提供错误信息'}")

        assert last is not None  # max_attempts >= 1 保证至少执行一次
        return last, failures

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------
    def _resolve_attempt_timeout(self) -> float:
        """单次执行超时：显式配置优先，否则 model_timeout × max_model_turns。"""
        if self._attempt_timeout is not None:
            return float(self._attempt_timeout)
        return float(self._settings.model_timeout) * float(self._settings.max_model_turns)

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

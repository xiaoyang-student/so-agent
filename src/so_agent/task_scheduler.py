"""长任务调度器（PeriodicTaskPoller）。

异步守护进程，持续轮询 TaskStore 中到期的持久化任务并触发执行。
支持：
- 定时轮询（可配置间隔，见 ``Settings.scheduler_poll_interval``）
- 周期任务自动重调度；cron 可按任务时区（``ScheduleSpec.timezone``）计算
- 错过触发窗口按 ``scheduler_misfire_policy`` 补跑一次（coalesce）、
  有界补跑（catch_up）或跳过（skip）
- 失败按 ``scheduler_task_max_retries`` / 退避基数重排，耗尽后保存
  检查点并暂停
- 每次执行（含跳过）写入 ``task_runs``，供 Web 端逐次查看
- 优雅停机（``stop()`` 置位运行标志，下一轮轮询后退出）

执行动作说明：到期动作经四象限路由执行——简单动作仍由单次 LLM
完成；目标或验收标准要求文件、命令等工具时，自动升级为完整工作流，
获得沙箱工具能力。每个调度任务使用独立子沙箱，避免并发产物冲突。
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from so_agent.config import Settings
from so_agent.context import ProjectContext
from so_agent.cron import parse_cron
from so_agent.models import (
    ClassificationResult,
    RouteStep,
    ScheduleSpec,
    TaskStatus,
    TaskType,
)
from so_agent.persistence import TaskRecord, TaskStore
from so_agent.runtime.logging_setup import get_logger

logger = get_logger("scheduler")


def _checkpoint_text(checkpoint: dict[str, Any], limit: int = 2000) -> str:
    """序列化检查点用于提示词，超长截断避免挤占上下文。"""
    text = json.dumps(checkpoint, ensure_ascii=False)
    if len(text) <= limit:
        return text
    return text[:limit] + "…（已截断）"


def _parse_iso(value: str | None) -> datetime | None:
    """解析存储时间；非法或空值返回 None。"""
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _resolve_timezone(name: str | None):
    """解析 IANA 时区；空值按 UTC 处理，未知时区抛 ValueError。"""
    if not name or not str(name).strip():
        return timezone.utc
    try:
        return ZoneInfo(str(name).strip())
    except Exception as exc:
        raise ValueError(f"未知时区：{name}") from exc


def _compute_next_run(
    spec: ScheduleSpec, now: datetime | None = None
) -> str | None:
    """根据调度规格计算下次执行时间（UTC ISO 格式）。

    - ``interval``：从基准时刻起算固定间隔（与时区无关）；
    - ``cron``：按真实 cron 表达式求解；配置 ``timezone`` 时按该时区的
      本地墙钟计算再换算回 UTC，否则按 UTC；
    - ``once`` / ``event`` / 非周期：返回 ``None``。
    """
    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    current = current.astimezone(timezone.utc)
    if not spec.is_recurring:
        return None
    if spec.trigger == "interval":
        if not spec.interval_seconds or spec.interval_seconds <= 0:
            raise ValueError("周期任务的 interval_seconds 必须为正整数")
        return (current + timedelta(seconds=spec.interval_seconds)).isoformat()
    if spec.trigger == "cron":
        if not spec.cron or not spec.cron.strip():
            raise ValueError("周期任务的 cron 表达式不能为空")
        expression = parse_cron(spec.cron)
        if not spec.timezone:
            return expression.next_after(current).isoformat()
        tz = _resolve_timezone(spec.timezone)
        # CronExpression 以“UTC 标签”表示墙钟字段：先转本地时间并去掉
        # 时区参与求解，再把结果按目标时区解释回真实时刻。
        wall_clock = current.astimezone(tz).replace(tzinfo=timezone.utc)
        local_next = expression.next_after(wall_clock)
        return (
            local_next.replace(tzinfo=tz)
            .astimezone(timezone.utc)
            .isoformat()
        )
    return None


def _retry_backoff_seconds(
    backoff_base: float, failures: int, cap: float = 3600.0
) -> float:
    """第 N 次失败后的退避秒数（指数增长，单次上限 1 小时）。"""
    if backoff_base <= 0:
        return 0.0
    return min(cap, backoff_base * (2 ** max(0, failures - 1)))


class PeriodicTaskPoller:
    """长任务调度器：定时轮询到期任务并触发执行。"""

    def __init__(
        self,
        store: TaskStore,
        context: ProjectContext,
        settings: Settings | None = None,
    ):
        self._store = store
        self._context = context
        self._settings = settings or Settings()
        self._running = False
        self._poll_interval = self._settings.scheduler_poll_interval
        self._max_concurrent = max(1, self._settings.scheduler_max_concurrent_tasks)
        self._semaphore = asyncio.Semaphore(self._max_concurrent)
        self._inflight: dict[str, asyncio.Task[None]] = {}
        self._stop_event = asyncio.Event()

    async def start(self) -> None:
        """启动调度循环（阻塞运行直到 stop()）。

        并发上限由 ``scheduler_max_concurrent_tasks`` 控制；同一任务的
        重复到期不会叠加执行（``_inflight`` 去重 + ``TaskStore.claim``
        条件更新双重保护）。
        """
        self._running = True
        self._stop_event.clear()
        logger.info(
            "调度器已启动，轮询间隔 %ss，并发上限 %s",
            self._poll_interval,
            self._max_concurrent,
        )
        try:
            while self._running:
                try:
                    self._dispatch_due_tasks()
                except Exception as exc:
                    logger.exception("调度器轮询异常：%s", exc)
                await self._wait_for_stop_or_tick()
        finally:
            await self._drain_inflight()

    def _dispatch_due_tasks(self) -> None:
        for task in self._store.get_due_tasks():
            if task.task_id in self._inflight:
                continue
            scheduled = asyncio.create_task(self._execute_guarded(task))
            self._inflight[task.task_id] = scheduled
            scheduled.add_done_callback(
                lambda _task, task_id=task.task_id: self._inflight.pop(task_id, None)
            )

    async def _wait_for_stop_or_tick(self) -> None:
        try:
            await asyncio.wait_for(self._stop_event.wait(), timeout=self._poll_interval)
        except asyncio.TimeoutError:
            pass

    async def _execute_guarded(self, task: TaskRecord) -> None:
        async with self._semaphore:
            if not self._running and self._stop_event.is_set():
                return
            await self._execute_task(task)

    async def _drain_inflight(self) -> None:
        """停机时取消并回收在途任务（状态回到 paused，可恢复）。"""
        pending = list(self._inflight.values())
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        self._inflight.clear()

    async def _execute_task(self, task: TaskRecord) -> None:
        """执行单个到期任务（含检查点恢复、错过补跑与失败重试）。

        先以条件更新抢占任务（仅 ``scheduled`` 可被抢占），抢占失败说明
        已被其它调度进程或上一轮执行处理，直接返回，避免重复执行。
        """
        if not self._store.claim(task.task_id):
            return
        logger.info("调度执行任务 %s：%s", task.task_id, task.objective[:50])

        spec = task.schedule_spec
        now = datetime.now(timezone.utc)
        scheduled_for = task.next_run_at or now.isoformat()
        policy = self._settings.scheduler_misfire_policy
        overdue = max(0.0, (now - (_parse_iso(scheduled_for) or now)).total_seconds())

        if (
            policy == "skip"
            and spec is not None
            and spec.is_recurring
            and overdue > self._settings.scheduler_misfire_grace_seconds
        ):
            next_run = _compute_next_run(spec, now)
            self._store.record_skipped_run(
                task.task_id,
                scheduled_for,
                f"错过触发窗口 {int(overdue)} 秒，按策略跳过",
                next_run_at=next_run,
            )
            if next_run:
                self._store.reschedule(task.task_id, next_run)
            else:
                self._store.update_status(task.task_id, "completed")
            self._store.add_log(
                task.task_id,
                "execution_skipped",
                {"scheduled_for": scheduled_for, "next_run_at": next_run},
            )
            logger.info("任务 %s 错过触发窗口，已跳过并重排：%s", task.task_id, next_run)
            return

        checkpoint = self._store.load_checkpoint(task.task_id)
        catch_up = policy == "catch_up" and spec is not None and spec.is_recurring
        run_budget = (
            self._settings.scheduler_max_catch_up_runs if catch_up else 1
        )
        current_scheduled = scheduled_for
        runs_done = 0

        while True:
            run_id = self._store.start_run(task.task_id, current_scheduled)
            self._store.add_log(
                task.task_id,
                "execution_start",
                {
                    "checkpoint": checkpoint is not None,
                    "scheduled_for": current_scheduled,
                    "catch_up": catch_up,
                },
            )
            try:
                result = await self._run_task_action(task, checkpoint)
            except asyncio.CancelledError:
                self._store.finish_run(
                    run_id, "cancelled", error="调度器停止，任务被中断"
                )
                self._store.pause(task.task_id, error="调度器停止，任务被中断")
                self._store.add_log(task.task_id, "execution_interrupted")
                raise
            except Exception as exc:
                self._handle_failure(task, run_id, exc, checkpoint)
                return

            runs_done += 1
            finished = datetime.now(timezone.utc)
            next_run = None
            if spec is not None and spec.is_recurring:
                base = (
                    (_parse_iso(current_scheduled) or finished)
                    if catch_up
                    else finished
                )
                candidate = _compute_next_run(spec, base)
                candidate_dt = _parse_iso(candidate)
                if (
                    catch_up
                    and candidate_dt is not None
                    and candidate_dt <= finished
                    and runs_done < run_budget
                ):
                    # 仍在补跑预算内：立即执行下一个错过的触发点。
                    self._store.finish_run(
                        run_id, "completed", result=result, next_run_at=candidate
                    )
                    self._store.record_success(task.task_id, at=finished.isoformat())
                    self._store.add_log(
                        task.task_id,
                        "execution_complete",
                        {"scheduled_for": current_scheduled, "catch_up": True},
                    )
                    current_scheduled = candidate
                    continue
                if candidate_dt is not None and candidate_dt <= finished:
                    candidate = _compute_next_run(spec, finished)
                next_run = candidate

            self._store.update_status(task.task_id, "completed", result=result)
            self._store.record_success(task.task_id, at=finished.isoformat())
            self._store.finish_run(run_id, "completed", result=result, next_run_at=next_run)
            self._store.add_log(task.task_id, "execution_complete")
            if next_run:
                self._store.reschedule(task.task_id, next_run)
                logger.info("周期任务 %s 下次执行：%s", task.task_id, next_run)
            return

    def _handle_failure(
        self,
        task: TaskRecord,
        run_id: str,
        exc: Exception,
        checkpoint: dict[str, Any] | None,
    ) -> None:
        """失败收口：按配置重排重试，耗尽后保存检查点并暂停。"""
        error = str(exc)
        self._store.save_checkpoint(
            task.task_id,
            {"error": error, "partial_state": checkpoint},
        )
        max_retries = self._settings.scheduler_task_max_retries
        failures = task.consecutive_failures + 1
        if max_retries > 0 and failures <= max_retries:
            delay = _retry_backoff_seconds(
                self._settings.scheduler_task_retry_backoff_seconds, failures
            )
            retry_at = (
                datetime.now(timezone.utc) + timedelta(seconds=delay)
            ).isoformat()
            self._store.finish_run(
                run_id, "failed", error=error, next_run_at=retry_at
            )
            self._store.schedule_retry(task.task_id, retry_at, error=error)
            self._store.add_log(
                task.task_id,
                "execution_retry_scheduled",
                {"error": error, "attempt": failures, "next_run_at": retry_at},
            )
            logger.warning(
                "任务 %s 执行失败（第 %s 次），%ss 后重试：%s",
                task.task_id,
                failures,
                delay,
                error,
            )
            return
        self._store.finish_run(run_id, "failed", error=error)
        self._store.pause(task.task_id, error=error)
        self._store.add_log(task.task_id, "execution_error", {"error": error})
        logger.warning("任务 %s 执行失败，已暂停：%s", task.task_id, exc)

    async def _run_task_action(
        self, task: TaskRecord, checkpoint: dict[str, Any] | None
    ) -> str:
        """通过四象限路由执行到期动作（工具任务进入完整工作流）。

        到期动作与普通任务共用同一套分类规则：简单/推理动作仍由单次
        LLM 完成；目标或验收标准要求文件、命令等工具时，由确定性升级
        规则改走完整工作流，获得沙箱工具能力。每个调度任务使用独立
        子沙箱（``<sandbox>/scheduled/<task_id>``），并发任务产物互不
        覆盖。

        异常处理约定：动作未成功（非 ``completed`` 终态）或调用异常
        直接向上抛给 ``_execute_task``，由上层保存检查点并把任务置为
        ``paused``。
        """
        action = ""
        if task.schedule_spec:
            action = task.schedule_spec.action or ""
        action = action or task.objective

        # 延迟导入：router 依赖 orchestrator，调度器保持轻量导入
        from so_agent.router import TaskRouter

        execution_context = ProjectContext(
            sandbox_dir=self._task_sandbox(task.task_id),
            project_name=f"scheduled-{task.task_id}",
            task_store_dir=self._context.task_store_dir,
            config=self._settings,
        )
        router = TaskRouter(execution_context, self._settings)
        classification = await router.classify(
            action, task.constraints, task.acceptance_criteria
        )
        classification = self._guard_nested_schedule(classification)

        # 检查点只在执行阶段注入，不参与分类，避免历史错误信息干扰选路
        run_constraints = task.constraints
        if checkpoint:
            run_constraints = (
                f"{run_constraints}\n上次执行的检查点信息："
                f"{_checkpoint_text(checkpoint)}"
            ).strip()

        output = await router.route(
            action,
            run_constraints,
            task.acceptance_criteria,
            classification=classification,
        )
        self._store.add_log(
            task.task_id,
            "execution_route",
            {
                "task_type": classification.task_type.value,
                "reasoning": classification.reasoning,
                "steps": [step.model_dump() for step in classification.steps],
                "status": output.status.value,
            },
        )
        if output.status is not TaskStatus.COMPLETED:
            raise RuntimeError(
                f"调度动作未成功（{output.status.value}）："
                f"{output.final_result or ''}"
            )
        return output.final_result or ""

    def _task_sandbox(self, task_id: str) -> Path:
        """调度任务执行沙箱：按任务隔离，避免并发产物互相覆盖。"""
        root = (
            Path(self._context.sandbox_dir).expanduser().resolve()
            / "scheduled"
            / task_id
        )
        root.mkdir(parents=True, exist_ok=True)
        return root

    @staticmethod
    def _guard_nested_schedule(
        classification: ClassificationResult,
    ) -> ClassificationResult:
        """到期动作不得再创建嵌套调度任务，统一改走完整工作流。"""
        if classification.task_type is not TaskType.PERSISTENT_TASK:
            return classification
        return classification.model_copy(
            update={
                "task_type": TaskType.FULL_WORKFLOW,
                "reasoning": (
                    f"{classification.reasoning}；到期动作不再创建嵌套调度"
                    "任务，按完整工作流执行"
                ),
                "steps": [
                    *classification.steps,
                    RouteStep(
                        stage="scheduler_nested_task_guard",
                        status="upgraded",
                        detail="到期动作不接受嵌套调度，改走完整工作流",
                    ),
                ],
            }
        )

    def stop(self) -> None:
        """停止调度循环并取消在途任务（任务落回 paused，可恢复）。"""
        self._running = False
        self._stop_event.set()
        for task in self._inflight.values():
            task.cancel()
        logger.info("调度器已停止")

    def register(self, task: TaskRecord) -> None:
        """注册任务到调度器（计算首次执行时间）。"""
        if task.schedule_spec and not task.next_run_at:
            next_run = _compute_next_run(task.schedule_spec)
            if next_run:
                self._store.reschedule(task.task_id, next_run)
            elif task.schedule_spec.trigger == "once":
                # 一次性任务立即调度
                self._store.reschedule(
                    task.task_id,
                    datetime.now(timezone.utc).isoformat(),
                )

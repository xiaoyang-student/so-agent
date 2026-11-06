"""Web 控制台任务管理器。

每个 Web 任务拥有独立的沙箱目录与 ``ProjectContext``，从而实现任务间
状态隔离（Agent 注册表、工具授权、事件日志互不串扰）。管理器负责：

- 提交任务并在后台执行（信号量限制并发，超出排队）；
- 四象限分类后按路径执行：完整工作流持有 ``WorkflowEngine`` 引用，
  以便实时读取计划版本、子任务结果并生成流程图数据；
- 组织只读快照（阶段、计划 DAG、时间线、最终产物与沙箱文件）；
- 取消任务（协作式取消 + 取消 asyncio 任务）与进程退出清理。

安全边界：所有文件浏览均限制在任务沙箱目录内；内部目录（生成工具、
持久化存储、缓存）与隐藏文件不对外暴露。
"""

from __future__ import annotations

import asyncio
import copy
import json
import mimetypes
import os
import re
import shutil
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

# 必须在导入 OpenAI Agents SDK（经 router/workflow 间接导入）之前设置
os.environ.setdefault("OPENAI_AGENTS_DISABLE_TRACING", "1")

from so_agent.config import Settings, get_settings
from so_agent.context import ProjectContext
from so_agent.models import (
    ClassificationResult,
    ExecutionResult,
    HumanGuidance,
    RouteStep,
    SubtaskSpec,
    TaskStatus,
    TaskType,
)
from so_agent.orchestrator import OrchestratorOutput
from so_agent.paths.single_file import maybe_run_single_file_fast_path
from so_agent.persistence import TaskRecord, TaskStore, resolve_task_store_path
from so_agent.router import TaskRouter
from so_agent.runtime.config_loader import ConfigLoadError
from so_agent.runtime.logging_setup import get_logger
from so_agent.runtime.metrics import aggregate_events, merge_event_summaries
from so_agent.runtime.notification_outbox import (
    NotificationOutbox,
    NotificationOutboxWorker,
    resolve_outbox_path,
)
from so_agent.runtime.sandbox import available_sandbox_backends
from so_agent.runtime.sandbox_container import detect_container_runtime
from so_agent.runtime.sandbox_health import resolve_backend_name
from so_agent.runtime.tool_management import (
    load_generated_manifests,
    summarize_manifests,
)
from so_agent.runtime.tool_store import resolve_mcp_servers_dir
from so_agent.mcp.directory import scan_mcp_directory
from so_agent.task_scheduler import PeriodicTaskPoller
from so_agent.web.store import WebTaskStore
from so_agent.web.result_archive import FileResultArchive, JsonlHistoryStore
from so_agent.web.text import tr
from so_agent.workflow import HumanGuidanceError, WorkflowEngine

from datetime import datetime
from datetime import timedelta
from datetime import timezone
from pathlib import Path
from so_agent.config import Settings
from so_agent.config import get_settings
from so_agent.context import ProjectContext
from so_agent.models import ClassificationResult
from so_agent.models import HumanGuidance
from so_agent.models import TaskStatus
from so_agent.models import TaskType
from so_agent.orchestrator import OrchestratorOutput
from so_agent.persistence import TaskStore
from so_agent.persistence import resolve_task_store_path
from so_agent.runtime.metrics import aggregate_events
from so_agent.web.text import tr
from typing import Any
import asyncio
import json
import shutil
import uuid

from so_agent.web.common import (
    ArtifactError,
    GuidanceStateError,
    MAX_ARTIFACTS,
    MAX_ARTIFACT_BYTES,
    MAX_OBJECTIVE_CHARS,
    MAX_TASKS_IN_MEMORY,
    TERMINAL_STATUSES,
    WebTask,
    _GENERATED_TOOL_ACTIONS,
    _GENERATED_TOOL_STEP_LABELS,
    _GENERATED_TOOL_STEP_LABELS_EN,
    _INTERNAL_DIR_NAMES,
    _INTERNAL_DIR_PREFIXES,
    _INTERNAL_FILE_NAMES,
    _MAX_PREVIEW_CHARS,
    _MAX_TIMELINE_EVENTS,
    _MAX_TOOL_CALLS,
    _PHASE_RANK,
    _ROUTE_BRANCHES,
    _ROUTE_BRANCHES_EN,
    _ROUTE_LABELS,
    _ROUTE_LABELS_EN,
    _ROUTE_METHOD_LABELS,
    _ROUTE_METHOD_LABELS_EN,
    _STATUS_LABELS,
    _STATUS_LABELS_EN,
    _TASK_ID_PATTERN,
    _TEXT_EXTENSIONS,
    _preview,
    _route_labels,
    _split_items,
    _status_label,
    _utc_now_iso,
    logger,
)
from so_agent.web.artifacts import ArtifactMixin
from so_agent.web.journey import JourneyMixin
from so_agent.web.metrics import MetricsMixin
from so_agent.web.scheduled_tasks import ScheduledTaskMixin
from so_agent.web.snapshot import SnapshotMixin
from so_agent.web.timeline import TimelineMixin



class TaskManager(
    ScheduledTaskMixin,
    MetricsMixin,
    SnapshotMixin,
    JourneyMixin,
    TimelineMixin,
    ArtifactMixin,
):
    """Web 任务生命周期管理器。"""

    def __init__(
        self,
        base_dir: Path,
        settings: Settings | None = None,
        *,
        max_concurrent: int | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._base_dir = Path(base_dir).expanduser().resolve()
        self._base_dir.mkdir(parents=True, exist_ok=True)
        (self._base_dir / "logs").mkdir(parents=True, exist_ok=True)
        concurrency = (
            max_concurrent
            if max_concurrent is not None
            else int(self._settings.web_max_concurrent_tasks)
        )
        self._semaphore = asyncio.Semaphore(max(1, concurrency))
        self._tasks: dict[str, WebTask] = {}
        self._closing = False
        self._started_at = datetime.now(timezone.utc)
        self._persistent_root = self._base_dir / "persistent"
        self._persistent_root.mkdir(parents=True, exist_ok=True)
        self._scheduler: PeriodicTaskPoller | None = None
        self._scheduler_runner: asyncio.Task[None] | None = None
        self._outbox_worker: NotificationOutboxWorker | None = None
        self._store = WebTaskStore(self._base_dir / "web_tasks.db")
        results_dir = self._base_dir / self._settings.web_results_dir
        self._keep_history = bool(self._settings.web_keep_history)
        self._archive = FileResultArchive(
            results_dir,
            keep_history=self._keep_history,
            history=(
                JsonlHistoryStore(results_dir / "history.jsonl")
                if self._keep_history
                else None
            ),
        )
        self._pending_recovery: list[str] = []
        resume_enabled = bool(self._settings.web_resume_interrupted_tasks)
        self._store.mark_interrupted(
            running_reason=(
                "服务重启，任务自动恢复次数已用尽；请重新提交。"
                if resume_enabled
                else "服务重启，任务被中断；请重新提交。"
            ),
            human_reason=(
                "服务重启导致运行时状态丢失，人工指导无法恢复；请重新提交。"
            ),
            provisional_reason=(
                "服务重启，后台评审被中断；此前交付的结果已保留。"
            ),
            max_resumes=1 if resume_enabled else None,
        )
        if resume_enabled:
            try:
                self._pending_recovery = self._store.recover_interrupted(
                    max_resumes=1
                )
            except Exception as exc:
                logger.warning("Web 任务自动恢复排队失败：%s", exc)
        self._cleanup_retention()
        for row in self._store.list_recent(MAX_TASKS_IN_MEMORY):
            task = self._task_from_record(row)
            self._tasks[task.task_id] = task

    @property
    def base_dir(self) -> Path:
        """Web 任务沙箱根目录。"""
        return self._base_dir

    # ------------------------------------------------------------------
    # 提交与执行
    # ------------------------------------------------------------------
    def submit(
        self,
        objective: str,
        constraints: str = "",
        acceptance_criteria: str = "",
        *,
        lang: str = "zh",
    ) -> WebTask:
        """提交一个 Web 任务并立即返回记录（执行在后台进行）。"""
        if self._closing:
            raise RuntimeError(
                tr(
                    lang,
                    "Web 控制台正在关闭，不接受新任务",
                    "Web console is shutting down; new tasks are rejected",
                )
            )
        objective = objective.strip()
        if not objective:
            raise ValueError(
                tr(lang, "任务目标不能为空", "Task objective must not be empty")
            )
        if len(objective) > MAX_OBJECTIVE_CHARS:
            raise ValueError(
                tr(
                    lang,
                    f"任务目标超过 {MAX_OBJECTIVE_CHARS} 字符上限",
                    f"Task objective exceeds the {MAX_OBJECTIVE_CHARS}-character limit",
                )
            )
        if len(constraints) > MAX_OBJECTIVE_CHARS:
            raise ValueError(
                tr(
                    lang,
                    f"约束条件超过 {MAX_OBJECTIVE_CHARS} 字符上限",
                    f"Constraints exceed the {MAX_OBJECTIVE_CHARS}-character limit",
                )
            )
        if len(acceptance_criteria) > MAX_OBJECTIVE_CHARS:
            raise ValueError(
                tr(
                    lang,
                    f"验收标准超过 {MAX_OBJECTIVE_CHARS} 字符上限",
                    f"Acceptance criteria exceed the {MAX_OBJECTIVE_CHARS}-character limit",
                )
            )
        self._prune_finished()
        self._prepare_new_run()
        if len(self._tasks) >= MAX_TASKS_IN_MEMORY:
            raise RuntimeError(
                tr(
                    lang,
                    f"任务数量已达上限（{MAX_TASKS_IN_MEMORY}），请先清理已完成任务",
                    f"Task limit reached ({MAX_TASKS_IN_MEMORY}); "
                    "clear finished tasks first",
                )
            )

        task_id = uuid.uuid4().hex
        sandbox_dir = self._base_dir / task_id
        sandbox_dir.mkdir(parents=True, exist_ok=True)
        context = ProjectContext(
            sandbox_dir=sandbox_dir,
            project_name=f"web-{task_id[:8]}",
            task_store_dir=self._persistent_root,
            config=self._settings,
        )
        task = WebTask(
            task_id=task_id,
            objective=objective,
            constraints=constraints.strip(),
            acceptance_criteria=acceptance_criteria.strip(),
            sandbox_dir=sandbox_dir,
            context=context,
        )
        self._tasks[task_id] = task
        try:
            self._store.create(task)
        except Exception as exc:
            logger.warning("Web 任务记录写入失败：%s（%s）", task_id, exc)
        task.runner = asyncio.create_task(
            self._execute(task), name=f"web-task-{task_id[:8]}"
        )
        logger.info("Web 任务已提交：%s", task_id)
        return task

    async def _execute(self, task: WebTask) -> None:
        """在信号量约束下执行任务，并把终态写回记录。"""
        try:
            async with self._semaphore:
                if task.cancel_requested:
                    self._mark_cancelled(task)
                    return
                task.status = "running"
                task.started_at = _utc_now_iso()
                task.error = None
                self._persist_task_record(task)
                logger.info("Web 任务开始执行：%s", task.task_id)

                router = TaskRouter(
                    task.context,
                    self._settings,
                    log_file=self._base_dir / "logs" / f"{task.task_id}.log",
                )
                try:
                    classification = await router.classify(
                        task.objective, task.constraints, task.acceptance_criteria
                    )
                except ConfigLoadError as exc:
                    task.status = "failed"
                    task.error = f"配置错误：{exc}"
                    return
                task.classification = classification

                if task.cancel_requested:
                    self._mark_cancelled(task)
                    return

                if classification.task_type == TaskType.FULL_WORKFLOW:
                    outcome = await maybe_run_single_file_fast_path(
                        task.context,
                        self._settings,
                        task.objective,
                        task.constraints,
                        task.acceptance_criteria,
                        task_id=task.task_id,
                    )
                    if outcome.applied and outcome.output is not None:
                        classification.steps.append(
                            RouteStep(
                                stage="fast_path",
                                status="hit",
                                detail=(
                                    f"单文件纯写入快通道：{outcome.target}"
                                ),
                            )
                        )
                        output = outcome.output
                    else:
                        if outcome.reason not in {
                            "disabled",
                            "not_applicable",
                        }:
                            classification.steps.append(
                                RouteStep(
                                    stage="fast_path",
                                    status="fallback",
                                    detail=(
                                        f"快通道未完成（{outcome.reason}），"
                                        "回退完整工作流"
                                    ),
                                )
                            )
                        engine = WorkflowEngine(
                            task.context,
                            settings=self._settings,
                            log_file=(
                                self._base_dir
                                / "logs"
                                / f"{task.task_id}.log"
                            ),
                        )
                        task.engine = engine
                        from so_agent.runtime.orchestration_runtime import (
                            create_orchestration_runtime,
                        )

                        runtime = create_orchestration_runtime(
                            engine, self._settings
                        )
                        output = await runtime.run(
                            task.objective,
                            task.constraints,
                            task.acceptance_criteria,
                        )
                else:
                    output = await router.route(
                        task.objective,
                        task.constraints,
                        task.acceptance_criteria,
                        classification=classification,
                    )
                task.output = output
                task.status = output.status.value
                # 异步 provisional：先持久化初步结果供前端展示，再等待
                # 后台评审收口（不阻塞用户看到结果）
                if (
                    task.engine is not None
                    and output.status is TaskStatus.COMPLETED_PROVISIONAL
                ):
                    self._persist_task_record(task, with_snapshots=True)
                    final = await task.engine.wait_for_provisional()
                    if (
                        final is not None
                        and not task.cancel_requested
                        and not self._closing
                    ):
                        task.output = final
                        task.status = final.status.value
                        logger.info(
                            "Web 任务异步评审完成：%s（%s）",
                            task.task_id,
                            task.status,
                        )
                logger.info(
                    "Web 任务结束：%s（%s）", task.task_id, task.status
                )
        except asyncio.CancelledError:
            self._mark_cancelled(task)
            raise
        except ConfigLoadError as exc:
            task.status = "failed"
            task.error = f"配置错误：{exc}"
        except Exception as exc:  # 任务失败不得影响其他任务
            task.status = "failed"
            task.error = f"{type(exc).__name__}: {exc}"
            logger.exception("Web 任务异常：%s", task.task_id)
        finally:
            task.finished_at = _utc_now_iso()
            self._persist_task_record(task, with_snapshots=True)
            self._publish_result(task)
            try:
                self._cleanup_retention()
            except Exception as exc:
                logger.warning("Web 任务保留清理失败：%s", exc)

    @staticmethod
    def _mark_cancelled(task: WebTask) -> None:
        task.cancel_requested = True
        task.status = "cancelled"
        if task.output is None:
            task.error = task.error or "任务已取消"

    # ------------------------------------------------------------------
    # 人工指导恢复
    # ------------------------------------------------------------------
    def submit_guidance(
        self, task_id: str, guidance: HumanGuidance, *, lang: str = "zh"
    ) -> bool:
        """提交人工指导并在后台恢复执行；非法状态抛 GuidanceStateError。"""
        task = self._tasks.get(task_id)
        if task is None:
            return False
        if task.status != "human_required" or task.engine is None:
            raise GuidanceStateError(
                tr(
                    lang,
                    f"任务当前状态为 {task.status!r}，仅 human_required 可提交指导",
                    f"Task is in state {task.status!r}; "
                    "guidance is only accepted in human_required",
                )
            )
        if task.runner is not None and not task.runner.done():
            raise GuidanceStateError(
                tr(
                    lang,
                    "任务仍在运行，无法提交指导",
                    "Task is still running; guidance cannot be submitted",
                )
            )
        if self._closing:
            raise GuidanceStateError(
                tr(
                    lang,
                    "Web 控制台正在关闭，不接受新动作",
                    "Web console is shutting down; no new actions accepted",
                )
            )
        task.last_guidance = guidance.model_copy(deep=True)
        task.error = None
        task.status = "running"
        task.cancel_requested = False
        task.finished_at = None
        self._persist_task_record(task)
        try:
            self._store.clear_snapshots(task_id)
        except Exception as exc:
            logger.warning("Web 任务快照清理失败：%s（%s）", task_id, exc)
        task.runner = asyncio.create_task(
            self._resume_with_guidance(task, guidance),
            name=f"web-task-resume-{task_id[:8]}",
        )
        logger.info("Web 任务收到人工指导：%s（%s）", task_id, guidance.action)
        return True

    async def _resume_with_guidance(
        self, task: WebTask, guidance: HumanGuidance
    ) -> None:
        """在信号量约束下应用指导并恢复执行，终态写回任务记录。"""
        try:
            async with self._semaphore:
                engine = task.engine
                assert engine is not None
                output = await engine.resume_with_guidance(guidance)
                task.output = output
                task.status = output.status.value
                logger.info(
                    "Web 任务恢复结束：%s（%s）", task.task_id, task.status
                )
        except asyncio.CancelledError:
            self._mark_cancelled(task)
            raise
        except HumanGuidanceError as exc:
            task.status = "human_required"
            task.error = f"人工指导无效：{exc}"
            logger.warning("Web 任务指导无效：%s（%s）", task.task_id, exc)
        except Exception as exc:  # 恢复失败不得影响其他任务
            task.status = "failed"
            task.error = f"{type(exc).__name__}: {exc}"
            logger.exception("Web 任务恢复异常：%s", task.task_id)
        finally:
            task.finished_at = _utc_now_iso()
            self._persist_task_record(task, with_snapshots=True)
            self._publish_result(task)

    def resume_interrupted(self) -> list[str]:
        """把启动时重新排队的任务交给事件循环执行。

        必须在运行中的事件循环内调用（Web lifespan 启动阶段）。任务从
        零开始重新执行：主管与评审的运行时状态无法序列化，但目标、
        约束、验收标准与沙箱产物均已持久化或保留在任务目录中。
        """
        if self._closing:
            return []
        pending = self._pending_recovery
        self._pending_recovery = []
        started: list[str] = []
        for task_id in pending:
            task = self._tasks.get(task_id)
            if task is None or task.runner is not None:
                continue
            task.restored = False
            task.restored_snapshot.clear()
            task.restored_tokens = {}
            task.output = None
            task.classification = None
            task.status = "queued"
            task.started_at = None
            task.finished_at = None
            task.cancel_requested = False
            task.runner = asyncio.create_task(
                self._execute(task),
                name=f"web-task-recover-{task_id[:8]}",
            )
            started.append(task_id)
        if started:
            logger.info("Web 任务自动恢复启动：%s", started)
        return started

    # ------------------------------------------------------------------
    # 持久化任务调度器
    # ------------------------------------------------------------------
    def start_scheduler(self) -> None:
        """启动 Web 内置持久化任务调度器（随应用生命周期启停）。"""
        if self._closing or not self._settings.web_scheduler_enabled:
            return
        if (
            self._scheduler_runner is not None
            and not self._scheduler_runner.done()
        ):
            return
        context = ProjectContext(
            sandbox_dir=self._persistent_root,
            project_name="web-persistent",
            task_store_dir=self._persistent_root,
            config=self._settings,
        )
        store_path = resolve_task_store_path(context, self._settings)
        self._scheduler = PeriodicTaskPoller(
            TaskStore(store_path), context, self._settings
        )
        self._scheduler_runner = asyncio.create_task(
            self._scheduler.start(), name="web-persistent-scheduler"
        )
        logger.info("Web 内置持久化任务调度器已启动：%s", store_path)

    async def _stop_scheduler(self) -> None:
        if self._scheduler is None:
            return
        self._scheduler.stop()
        runner = self._scheduler_runner
        if runner is not None and not runner.done():
            try:
                await asyncio.wait_for(runner, timeout=15)
            except asyncio.TimeoutError:
                runner.cancel()
                await asyncio.gather(runner, return_exceptions=True)
            except Exception:
                pass
        self._scheduler = None
        self._scheduler_runner = None

    def scheduler_running(self) -> bool:
        return (
            self._scheduler_runner is not None
            and not self._scheduler_runner.done()
        )

    # ------------------------------------------------------------------
    # 人工升级通知 outbox
    # ------------------------------------------------------------------
    async def start_notification_outbox(self) -> None:
        """启动后台通知排空任务（仅在启用 outbox 且配置 Webhook 时）。"""
        if self._closing or not self._settings.notification_outbox_enabled:
            return
        if not (self._settings.escalation_webhook_url or "").strip():
            return
        if self._outbox_worker is not None and self._outbox_worker.running:
            return
        try:
            worker = NotificationOutboxWorker(self._settings)
            await worker.start()
        except Exception as exc:  # outbox 失败不阻塞控制台
            logger.warning("通知 outbox 启动失败：%s", exc)
            return
        self._outbox_worker = worker

    async def _stop_notification_outbox(self) -> None:
        worker = self._outbox_worker
        self._outbox_worker = None
        if worker is not None:
            await worker.stop()

    def notification_outbox_stats(self) -> dict[str, Any]:
        """返回 outbox 队列健康指标；未启用时仅返回 enabled=False。"""
        if not self._settings.notification_outbox_enabled:
            return {"enabled": False}
        path = resolve_outbox_path(self._settings)
        if path is None:
            return {"enabled": True, "error": "notification_outbox_path 为空"}
        try:
            return NotificationOutbox(path).stats()
        except Exception as exc:
            return {"enabled": True, "error": str(exc)[:200]}

    # ------------------------------------------------------------------
    # 持久化任务查询（逐次运行可见性）
    # ------------------------------------------------------------------




    # ------------------------------------------------------------------
    # 取消与关闭
    # ------------------------------------------------------------------
    def cancel(self, task_id: str) -> bool:
        """请求取消任务；终态任务或不存在时返回 False。"""
        task = self._tasks.get(task_id)
        if task is None or task.status in TERMINAL_STATUSES:
            return False
        task.cancel_requested = True
        if task.engine is not None:
            task.engine.cancel()
        if task.runner is not None and not task.runner.done():
            task.runner.cancel()
        logger.info("Web 任务收到取消请求：%s", task_id)
        return True

    async def shutdown(self) -> None:
        """停止调度器、取消全部在跑任务并等待清理完成。"""
        self._closing = True
        await self._stop_notification_outbox()
        await self._stop_scheduler()
        runners: list[asyncio.Task[None]] = []
        for task in self._tasks.values():
            if task.runner is not None and not task.runner.done():
                if task.engine is not None:
                    task.engine.cancel()
                task.runner.cancel()
            if task.runner is not None:
                runners.append(task.runner)
        if runners:
            await asyncio.gather(*runners, return_exceptions=True)
        try:
            self._cleanup_retention()
        except Exception as exc:  # 清理失败不得阻塞退出
            logger.warning("Web 任务保留清理失败：%s", exc)

    # ------------------------------------------------------------------
    # 持久化与保留
    # ------------------------------------------------------------------
    def _prepare_new_run(self) -> None:
        """新任务提交时清理上一次结果（默认不保留历史）。

        ``human_required`` 任务保留，等待人工处置；运行中的任务不受影响。
        """
        self._archive.begin_run()
        if self._keep_history:
            return
        removable = [
            task
            for task in list(self._tasks.values())
            if task.status in {"completed", "failed", "cancelled"}
            and (task.runner is None or task.runner.done())
        ]
        for task in removable:
            self._tasks.pop(task.task_id, None)
            self._delete_task_files(task.task_id)
            try:
                self._store.delete(task.task_id)
            except Exception as exc:  # 单条清理失败不阻塞新任务
                logger.warning(
                    "Web 任务记录清理失败：%s（%s）", task.task_id, exc
                )
        if removable:
            logger.info("Web 已清理上一次任务结果：%d 个", len(removable))

    def _publish_result(self, task: WebTask) -> None:
        try:
            self._archive.publish(task.task_id, self._result_payload(task))
        except Exception as exc:
            logger.warning("Web 结果沉淀失败：%s（%s）", task.task_id, exc)

    def _result_payload(self, task: WebTask) -> dict[str, Any]:
        output = task.output
        tokens = self._task_token_totals(task)
        plan: list[dict[str, Any]] = []
        if output is not None and output.plan_versions:
            try:
                latest_status = {
                    result.subtask_id: result.success
                    for result in (
                        task.engine.results.get(task.engine.plan_version, [])
                        if task.engine is not None
                        else []
                    )
                }
            except Exception:
                latest_status = {}
            for spec in output.plan_versions[-1]:
                plan.append(
                    {
                        "subtask_id": spec.subtask_id,
                        "title": spec.title,
                        "status": (
                            "completed"
                            if latest_status.get(spec.subtask_id)
                            else "unknown"
                        ),
                    }
                )
        payload: dict[str, Any] = {
            "task_id": task.task_id,
            "objective": task.objective,
            "constraints": task.constraints,
            "acceptance_criteria": task.acceptance_criteria,
            "status": task.status,
            "task_type": (
                task.classification.task_type.value
                if task.classification is not None
                else ""
            ),
            "created_at": task.created_at,
            "started_at": task.started_at,
            "finished_at": task.finished_at,
            "error": task.error,
            "final_result": output.final_result if output is not None else "",
            "review_summary": output.review_summary if output is not None else "",
            "notice": output.notice if output is not None else "",
            "previous_result": (
                output.previous_result if output is not None else ""
            ),
            "provisional": bool(output.provisional) if output is not None else False,
            "total_tokens": tokens.get("total_tokens", 0),
            "tokens_by_agent": tokens.get("by_agent", {}),
            "plan": plan,
            "audit": (
                output.audit.model_dump(mode="json")
                if output is not None and output.audit is not None
                else None
            ),
            "human_escalation": (
                output.human_escalation.model_dump(mode="json")
                if output is not None and output.human_escalation is not None
                else None
            ),
            "artifacts": self._scan_artifacts(task),
        }
        return payload

    def latest_result(self) -> dict[str, Any] | None:
        """返回最近一次沉淀的结果载荷（未生成时为 None）。"""
        return self._archive.latest()

    def latest_result_markdown(self) -> str | None:
        return self._archive.latest_markdown()

    def _persist_task_record(
        self, task: WebTask, *, with_snapshots: bool = False
    ) -> None:
        try:
            if self._store.save(task) == 0:
                self._store.create(task)
        except Exception as exc:
            logger.warning("Web 任务记录写入失败：%s（%s）", task.task_id, exc)
            return
        if not with_snapshots:
            return
        snapshots: dict[str, dict[str, Any]] = {}
        for lang in ("zh", "en"):
            try:
                snapshot = self.snapshot(task.task_id, lang)
            except Exception as exc:
                logger.warning(
                    "Web 任务快照生成失败：%s（%s，%s）",
                    task.task_id,
                    lang,
                    exc,
                )
                continue
            if snapshot is not None:
                snapshots[lang] = snapshot
        try:
            self._store.save_snapshots(
                task.task_id, snapshots, self._task_token_totals(task)
            )
        except Exception as exc:
            logger.warning("Web 任务快照写入失败：%s（%s）", task.task_id, exc)

    @staticmethod
    def _task_token_totals(task: WebTask) -> dict[str, Any]:
        total = 0
        input_total = 0
        cached = 0
        cache_write = 0
        per_agent: dict[str, int] = {}
        for record in task.context.event_log:
            if record.get("event_type") != "token_usage":
                continue
            payload = record.get("payload") or {}
            tokens = int(payload.get("total_tokens") or 0)
            total += tokens
            input_total += int(payload.get("input_tokens") or 0)
            cached += int(payload.get("cached_tokens") or 0)
            cache_write += int(payload.get("cache_write_tokens") or 0)
            agent = str(payload.get("agent") or "unknown")
            per_agent[agent] = per_agent.get(agent, 0) + tokens
        if total == 0 and task.engine is not None:
            try:
                snapshot = task.engine.token_budget_snapshot() or {}
            except Exception:
                snapshot = {}
            total = int(snapshot.get("total_tokens") or 0)
            per_agent = {
                str(agent): int(value or 0)
                for agent, value in (snapshot.get("per_agent") or {}).items()
            }
        # 事件计数随 token 统计一并落库：重启后内存事件日志为空，
        # 指标聚合需要从持久化快照补齐，避免工具/评审计数归零。
        return {
            "total_tokens": total,
            "input_tokens": input_total,
            "cached_tokens": cached,
            "cache_write_tokens": cache_write,
            "by_agent": per_agent,
            "events": aggregate_events([task.context.event_log]),
        }

    def _task_from_record(self, row: dict[str, Any]) -> WebTask:
        task_id = str(row["task_id"])
        sandbox_dir = self._base_dir / task_id
        context = ProjectContext(
            sandbox_dir=sandbox_dir,
            project_name=f"web-{task_id[:8]}",
            task_store_dir=self._persistent_root,
            config=self._settings,
        )
        task = WebTask(
            task_id=task_id,
            objective=str(row.get("objective") or ""),
            constraints=str(row.get("constraints") or ""),
            acceptance_criteria=str(row.get("acceptance_criteria") or ""),
            sandbox_dir=sandbox_dir,
            context=context,
            status=str(row.get("status") or "failed"),
            created_at=str(row.get("created_at") or _utc_now_iso()),
            started_at=row.get("started_at"),
            finished_at=row.get("finished_at"),
            error=row.get("error"),
            cancel_requested=bool(row.get("cancel_requested")),
            restored=True,
        )
        classification_json = row.get("classification_json")
        if classification_json:
            try:
                task.classification = ClassificationResult.model_validate_json(
                    classification_json
                )
            except Exception:
                pass
        output_json = row.get("output_json")
        if output_json:
            try:
                task.output = OrchestratorOutput.model_validate_json(
                    output_json
                )
            except Exception:
                pass
        guidance_json = row.get("last_guidance_json")
        if guidance_json:
            try:
                task.last_guidance = HumanGuidance.model_validate_json(
                    guidance_json
                )
            except Exception:
                pass
        for column, lang in (("snapshot_zh", "zh"), ("snapshot_en", "en")):
            raw = row.get(column)
            if not raw:
                continue
            try:
                task.restored_snapshot[lang] = json.loads(raw)
            except (TypeError, ValueError):
                continue
        tokens_json = row.get("tokens_json")
        if tokens_json:
            try:
                task.restored_tokens = json.loads(tokens_json)
            except (TypeError, ValueError):
                task.restored_tokens = {}
        return task


    def _cleanup_retention(self) -> None:
        days = int(self._settings.web_task_retention_days)
        max_records = int(self._settings.web_max_retained_tasks)
        try:
            pruned = self._store.prune(
                retention_days=days, max_records=max_records
            )
        except Exception as exc:
            logger.warning("Web 任务保留清理失败：%s", exc)
            return
        for task_id in pruned:
            self._delete_task_files(task_id)
        if days > 0:
            cutoff = datetime.now(timezone.utc) - timedelta(days=days)
            self._cleanup_orphans(cutoff)
        if pruned:
            logger.info("Web 任务保留清理完成：删除 %d 个任务", len(pruned))

    def _delete_task_files(self, task_id: str) -> None:
        if not _TASK_ID_PATTERN.match(task_id):
            return
        path = self._base_dir / task_id
        if path.is_symlink():
            try:
                path.unlink()
            except OSError:
                pass
        elif path.is_dir():
            try:
                resolved = path.resolve()
            except OSError:
                return
            if resolved.parent != self._base_dir:
                return
            shutil.rmtree(resolved, ignore_errors=True)
        log_path = self._base_dir / "logs" / f"{task_id}.log"
        try:
            log_path.unlink(missing_ok=True)
        except OSError:
            pass

    def _cleanup_orphans(self, cutoff: datetime) -> None:
        try:
            known = self._store.all_ids()
        except Exception:
            known = set()
        try:
            entries = list(self._base_dir.iterdir())
        except OSError:
            entries = []
        for child in entries:
            if (
                not child.is_dir()
                or child.is_symlink()
                or not _TASK_ID_PATTERN.match(child.name)
                or child.name in known
            ):
                continue
            try:
                modified = datetime.fromtimestamp(
                    child.stat().st_mtime, tz=timezone.utc
                )
            except OSError:
                continue
            if modified < cutoff:
                self._delete_task_files(child.name)
        log_dir = self._base_dir / "logs"
        if not log_dir.is_dir():
            return
        for log_path in log_dir.glob("*.log"):
            if not _TASK_ID_PATTERN.match(log_path.stem):
                continue
            if log_path.stem in known:
                continue
            try:
                modified = datetime.fromtimestamp(
                    log_path.stat().st_mtime, tz=timezone.utc
                )
            except OSError:
                continue
            if modified >= cutoff:
                continue
            try:
                log_path.unlink()
            except OSError:
                pass

    # ------------------------------------------------------------------
    # 查询
    # ------------------------------------------------------------------
    def get_task(self, task_id: str) -> WebTask | None:
        task = self._tasks.get(task_id)
        if task is not None:
            return task
        try:
            row = self._store.get(task_id)
        except Exception:
            row = None
        return self._task_from_record(row) if row is not None else None

    def list_tasks(self) -> list[dict[str, Any]]:
        views: dict[str, dict[str, Any]] = {}
        for task in self._tasks.values():
            views[task.task_id] = self._task_summary(task)
        try:
            limit = max(
                MAX_TASKS_IN_MEMORY,
                int(self._settings.web_max_retained_tasks) or 0,
            )
            rows = self._store.list_recent(limit)
        except Exception:
            rows = []
        for row in rows:
            task_id = str(row.get("task_id") or "")
            if not task_id or task_id in views:
                continue
            views[task_id] = {
                "task_id": task_id,
                "objective": _preview(str(row.get("objective") or ""), 120),
                "status": str(row.get("status") or "failed"),
                "created_at": str(row.get("created_at") or ""),
                "finished_at": row.get("finished_at"),
                "task_type": row.get("task_type"),
            }
        return sorted(
            views.values(), key=lambda item: item["created_at"], reverse=True
        )

    @staticmethod
    def _task_summary(task: WebTask) -> dict[str, Any]:
        return {
            "task_id": task.task_id,
            "objective": _preview(task.objective, 120),
            "status": task.status,
            "created_at": task.created_at,
            "finished_at": task.finished_at,
            "task_type": (
                task.classification.task_type.value
                if task.classification is not None
                else None
            ),
        }



    # ------------------------------------------------------------------
    # 快照子结构
    # ------------------------------------------------------------------






    # ------------------------------------------------------------------
    # 全链路（路由判断 → 分支 → 工作流 → 工具 / Agent）
    # ------------------------------------------------------------------












    # ------------------------------------------------------------------
    # 产物
    # ------------------------------------------------------------------




    def _prune_finished(self) -> None:
        """内存满时回收终态任务，保留低水位余量；持久化记录不受影响。"""
        if len(self._tasks) < MAX_TASKS_IN_MEMORY:
            return
        target = max(0, MAX_TASKS_IN_MEMORY - 20)
        removable = [
            task
            for task in self._tasks.values()
            if task.status in TERMINAL_STATUSES
            and (task.runner is None or task.runner.done())
        ]
        excess = len(self._tasks) - target
        for task in sorted(
            removable, key=lambda item: item.created_at
        )[:excess]:
            self._tasks.pop(task.task_id, None)


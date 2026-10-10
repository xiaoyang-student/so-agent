"""长任务调度器（PeriodicTaskPoller）。

异步守护进程，持续轮询 TaskStore 中到期的持久化任务并触发执行。
支持：
- 定时轮询（可配置间隔，见 ``Settings.scheduler_poll_interval``）
- 周期任务自动重调度
- 异常时保存检查点并暂停
- 优雅停机（``stop()`` 置位运行标志，下一轮轮询后退出）

执行动作说明：当前实现用一次 LLM 调用执行 ``schedule_spec.action``，
后续可扩展为沙箱代码执行、文件系统操作等。
"""

from __future__ import annotations

import asyncio
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from so_agent.config import Settings
from so_agent.context import ProjectContext
from so_agent.models import ScheduleSpec
from so_agent.persistence import TaskRecord, TaskStore


def _compute_next_run(spec: ScheduleSpec) -> str | None:
    """根据调度规格计算下次执行时间（ISO 格式）。"""
    now = datetime.now(timezone.utc)
    if not spec.is_recurring:
        return None
    if spec.trigger == "interval" and spec.interval_seconds:
        return (now + timedelta(seconds=spec.interval_seconds)).isoformat()
    if spec.trigger == "cron" and spec.cron:
        # 简化实现：cron 解析较复杂，此处用固定间隔 24h 近似
        # 后续可引入 croniter 库做精确解析
        return (now + timedelta(hours=24)).isoformat()
    return None


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

    async def start(self) -> None:
        """启动调度循环（阻塞运行直到 stop()）。"""
        self._running = True
        print(
            f"[so-agent] 调度器已启动，轮询间隔 {self._poll_interval}s",
            file=sys.stderr,
            flush=True,
        )
        while self._running:
            try:
                due_tasks = self._store.get_due_tasks()
                for task in due_tasks:
                    asyncio.create_task(self._execute_task(task))
            except Exception as exc:
                print(
                    f"[so-agent] 调度器轮询异常：{exc}",
                    file=sys.stderr,
                    flush=True,
                )
            await asyncio.sleep(self._poll_interval)

    async def _execute_task(self, task: TaskRecord) -> None:
        """执行单个到期任务（含检查点恢复）。"""
        print(
            f"[so-agent] 调度执行任务 {task.task_id}：{task.objective[:50]}",
            file=sys.stderr,
            flush=True,
        )
        checkpoint = self._store.load_checkpoint(task.task_id)
        try:
            self._store.update_status(task.task_id, "running")
            self._store.add_log(
                task.task_id,
                "execution_start",
                {"checkpoint": checkpoint is not None},
            )

            # 根据任务执行动作
            result = await self._run_task_action(task, checkpoint)

            self._store.update_status(
                task.task_id,
                "completed",
                result=result,
            )
            self._store.add_log(task.task_id, "execution_complete")

            # 周期任务：计算下次执行并重调度
            if task.schedule_spec and task.schedule_spec.is_recurring:
                next_run = _compute_next_run(task.schedule_spec)
                if next_run:
                    self._store.reschedule(task.task_id, next_run)
                    print(
                        f"[so-agent] 周期任务 {task.task_id} 下次执行：{next_run}",
                        file=sys.stderr,
                        flush=True,
                    )

        except Exception as exc:
            # 异常时保存检查点并暂停
            self._store.save_checkpoint(
                task.task_id,
                {
                    "error": str(exc),
                    "partial_state": checkpoint,
                },
            )
            self._store.update_status(
                task.task_id,
                "paused",
                error=str(exc),
            )
            self._store.add_log(
                task.task_id,
                "execution_error",
                {"error": str(exc)},
            )
            print(
                f"[so-agent] 任务 {task.task_id} 执行失败，已暂停：{exc}",
                file=sys.stderr,
                flush=True,
            )

    async def _run_task_action(
        self, task: TaskRecord, checkpoint: dict[str, Any] | None
    ) -> str:
        """执行任务的实际动作。

        当前实现：对于简单+长任务，用一次 LLM 调用执行 action。
        后续可扩展为：沙箱代码执行、文件系统操作等。

        异常处理约定：此处不吞异常，调用失败（含网络/服务异常）
        直接向上抛给 ``_execute_task``，由上层统一保存检查点并把
        任务状态置为 ``paused``。客户端在 finally 中关闭，避免
        连接泄漏。HTTP 超时经 ``resolve_model_http_timeout`` 解析
        （任务 59，与 orchestrator / router 客户端一致）：分维度
        未配置时等价旧单一 float timeout，非法配置在构造客户端前
        fail fast。
        """
        # 延迟导入重依赖：避免模块导入阶段即加载 openai 客户端
        from openai import AsyncOpenAI

        from so_agent.runtime.config_loader import (
            load_agent_api_config,
            resolve_model_http_timeout,
        )

        api_config_path = Path(__file__).resolve().parent / "api_config.yaml"
        api_cfg = load_agent_api_config(api_config_path, resolve_secrets=True)
        http_timeout = resolve_model_http_timeout(api_cfg)
        client = AsyncOpenAI(
            api_key=api_cfg.resolve_api_key(),
            base_url=api_cfg.base_url or None,
            timeout=http_timeout,
            max_retries=api_cfg.retry,
        )

        action = ""
        if task.schedule_spec:
            action = task.schedule_spec.action or task.objective

        prompt = f"执行以下任务：{action}"
        if checkpoint:
            prompt += f"\n\n上次执行的检查点信息：{json.dumps(checkpoint, ensure_ascii=False)}"

        try:
            response = await client.chat.completions.create(
                model=api_cfg.model,
                messages=[{"role": "user", "content": prompt}],
                timeout=http_timeout,
            )
            return response.choices[0].message.content or ""
        finally:
            await client.close()

    def stop(self) -> None:
        """停止调度循环。"""
        self._running = False
        print("[so-agent] 调度器已停止", file=sys.stderr, flush=True)

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

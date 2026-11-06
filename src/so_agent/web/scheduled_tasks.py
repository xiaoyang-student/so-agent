"""持久化调度任务：只读列表/详情与运行历史摘要。"""

from __future__ import annotations

from so_agent.context import ProjectContext
from so_agent.persistence import TaskRecord
from so_agent.persistence import TaskStore
from so_agent.persistence import resolve_task_store_path
from typing import Any

from so_agent.web.common import (
    _preview,
    logger,
)


class ScheduledTaskMixin:

    def _open_scheduled_store(self) -> TaskStore | None:
        """以只读意图打开调度任务库；库尚不存在时返回 None。"""
        context = ProjectContext(
            sandbox_dir=self._persistent_root,
            project_name="web-persistent",
            task_store_dir=self._persistent_root,
            config=self._settings,
        )
        path = resolve_task_store_path(context, self._settings)
        if not path.is_file():
            return None
        return TaskStore(path)


    def list_scheduled_tasks(self) -> list[dict[str, Any]]:
        """列出所有持久化调度任务及其最近一次运行。"""
        store = self._open_scheduled_store()
        if store is None:
            return []
        try:
            records = [
                record
                for record in store.list_tasks()
                if record.schedule_spec is not None
            ]
            records.sort(
                key=lambda record: (
                    0 if record.status == "scheduled" else 1,
                    record.next_run_at or "",
                    record.updated_at or "",
                )
            )
            views: list[dict[str, Any]] = []
            for record in records[:200]:
                runs = store.list_runs(record.task_id, limit=1)
                views.append(
                    self._scheduled_summary(record, runs[0] if runs else None)
                )
            return views
        except Exception as exc:
            logger.warning("读取调度任务失败：%s", exc)
            return []
        finally:
            store.close()


    def get_scheduled_task(self, task_id: str) -> dict[str, Any] | None:
        """返回调度任务详情与最近运行历史。"""
        store = self._open_scheduled_store()
        if store is None:
            return None
        try:
            record = store.get_task(task_id)
            if record is None or record.schedule_spec is None:
                return None
            runs = store.list_runs(task_id, limit=50)
            view = self._scheduled_summary(record, runs[0] if runs else None)
            view["runs"] = runs
            return view
        except Exception as exc:
            logger.warning("读取调度任务详情失败：%s", exc)
            return None
        finally:
            store.close()


    @staticmethod
    def _scheduled_summary(
        record: TaskRecord, latest_run: dict[str, Any] | None
    ) -> dict[str, Any]:
        spec = record.schedule_spec
        schedule = None
        if spec is not None:
            schedule = {
                "trigger": spec.trigger,
                "cron": spec.cron,
                "interval_seconds": spec.interval_seconds,
                "timezone": spec.timezone,
                "summary": spec.summary,
                "action": spec.action,
                "is_recurring": spec.is_recurring,
            }
        return {
            "task_id": record.task_id,
            "objective": _preview(record.objective, 160),
            "status": record.status,
            "task_type": record.task_type.value,
            "schedule": schedule,
            "next_run_at": record.next_run_at,
            "last_run_at": record.last_run_at,
            "consecutive_failures": record.consecutive_failures,
            "created_at": record.created_at,
            "updated_at": record.updated_at,
            "latest_run": latest_run,
        }


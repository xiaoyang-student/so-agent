"""调度语义：时区、错过策略、失败重试与运行历史。"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from so_agent.config import Settings
from so_agent.context import ProjectContext
from so_agent.models import ScheduleSpec, TaskType
from so_agent.persistence import TaskStore
from so_agent.task_scheduler import (
    PeriodicTaskPoller,
    _compute_next_run,
    _retry_backoff_seconds,
)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


@pytest.fixture
def store(tmp_path: Path) -> TaskStore:
    store = TaskStore(tmp_path / "tasks.db")
    yield store
    store.close()


def _poller(store: TaskStore, tmp_path: Path, **settings) -> PeriodicTaskPoller:
    values = {"_env_file": None}
    values.update(settings)
    return PeriodicTaskPoller(
        store,
        ProjectContext(sandbox_dir=tmp_path, project_name="sched"),
        Settings(**values),
    )


def _recurring_task(store: TaskStore, spec: ScheduleSpec, next_run_at: str):
    return store.create_task(
        objective="周期检查",
        task_type=TaskType.PERSISTENT_TASK,
        schedule_spec=spec,
        status="scheduled",
        next_run_at=next_run_at,
    )


class TestTimezone:
    def test_cron_uses_task_timezone(self):
        spec = ScheduleSpec(
            trigger="cron",
            cron="0 9 * * *",
            timezone="Asia/Shanghai",
            is_recurring=True,
        )
        # 00:30 UTC = 08:30 上海，下一次本地 09:00 = 01:00 UTC
        now = datetime(2026, 1, 1, 0, 30, tzinfo=timezone.utc)
        assert _compute_next_run(spec, now) == "2026-01-01T01:00:00+00:00"

    def test_cron_without_timezone_stays_utc(self):
        spec = ScheduleSpec(trigger="cron", cron="0 9 * * *", is_recurring=True)
        now = datetime(2026, 1, 1, 0, 30, tzinfo=timezone.utc)
        assert _compute_next_run(spec, now) == "2026-01-01T09:00:00+00:00"

    def test_unknown_timezone_rejected_before_persisting(self):
        with pytest.raises(ValueError):
            ScheduleSpec(
                trigger="cron",
                cron="* * * * *",
                timezone="Mars/Olympus",
                is_recurring=True,
            )


class TestMisfirePolicy:
    async def test_skip_records_run_and_moves_to_future(
        self, store, tmp_path, monkeypatch
    ):
        poller = _poller(
            store,
            tmp_path,
            scheduler_misfire_policy="skip",
            scheduler_misfire_grace_seconds=60,
        )
        spec = ScheduleSpec(trigger="interval", interval_seconds=3600, is_recurring=True)
        past = (_utc_now() - timedelta(hours=2)).isoformat()
        task = _recurring_task(store, spec, past)
        calls = 0

        async def fake_action(record, checkpoint):
            nonlocal calls
            calls += 1
            return "ok"

        monkeypatch.setattr(poller, "_run_task_action", fake_action)
        await poller._execute_task(task)

        assert calls == 0
        record = store.get_task(task.task_id)
        assert record is not None
        assert record.status == "scheduled"
        assert record.next_run_at is not None
        assert record.next_run_at > _utc_now().isoformat()
        runs = store.list_runs(task.task_id)
        assert [run["status"] for run in runs] == ["skipped"]

    async def test_catch_up_is_bounded_and_schedules_future(
        self, store, tmp_path, monkeypatch
    ):
        poller = _poller(
            store,
            tmp_path,
            scheduler_misfire_policy="catch_up",
            scheduler_max_catch_up_runs=3,
        )
        spec = ScheduleSpec(trigger="interval", interval_seconds=600, is_recurring=True)
        task = _recurring_task(store, spec, (_utc_now() - timedelta(minutes=25)).isoformat())
        calls = 0

        async def fake_action(record, checkpoint):
            nonlocal calls
            calls += 1
            return f"run-{calls}"

        monkeypatch.setattr(poller, "_run_task_action", fake_action)
        await poller._execute_task(task)

        assert calls == 3
        assert [run["status"] for run in store.list_runs(task.task_id)] == [
            "completed",
            "completed",
            "completed",
        ]
        record = store.get_task(task.task_id)
        assert record is not None
        assert record.status == "scheduled"
        assert record.next_run_at is not None
        assert record.next_run_at > _utc_now().isoformat()


class TestFailureRetry:
    async def test_retry_schedules_until_budget_exhausted(
        self, store, tmp_path, monkeypatch
    ):
        poller = _poller(
            store,
            tmp_path,
            scheduler_task_max_retries=2,
            scheduler_task_retry_backoff_seconds=0,
        )
        spec = ScheduleSpec(trigger="interval", interval_seconds=600, is_recurring=True)
        task = _recurring_task(store, spec, _utc_now().isoformat())

        async def failing_action(record, checkpoint):
            raise RuntimeError("模型调用失败")

        monkeypatch.setattr(poller, "_run_task_action", failing_action)

        await poller._execute_task(task)
        record = store.get_task(task.task_id)
        assert record is not None
        assert record.status == "scheduled"
        assert record.consecutive_failures == 1

        await poller._execute_task(record)
        record = store.get_task(task.task_id)
        assert record is not None
        assert record.status == "scheduled"
        assert record.consecutive_failures == 2

        await poller._execute_task(record)
        record = store.get_task(task.task_id)
        assert record is not None
        assert record.status == "paused"
        assert record.next_run_at is None
        statuses = [run["status"] for run in store.list_runs(task.task_id)]
        assert statuses == ["failed", "failed", "failed"]

    async def test_success_resets_failure_counter(self, store, tmp_path, monkeypatch):
        poller = _poller(
            store,
            tmp_path,
            scheduler_task_max_retries=1,
            scheduler_task_retry_backoff_seconds=0,
        )
        spec = ScheduleSpec(trigger="interval", interval_seconds=600, is_recurring=True)
        task = _recurring_task(store, spec, _utc_now().isoformat())

        async def failing_action(record, checkpoint):
            raise RuntimeError("临时失败")

        monkeypatch.setattr(poller, "_run_task_action", failing_action)
        await poller._execute_task(task)
        assert store.get_task(task.task_id).consecutive_failures == 1  # type: ignore[union-attr]

        async def ok_action(record, checkpoint):
            return "ok"

        monkeypatch.setattr(poller, "_run_task_action", ok_action)
        await poller._execute_task(store.get_task(task.task_id))  # type: ignore[arg-type]
        record = store.get_task(task.task_id)
        assert record is not None
        assert record.consecutive_failures == 0
        assert record.last_run_at is not None

    def test_backoff_grows_and_is_capped(self):
        assert _retry_backoff_seconds(60, 1) == 60
        assert _retry_backoff_seconds(60, 2) == 120
        assert _retry_backoff_seconds(60, 10) == 3600
        assert _retry_backoff_seconds(0, 5) == 0


class TestSchemaMigration:
    def test_v1_database_is_migrated_in_place(self, tmp_path):
        path = tmp_path / "legacy.db"
        conn = sqlite3.connect(path)
        conn.executescript(
            """
            CREATE TABLE tasks (
                task_id TEXT PRIMARY KEY,
                objective TEXT NOT NULL,
                constraints TEXT DEFAULT '',
                acceptance_criteria TEXT DEFAULT '',
                task_type TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                schedule_spec TEXT,
                checkpoint TEXT,
                result TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                next_run_at TEXT
            );
            CREATE TABLE task_logs (
                log_id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id TEXT NOT NULL REFERENCES tasks(task_id),
                event_type TEXT NOT NULL,
                payload TEXT,
                created_at TEXT NOT NULL
            );
            CREATE TABLE schema_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            INSERT INTO schema_meta (key, value)
                VALUES ('task_store_schema_version', '1');
            INSERT INTO tasks (task_id, objective, task_type, status, created_at, updated_at)
                VALUES ('t1', '历史任务', 'persistent_task', 'scheduled', '2026-01-01', '2026-01-01');
            """
        )
        conn.commit()
        conn.close()

        store = TaskStore(path)
        try:
            record = store.get_task("t1")
            assert record is not None
            assert record.last_run_at is None
            assert record.consecutive_failures == 0
            run_id = store.start_run("t1", "2026-01-01T00:00:00+00:00")
            store.finish_run(run_id, "completed", result="ok")
            assert store.list_runs("t1")[0]["status"] == "completed"
        finally:
            store.close()

        check = sqlite3.connect(path)
        version = check.execute(
            "SELECT value FROM schema_meta WHERE key='task_store_schema_version'"
        ).fetchone()
        check.close()
        assert version == ("2",)

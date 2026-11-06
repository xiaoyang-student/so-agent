"""单元测试：长任务调度器（cron 精确性、暂停语义、并发上限）。"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from so_agent.config import Settings
from so_agent.context import ProjectContext
from so_agent.models import ScheduleSpec, TaskType
from so_agent.persistence import (
    TaskStore,
    resolve_task_store_path,
)
from so_agent.task_scheduler import PeriodicTaskPoller, _compute_next_run


def _due_now() -> str:
    return (datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat()


@pytest.fixture
def store(tmp_path: Path) -> TaskStore:
    store = TaskStore(tmp_path / "tasks.db")
    yield store
    store.close()


@pytest.fixture
def poller(store: TaskStore, tmp_path: Path) -> PeriodicTaskPoller:
    context = ProjectContext(sandbox_dir=tmp_path, project_name="scheduler")
    return PeriodicTaskPoller(store, context, Settings(_env_file=None))


def _scheduled_task(
    store: TaskStore,
    spec: ScheduleSpec | None = None,
    next_run_at: str | None = None,
):
    return store.create_task(
        objective="检查服务状态",
        task_type=TaskType.PERSISTENT_TASK,
        schedule_spec=spec,
        status="scheduled",
        next_run_at=next_run_at or _due_now(),
    )


class TestComputeNextRun:
    def test_cron_is_precise_not_24h(self):
        spec = ScheduleSpec(
            trigger="cron", cron="*/15 * * * *", is_recurring=True
        )
        now = datetime(2026, 1, 1, 10, 7, tzinfo=timezone.utc)
        assert _compute_next_run(spec, now) == "2026-01-01T10:15:00+00:00"

    def test_interval(self):
        spec = ScheduleSpec(
            trigger="interval", interval_seconds=300, is_recurring=True
        )
        now = datetime(2026, 1, 1, 10, 0, tzinfo=timezone.utc)
        assert _compute_next_run(spec, now) == "2026-01-01T10:05:00+00:00"

    def test_once_returns_none(self):
        spec = ScheduleSpec(trigger="once", is_recurring=False)
        assert _compute_next_run(spec) is None

    def test_invalid_cron_raises_value_error(self):
        spec = ScheduleSpec(trigger="cron", cron="bad", is_recurring=True)
        with pytest.raises(ValueError):
            _compute_next_run(spec)

    def test_non_positive_interval_raises(self):
        spec = ScheduleSpec(
            trigger="interval", interval_seconds=0, is_recurring=True
        )
        with pytest.raises(ValueError):
            _compute_next_run(spec)


class TestTaskStorePause:
    def test_paused_task_is_not_due(self, store):
        task = _scheduled_task(store)
        store.pause(task.task_id, error="boom")
        assert store.get_due_tasks() == []
        record = store.get_task(task.task_id)
        assert record is not None
        assert record.status == "paused"
        assert record.next_run_at is None

    def test_claim_is_atomic(self, store):
        task = _scheduled_task(store)
        assert store.claim(task.task_id) is True
        assert store.claim(task.task_id) is False
        assert store.get_task(task.task_id).status == "running"  # type: ignore[union-attr]

    def test_resume_via_reschedule(self, store):
        task = _scheduled_task(store)
        store.pause(task.task_id)
        store.reschedule(task.task_id, _due_now())
        assert [t.task_id for t in store.get_due_tasks()] == [task.task_id]


class TestResolveTaskStorePath:
    def test_relative_to_sandbox_without_double_segment(self, tmp_path):
        context = ProjectContext(sandbox_dir=tmp_path, project_name="p")
        settings = Settings(_env_file=None)
        assert resolve_task_store_path(context, settings) == tmp_path / "task_store.db"

    def test_explicit_sandbox_prefix_is_respected(self, tmp_path):
        context = ProjectContext(sandbox_dir=tmp_path, project_name="p")
        settings = Settings(_env_file=None, task_store_path="sandbox/custom.db")
        assert (
            resolve_task_store_path(context, settings)
            == tmp_path / "sandbox" / "custom.db"
        )

    def test_legacy_database_is_copied_forward(self, tmp_path):
        legacy = tmp_path / "sandbox" / "task_store.db"
        legacy.parent.mkdir(parents=True)
        old = TaskStore(legacy)
        old.create_task(objective="历史任务", task_type=TaskType.PERSISTENT_TASK)
        old.close()

        context = ProjectContext(sandbox_dir=tmp_path, project_name="p")
        path = resolve_task_store_path(context, Settings(_env_file=None))
        assert path == tmp_path / "task_store.db"
        assert path.is_file()
        migrated = TaskStore(path)
        try:
            assert [t.objective for t in migrated.list_tasks()] == ["历史任务"]
        finally:
            migrated.close()


class TestExecuteTask:
    async def test_success_marks_completed(self, poller, store, monkeypatch):
        task = _scheduled_task(store)

        async def fake_action(record, checkpoint):
            return "执行完成"

        monkeypatch.setattr(poller, "_run_task_action", fake_action)
        await poller._execute_task(task)

        record = store.get_task(task.task_id)
        assert record is not None
        assert record.status == "completed"
        assert record.result == "执行完成"

    async def test_recurring_cron_reschedules(self, poller, store, monkeypatch):
        spec = ScheduleSpec(
            trigger="cron", cron="*/5 * * * *", is_recurring=True
        )
        task = _scheduled_task(store, spec=spec)

        async def fake_action(record, checkpoint):
            return "ok"

        monkeypatch.setattr(poller, "_run_task_action", fake_action)
        await poller._execute_task(task)

        record = store.get_task(task.task_id)
        assert record is not None
        assert record.status == "scheduled"
        assert record.next_run_at is not None
        assert record.next_run_at.endswith("+00:00")

    async def test_failure_pauses_and_clears_next_run(
        self, poller, store, monkeypatch
    ):
        task = _scheduled_task(store)

        async def failing_action(record, checkpoint):
            raise RuntimeError("模型调用失败")

        monkeypatch.setattr(poller, "_run_task_action", failing_action)
        await poller._execute_task(task)

        record = store.get_task(task.task_id)
        assert record is not None
        assert record.status == "paused"
        assert record.next_run_at is None
        assert record.checkpoint is not None
        assert "模型调用失败" in record.checkpoint["error"]
        assert store.get_due_tasks() == []

    async def test_already_claimed_task_is_skipped(self, poller, store, monkeypatch):
        task = _scheduled_task(store)
        store.claim(task.task_id)
        calls = 0

        async def fake_action(record, checkpoint):
            nonlocal calls
            calls += 1
            return "ok"

        monkeypatch.setattr(poller, "_run_task_action", fake_action)
        await poller._execute_task(task)
        assert calls == 0


class TestDispatchConcurrency:
    async def test_semaphore_limits_concurrency(self, store, tmp_path, monkeypatch):
        settings = Settings(_env_file=None, scheduler_max_concurrent_tasks=2)
        poller = PeriodicTaskPoller(
            store,
            ProjectContext(sandbox_dir=tmp_path, project_name="c"),
            settings,
        )
        for _ in range(4):
            _scheduled_task(store)

        started = 0
        release = asyncio.Event()

        async def blocked(record):
            nonlocal started
            started += 1
            await release.wait()

        monkeypatch.setattr(poller, "_execute_task", blocked)
        poller._running = True
        try:
            poller._dispatch_due_tasks()
            await asyncio.sleep(0.05)
            assert started == 2
        finally:
            poller._running = False
            release.set()
            await poller._drain_inflight()

    async def test_start_exits_promptly_after_stop(self, store, tmp_path):
        settings = Settings(
            _env_file=None, scheduler_poll_interval=60, scheduler_max_concurrent_tasks=2
        )
        poller = PeriodicTaskPoller(
            store,
            ProjectContext(sandbox_dir=tmp_path, project_name="s"),
            settings,
        )
        runner = asyncio.create_task(poller.start())
        await asyncio.sleep(0.02)
        poller.stop()
        await asyncio.wait_for(runner, timeout=2)
        assert poller._inflight == {}

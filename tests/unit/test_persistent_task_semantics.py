"""持久化任务语义测试：调度状态、event 拒绝与共享存储位置。"""

from __future__ import annotations

from so_agent.config import Settings
from so_agent.context import ProjectContext
from so_agent.models import (
    ClassificationResult,
    ScheduleSpec,
    TaskStatus,
    TaskType,
)
from so_agent.paths.persistent_task import PersistentTaskPath
from so_agent.persistence import TaskStore, resolve_task_store_path


def _classification() -> ClassificationResult:
    return ClassificationResult(
        task_type=TaskType.PERSISTENT_TASK, reasoning="scheduled"
    )


async def _run(tmp_path, monkeypatch, spec: ScheduleSpec):
    context = ProjectContext(
        sandbox_dir=tmp_path / "task",
        project_name="persistent-test",
        task_store_dir=tmp_path / "shared",
        config=Settings(),
    )
    path = PersistentTaskPath(context, Settings())

    async def fake_parse(objective):
        return spec

    monkeypatch.setattr(path, "_parse_schedule", fake_parse)
    output = await path.run("每小时检查服务", "", "", _classification())
    store_path = resolve_task_store_path(context, Settings())
    return output, store_path


async def test_created_task_reports_scheduled(tmp_path, monkeypatch):
    output, store_path = await _run(
        tmp_path,
        monkeypatch,
        ScheduleSpec(
            trigger="interval",
            interval_seconds=3600,
            action="检查服务",
            summary="每小时检查",
            is_recurring=True,
        ),
    )
    assert output.status is TaskStatus.SCHEDULED
    assert output.task_id
    assert store_path == tmp_path / "shared" / "task_store.db"
    record = TaskStore(store_path).get_task(output.task_id)
    assert record is not None
    assert record.status == "scheduled"
    assert record.next_run_at


async def test_event_trigger_is_rejected_explicitly(tmp_path, monkeypatch):
    output, store_path = await _run(
        tmp_path,
        monkeypatch,
        ScheduleSpec(
            trigger="event",
            event_source="webhook",
            action="响应事件",
            summary="事件触发",
        ),
    )
    assert output.status is TaskStatus.FAILED
    assert "event" in output.final_result
    assert not store_path.exists()


async def test_shared_store_path_is_used_for_web_context(tmp_path):
    context = ProjectContext(
        sandbox_dir=tmp_path / "task-a",
        project_name="a",
        task_store_dir=tmp_path / "shared",
        config=Settings(),
    )
    assert (
        resolve_task_store_path(context, Settings())
        == tmp_path / "shared" / "task_store.db"
    )
    legacy_context = ProjectContext(
        sandbox_dir=tmp_path / "task-b",
        project_name="b",
        config=Settings(),
    )
    assert (
        resolve_task_store_path(legacy_context, Settings())
        == tmp_path / "task-b" / "task_store.db"
    )


def test_web_scheduler_starts_with_app(tmp_path):
    from fastapi.testclient import TestClient

    from so_agent.web.app import create_app

    app = create_app(settings=Settings(), base_dir=tmp_path)
    with TestClient(app):
        assert app.state.manager.scheduler_running()
    assert not app.state.manager.scheduler_running()

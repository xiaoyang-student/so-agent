"""Web 重启后未完成任务自动恢复测试（假模型路径）。"""

from __future__ import annotations

import time
from types import SimpleNamespace

from fastapi.testclient import TestClient

from so_agent.config import Settings
from so_agent.models import ClassificationResult, TaskStatus, TaskType
from so_agent.orchestrator import OrchestratorOutput
from so_agent.web import manager as web_manager
from so_agent.web.app import create_app
from so_agent.web.manager import TaskManager
from so_agent.web.store import WebTaskStore


def _router_class():
    classification = ClassificationResult(
        task_type=TaskType.DIRECT_ANSWER, reasoning="simple"
    )
    output = OrchestratorOutput(
        task_id="recovered",
        status=TaskStatus.COMPLETED,
        final_result="恢复执行完成",
    )

    class FakeRouter:
        def __init__(self, *args, **kwargs) -> None:
            pass

        async def classify(self, objective, constraints, acceptance_criteria):
            return classification

        async def route(
            self,
            objective,
            constraints,
            acceptance_criteria,
            *,
            classification=None,
        ):
            return output

    return FakeRouter


def _record(task_id: str, *, status: str = "running"):
    return SimpleNamespace(
        task_id=task_id,
        objective="被重启中断的任务",
        constraints="",
        acceptance_criteria="",
        status=status,
        created_at="2026-01-01T00:00:00+00:00",
        started_at="2026-01-01T00:00:01+00:00",
        finished_at=None,
        classification=None,
        output=None,
        error=None,
        cancel_requested=False,
        last_guidance=None,
    )


def _seed_running_task(root, task_id: str = "r" * 32) -> None:
    store = WebTaskStore(root / "web_tasks.db")
    store.create(_record(task_id))
    store.close()
    (root / task_id).mkdir(parents=True, exist_ok=True)


class TestRecoverInterrupted:
    def test_requeues_and_increments_resume_count(self, tmp_path):
        store = WebTaskStore(tmp_path / "web_tasks.db")
        store.create(_record("t" * 32))
        store.close()

        store = WebTaskStore(tmp_path / "web_tasks.db")
        assert store.recover_interrupted(max_resumes=1) == ["t" * 32]
        row = store.get("t" * 32)
        assert row["status"] == "queued"
        assert row["resume_count"] == 1
        assert row["started_at"] is None
        assert store.recover_interrupted(max_resumes=1) == []
        store.close()

    def test_exhausted_task_is_marked_interrupted(self, tmp_path):
        store = WebTaskStore(tmp_path / "web_tasks.db")
        store.create(_record("u" * 32))
        store.close()

        store = WebTaskStore(tmp_path / "web_tasks.db")
        assert store.recover_interrupted(max_resumes=1) == ["u" * 32]
        store._conn.execute(
            "UPDATE web_tasks SET status = 'running' WHERE task_id = ?",
            ("u" * 32,),
        )
        store._conn.commit()
        assert store.recover_interrupted(max_resumes=1) == []
        store.mark_interrupted(
            running_reason="恢复次数用尽",
            human_reason="人工任务无法恢复",
        )
        row = store.get("u" * 32)
        assert row["status"] == "failed"
        assert row["error"] == "恢复次数用尽"
        store.close()


async def test_manager_requeues_and_reruns_interrupted_task(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(web_manager, "TaskRouter", _router_class())
    task_id = "r" * 32
    _seed_running_task(tmp_path, task_id)

    manager = TaskManager(
        tmp_path,
        Settings(
            _env_file=None,
            web_scheduler_enabled=False,
            web_resume_interrupted_tasks=True,
        ),
        max_concurrent=1,
    )
    task = manager._tasks[task_id]
    assert task.status == "queued"

    assert manager.resume_interrupted() == [task_id]
    await task.runner

    assert task.status == "completed"
    assert task.output is not None
    assert task.output.final_result == "恢复执行完成"
    assert task.error is None
    row = WebTaskStore(tmp_path / "web_tasks.db").get(task_id)
    assert row["status"] == "completed"
    assert row["resume_count"] == 1


def test_app_lifespan_resumes_interrupted_task(tmp_path, monkeypatch):
    monkeypatch.setattr(web_manager, "TaskRouter", _router_class())
    task_id = "l" * 32
    _seed_running_task(tmp_path, task_id)
    app = create_app(
        settings=Settings(
            _env_file=None,
            web_scheduler_enabled=False,
            web_resume_interrupted_tasks=True,
        ),
        base_dir=tmp_path,
    )

    with TestClient(app):
        deadline = time.time() + 3.0
        task = app.state.manager.get_task(task_id)
        while time.time() < deadline:
            task = app.state.manager.get_task(task_id)
            if task is not None and task.status in {
                "completed",
                "failed",
                "cancelled",
            }:
                break
            time.sleep(0.01)
        assert task is not None
        assert task.status == "completed"


async def test_disabled_recovery_keeps_old_failure_behavior(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(web_manager, "TaskRouter", _router_class())
    task_id = "d" * 32
    _seed_running_task(tmp_path, task_id)

    manager = TaskManager(
        tmp_path,
        Settings(
            _env_file=None,
            web_scheduler_enabled=False,
            web_resume_interrupted_tasks=False,
        ),
        max_concurrent=1,
    )
    assert manager.resume_interrupted() == []
    task = manager._tasks[task_id]
    assert task.status == "failed"
    assert "任务被中断" in (task.error or "")

"""Web 任务持久化、重启恢复与保留清理测试（全部使用假模型路径）。"""

from __future__ import annotations

import os
import time
from pathlib import Path
from types import SimpleNamespace

from so_agent.config import Settings
from so_agent.models import ClassificationResult, TaskStatus, TaskType
from so_agent.orchestrator import OrchestratorOutput
from so_agent.web import manager as web_manager
from so_agent.web.manager import TaskManager
from so_agent.web.store import WebTaskStore


def _router_class():
    classification = ClassificationResult(
        task_type=TaskType.DIRECT_ANSWER, reasoning="simple"
    )
    output = OrchestratorOutput(
        task_id="t-restored",
        status=TaskStatus.COMPLETED,
        final_result="持久化结果",
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


async def test_task_survives_restart(tmp_path, monkeypatch):
    monkeypatch.setattr(web_manager, "TaskRouter", _router_class())
    manager = TaskManager(tmp_path, Settings(), max_concurrent=1)
    task = manager.submit("记录一个可恢复任务", "约束", "验收")
    await task.runner
    task_id = task.task_id
    assert task.status == "completed"

    restored = TaskManager(tmp_path, Settings(), max_concurrent=1)
    listed = restored.list_tasks()
    assert [item["task_id"] for item in listed] == [task_id]
    assert listed[0]["status"] == "completed"

    snapshot = restored.snapshot(task_id)
    assert snapshot is not None
    assert snapshot["status"] == "completed"
    assert snapshot["final_result"] == "持久化结果"
    assert snapshot["restored"] is True
    assert snapshot["objective"] == "记录一个可恢复任务"

    english = restored.snapshot(task_id, "en")
    assert english is not None
    assert english["classification"]["label"] == "Direct answer"


async def test_restart_marks_running_task_interrupted(tmp_path):
    store = WebTaskStore(tmp_path / "web_tasks.db")
    store.create(
        SimpleNamespace(
            task_id="a" * 32,
            objective="被中断的任务",
            constraints="",
            acceptance_criteria="",
            status="running",
            created_at="2026-01-01T00:00:00+00:00",
            started_at="2026-01-01T00:00:01+00:00",
            finished_at=None,
        )
    )
    store.close()

    manager = TaskManager(
        tmp_path,
        Settings(_env_file=None, web_resume_interrupted_tasks=False),
        max_concurrent=1,
    )
    task = manager.get_task("a" * 32)
    assert task is not None
    assert task.status == "failed"
    assert "服务重启" in (task.error or "")
    listed = manager.list_tasks()
    assert listed[0]["status"] == "failed"


async def test_restart_marks_human_required_unrecoverable(tmp_path):
    store = WebTaskStore(tmp_path / "web_tasks.db")
    store.create(
        SimpleNamespace(
            task_id="b" * 32,
            objective="待人工任务",
            constraints="",
            acceptance_criteria="",
            status="human_required",
            created_at="2026-01-01T00:00:00+00:00",
            started_at="2026-01-01T00:00:01+00:00",
            finished_at="2026-01-01T00:01:00+00:00",
        )
    )
    store.close()

    manager = TaskManager(tmp_path, Settings(), max_concurrent=1)
    task = manager.get_task("b" * 32)
    assert task is not None
    assert task.status == "failed"
    assert "人工指导无法恢复" in (task.error or "")


async def test_restart_marks_provisional_review_interrupted(tmp_path):
    """重启后 provisional 后台评审已丢失：标记失败，避免永久非终态。"""
    store = WebTaskStore(tmp_path / "web_tasks.db")
    store.create(
        SimpleNamespace(
            task_id="c" * 32,
            objective="先行交付任务",
            constraints="",
            acceptance_criteria="",
            status="completed_provisional",
            created_at="2026-01-01T00:00:00+00:00",
            started_at="2026-01-01T00:00:01+00:00",
            finished_at=None,
        )
    )
    store.close()

    manager = TaskManager(tmp_path, Settings(), max_concurrent=1)
    task = manager.get_task("c" * 32)
    assert task is not None
    assert task.status == "failed"
    assert "后台评审" in (task.error or "")


async def test_retention_prunes_old_records_and_files(tmp_path, monkeypatch):
    monkeypatch.setattr(web_manager, "TaskRouter", _router_class())
    # 保留策略与“单任务覆盖”是两条互补路径：本用例显式开启历史，
    # 以便验证按时间/数量保留清理仍然生效。
    manager = TaskManager(
        tmp_path, Settings(web_keep_history=True), max_concurrent=1
    )
    first = manager.submit("旧任务")
    await first.runner
    second = manager.submit("新任务")
    await second.runner

    first.finished_at = "2020-01-01T00:00:00+00:00"
    first.created_at = "2020-01-01T00:00:00+00:00"
    manager._store.save(first)
    log_path = tmp_path / "logs" / f"{first.task_id}.log"
    log_path.write_text("old log", encoding="utf-8")
    assert (tmp_path / first.task_id).is_dir()
    assert log_path.is_file()

    restored = TaskManager(
        tmp_path,
        Settings(web_max_retained_tasks=1),
        max_concurrent=1,
    )
    assert restored.get_task(first.task_id) is None
    assert not (tmp_path / first.task_id).exists()
    assert not (tmp_path / "logs" / f"{first.task_id}.log").exists()
    assert restored.get_task(second.task_id) is not None


async def test_retention_removes_old_orphan_dirs(tmp_path):
    orphan = tmp_path / ("c" * 32)
    orphan.mkdir()
    (orphan / "artifact.txt").write_text("old", encoding="utf-8")
    old = time.time() - 30 * 24 * 3600
    os.utime(orphan, (old, old))

    TaskManager(
        tmp_path,
        Settings(web_task_retention_days=7),
        max_concurrent=1,
    )
    assert not orphan.exists()

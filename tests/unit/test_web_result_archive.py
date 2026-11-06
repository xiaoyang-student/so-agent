"""单元测试：Web 结果沉淀、单任务覆盖与历史接口（任务 102）。"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from so_agent.config import Settings
from so_agent.models import ClassificationResult, TaskStatus, TaskType
from so_agent.orchestrator import OrchestratorOutput
from so_agent.web import manager as web_manager
from so_agent.web.app import create_app
from so_agent.web.manager import TaskManager
from so_agent.web.result_archive import (
    FileResultArchive,
    JsonlHistoryStore,
    render_markdown,
)


def _settings(tmp_path: Path, **overrides: object) -> Settings:
    base: dict[str, object] = {
        "_env_file": None,
        "sandbox_backend": "local",
        "tool_store_dir": str(tmp_path / "store"),
        "web_scheduler_enabled": False,
    }
    base.update(overrides)
    return Settings(**base)  # type: ignore[arg-type]


class _FakeRouter:
    def __init__(self, *args, **kwargs) -> None:
        pass

    async def classify(self, objective, constraints, acceptance_criteria):
        return ClassificationResult(
            task_type=TaskType.DIRECT_ANSWER, reasoning="result archive test"
        )

    async def route(
        self, objective, constraints, acceptance_criteria, *, classification=None
    ):
        return OrchestratorOutput(
            task_id="fake",
            status=TaskStatus.COMPLETED,
            final_result=f"result:{objective}",
            review_summary="ok",
        )


def _payload(task_id: str, result: str) -> dict[str, object]:
    return {
        "task_id": task_id,
        "status": "completed",
        "final_result": result,
        "total_tokens": 12,
        "tokens_by_agent": {"supervisor": 12},
    }


def test_archive_latest_is_replaced_on_begin_run(tmp_path: Path) -> None:
    archive = FileResultArchive(tmp_path / "results")
    archive.publish("t1", _payload("t1", "first"))
    assert archive.latest()["final_result"] == "first"
    assert "first" in (archive.latest_markdown() or "")

    archive.begin_run()
    assert archive.latest() is None
    assert archive.latest_markdown() is None

    archive.publish("t2", _payload("t2", "second"))
    assert archive.latest()["task_id"] == "t2"
    assert "second" in (archive.latest_markdown() or "")


def test_archive_keeps_history_as_jsonl(tmp_path: Path) -> None:
    history = JsonlHistoryStore(tmp_path / "history.jsonl")
    archive = FileResultArchive(
        tmp_path / "results", keep_history=True, history=history
    )
    archive.publish("t1", _payload("t1", "first"))
    archive.begin_run()
    archive.publish("t2", _payload("t2", "second"))

    records = history.list_recent()
    assert [item["task_id"] for item in records] == ["t2", "t1"]
    assert archive.latest()["task_id"] == "t2"
    line = (tmp_path / "history.jsonl").read_text(encoding="utf-8").splitlines()[0]
    assert json.loads(line)["final_result"] == "first"


def test_render_markdown_includes_sections() -> None:
    text = render_markdown(
        {
            **_payload("t1", "done"),
            "objective": "统计文件",
            "artifacts": [{"path": "out.txt", "size": 2048}],
            "audit": {"verdict": "pass", "event_count": 3, "findings": []},
        }
    )
    assert "# 任务结果" in text
    assert "## 目标" in text
    assert "## 最终结果" in text
    assert "out.txt" in text
    assert "pass" in text


async def test_new_task_clears_previous_terminal_result(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(web_manager, "TaskRouter", _FakeRouter)
    monkeypatch.setattr(web_manager, "WorkflowEngine", _FakeRouter)
    manager = TaskManager(
        tmp_path / "web", _settings(tmp_path), max_concurrent=1
    )
    first = manager.submit("first")
    await first.runner
    latest = manager.latest_result()
    assert latest is not None and latest["final_result"] == "result:first"
    first_dir = first.sandbox_dir
    assert first_dir.is_dir()

    second = manager.submit("second")
    assert first.task_id not in manager._tasks
    assert not first_dir.exists()
    assert manager._store.get(first.task_id) is None
    await second.runner

    latest = manager.latest_result()
    assert latest is not None and latest["final_result"] == "result:second"
    markdown = manager.latest_result_markdown() or ""
    assert "result:second" in markdown
    assert "result:first" not in markdown
    await manager.shutdown()


async def test_keep_history_retains_previous_task(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(web_manager, "TaskRouter", _FakeRouter)
    monkeypatch.setattr(web_manager, "WorkflowEngine", _FakeRouter)
    manager = TaskManager(
        tmp_path / "web",
        _settings(tmp_path, web_keep_history=True),
        max_concurrent=1,
    )
    first = manager.submit("first")
    await first.runner
    second = manager.submit("second")
    await second.runner

    assert manager._store.get(first.task_id) is not None
    assert first.sandbox_dir.is_dir()
    history_path = manager.base_dir / "results" / "history.jsonl"
    records = [
        json.loads(line)
        for line in history_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [item["final_result"] for item in records] == [
        "result:first",
        "result:second",
    ]
    await manager.shutdown()


def test_latest_result_endpoint_requires_result(tmp_path: Path) -> None:
    app = create_app(
        settings=_settings(tmp_path),
        base_dir=tmp_path / "web",
    )
    with TestClient(app) as client:
        response = client.get("/api/results/latest")
        assert response.status_code == 404

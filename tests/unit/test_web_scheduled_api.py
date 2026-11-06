"""Web API：持久化调度任务列表与逐次运行可见性。"""

from __future__ import annotations

from fastapi.testclient import TestClient

from so_agent.config import Settings
from so_agent.models import ScheduleSpec, TaskType
from so_agent.persistence import TaskStore
from so_agent.web.app import create_app


def _seed_scheduled_task(persistent) -> str:
    store = TaskStore(persistent / "task_store.db")
    try:
        spec = ScheduleSpec(
            trigger="cron",
            cron="0 9 * * *",
            timezone="Asia/Shanghai",
            action="汇总数据",
            summary="每天北京时间 9 点汇总",
            is_recurring=True,
        )
        task = store.create_task(
            objective="每日汇总数据",
            task_type=TaskType.PERSISTENT_TASK,
            schedule_spec=spec,
            status="scheduled",
            next_run_at="2099-01-01T00:00:00+00:00",
        )
        run_id = store.start_run(task.task_id, "2098-12-31T00:00:00+00:00")
        store.finish_run(run_id, "completed", result="汇总完成")
        return task.task_id
    finally:
        store.close()


def test_scheduled_tasks_list_and_runs(tmp_path):
    persistent = tmp_path / "persistent"
    persistent.mkdir()
    task_id = _seed_scheduled_task(persistent)

    app = create_app(settings=Settings(_env_file=None), base_dir=tmp_path)
    with TestClient(app) as client:
        response = client.get("/api/scheduled-tasks")
        assert response.status_code == 200
        data = response.json()
        assert data["policy"]["misfire"] == "coalesce"
        tasks = data["tasks"]
        assert len(tasks) == 1
        view = tasks[0]
        assert view["task_id"] == task_id
        assert view["status"] == "scheduled"
        assert view["schedule"]["cron"] == "0 9 * * *"
        assert view["schedule"]["timezone"] == "Asia/Shanghai"
        assert view["next_run_at"] == "2099-01-01T00:00:00+00:00"
        assert view["latest_run"]["status"] == "completed"

        detail = client.get(f"/api/scheduled-tasks/{task_id}")
        assert detail.status_code == 200
        payload = detail.json()
        assert payload["runs"][0]["result"] == "汇总完成"
        assert payload["runs"][0]["status"] == "completed"


def test_unknown_scheduled_task_returns_404(tmp_path):
    app = create_app(settings=Settings(_env_file=None), base_dir=tmp_path)
    with TestClient(app) as client:
        assert client.get("/api/scheduled-tasks/missing").status_code == 404

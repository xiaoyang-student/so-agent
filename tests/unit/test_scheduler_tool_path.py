"""单元测试：调度动作经四象限路由执行（工具路径接入）。

验证：
- 到期动作使用 ``schedule_spec.action`` 作为目标，约束与验收标准透传；
- 每个调度任务使用独立子沙箱（``<sandbox>/scheduled/<task_id>``）；
- 检查点只在执行阶段注入，不参与分类；
- 嵌套调度分类被强制改走完整工作流，避免每次到期重复建任务；
- 非 ``completed`` 终态按失败处理并保存检查点暂停。
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

import so_agent.router as router_mod
from so_agent.config import Settings
from so_agent.context import ProjectContext
from so_agent.models import (
    ClassificationResult,
    ScheduleSpec,
    TaskStatus,
    TaskType,
)
from so_agent.orchestrator import OrchestratorOutput as _Output
from so_agent.persistence import TaskStore
from so_agent.task_scheduler import PeriodicTaskPoller


def _due_now() -> str:
    return (datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat()


@pytest.fixture
def store(tmp_path: Path) -> TaskStore:
    task_store = TaskStore(tmp_path / "tasks.db")
    yield task_store
    task_store.close()


@pytest.fixture
def context(tmp_path: Path) -> ProjectContext:
    return ProjectContext(
        sandbox_dir=tmp_path / "sandbox",
        project_name="scheduler-tool",
        task_store_dir=tmp_path / "persistent",
    )


@pytest.fixture
def poller(store: TaskStore, context: ProjectContext) -> PeriodicTaskPoller:
    return PeriodicTaskPoller(store, context, Settings(_env_file=None))


def _once_task(store: TaskStore, action: str = "检查服务状态"):
    return store.create_task(
        objective="每天检查服务状态",
        task_type=TaskType.PERSISTENT_TASK,
        constraints="不要修改系统配置",
        acceptance_criteria="输出包含健康状态",
        schedule_spec=ScheduleSpec(
            trigger="once", action=action, is_recurring=False
        ),
        status="scheduled",
        next_run_at=_due_now(),
    )


def install_fake_router(
    monkeypatch: pytest.MonkeyPatch,
    *,
    task_type: TaskType = TaskType.FULL_WORKFLOW,
    status: TaskStatus = TaskStatus.COMPLETED,
    result: str = "工具执行完成",
) -> list:
    """替换 TaskRouter，捕获调度动作的选路参数。"""
    created: list = []

    class _FakeRouter:
        def __init__(self, context: ProjectContext, settings: Settings):
            self.context = context
            self.settings = settings
            self.classify_args: tuple | None = None
            self.route_args: dict | None = None
            created.append(self)

        async def classify(
            self, objective: str, constraints: str, acceptance_criteria: str
        ) -> ClassificationResult:
            self.classify_args = (objective, constraints, acceptance_criteria)
            return ClassificationResult(
                task_type=task_type, reasoning="测试分类", steps=[]
            )

        async def route(
            self,
            objective: str,
            constraints: str,
            acceptance_criteria: str,
            *,
            classification: ClassificationResult | None = None,
        ) -> _Output:
            self.route_args = {
                "objective": objective,
                "constraints": constraints,
                "acceptance_criteria": acceptance_criteria,
                "classification": classification,
            }
            return _Output(
                task_id="fake-task", status=status, final_result=result
            )

    monkeypatch.setattr(router_mod, "TaskRouter", _FakeRouter)
    return created


def capture_logs(store: TaskStore, monkeypatch: pytest.MonkeyPatch) -> list:
    logs: list = []
    original = store.add_log

    def _capture(task_id: str, event_type: str, payload=None):
        logs.append((task_id, event_type, payload))
        return original(task_id, event_type, payload)

    monkeypatch.setattr(store, "add_log", _capture)
    return logs


class TestScheduledActionRouting:
    async def test_action_uses_action_text_and_isolated_sandbox(
        self, poller, store, monkeypatch
    ):
        created = install_fake_router(monkeypatch)
        logs = capture_logs(store, monkeypatch)
        task = _once_task(store, action="检查服务并生成健康报告")

        result = await poller._run_task_action(task, None)

        assert result == "工具执行完成"
        router = created[0]
        assert router.classify_args == (
            "检查服务并生成健康报告",
            "不要修改系统配置",
            "输出包含健康状态",
        )
        assert router.route_args["objective"] == "检查服务并生成健康报告"
        assert (
            router.route_args["classification"].task_type
            is TaskType.FULL_WORKFLOW
        )

        expected_sandbox = (
            (poller._context.sandbox_dir / "scheduled" / task.task_id)
            .expanduser()
            .resolve()
        )
        assert router.context.sandbox_dir == expected_sandbox
        assert expected_sandbox.is_dir()
        assert router.context.task_store_dir == poller._context.task_store_dir

        route_logs = [
            payload
            for _, event_type, payload in logs
            if event_type == "execution_route"
        ]
        assert route_logs[0]["task_type"] == "full_workflow"
        assert route_logs[0]["status"] == "completed"

    async def test_empty_action_falls_back_to_objective(
        self, poller, store, monkeypatch
    ):
        created = install_fake_router(monkeypatch)
        task = _once_task(store, action="")

        await poller._run_task_action(task, None)

        assert created[0].classify_args[0] == "每天检查服务状态"

    async def test_checkpoint_only_injected_into_execution(
        self, poller, store, monkeypatch
    ):
        created = install_fake_router(monkeypatch)
        task = _once_task(store)
        checkpoint = {"error": "上一次失败", "partial_state": None}

        await poller._run_task_action(task, checkpoint)

        router = created[0]
        assert "上次执行的检查点信息" not in router.classify_args[1]
        assert "上次执行的检查点信息" in router.route_args["constraints"]
        assert "上一次失败" in router.route_args["constraints"]

    async def test_nested_schedule_is_forced_to_workflow(
        self, poller, store, monkeypatch
    ):
        created = install_fake_router(
            monkeypatch, task_type=TaskType.PERSISTENT_TASK
        )
        task = _once_task(store)

        await poller._run_task_action(task, None)

        classification = created[0].route_args["classification"]
        assert classification.task_type is TaskType.FULL_WORKFLOW
        assert (
            classification.steps[-1].stage
            == "scheduler_nested_task_guard"
        )

    async def test_failed_route_pauses_and_saves_checkpoint(
        self, poller, store, monkeypatch
    ):
        install_fake_router(
            monkeypatch,
            status=TaskStatus.FAILED,
            result="工具执行失败",
        )
        task = _once_task(store)

        await poller._execute_task(task)

        record = store.get_task(task.task_id)
        assert record is not None
        assert record.status == "paused"
        assert record.next_run_at is None
        assert record.checkpoint is not None
        assert "调度动作未成功" in record.checkpoint["error"]
        assert store.get_due_tasks() == []

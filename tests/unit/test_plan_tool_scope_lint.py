"""计划工具范围一致性 lint：只补齐明确要求的标准内置工具。"""

from __future__ import annotations

from types import SimpleNamespace

from so_agent.config import Settings
from so_agent.context import ProjectContext
from so_agent.models import SubtaskSpec, TaskRequest
from so_agent.planning_coordinator import PlanningCoordinator
from so_agent.runtime.events import (
    EVENT_PLAN_TOOL_SCOPE_REPAIRED,
    EventLogger,
)
from so_agent.state_machine import WorkflowState


def _spec(instructions: str, tools: list[str]) -> SubtaskSpec:
    return SubtaskSpec(
        subtask_id="s1",
        title="子任务",
        instructions=instructions,
        role="code",
        allowed_tools=list(tools),
    )


def _coordinator(tmp_path, **settings) -> tuple[PlanningCoordinator, EventLogger]:
    events = EventLogger(
        context=ProjectContext(sandbox_dir=tmp_path, project_name="scope-lint")
    )
    coordinator = object.__new__(PlanningCoordinator)
    coordinator._deps = SimpleNamespace(
        events=events,
        settings=Settings(_env_file=None, **settings),
        tool_registry=SimpleNamespace(get_tool=lambda name: None),
    )
    coordinator._state = WorkflowState(events=events)
    coordinator._state.plan_version = 1
    coordinator._task_request = TaskRequest(task_id="t-lint", objective="工具范围 lint")
    return coordinator, events


def _repair_events(events: EventLogger) -> list[dict]:
    return [
        event
        for event in events.events
        if event["event_type"] == EVENT_PLAN_TOOL_SCOPE_REPAIRED
    ]


class TestPlanToolScopeLint:
    def test_execution_intent_adds_missing_run_tool(self, tmp_path):
        coordinator, events = _coordinator(tmp_path)
        spec = _spec("运行 python analyze.py 生成 metrics.json", ["read_file"])
        coordinator._lint_tool_scope(spec)
        assert "run_in_sandbox" in spec.allowed_tools
        assert "write_file" in spec.allowed_tools
        repaired = _repair_events(events)
        assert len(repaired) == 1
        assert set(repaired[0]["payload"]["added_tools"]) == {
            "run_in_sandbox",
            "write_file",
        }

    def test_read_only_instruction_not_expanded(self, tmp_path):
        coordinator, events = _coordinator(tmp_path)
        spec = _spec("读取 access.log 并撰写 incident_report.md", ["read_file", "write_file"])
        original = list(spec.allowed_tools)
        coordinator._lint_tool_scope(spec)
        assert spec.allowed_tools == original
        assert _repair_events(events) == []

    def test_existing_tools_not_duplicated(self, tmp_path):
        coordinator, events = _coordinator(tmp_path)
        spec = _spec("运行脚本并保存 result.json", ["run_in_sandbox", "write_file"])
        original = list(spec.allowed_tools)
        coordinator._lint_tool_scope(spec)
        assert spec.allowed_tools == original
        assert _repair_events(events) == []

    def test_empty_allowed_tools_left_untouched(self, tmp_path):
        coordinator, events = _coordinator(tmp_path)
        spec = _spec("运行 python a.py", [])
        coordinator._lint_tool_scope(spec)
        assert spec.allowed_tools == []
        assert _repair_events(events) == []

    def test_disabled_by_settings(self, tmp_path):
        coordinator, events = _coordinator(tmp_path, plan_tool_scope_lint=False)
        spec = _spec("运行 python a.py", ["read_file"])
        coordinator._lint_tool_scope(spec)
        assert spec.allowed_tools == ["read_file"]
        assert _repair_events(events) == []

    def test_adopt_plan_applies_lint(self, tmp_path):
        coordinator, _ = _coordinator(tmp_path)
        spec = _spec("执行脚本输出 out.json", ["read_file"])
        adopted = coordinator._adopt_plan([spec])
        assert "run_in_sandbox" in adopted[0].allowed_tools

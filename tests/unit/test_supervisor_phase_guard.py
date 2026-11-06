"""单元测试：规划阶段确定性拒绝主管工具调用。

背景：提示词已禁止 planning / replanning 阶段调用工具，但真实运行中
仍观察到主管在 planning 阶段调用 code_agent 生成占位数据。控制层在
每次主管调用前经 ContextVar 注入阶段，工具桥在进入 MCP 服务器前
拒绝固定工具入口，并写入 ``tool_denied`` 审计事件。
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import so_agent.workflow as workflow_mod
from so_agent.config import Settings
from so_agent.context import ProjectContext
from so_agent.orchestrator import (
    PlanningOutput,
    _supervisor_phase_guard,
    current_supervisor_phase,
    supervisor_phase_scope,
)
from so_agent.runtime.events import EVENT_TOOL_DENIED, EventLogger
from so_agent.workflow import WorkflowEngine


class TestPhaseScope:
    def test_default_phase_is_none(self):
        assert current_supervisor_phase() is None

    def test_scope_sets_and_resets_phase(self):
        with supervisor_phase_scope("planning"):
            assert current_supervisor_phase() == "planning"
            with supervisor_phase_scope("execute_subtask"):
                assert current_supervisor_phase() == "execute_subtask"
            assert current_supervisor_phase() == "planning"
        assert current_supervisor_phase() is None


class TestPhaseGuard:
    def test_outside_supervisor_call_passes(self):
        guard = _supervisor_phase_guard(EventLogger())
        assert guard("code_agent", {}) is None

    def test_planning_phase_rejects_all_supervisor_tools(self):
        events = EventLogger()
        guard = _supervisor_phase_guard(events)
        with supervisor_phase_scope("planning"):
            for tool_name in ("code_agent", "subagent_creator", "review_agent"):
                payload = json.loads(guard(tool_name, {"x": 1}))
                assert payload["status"] == "error"
                assert payload["code"] == "phase_tool_forbidden"
                assert tool_name in payload["error"]
        denied = [
            record
            for record in events.events
            if record["event_type"] == EVENT_TOOL_DENIED
        ]
        assert len(denied) == 3
        assert denied[0]["payload"]["phase"] == "planning"
        assert denied[0]["payload"]["reason"] == "规划阶段禁止调用工具"

    def test_replanning_phase_rejects(self):
        guard = _supervisor_phase_guard(EventLogger())
        with supervisor_phase_scope("replanning"):
            payload = json.loads(guard("code_agent", {}))
        assert payload["code"] == "phase_tool_forbidden"

    def test_execution_and_aggregation_phases_pass(self):
        guard = _supervisor_phase_guard(EventLogger())
        for phase in ("execute_subtask", "aggregating"):
            with supervisor_phase_scope(phase):
                assert guard("code_agent", {}) is None

    def test_guard_without_logger_still_rejects(self):
        guard = _supervisor_phase_guard(None)
        with supervisor_phase_scope("planning"):
            payload = json.loads(guard("code_agent", {}))
        assert payload["code"] == "phase_tool_forbidden"

    def test_arguments_preview_is_bounded(self):
        events = EventLogger()
        guard = _supervisor_phase_guard(events)
        with supervisor_phase_scope("planning"):
            guard("code_agent", {"blob": "x" * 1000})
        denied = events.events[0]["payload"]
        assert len(denied["arguments_preview"]) <= 200

    def test_identical_calls_are_broken_after_threshold(self):
        events = EventLogger()
        guard = _supervisor_phase_guard(
            events, SimpleNamespace(recovery_retry_suppress_threshold=2)
        )
        args = {"phase": "execute_subtask", "subtask": {"subtask_id": "s1"}}
        with supervisor_phase_scope("execute_subtask"):
            assert guard("code_agent", args) is None
            assert guard("code_agent", args) is None
            payload = json.loads(guard("code_agent", args))
        assert payload["code"] == "repeated_identical_call"
        denied = [
            record
            for record in events.events
            if record["event_type"] == EVENT_TOOL_DENIED
        ]
        assert denied[-1]["payload"]["reason"] == "相同参数的重复调用已熔断"

    def test_changed_arguments_are_not_treated_as_repeat(self):
        guard = _supervisor_phase_guard(
            EventLogger(),
            SimpleNamespace(recovery_retry_suppress_threshold=1),
        )
        with supervisor_phase_scope("execute_subtask"):
            assert guard("code_agent", {"attempt": 1}) is None
            assert json.loads(guard("code_agent", {"attempt": 1}))[
                "code"
            ] == "repeated_identical_call"
            # 参数变化（新尝试）应放行，不能阻断正常重试
            assert guard("code_agent", {"attempt": 2}) is None


class TestWorkflowPhaseWiring:
    async def test_run_supervisor_exposes_phase_to_tool_bridge(
        self, tmp_path, monkeypatch
    ):
        seen: list[str | None] = []

        class FakeStreamedResult:
            def __init__(self, final_output: str):
                self.final_output = final_output
                self.raw_responses = []

            def cancel(self, mode: str = "immediate") -> None:
                pass

            async def stream_events(self):
                seen.append(current_supervisor_phase())
                yield {"type": "fake_event"}

        class FakeRunner:
            @classmethod
            def run_streamed(
                cls, starting_agent, input=None, *, max_turns=None, **kwargs
            ):
                return FakeStreamedResult('{"subtasks": []}')

        monkeypatch.setattr(workflow_mod, "Runner", FakeRunner)
        engine = WorkflowEngine(
            ProjectContext(sandbox_dir=tmp_path, project_name="phase-wiring"),
            settings=Settings(_env_file=None),
            orchestrator=SimpleNamespace(name="fake-supervisor"),
            review_agent=SimpleNamespace(name="fake-reviewer"),
        )

        await engine._run_supervisor({"phase": "planning"}, PlanningOutput)

        assert seen == ["planning"]
        assert current_supervisor_phase() is None

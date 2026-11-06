"""计划结构确定性校验、评审分级与评审工具宽容解析。

覆盖：
- 结构硬校验（空/重复 ID、自依赖、依赖缺失、成环、越界写入路径）在
  进入模型评审前拦截，并写入 ``plan_validation_failed`` 审计事件；
- 授权评审分级：候选满足确定性最小化条件时跳过模型授权评审，
  非确定性候选仍保留独立评审；
- 评审工具对模型精简输入（缺字段、字符串列表、非 JSON 验收标准）的
  宽容解析，以及对缺失 success 的 fail-closed 处理。
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from agents.tool_context import ToolContext

from so_agent.config import Settings
from so_agent.context import ProjectContext
from so_agent.models import ReviewDecision, SubtaskSpec, TaskRequest, ToolGrant
from so_agent.planning_coordinator import PlanningCoordinator
from so_agent.runtime.events import (
    EVENT_PLAN_VALIDATION_FAILED,
    EVENT_REVIEW_SKIPPED,
    EventLogger,
)
from so_agent.tool_packages.review_agent import agent as review_mod
from so_agent.workflow import WorkflowEngine


def make_ctx(tool_name: str) -> ToolContext:
    return ToolContext(
        context=None,
        tool_name=tool_name,
        tool_call_id=f"call-{tool_name}-1",
        tool_arguments="{}",
    )


def spec(
    subtask_id: str,
    *,
    deps: list[str] | None = None,
    write_scope: list[str] | None = None,
    tools: list[str] | None = None,
) -> SubtaskSpec:
    return SubtaskSpec(
        subtask_id=subtask_id,
        title=f"子任务 {subtask_id}",
        instructions="执行",
        role="code",
        dependencies=list(deps or []),
        allowed_tools=list(tools or []),
        write_scope=list(write_scope or []),
    )


def make_validator(subtasks: list[SubtaskSpec]) -> SimpleNamespace:
    events = EventLogger()
    return SimpleNamespace(
        _subtasks=subtasks,
        _plan_version=1,
        _task_request=TaskRequest(task_id="t1", objective="目标"),
        _events=events,
    )


class TestDeterministicPlanValidation:
    def test_valid_plan_passes(self):
        validator = make_validator([spec("s1"), spec("s2", deps=["s1"])])
        assert PlanningCoordinator._validate_plan_structure(validator) is None
        assert not validator._events.events

    def test_duplicate_ids_rejected(self):
        validator = make_validator([spec("s1"), spec("s1")])
        decision = PlanningCoordinator._validate_plan_structure(validator)
        assert decision is not None
        assert decision.passed is False
        assert any("重复" in issue for issue in decision.issues)
        assert decision.required_fixes
        records = validator._events.get_events(
            event_type=EVENT_PLAN_VALIDATION_FAILED
        )
        assert len(records) == 1

    def test_empty_id_rejected(self):
        validator = make_validator([spec("")])
        decision = PlanningCoordinator._validate_plan_structure(validator)
        assert decision is not None and decision.passed is False

    def test_missing_dependency_rejected(self):
        validator = make_validator([spec("s1", deps=["ghost"])])
        decision = PlanningCoordinator._validate_plan_structure(validator)
        assert decision is not None
        assert any("依赖不存在" in issue for issue in decision.issues)

    def test_self_dependency_rejected(self):
        validator = make_validator([spec("s1", deps=["s1"])])
        decision = PlanningCoordinator._validate_plan_structure(validator)
        assert decision is not None
        assert any("自依赖" in issue for issue in decision.issues)

    def test_cycle_rejected(self):
        validator = make_validator(
            [spec("s1", deps=["s2"]), spec("s2", deps=["s1"])]
        )
        decision = PlanningCoordinator._validate_plan_structure(validator)
        assert decision is not None
        assert any("环" in issue for issue in decision.issues)

    @pytest.mark.parametrize("entry", ["../secret.txt", "/etc/passwd", "C:\\tmp\\x"])
    def test_unsafe_write_scope_rejected(self, entry):
        validator = make_validator([spec("s1", write_scope=[entry])])
        decision = PlanningCoordinator._validate_plan_structure(validator)
        assert decision is not None
        assert any("越界写入" in issue for issue in decision.issues)

    async def test_plan_review_short_circuits_without_model_call(
        self, tmp_path, monkeypatch
    ):
        context = ProjectContext(
            sandbox_dir=tmp_path, project_name="plan-validation"
        )
        engine = WorkflowEngine(
            context,
            settings=Settings(_env_file=None, max_replan_attempts=0),
            orchestrator=SimpleNamespace(name="fake-supervisor"),
            review_agent=SimpleNamespace(name="fake-reviewer"),
        )
        engine._create_task("目标", "", "")

        async def fake_request_plan(*, issue, feedback):
            # 环不会被 _adopt_plan 清理，必须由确定性校验拦截
            return [spec("s1", deps=["s2"]), spec("s2", deps=["s1"])], []

        engine._planning_coordinator._request_plan = fake_request_plan
        review_calls: list = []

        async def fake_review(review_input):
            review_calls.append(review_input)
            return ReviewDecision(stage="plan_review", passed=True, summary="模拟")

        monkeypatch.setattr(engine._deps, "call_review_agent", fake_review)

        accepted = await engine._plan_and_review()

        assert accepted is False
        assert review_calls == []
        assert engine._events.get_events(
            event_type=EVENT_PLAN_VALIDATION_FAILED
        )


class TestAuthorizationReviewTiering:
    def _engine(self, tmp_path, *, mode: str | None = None):
        settings = Settings(_env_file=None)
        if mode is not None:
            settings = settings.model_copy(
                update={"authorization_review_mode": mode}
            )
        context = ProjectContext(
            sandbox_dir=tmp_path, project_name="plan-tiering"
        )
        engine = WorkflowEngine(
            context,
            settings=settings,
            orchestrator=SimpleNamespace(name="fake-supervisor"),
            review_agent=SimpleNamespace(name="fake-reviewer"),
        )
        engine._task_request = TaskRequest(task_id="t1", objective="目标")
        engine._subtasks = [spec("s1", tools=["write_file"])]
        engine._plan_version = 1
        return engine

    def _grant(self, *tools: str, slot: str = "s1") -> ToolGrant:
        return ToolGrant(
            grant_id="proposal-1",
            task_id="t1",
            agent_id=f"t1:{slot}:grant-slot",
            allowed_tools=list(tools),
            issued_by="supervisor_agent",
        )

    def test_mechanical_candidate_is_deterministic(self, tmp_path):
        engine = self._engine(tmp_path)
        engine._plan_grant_proposals = [self._grant("write_file")]
        assert engine._authorization_review_is_deterministic() is True

    def test_always_mode_never_deterministic(self, tmp_path):
        engine = self._engine(tmp_path, mode="always")
        engine._plan_grant_proposals = [self._grant("write_file")]
        assert engine._authorization_review_is_deterministic() is False

    def test_over_granted_candidate_not_deterministic(self, tmp_path):
        engine = self._engine(tmp_path)
        engine._plan_grant_proposals = [
            self._grant("write_file", "run_in_sandbox")
        ]
        assert engine._authorization_review_is_deterministic() is False

    def test_unknown_slot_not_deterministic(self, tmp_path):
        engine = self._engine(tmp_path)
        engine._plan_grant_proposals = [
            self._grant("write_file", slot="ghost")
        ]
        assert engine._authorization_review_is_deterministic() is False

    async def test_deterministic_path_logs_skip_and_keeps_plan_review(
        self, tmp_path, monkeypatch
    ):
        engine = self._engine(tmp_path)
        engine._plan_grant_proposals = [self._grant("write_file")]
        captured: list = []

        async def fake_review(review_input):
            captured.append(review_input)
            return ReviewDecision(stage="plan_review", passed=True, summary="模拟")

        monkeypatch.setattr(engine._deps, "call_review_agent", fake_review)

        decision = await engine._review_plan()

        assert decision.passed is True
        assert len(captured) == 1
        skipped = engine._events.get_events(event_type=EVENT_REVIEW_SKIPPED)
        assert skipped[0]["payload"]["reason"] == "authorization_deterministic"


class TestTolerantReviewToolInputs:
    async def _invoke(self, tmp_path, tool_name: str, args: dict):
        context = ProjectContext(
            sandbox_dir=tmp_path, project_name="review-tools"
        )
        tools = review_mod.build_review_tools(context)
        raw = await tools[tool_name].on_invoke_tool(
            make_ctx(tool_name), json.dumps(args, ensure_ascii=False)
        )
        return json.loads(raw)

    async def test_minimal_subtasks_do_not_break_plan_check(self, tmp_path):
        payload = await self._invoke(
            tmp_path,
            "check_plan_completeness",
            {
                "subtasks_json": json.dumps(
                    [
                        {"id": "s1"},
                        {"id": "s2", "dependencies": "s1"},
                    ]
                )
            },
        )
        assert payload["status"] == "ok"
        assert payload["passed"] is True
        assert payload["duplicate_ids"] == []
        assert any("非结构字段" in note for note in payload["notes"])

    async def test_explicit_empty_field_still_flagged(self, tmp_path):
        payload = await self._invoke(
            tmp_path,
            "check_plan_completeness",
            {
                "subtasks_json": json.dumps(
                    [
                        {
                            "subtask_id": "s1",
                            "title": "",
                            "instructions": "执行",
                            "role": "code",
                        }
                    ]
                )
            },
        )
        assert payload["passed"] is False
        assert payload["empty_fields"] == [
            {"subtask_id": "s1", "missing": ["title"]}
        ]

    async def test_string_list_fields_are_normalized(self, tmp_path):
        payload = await self._invoke(
            tmp_path,
            "check_plan_completeness",
            {
                "subtasks_json": json.dumps(
                    [
                        {
                            "subtask_id": "s1",
                            "title": "t",
                            "instructions": "i",
                            "role": "code",
                            "dependencies": "s2, s3",
                        },
                        {"subtask_id": "s2"},
                        {"subtask_id": "s3"},
                    ]
                )
            },
        )
        assert payload["status"] == "ok"
        assert payload["missing_dependencies"] == []

    async def test_grant_without_metadata_is_accepted(self, tmp_path):
        payload = await self._invoke(
            tmp_path,
            "check_tool_necessity",
            {
                "subtasks_json": json.dumps(
                    [{"subtask_id": "s1", "allowed_tools": ["write_file"]}]
                ),
                "grants_json": json.dumps(
                    [{"grant_id": "g1", "allowed_tools": "write_file"}]
                ),
            },
        )
        assert payload["status"] == "ok"
        assert payload["granted_tools"] == ["write_file"]
        assert payload["over_granted"] == []
        assert payload["passed"] is True

    async def test_missing_success_fails_closed(self, tmp_path):
        payload = await self._invoke(
            tmp_path,
            "check_result_correctness",
            {
                "results_json": json.dumps(
                    [{"subtask_id": "s1", "output": "完成"}]
                ),
                "criteria_json": "标准一；标准二",
            },
        )
        assert payload["status"] == "ok"
        assert payload["failed_subtasks"] == [
            {"subtask_id": "s1", "error": None}
        ]
        assert payload["passed"] is False
        assert payload["criteria"] == ["标准一", "标准二"]
        assert any("fail-closed" in note for note in payload["notes"])

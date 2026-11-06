"""单元测试：计划评审携带可用工具清单（``ReviewInput.available_tools``）。

锁定：
- ``ReviewInput.available_tools`` 可选字段（默认 None、JSON 往返一致）；
- ``check_plan_completeness`` 的工具越界检查：子任务声明的
  allowed_tools 不在可用工具列表中时输出 ``out_of_scope_tools`` 提醒，
  但不构成失败（待创建的动态工具属正常流程）；未提供 available_tools
  时跳过该检查；
- ``_review_plan`` 把当前已批准可分配的工具名称清单传给两份
  ReviewInput（计划评审 + 授权评审）。
"""

from __future__ import annotations

import json
from types import SimpleNamespace

from agents.tool_context import ToolContext

from so_agent.config import Settings
from so_agent.context import ProjectContext
from so_agent.models import (
    ReviewDecision,
    SubtaskSpec,
    TaskRequest,
    TaskStatus,
    ToolGrant,
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


def make_spec(subtask_id: str, allowed_tools: list[str]) -> SubtaskSpec:
    return SubtaskSpec(
        subtask_id=subtask_id,
        title=f"模拟子任务 {subtask_id}",
        instructions=f"完成子任务 {subtask_id}",
        role="code",
        allowed_tools=allowed_tools,
    )


async def invoke_plan_check(tmp_path, specs, available_tools):
    """调用 check_plan_completeness 并解析 JSON 返回值。

    Args:
        available_tools: None 表示不传 available_tools_json（默认参数）。
    """
    context = ProjectContext(sandbox_dir=tmp_path, project_name="plan-review-tools")
    tools = review_mod.build_review_tools(context)
    args: dict[str, str] = {
        "subtasks_json": json.dumps(
            [spec.model_dump(mode="json") for spec in specs], ensure_ascii=False
        ),
    }
    if available_tools is not None:
        args["available_tools_json"] = json.dumps(available_tools, ensure_ascii=False)
    raw = await tools["check_plan_completeness"].on_invoke_tool(
        make_ctx("check_plan_completeness"),
        json.dumps(args, ensure_ascii=False),
    )
    return json.loads(raw)


class TestReviewInputAvailableTools:
    """ReviewInput.available_tools 字段契约。"""

    def test_defaults_to_none(self):
        review_input = review_mod.ReviewInput(
            stage=review_mod.ReviewStage.PLAN_REVIEW,
            task_request=TaskRequest(task_id="t1", objective="目标"),
        )
        assert review_input.available_tools is None

    def test_json_roundtrip(self):
        review_input = review_mod.ReviewInput(
            stage=review_mod.ReviewStage.PLAN_REVIEW,
            task_request=TaskRequest(task_id="t1", objective="目标"),
            available_tools=["write_file", "demo_tool"],
        )
        data = json.loads(review_input.model_dump_json())
        assert data["available_tools"] == ["write_file", "demo_tool"]
        parsed = review_mod.ReviewInput.model_validate(data)
        assert parsed.available_tools == ["write_file", "demo_tool"]

    def test_extra_instructions_mention_available_tools(self):
        text = review_mod._EXTRA_INSTRUCTIONS
        assert "available_tools" in text
        assert "available_tools_json" in text
        # 越界工具：issues 指出并建议替代方案，但不构成拒绝理由
        assert "建议替代" in text
        assert "不构成拒绝理由" in text


class TestPlanCheckOutOfScopeTools:
    """check_plan_completeness 的工具越界检查。"""

    async def test_out_of_scope_reported_but_passing(self, tmp_path):
        specs = [make_spec("s1", ["write_file", "mystery_tool"])]
        payload = await invoke_plan_check(
            tmp_path, specs, ["write_file", "run_in_sandbox"]
        )
        assert payload["status"] == "ok"
        assert payload["out_of_scope_tools"] == ["mystery_tool"]
        assert payload["passed"] is True  # 仅提醒，不构成失败
        assert any("不在可用工具列表" in note for note in payload["notes"])

    async def test_skipped_without_available_tools(self, tmp_path):
        specs = [make_spec("s1", ["mystery_tool"])]
        payload = await invoke_plan_check(tmp_path, specs, None)
        assert payload["out_of_scope_tools"] == []
        assert payload["passed"] is True

    async def test_all_in_scope_reports_empty(self, tmp_path):
        specs = [make_spec("s1", ["write_file"])]
        payload = await invoke_plan_check(tmp_path, specs, ["write_file"])
        assert payload["out_of_scope_tools"] == []
        assert payload["passed"] is True

    async def test_structural_issues_still_fail(self, tmp_path):
        # 工具越界不豁免结构问题：重复 ID 仍然判定不通过
        specs = [make_spec("s1", []), make_spec("s1", [])]
        payload = await invoke_plan_check(tmp_path, specs, ["write_file"])
        assert payload["passed"] is False
        assert payload["duplicate_ids"] == ["s1"]


class TestReviewPlanCarriesAvailableTools:
    """_review_plan 把可用工具名称清单传给 ReviewInput。"""

    async def test_authorization_review_skipped_without_proposals(
        self, tmp_path, monkeypatch
    ):
        # 无授权候选时授权评审没有可审对象：确定性跳过，只调用一次
        context = ProjectContext(sandbox_dir=tmp_path, project_name="plan-review-tools")
        engine = WorkflowEngine(
            context,
            settings=Settings(_env_file=None),
            orchestrator=SimpleNamespace(name="fake-supervisor"),
            review_agent=SimpleNamespace(name="fake-reviewer"),
        )
        engine._task_request = TaskRequest(task_id="t1", objective="目标")
        engine._subtasks = [make_spec("s1", [])]
        engine._transition(TaskStatus.PLANNING)

        captured: list = []

        async def fake_review(review_input):
            captured.append(review_input)
            return ReviewDecision(stage="plan_review", passed=True, summary="模拟")

        monkeypatch.setattr(engine._deps, "call_review_agent", fake_review)

        decision = await engine._review_plan()

        assert decision.passed is True
        assert len(captured) == 1
        assert captured[0].stage == review_mod.ReviewStage.PLAN_REVIEW
        assert captured[0].available_tools is not None
        assert "write_file" in captured[0].available_tools
        skipped = engine._events.get_events(event_type="review_skipped")
        assert len(skipped) == 1
        assert skipped[0]["payload"]["reason"] == "no_grant_proposals"

    def _make_engine(self, tmp_path, *, settings=None, grants=None):
        context = ProjectContext(
            sandbox_dir=tmp_path, project_name="plan-review-tools"
        )
        engine = WorkflowEngine(
            context,
            settings=settings or Settings(_env_file=None),
            orchestrator=SimpleNamespace(name="fake-supervisor"),
            review_agent=SimpleNamespace(name="fake-reviewer"),
        )
        engine._task_request = TaskRequest(task_id="t1", objective="目标")
        engine._subtasks = [make_spec("s1", ["write_file"])]
        engine._plan_grant_proposals = grants or [
            ToolGrant(
                grant_id="g1",
                task_id="t1",
                agent_id="t1:s1:grant-slot",
                allowed_tools=["write_file"],
                issued_by="supervisor_agent",
            )
        ]
        engine._transition(TaskStatus.PLANNING)
        return engine

    async def test_deterministic_authorization_skips_second_review(
        self, tmp_path, monkeypatch
    ):
        # 候选机械满足最小化条件：授权评审由控制层完成，只调用一次模型
        engine = self._make_engine(tmp_path)
        captured: list = []

        async def fake_review(review_input):
            captured.append(review_input)
            return ReviewDecision(stage="plan_review", passed=True, summary="模拟")

        monkeypatch.setattr(engine._deps, "call_review_agent", fake_review)

        decision = await engine._review_plan()

        assert decision.passed is True
        assert len(captured) == 1
        assert captured[0].stage == review_mod.ReviewStage.PLAN_REVIEW
        assert captured[0].available_tools is not None
        assert "write_file" in captured[0].available_tools
        skipped = engine._events.get_events(event_type="review_skipped")
        assert any(
            record["payload"]["reason"] == "authorization_deterministic"
            for record in skipped
        )

    async def test_always_mode_keeps_both_review_inputs(
        self, tmp_path, monkeypatch
    ):
        engine = self._make_engine(
            tmp_path,
            settings=Settings(_env_file=None, authorization_review_mode="always"),
        )
        captured: list = []

        async def fake_review(review_input):
            captured.append(review_input)
            return ReviewDecision(stage="plan_review", passed=True, summary="模拟")

        monkeypatch.setattr(engine._deps, "call_review_agent", fake_review)

        decision = await engine._review_plan()

        assert decision.passed is True
        assert len(captured) == 2
        stages = {item.stage for item in captured}
        assert stages == {
            review_mod.ReviewStage.PLAN_REVIEW,
            review_mod.ReviewStage.AUTHORIZATION_REVIEW,
        }
        for item in captured:
            # 默认注册表包含预置可分配工具 write_file / read_file / run_in_sandbox
            assert item.available_tools is not None
            assert "write_file" in item.available_tools

    async def test_over_granted_candidate_falls_back_to_model_review(
        self, tmp_path, monkeypatch
    ):
        # 候选工具不在子任务声明范围内：确定性条件不成立，保留独立授权评审
        engine = self._make_engine(
            tmp_path,
            grants=[
                ToolGrant(
                    grant_id="g1",
                    task_id="t1",
                    agent_id="t1:s1:grant-slot",
                    allowed_tools=["write_file", "run_in_sandbox"],
                    issued_by="supervisor_agent",
                )
            ],
        )
        captured: list = []

        async def fake_review(review_input):
            captured.append(review_input)
            return ReviewDecision(stage="plan_review", passed=True, summary="模拟")

        monkeypatch.setattr(engine._deps, "call_review_agent", fake_review)

        decision = await engine._review_plan()

        assert decision.passed is True
        assert len(captured) == 2
        assert any(
            item.stage == review_mod.ReviewStage.AUTHORIZATION_REVIEW
            for item in captured
        )


class TestPlanningPayloadToolSlimming:
    """主管规划载荷的工具清单瘦身（默认开启）。"""

    def make_engine(self, tmp_path, **settings):
        context = ProjectContext(
            sandbox_dir=tmp_path, project_name="planning-slimming"
        )
        engine = WorkflowEngine(
            context,
            settings=Settings(_env_file=None, **settings),
            orchestrator=SimpleNamespace(name="fake-supervisor"),
            review_agent=SimpleNamespace(name="fake-reviewer"),
        )
        engine._create_task("目标", "", "")
        engine._tool_registry.register_builtin_tool(
            "long_tool",
            "描" * 500,
            "so_agent.tool_packages.code_agent",
        )
        return engine

    def test_default_keeps_names_drops_metadata_and_truncates(
        self, tmp_path
    ):
        engine = self.make_engine(tmp_path)
        payload = engine._build_planning_payload(issue=None, feedback=None)
        tools = payload["available_tools"]
        by_name = {item["name"]: item for item in tools}
        assert "write_file" in by_name
        assert "long_tool" in by_name
        for item in tools:
            assert set(item) == {"name", "description"}

        long_description = by_name["long_tool"]["description"]
        assert len(long_description) <= 161
        assert long_description.endswith("…")
        assert long_description.startswith("描")

        # 拆解决策输入不受瘦身影响
        assert payload["orchestrator_input"]["objective"] == "目标"

    def test_disabled_keeps_full_metadata(self, tmp_path):
        engine = self.make_engine(
            tmp_path, supervisor_context_slimming=False
        )
        payload = engine._build_planning_payload(issue=None, feedback=None)
        long_tool = next(
            item
            for item in payload["available_tools"]
            if item["name"] == "long_tool"
        )
        assert long_tool["description"] == "描" * 500
        assert long_tool["source"] == "builtin"
        assert long_tool["approved"] is True

    def test_zero_limit_disables_truncation(self, tmp_path):
        engine = self.make_engine(
            tmp_path, supervisor_tool_description_max_chars=0
        )
        payload = engine._build_planning_payload(issue=None, feedback=None)
        long_tool = next(
            item
            for item in payload["available_tools"]
            if item["name"] == "long_tool"
        )
        assert long_tool["description"] == "描" * 500

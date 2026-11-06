"""控制层预计算评审检查结果（P0：评审单轮化）。

锁定契约：
- 计划 / 授权 / 结果评审的确定性检查由控制层预计算，结果随
  ``ReviewInput.deterministic_findings`` 下发；
- 工作流内部构造的评审 Agent 不再绑定校验工具（单轮结论），
  外部 MCP / as_tool 入口仍保留工具；
- ``_call_review_agent`` 在调用模型前注入预计算结果。
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

import so_agent.workflow as workflow_mod
from so_agent.config import Settings
from so_agent.context import ProjectContext
from so_agent.models import (
    ExecutionResult,
    SubtaskSpec,
    TaskRequest,
    ToolGrant,
)
from so_agent.tool_packages.review_agent import agent as review_mod
from so_agent.tool_packages.review_agent.agent import (
    ReviewInput,
    ReviewStage,
    build_deterministic_findings,
    create_review_agent,
)
from so_agent.workflow import WorkflowEngine


def make_spec(subtask_id: str = "s1") -> SubtaskSpec:
    return SubtaskSpec(
        subtask_id=subtask_id,
        title="分析日志",
        instructions="读取并分析日志",
        role="code",
        allowed_tools=["read_file"],
    )


def make_request() -> TaskRequest:
    return TaskRequest(
        task_id="t-precomputed",
        objective="分析日志",
        acceptance_criteria=["输出指标 JSON"],
    )


def test_plan_review_findings_include_completeness(tmp_path):
    review_input = ReviewInput(
        stage=ReviewStage.PLAN_REVIEW,
        task_request=make_request(),
        subtasks=[make_spec()],
        available_tools=["read_file"],
    )
    findings = build_deterministic_findings(review_input, tmp_path)
    assert findings["plan_completeness"]["passed"] is True
    assert "tool_necessity" not in findings


def test_plan_findings_flag_structural_problems(tmp_path):
    review_input = ReviewInput(
        stage=ReviewStage.PLAN_REVIEW,
        task_request=make_request(),
        subtasks=[make_spec("s1"), make_spec("s1")],
    )
    findings = build_deterministic_findings(review_input, tmp_path)
    assert findings["plan_completeness"]["passed"] is False
    assert findings["plan_completeness"]["duplicate_ids"] == ["s1"]


def test_authorization_review_includes_tool_necessity(tmp_path):
    review_input = ReviewInput(
        stage=ReviewStage.AUTHORIZATION_REVIEW,
        task_request=make_request(),
        subtasks=[make_spec()],
        tool_grants=[
            ToolGrant(
                grant_id="g1",
                task_id="t-precomputed",
                agent_id="code_agent:s1",
                allowed_tools=["read_file", "write_file"],
                issued_by="supervisor",
            )
        ],
    )
    findings = build_deterministic_findings(review_input, tmp_path)
    assert findings["tool_necessity"]["over_granted"] == ["write_file"]
    assert findings["tool_necessity"]["passed"] is False


def test_result_review_includes_correctness(tmp_path):
    review_input = ReviewInput(
        stage=ReviewStage.RESULT_REVIEW,
        task_request=make_request(),
        subtasks=[make_spec()],
        execution_results=[
            ExecutionResult(
                subtask_id="s1",
                success=True,
                output="完成",
                evidence=["read_file: access.log"],
            )
        ],
    )
    findings = build_deterministic_findings(review_input, tmp_path)
    correctness = findings["result_correctness"]
    assert correctness["passed"] is True
    assert correctness["criteria"] == ["输出指标 JSON"]


class _CaptureRunner:
    captured: dict = {}

    @classmethod
    async def run(cls, starting_agent, input=None, *, max_turns=None, **kwargs):
        cls.captured = json.loads(input)
        return SimpleNamespace(
            final_output={
                "stage": "plan_review",
                "passed": True,
                "issues": [],
                "suggestions": [],
                "required_fixes": [],
                "retry_target": None,
                "summary": "通过",
            },
            raw_responses=[],
        )


@pytest.mark.asyncio
async def test_call_review_agent_injects_findings(tmp_path, monkeypatch):
    context = ProjectContext(sandbox_dir=tmp_path, project_name="precomputed")
    engine = WorkflowEngine(
        context,
        settings=Settings(_env_file=None),
        orchestrator=SimpleNamespace(name="fake-supervisor"),
        review_agent=SimpleNamespace(name="fake-reviewer"),
    )
    monkeypatch.setattr(workflow_mod, "Runner", _CaptureRunner)
    review_input = ReviewInput(
        stage=ReviewStage.PLAN_REVIEW,
        task_request=make_request(),
        subtasks=[make_spec()],
    )
    decision = await engine._call_review_agent(review_input)
    assert decision.passed is True
    findings = _CaptureRunner.captured["deterministic_findings"]
    assert findings["plan_completeness"]["passed"] is True


def test_workflow_review_agent_has_no_tools(tmp_path, monkeypatch):
    captured: dict = {}

    class _FakeAgent:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(review_mod, "Agent", _FakeAgent)
    agent = create_review_agent(
        ProjectContext(sandbox_dir=tmp_path, project_name="no-tools"),
        settings=Settings(_env_file=None),
        model=object(),
    )
    assert agent is not None
    assert captured["tools"] == []


def test_external_entry_keeps_tools(tmp_path, monkeypatch):
    captured: dict = {}

    class _FakeAgent:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(review_mod, "Agent", _FakeAgent)
    create_review_agent(
        ProjectContext(sandbox_dir=tmp_path, project_name="with-tools"),
        settings=Settings(_env_file=None),
        model=object(),
        with_tools=True,
    )
    assert {tool.name for tool in captured["tools"]} == {
        "check_plan_completeness",
        "check_tool_necessity",
        "check_result_correctness",
    }

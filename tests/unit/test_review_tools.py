"""单元测试：评审 Agent 内部校验工具的授权评审策略（check_tool_necessity）。

锁定授权评审的通过/拒绝策略（与 _EXTRA_INSTRUCTIONS 提示词一致）：
- under_granted（声明但未授权）不构成拒绝理由——动态工具需在执行阶段
  创建后才能授权，计划阶段无法预先签发，属正常流程：此时 passed=true，
  仅在 notes 中提示主管在执行前补发授权；
- over_granted（超额授权）/ empty_grants（空授权）/ suspicious_tools
  （宽泛高风险工具）才构成拒绝（passed=false）。
"""

from __future__ import annotations

import json

import pytest
from agents.tool_context import ToolContext

from so_agent.context import ProjectContext
from so_agent.models import SubtaskSpec, ToolGrant
from so_agent.tool_packages.review_agent import agent as review_mod


@pytest.fixture
def tools(tmp_path):
    context = ProjectContext(sandbox_dir=tmp_path, project_name="review-tools-test")
    return review_mod.build_review_tools(context)


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


def make_grant(grant_id: str, allowed_tools: list[str]) -> ToolGrant:
    return ToolGrant(
        grant_id=grant_id,
        task_id="t1",
        agent_id="slot",
        allowed_tools=allowed_tools,
        issued_by="supervisor_agent",
    )


async def invoke_necessity(
    tools: dict, specs: list[SubtaskSpec], grants: list[ToolGrant]
) -> dict:
    """以 FunctionTool 协议调用 check_tool_necessity 并解析 JSON 返回值。"""
    raw = await tools["check_tool_necessity"].on_invoke_tool(
        make_ctx("check_tool_necessity"),
        json.dumps(
            {
                "subtasks_json": json.dumps(
                    [spec.model_dump(mode="json") for spec in specs], ensure_ascii=False
                ),
                "grants_json": json.dumps(
                    [grant.model_dump(mode="json") for grant in grants],
                    ensure_ascii=False,
                ),
            },
            ensure_ascii=False,
        ),
    )
    return json.loads(raw)


class TestUnderGrantedPolicy:
    """under_granted 不构成拒绝理由（动态工具在执行阶段才创建并授权）。"""

    async def test_under_granted_still_passes_with_hint(self, tools):
        # 声明 write_file + 尚未注册的动态工具，当前仅授权 write_file
        specs = [make_spec("s1", ["write_file", "gen_tool_x"])]
        grants = [make_grant("g1", ["write_file"])]
        payload = await invoke_necessity(tools, specs, grants)
        assert payload["status"] == "ok"
        assert payload["under_granted"] == ["gen_tool_x"]
        assert payload["over_granted"] == []
        assert payload["empty_grants"] == []
        assert payload["suspicious_tools"] == []
        assert payload["passed"] is True  # under_granted 不在 passed 判定条件中
        # notes 为提示性措辞：提醒补发授权而非判定失败
        assert any("尚未授权" in note for note in payload["notes"])
        assert any("自动授权" in note for note in payload["notes"])

    async def test_all_tools_ungranted_still_passes(self, tools):
        # 全部声明工具均未授权（如全部为待创建的动态工具）→ 仍通过
        specs = [make_spec("s1", ["gen_tool_a", "gen_tool_b"])]
        payload = await invoke_necessity(tools, specs, [])
        assert payload["under_granted"] == ["gen_tool_a", "gen_tool_b"]
        assert payload["passed"] is True

    async def test_exact_match_passes_without_notes(self, tools):
        specs = [make_spec("s1", ["write_file"])]
        grants = [make_grant("g1", ["write_file"])]
        payload = await invoke_necessity(tools, specs, grants)
        assert payload["passed"] is True
        assert payload["under_granted"] == []
        assert payload["over_granted"] == []
        assert payload["notes"] == []


class TestRejectionConditions:
    """超额授权/空授权/可疑宽泛工具才构成拒绝（passed=false）。"""

    async def test_over_granted_fails(self, tools):
        specs = [make_spec("s1", ["write_file"])]
        grants = [make_grant("g1", ["write_file", "extra_tool"])]
        payload = await invoke_necessity(tools, specs, grants)
        assert payload["over_granted"] == ["extra_tool"]
        assert payload["passed"] is False
        assert any("收紧授权" in note for note in payload["notes"])

    async def test_empty_grants_fails(self, tools):
        specs = [make_spec("s1", ["write_file"])]
        grants = [make_grant("g1", [])]
        payload = await invoke_necessity(tools, specs, grants)
        assert payload["empty_grants"] == ["g1"]
        assert payload["passed"] is False

    async def test_suspicious_tool_fails(self, tools):
        specs = [make_spec("s1", ["shell"])]
        grants = [make_grant("g1", ["shell"])]
        payload = await invoke_necessity(tools, specs, grants)
        assert payload["suspicious_tools"] == ["shell"]
        assert payload["passed"] is False

    async def test_invalid_input_returns_error(self, tools):
        raw = await tools["check_tool_necessity"].on_invoke_tool(
            make_ctx("check_tool_necessity"),
            json.dumps({"subtasks_json": "not-json", "grants_json": "[]"}),
        )
        assert json.loads(raw)["status"] == "error"


class TestExtraInstructionsPolicy:
    """提示词中的授权评审通过/拒绝策略与工具行为保持一致。"""

    def test_policy_declared_in_extra_instructions(self):
        text = review_mod._EXTRA_INSTRUCTIONS
        assert "授权评审的通过/拒绝策略" in text
        assert "under_granted（声明但未授权）不构成拒绝理由" in text
        assert "动态工具需在执行阶段创建后才能授权" in text

    def test_prompt_loads_policy(self):
        # _load_prompt 拼接 prompt.md 与 _EXTRA_INSTRUCTIONS，策略随之生效
        assert "授权评审的通过/拒绝策略" in review_mod._load_prompt()

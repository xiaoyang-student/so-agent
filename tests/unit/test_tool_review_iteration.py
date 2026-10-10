"""单元测试：工具评审不通过后的 1 次迭代修复流程（``_create_and_approve_tool``）。

锁定迭代修复策略：
- 首次评审不通过 → 评审反馈传回代码 Agent（修正模式）→ 重走沙箱验收
  + 评审；
- 第 2 次评审通过 → 注册（approved）；第 2 次仍不通过 → 才真正标记
  失败（gap_unresolved）；
- 修正调用本身失败 → 直接失败，不再重试；
- 首次评审即通过 → 不触发修正调用。
"""

from __future__ import annotations

from types import SimpleNamespace

import so_agent.tool_packages.code_agent.agent as code_agent_agent_mod
from so_agent.config import Settings
from so_agent.context import ProjectContext
from so_agent.models import GeneratedToolManifest, ReviewDecision, TaskRequest
from so_agent.runtime.tool_registry import ToolRegistry
from so_agent.workflow import WorkflowEngine

TOOL = "demo_tool"


def make_registry(tmp_path) -> ToolRegistry:
    """预置一个 pending 状态的候选工具（无 test_cases → 沙箱验收跳过）。"""
    registry = ToolRegistry(generated_tools_dir=tmp_path / "generated_tools")
    manifest = GeneratedToolManifest(
        tool_id="t1",
        tool_name=TOOL,
        entry_file=f"{TOOL}.py",
        description="演示工具",
        created_by_task="seed",
        review_status="pending",
    )
    registry.register_generated_tool(manifest)
    return registry


def make_engine(tmp_path, monkeypatch, registry) -> WorkflowEngine:
    context = ProjectContext(sandbox_dir=tmp_path, project_name="tool-review-iteration")
    engine = WorkflowEngine(
        context,
        settings=Settings(_env_file=None),
        orchestrator=SimpleNamespace(name="fake-supervisor"),
        review_agent=SimpleNamespace(name="fake-reviewer"),
        tool_registry=registry,
    )
    engine._task_request = TaskRequest(
        task_id="t1", objective="演示工具评审迭代修复流程"
    )
    # 注册/写回路径指向临时目录，避免污染真实 sandbox/generated_tools
    monkeypatch.setattr(
        code_agent_agent_mod, "GENERATED_REGISTRY_PATH", tmp_path / "registry.json"
    )
    monkeypatch.setattr(
        code_agent_agent_mod, "GENERATED_TOOLS_DIR", tmp_path / "generated_tools"
    )
    return engine


def install_review_script(engine, monkeypatch, results):
    """按调用次序返回评审结论（超出脚本长度时钳制到最后一项）。"""
    calls: list = []

    async def fake_review(review_input):
        calls.append(review_input)
        passed = results[min(len(calls) - 1, len(results) - 1)]
        return ReviewDecision(
            stage="plan_review",
            passed=passed,
            issues=[] if passed else ["候选工具输入输出契约不完整"],
            required_fixes=[] if passed else ["补充 input_schema 与错误处理"],
            summary="工具评审（模拟）",
        )

    monkeypatch.setattr(engine._deps, "call_review_agent", fake_review)
    return calls


def install_fix_stub(engine, monkeypatch, *, error: str | None = None):
    """替换代码 Agent 调用（记录工具名与评审反馈；可选返回失败原因）。"""
    fix_calls: list[tuple[str, str | None]] = []

    async def fake_fix(tool_name, *, feedback=None):
        fix_calls.append((tool_name, feedback))
        return error

    monkeypatch.setattr(
        engine._execution_coordinator, "_invoke_code_agent_for_tool", fake_fix
    )
    return fix_calls


class TestReviewIteration:
    """评审不通过后的迭代修复流程。"""

    async def test_first_fail_then_fix_and_pass(self, tmp_path, monkeypatch):
        registry = make_registry(tmp_path)
        engine = make_engine(tmp_path, monkeypatch, registry)
        review_calls = install_review_script(engine, monkeypatch, [False, True])
        fix_calls = install_fix_stub(engine, monkeypatch)

        reason = await engine._create_and_approve_tool(TOOL)

        assert reason is None
        assert len(review_calls) == 2  # 修正后重新送审
        assert len(fix_calls) == 1
        tool_name, feedback = fix_calls[0]
        assert tool_name == TOOL
        assert feedback is not None and "输入输出契约不完整" in feedback
        assert registry.is_approved(TOOL) is True
        assert registry.get_manifest(TOOL).review_status == "approved"

    async def test_second_fail_marks_unresolved(self, tmp_path, monkeypatch):
        registry = make_registry(tmp_path)
        engine = make_engine(tmp_path, monkeypatch, registry)
        review_calls = install_review_script(engine, monkeypatch, [False, False])
        fix_calls = install_fix_stub(engine, monkeypatch)

        reason = await engine._create_and_approve_tool(TOOL)

        assert reason is not None
        assert "工具评审未通过" in reason
        assert len(review_calls) == 2
        assert len(fix_calls) == 1  # 仅一次修正机会
        assert registry.is_approved(TOOL) is False
        assert registry.get_manifest(TOOL).review_status == "pending"

    async def test_first_pass_skips_fix(self, tmp_path, monkeypatch):
        registry = make_registry(tmp_path)
        engine = make_engine(tmp_path, monkeypatch, registry)
        review_calls = install_review_script(engine, monkeypatch, [True])
        fix_calls = install_fix_stub(engine, monkeypatch)

        reason = await engine._create_and_approve_tool(TOOL)

        assert reason is None
        assert len(review_calls) == 1
        assert fix_calls == []
        assert registry.is_approved(TOOL) is True

    async def test_fix_invocation_failure_fails_fast(self, tmp_path, monkeypatch):
        registry = make_registry(tmp_path)
        engine = make_engine(tmp_path, monkeypatch, registry)
        review_calls = install_review_script(engine, monkeypatch, [False])
        fix_calls = install_fix_stub(
            engine, monkeypatch, error="代码 Agent MCP 通道不可用"
        )

        reason = await engine._create_and_approve_tool(TOOL)

        assert reason is not None
        assert "工具评审未通过且修正调用失败" in reason
        assert len(review_calls) == 1
        assert len(fix_calls) == 1
        assert registry.is_approved(TOOL) is False

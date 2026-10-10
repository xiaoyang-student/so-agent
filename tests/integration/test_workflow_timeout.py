"""集成测试：Agent 级超时控制（pipeline 整体超时 + 主管单次调用超时）。

模拟场景（无真实模型调用，monkeypatch 替换 workflow 模块内的 Runner）：

1. pipeline 整体超时：Runner 长时间挂起 + 极短 ``max_pipeline_timeout``
   → 任务优雅降级为 ``human_required``，且携带已完成的部分结果
   （本地确定性摘要并入 final_result），不崩溃、不丢已完成工作；
2. 主管单次调用超时：``_run_supervisor`` 超过
   ``supervisor_total_timeout`` 时抛 ``asyncio.TimeoutError``
   （任务 33 嵌套修复后，主管绝对总预算独立于子 Agent 的
   ``agent_total_timeout``）；
3. 执行阶段主管卡死：单次调用超时被映射为该次尝试的失败结果
   （不崩溃），走完"三次尝试 → 替换 → 分解问题 → 重规划额度耗尽"
   升级链路后转人工兜底，失败历史中留有 TimeoutError 证据。
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

import so_agent.workflow as workflow_mod
from so_agent.config import Settings
from so_agent.context import ProjectContext
from so_agent.models import (
    ExecutionResult,
    ReviewDecision,
    SubtaskSpec,
    TaskStatus,
)
from so_agent.orchestrator import OrchestratorOutput, PlanningOutput
from so_agent.workflow import WorkflowEngine


def make_spec(subtask_id: str, *, tools: list[str] | None = None) -> SubtaskSpec:
    return SubtaskSpec(
        subtask_id=subtask_id,
        title=f"模拟子任务 {subtask_id}",
        instructions=f"完成子任务 {subtask_id}",
        role="code",
        allowed_tools=list(tools or []),
        max_attempts=3,
    )


class FakeRunResult:
    """假 RunResult：只保留 final_output。"""

    def __init__(self, final_output):
        self.final_output = final_output


class FakeStreamedResult:
    """假 RunResultStreaming：首次取流式事件时求值协程工厂。

    挂起场景（handler 长时间不返回）下，``stream_events`` 的首次
    ``__anext__`` 会悬挂，由 ``run_agent_streamed`` 的 wait_for 取消并
    抛 ``asyncio.TimeoutError``；``cancel`` 记录超时收尾路径。
    """

    def __init__(self, run_call):
        self._run_call = run_call
        self.final_output = None
        self.raw_responses = []
        self.cancelled = False

    def cancel(self, mode="immediate"):
        self.cancelled = True

    async def stream_events(self):
        result = await self._run_call()
        self.final_output = result.final_output
        self.raw_responses = getattr(result, "raw_responses", None) or []
        yield {"type": "fake_event"}


async def _hang_forever(is_review: bool, raw_input: str):
    """所有阶段都长时间挂起，等待外层 wait_for 超时取消。"""
    await asyncio.sleep(30)


async def _plan_ok_then_hang(is_review: bool, raw_input: str):
    """计划与计划评审正常返回；之后的一切主管调用都挂起。"""
    if is_review:
        return ReviewDecision(
            stage="plan_review", passed=True, summary="计划评审（模拟）"
        )
    payload = json.loads(raw_input)
    phase = payload.get("phase")
    if phase in ("planning", "replanning"):
        specs = [make_spec("s1")]
        return OrchestratorOutput(
            task_id=str(payload.get("task_id") or "stub-task"),
            status=TaskStatus.PLANNING,
            plan_versions=[[spec.model_copy(deep=True) for spec in specs]],
        )
    await asyncio.sleep(30)  # execute_subtask / aggregating 等阶段全部挂起


def build_engine(tmp_path, monkeypatch, *, settings, handler) -> WorkflowEngine:
    """组装工作流引擎：注入按身份分派的假 Runner（handler(is_review, input)）。"""
    context = ProjectContext(sandbox_dir=tmp_path, project_name="timeout-test")
    orchestrator = SimpleNamespace(name="fake-supervisor")
    review_agent = SimpleNamespace(name="fake-reviewer")

    class FakeRunner:
        @classmethod
        async def run(cls, starting_agent, input=None, *, max_turns=None, **kwargs):
            output = await handler(starting_agent is review_agent, input or "")
            return FakeRunResult(output)

        @classmethod
        def run_streamed(cls, starting_agent, input=None, *, max_turns=None, **kwargs):
            return FakeStreamedResult(
                lambda: cls.run(
                    starting_agent, input=input, max_turns=max_turns, **kwargs
                )
            )

    monkeypatch.setattr(workflow_mod, "Runner", FakeRunner)
    return WorkflowEngine(
        context,
        settings=settings,
        orchestrator=orchestrator,
        review_agent=review_agent,
    )


class TestPipelineTimeout:
    """流水线整体超时：优雅降级并携带已完成的部分结果。"""

    async def test_timeout_degrades_with_partial_results(self, tmp_path, monkeypatch):
        settings = Settings(_env_file=None, max_pipeline_timeout=0.1)
        engine = build_engine(
            tmp_path, monkeypatch, settings=settings, handler=_hang_forever
        )
        # 预置一个已完成的子任务与计划：超时降级必须携带它返回
        engine._subtasks = [make_spec("s_done"), make_spec("s_hang")]
        engine._results["s_done"] = ExecutionResult(
            subtask_id="s_done",
            success=True,
            output="已完成的部分产出（模拟），用于验证超时降级保留成果",
        )

        output = await engine.run("永远完不成的任务")

        # 优雅降级：人工兜底终态 + 部分结果摘要 + 升级材料包
        assert output.status is TaskStatus.HUMAN_REQUIRED
        assert "超时" in output.final_result
        assert "s_done" in output.final_result
        assert "已完成的部分产出" in output.final_result
        assert output.human_escalation is not None
        assert "1/2" in output.final_result

    async def test_timeout_without_partial_results_still_degrades(
        self, tmp_path, monkeypatch
    ):
        settings = Settings(_env_file=None, max_pipeline_timeout=0.05)
        engine = build_engine(
            tmp_path, monkeypatch, settings=settings, handler=_hang_forever
        )

        output = await engine.run("没有任何进展的任务")

        # 无已完成结果时同样优雅降级（不崩溃、不悬挂）
        assert output.status is TaskStatus.HUMAN_REQUIRED
        assert "超时" in output.final_result
        assert "0/0" in output.final_result


class TestSupervisorCallTimeout:
    """主管单次调用超时：_run_supervisor 超过 supervisor_total_timeout 抛错。"""

    async def test_supervisor_timeout_raises(self, tmp_path, monkeypatch):
        settings = Settings(
            _env_file=None,
            supervisor_total_timeout=0.05,
            subagent_tool_execution_timeout=0.04,
        )
        engine = build_engine(
            tmp_path, monkeypatch, settings=settings, handler=_hang_forever
        )

        with pytest.raises(asyncio.TimeoutError):
            await engine._run_supervisor({"phase": "planning"}, PlanningOutput)


class TestExecutionTimeoutEscalation:
    """执行阶段主管卡死：单次超时映射为失败结果，走升级链路转人工。"""

    async def test_execution_timeout_maps_to_failure_and_escalates(
        self, tmp_path, monkeypatch
    ):
        settings = Settings(
            _env_file=None,
            supervisor_total_timeout=0.05,
            subagent_tool_execution_timeout=0.04,
        )
        engine = build_engine(
            tmp_path, monkeypatch, settings=settings, handler=_plan_ok_then_hang
        )

        output = await engine.run("执行阶段卡死的任务")

        # 不崩溃：超时被优雅处理，最终转人工兜底
        assert output.status is TaskStatus.HUMAN_REQUIRED
        assert output.human_escalation is not None
        # 失败历史中留有超时证据（来自单次调用超时的失败结果）
        assert any(
            "TimeoutError" in item
            for item in output.human_escalation.agent_failure_history
        )

"""无进展看门狗：总时长充足但长时间无进展时优雅降级。

与 ``test_workflow_timeout`` 的绝对超时互补：
- 绝对上限充足（30s）时不因总时长误杀；
- 连续 ``pipeline_no_progress_timeout`` 无进展才判定停滞，转人工兜底
  并保留检查点，续跑时只补未完成子任务。
"""

from __future__ import annotations

import asyncio
import json
import time
from types import SimpleNamespace

import so_agent.workflow as workflow_mod
from so_agent.config import Settings
from so_agent.context import ProjectContext
from so_agent.models import ReviewDecision, SubtaskSpec, TaskStatus
from so_agent.orchestrator import OrchestratorOutput
from so_agent.workflow import WorkflowEngine


class FakeRunResult:
    def __init__(self, final_output):
        self.final_output = final_output


class FakeStreamedResult:
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


def make_spec(subtask_id: str) -> SubtaskSpec:
    return SubtaskSpec(
        subtask_id=subtask_id,
        title=f"模拟子任务 {subtask_id}",
        instructions=f"完成子任务 {subtask_id}",
        role="code",
        allowed_tools=[],
        max_attempts=3,
    )


async def _plan_ok_then_hang(is_review: bool, raw_input: str):
    if is_review:
        return ReviewDecision(
            stage="plan_review", passed=True, summary="计划评审（模拟）"
        )
    payload = json.loads(raw_input)
    phase = payload.get("phase")
    if phase in ("planning", "replanning"):
        return OrchestratorOutput(
            task_id=str(payload.get("task_id") or "stub-task"),
            status=TaskStatus.PLANNING,
            plan_versions=[[make_spec("s_hang")]],
        )
    await asyncio.sleep(30)


def build_engine(tmp_path, monkeypatch, *, settings) -> WorkflowEngine:
    context = ProjectContext(
        sandbox_dir=tmp_path, project_name="no-progress-test"
    )
    orchestrator = SimpleNamespace(name="fake-supervisor")
    review_agent = SimpleNamespace(name="fake-reviewer")

    class FakeRunner:
        @classmethod
        async def run(cls, starting_agent, input=None, *, max_turns=None, **kwargs):
            output = await _plan_ok_then_hang(
                starting_agent is review_agent, input or ""
            )
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


class TestNoProgressWatchdog:
    async def test_stall_escalates_before_absolute_timeout(
        self, tmp_path, monkeypatch
    ):
        settings = Settings(
            _env_file=None,
            pipeline_no_progress_timeout=0.2,
            max_pipeline_timeout=30.0,
            supervisor_total_timeout=10.0,
        )
        engine = build_engine(tmp_path, monkeypatch, settings=settings)

        started = time.monotonic()
        output = await engine.run("执行阶段长时间无进展的任务")
        elapsed = time.monotonic() - started

        assert output.status is TaskStatus.HUMAN_REQUIRED
        # 停滞原因与"未被固定总时长判定"语义同时写入最终结果
        assert "无进展" in output.final_result
        assert "未被固定总时长判定" in output.final_result
        assert engine._gate.stop_reason.startswith("no_progress_timeout")
        # 明确早于 30s 绝对兜底到期
        assert elapsed < 5.0
        # 停滞兜底同样携带检查点，供后续只补未完成子任务
        checkpoint = (
            tmp_path / "checkpoints" / f"{engine.task_request.task_id}.json"
        )
        assert checkpoint.is_file()
        assert output.human_escalation is not None

    async def test_disabled_watchdog_keeps_absolute_backstop(
        self, tmp_path, monkeypatch
    ):
        settings = Settings(
            _env_file=None,
            pipeline_no_progress_timeout=0.0,
            max_pipeline_timeout=0.2,
            supervisor_total_timeout=10.0,
        )
        engine = build_engine(tmp_path, monkeypatch, settings=settings)

        output = await engine.run("禁用看门狗后仅剩绝对兜底")

        assert output.status is TaskStatus.HUMAN_REQUIRED
        assert "超时" in output.final_result
        assert engine._gate.stop_reason == "pipeline_timeout"

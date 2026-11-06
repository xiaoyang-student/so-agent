"""断点/人工指导续跑：只执行未完成子任务，已完成成果不重跑。"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

import so_agent.workflow as workflow_mod
from so_agent.config import Settings
from so_agent.context import ProjectContext
from so_agent.models import (
    ExecutionResult,
    HumanEscalationRequest,
    HumanGuidance,
    ReviewDecision,
    SubtaskSpec,
    TaskRequest,
    TaskStatus,
)
from so_agent.orchestrator import OrchestratorOutput
from so_agent.state_machine import HumanGuidanceError
from so_agent.tool_packages.review_agent.agent import ReviewStage
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


def make_spec(subtask_id: str, deps: list[str] | None = None) -> SubtaskSpec:
    return SubtaskSpec(
        subtask_id=subtask_id,
        title=f"子任务 {subtask_id}",
        instructions=f"完成 {subtask_id}",
        role="code",
        dependencies=deps or [],
        allowed_tools=[],
        max_attempts=3,
    )


class FakeSupervisor:
    """只接受执行与汇总阶段；记录实际被执行的子任务。"""

    def __init__(self) -> None:
        self.executed: list[str] = []

    def __call__(self, payload: dict) -> OrchestratorOutput:
        phase = payload.get("phase")
        if phase == "execute_subtask":
            subtask_id = payload["subtask"]["subtask_id"]
            self.executed.append(subtask_id)
            result = ExecutionResult(
                subtask_id=subtask_id,
                success=True,
                output=f"{subtask_id} 的产出内容足够长，用于通过产出充分性检查。",
            )
            return OrchestratorOutput(
                task_id=str(payload.get("task_id") or "stub-task"),
                status=TaskStatus.EXECUTING,
                final_result=result.model_dump_json(),
            )
        if phase == "aggregating":
            return OrchestratorOutput(
                task_id=str(payload.get("task_id") or "stub-task"),
                status=TaskStatus.AGGREGATING,
                final_result="汇总结果",
            )
        raise AssertionError(f"续跑不应调用阶段：{phase}")


def review_decision(review_input) -> ReviewDecision:
    stage = (
        "result_review"
        if review_input.stage == ReviewStage.RESULT_REVIEW
        else "plan_review"
    )
    return ReviewDecision(
        stage=stage, passed=True, summary="评审通过（模拟）"
    )


def build_engine(tmp_path, monkeypatch) -> tuple[WorkflowEngine, FakeSupervisor]:
    context = ProjectContext(
        sandbox_dir=tmp_path, project_name="resume-incremental"
    )
    orchestrator = SimpleNamespace(name="fake-supervisor")
    review_agent = SimpleNamespace(name="fake-reviewer")
    supervisor = FakeSupervisor()

    class FakeRunner:
        @classmethod
        async def run(cls, starting_agent, input=None, *, max_turns=None, **kwargs):
            if starting_agent is review_agent:
                from so_agent.tool_packages.review_agent.agent import (
                    ReviewInput,
                )

                return FakeRunResult(
                    review_decision(ReviewInput.model_validate_json(input))
                )
            return FakeRunResult(supervisor(json.loads(input)))

        @classmethod
        def run_streamed(cls, starting_agent, input=None, *, max_turns=None, **kwargs):
            return FakeStreamedResult(
                lambda: cls.run(
                    starting_agent, input=input, max_turns=max_turns, **kwargs
                )
            )

    monkeypatch.setattr(workflow_mod, "Runner", FakeRunner)
    engine = WorkflowEngine(
        context,
        settings=Settings(_env_file=None),
        orchestrator=orchestrator,
        review_agent=review_agent,
    )
    return engine, supervisor


def seed_human_required(
    engine: WorkflowEngine,
    specs: list[SubtaskSpec],
    results: dict[str, ExecutionResult],
) -> None:
    request = TaskRequest(task_id="task-resume", objective="续跑测试")
    engine._task_request = request
    engine._state.current_status = TaskStatus.HUMAN_REQUIRED
    engine._subtasks = specs
    engine._plan_history = [[spec.model_copy(deep=True) for spec in specs]]
    engine._plan_version = 1
    engine._results = results
    engine._human_escalation = HumanEscalationRequest(
        task_id=request.task_id,
        original_request=request,
        reason="无进展超时",
    )


class TestResumeOnlyUnfinished:
    async def test_resume_skips_completed_subtask(
        self, tmp_path, monkeypatch
    ):
        engine, supervisor = build_engine(tmp_path, monkeypatch)
        seed_human_required(
            engine,
            [make_spec("s1"), make_spec("s2")],
            {
                "s1": ExecutionResult(
                    subtask_id="s1", success=True, output="s1 已完成"
                ),
                "s2": ExecutionResult(
                    subtask_id="s2", success=False, error="无进展超时"
                ),
            },
        )

        output = await engine.resume_with_guidance(
            HumanGuidance(action="resume")
        )

        assert output.status is TaskStatus.COMPLETED
        # 只有未完成的 s2 被重新执行，s1 的既有成果不重跑
        assert supervisor.executed == ["s2"]

    async def test_resume_with_all_completed_goes_straight_to_review(
        self, tmp_path, monkeypatch
    ):
        engine, supervisor = build_engine(tmp_path, monkeypatch)
        seed_human_required(
            engine,
            [make_spec("s1"), make_spec("s2")],
            {
                "s1": ExecutionResult(
                    subtask_id="s1", success=True, output="s1 已完成"
                ),
                "s2": ExecutionResult(
                    subtask_id="s2", success=True, output="s2 已完成"
                ),
            },
        )

        output = await engine.resume_with_guidance(
            HumanGuidance(action="resume")
        )

        assert output.status is TaskStatus.COMPLETED
        assert supervisor.executed == []

    async def test_restored_cancelled_checkpoint_resumes_unfinished(
        self, tmp_path, monkeypatch
    ):
        source, _ = build_engine(tmp_path, monkeypatch)
        seed_human_required(
            source,
            [make_spec("s1"), make_spec("s2", deps=["s1"])],
            {
                "s1": ExecutionResult(
                    subtask_id="s1", success=True, output="s1 已完成"
                )
            },
        )
        # 外部取消不会产生人工兜底请求，检查点状态仍是执行中
        source._human_escalation = None
        source._state.current_status = TaskStatus.EXECUTING
        payload = source.export_checkpoint()

        target, supervisor = build_engine(tmp_path, monkeypatch)
        target.restore_checkpoint(payload)
        output = await target.resume_with_guidance(
            HumanGuidance(action="resume")
        )

        assert output.status is TaskStatus.COMPLETED
        assert supervisor.executed == ["s2"]

    async def test_live_executing_engine_still_rejects_guidance(
        self, tmp_path, monkeypatch
    ):
        engine, _ = build_engine(tmp_path, monkeypatch)
        seed_human_required(
            engine,
            [make_spec("s1")],
            {
                "s1": ExecutionResult(
                    subtask_id="s1", success=True, output="s1 已完成"
                )
            },
        )
        engine._human_escalation = None
        engine._state.current_status = TaskStatus.EXECUTING

        with pytest.raises(HumanGuidanceError):
            await engine.resume_with_guidance(HumanGuidance(action="resume"))

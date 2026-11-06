"""集成测试：执行阶段持续失败 → 拆分额度耗尽 → 人工兜底（模拟 Runner）。

模拟场景（无真实模型调用，monkeypatch 替换 workflow 模块内的 Runner）：
1. 计划评审一次性通过（拆解本身没有问题），但执行阶段持续失败：
   首批触发"同因两次尝试 + 一次替换申请 + 替代同因两次失败 → 分解
   问题"，后续批次替换额度耗尽直接上报分解问题；
2. 每批失败消耗一次拆分修改额度并重规划（计划评审继续通过）；
3. 3 次额度耗尽 → 第四次失败批次后转入 human_required 终态，
   组装完整的人工升级材料包（原始任务、历代计划、失败历史、
   工具调用证据、最终评审结论）并交接处理器；
4. 状态链完整包含替换链（executing → replacement_requested →
   replacement_executing）与重规划循环（decomposition_issue → replanning）。
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
    HumanGuidance,
    ReviewDecision,
    SubtaskSpec,
    TaskStatus,
)
from so_agent.orchestrator import OrchestratorOutput
from so_agent.runtime.events import (
    EVENT_DECOMPOSITION_ISSUE,
    EVENT_HUMAN_ESCALATION,
    EVENT_HUMAN_GUIDANCE_APPLIED,
    EVENT_MESSAGE_ROUTED,
    EVENT_REPLACEMENT_REQUESTED,
    EVENT_REPLAN,
    EVENT_STATUS_TRANSITION,
    EVENT_TASK_DISPATCHED,
    EVENT_TOOL_GRANTED,
    ALL_EVENT_TYPES,
)
from so_agent.tool_packages.review_agent.agent import ReviewInput, ReviewStage
from so_agent.workflow import (
    HumanEscalationHandler,
    HumanGuidanceError,
    WorkflowEngine,
)


class FakeRunResult:
    """假 RunResult：只保留 final_output，供控制层结构与文本解析。"""

    def __init__(self, final_output):
        self.final_output = final_output


class FakeStreamedResult:
    """假 RunResultStreaming：首次取流式事件时求值协程工厂并回填输出。

    正常路径下取到假事件后迭代结束，返回对象（final_output 已就绪）；
    ``cancel`` 记录超时收尾路径。
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


def install_fake_runner(
    monkeypatch, *, orchestrator, review_agent, supervisor, reviews
):
    """把 workflow 模块内的 Runner 替换为按 Agent 身份分派的假 Runner。

    主管 Agent 经 ``run_streamed``（run_agent_streamed 的流式入口），
    评审仍走 ``run``。
    """

    class FakeRunner:
        @classmethod
        async def run(cls, starting_agent, input=None, *, max_turns=None, **kwargs):
            if starting_agent is review_agent:
                review_input = ReviewInput.model_validate_json(input)
                return FakeRunResult(reviews(review_input))
            payload = json.loads(input)
            return FakeRunResult(supervisor(payload))

        @classmethod
        def run_streamed(cls, starting_agent, input=None, *, max_turns=None, **kwargs):
            return FakeStreamedResult(
                lambda: cls.run(
                    starting_agent, input=input, max_turns=max_turns, **kwargs
                )
            )

    monkeypatch.setattr(workflow_mod, "Runner", FakeRunner)


class FakeSupervisor:
    """按 phase 分派的主管 Agent 模拟对象（执行阶段可配置成败）。"""

    def __init__(self, *, exec_success: bool = False):
        self.exec_success = exec_success
        self.payloads: list[dict] = []

    @property
    def phases(self) -> list[str]:
        return [payload.get("phase") for payload in self.payloads]

    def __call__(self, payload: dict) -> OrchestratorOutput:
        self.payloads.append(payload)
        phase = payload.get("phase")
        if phase in ("planning", "replanning"):
            return self._plan_output(payload)
        if phase == "execute_subtask":
            return self._execution_output(payload)
        if phase == "aggregating":
            return OrchestratorOutput(
                task_id=str(payload.get("task_id") or "stub-task"),
                status=TaskStatus.AGGREGATING,
                final_result="最终结果文本",
            )
        raise AssertionError(f"未预期的阶段载荷：{phase}")

    def _plan_output(self, payload: dict) -> OrchestratorOutput:
        spec = SubtaskSpec(
            subtask_id="s1",
            title="模拟子任务",
            instructions="完成模拟工作",
            role="code",
            allowed_tools=["write_file"],
            max_attempts=3,
        )
        return OrchestratorOutput(
            task_id=str(payload.get("task_id") or "stub-task"),
            status=(
                TaskStatus.REPLANNING
                if payload.get("phase") == "replanning"
                else TaskStatus.PLANNING
            ),
            plan_versions=[[spec]],
        )

    def _execution_output(self, payload: dict) -> OrchestratorOutput:
        subtask_id = payload["subtask"]["subtask_id"]
        attempt = payload.get("attempt", 1)
        if self.exec_success:
            result = ExecutionResult(
                subtask_id=subtask_id, success=True, output="模拟子任务产出"
            )
        else:
            result = ExecutionResult(
                subtask_id=subtask_id,
                success=False,
                error=f"模拟执行失败（第 {attempt} 次尝试）",
                evidence=[f"evidence-attempt-{attempt}"],
            )
        return OrchestratorOutput(
            task_id=str(payload.get("task_id") or "stub-task"),
            status=TaskStatus.EXECUTING,
            final_result=result.model_dump_json(),
        )


class ReviewScript:
    """计划评审按批次脚本返回结论（每批两轮）；结果评审固定通过。"""

    def __init__(self, plan_batch_results, *, result_passed: bool = True):
        self.plan_batch_results = list(plan_batch_results) or [True]
        self.result_passed = result_passed
        self.calls: list[ReviewInput] = []
        self.decisions: list[ReviewDecision] = []

    def __call__(self, review_input: ReviewInput) -> ReviewDecision:
        self.calls.append(review_input)
        if review_input.stage == ReviewStage.RESULT_REVIEW:
            decision = ReviewDecision(
                stage="result_review",
                passed=self.result_passed,
                issues=[] if self.result_passed else ["整体结果不满足验收标准"],
                required_fixes=[] if self.result_passed else ["修复整体结果后重新提交"],
                summary="结果评审（模拟）",
            )
        else:
            plan_calls = sum(
                1 for call in self.calls if call.stage != ReviewStage.RESULT_REVIEW
            )
            index = min((plan_calls - 1) // 2, len(self.plan_batch_results) - 1)
            passed = self.plan_batch_results[index]
            decision = ReviewDecision(
                stage="plan_review",
                passed=passed,
                issues=[] if passed else ["子任务拆分粒度过粗"],
                required_fixes=[] if passed else ["重新拆分并补充依赖"],
                summary=f"计划评审（模拟，第 {index + 1} 批）",
            )
        self.decisions.append(decision)
        return decision


def build_engine(tmp_path, monkeypatch, *, supervisor, reviews, settings=None):
    """组装工作流引擎：注入假主管/评审 Agent 与记录型人工兜底处理器。"""
    context = ProjectContext(sandbox_dir=tmp_path, project_name="escalation-flow")
    effective_settings = (
        settings or Settings(_env_file=None, max_replan_attempts=3)
    ).model_copy(
        update={
            "skip_plan_review_for_simple": False,
            "authorization_review_mode": "always",
        }
    )
    orchestrator = SimpleNamespace(name="fake-supervisor")
    review_agent = SimpleNamespace(name="fake-reviewer")
    handler = HumanEscalationHandler()
    install_fake_runner(
        monkeypatch,
        orchestrator=orchestrator,
        review_agent=review_agent,
        supervisor=supervisor,
        reviews=reviews,
    )
    engine = WorkflowEngine(
        context,
        settings=effective_settings,
        orchestrator=orchestrator,
        review_agent=review_agent,
        escalation_handler=handler,
    )
    return context, engine, handler


def status_transitions(context) -> list[tuple[str, str]]:
    """提取状态迁移序列 ``[(from_status, to_status), ...]``。"""
    return [
        (record["payload"]["from_status"], record["payload"]["to_status"])
        for record in context.event_log
        if record["event_type"] == EVENT_STATUS_TRANSITION
    ]


class TestPersistentExecutionFailure:
    async def test_escalates_after_replan_quota_exhausted(self, tmp_path, monkeypatch):
        supervisor = FakeSupervisor(exec_success=False)
        reviews = ReviewScript([True])
        context, engine, handler = build_engine(
            tmp_path, monkeypatch, supervisor=supervisor, reviews=reviews
        )

        output = await engine.run("实现数据处理管道")

        # 终态与治理额度
        assert output.status == TaskStatus.HUMAN_REQUIRED
        assert engine.current_status == TaskStatus.HUMAN_REQUIRED
        assert engine.replan_count == 3
        assert engine.plan_version == 4
        assert len(output.plan_versions) == 4
        assert output.final_result == (
            "任务需要人工介入：执行阶段持续失败：任务拆分修改额度耗尽"
        )
        # 最后一次评审（计划评审）是通过的：问题出在执行侧
        assert "通过" in output.review_summary

        # 主管调用统计：
        # - 规划 1 次 + 重规划 3 次（每批失败消耗一次额度）
        # - 执行子任务 10 次 = 首批 4 次（原始 2 + 替代 2，同因抑制）
        #   + 其后每批 2 次（替换额度耗尽后仅原始 Agent 两次尝试）
        # - 从未进入汇总阶段
        phases = supervisor.phases
        assert phases.count("planning") == 1
        assert phases.count("replanning") == 3
        assert phases.count("execute_subtask") == 10
        assert phases.count("aggregating") == 0

        # 计划评审 4 批 × 2 轮全部通过；从未进行结果评审
        assert len(reviews.calls) == 8
        assert all(call.stage != ReviewStage.RESULT_REVIEW for call in reviews.calls)
        assert all(decision.passed for decision in reviews.decisions)

        # Agent 生命周期记录留存：首批 2 个（原始 + 替代），其后每批 1 个
        assert len(output.agent_history) == 5

        # 人工兜底已触发
        assert len(handler.pending_requests) == 1
        assert output.human_escalation is not None

    async def test_status_sequence_includes_replacement_chain(
        self, tmp_path, monkeypatch
    ):
        supervisor = FakeSupervisor(exec_success=False)
        reviews = ReviewScript([True])
        context, engine, _ = build_engine(
            tmp_path, monkeypatch, supervisor=supervisor, reviews=reviews
        )

        await engine.run("实现数据处理管道")

        assert status_transitions(context) == [
            ("received", "planning"),
            ("planning", "plan_review"),
            ("plan_review", "executing"),
            ("executing", "replacement_requested"),
            ("replacement_requested", "replacement_executing"),
            ("replacement_executing", "decomposition_issue"),
            ("decomposition_issue", "replanning"),
            ("replanning", "plan_review"),
            ("plan_review", "executing"),
            ("executing", "decomposition_issue"),
            ("decomposition_issue", "replanning"),
            ("replanning", "plan_review"),
            ("plan_review", "executing"),
            ("executing", "decomposition_issue"),
            ("decomposition_issue", "replanning"),
            ("replanning", "plan_review"),
            ("plan_review", "executing"),
            ("executing", "decomposition_issue"),
            ("decomposition_issue", "human_required"),
        ]


class TestEscalationMaterialPackage:
    async def test_material_package_contains_full_evidence(self, tmp_path, monkeypatch):
        supervisor = FakeSupervisor(exec_success=False)
        reviews = ReviewScript([True])
        context, engine, handler = build_engine(
            tmp_path, monkeypatch, supervisor=supervisor, reviews=reviews
        )

        output = await engine.run("实现数据处理管道")

        request = handler.pending_requests[0]
        assert request.task_id == output.task_id
        assert request.original_request.objective == "实现数据处理管道"

        # 历代计划快照：首版 + 3 次重规划 = 4 版
        assert len(request.plan_history) == 4
        assert all(plan and plan[0].subtask_id == "s1" for plan in request.plan_history)

        # 失败历史：1 条替换申请 + 4 条分解问题（每批一条）
        assert len(request.agent_failure_history) == 5
        assert any("替换申请" in item for item in request.agent_failure_history)
        assert any("子任务 s1" in item for item in request.agent_failure_history)

        # 工具调用证据：包含调度、授权与 Agent 生命周期事件（JSON 文本）
        assert request.tool_call_evidence
        evidence_text = "\n".join(request.tool_call_evidence)
        assert EVENT_TASK_DISPATCHED in evidence_text
        assert EVENT_TOOL_GRANTED in evidence_text
        assert EVENT_REPLACEMENT_REQUESTED in evidence_text

        # 最终评审结论为通过（拆解无问题，失败在执行侧）
        assert request.final_review is not None
        assert request.final_review.passed is True

        # 首版人工处理器不可解决（占位实现）
        assert await handler.check_resolution(output.task_id) is False

        # 材料包可完整 JSON 序列化（审计载体）
        payload = request.model_dump(mode="json")
        assert payload["task_id"] == output.task_id

        # 确定性审计报告随终态输出返回
        assert output.audit is not None
        assert output.audit.status == "human_required"
        assert output.audit.verdict == "warn"
        assert output.audit.counts["failed"] >= 1
        assert "human_required" in {
            finding.code for finding in output.audit.findings
        }

    async def test_escalation_event_and_core_event_chain(self, tmp_path, monkeypatch):
        supervisor = FakeSupervisor(exec_success=False)
        reviews = ReviewScript([True])
        context, engine, handler = build_engine(
            tmp_path, monkeypatch, supervisor=supervisor, reviews=reviews
        )

        await engine.run("实现数据处理管道")

        def count(event_type: str) -> int:
            return sum(
                1
                for record in context.event_log
                if record["event_type"] == event_type
            )

        # 每条事件类型均合法（记录时已校验），核心升级链计数如下：
        assert count(EVENT_REPLACEMENT_REQUESTED) == 1  # 替换额度仅 1 次
        assert count(EVENT_DECOMPOSITION_ISSUE) == 4  # 每批失败一次
        assert count(EVENT_REPLAN) == 3  # 重规划事件（计划版本 2/3/4）
        assert count(EVENT_HUMAN_ESCALATION) == 1
        # 1 次替换批准上报 + 4 次分解问题上报
        assert count(EVENT_MESSAGE_ROUTED) == 5

        escalation_records = [
            record
            for record in context.event_log
            if record["event_type"] == EVENT_HUMAN_ESCALATION
        ]
        payload = escalation_records[0]["payload"]
        assert "执行阶段持续失败" in payload["reason"]
        assert payload["request"]["task_id"] == engine.task_request.task_id
        assert payload["request"]["plan_history"]

        assert all(
            record["event_type"] in ALL_EVENT_TYPES for record in context.event_log
        )


class TestHumanGuidanceResume:
    """人工指导闭环：材料包结构化进展 + 指定回退点后恢复执行。"""

    async def _escalated_engine(self, tmp_path, monkeypatch):
        supervisor = FakeSupervisor(exec_success=False)
        reviews = ReviewScript([True])
        context, engine, handler = build_engine(
            tmp_path, monkeypatch, supervisor=supervisor, reviews=reviews
        )
        output = await engine.run("实现数据处理管道")
        assert output.status == TaskStatus.HUMAN_REQUIRED
        return supervisor, context, engine, handler, output

    async def test_material_contains_progress_and_rollback_targets(
        self, tmp_path, monkeypatch
    ):
        _, _, _, _, output = await self._escalated_engine(tmp_path, monkeypatch)

        request = output.human_escalation
        assert request is not None
        assert "执行阶段持续失败" in request.reason
        assert request.rollback_targets == ["s1"]
        assert len(request.progress) == 1
        progress = request.progress[0]
        assert progress.subtask_id == "s1"
        assert progress.status == "failed"
        assert "模拟执行失败" in progress.error

    async def test_rollback_resumes_and_completes(self, tmp_path, monkeypatch):
        supervisor, context, engine, _, _ = await self._escalated_engine(
            tmp_path, monkeypatch
        )
        supervisor.exec_success = True
        review_calls_before = len(
            [p for p in supervisor.payloads if p.get("phase") == "execute_subtask"]
        )

        output = await engine.resume_with_guidance(
            HumanGuidance(
                action="rollback",
                rollback_to=["s1"],
                instructions="修正输出格式并重跑",
                additional_info="产物必须使用 UTF-8",
            )
        )

        assert output.status == TaskStatus.COMPLETED
        assert engine.current_status == TaskStatus.COMPLETED
        assert output.human_escalation is None

        # 确定性审计报告随终态输出返回（历史失败会被如实记为警告，但不判失败）
        assert output.audit is not None
        assert output.audit.status == "completed"
        assert output.audit.verdict in {"pass", "warn"}
        assert all(finding.level != "error" for finding in output.audit.findings)
        assert output.audit.counts["guidance_applied"] == 1
        assert output.audit.counts["completed"] == 1

        # 指导文本进入回退子任务的执行指令，并随执行载荷下发
        instructions = engine.subtasks[0].instructions
        assert "人工指导：修正输出格式并重跑" in instructions
        assert "人工补充信息：产物必须使用 UTF-8" in instructions
        exec_payloads = [
            payload
            for payload in supervisor.payloads
            if payload.get("phase") == "execute_subtask"
        ]
        assert len(exec_payloads) == review_calls_before + 1
        assert "人工指导" in exec_payloads[-1]["subtask"]["instructions"]

        # 审计事件：动作、影响集与补充信息标记（不含全文）
        records = [
            record
            for record in context.event_log
            if record["event_type"] == EVENT_HUMAN_GUIDANCE_APPLIED
        ]
        assert len(records) == 1
        payload = records[0]["payload"]
        assert payload["action"] == "rollback"
        assert payload["rollback_to"] == ["s1"]
        assert payload["has_instructions"] is True
        assert payload["has_additional_info"] is True
        assert "instructions" not in payload

    async def test_resume_retries_failed_subtask_without_targets(
        self, tmp_path, monkeypatch
    ):
        supervisor, _, engine, _, _ = await self._escalated_engine(
            tmp_path, monkeypatch
        )
        supervisor.exec_success = True

        output = await engine.resume_with_guidance(
            HumanGuidance(action="resume", additional_info="请重试")
        )

        assert output.status == TaskStatus.COMPLETED
        assert "请重试" in engine.subtasks[0].instructions

    async def test_abort_keeps_human_required(self, tmp_path, monkeypatch):
        _, _, engine, _, _ = await self._escalated_engine(tmp_path, monkeypatch)

        output = await engine.resume_with_guidance(
            HumanGuidance(action="abort")
        )

        assert output.status == TaskStatus.HUMAN_REQUIRED
        assert engine.current_status == TaskStatus.HUMAN_REQUIRED
        assert engine.human_escalation is not None

    async def test_unknown_rollback_target_is_rejected(self, tmp_path, monkeypatch):
        _, _, engine, _, _ = await self._escalated_engine(tmp_path, monkeypatch)

        with pytest.raises(HumanGuidanceError, match="不在当前计划中"):
            await engine.resume_with_guidance(
                HumanGuidance(action="rollback", rollback_to=["missing"])
            )
        assert engine.current_status == TaskStatus.HUMAN_REQUIRED

    async def test_resume_requires_human_required_state(self, tmp_path, monkeypatch):
        supervisor = FakeSupervisor(exec_success=True)
        reviews = ReviewScript([True], result_passed=True)
        _, engine, _ = build_engine(
            tmp_path, monkeypatch, supervisor=supervisor, reviews=reviews
        )

        with pytest.raises(HumanGuidanceError, match="不在人工兜底状态"):
            await engine.resume_with_guidance(HumanGuidance(action="resume"))

"""集成测试：任务拆分修改（replan）循环与人工兜底（模拟 Runner，不消耗真实 API）。

模拟场景（无真实模型调用，monkeypatch 替换 workflow 模块内的 Runner）：
1. 计划评审连续不通过 → 每轮消耗一次拆分修改额度（默认上限 3 次），
   额度耗尽后转入 human_required 终态，并把人工升级材料包交接给处理器；
2. 重规划输入必须携带上一轮评审反馈（review_feedback）与上一版计划
   快照（previous_plan），阶段标记按 planning → replanning 演进；
3. 每版被采纳的计划都记录 plan_version / replan_count 递增的重规划事件；
4. 拆解结果结构非法（未通过基础校验）时同样消耗额度且不进入评审阶段；
5. 额度以内（第三次拆解）评审通过 → 继续执行、汇总与终审并完成任务；
6. 主管 Agent 网络异常（超时 / 连接失败）与计划结构问题分开处理：
   独立网络重试预算（3 次）不消耗拆分修改额度，重试进度提示打印到
   stderr；网络连续失败耗尽预算 → human_required（原因网络连续失败）。
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import httpx2
import openai

import so_agent.workflow as workflow_mod
from so_agent.config import Settings
from so_agent.context import ProjectContext
from so_agent.models import (
    ExecutionResult,
    ReviewDecision,
    SubtaskSpec,
    TaskStatus,
)
from so_agent.orchestrator import OrchestratorOutput
from so_agent.runtime.events import (
    EVENT_HUMAN_ESCALATION,
    EVENT_REPLAN,
    EVENT_STATUS_TRANSITION,
)
from so_agent.tool_packages.review_agent.agent import ReviewInput, ReviewStage
from so_agent.workflow import HumanEscalationHandler, WorkflowEngine


class FakeRunResult:
    """假 RunResult：只保留 final_output，供控制层结构与文本解析。"""

    def __init__(self, final_output):
        self.final_output = final_output


def install_fake_runner(
    monkeypatch, *, orchestrator, review_agent, supervisor, reviews
):
    """把 workflow 模块内的 Runner 替换为按 Agent 身份分派的假 Runner。

    workflow 模块以 ``Runner.run(agent, input=...)`` 调用主管/评审 Agent；
    假 Runner 按 ``starting_agent`` 身份把输入分派给 supervisor / reviews
    两个可调用对象（返回 OrchestratorOutput / ReviewDecision）。
    """

    class FakeRunner:
        @classmethod
        async def run(cls, starting_agent, input=None, *, max_turns=None, **kwargs):
            if starting_agent is review_agent:
                review_input = ReviewInput.model_validate_json(input)
                return FakeRunResult(reviews(review_input))
            payload = json.loads(input)
            return FakeRunResult(supervisor(payload))

    monkeypatch.setattr(workflow_mod, "Runner", FakeRunner)


class FakeSupervisor:
    """按 phase 分派的主管 Agent 模拟对象，并记录收到的全部 JSON 载荷。"""

    def __init__(self, *, invalid_plan: bool = False):
        self.invalid_plan = invalid_plan
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
        if self.invalid_plan:
            # 结构非法：缺少 plan_versions（触发基础校验失败分支）
            plan_versions: list[list[SubtaskSpec]] = []
        else:
            plan_versions = [
                [
                    SubtaskSpec(
                        subtask_id="s1",
                        title="模拟子任务",
                        instructions="完成模拟工作",
                        role="code",
                        allowed_tools=["write_file"],
                        max_attempts=3,
                    )
                ]
            ]
        return OrchestratorOutput(
            task_id=str(payload.get("task_id") or "stub-task"),
            status=(
                TaskStatus.REPLANNING
                if payload.get("phase") == "replanning"
                else TaskStatus.PLANNING
            ),
            plan_versions=plan_versions,
        )

    def _execution_output(self, payload: dict) -> OrchestratorOutput:
        subtask_id = payload["subtask"]["subtask_id"]
        result = ExecutionResult(
            subtask_id=subtask_id, success=True, output="模拟子任务产出"
        )
        return OrchestratorOutput(
            task_id=str(payload.get("task_id") or "stub-task"),
            status=TaskStatus.EXECUTING,
            final_result=result.model_dump_json(),
        )


def _network_timeout_error() -> openai.APITimeoutError:
    """构造真实的 openai 超时异常（携带最小 httpx2 请求对象）。"""
    return openai.APITimeoutError(
        request=httpx2.Request("POST", "https://example.invalid/v1/chat/completions")
    )


class FlakyNetworkSupervisor:
    """网络抖动模拟：前 ``failures`` 次调用抛超时异常，之后交给 FakeSupervisor。

    ``failures=None`` 表示始终抛网络异常（用于验证网络重试预算耗尽）；
    ``calls`` 记录总调用次数（含失败调用）。
    """

    def __init__(self, *, failures: int | None) -> None:
        self.failures = failures
        self.calls = 0
        self.inner = FakeSupervisor()

    @property
    def phases(self) -> list[str]:
        return self.inner.phases

    def __call__(self, payload: dict) -> OrchestratorOutput:
        self.calls += 1
        if self.failures is None or self.calls <= self.failures:
            raise _network_timeout_error()
        return self.inner(payload)


class ReviewScript:
    """评审 Agent 模拟脚本。

    - 计划评审按"批次"给出结论：每批含两轮调用（计划评审 + 授权评审），
      同批两轮返回同一结论；``plan_batch_results`` 指定每批是否通过。
    - 结果评审按 ``result_passed`` 固定返回。
    - 记录全部调用输入（``calls``）与结论（``decisions``）供断言。
    """

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
    context = ProjectContext(sandbox_dir=tmp_path, project_name="replan-flow")
    effective_settings = settings or Settings(_env_file=None, max_replan_attempts=3)
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


class TestPlanReviewQuotaExhaustion:
    async def test_review_failures_exhaust_quota_and_escalate(self, tmp_path, monkeypatch):
        supervisor = FakeSupervisor()
        reviews = ReviewScript([False, False, False, False])
        context, engine, handler = build_engine(
            tmp_path, monkeypatch, supervisor=supervisor, reviews=reviews
        )

        output = await engine.run(
            "编写一个演示 CLI 工具", constraints="使用 Python; 保持简洁"
        )

        # 终态与治理额度：修改 3 次后额度耗尽
        assert output.status == TaskStatus.HUMAN_REQUIRED
        assert engine.current_status == TaskStatus.HUMAN_REQUIRED
        assert engine.replan_count == 3
        assert engine.plan_version == 4
        assert "任务拆分修改额度耗尽" in output.final_result
        assert "未通过" in output.review_summary

        # 首次拆解 + 三次重拆；每轮两阶段评审（计划 + 授权）
        assert supervisor.phases == ["planning", "replanning", "replanning", "replanning"]
        assert [call.stage for call in reviews.calls] == [
            ReviewStage.PLAN_REVIEW,
            ReviewStage.AUTHORIZATION_REVIEW,
        ] * 4
        assert [decision.passed for decision in reviews.decisions] == [False] * 8

        # 每版被采纳的计划都保留快照
        assert len(output.plan_versions) == 4

        # 人工升级材料包完整交接
        assert output.human_escalation is not None
        assert len(handler.pending_requests) == 1
        request = handler.pending_requests[0]
        assert request.task_id == output.task_id
        assert request.original_request.objective == "编写一个演示 CLI 工具"
        assert request.original_request.constraints == ["使用 Python", "保持简洁"]
        assert len(request.plan_history) == 4
        assert request.final_review is not None
        assert request.final_review.passed is False
        # 每版计划均签发工具授权 → 至少包含对应授权事件
        assert len(request.tool_call_evidence) == 4

        # 升级事件
        escalation_events = [
            record
            for record in context.event_log
            if record["event_type"] == EVENT_HUMAN_ESCALATION
        ]
        assert len(escalation_events) == 1
        assert "计划评审连续未通过" in escalation_events[0]["payload"]["reason"]

    async def test_status_sequence_matches_planning_loop(self, tmp_path, monkeypatch):
        supervisor = FakeSupervisor()
        reviews = ReviewScript([False, False, False, False])
        context, engine, _ = build_engine(
            tmp_path, monkeypatch, supervisor=supervisor, reviews=reviews
        )

        await engine.run("目标")

        assert status_transitions(context) == [
            ("received", "planning"),
            ("planning", "plan_review"),
            ("plan_review", "replanning"),
            ("replanning", "plan_review"),
            ("plan_review", "replanning"),
            ("replanning", "plan_review"),
            ("plan_review", "replanning"),
            ("replanning", "plan_review"),
            ("plan_review", "human_required"),
        ]


class TestReplanPayloadAndEvents:
    async def test_replan_inputs_carry_feedback_and_previous_plan(
        self, tmp_path, monkeypatch
    ):
        supervisor = FakeSupervisor()
        reviews = ReviewScript([False] * 4)
        _, engine, _ = build_engine(
            tmp_path, monkeypatch, supervisor=supervisor, reviews=reviews
        )

        await engine.run("目标")

        first = supervisor.payloads[0]
        assert first["phase"] == "planning"
        assert first["plan_version"] == 1
        assert first["replan_count"] == 0
        assert first["previous_plan"] == []
        assert "review_feedback" not in first
        assert [item["name"] for item in first["available_tools"]] == [
            "read_file",
            "review_agent",
            "run_in_sandbox",
            "write_file",
        ]
        assert first["max_replan_attempts"] == 3

        second = supervisor.payloads[1]
        assert second["phase"] == "replanning"
        assert second["plan_version"] == 2
        assert second["replan_count"] == 1
        assert second["previous_plan"][0]["subtask_id"] == "s1"
        assert second["review_feedback"]["passed"] is False
        assert second["review_feedback"]["stage"] == "plan_review"
        assert "子任务拆分粒度过粗" in second["review_feedback"]["issues"]

        # 额度消耗随循环推进递增
        assert [payload["replan_count"] for payload in supervisor.payloads] == [
            0,
            1,
            2,
            3,
        ]

    async def test_replan_events_recorded_per_version(self, tmp_path, monkeypatch):
        supervisor = FakeSupervisor()
        reviews = ReviewScript([False] * 4)
        context, engine, _ = build_engine(
            tmp_path, monkeypatch, supervisor=supervisor, reviews=reviews
        )

        await engine.run("目标")

        replan_events = [
            record
            for record in context.event_log
            if record["event_type"] == EVENT_REPLAN
        ]
        assert [record["payload"]["plan_version"] for record in replan_events] == [
            2,
            3,
            4,
        ]
        assert [record["payload"]["replan_count"] for record in replan_events] == [
            1,
            2,
            3,
        ]
        assert all(
            record["payload"]["subtask_ids"] == ["s1"] for record in replan_events
        )


class TestInvalidPlanStructure:
    async def test_structural_problems_consume_quota_without_review(
        self, tmp_path, monkeypatch
    ):
        supervisor = FakeSupervisor(invalid_plan=True)
        reviews = ReviewScript([True])
        context, engine, handler = build_engine(
            tmp_path, monkeypatch, supervisor=supervisor, reviews=reviews
        )

        output = await engine.run("目标")

        # 结构非法：每次拆解都消耗额度，耗尽后人工兜底
        assert output.status == TaskStatus.HUMAN_REQUIRED
        assert engine.replan_count == 3
        assert engine.plan_version == 0
        # 非法计划从未被采纳：plan_version 保持 0，阶段标记始终为 planning
        # （重拆标记 replanning 仅在上一版计划通过基础校验并被采纳后出现）
        assert supervisor.phases == ["planning"] * 4
        # 未通过基础校验：从未进入评审阶段、无计划被采纳、无重规划事件
        assert reviews.calls == []
        assert output.plan_versions == []
        # 状态无抖动：planning 只记录一次（重复迁移去重）
        assert status_transitions(context) == [
            ("received", "planning"),
            ("planning", "human_required"),
        ]
        assert [
            record
            for record in context.event_log
            if record["event_type"] == EVENT_REPLAN
        ] == []

        request = handler.pending_requests[0]
        assert request.final_review is not None
        # 阶段性精简契约（PlanningOutput）下的基础校验失败消息：拆分方案为空
        assert any("subtasks" in issue for issue in request.final_review.issues)


class TestReplanSucceedsWithinQuota:
    async def test_plan_passes_on_third_attempt_and_completes(
        self, tmp_path, monkeypatch
    ):
        supervisor = FakeSupervisor()
        reviews = ReviewScript([False, False, True])
        context, engine, handler = build_engine(
            tmp_path, monkeypatch, supervisor=supervisor, reviews=reviews
        )

        output = await engine.run("编写演示工具", constraints="使用 Python")

        # 第三次拆解通过评审 → 执行、汇总、终审通过 → completed
        assert output.status == TaskStatus.COMPLETED
        assert engine.current_status == TaskStatus.COMPLETED
        assert output.final_result == "最终结果文本"
        assert engine.replan_count == 2  # 前两轮不通过各消耗一次额度
        assert engine.plan_version == 3
        assert len(output.plan_versions) == 3
        assert handler.pending_requests == []  # 未触发人工兜底

        # 阶段调用序列：规划 → 执行（一次成功）→ 汇总
        assert supervisor.phases == [
            "planning",
            "replanning",
            "replanning",
            "execute_subtask",
            "aggregating",
        ]
        # 计划评审 3 批 × 2 轮 + 结果评审 1 次
        assert len(reviews.calls) == 7
        assert [decision.passed for decision in reviews.decisions] == [
            False,
            False,
            False,
            False,
            True,
            True,
            True,
        ]

        # 完整状态序列直至完成
        assert status_transitions(context) == [
            ("received", "planning"),
            ("planning", "plan_review"),
            ("plan_review", "replanning"),
            ("replanning", "plan_review"),
            ("plan_review", "replanning"),
            ("replanning", "plan_review"),
            ("plan_review", "executing"),
            ("executing", "aggregating"),
            ("aggregating", "result_review"),
            ("result_review", "completed"),
        ]


class TestNetworkFailuresDoNotConsumeReplanQuota:
    async def test_network_timeout_retries_then_succeeds(
        self, tmp_path, monkeypatch, capsys
    ):
        supervisor = FlakyNetworkSupervisor(failures=1)
        reviews = ReviewScript([True])
        context, engine, handler = build_engine(
            tmp_path, monkeypatch, supervisor=supervisor, reviews=reviews
        )

        output = await engine.run("网络抖动一次后恢复")

        # 网络失败不消耗拆分修改额度、不转为计划结构问题：重试后正常完成
        assert output.status == TaskStatus.COMPLETED
        assert engine.replan_count == 0
        assert engine.plan_version == 1
        # 1 次超时 + 规划重试 + 执行 + 汇总；重试后的阶段序列完整
        assert supervisor.calls == 4
        assert supervisor.phases == ["planning", "execute_subtask", "aggregating"]
        assert handler.pending_requests == []

        # 重试进度提示（脱敏：只报异常类名）
        stderr = capsys.readouterr().err
        assert "[so-agent] 网络请求失败（APITimeoutError），正在重试..." in stderr
        assert "[so-agent] 网络重试 1/3..." in stderr

    async def test_network_failures_exhaust_budget_and_escalate(
        self, tmp_path, monkeypatch, capsys
    ):
        supervisor = FlakyNetworkSupervisor(failures=None)
        reviews = ReviewScript([True])
        context, engine, handler = build_engine(
            tmp_path, monkeypatch, supervisor=supervisor, reviews=reviews
        )

        output = await engine.run("网络持续失败")

        # 首次尝试 + 3 次网络重试 = 4 次调用；不消耗重规划额度
        assert supervisor.calls == 4
        assert engine.replan_count == 0
        assert engine.plan_version == 0
        assert output.status == TaskStatus.HUMAN_REQUIRED
        assert "网络连续失败" in output.final_result

        # 网络耗尽而非计划结构问题：人工升级材料包不包含评审结论
        request = handler.pending_requests[0]
        assert request.final_review is None
        assert status_transitions(context) == [
            ("received", "planning"),
            ("planning", "human_required"),
        ]
        escalation_events = [
            record
            for record in context.event_log
            if record["event_type"] == EVENT_HUMAN_ESCALATION
        ]
        assert len(escalation_events) == 1
        assert "网络连续失败" in escalation_events[0]["payload"]["reason"]

        # 只在实际发起重试时打印提示；第 4 次失败直接转人工兜底
        stderr = capsys.readouterr().err
        assert stderr.count("[so-agent] 网络请求失败") == 3
        assert "[so-agent] 网络重试 3/3..." in stderr

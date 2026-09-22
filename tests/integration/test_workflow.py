"""集成测试：工作流主链路（快乐路径、结果返工、边界行为与主管组装）。

模拟场景（无真实模型调用，monkeypatch 替换 workflow 模块内的 Runner）：
1. 快乐路径：received → planning → plan_review → executing → aggregating
   → result_review → completed，主管三阶段调用与两阶段评审各就各位；
2. 计划工具净化：allowed_tools 裁剪到主管工具全集（去重、去未知），
   有工具的子任务签发 ToolGrant、无工具的子任务不签发（默认无工具）；
3. 结果评审返工：仅重跑被指出的子任务；无法定位返工目标或返工额度
   耗尽时转入人工兜底；
4. 汇总兜底：主管汇总返回空文本时回退为本地确定性摘要；
5. 边界：协作式取消、空目标失败终态；
6. 主管组装：create_orchestrator 的工具集严格等于三个固定工具入口。
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import so_agent.workflow as workflow_mod
from so_agent.config import Settings
from so_agent.context import ProjectContext
from so_agent.models import (
    ExecutionResult,
    ReviewDecision,
    SubtaskSpec,
    TaskStatus,
)
from so_agent.orchestrator import (
    AGENT_NAME,
    SUPERVISOR_TOOL_NAMES,
    OrchestratorOutput,
    create_orchestrator,
)
from so_agent.runtime.events import EVENT_STATUS_TRANSITION
from so_agent.tool_packages.review_agent.agent import ReviewInput, ReviewStage
from so_agent.workflow import WorkflowEngine


class FakeRunResult:
    """假 RunResult：只保留 final_output，供控制层结构与文本解析。"""

    def __init__(self, final_output):
        self.final_output = final_output


def install_fake_runner(
    monkeypatch, *, orchestrator, review_agent, supervisor, reviews
):
    """把 workflow 模块内的 Runner 替换为按 Agent 身份分派的假 Runner。"""

    class FakeRunner:
        @classmethod
        async def run(cls, starting_agent, input=None, *, max_turns=None, **kwargs):
            if starting_agent is review_agent:
                review_input = ReviewInput.model_validate_json(input)
                return FakeRunResult(reviews(review_input))
            payload = json.loads(input)
            return FakeRunResult(supervisor(payload))

    monkeypatch.setattr(workflow_mod, "Runner", FakeRunner)


def make_spec(subtask_id: str, *, tools: list[str] | None = None) -> SubtaskSpec:
    return SubtaskSpec(
        subtask_id=subtask_id,
        title=f"模拟子任务 {subtask_id}",
        instructions=f"完成子任务 {subtask_id}",
        role="code",
        allowed_tools=list(tools or []),
        max_attempts=3,
    )


class FakeSupervisor:
    """按 phase 分派的主管 Agent 模拟对象，记录载荷与执行序列。"""

    def __init__(self, *, specs=None, empty_summary: bool = False):
        self.specs = (
            list(specs)
            if specs is not None
            else [make_spec("s1", tools=["write_file"])]
        )
        self.empty_summary = empty_summary
        self.payloads: list[dict] = []
        self.exec_sequence: list[str] = []
        self.exec_payloads: dict[str, list[dict]] = {}

    @property
    def phases(self) -> list[str]:
        return [payload.get("phase") for payload in self.payloads]

    def __call__(self, payload: dict) -> OrchestratorOutput:
        self.payloads.append(payload)
        phase = payload.get("phase")
        if phase in ("planning", "replanning"):
            return self._plan_output(payload)
        if phase == "execute_subtask":
            subtask_id = payload["subtask"]["subtask_id"]
            self.exec_sequence.append(subtask_id)
            self.exec_payloads.setdefault(subtask_id, []).append(payload)
            return self._execution_output(payload)
        if phase == "aggregating":
            return OrchestratorOutput(
                task_id=str(payload.get("task_id") or "stub-task"),
                status=TaskStatus.AGGREGATING,
                final_result="" if self.empty_summary else "最终结果文本",
            )
        raise AssertionError(f"未预期的阶段载荷：{phase}")

    def _plan_output(self, payload: dict) -> OrchestratorOutput:
        plan = [spec.model_copy(deep=True) for spec in self.specs]
        return OrchestratorOutput(
            task_id=str(payload.get("task_id") or "stub-task"),
            status=(
                TaskStatus.REPLANNING
                if payload.get("phase") == "replanning"
                else TaskStatus.PLANNING
            ),
            plan_versions=[plan],
        )

    def _execution_output(self, payload: dict) -> OrchestratorOutput:
        subtask_id = payload["subtask"]["subtask_id"]
        result = ExecutionResult(
            subtask_id=subtask_id, success=True, output=f"{subtask_id} 完成（模拟）"
        )
        return OrchestratorOutput(
            task_id=str(payload.get("task_id") or "stub-task"),
            status=TaskStatus.EXECUTING,
            final_result=result.model_dump_json(),
        )


class ReviewScript:
    """计划评审按批次脚本；结果评审按序列脚本（默认全部通过）。"""

    def __init__(
        self,
        plan_batch_results,
        *,
        result_results=True,
        result_retry_target: str | None = "s1",
    ):
        self.plan_batch_results = list(plan_batch_results) or [True]
        if isinstance(result_results, bool):
            result_results = [result_results]
        self.result_results = list(result_results)
        self.result_retry_target = result_retry_target
        self.calls: list[ReviewInput] = []
        self.decisions: list[ReviewDecision] = []
        self._result_index = 0

    def __call__(self, review_input: ReviewInput) -> ReviewDecision:
        self.calls.append(review_input)
        if review_input.stage == ReviewStage.RESULT_REVIEW:
            passed = self.result_results[
                min(self._result_index, len(self.result_results) - 1)
            ]
            self._result_index += 1
            decision = ReviewDecision(
                stage="result_review",
                passed=passed,
                issues=[] if passed else ["整体结果不满足验收标准"],
                retry_target=None if passed else self.result_retry_target,
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
    """组装工作流引擎：注入假主管/评审 Agent。"""
    context = ProjectContext(sandbox_dir=tmp_path, project_name="workflow-test")
    effective_settings = settings or Settings(_env_file=None)
    orchestrator = SimpleNamespace(name="fake-supervisor")
    review_agent = SimpleNamespace(name="fake-reviewer")
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
    )
    return context, engine


def status_transitions(context) -> list[tuple[str, str]]:
    """提取状态迁移序列 ``[(from_status, to_status), ...]``。"""
    return [
        (record["payload"]["from_status"], record["payload"]["to_status"])
        for record in context.event_log
        if record["event_type"] == EVENT_STATUS_TRANSITION
    ]


class TestHappyPath:
    async def test_full_pipeline_state_sequence_and_calls(self, tmp_path, monkeypatch):
        supervisor = FakeSupervisor()
        reviews = ReviewScript([True])
        context, engine = build_engine(
            tmp_path, monkeypatch, supervisor=supervisor, reviews=reviews
        )

        output = await engine.run("编写演示模块")

        # 快乐路径完整状态序列
        assert status_transitions(context) == [
            ("received", "planning"),
            ("planning", "plan_review"),
            ("plan_review", "executing"),
            ("executing", "aggregating"),
            ("aggregating", "result_review"),
            ("result_review", "completed"),
        ]
        assert output.status == TaskStatus.COMPLETED
        assert engine.current_status == TaskStatus.COMPLETED
        assert output.final_result == "最终结果文本"
        assert engine.plan_version == 1
        assert engine.replan_count == 0
        assert len(output.plan_versions) == 1

        # 主管调用：规划 → 执行（一次尝试成功）→ 汇总
        assert supervisor.phases == ["planning", "execute_subtask", "aggregating"]
        # 评审调用：计划 + 授权 + 结果（各一次）
        assert [call.stage for call in reviews.calls] == [
            ReviewStage.PLAN_REVIEW,
            ReviewStage.AUTHORIZATION_REVIEW,
            ReviewStage.RESULT_REVIEW,
        ]

        # 执行阶段：签发的 ToolGrant 覆盖唯一允许工具
        exec_payload = supervisor.exec_payloads["s1"][0]
        grant = exec_payload["grant"]
        assert grant is not None
        assert grant["allowed_tools"] == ["write_file"]
        assert grant["issued_by"] == "supervisor"
        assert exec_payload["subtask"]["allowed_tools"] == ["write_file"]

        # Agent 生命周期：执行成功并释放
        assert len(output.agent_history) == 1
        assert output.agent_history[0].status == "completed"

    async def test_plan_tools_sanitized_and_grants_issued(self, tmp_path, monkeypatch):
        supervisor = FakeSupervisor(
            specs=[
                make_spec(
                    "s1",
                    tools=["write_file", "code_agent", "subagent_creator", "write_file"],
                ),
                make_spec("s2", tools=[]),
            ]
        )
        reviews = ReviewScript([True])
        context, engine = build_engine(
            tmp_path, monkeypatch, supervisor=supervisor, reviews=reviews
        )

        output = await engine.run("混合工具子任务")

        assert output.status == TaskStatus.COMPLETED
        assert supervisor.exec_sequence == ["s1", "s2"]

        # s1：allowed_tools 去重；主管专属工具（code_agent / subagent_creator）
        # 被裁剪（assignable=False 严禁分配）；仅已批准的可分配工具获得授权
        s1_payload = supervisor.exec_payloads["s1"][0]
        assert s1_payload["subtask"]["allowed_tools"] == ["write_file"]
        assert s1_payload["grant"] is not None
        assert s1_payload["grant"]["allowed_tools"] == ["write_file"]

        # s2：无允许工具 → 不签发授权（子 Agent 默认无任何工具）
        s2_payload = supervisor.exec_payloads["s2"][0]
        assert s2_payload["subtask"]["allowed_tools"] == []
        assert s2_payload["grant"] is None


class TestResultRework:
    async def test_rework_failed_subtask_then_completes(self, tmp_path, monkeypatch):
        supervisor = FakeSupervisor(
            specs=[make_spec("s1", tools=["write_file"]), make_spec("s2")]
        )
        reviews = ReviewScript([True], result_results=[False, True])
        context, engine = build_engine(
            tmp_path, monkeypatch, supervisor=supervisor, reviews=reviews
        )

        output = await engine.run("含返工的任务")

        assert output.status == TaskStatus.COMPLETED
        # 返工只重跑被指出的子任务 s1
        assert supervisor.exec_sequence == ["s1", "s2", "s1"]
        # 两轮汇总与两次结果评审
        assert supervisor.phases.count("aggregating") == 2
        result_calls = [
            call for call in reviews.calls if call.stage == ReviewStage.RESULT_REVIEW
        ]
        assert len(result_calls) == 2
        assert [decision.passed for decision in reviews.decisions][-2:] == [False, True]

        assert status_transitions(context) == [
            ("received", "planning"),
            ("planning", "plan_review"),
            ("plan_review", "executing"),
            ("executing", "aggregating"),
            ("aggregating", "result_review"),
            ("result_review", "executing"),
            ("executing", "aggregating"),
            ("aggregating", "result_review"),
            ("result_review", "completed"),
        ]

    async def test_unresolvable_rework_target_escalates(self, tmp_path, monkeypatch):
        supervisor = FakeSupervisor()
        reviews = ReviewScript(
            [True], result_results=False, result_retry_target=None
        )
        context, engine = build_engine(
            tmp_path, monkeypatch, supervisor=supervisor, reviews=reviews
        )

        output = await engine.run("无法定位返工目标")

        assert output.status == TaskStatus.HUMAN_REQUIRED
        assert "无法定位返工目标" in output.final_result
        # 结果评审不通过且 retry_target 为空：不进入返工，一次执行/汇总后直接兜底
        assert supervisor.exec_sequence == ["s1"]
        assert supervisor.phases.count("aggregating") == 1
        assert output.human_escalation is not None
        assert output.human_escalation.final_review is not None
        assert output.human_escalation.final_review.passed is False

    async def test_rework_quota_exhausted_escalates(self, tmp_path, monkeypatch):
        supervisor = FakeSupervisor()
        reviews = ReviewScript([True], result_results=False)
        context, engine = build_engine(
            tmp_path, monkeypatch, supervisor=supervisor, reviews=reviews
        )

        output = await engine.run("返工额度耗尽")

        assert output.status == TaskStatus.HUMAN_REQUIRED
        assert "结果评审返工次数耗尽仍未通过" in output.final_result
        # 4 轮执行/汇总/评审：第 4 次不通过后返工额度（3 次）已耗尽
        assert supervisor.exec_sequence == ["s1"] * 4
        assert supervisor.phases.count("aggregating") == 4
        assert [decision.passed for decision in reviews.decisions] == [
            True,
            True,
            False,
            False,
            False,
            False,
        ]

        assert status_transitions(context) == [
            ("received", "planning"),
            ("planning", "plan_review"),
            ("plan_review", "executing"),
            ("executing", "aggregating"),
            ("aggregating", "result_review"),
            ("result_review", "executing"),
            ("executing", "aggregating"),
            ("aggregating", "result_review"),
            ("result_review", "executing"),
            ("executing", "aggregating"),
            ("aggregating", "result_review"),
            ("result_review", "executing"),
            ("executing", "aggregating"),
            ("aggregating", "result_review"),
            ("result_review", "human_required"),
        ]


class TestWorkflowEdgeCases:
    async def test_empty_supervisor_summary_falls_back_locally(
        self, tmp_path, monkeypatch
    ):
        supervisor = FakeSupervisor(empty_summary=True)
        reviews = ReviewScript([True])
        context, engine = build_engine(
            tmp_path, monkeypatch, supervisor=supervisor, reviews=reviews
        )

        output = await engine.run("汇总兜底")

        assert output.status == TaskStatus.COMPLETED
        # 主管汇总不可用/为空 → 本地确定性摘要兜底
        assert "共完成 1 个子任务" in output.final_result
        assert "[s1] s1 完成（模拟）" in output.final_result
        assert "（汇总说明：主管 Agent 未返回汇总文本）" in output.final_result

    async def test_cancel_before_start_returns_cancelled(self, tmp_path, monkeypatch):
        supervisor = FakeSupervisor()
        reviews = ReviewScript([True])
        context, engine = build_engine(
            tmp_path, monkeypatch, supervisor=supervisor, reviews=reviews
        )

        engine.cancel()
        output = await engine.run("立即取消")

        assert output.status == TaskStatus.CANCELLED
        assert output.final_result == "任务已取消"
        # 取消发生在第一个检查点：未产生任何模型调用
        assert supervisor.payloads == []
        assert reviews.calls == []
        assert status_transitions(context) == [
            ("received", "planning"),
            ("planning", "cancelled"),
        ]

    async def test_empty_objective_returns_failed(self, tmp_path, monkeypatch):
        supervisor = FakeSupervisor()
        reviews = ReviewScript([True])
        context, engine = build_engine(
            tmp_path, monkeypatch, supervisor=supervisor, reviews=reviews
        )

        output = await engine.run("   ")

        assert output.status == TaskStatus.FAILED
        assert "WorkflowError" in output.final_result
        assert "objective" in output.final_result
        assert supervisor.payloads == []
        assert status_transitions(context) == [("received", "failed")]


class TestSupervisorAssembly:
    def test_orchestrator_has_exactly_three_fixed_tools(self, tmp_path, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        context = ProjectContext(sandbox_dir=tmp_path, project_name="assembly")

        agent = create_orchestrator(context, settings=Settings(_env_file=None))

        assert agent.name == AGENT_NAME == "supervisor_agent"
        # 主管工具集严格等于三个固定工具入口（顺序一致），此外不持有任何工具
        assert [tool.name for tool in agent.tools] == list(SUPERVISOR_TOOL_NAMES)
        assert [tool.name for tool in agent.tools] == [
            "code_agent",
            "subagent_creator",
            "review_agent",
        ]
        assert agent.output_type is None  # 纯文本返回：由控制层手动解析（兼容千问围栏行为）


class TestFencedJsonOutput:
    """千问兼容接口回归：围栏包裹的纯文本 JSON 输出仍可端到端完成。

    模拟真实行为：主管/评审 Agent 的 output_type 均为 None，模型把
    内容正确的 JSON 用 ```json 围栏包裹后以纯文本返回；控制层
    （_coerce_structured_output）应剥离围栏后按阶段性精简契约
    （PlanningOutput / ExecutionOutput / AggregationOutput /
    ReviewDecision）手动校验，全流程不受影响。
    """

    async def test_fenced_json_supervisor_and_review_complete(
        self, tmp_path, monkeypatch
    ):
        class FencedSupervisor:
            """每个阶段都返回 ```json 围栏包裹的精简契约 JSON 文本。"""

            def __init__(self):
                self.payloads: list[dict] = []

            @property
            def phases(self) -> list[str]:
                return [payload.get("phase") for payload in self.payloads]

            def __call__(self, payload: dict) -> str:
                self.payloads.append(payload)
                phase = payload.get("phase")
                if phase in ("planning", "replanning"):
                    spec = make_spec("s1", tools=["write_file"])
                    body = json.dumps(
                        {"subtasks": [spec.model_dump(mode="json")]},
                        ensure_ascii=False,
                    )
                elif phase == "execute_subtask":
                    subtask_id = payload["subtask"]["subtask_id"]
                    result = ExecutionResult(
                        subtask_id=subtask_id,
                        success=True,
                        output=f"{subtask_id} 完成（模拟）",
                    )
                    body = json.dumps(
                        {"final_result": result.model_dump_json()}, ensure_ascii=False
                    )
                elif phase == "aggregating":
                    body = json.dumps(
                        {"final_result": "最终结果文本", "status": "aggregating"},
                        ensure_ascii=False,
                    )
                else:
                    raise AssertionError(f"未预期的阶段载荷：{phase}")
                return f"```json\n{body}\n```"

        def fenced_reviews(review_input: ReviewInput) -> str:
            """评审 Agent 同样以围栏纯文本返回 ReviewDecision JSON。"""
            if review_input.stage == ReviewStage.RESULT_REVIEW:
                decision = ReviewDecision(
                    stage="result_review",
                    passed=True,
                    issues=[],
                    summary="结果评审（模拟）",
                )
            else:
                decision = ReviewDecision(
                    stage="plan_review",
                    passed=True,
                    issues=[],
                    summary="计划评审（模拟）",
                )
            return f"```json\n{decision.model_dump_json()}\n```"

        supervisor = FencedSupervisor()
        reviews = fenced_reviews
        context, engine = build_engine(
            tmp_path, monkeypatch, supervisor=supervisor, reviews=reviews
        )

        output = await engine.run("编写演示模块")

        # 围栏包裹的纯文本输出不影响全流程：完成终态、结果正确
        assert output.status == TaskStatus.COMPLETED
        assert output.final_result == "最终结果文本"
        assert supervisor.phases == ["planning", "execute_subtask", "aggregating"]
        assert status_transitions(context) == [
            ("received", "planning"),
            ("planning", "plan_review"),
            ("plan_review", "executing"),
            ("executing", "aggregating"),
            ("aggregating", "result_review"),
            ("result_review", "completed"),
        ]

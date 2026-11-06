"""单元测试：结果评审策略（快速通道产出门槛与无验收标准兜底评审）。

覆盖 Fix 3.1 的行为约定（``WorkflowEngine._run_pipeline_main``）：

1. 删除"无验收标准跳过结果评审"：全部成功且无验收标准的任务仍走
   结果评审（对照 objective 校验整体结果），不再条件放行；
2. 简单任务快速通道增加产出充分性门槛：全部成功子任务的 output
   非空且长度 > 50 字符才允许跳过汇总与结果评审；产出空白或过短
   时仍走完整汇总与结果评审链路；
3. ``skip_result_review_for_simple`` 配置：True（默认）时满足门槛的
   简单任务走快速通道；False 时所有任务一律走结果评审；
4. 复杂任务（> 2 个子任务或存在依赖）无论配置与产出长度都走结果
   评审。

模拟场景（无真实模型调用，monkeypatch 替换 workflow 模块内的
Runner）：主管按脚本返回计划与执行产出；评审按脚本返回结论并记录
全部调用，便于断言评审是否被触发。
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import so_agent.workflow as workflow_mod
from so_agent.config import Settings, get_settings
from so_agent.context import ProjectContext
from so_agent.models import (
    ExecutionResult,
    ReviewDecision,
    SubtaskSpec,
    TaskStatus,
)
from so_agent.orchestrator import OrchestratorOutput
from so_agent.runtime.events import (
    EVENT_PROVISIONAL_DELIVERED,
    EVENT_PROVISIONAL_ESCALATED,
    EVENT_PROVISIONAL_REVIEW_FAILED,
    EVENT_PROVISIONAL_REVIEW_PASSED,
    EVENT_PROVISIONAL_REVISED,
    EVENT_STATUS_TRANSITION,
)
from so_agent.tool_packages.review_agent.agent import ReviewInput, ReviewStage
from so_agent.workflow import WorkflowEngine

# 快速通道产出充分性阈值与 workflow.py 保持一致（严格大于）
_MIN_OUTPUT_CHARS = 50

# 充分产出（> 50 字符）与不充分产出
LONG_OUTPUT = "充分长度的模拟产出文本，" * 6  # 60 字符
SHORT_OUTPUT = "短产出"


def make_spec(
    subtask_id: str,
    *,
    dependencies: list[str] | None = None,
    write_scope: list[str] | None = None,
) -> SubtaskSpec:
    """构造无工具需求的模拟子任务规格。"""
    return SubtaskSpec(
        subtask_id=subtask_id,
        title=f"模拟子任务 {subtask_id}",
        instructions=f"完成子任务 {subtask_id}",
        role="code",
        allowed_tools=[],
        dependencies=list(dependencies or []),
        write_scope=list(write_scope or []),
        max_attempts=3,
    )


class FakeRunResult:
    """假 RunResult：只保留 final_output。"""

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


class ScriptedSupervisor:
    """按脚本分派的主管模拟对象：计划按 specs、执行按 outputs 映射。"""

    def __init__(self, *, specs: list[SubtaskSpec], outputs: dict[str, str]):
        self.specs = list(specs)
        self.outputs = dict(outputs)
        self.payloads: list[dict] = []

    @property
    def phases(self) -> list[str]:
        return [payload.get("phase") for payload in self.payloads]

    def __call__(self, payload: dict) -> OrchestratorOutput:
        self.payloads.append(payload)
        phase = payload.get("phase")
        if phase == "planning":
            plan = [spec.model_copy(deep=True) for spec in self.specs]
            return OrchestratorOutput(
                task_id=str(payload.get("task_id") or "stub-task"),
                status=TaskStatus.PLANNING,
                plan_versions=[plan],
            )
        if phase == "execute_subtask":
            subtask_id = payload["subtask"]["subtask_id"]
            result = ExecutionResult(
                subtask_id=subtask_id,
                success=True,
                output=self.outputs.get(subtask_id, ""),
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
                final_result="主管汇总结果文本",
            )
        raise AssertionError(f"未预期的阶段载荷：{phase}")


class ScriptedReviews:
    """评审模拟：计划/授权评审固定通过；结果评审按脚本并记录调用。"""

    def __init__(
        self,
        *,
        result_passed: tuple[bool, ...] = (True,),
        result_retry_targets: tuple[str | None, ...] = (),
    ):
        self.result_passed = list(result_passed) or [True]
        self.result_retry_targets = list(result_retry_targets)
        self.calls: list[ReviewStage] = []
        self.decisions: list[ReviewDecision] = []
        self._index = 0

    def __call__(self, review_input: ReviewInput) -> ReviewDecision:
        self.calls.append(review_input.stage)
        if review_input.stage == ReviewStage.RESULT_REVIEW:
            index = min(self._index, len(self.result_passed) - 1)
            passed = self.result_passed[index]
            retry_target = (
                self.result_retry_targets[
                    min(index, len(self.result_retry_targets) - 1)
                ]
                if self.result_retry_targets
                else None
            )
            self._index += 1
            decision = ReviewDecision(
                stage="result_review",
                passed=passed,
                issues=[] if passed else ["整体结果不满足验收标准"],
                required_fixes=[] if passed else ["修复整体结果后重新提交"],
                retry_target=retry_target,
                summary="结果评审（模拟）",
            )
        else:
            decision = ReviewDecision(
                stage="plan_review",
                passed=True,
                summary="计划评审（模拟）",
            )
        self.decisions.append(decision)
        return decision


def build_engine(
    tmp_path, monkeypatch, *, supervisor, reviews, settings=None
) -> tuple[ProjectContext, WorkflowEngine]:
    """组装工作流引擎：注入假主管/评审 Agent（Runner 按身份分派）。"""
    context = ProjectContext(sandbox_dir=tmp_path, project_name="review-policy")
    effective_settings = (settings or Settings(_env_file=None)).model_copy(
        update={"skip_plan_review_for_simple": False}
    )
    orchestrator = SimpleNamespace(name="fake-supervisor")
    review_agent = SimpleNamespace(name="fake-reviewer")

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


class TestConfigPolicy:
    """skip_result_review_for_simple 配置：默认开启、支持环境变量覆盖。"""

    def test_default_flag_enabled(self):
        settings = Settings(_env_file=None)
        assert settings.skip_result_review_for_simple is True

    def test_env_override_disables_flag(self, monkeypatch):
        monkeypatch.setenv("SO_AGENT_SKIP_RESULT_REVIEW_FOR_SIMPLE", "false")
        settings = Settings(_env_file=None)
        assert settings.skip_result_review_for_simple is False

    def test_env_cache_clear_picks_up_flag(self, monkeypatch):
        get_settings.cache_clear()
        try:
            monkeypatch.setenv("SO_AGENT_SKIP_RESULT_REVIEW_FOR_SIMPLE", "false")
            get_settings.cache_clear()
            assert get_settings().skip_result_review_for_simple is False
        finally:
            get_settings.cache_clear()


class TestNoCriteriaStillReviews:
    """删除"无验收标准跳过结果评审"：无验收标准的任务仍走结果评审。"""

    async def test_result_review_runs_without_acceptance_criteria(
        self, tmp_path, monkeypatch
    ):
        """复杂任务全部成功且无验收标准：汇总后照常结果评审。"""
        specs = [make_spec("s1"), make_spec("s2"), make_spec("s3")]
        supervisor = ScriptedSupervisor(
            specs=specs,
            outputs={"s1": LONG_OUTPUT, "s2": LONG_OUTPUT, "s3": LONG_OUTPUT},
        )
        reviews = ScriptedReviews()
        context, engine = build_engine(
            tmp_path, monkeypatch, supervisor=supervisor, reviews=reviews
        )

        output = await engine.run("无验收标准的演示任务")

        assert output.status == TaskStatus.COMPLETED
        # 结果评审必须触发（无验收标准不再跳过）
        assert ReviewStage.RESULT_REVIEW in reviews.calls
        assert output.review_summary != ""
        assert status_transitions(context) == [
            ("received", "planning"),
            ("planning", "plan_review"),
            ("plan_review", "executing"),
            ("executing", "aggregating"),
            ("aggregating", "result_review"),
            ("result_review", "completed"),
        ]


class TestSimpleFastPath:
    """简单任务快速通道：产出充分才放行，过短/空白仍走结果评审。"""

    async def test_fast_path_with_sufficient_output(self, tmp_path, monkeypatch):
        """单/双子任务 + 全部成功 + 产出充分 + 默认配置 → 跳过汇总与评审。"""
        for specs, outputs in (
            ([make_spec("s1")], {"s1": LONG_OUTPUT}),
            (
                [make_spec("s1"), make_spec("s2")],
                {"s1": LONG_OUTPUT, "s2": LONG_OUTPUT},
            ),
        ):
            supervisor = ScriptedSupervisor(specs=specs, outputs=outputs)
            reviews = ScriptedReviews()
            context, engine = build_engine(
                tmp_path, monkeypatch, supervisor=supervisor, reviews=reviews
            )

            output = await engine.run("简单的演示任务")

            assert output.status == TaskStatus.COMPLETED
            assert ReviewStage.RESULT_REVIEW not in reviews.calls
            assert "aggregating" not in supervisor.phases
            # 快速通道使用本地确定性汇总
            assert "共完成" in output.final_result
            assert status_transitions(context) == [
                ("received", "planning"),
                ("planning", "plan_review"),
                ("plan_review", "executing"),
                ("executing", "completed"),
            ]

    async def test_short_output_falls_back_to_result_review(
        self, tmp_path, monkeypatch
    ):
        """简单任务全部成功但产出过短（<= 50 字符）→ 走完整汇总与结果评审。"""
        supervisor = ScriptedSupervisor(
            specs=[make_spec("s1")], outputs={"s1": SHORT_OUTPUT}
        )
        reviews = ScriptedReviews()
        context, engine = build_engine(
            tmp_path, monkeypatch, supervisor=supervisor, reviews=reviews
        )

        output = await engine.run("产出过短的简单任务")

        assert output.status == TaskStatus.COMPLETED
        assert ReviewStage.RESULT_REVIEW in reviews.calls
        assert status_transitions(context) == [
            ("received", "planning"),
            ("planning", "plan_review"),
            ("plan_review", "executing"),
            ("executing", "aggregating"),
            ("aggregating", "result_review"),
            ("result_review", "completed"),
        ]

    async def test_blank_output_falls_back_to_result_review(
        self, tmp_path, monkeypatch
    ):
        """产出为空白文本 → 不走快速通道，走结果评审。"""
        supervisor = ScriptedSupervisor(
            specs=[make_spec("s1")], outputs={"s1": "   \n  "}
        )
        reviews = ScriptedReviews()
        _, engine = build_engine(
            tmp_path, monkeypatch, supervisor=supervisor, reviews=reviews
        )

        output = await engine.run("产出空白的简单任务")

        assert output.status == TaskStatus.COMPLETED
        assert ReviewStage.RESULT_REVIEW in reviews.calls

    async def test_output_length_boundary(self, tmp_path, monkeypatch):
        """产出长度边界：50 字符走结果评审，51 字符走快速通道。"""
        for length, expect_fast_path in ((50, False), (51, True)):
            supervisor = ScriptedSupervisor(
                specs=[make_spec("s1")], outputs={"s1": "字" * length}
            )
            reviews = ScriptedReviews()
            _, engine = build_engine(
                tmp_path, monkeypatch, supervisor=supervisor, reviews=reviews
            )

            output = await engine.run(f"产出长度 {length} 的简单任务")

            assert output.status == TaskStatus.COMPLETED
            reviewed = ReviewStage.RESULT_REVIEW in reviews.calls
            assert reviewed is (not expect_fast_path), (
                f"长度 {length}：expect_fast_path={expect_fast_path}, reviewed={reviewed}"
            )


class TestDisabledFlag:
    """skip_result_review_for_simple=False：所有任务一律走结果评审。"""

    async def test_flag_disabled_forces_review_even_with_sufficient_output(
        self, tmp_path, monkeypatch
    ):
        """简单任务 + 产出充分 + 配置关闭 → 不走快速通道，走结果评审。"""
        settings = Settings(_env_file=None, skip_result_review_for_simple=False)
        supervisor = ScriptedSupervisor(
            specs=[make_spec("s1")], outputs={"s1": LONG_OUTPUT}
        )
        reviews = ScriptedReviews()
        _, engine = build_engine(
            tmp_path,
            monkeypatch,
            supervisor=supervisor,
            reviews=reviews,
            settings=settings,
        )

        output = await engine.run("配置关闭后的简单任务")

        assert output.status == TaskStatus.COMPLETED
        assert ReviewStage.RESULT_REVIEW in reviews.calls
        # 单子任务汇总走透传快速路径（不调主管 aggregating）：
        # 最终结果为透传产出而非快速通道的本地确定性汇总
        assert output.final_result == LONG_OUTPUT
        assert "共完成" not in output.final_result


class TestComplexAlwaysReviewed:
    """复杂任务无论配置与产出长度都走结果评审。"""

    async def test_complex_task_reviewed_regardless_of_flag(
        self, tmp_path, monkeypatch
    ):
        """3 个子任务 + 产出充分：开关为 True/False 都触发结果评审。"""
        for enabled in (True, False):
            settings = Settings(
                _env_file=None, skip_result_review_for_simple=enabled
            )
            specs = [make_spec("s1"), make_spec("s2"), make_spec("s3")]
            supervisor = ScriptedSupervisor(
                specs=specs,
                outputs={"s1": LONG_OUTPUT, "s2": LONG_OUTPUT, "s3": LONG_OUTPUT},
            )
            reviews = ScriptedReviews()
            _, engine = build_engine(
                tmp_path,
                monkeypatch,
                supervisor=supervisor,
                reviews=reviews,
                settings=settings,
            )

            output = await engine.run(
                "复杂任务", acceptance_criteria="通过验收"
            )

            assert output.status == TaskStatus.COMPLETED
            assert ReviewStage.RESULT_REVIEW in reviews.calls, (
                f"skip_result_review_for_simple={enabled} 时复杂任务必须走结果评审"
            )

    async def test_dependent_subtasks_not_simple_and_reviewed(
        self, tmp_path, monkeypatch
    ):
        """双子任务但存在依赖（非简单计划）→ 即便产出充分也走结果评审。"""
        specs = [
            make_spec("s1"),
            make_spec("s2", dependencies=["s1"]),
        ]
        supervisor = ScriptedSupervisor(
            specs=specs, outputs={"s1": LONG_OUTPUT, "s2": LONG_OUTPUT}
        )
        reviews = ScriptedReviews()
        _, engine = build_engine(
            tmp_path, monkeypatch, supervisor=supervisor, reviews=reviews
        )

        output = await engine.run("带依赖的演示任务")

        assert output.status == TaskStatus.COMPLETED
        assert ReviewStage.RESULT_REVIEW in reviews.calls


class TestResultReviewMode:
    """result_review_mode / 并行评审：分级跳过与审计事件。"""

    def test_default_mode_is_always(self):
        assert Settings(_env_file=None).result_review_mode == "always"

    async def test_off_mode_skips_review_and_records_event(
        self, tmp_path, monkeypatch
    ):
        settings = Settings(
            _env_file=None,
            result_review_mode="off",
            skip_result_review_for_simple=False,
        )
        specs = [make_spec("s1"), make_spec("s2"), make_spec("s3")]
        supervisor = ScriptedSupervisor(
            specs=specs,
            outputs={"s1": LONG_OUTPUT, "s2": LONG_OUTPUT, "s3": LONG_OUTPUT},
        )
        reviews = ScriptedReviews()
        context, engine = build_engine(
            tmp_path,
            monkeypatch,
            supervisor=supervisor,
            reviews=reviews,
            settings=settings,
        )

        output = await engine.run("关闭评审的演示任务")

        assert output.status == TaskStatus.COMPLETED
        assert ReviewStage.RESULT_REVIEW not in reviews.calls
        skipped = [
            record
            for record in context.event_log
            if record["event_type"] == "review_skipped"
            and record["payload"].get("stage") == "result_review"
        ]
        assert skipped
        assert skipped[0]["payload"]["reason"] == "disabled"

    async def test_risk_based_skips_when_all_outputs_present(
        self, tmp_path, monkeypatch
    ):
        settings = Settings(
            _env_file=None,
            result_review_mode="risk_based",
            skip_result_review_for_simple=False,
        )
        specs = [make_spec("s1"), make_spec("s2"), make_spec("s3")]
        supervisor = ScriptedSupervisor(
            specs=specs,
            outputs={"s1": LONG_OUTPUT, "s2": LONG_OUTPUT, "s3": LONG_OUTPUT},
        )
        reviews = ScriptedReviews()
        context, engine = build_engine(
            tmp_path,
            monkeypatch,
            supervisor=supervisor,
            reviews=reviews,
            settings=settings,
        )

        output = await engine.run("低风险演示任务")

        assert output.status == TaskStatus.COMPLETED
        assert ReviewStage.RESULT_REVIEW not in reviews.calls
        skipped = [
            record
            for record in context.event_log
            if record["event_type"] == "review_skipped"
            and record["payload"].get("stage") == "result_review"
        ]
        assert skipped[0]["payload"]["reason"] == "risk_based_low_risk"

    async def test_risk_based_reviews_when_output_empty(
        self, tmp_path, monkeypatch
    ):
        settings = Settings(
            _env_file=None,
            result_review_mode="risk_based",
            skip_result_review_for_simple=False,
        )
        specs = [make_spec("s1"), make_spec("s2"), make_spec("s3")]
        supervisor = ScriptedSupervisor(
            specs=specs,
            outputs={"s1": LONG_OUTPUT, "s2": "", "s3": LONG_OUTPUT},
        )
        reviews = ScriptedReviews()
        _, engine = build_engine(
            tmp_path,
            monkeypatch,
            supervisor=supervisor,
            reviews=reviews,
            settings=settings,
        )

        output = await engine.run("空产出演示任务")

        assert output.status == TaskStatus.COMPLETED
        assert ReviewStage.RESULT_REVIEW in reviews.calls

    async def test_parallel_aggregation_and_review(self, tmp_path, monkeypatch):
        settings = Settings(
            _env_file=None,
            result_review_parallel_with_aggregation=True,
            skip_result_review_for_simple=False,
        )
        specs = [make_spec("s1"), make_spec("s2"), make_spec("s3")]
        supervisor = ScriptedSupervisor(
            specs=specs,
            outputs={"s1": LONG_OUTPUT, "s2": LONG_OUTPUT, "s3": LONG_OUTPUT},
        )
        reviews = ScriptedReviews()
        context, engine = build_engine(
            tmp_path,
            monkeypatch,
            supervisor=supervisor,
            reviews=reviews,
            settings=settings,
        )

        output = await engine.run("并行汇总与评审的演示任务")

        assert output.status == TaskStatus.COMPLETED
        assert "aggregating" in supervisor.phases
        assert ReviewStage.RESULT_REVIEW in reviews.calls
        assert output.final_result == "主管汇总结果文本"
        transitions = status_transitions(context)
        assert ("executing", "aggregating") in transitions
        assert ("aggregating", "result_review") in transitions
        assert ("result_review", "completed") in transitions


def event_payloads(context, event_type: str) -> list[dict]:
    """提取指定事件类型的载荷列表。"""
    return [
        record["payload"]
        for record in context.event_log
        if record["event_type"] == event_type
    ]


class TestProvisionalDelivery:
    """异步 provisional 交付：低风险先交付、后台评审与纠正闭环。"""

    @staticmethod
    def _settings(**overrides):
        base = {
            "result_review_async_provisional": True,
            "skip_result_review_for_simple": False,
        }
        base.update(overrides)
        return Settings(_env_file=None, **base)

    async def test_low_risk_delivers_then_review_passes(
        self, tmp_path, monkeypatch
    ):
        """低风险任务先交付（completed_provisional），后台评审通过转 completed。"""
        specs = [make_spec("s1"), make_spec("s2"), make_spec("s3")]
        supervisor = ScriptedSupervisor(
            specs=specs,
            outputs={"s1": LONG_OUTPUT, "s2": LONG_OUTPUT, "s3": LONG_OUTPUT},
        )
        reviews = ScriptedReviews()
        context, engine = build_engine(
            tmp_path,
            monkeypatch,
            supervisor=supervisor,
            reviews=reviews,
            settings=self._settings(),
        )

        output = await engine.run("低风险演示任务")

        assert output.status == TaskStatus.COMPLETED_PROVISIONAL
        assert output.provisional is True
        assert output.notice
        assert output.final_result == "主管汇总结果文本"

        final = await engine.wait_for_provisional()

        assert final is not None
        assert final.status == TaskStatus.COMPLETED
        assert final.final_result == output.final_result
        assert ReviewStage.RESULT_REVIEW in reviews.calls
        transitions = status_transitions(context)
        assert ("executing", "aggregating") in transitions
        assert ("aggregating", "completed_provisional") in transitions
        assert ("completed_provisional", "result_review") in transitions
        assert ("result_review", "completed") in transitions
        assert event_payloads(context, EVENT_PROVISIONAL_DELIVERED)
        assert event_payloads(context, EVENT_PROVISIONAL_REVIEW_PASSED)

    async def test_side_effects_stay_synchronous(self, tmp_path, monkeypatch):
        """存在写入副作用（write_scope）→ 不允许先交付后纠正。"""
        (tmp_path / "report.md").write_text("销售额 100", encoding="utf-8")
        specs = [
            make_spec("s1", write_scope=["report.md"]),
            make_spec("s2"),
            make_spec("s3"),
        ]
        supervisor = ScriptedSupervisor(
            specs=specs,
            outputs={"s1": LONG_OUTPUT, "s2": LONG_OUTPUT, "s3": LONG_OUTPUT},
        )
        reviews = ScriptedReviews()
        context, engine = build_engine(
            tmp_path,
            monkeypatch,
            supervisor=supervisor,
            reviews=reviews,
            settings=self._settings(subtask_tool_evidence_required=False),
        )

        output = await engine.run("有副作用的演示任务")

        assert output.status == TaskStatus.COMPLETED
        assert output.provisional is False
        assert not event_payloads(context, EVENT_PROVISIONAL_DELIVERED)
        transitions = status_transitions(context)
        assert ("aggregating", "result_review") in transitions

    async def test_acceptance_criteria_stay_synchronous(
        self, tmp_path, monkeypatch
    ):
        """存在验收标准 → 保持同步阻塞评审。"""
        specs = [make_spec("s1"), make_spec("s2"), make_spec("s3")]
        supervisor = ScriptedSupervisor(
            specs=specs,
            outputs={"s1": LONG_OUTPUT, "s2": LONG_OUTPUT, "s3": LONG_OUTPUT},
        )
        reviews = ScriptedReviews()
        context, engine = build_engine(
            tmp_path,
            monkeypatch,
            supervisor=supervisor,
            reviews=reviews,
            settings=self._settings(),
        )

        output = await engine.run(
            "带验收标准的演示任务", acceptance_criteria="结果必须包含销售总额"
        )

        assert output.status == TaskStatus.COMPLETED
        assert output.provisional is False
        assert not event_payloads(context, EVENT_PROVISIONAL_DELIVERED)

    async def test_failed_review_revises_and_keeps_previous(
        self, tmp_path, monkeypatch
    ):
        """评审失败一次 → 自动返工并重新评审通过，旧版本保留。"""
        specs = [make_spec("s1"), make_spec("s2"), make_spec("s3")]
        supervisor = ScriptedSupervisor(
            specs=specs,
            outputs={"s1": LONG_OUTPUT, "s2": LONG_OUTPUT, "s3": LONG_OUTPUT},
        )
        reviews = ScriptedReviews(
            result_passed=(False, True), result_retry_targets=("s1",)
        )
        context, engine = build_engine(
            tmp_path,
            monkeypatch,
            supervisor=supervisor,
            reviews=reviews,
            settings=self._settings(),
        )

        delivered = await engine.run("需要修订的演示任务")
        assert delivered.status == TaskStatus.COMPLETED_PROVISIONAL

        final = await engine.wait_for_provisional()

        assert final is not None
        assert final.status == TaskStatus.COMPLETED
        assert final.previous_result == delivered.final_result
        assert final.notice
        assert event_payloads(context, EVENT_PROVISIONAL_REVIEW_FAILED)
        revised = event_payloads(context, EVENT_PROVISIONAL_REVISED)
        assert revised
        assert revised[0]["rework_subtasks"] == ["s1"]
        # 初审 + 返工后复核共 2 次结果评审；返工执行 s1
        assert len([c for c in reviews.calls if c == ReviewStage.RESULT_REVIEW]) == 2
        assert supervisor.phases.count("execute_subtask") == 4

    async def test_failed_review_without_target_escalates(
        self, tmp_path, monkeypatch
    ):
        """评审失败且无法定位返工目标 → 人工兜底，保留已交付结果。"""
        specs = [make_spec("s1"), make_spec("s2"), make_spec("s3")]
        supervisor = ScriptedSupervisor(
            specs=specs,
            outputs={"s1": LONG_OUTPUT, "s2": LONG_OUTPUT, "s3": LONG_OUTPUT},
        )
        reviews = ScriptedReviews(result_passed=(False,))
        context, engine = build_engine(
            tmp_path,
            monkeypatch,
            supervisor=supervisor,
            reviews=reviews,
            settings=self._settings(),
        )

        delivered = await engine.run("无法定位返工目标的演示任务")
        final = await engine.wait_for_provisional()

        assert final is not None
        assert final.status == TaskStatus.HUMAN_REQUIRED
        assert final.previous_result == delivered.final_result
        assert event_payloads(context, EVENT_PROVISIONAL_REVIEW_FAILED)
        assert event_payloads(context, EVENT_PROVISIONAL_ESCALATED)

    async def test_cancel_during_background_review_marks_cancelled(
        self, tmp_path, monkeypatch
    ):
        """后台评审期间取消 → 最终输出为 cancelled，不误判为完成。"""
        specs = [make_spec("s1"), make_spec("s2"), make_spec("s3")]
        supervisor = ScriptedSupervisor(
            specs=specs,
            outputs={"s1": LONG_OUTPUT, "s2": LONG_OUTPUT, "s3": LONG_OUTPUT},
        )
        scripted = ScriptedReviews()
        holder: dict[str, WorkflowEngine] = {}

        def canceling_reviews(review_input):
            decision = scripted(review_input)
            if review_input.stage == ReviewStage.RESULT_REVIEW:
                holder["engine"].cancel()
            return decision

        context, engine = build_engine(
            tmp_path,
            monkeypatch,
            supervisor=supervisor,
            reviews=canceling_reviews,
            settings=self._settings(),
        )
        holder["engine"] = engine

        delivered = await engine.run("评审期间取消的演示任务")
        assert delivered.status == TaskStatus.COMPLETED_PROVISIONAL

        final = await engine.wait_for_provisional()

        assert final is not None
        assert final.status == TaskStatus.CANCELLED
        transitions = status_transitions(context)
        assert ("result_review", "cancelled") in transitions

    async def test_disabled_by_default_keeps_synchronous_review(
        self, tmp_path, monkeypatch
    ):
        """默认配置关闭 provisional：行为与同步评审一致。"""
        assert (
            Settings(_env_file=None).result_review_async_provisional is False
        )
        settings = Settings(_env_file=None, skip_result_review_for_simple=False)
        specs = [make_spec("s1"), make_spec("s2"), make_spec("s3")]
        supervisor = ScriptedSupervisor(
            specs=specs,
            outputs={"s1": LONG_OUTPUT, "s2": LONG_OUTPUT, "s3": LONG_OUTPUT},
        )
        reviews = ScriptedReviews()
        context, engine = build_engine(
            tmp_path,
            monkeypatch,
            supervisor=supervisor,
            reviews=reviews,
            settings=settings,
        )

        output = await engine.run("默认配置的演示任务")

        assert output.status == TaskStatus.COMPLETED
        assert output.provisional is False
        assert engine.provisional_review_pending is False
        assert await engine.wait_for_provisional() is None
        assert not event_payloads(context, EVENT_PROVISIONAL_DELIVERED)
        transitions = status_transitions(context)
        assert ("result_review", "completed") in transitions

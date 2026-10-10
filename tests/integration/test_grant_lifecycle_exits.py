"""集成测试：授权生命周期退出路径（Task #31）。

模拟场景（无真实模型调用，monkeypatch 替换 workflow 模块内的 Runner）：

1. 内容评审失败 → 转人工，被拒计划在网关无任何真实签发记录；
2. 授权评审失败 → 转人工，授权评审输入只含候选（proposal）；
3. 授权评审先拒后过 → 每次授权评审时刻均无真实签发（grant_counts 时序
   快照），通过后恰好签发一次、执行载荷使用真实授权；
4. 正常完成 → 执行窗口结束回收全部授权（reason=execution_window_closed），
   完成后旧授权不可复用；
5. 结果评审失败（无法定位返工目标）→ 转人工，无活跃授权残留；
6. pipeline timeout → 统一清理以 lifecycle_cleanup:pipeline_timeout 留下审计；
7. 外部取消 → 统一清理以 lifecycle_cleanup:cancelled 留下审计；
8. 未捕获异常（非 SchedulerError 穿透批次回收）→ FAILED 且统一清理以
   lifecycle_cleanup:unhandled_exception 留下审计；
9. 重复清理幂等：完成后再次清理为空操作、无重复事件。
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
from so_agent.orchestrator import OrchestratorOutput
from so_agent.runtime.events import EVENT_TOOL_GRANTED, EVENT_TOOL_REVOKED
from so_agent.tool_packages.review_agent.agent import ReviewInput, ReviewStage
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
    """假 RunResultStreaming：首次取流式事件时求值协程工厂并回填输出。"""

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


class GrantSupervisor:
    """按 phase 分派的假主管：规划携带工具、执行成功、汇总返回文本。

    ``hang_execute=True`` 时执行阶段长时间挂起（等待外层 pipeline 超时
    或任务取消）；``on_execute`` 在执行阶段开始时回调（供取消测试同步）。
    """

    def __init__(self, *, specs=None, hang_execute: bool = False, on_execute=None):
        self.specs = (
            list(specs)
            if specs is not None
            else [
                make_spec("s1", tools=["write_file"]),
                make_spec("s2"),
                make_spec("s3"),
            ]
        )
        self.hang_execute = hang_execute
        self.on_execute = on_execute
        self.payloads: list[dict] = []
        self.exec_sequence: list[str] = []
        self.exec_payloads: dict[str, list[dict]] = {}

    @property
    def phases(self) -> list[str]:
        return [payload.get("phase") for payload in self.payloads]

    async def __call__(self, payload: dict) -> OrchestratorOutput:
        self.payloads.append(payload)
        phase = payload.get("phase")
        if phase in ("planning", "replanning"):
            plan = [spec.model_copy(deep=True) for spec in self.specs]
            return OrchestratorOutput(
                task_id=str(payload.get("task_id") or "stub-task"),
                status=(
                    TaskStatus.REPLANNING
                    if phase == "replanning"
                    else TaskStatus.PLANNING
                ),
                plan_versions=[plan],
            )
        if phase == "execute_subtask":
            subtask_id = payload["subtask"]["subtask_id"]
            self.exec_sequence.append(subtask_id)
            self.exec_payloads.setdefault(subtask_id, []).append(payload)
            if self.on_execute is not None:
                self.on_execute(payload)
            if self.hang_execute:
                await asyncio.sleep(30)  # 等待外层 pipeline 超时 / 任务取消
            result = ExecutionResult(
                subtask_id=subtask_id, success=True, output=f"{subtask_id} 完成（模拟）"
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
                final_result="最终结果文本",
            )
        raise AssertionError(f"未预期的阶段载荷：{phase}")


class StageReview:
    """按 stage 分派脚本的评审假实现（计划/授权/结果三路独立脚本）。

    ``grant_counter`` 非空时，每次授权评审都会快照当时的 tool_granted
    事件数（时序证据：授权评审时刻不得存在真实签发）。
    """

    def __init__(
        self,
        *,
        plan=None,
        authorization=None,
        result=None,
        result_retry_target=None,
    ):
        self._plan_script = list(plan) if plan is not None else [True]
        self._auth_script = (
            list(authorization) if authorization is not None else [True]
        )
        self._result_script = list(result) if result is not None else [True]
        self.result_retry_target = result_retry_target
        self.calls: list[ReviewInput] = []
        self.grant_counts: list[int] = []
        self.grant_counter = None
        self._plan_index = 0
        self._auth_index = 0
        self._result_index = 0

    def __call__(self, review_input: ReviewInput) -> ReviewDecision:
        self.calls.append(review_input)
        if review_input.stage == ReviewStage.RESULT_REVIEW:
            passed = self._result_script[
                min(self._result_index, len(self._result_script) - 1)
            ]
            self._result_index += 1
            return ReviewDecision(
                stage="result_review",
                passed=passed,
                issues=[] if passed else ["整体结果不满足验收标准"],
                required_fixes=[] if passed else ["修复整体结果后重新提交"],
                retry_target=None if passed else self.result_retry_target,
                summary="结果评审（模拟）",
            )
        if review_input.stage == ReviewStage.AUTHORIZATION_REVIEW:
            if self.grant_counter is not None:
                self.grant_counts.append(self.grant_counter())
            passed = self._auth_script[
                min(self._auth_index, len(self._auth_script) - 1)
            ]
            self._auth_index += 1
            return ReviewDecision(
                stage="authorization_review",
                passed=passed,
                issues=[] if passed else ["授权范围与计划不匹配"],
                required_fixes=[] if passed else ["收紧授权范围后重新提交"],
                summary="授权评审（模拟）",
            )
        passed = self._plan_script[
            min(self._plan_index, len(self._plan_script) - 1)
        ]
        self._plan_index += 1
        return ReviewDecision(
            stage="plan_review",
            passed=passed,
            issues=[] if passed else ["子任务拆分粒度过粗"],
            required_fixes=[] if passed else ["重新拆分并补充依赖"],
            summary="计划评审（模拟）",
        )


def install_fake_runner(monkeypatch, *, orchestrator, review_agent, supervisor, reviews):
    """把 workflow 模块内的 Runner 替换为按 Agent 身份分派的假 Runner。"""

    class FakeRunner:
        @classmethod
        async def run(cls, starting_agent, input=None, *, max_turns=None, **kwargs):
            if starting_agent is review_agent:
                review_input = ReviewInput.model_validate_json(input)
                return FakeRunResult(reviews(review_input))
            payload = json.loads(input)
            return FakeRunResult(await supervisor(payload))

        @classmethod
        def run_streamed(cls, starting_agent, input=None, *, max_turns=None, **kwargs):
            return FakeStreamedResult(
                lambda: cls.run(
                    starting_agent, input=input, max_turns=max_turns, **kwargs
                )
            )

    monkeypatch.setattr(workflow_mod, "Runner", FakeRunner)


def build_engine(
    tmp_path, monkeypatch, *, supervisor, reviews, settings=None, scheduler=None
):
    """组装工作流引擎：注入假主管/评审 Agent（可选自定义调度器）。"""
    context = ProjectContext(sandbox_dir=tmp_path, project_name="lifecycle-exits")
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
    kwargs = {"scheduler": scheduler} if scheduler is not None else {}
    engine = WorkflowEngine(
        context,
        settings=effective_settings,
        orchestrator=orchestrator,
        review_agent=review_agent,
        **kwargs,
    )
    return context, engine


def granted_records(context) -> list[dict]:
    return [r for r in context.event_log if r["event_type"] == EVENT_TOOL_GRANTED]


def revoked_records(context) -> list[dict]:
    return [r for r in context.event_log if r["event_type"] == EVENT_TOOL_REVOKED]


def revoked_reasons(context) -> set[str]:
    return {r["payload"]["reason"] for r in revoked_records(context)}


# ===========================================================================
# 一、计划评审失败路径：被拒计划无真实签发
# ===========================================================================
class TestPlanReviewRejection:
    async def test_rejected_content_review_leaves_no_real_grant(
        self, tmp_path, monkeypatch
    ):
        settings = Settings(_env_file=None, max_replan_attempts=1)
        supervisor = GrantSupervisor()
        reviews = StageReview(plan=[False])  # 内容评审持续失败
        context, engine = build_engine(
            tmp_path, monkeypatch, supervisor=supervisor, reviews=reviews,
            settings=settings,
        )

        output = await engine.run("内容评审持续失败的任务")

        assert output.status == TaskStatus.HUMAN_REQUIRED
        assert "计划评审连续未通过" in output.final_result
        # 被拒计划：无 tool_granted 事件、网关无任何签发记录
        assert granted_records(context) == []
        assert engine._gateway.grants == {}

    async def test_rejected_authorization_review_leaves_no_real_grant(
        self, tmp_path, monkeypatch
    ):
        settings = Settings(_env_file=None, max_replan_attempts=1)
        supervisor = GrantSupervisor()
        reviews = StageReview(plan=[True], authorization=[False])
        context, engine = build_engine(
            tmp_path, monkeypatch, supervisor=supervisor, reviews=reviews,
            settings=settings,
        )

        output = await engine.run("授权评审持续失败的任务")

        assert output.status == TaskStatus.HUMAN_REQUIRED
        assert granted_records(context) == []
        assert engine._gateway.grants == {}
        # 授权评审输入只含候选（评审能看到预计授权工具，但无真实凭证）
        auth_calls = [
            c for c in reviews.calls if c.stage == ReviewStage.AUTHORIZATION_REVIEW
        ]
        assert auth_calls
        assert all(call.tool_grants for call in auth_calls)
        assert all(
            grant.grant_id.startswith("proposal-")
            for call in auth_calls
            for grant in call.tool_grants
        )


# ===========================================================================
# 二、通过路径：两路评审均通过后才真实签发
# ===========================================================================
class TestBothReviewsPassIssuance:
    async def test_reject_then_pass_issues_only_after_reviews(
        self, tmp_path, monkeypatch
    ):
        settings = Settings(_env_file=None, max_replan_attempts=2)
        supervisor = GrantSupervisor()
        reviews = StageReview(plan=[True], authorization=[False, True])
        context, engine = build_engine(
            tmp_path, monkeypatch, supervisor=supervisor, reviews=reviews,
            settings=settings,
        )
        reviews.grant_counter = lambda: len(granted_records(context))

        output = await engine.run("授权评审先拒后过的任务", acceptance_criteria="通过验收")

        assert output.status == TaskStatus.COMPLETED
        # 时序证据：每次授权评审时刻，网关均无真实签发
        assert reviews.grant_counts == [0, 0]
        # 通过后恰好签发一次
        assert len(granted_records(context)) == 1
        # 执行载荷的 grant 是真实签发（非候选前缀，工具白名单一致）
        payload = supervisor.exec_payloads["s1"][0]
        assert payload["grant"] is not None
        assert not payload["grant"]["grant_id"].startswith("proposal-")
        assert payload["grant"]["allowed_tools"] == ["write_file"]
        # 执行窗口结束后回收
        assert "execution_window_closed" in revoked_reasons(context)


# ===========================================================================
# 三、正常完成：执行窗口回收 + 完成后旧授权不可复用
# ===========================================================================
class TestCompletedPath:
    async def test_completed_task_revokes_grants(self, tmp_path, monkeypatch):
        supervisor = GrantSupervisor()
        reviews = StageReview()
        context, engine = build_engine(
            tmp_path, monkeypatch, supervisor=supervisor, reviews=reviews
        )

        output = await engine.run("正常完成的任务", acceptance_criteria="通过验收")

        assert output.status == TaskStatus.COMPLETED
        # 唯一带工具的子任务（s1）经历一次签发与一次批次回收
        records = revoked_records(context)
        assert len(records) == 1
        assert records[0]["payload"]["reason"] == "execution_window_closed"
        # 全部授权记录均为已撤销；完成后旧授权无法通过校验
        grants = list(engine._gateway.grants.values())
        assert grants and all(grant.revoked for grant in grants)
        task_id = engine._task_request.task_id
        assert (
            engine._gateway.verify_grant(
                f"{task_id}:s1:grant-slot", "write_file", task_id
            )
            is False
        )

    async def test_repeated_cleanup_is_idempotent(self, tmp_path, monkeypatch):
        supervisor = GrantSupervisor()
        reviews = StageReview()
        context, engine = build_engine(
            tmp_path, monkeypatch, supervisor=supervisor, reviews=reviews
        )
        output = await engine.run("重复清理的任务", acceptance_criteria="通过验收")
        assert output.status == TaskStatus.COMPLETED
        before = len(revoked_records(context))
        assert before == 1

        extra = engine._revoke_active_grants(reason="lifecycle_cleanup:repeat")

        assert extra == []
        assert len(revoked_records(context)) == before


# ===========================================================================
# 四、结果评审失败 → 人工升级：无活跃授权残留
# ===========================================================================
class TestResultReviewFailure:
    async def test_unresolvable_result_rework_escalates_clean(
        self, tmp_path, monkeypatch
    ):
        settings = Settings(_env_file=None, max_replan_attempts=1)
        supervisor = GrantSupervisor()
        # retry_target=None：无法定位返工目标 → 直接人工兜底
        reviews = StageReview(result=[False], result_retry_target=None)
        context, engine = build_engine(
            tmp_path, monkeypatch, supervisor=supervisor, reviews=reviews,
            settings=settings,
        )

        output = await engine.run("结果评审不通过的任务", acceptance_criteria="通过验收")

        assert output.status == TaskStatus.HUMAN_REQUIRED
        assert "无法定位返工目标" in output.final_result
        grants = list(engine._gateway.grants.values())
        assert grants and all(grant.revoked for grant in grants)
        # 批次回收完成；最外层清理为空操作（不产生额外事件）
        assert revoked_reasons(context) == {"execution_window_closed"}


# ===========================================================================
# 五、pipeline timeout：统一清理带原因审计
# ===========================================================================
class TestPipelineTimeoutCleanup:
    async def test_timeout_cleans_with_reason(self, tmp_path, monkeypatch):
        settings = Settings(_env_file=None, max_pipeline_timeout=0.5)
        supervisor = GrantSupervisor(hang_execute=True)
        reviews = StageReview()
        context, engine = build_engine(
            tmp_path, monkeypatch, supervisor=supervisor, reviews=reviews,
            settings=settings,
        )

        output = await engine.run("执行阶段挂起的任务")

        assert output.status == TaskStatus.HUMAN_REQUIRED
        assert "超时" in output.final_result
        # 授权在挂起前已签发；超时取消跳过批次回收，由最外层统一清理
        assert granted_records(context)
        assert "lifecycle_cleanup:pipeline_timeout" in revoked_reasons(context)
        grants = list(engine._gateway.grants.values())
        assert grants and all(grant.revoked for grant in grants)


# ===========================================================================
# 六、外部取消：统一清理带原因审计
# ===========================================================================
class TestCancellationCleanup:
    async def test_cancel_cleans_with_reason(self, tmp_path, monkeypatch):
        started = asyncio.Event()
        supervisor = GrantSupervisor(
            hang_execute=True, on_execute=lambda payload: started.set()
        )
        reviews = StageReview()
        context, engine = build_engine(
            tmp_path, monkeypatch, supervisor=supervisor, reviews=reviews
        )

        task = asyncio.create_task(engine.run("执行中被取消的任务"))
        await asyncio.wait_for(started.wait(), timeout=10)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        assert engine.current_status == TaskStatus.CANCELLED
        assert "lifecycle_cleanup:cancelled" in revoked_reasons(context)
        grants = list(engine._gateway.grants.values())
        assert grants and all(grant.revoked for grant in grants)


# ===========================================================================
# 七、未捕获异常：FAILED 终态 + 统一清理带原因审计
# ===========================================================================
class ExplodingScheduler:
    """调度器抛非 SchedulerError 异常：穿透批次回收，依赖最外层清理兜底。"""

    async def schedule(self, task_id, specs):
        raise RuntimeError("scheduler exploded")


class TestUnhandledExceptionCleanup:
    async def test_unhandled_exception_cleans_with_reason(
        self, tmp_path, monkeypatch
    ):
        supervisor = GrantSupervisor()
        reviews = StageReview()
        context, engine = build_engine(
            tmp_path, monkeypatch, supervisor=supervisor, reviews=reviews,
            scheduler=ExplodingScheduler(),
        )

        output = await engine.run("调度器爆炸的任务")

        assert output.status == TaskStatus.FAILED
        assert "RuntimeError" in output.final_result
        # 授权已签发且批次回收被异常跳过 → 由最外层统一清理
        assert granted_records(context)
        assert "lifecycle_cleanup:unhandled_exception" in revoked_reasons(context)
        grants = list(engine._gateway.grants.values())
        assert grants and all(grant.revoked for grant in grants)

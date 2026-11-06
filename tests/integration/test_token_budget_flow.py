"""集成测试：任务级 token 预算熔断（模拟 Runner，不消耗真实 API）。

覆盖三条路径：

1. 顶层模型调用超限 → ``TokenBudgetExceeded`` 穿透协调器宽捕获，任务
   直接失败，不再进入评审 / 执行 / 重规划；
2. 嵌套 Agent 用量超限（顶层记账标签仍是 supervisor）→ 同样熔断；
3. 显式关闭预算 → 只累计不拦截，任务正常完成（默认启用后的回退口）。
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

import so_agent.workflow as workflow_mod
from so_agent.config import Settings
from so_agent.context import ProjectContext
from so_agent.models import TaskStatus
from so_agent.runtime.budget import BudgetViolation, TokenBudgetExceeded
from so_agent.runtime.events import EVENT_TOKEN_BUDGET_EXCEEDED
from so_agent.workflow import WorkflowEngine

from tests.integration.test_replan_flow import FakeSupervisor, ReviewScript


class UsageRunResult:
    """假 RunResult：携带 usage 的 raw_responses，供预算记账提取。"""

    def __init__(self, final_output, tokens: int) -> None:
        self.final_output = final_output
        usage = SimpleNamespace(
            input_tokens=tokens, output_tokens=0, total_tokens=tokens
        )
        self.raw_responses = [SimpleNamespace(usage=usage)]


class UsageStreamedResult:
    """假 RunResultStreaming：首次取流式事件时求值并回填输出与响应。"""

    def __init__(self, run_call) -> None:
        self._run_call = run_call
        self.final_output = None
        self.raw_responses = []

    def cancel(self, mode="immediate") -> None:
        pass

    async def stream_events(self):
        result = await self._run_call()
        self.final_output = result.final_output
        self.raw_responses = getattr(result, "raw_responses", None) or []
        yield {"type": "fake_event"}


def install_usage_runner(
    monkeypatch, *, orchestrator, review_agent, supervisor, reviews, tokens: int
) -> None:
    """安装按 Agent 身份分派的假 Runner，每次主管调用记 ``tokens`` 用量。"""

    class FakeRunner:
        @classmethod
        async def run(cls, starting_agent, input=None, *, max_turns=None, **kwargs):
            if starting_agent is review_agent:
                from so_agent.tool_packages.review_agent.agent import ReviewInput

                review_input = ReviewInput.model_validate_json(input)
                return UsageRunResult(reviews(review_input), 0)
            payload = json.loads(input)
            return UsageRunResult(supervisor(payload), tokens)

        @classmethod
        def run_streamed(cls, starting_agent, input=None, *, max_turns=None, **kwargs):
            return UsageStreamedResult(
                lambda: cls.run(
                    starting_agent, input=input, max_turns=max_turns, **kwargs
                )
            )

    monkeypatch.setattr(workflow_mod, "Runner", FakeRunner)


def build_engine(
    tmp_path,
    monkeypatch,
    *,
    supervisor,
    reviews,
    settings: Settings,
    tokens: int,
) -> tuple[ProjectContext, WorkflowEngine]:
    context = ProjectContext(sandbox_dir=tmp_path, project_name="budget-flow")
    effective_settings = settings.model_copy(
        update={"skip_plan_review_for_simple": False}
    )
    orchestrator = SimpleNamespace(name="fake-supervisor")
    review_agent = SimpleNamespace(name="fake-reviewer")
    install_usage_runner(
        monkeypatch,
        orchestrator=orchestrator,
        review_agent=review_agent,
        supervisor=supervisor,
        reviews=reviews,
        tokens=tokens,
    )
    engine = WorkflowEngine(
        context,
        settings=effective_settings,
        orchestrator=orchestrator,
        review_agent=review_agent,
    )
    return context, engine


class TestTaskBudgetTrip:
    async def test_top_level_usage_over_limit_fails_without_retry(
        self, tmp_path, monkeypatch
    ):
        supervisor = FakeSupervisor()
        reviews = ReviewScript([True])
        settings = Settings(
            _env_file=None,
            token_budget_enabled=True,
            token_budget_max_total_tokens=10,
            token_budget_max_agent_tokens=0,
        )
        context, engine = build_engine(
            tmp_path,
            monkeypatch,
            supervisor=supervisor,
            reviews=reviews,
            settings=settings,
            tokens=20,
        )

        output = await engine.run("预算熔断场景")

        assert output.status == TaskStatus.FAILED
        assert engine.current_status == TaskStatus.FAILED
        assert "token 预算熔断" in output.final_result
        # 首次规划调用即超限：不进入评审、执行或重规划
        assert supervisor.phases == ["planning"]
        assert reviews.calls == []
        assert engine.replan_count == 0
        events = context.event_log
        exceeded = [
            record
            for record in events
            if record["event_type"] == EVENT_TOKEN_BUDGET_EXCEEDED
        ]
        assert len(exceeded) == 1
        assert exceeded[0]["payload"]["scope"] == "task"
        assert exceeded[0]["payload"]["limit"] == 10
        assert exceeded[0]["payload"]["observed"] == 20

    async def test_nested_agent_over_limit_fails_even_when_total_is_fine(
        self, tmp_path, monkeypatch
    ):
        class NestedUsageSupervisor(FakeSupervisor):
            def __call__(self, payload: dict):
                result = super().__call__(payload)
                if payload.get("phase") == "planning":
                    context.token_budget.observe(
                        "code_agent",
                        {"requests": 1, "input_tokens": 80,
                         "output_tokens": 0, "total_tokens": 80},
                    )
                return result

        supervisor = NestedUsageSupervisor()
        reviews = ReviewScript([True])
        settings = Settings(
            _env_file=None,
            token_budget_enabled=True,
            token_budget_max_total_tokens=0,
            token_budget_max_agent_tokens=50,
        )
        context, engine = build_engine(
            tmp_path,
            monkeypatch,
            supervisor=supervisor,
            reviews=reviews,
            settings=settings,
            tokens=5,
        )

        output = await engine.run("嵌套 Agent 超限")

        assert output.status == TaskStatus.FAILED
        assert "code_agent" in output.final_result
        assert supervisor.phases == ["planning"]
        assert reviews.calls == []


class TestProvisionalBudgetTrip:
    async def test_background_review_budget_trip_marks_failed(
        self, tmp_path, monkeypatch
    ):
        """后台评审中的预算熔断必须收口为 failed，不得逃逸为未处理异常。"""
        settings = Settings(
            _env_file=None,
            token_budget_enabled=True,
            token_budget_max_total_tokens=10,
            token_budget_max_agent_tokens=0,
        )
        _, engine = build_engine(
            tmp_path,
            monkeypatch,
            supervisor=FakeSupervisor(),
            reviews=ReviewScript([True]),
            settings=settings,
            tokens=0,
        )
        transitions: list[TaskStatus] = []

        async def raise_budget(**kwargs):
            raise TokenBudgetExceeded(
                BudgetViolation(
                    scope="task",
                    limit=10,
                    observed=20,
                    message="任务 token 用量 20 超过上限 10，已触发预算熔断",
                )
            )

        monkeypatch.setattr(engine, "_final_review", raise_budget)
        monkeypatch.setattr(
            engine, "_transition", lambda status, **kw: transitions.append(status)
        )

        await engine._run_provisional_review("已交付的初步结果")

        output = engine._provisional_output
        assert output is not None
        assert output.status == TaskStatus.FAILED
        assert "token 预算熔断" in output.final_result
        assert output.previous_result == "已交付的初步结果"
        assert output.notice
        assert TaskStatus.FAILED in transitions


class TestBudgetDisabled:
    async def test_disabled_budget_still_completes(
        self, tmp_path, monkeypatch
    ):
        supervisor = FakeSupervisor()
        reviews = ReviewScript([True])
        settings = Settings(
            _env_file=None,
            token_budget_enabled=False,
            token_budget_max_total_tokens=10,
            token_budget_max_agent_tokens=0,
        )
        context, engine = build_engine(
            tmp_path,
            monkeypatch,
            supervisor=supervisor,
            reviews=reviews,
            settings=settings,
            tokens=999,
        )

        output = await engine.run("关闭预算后正常完成")

        assert output.status == TaskStatus.COMPLETED
        snapshot = engine.token_budget_snapshot()
        assert snapshot["active"] is False
        assert snapshot["total_tokens"] > 10
        assert [
            record
            for record in context.event_log
            if record["event_type"] == EVENT_TOKEN_BUDGET_EXCEEDED
        ] == []

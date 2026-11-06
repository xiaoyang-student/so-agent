"""单元测试：token 预算接口、默认启用与熔断事件（任务 33）。"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from so_agent.config import Settings
from so_agent.runtime.budget import (
    TokenBudgetExceeded,
    TokenBudgetLimits,
    TokenBudgetTracker,
)
from so_agent.runtime.events import (
    EVENT_TOKEN_BUDGET_EXCEEDED,
    EVENT_TOKEN_USAGE,
    EventLogger,
)
from so_agent.runtime.usage import (
    log_model_usage,
    make_usage_recorder,
    record_completion_usage,
)
from so_agent.workflow import WorkflowEngine


def _usage(tokens: int) -> dict[str, int]:
    return {
        "requests": 1,
        "input_tokens": tokens,
        "output_tokens": 0,
        "total_tokens": tokens,
    }


def _fake_result(tokens: int) -> SimpleNamespace:
    usage = SimpleNamespace(
        input_tokens=tokens, output_tokens=0, total_tokens=tokens
    )
    return SimpleNamespace(raw_responses=[SimpleNamespace(usage=usage)])


class TestLimitsAndConfig:
    def test_default_settings_enable_control(self) -> None:
        settings = Settings()
        limits = TokenBudgetLimits.from_settings(settings)
        assert limits.enabled is True
        assert limits.active is True
        assert limits.max_total_tokens == 800_000
        assert limits.max_agent_tokens == 400_000

    def test_enabled_without_limits_is_config_error(self) -> None:
        with pytest.raises(ValueError, match="至少配置一个正上限"):
            Settings(
                token_budget_enabled=True,
                token_budget_max_total_tokens=0,
                token_budget_max_agent_tokens=0,
            )

    def test_explicit_disable_is_allowed_without_limits(self) -> None:
        limits = TokenBudgetLimits.from_settings(
            Settings(
                token_budget_enabled=False,
                token_budget_max_total_tokens=0,
                token_budget_max_agent_tokens=0,
            )
        )
        assert limits.active is False

    def test_negative_limit_is_config_error(self) -> None:
        with pytest.raises(ValueError, match="不能为负数"):
            Settings(token_budget_max_total_tokens=-1)

    def test_enabled_with_limit_is_active(self) -> None:
        limits = TokenBudgetLimits.from_settings(
            Settings(
                token_budget_enabled=True,
                token_budget_max_total_tokens=100,
            )
        )
        assert limits.active is True


class TestTrackerDefaultOff:
    def test_accumulates_without_enforcement_or_events(self) -> None:
        events = EventLogger()
        tracker = TokenBudgetTracker(
            TokenBudgetLimits(enabled=False, max_total_tokens=10),
            events=events,
        )
        tracker.observe("supervisor", _usage(50))
        tracker.observe("code_agent", _usage(60))
        assert tracker.total_tokens == 110
        assert tracker.tokens_for("supervisor") == 50
        assert tracker.check("supervisor") is None
        tracker.enforce("supervisor")  # 不抛异常
        assert events.get_events(event_type=EVENT_TOKEN_BUDGET_EXCEEDED) == []

    def test_snapshot_reports_accumulation(self) -> None:
        tracker = TokenBudgetTracker(TokenBudgetLimits())
        tracker.observe("review_agent", _usage(7))
        snapshot = tracker.snapshot()
        assert snapshot["active"] is False
        assert snapshot["total_tokens"] == 7
        assert snapshot["per_agent"] == {"review_agent": 7}
        assert snapshot["violations"] == []


class TestTrackerEnforcement:
    def test_task_limit_logs_once_and_enforce_raises(self) -> None:
        events = EventLogger()
        tracker = TokenBudgetTracker(
            TokenBudgetLimits(enabled=True, max_total_tokens=100),
            events=events,
            task_id="task-1",
        )
        tracker.observe("supervisor", _usage(60))
        assert tracker.check() is None
        tracker.observe("supervisor", _usage(60))
        assert tracker.check() is not None
        with pytest.raises(TokenBudgetExceeded) as excinfo:
            tracker.enforce("supervisor")
        assert excinfo.value.violation.scope == "task"
        records = events.get_events(event_type=EVENT_TOKEN_BUDGET_EXCEEDED)
        assert len(records) == 1
        payload = records[0]["payload"]
        assert payload["scope"] == "task"
        assert payload["limit"] == 100
        assert payload["observed"] == 120
        assert records[0]["task_id"] == "task-1"

    def test_agent_limit_isolated_per_agent(self) -> None:
        tracker = TokenBudgetTracker(
            TokenBudgetLimits(enabled=True, max_agent_tokens=50)
        )
        tracker.observe("supervisor", _usage(30))
        tracker.observe("code_agent", _usage(60))
        assert tracker.check("supervisor") is None
        violation = tracker.check("code_agent")
        assert violation is not None
        assert violation.scope == "agent"
        assert violation.agent_id == "code_agent"

    def test_check_without_agent_scans_all_agents(self) -> None:
        tracker = TokenBudgetTracker(
            TokenBudgetLimits(enabled=True, max_agent_tokens=50)
        )
        tracker.observe("supervisor", _usage(30))
        tracker.observe("code_agent", _usage(60))
        violation = tracker.check()
        assert violation is not None
        assert violation.scope == "agent"
        assert violation.agent_id == "code_agent"

    def test_repeated_reports_are_deduplicated(self) -> None:
        events = EventLogger()
        tracker = TokenBudgetTracker(
            TokenBudgetLimits(enabled=True, max_total_tokens=10),
            events=events,
        )
        tracker.observe("supervisor", _usage(20))
        tracker.observe("supervisor", _usage(20))
        assert len(
            events.get_events(event_type=EVENT_TOKEN_BUDGET_EXCEEDED)
        ) == 1


class TestUsageWiring:
    def test_log_model_usage_feeds_tracker(self) -> None:
        events = EventLogger()
        tracker = TokenBudgetTracker(
            TokenBudgetLimits(enabled=True, max_total_tokens=100),
            events=events,
        )
        log_model_usage(events, _fake_result(40), "supervisor", budget=tracker)
        assert tracker.total_tokens == 40
        assert events.get_events(event_type=EVENT_TOKEN_USAGE)[0]["payload"][
            "total_tokens"
        ] == 40

    def test_recorder_picks_up_context_tracker_lazily(self) -> None:
        events = EventLogger()
        context = SimpleNamespace(event_logger=events, token_budget=None)
        recorder = make_usage_recorder(context, "subagent_creator")
        tracker = TokenBudgetTracker(TokenBudgetLimits())
        context.token_budget = tracker
        recorder(_fake_result(25))
        assert tracker.tokens_for("subagent_creator") == 25

    def test_record_completion_usage_logs_chat_completion(self) -> None:
        events = EventLogger()
        context = SimpleNamespace(event_logger=events, token_budget=None)
        response = SimpleNamespace(
            usage=SimpleNamespace(
                prompt_tokens=11, completion_tokens=7, total_tokens=18
            )
        )
        record_completion_usage(context, response, "direct_answer")
        record = events.get_events(event_type=EVENT_TOKEN_USAGE)[0]
        assert record["payload"]["agent"] == "direct_answer"
        assert record["payload"]["input_tokens"] == 11
        assert record["payload"]["output_tokens"] == 7
        assert record["payload"]["total_tokens"] == 18
        assert record["payload"]["requests"] == 1

    def test_record_completion_usage_without_usage_field(self) -> None:
        events = EventLogger()
        context = SimpleNamespace(event_logger=events, token_budget=None)
        record_completion_usage(context, SimpleNamespace(), "router_classifier")
        payload = events.get_events(event_type=EVENT_TOKEN_USAGE)[0]["payload"]
        assert payload["requests"] == 0
        assert payload["total_tokens"] == 0

    def test_engine_record_usage_is_noop_when_disabled(self) -> None:
        events = EventLogger()
        stub = SimpleNamespace(_events=events, _token_budget=None)
        stub._enforce_token_budget = lambda: WorkflowEngine._enforce_token_budget(
            stub
        )
        WorkflowEngine._record_model_usage(
            stub, _fake_result(999), "supervisor"
        )
        assert events.get_events(event_type=EVENT_TOKEN_USAGE)

    def test_engine_record_usage_raises_when_enabled_and_exceeded(self) -> None:
        events = EventLogger()
        tracker = TokenBudgetTracker(
            TokenBudgetLimits(enabled=True, max_total_tokens=10),
            events=events,
        )
        stub = SimpleNamespace(_events=events, _token_budget=tracker)
        stub._enforce_token_budget = lambda: WorkflowEngine._enforce_token_budget(
            stub
        )
        with pytest.raises(TokenBudgetExceeded):
            WorkflowEngine._record_model_usage(
                stub, _fake_result(11), "supervisor"
            )

    def test_engine_record_usage_checks_nested_agent_limits(self) -> None:
        """顶层记账标签是 supervisor，但嵌套 Agent 累计超限也必须熔断。"""
        events = EventLogger()
        tracker = TokenBudgetTracker(
            TokenBudgetLimits(enabled=True, max_agent_tokens=50),
            events=events,
        )
        stub = SimpleNamespace(_events=events, _token_budget=tracker)
        stub._enforce_token_budget = lambda: WorkflowEngine._enforce_token_budget(
            stub
        )
        tracker.observe("code_agent", _usage(80))
        with pytest.raises(TokenBudgetExceeded) as excinfo:
            WorkflowEngine._record_model_usage(
                stub, _fake_result(1), "supervisor"
            )
        assert excinfo.value.violation.agent_id == "code_agent"

    def test_remaining_tokens_reports_headroom(self) -> None:
        tracker = TokenBudgetTracker(
            TokenBudgetLimits(enabled=True, max_total_tokens=100)
        )
        tracker.observe("supervisor", _usage(30))
        assert tracker.remaining_tokens == 70
        assert tracker.snapshot()["remaining_tokens"] == 70

    def test_remaining_tokens_is_none_without_task_limit(self) -> None:
        tracker = TokenBudgetTracker(
            TokenBudgetLimits(enabled=True, max_agent_tokens=50)
        )
        tracker.observe("supervisor", _usage(30))
        assert tracker.remaining_tokens is None

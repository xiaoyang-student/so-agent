"""确定性审计报告测试：计数、判定、证据引用与任务隔离。"""

from __future__ import annotations

import hashlib

from so_agent.models import (
    AgentRecord,
    ExecutionResult,
    SubtaskSpec,
    TaskStatus,
)
from so_agent.runtime.audit import build_audit_report
from so_agent.runtime.events import (
    EVENT_HUMAN_ESCALATION,
    EVENT_REVIEW_AGENT_OUTPUT,
    EVENT_RUNTIME_BREAKER_TRIPPED,
    EVENT_TASK_DISPATCHED,
    EVENT_TOKEN_BUDGET_EXCEEDED,
    EVENT_TOKEN_USAGE,
    EVENT_TOOL_GRANTED,
    EVENT_TOOL_REVOKED,
    EVENT_WRITE_SCOPE_UNMET,
)


def _event(
    event_id: str,
    event_type: str,
    payload: dict | None = None,
    *,
    task_id: str = "t1",
    timestamp: str = "2026-01-01T00:00:00+00:00",
) -> dict:
    return {
        "event_id": event_id,
        "event_type": event_type,
        "timestamp": timestamp,
        "task_id": task_id,
        "agent_id": None,
        "payload": payload or {},
    }


def _plans() -> list[list[SubtaskSpec]]:
    version1 = [
        SubtaskSpec(
            subtask_id="s1",
            plan_version=1,
            title="采集数据",
            instructions="采集",
            role="code",
        )
    ]
    version2 = [
        SubtaskSpec(
            subtask_id="s1",
            plan_version=2,
            title="采集数据",
            instructions="采集",
            role="code",
        ),
        SubtaskSpec(
            subtask_id="s2",
            plan_version=2,
            title="生成报告",
            instructions="生成",
            role="code",
            dependencies=["s1"],
        ),
    ]
    return [version1, version2]


def _results() -> dict[str, ExecutionResult]:
    return {
        "s1": ExecutionResult(
            subtask_id="s1", success=True, output="采集完成", artifacts=["data.csv"]
        ),
        "s2": ExecutionResult(
            subtask_id="s2",
            success=False,
            error="生成失败",
            evidence=["evidence-1", "evidence-2"],
        ),
    }


def _agents() -> list[AgentRecord]:
    return [
        AgentRecord(
            agent_id="a1",
            agent_type="code_agent",
            parent_task_id="t1",
            plan_version=2,
            allowed_tools=["read_file", "write_file"],
            attempt_count=2,
            status="completed",
        )
    ]


def _events() -> list[dict]:
    return [
        _event("e1", EVENT_TASK_DISPATCHED, {"subtask_id": "s1", "attempt": 1}),
        _event("e2", EVENT_TASK_DISPATCHED, {"subtask_id": "s2", "attempt": 2}),
        _event(
            "e3",
            EVENT_TOOL_GRANTED,
            {
                "grant": {
                    "grant_id": "g1",
                    "agent_id": "a1",
                    "allowed_tools": ["read_file", "write_file"],
                }
            },
        ),
        _event(
            "e4",
            EVENT_TOOL_REVOKED,
            {"grant": {"grant_id": "g1", "agent_id": "a1"}},
        ),
        _event(
            "e5",
            EVENT_REVIEW_AGENT_OUTPUT,
            {"stage": "plan_review", "passed": False},
        ),
        _event(
            "e6",
            EVENT_TOKEN_USAGE,
            {
                "agent": "supervisor",
                "requests": 1,
                "input_tokens": 60,
                "output_tokens": 40,
                "total_tokens": 100,
            },
        ),
        _event(
            "e7",
            EVENT_TOKEN_USAGE,
            {
                "agent": "code_agent",
                "requests": 2,
                "input_tokens": 20,
                "output_tokens": 10,
            },
        ),
        _event("e8", EVENT_WRITE_SCOPE_UNMET, {"subtask_id": "s2"}),
        _event(
            "e9",
            EVENT_HUMAN_ESCALATION,
            {"reason": "执行阶段持续失败"},
            timestamp="2026-01-01T00:00:02+00:00",
        ),
        _event(
            "other",
            EVENT_TOKEN_USAGE,
            {"agent": "supervisor", "total_tokens": 999},
            task_id="t2",
        ),
    ]


def test_report_is_deterministic_and_task_scoped():
    kwargs = dict(
        events=_events(),
        task_id="t1",
        status=TaskStatus.HUMAN_REQUIRED,
        final_result="任务需要人工介入",
        review_summary="[result_review] 未通过",
        plan_versions=_plans(),
        agent_history=_agents(),
        results=_results(),
    )
    first = build_audit_report(**kwargs)
    second = build_audit_report(**kwargs)

    assert first.model_dump(mode="json") == second.model_dump(mode="json")
    assert first.task_id == "t1"
    assert first.event_count == 9
    assert first.event_counts[EVENT_TOKEN_USAGE] == 2


def test_report_aggregates_counts_results_and_usage():
    report = build_audit_report(
        events=_events(),
        task_id="t1",
        status=TaskStatus.HUMAN_REQUIRED,
        final_result="任务需要人工介入",
        plan_versions=_plans(),
        agent_history=_agents(),
        results=_results(),
    )

    assert [plan.version for plan in report.plan_versions] == [1, 2]
    assert report.plan_versions[-1].subtask_ids == ["s1", "s2"]
    assert [item.subtask_id for item in report.results] == ["s1", "s2"]
    assert report.results[0].artifact_count == 1
    assert report.results[1].evidence_count == 2
    assert report.counts["plans"] == 2
    assert report.counts["subtasks"] == 2
    assert report.counts["completed"] == 1
    assert report.counts["failed"] == 1
    assert report.counts["dispatches"] == 2
    assert report.counts["reviews"] == 1
    assert report.counts["review_failures"] == 1
    assert report.counts["human_escalations"] == 1

    assert [entry.agent for entry in report.usage] == ["code_agent", "supervisor"]
    assert report.total_tokens == 130
    assert report.usage[0].requests == 2
    assert report.usage[1].input_tokens == 60

    assert report.grants[0].grant_id == "g1"
    assert report.grants[0].revoked is True
    assert report.agents[0].attempt_count == 2


def test_report_findings_and_verdict_for_escalation():
    report = build_audit_report(
        events=_events(),
        task_id="t1",
        status=TaskStatus.HUMAN_REQUIRED,
        final_result="任务需要人工介入",
        plan_versions=_plans(),
        results=_results(),
    )
    codes = {finding.code for finding in report.findings}

    assert report.verdict == "warn"
    assert {
        "human_required",
        "subtask_failures",
        "review_rejection",
        "write_scope_unmet",
    } <= codes
    human = next(f for f in report.findings if f.code == "human_required")
    assert human.refs == ["e9"]
    failures = next(f for f in report.findings if f.code == "subtask_failures")
    assert failures.refs == ["s2"]


def test_report_clean_completed_run_passes():
    report = build_audit_report(
        events=[_event("e1", EVENT_TASK_DISPATCHED, {"subtask_id": "s1"})],
        task_id="t1",
        status=TaskStatus.COMPLETED,
        final_result="全部完成",
        plan_versions=_plans()[:1],
        results={
            "s1": ExecutionResult(
                subtask_id="s1", success=True, output="完成"
            )
        },
    )

    assert report.verdict == "pass"
    assert report.findings == []
    assert report.final_result_chars == len("全部完成")
    assert report.final_result_sha256 == hashlib.sha256(
        "全部完成".encode("utf-8")
    ).hexdigest()
    assert report.last_event_at == "2026-01-01T00:00:00+00:00"


def test_report_marks_errors_for_failed_status_and_budget():
    events = [
        _event("b1", EVENT_TOKEN_BUDGET_EXCEEDED, {"scope": "task", "limit": 10}),
        _event("r1", EVENT_RUNTIME_BREAKER_TRIPPED, {"reason": "provider_pressure"}),
    ]
    report = build_audit_report(
        events=events, task_id="t1", status=TaskStatus.FAILED
    )

    codes = {finding.code for finding in report.findings}
    assert report.verdict == "fail"
    assert {"task_failed", "token_budget_exceeded", "runtime_breaker_tripped"} <= codes
    budget = next(f for f in report.findings if f.code == "token_budget_exceeded")
    assert budget.level == "error"
    assert budget.refs == ["b1"]

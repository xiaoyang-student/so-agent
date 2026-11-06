"""确定性审计报告构建。

从既有事件流、计划版本、Agent 生命周期与执行结果推导结构化审计报告：
不调用模型、不读取当前时间、不复制完整业务产出，同一输入必得同一输出。
报告用于终结态评估、Web 展示与外部审计复核。
"""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from so_agent.models import (
    AgentRecord,
    AuditAgentSummary,
    AuditFinding,
    AuditGrantSummary,
    AuditPlanSummary,
    AuditReport,
    AuditSubtaskResult,
    AuditUsageEntry,
    ExecutionResult,
    SubtaskSpec,
    TaskStatus,
)
from so_agent.runtime.events import (
    EVENT_AGENT_CREATED,
    EVENT_DECOMPOSITION_ISSUE,
    EVENT_HUMAN_ESCALATION,
    EVENT_HUMAN_GUIDANCE_APPLIED,
    EVENT_REPLAN,
    EVENT_REPLACEMENT_REQUESTED,
    EVENT_RESULT_REUSED,
    EVENT_RETRY_SUPPRESSED,
    EVENT_REUSE_REJECTED,
    EVENT_REVIEW_AGENT_OUTPUT,
    EVENT_REVIEW_CONTRACT_ERROR,
    EVENT_REVIEW_SKIPPED,
    EVENT_RUNTIME_BREAKER_TRIPPED,
    EVENT_TASK_DISPATCHED,
    EVENT_TOKEN_BUDGET_EXCEEDED,
    EVENT_TOKEN_USAGE,
    EVENT_TOOL_CALLED,
    EVENT_TOOL_DENIED,
    EVENT_TOOL_GRANTED,
    EVENT_TOOL_REVOKED,
    EVENT_WRITE_SCOPE_UNMET,
)

_FAILED_STATUSES: frozenset[str] = frozenset(
    {TaskStatus.FAILED.value, TaskStatus.CANCELLED.value}
)
_WARNING_STATUSES: frozenset[str] = frozenset({TaskStatus.HUMAN_REQUIRED.value})
_REVIEW_STAGES: frozenset[str] = frozenset({"plan_review", "result_review"})


def _as_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _status_value(status: TaskStatus | str) -> str:
    return status.value if isinstance(status, TaskStatus) else str(status)


def _scoped_events(
    events: Iterable[dict[str, Any]], task_id: str
) -> list[dict[str, Any]]:
    """按任务过滤事件；无 task_id 的事件（全局事件）一并保留。"""
    scoped: list[dict[str, Any]] = []
    for record in events:
        if not isinstance(record, dict):
            continue
        record_task = record.get("task_id")
        if task_id and record_task not in (None, task_id):
            continue
        scoped.append(record)
    return scoped


def _plan_summaries(
    plan_versions: Sequence[Sequence[SubtaskSpec]],
) -> list[AuditPlanSummary]:
    return [
        AuditPlanSummary(
            version=index,
            subtask_ids=[spec.subtask_id for spec in specs],
        )
        for index, specs in enumerate(plan_versions, start=1)
    ]


def _result_summaries(
    results: Mapping[str, ExecutionResult] | None,
    order: Sequence[str],
) -> list[AuditSubtaskResult]:
    if not results:
        return []
    summaries: list[AuditSubtaskResult] = []
    seen: set[str] = set()
    for subtask_id in order:
        result = results.get(subtask_id)
        if result is None:
            continue
        seen.add(subtask_id)
        summaries.append(_result_summary(result))
    for subtask_id, result in results.items():
        if subtask_id not in seen:
            summaries.append(_result_summary(result))
    return summaries


def _result_summary(result: ExecutionResult) -> AuditSubtaskResult:
    return AuditSubtaskResult(
        subtask_id=result.subtask_id,
        success=bool(result.success),
        duration=float(result.duration or 0.0),
        evidence_count=len(result.evidence or []),
        artifact_count=len(result.artifacts or []),
        output_chars=len(result.output or ""),
        error=str(result.error or ""),
    )


def _agent_summaries(
    agent_history: Sequence[AgentRecord] | None,
) -> list[AuditAgentSummary]:
    return [
        AuditAgentSummary(
            agent_id=record.agent_id,
            agent_type=record.agent_type,
            status=str(record.status),
            plan_version=record.plan_version,
            attempt_count=record.attempt_count,
            allowed_tools=list(record.allowed_tools or []),
            replacement_requested=record.replacement_requested,
            replacement_of=record.replacement_of,
        )
        for record in (agent_history or [])
    ]


def _grant_id(record: dict[str, Any]) -> str:
    payload = _as_dict(record.get("payload"))
    return str(
        payload.get("grant_id")
        or _as_dict(payload.get("grant")).get("grant_id")
        or ""
    )


def _grant_summaries(events: Sequence[dict[str, Any]]) -> list[AuditGrantSummary]:
    grants: list[AuditGrantSummary] = []
    seen: set[str] = set()
    revoked_ids = {
        _grant_id(record)
        for record in events
        if record.get("event_type") == EVENT_TOOL_REVOKED
    }
    revoked_ids.discard("")
    for record in events:
        if record.get("event_type") != EVENT_TOOL_GRANTED:
            continue
        grant = _as_dict(_as_dict(record.get("payload")).get("grant"))
        grant_id = str(grant.get("grant_id") or "")
        if not grant_id or grant_id in seen:
            continue
        seen.add(grant_id)
        grants.append(
            AuditGrantSummary(
                grant_id=grant_id,
                agent_id=str(grant.get("agent_id") or ""),
                allowed_tools=[str(item) for item in _as_list(grant.get("allowed_tools"))],
                revoked=grant_id in revoked_ids,
            )
        )
    return grants


def _usage_summaries(
    events: Sequence[dict[str, Any]],
) -> tuple[list[AuditUsageEntry], int]:
    buckets: dict[str, AuditUsageEntry] = {}
    for record in events:
        if record.get("event_type") != EVENT_TOKEN_USAGE:
            continue
        payload = _as_dict(record.get("payload"))
        agent = str(payload.get("agent") or "unknown")
        entry = buckets.setdefault(agent, AuditUsageEntry(agent=agent))
        entry.requests += max(1, _as_int(payload.get("requests")))
        entry.input_tokens += _as_int(payload.get("input_tokens"))
        entry.output_tokens += _as_int(payload.get("output_tokens"))
        entry.cached_tokens += _as_int(payload.get("cached_tokens"))
        entry.cache_write_tokens += _as_int(payload.get("cache_write_tokens"))
        reported_total = _as_int(payload.get("total_tokens"))
        entry.total_tokens += reported_total or (
            _as_int(payload.get("input_tokens"))
            + _as_int(payload.get("output_tokens"))
        )
    usage = [buckets[key] for key in sorted(buckets)]
    return usage, sum(entry.total_tokens for entry in usage)


def _review_findings(
    events: Sequence[dict[str, Any]],
) -> tuple[int, list[AuditFinding]]:
    failures: list[AuditFinding] = []
    contract_errors: list[str] = []
    skipped: list[str] = []
    for record in events:
        event_type = record.get("event_type")
        payload = _as_dict(record.get("payload"))
        event_id = str(record.get("event_id") or "")
        if event_type == EVENT_REVIEW_AGENT_OUTPUT:
            stage = str(payload.get("stage") or "")
            if stage in _REVIEW_STAGES and payload.get("passed") is False:
                failures.append(
                    AuditFinding(
                        code="review_rejection",
                        level="warning",
                        message=f"{stage} 未通过",
                        refs=[event_id],
                    )
                )
        elif event_type == EVENT_REVIEW_CONTRACT_ERROR:
            contract_errors.append(event_id)
        elif event_type == EVENT_REVIEW_SKIPPED:
            skipped.append(event_id)
    findings = failures
    if contract_errors:
        findings.append(
            AuditFinding(
                code="review_contract_errors",
                level="warning",
                message=f"评审契约错误 {len(contract_errors)} 次",
                refs=contract_errors,
            )
        )
    if skipped:
        findings.append(
            AuditFinding(
                code="review_skipped",
                level="info",
                message=f"确定性条件跳过评审 {len(skipped)} 次",
                refs=skipped,
            )
        )
    return len(failures), findings


def build_audit_report(
    *,
    events: Iterable[dict[str, Any]],
    task_id: str = "",
    status: TaskStatus | str = "",
    final_result: str = "",
    review_summary: str = "",
    plan_versions: Sequence[Sequence[SubtaskSpec]] | None = None,
    agent_history: Sequence[AgentRecord] | None = None,
    results: Mapping[str, ExecutionResult] | None = None,
) -> AuditReport:
    """构建确定性审计报告（纯推导，无副作用）。"""
    plans = list(plan_versions or [])
    status_value = _status_value(status)
    scoped = _scoped_events(events, task_id)
    event_counts = Counter(str(record.get("event_type")) for record in scoped)
    latest_ids = [spec.subtask_id for spec in plans[-1]] if plans else []
    result_summaries = _result_summaries(results, latest_ids)
    usage, total_tokens = _usage_summaries(scoped)

    def refs(event_type: str, limit: int = 20) -> list[str]:
        return [
            str(record.get("event_id") or "")
            for record in scoped
            if record.get("event_type") == event_type
        ][:limit]

    findings: list[AuditFinding] = []
    if status_value in _FAILED_STATUSES:
        findings.append(
            AuditFinding(
                code=f"task_{status_value}",
                level="error",
                message=f"任务终态为 {status_value}",
            )
        )
    elif status_value in _WARNING_STATUSES:
        findings.append(
            AuditFinding(
                code="human_required",
                level="warning",
                message="任务需要人工介入",
                refs=refs(EVENT_HUMAN_ESCALATION),
            )
        )

    failed_subtasks = [
        item.subtask_id for item in result_summaries if not item.success
    ]
    if failed_subtasks:
        findings.append(
            AuditFinding(
                code="subtask_failures",
                level="warning",
                message="失败子任务：" + "、".join(failed_subtasks),
                refs=failed_subtasks,
            )
        )

    review_failures, review_findings = _review_findings(scoped)
    findings.extend(review_findings)

    if event_counts.get(EVENT_TOOL_DENIED):
        findings.append(
            AuditFinding(
                code="tool_denials",
                level="warning",
                message=f"工具调用被拒绝 {event_counts[EVENT_TOOL_DENIED]} 次",
                refs=refs(EVENT_TOOL_DENIED),
            )
        )
    if event_counts.get(EVENT_WRITE_SCOPE_UNMET):
        findings.append(
            AuditFinding(
                code="write_scope_unmet",
                level="warning",
                message=(
                    "写入范围未满足："
                    f"{event_counts[EVENT_WRITE_SCOPE_UNMET]} 次"
                ),
                refs=refs(EVENT_WRITE_SCOPE_UNMET),
            )
        )
    if event_counts.get(EVENT_TOKEN_BUDGET_EXCEEDED):
        findings.append(
            AuditFinding(
                code="token_budget_exceeded",
                level="error",
                message=(
                    "token 预算超限："
                    f"{event_counts[EVENT_TOKEN_BUDGET_EXCEEDED]} 次"
                ),
                refs=refs(EVENT_TOKEN_BUDGET_EXCEEDED),
            )
        )
    if event_counts.get(EVENT_RETRY_SUPPRESSED):
        findings.append(
            AuditFinding(
                code="retry_suppressed",
                level="warning",
                message=f"重试被抑制 {event_counts[EVENT_RETRY_SUPPRESSED]} 次",
                refs=refs(EVENT_RETRY_SUPPRESSED),
            )
        )
    if event_counts.get(EVENT_RUNTIME_BREAKER_TRIPPED):
        findings.append(
            AuditFinding(
                code="runtime_breaker_tripped",
                level="warning",
                message="运行时熔断触发（并发收敛为串行）",
                refs=refs(EVENT_RUNTIME_BREAKER_TRIPPED),
            )
        )
    if event_counts.get(EVENT_REUSE_REJECTED):
        findings.append(
            AuditFinding(
                code="reuse_rejected",
                level="warning",
                message=(
                    "结果复用声明被拒绝："
                    f"{event_counts[EVENT_REUSE_REJECTED]} 次"
                ),
                refs=refs(EVENT_REUSE_REJECTED),
            )
        )

    levels = {finding.level for finding in findings}
    if levels & {"error"}:
        verdict = "fail"
    elif levels & {"warning"}:
        verdict = "warn"
    else:
        verdict = "pass"

    raw = (final_result or "").encode("utf-8")

    def count(event_type: str) -> int:
        return int(event_counts.get(event_type, 0))

    return AuditReport(
        task_id=task_id,
        status=status_value,
        verdict=verdict,
        plan_versions=_plan_summaries(plans),
        results=result_summaries,
        agents=_agent_summaries(agent_history),
        grants=_grant_summaries(scoped),
        usage=usage,
        total_tokens=total_tokens,
        counts={
            "plans": len(plans),
            "subtasks": len(latest_ids),
            "results": len(result_summaries),
            "completed": sum(1 for item in result_summaries if item.success),
            "failed": len(failed_subtasks),
            "dispatches": count(EVENT_TASK_DISPATCHED),
            "replans": count(EVENT_REPLAN),
            "decomposition_issues": count(EVENT_DECOMPOSITION_ISSUE),
            "replacements_requested": count(EVENT_REPLACEMENT_REQUESTED),
            "agents_created": count(EVENT_AGENT_CREATED),
            "tool_grants": count(EVENT_TOOL_GRANTED),
            "tool_calls": count(EVENT_TOOL_CALLED),
            "tool_denials": count(EVENT_TOOL_DENIED),
            "reviews": count(EVENT_REVIEW_AGENT_OUTPUT),
            "review_failures": review_failures,
            "human_escalations": count(EVENT_HUMAN_ESCALATION),
            "guidance_applied": count(EVENT_HUMAN_GUIDANCE_APPLIED),
            "result_reused": count(EVENT_RESULT_REUSED),
            "usage_calls": count(EVENT_TOKEN_USAGE),
        },
        findings=findings,
        review_summary=review_summary,
        final_result_sha256=hashlib.sha256(raw).hexdigest(),
        final_result_chars=len(final_result or ""),
        event_count=len(scoped),
        event_counts=dict(sorted(event_counts.items())),
        last_event_at=str(scoped[-1].get("timestamp") or "") if scoped else "",
    )

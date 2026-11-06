"""时间线与事件描述。"""

from __future__ import annotations

from so_agent.web.text import tr
from typing import Any

from so_agent.web.common import (
    WebTask,
    _GENERATED_TOOL_STEP_LABELS,
    _GENERATED_TOOL_STEP_LABELS_EN,
    _MAX_TIMELINE_EVENTS,
    _preview,
    _route_labels,
)


class TimelineMixin:

    @staticmethod
    def _build_agent_history(task: WebTask) -> list[dict[str, Any]]:
        if task.output is None:
            return []
        return [
            {
                "agent_id": record.agent_id,
                "agent_type": record.agent_type,
                "status": record.status,
                "attempt_count": record.attempt_count,
                "replacement_of": record.replacement_of,
                "created_at": record.created_at.isoformat(),
            }
            for record in task.output.agent_history[-50:]
        ]


    def _build_timeline(
        self, task: WebTask, lang: str = "zh"
    ) -> list[dict[str, Any]]:
        timeline: list[dict[str, Any]] = [
            {
                "type": "web_submitted",
                "timestamp": task.created_at,
                "level": "info",
                "message": tr(
                    lang,
                    "任务已提交到 Web 控制台",
                    "Task submitted to the web console",
                ),
                "subtask_id": None,
            }
        ]
        if task.classification is not None:
            label = _route_labels(lang).get(
                task.classification.task_type,
                task.classification.task_type.value,
            )
            reasoning = task.classification.reasoning or tr(
                lang, "无说明", "no details"
            )
            timeline.append(
                {
                    "type": "classification",
                    "timestamp": task.started_at or task.created_at,
                    "level": "info",
                    "message": tr(
                        lang,
                        f"分类结果：{label}（{reasoning}）",
                        f"Classification: {label} ({reasoning})",
                    ),
                    "subtask_id": None,
                }
            )
        events = task.context.event_log[-_MAX_TIMELINE_EVENTS:]
        for record in events:
            timeline.append(self._describe_event(record, lang))
        if task.finished_at and not any(
            item["type"] == "web_finished" for item in timeline
        ):
            level = "info"
            if task.status in {"failed", "human_required"}:
                level = "error"
            elif task.status == "cancelled":
                level = "warn"
            timeline.append(
                {
                    "type": "web_finished",
                    "timestamp": task.finished_at,
                    "level": level,
                    "message": tr(
                        lang,
                        f"任务结束：{task.status}"
                        + (f"（{task.error}）" if task.error else ""),
                        f"Task finished: {task.status}"
                        + (f" ({task.error})" if task.error else ""),
                    ),
                    "subtask_id": None,
                }
            )
        return timeline


    @staticmethod
    def _describe_event(
        record: dict[str, Any], lang: str = "zh"
    ) -> dict[str, Any]:
        event_type = str(record.get("event_type") or "")
        payload = record.get("payload") or {}
        level = "info"
        subtask_id = payload.get("subtask_id")
        if event_type == "status_transition":
            message = tr(
                lang,
                f"阶段迁移：{payload.get('from_status')} → "
                f"{payload.get('to_status')}",
                f"Stage transition: {payload.get('from_status')} → "
                f"{payload.get('to_status')}",
            )
            reason = str(payload.get("reason") or "").strip()
            if reason:
                message += tr(
                    lang,
                    f"（{_preview(reason, 160)}）",
                    f" ({_preview(reason, 160)})",
                )
            if payload.get("to_status") in {"failed", "human_required"}:
                level = "error"
            elif payload.get("to_status") == "cancelled":
                level = "warn"
        elif event_type == "task_dispatched":
            repair = (
                tr(lang, "（契约修复轮）", " (contract repair round)")
                if payload.get("contract_repair")
                else ""
            )
            message = tr(
                lang,
                f"派发子任务 {subtask_id}：第 {payload.get('attempt')}"
                f"/{payload.get('max_attempts')} 次{repair}",
                f"Dispatched subtask {subtask_id}: attempt "
                f"{payload.get('attempt')}/{payload.get('max_attempts')}{repair}",
            )
        elif event_type == "tool_called":
            action = str(payload.get("action") or "")
            tool_name = payload.get("tool_name") or tr(
                lang, "未知工具", "unknown tool"
            )
            if action == "invoke_started":
                message = tr(
                    lang,
                    f"调用工具 {tool_name}",
                    f"Calling tool {tool_name}",
                )
                if payload.get("agent"):
                    message += tr(
                        lang,
                        f"（{payload.get('agent')}）",
                        f" ({payload.get('agent')})",
                    )
                if payload.get("arguments_preview"):
                    message += tr(
                        lang,
                        f"：{_preview(str(payload['arguments_preview']), 120)}",
                        f": {_preview(str(payload['arguments_preview']), 120)}",
                    )
            elif action in {"invoke_completed", "invoke_failed"}:
                verdict = (
                    tr(lang, "失败", "failed")
                    if action == "invoke_failed"
                    else tr(lang, "完成", "completed")
                )
                message = tr(
                    lang,
                    f"工具 {tool_name} {verdict}",
                    f"Tool {tool_name} {verdict}",
                )
                if payload.get("duration") is not None:
                    message += tr(
                        lang,
                        f"（{payload.get('duration')}s）",
                        f" ({payload.get('duration')}s)",
                    )
                if action == "invoke_failed":
                    level = "warn"
                if payload.get("output_preview"):
                    message += tr(
                        lang,
                        f"：{_preview(str(payload['output_preview']), 100)}",
                        f": {_preview(str(payload['output_preview']), 100)}",
                    )
            else:
                labels = (
                    _GENERATED_TOOL_STEP_LABELS_EN
                    if lang == "en"
                    else _GENERATED_TOOL_STEP_LABELS
                )
                label = labels.get(action, action)
                message = tr(
                    lang,
                    f"动态工具 {tool_name} · {label}",
                    f"Dynamic tool {tool_name} · {label}",
                )
                if payload.get("reason"):
                    message += tr(
                        lang,
                        f"：{_preview(str(payload['reason']), 120)}",
                        f": {_preview(str(payload['reason']), 120)}",
                    )
                if action == "gap_unresolved":
                    level = "warn"
        elif event_type == "agent_created":
            nested = payload.get("agent") or {}
            agent_type = payload.get("agent_type") or (
                nested.get("agent_type") if isinstance(nested, dict) else ""
            )
            message = tr(
                lang,
                f"创建 Agent：{payload.get('agent_id')}",
                f"Created agent: {payload.get('agent_id')}",
            )
            if agent_type:
                message += tr(
                    lang, f"（{agent_type}）", f" ({agent_type})"
                )
            if payload.get("subtask_id"):
                message += tr(
                    lang,
                    f"，负责子任务 {payload.get('subtask_id')}",
                    f", assigned subtask {payload.get('subtask_id')}",
                )
        elif event_type == "agent_released":
            message = tr(
                lang,
                f"释放 Agent：{payload.get('agent_id')}"
                f"（{payload.get('status') or 'released'}）",
                f"Released agent: {payload.get('agent_id')}"
                f" ({payload.get('status') or 'released'})",
            )
            if payload.get("attempt_count"):
                message += tr(
                    lang,
                    f"，共尝试 {payload.get('attempt_count')} 次",
                    f", attempts: {payload.get('attempt_count')}",
                )
        elif event_type == "tool_granted":
            grant = payload.get("grant") or {}
            tools = grant.get("allowed_tools") or []
            message = tr(
                lang,
                f"签发工具授权：{payload.get('agent_id')}"
                f" → {', '.join(map(str, tools)) or '空'}",
                f"Tool grant issued: {payload.get('agent_id')}"
                f" → {', '.join(map(str, tools)) or 'none'}",
            )
        elif event_type == "tool_revoked":
            message = tr(
                lang,
                f"撤销工具授权：{payload.get('grant_id') or payload.get('agent_id')}",
                f"Tool grant revoked: {payload.get('grant_id') or payload.get('agent_id')}",
            )
        elif event_type == "replacement_requested":
            message = tr(
                lang,
                f"子任务 {subtask_id} 请求替换 Agent",
                f"Subtask {subtask_id} requested agent replacement",
            )
            level = "warn"
        elif event_type == "decomposition_issue":
            message = tr(
                lang,
                f"发现任务分解问题：{_preview(str(payload.get('recommendation') or ''), 160)}",
                f"Decomposition issue: {_preview(str(payload.get('recommendation') or ''), 160)}",
            )
            level = "warn"
        elif event_type == "replan":
            message = tr(
                lang,
                f"重规划：计划版本 v{payload.get('plan_version')}"
                f"（第 {payload.get('replan_count')} 次）",
                f"Replanned: plan version v{payload.get('plan_version')}"
                f" (replan #{payload.get('replan_count')})",
            )
            level = "warn"
        elif event_type == "result_reused":
            message = tr(
                lang,
                f"复用结果：{payload.get('source_subtask_id')} → "
                f"{payload.get('subtask_id')}（来源 v{payload.get('source_plan_version')}）",
                f"Reused result: {payload.get('source_subtask_id')} → "
                f"{payload.get('subtask_id')} (from v{payload.get('source_plan_version')})",
            )
        elif event_type == "reuse_rejected":
            message = tr(
                lang,
                f"拒绝复用声明：{payload.get('subtask_id')}（{payload.get('reason')}）",
                f"Reuse claim rejected: {payload.get('subtask_id')} ({payload.get('reason')})",
            )
            level = "warn"
        elif event_type == "failure_classified":
            message = tr(
                lang,
                f"失败分类：{subtask_id} → {payload.get('failure_type')}"
                f"（同指纹已出现 {payload.get('occurrences')} 次）",
                f"Failure classified: {subtask_id} → {payload.get('failure_type')}"
                f" (fingerprint seen {payload.get('occurrences')} times)",
            )
            level = "warn"
        elif event_type == "recovery_decided":
            message = tr(
                lang,
                f"恢复决策：{subtask_id} → {payload.get('action')}"
                f"（{_preview(str(payload.get('reason') or ''), 120)}）",
                f"Recovery decision: {subtask_id} → {payload.get('action')}"
                f" ({_preview(str(payload.get('reason') or ''), 120)})",
            )
        elif event_type == "retry_suppressed":
            message = tr(
                lang,
                f"抑制重试：{subtask_id}（同因失败达到阈值）",
                f"Retry suppressed: {subtask_id} (same failure reached threshold)",
            )
            level = "warn"
        elif event_type == "review_agent_output":
            verdict = (
                tr(lang, "通过", "passed")
                if payload.get("passed")
                else tr(lang, "未通过", "failed")
            )
            message = tr(
                lang,
                f"评审结论（{payload.get('stage')}）：{verdict}",
                f"Review result ({payload.get('stage')}): {verdict}",
            )
            if not payload.get("contract_valid", True):
                message += tr(
                    lang,
                    "（契约无效，已按未通过处理）",
                    " (invalid contract, treated as failed)",
                )
                level = "warn"
        elif event_type == "review_parse_error":
            message = tr(
                lang,
                f"评审解析失败（{payload.get('stage')}），保守判定未通过",
                f"Review parse failed ({payload.get('stage')}); conservatively marked failed",
            )
            level = "warn"
        elif event_type == "review_contract_error":
            message = tr(
                lang,
                f"评审契约错误（{payload.get('stage')}），按未通过处理",
                f"Review contract error ({payload.get('stage')}); treated as failed",
            )
            level = "warn"
        elif event_type == "review_contract_repair":
            message = tr(
                lang,
                f"评审契约修复（{payload.get('stage')}）："
                f"{payload.get('final_status')}",
                f"Review contract repaired ({payload.get('stage')}): "
                f"{payload.get('final_status')}",
            )
        elif event_type == "timeout_audit":
            message = tr(
                lang,
                f"超时审计：{payload.get('agent') or payload.get('subtask_id') or ''}"
                f" 阶段 {payload.get('phase')}，已耗时 {payload.get('elapsed')}",
                f"Timeout audit: {payload.get('agent') or payload.get('subtask_id') or ''}"
                f" phase {payload.get('phase')}, elapsed {payload.get('elapsed')}",
            )
            level = "warn"
        elif event_type == "human_escalation":
            message = tr(
                lang,
                f"升级人工处理：{_preview(str(payload.get('reason') or ''), 160)}",
                f"Escalated to human: {_preview(str(payload.get('reason') or ''), 160)}",
            )
            level = "error"
        elif event_type == "token_usage":
            message = tr(
                lang,
                f"模型用量：{payload.get('agent')}，"
                f"{payload.get('total_tokens')} tokens",
                f"Model usage: {payload.get('agent')}, "
                f"{payload.get('total_tokens')} tokens",
            )
        elif event_type == "parallelism_decided":
            message = tr(
                lang,
                f"并行度决策：{payload.get('mode')}，"
                f"并发 {payload.get('concurrency')}（{payload.get('reason')}）",
                f"Parallelism decision: {payload.get('mode')}, "
                f"concurrency {payload.get('concurrency')} ({payload.get('reason')})",
            )
        elif event_type == "runtime_breaker_tripped":
            message = tr(
                lang,
                f"运行时熔断：{payload.get('trip_reason')}，"
                f"剩余子任务收敛为串行",
                f"Runtime breaker tripped: {payload.get('trip_reason')}; "
                f"remaining subtasks run serially",
            )
            level = "warn"
        elif event_type == "review_skipped":
            message = tr(
                lang,
                f"评审跳过（{payload.get('stage')}）：{payload.get('reason')}",
                f"Review skipped ({payload.get('stage')}): {payload.get('reason')}",
            )
        elif event_type == "task_created":
            message = tr(
                lang, "任务已进入工作流", "Task entered the workflow"
            )
        else:
            message = event_type
        return {
            "type": event_type,
            "timestamp": record.get("timestamp"),
            "level": level,
            "message": message,
            "subtask_id": subtask_id,
        }


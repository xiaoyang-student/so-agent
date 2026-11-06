"""人工兜底、人工指导恢复与流水线降级（WorkflowEngine 混入）。"""

from __future__ import annotations

import json

from so_agent.models import (
    ExecutionResult,
    HumanEscalationRequest,
    HumanGuidance,
    SubtaskProgress,
    TaskStatus,
)
from so_agent.orchestrator import OrchestratorOutput
from so_agent.runtime.events import (
    EVENT_AGENT_CREATED,
    EVENT_AGENT_RELEASED,
    EVENT_DECOMPOSITION_ISSUE,
    EVENT_FAILURE_CLASSIFIED,
    EVENT_HUMAN_ESCALATION,
    EVENT_HUMAN_ESCALATION_NOTIFIED,
    EVENT_HUMAN_ESCALATION_NOTIFY_FAILED,
    EVENT_HUMAN_GUIDANCE_APPLIED,
    EVENT_RECOVERY_DECIDED,
    EVENT_REPLACEMENT_REQUESTED,
    EVENT_RETRY_SUPPRESSED,
    EVENT_TASK_DISPATCHED,
    EVENT_TOOL_GRANTED,
    EVENT_TOOL_REVOKED,
)
from so_agent.runtime.logging_setup import get_logger
from so_agent.runtime.registry import DEFAULT_SUPERVISOR_ID
from so_agent.state_machine import HumanGuidanceError

logger = get_logger("workflow.escalation")

# 人工兜底材料中单条产出 / 错误摘要的最大字符数（避免材料包过大）
_GUIDANCE_PREVIEW_CHARS: int = 300

# 计入"工具调用证据"的事件类型（人工升级材料组装用）
_EVIDENCE_EVENT_TYPES: frozenset[str] = frozenset(
    {
        EVENT_TASK_DISPATCHED,
        EVENT_TOOL_GRANTED,
        EVENT_TOOL_REVOKED,
        EVENT_AGENT_CREATED,
        EVENT_AGENT_RELEASED,
        EVENT_REPLACEMENT_REQUESTED,
        EVENT_DECOMPOSITION_ISSUE,
        EVENT_FAILURE_CLASSIFIED,
        EVENT_RECOVERY_DECIDED,
        EVENT_RETRY_SUPPRESSED,
    }
)


class EscalationMixin:
    async def _escalate_on_pipeline_stall(
        self, reason: str, window: float
    ) -> OrchestratorOutput:
        """无进展超时：携带已完成部分结果转人工兜底并保留检查点。"""
        self._lifecycle_exit_reason = "no_progress_timeout"
        completed = self._authoritative_success_results()
        total = max(len(self._subtasks), len(completed))
        snapshot = self._gate.progress_snapshot()
        self._persist_checkpoint(reason)
        message = (
            f"流水线连续 {window:g} 秒无进展（最后进展："
            f"{snapshot.get('label') or '无'}，已完成 "
            f"{len(completed)}/{total} 个子任务）；任务未被固定总时长判定"
            "失败，可从检查点续跑未完成子任务"
        )
        output = await self._escalate_to_human(message)
        if completed:
            partial = self._fallback_summary(completed)
            output = output.model_copy(
                update={
                    "final_result": (
                        f"{output.final_result}\n已完成的部分结果：\n{partial}"
                    )
                }
            )
        return output

    def _authoritative_success_results(self) -> list[ExecutionResult]:
        """权威成功结果：跨版本归档与当前/调度器结果的合并视图（深拷贝）。

        任务 45：pipeline 超时的完成计数依据。重规划采纳会清空
        ``self._results``，仅依赖当前计划的成功结果会低估真实完成量；
        合并顺序：``archive_results`` 按 (plan_version, subtask_id) 升序
        遍历（新版本覆盖旧版本）→ 当前 ``self._results`` 覆盖 → 调度器
        清理期落账的部分结果覆盖（不依赖被重规划修改的临时 pending
        集合），最后仅保留 success 项（部分结果同样准确）。
        """
        merged: dict[str, ExecutionResult] = {}
        archive = self._archive_results
        for version, subtask_id in sorted(
            archive.keys(), key=lambda key: (key[0], key[1])
        ):
            merged[subtask_id] = archive[(version, subtask_id)]

        def _merge(incoming: dict[str, ExecutionResult]) -> None:
            # 成功结果总是覆盖（最新成功优先）；失败结果不抹掉已有的成功
            # 记录：归档成功是已持久化的事实，当前版本重试 / 重规划失败
            # 不得低估真实完成量（任务 45：部分结果必须准确）
            for sid, result in incoming.items():
                existing = merged.get(sid)
                if result.success or existing is None or not existing.success:
                    merged[sid] = result

        _merge(self._results)
        scheduler = self._scheduler
        if scheduler is not None:
            _merge(getattr(scheduler, "results", {}) or {})
        return [
            item.model_copy(deep=True)
            for item in merged.values()
            if item.success
        ]

    async def _escalate_on_pipeline_timeout(
        self, timeout: float
    ) -> OrchestratorOutput:
        """流水线整体超时：携带已完成的部分结果优雅降级（转人工兑底）。

        与直接失败不同，此处保留权威成功结果（归档 + 当前 + 调度器
        落账，任务 45）的子任务产出，以本地确定性摘要并入 final_result，
        避免超时把已完成的工作一并丢弃；同时走标准人工兑底链路
        （状态迁移 + 升级材料包留痕）。
        """
        self._lifecycle_exit_reason = "pipeline_timeout"
        completed = self._authoritative_success_results()
        total = max(len(self._subtasks), len(completed))
        snapshot = self._gate.progress_snapshot()
        reason = (
            f"流水线绝对兜底超时（上限 {timeout:.0f} 秒）："
            f"已完成 {len(completed)}/{total} 个子任务；"
            f"最后进展：{snapshot.get('label') or '无'}，"
            f"已 {snapshot.get('seconds_since_progress', 0):.1f} 秒无进展"
        )
        output = await self._escalate_to_human(reason)
        if completed:
            partial = self._fallback_summary(completed)
            output = output.model_copy(
                update={
                    "final_result": (
                        f"{output.final_result}\n已完成的部分结果：\n{partial}"
                    )
                }
            )
        return output

    async def _escalate_to_human(self, reason: str) -> OrchestratorOutput:
        """人工兜底：组装升级材料包、交接处理器并把任务置为 human_required。

        首版 HumanEscalationHandler 仅记录请求，不实际通知人工；材料包
        含原始任务、历代计划、失败历史、工具调用证据与最终评审结论。

        Args:
            reason: 触发人工介入的原因说明。

        Returns:
            OrchestratorOutput：human_required 终态的结构化输出。
        """
        self._transition(TaskStatus.HUMAN_REQUIRED, reason=reason)
        request = HumanEscalationRequest(
            task_id=self._task_request.task_id,
            original_request=self._task_request.model_copy(deep=True),
            plan_history=self.plan_history,
            agent_failure_history=list(self._agent_failure_history),
            tool_call_evidence=self._collect_tool_evidence(),
            final_review=(
                self._last_review.model_copy(deep=True)
                if self._last_review is not None
                else None
            ),
            reason=reason,
            progress=self._build_subtask_progress(),
            rollback_targets=[spec.subtask_id for spec in self._subtasks],
        )
        self._human_escalation = request
        self._events.log_event(
            EVENT_HUMAN_ESCALATION,
            {
                "task_id": self._task_request.task_id,
                "agent_id": DEFAULT_SUPERVISOR_ID,
                "reason": reason,
                "request": request.model_dump(mode="json"),
            },
        )
        self._persist_checkpoint(reason)
        await self._escalation_handler.escalate(request)
        notification = getattr(
            self._escalation_handler, "last_notification", None
        )
        if notification is not None and not getattr(
            notification, "skipped", False
        ):
            self._events.log_event(
                (
                    EVENT_HUMAN_ESCALATION_NOTIFIED
                    if getattr(notification, "sent", False)
                    else EVENT_HUMAN_ESCALATION_NOTIFY_FAILED
                ),
                {
                    "task_id": self._task_request.task_id,
                    "agent_id": DEFAULT_SUPERVISOR_ID,
                    "status_code": getattr(notification, "status_code", None),
                    "error": getattr(notification, "error", ""),
                },
            )
        # 已完成成果不随升级一起丢弃：人工介入也应先拿到可用的部分结果
        partial = self._successful_results()
        final_text = f"任务需要人工介入：{reason}"
        if partial:
            final_text += "\n\n" + self._fallback_summary(
                partial, reason="部分子任务已完成，其余需要人工处理"
            )
        return self._build_output(
            TaskStatus.HUMAN_REQUIRED,
            final_result=final_text,
            review=self._last_review,
        )

    def _build_subtask_progress(self) -> list[SubtaskProgress]:
        """生成人工兜底材料中的子任务进展摘要（不含完整产出）。"""
        items: list[SubtaskProgress] = []
        for spec in self._subtasks:
            result = self._results.get(spec.subtask_id)
            if result is None:
                status = "pending"
                preview = ""
                error = ""
            else:
                status = "completed" if result.success else "failed"
                preview = (result.output or "").strip().replace("\n", " ")
                if len(preview) > _GUIDANCE_PREVIEW_CHARS:
                    preview = preview[:_GUIDANCE_PREVIEW_CHARS] + "…"
                error = (result.error or "").strip()
                if len(error) > _GUIDANCE_PREVIEW_CHARS:
                    error = error[:_GUIDANCE_PREVIEW_CHARS] + "…"
            items.append(
                SubtaskProgress(
                    subtask_id=spec.subtask_id,
                    title=spec.title,
                    status=status,
                    output_preview=preview,
                    error=error,
                )
            )
        return items

    async def resume_with_guidance(self, guidance: HumanGuidance) -> OrchestratorOutput:
        """应用人工指导并从指定子任务继续执行。

        语义（确定性，不接受自由文本推断）：

        - ``abort``：不执行任何子任务，任务保持 ``human_required``；
        - ``rollback`` / ``resume``：回退目标取 ``rollback_to``；为空时
          自动选择"失败 + 未完成"的子任务（按计划顺序），已成功子任务
          不重跑；全部已完成时直接进入汇总/结果评审（用于超时后续跑）；
          回退目标的下游依赖一并失效并重跑；未成功的上游依赖也纳入
          本批次，保证调度 DAG 可执行；
        - 指导文本追加到回退目标子任务的指令中，作为补充信息参与重跑。

        Raises:
            HumanGuidanceError: 任务不在人工兜底态、当前无计划、回退目标
                不在当前计划中，或没有可回退的子任务。
        """
        if not isinstance(guidance, HumanGuidance):
            raise HumanGuidanceError("guidance 必须是 HumanGuidance")
        status = self._state.current_status
        human_mode = (
            self._human_escalation is not None
            and status is TaskStatus.HUMAN_REQUIRED
        )
        # 检查点恢复允许续跑非终态/已取消任务：取消或进程中断不会
        # 生成人工兜底请求，但已完成子任务与未完成项仍可安全分离。
        recovery_mode = (
            self._restored_from_checkpoint
            and status not in {TaskStatus.COMPLETED, TaskStatus.FAILED}
        )
        if not (human_mode or recovery_mode):
            raise HumanGuidanceError("任务不在人工兜底状态，无法接受指导")
        if guidance.action == "abort":
            self._events.log_event(
                EVENT_HUMAN_GUIDANCE_APPLIED,
                {
                    "task_id": self._task_request.task_id,
                    "agent_id": DEFAULT_SUPERVISOR_ID,
                    "action": "abort",
                },
            )
            return self._build_output(
                TaskStatus.HUMAN_REQUIRED,
                final_result="人工已终止恢复：任务保持人工兜底状态",
                review=self._last_review,
            )
        plan_specs = list(self._subtasks)
        if not plan_specs:
            raise HumanGuidanceError(
                "当前任务没有可恢复的子任务计划，请重新提交任务或补充信息"
            )
        plan_ids = {spec.subtask_id for spec in plan_specs}
        requested = list(dict.fromkeys(guidance.rollback_to))
        unknown = [sid for sid in requested if sid not in plan_ids]
        if unknown:
            raise HumanGuidanceError(
                "回退目标不在当前计划中：" + "、".join(unknown)
            )
        # 默认续跑范围 = 失败项 + 从未完成的项（按计划顺序）。
        # 已成功子任务绝不重跑：超时/取消后的检查点恢复只补未完成
        # 工作，避免把已完成成果整体重做一遍（成本与完成率的关键）。
        succeeded = {
            sid
            for sid, result in self._results.items()
            if sid in plan_ids and result.success
        }
        if requested:
            targets = requested
        else:
            targets = [
                spec.subtask_id
                for spec in plan_specs
                if spec.subtask_id not in succeeded
            ]
        if not targets and guidance.action != "resume":
            raise HumanGuidanceError(
                "没有可回退的失败子任务；请指定 rollback_to 或补充信息后重试"
            )

        affected = self._guidance_affected_set(targets)
        note = self._format_guidance_note(guidance)
        self._patch_subtask_specs(set(targets), note)
        for subtask_id in affected:
            self._results.pop(subtask_id, None)
        self._pending_specs = [
            spec.model_copy(deep=True)
            for spec in self._subtasks
            if spec.subtask_id in affected
        ]

        self._human_escalation = None
        self._gate.reset()
        self._events.log_event(
            EVENT_HUMAN_GUIDANCE_APPLIED,
            {
                "task_id": self._task_request.task_id,
                "agent_id": DEFAULT_SUPERVISOR_ID,
                "action": guidance.action,
                "rollback_to": sorted(affected),
                "has_instructions": bool(guidance.instructions.strip()),
                "has_additional_info": bool(guidance.additional_info.strip()),
            },
        )
        self._transition(
            TaskStatus.EXECUTING,
            reason=f"应用人工指导恢复执行（{guidance.action}）",
        )
        output = await self._run_result_loop(is_simple=self._is_simple_plan())
        self._start_provisional_review_if_pending(output)
        return output

    def _guidance_affected_set(self, targets: list[str]) -> set[str]:
        """计算回退影响集：目标 + 下游依赖 + 未成功的上游依赖。"""
        plan = {spec.subtask_id: spec for spec in self._subtasks}
        affected = {sid for sid in targets if sid in plan}
        changed = True
        while changed:
            changed = False
            for sid, spec in plan.items():
                if sid in affected:
                    continue
                if any(dep in affected for dep in spec.dependencies):
                    affected.add(sid)
                    changed = True
        changed = True
        while changed:
            changed = False
            for sid in list(affected):
                for dep in plan[sid].dependencies:
                    if dep in plan and dep not in affected:
                        result = self._results.get(dep)
                        if result is None or not result.success:
                            affected.add(dep)
                            changed = True
        return affected

    @staticmethod
    def _format_guidance_note(guidance: HumanGuidance) -> str:
        """把人工指导格式化为追加到子任务指令的确定性文本块。"""
        parts: list[str] = []
        if guidance.instructions.strip():
            parts.append("人工指导：" + guidance.instructions.strip())
        if guidance.additional_info.strip():
            parts.append("人工补充信息：" + guidance.additional_info.strip())
        return "\n".join(parts)

    def _patch_subtask_specs(self, target_ids: set[str], note: str) -> None:
        """把指导文本追加到回退目标的指令，并同步当前计划版本快照。"""
        if not note:
            return
        patched = [
            spec.model_copy(
                update={"instructions": f"{spec.instructions}\n\n{note}"}
            )
            if spec.subtask_id in target_ids
            else spec
            for spec in self._subtasks
        ]
        self._subtasks = patched
        if self._plan_history:
            self._plan_history[-1] = [
                spec.model_copy(deep=True) for spec in patched
            ]

    def _collect_tool_evidence(self) -> list[str]:
        """从事件日志提取工具调用 / Agent 生命周期证据（JSON 文本行）。

        仅保留本任务相关事件（无 task_id 的事件一并保留），供人工升级
        材料组装与审计复核。
        """
        task_id = self._task_request.task_id if self._task_request else None
        evidence: list[str] = []
        for record in list(self._events.events):
            if not isinstance(record, dict):
                continue
            if record.get("event_type") not in _EVIDENCE_EVENT_TYPES:
                continue
            record_task = record.get("task_id")
            if task_id is not None and record_task not in (None, task_id):
                continue
            evidence.append(json.dumps(record, ensure_ascii=False, default=str))
        return evidence

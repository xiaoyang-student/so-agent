"""检查点导出、恢复与落盘（WorkflowEngine 混入）。"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from so_agent.models import (
    ExecutionResult,
    HumanEscalationRequest,
    ReviewDecision,
    SubtaskSpec,
    TaskRequest,
    TaskStatus,
    ToolGrant,
)
from so_agent.runtime.checkpoints import CheckpointError, save_checkpoint
from so_agent.runtime.events import (
    EVENT_CHECKPOINT_RESUMED,
    EVENT_CHECKPOINT_SAVED,
)
from so_agent.runtime.logging_setup import get_logger
from so_agent.runtime.registry import DEFAULT_SUPERVISOR_ID
from so_agent.state_machine import WorkflowError

logger = get_logger("workflow.checkpoint")


class CheckpointMixin:
    def export_checkpoint(self) -> dict[str, Any]:
        """导出当前工作流状态（JSON 兼容）；无任务请求时抛错。

        结果视图合并调度器已定型结果：子任务级检查点可能在批次尚未
        整体收尾时落盘，只读 ``self._results`` 会漏掉刚完成的兄弟
        子任务，导致续跑重复执行。
        """
        request = self._task_request
        if request is None:
            raise WorkflowError("当前没有可导出的任务请求")
        merged_results: dict[str, ExecutionResult] = {}
        scheduler = self._scheduler
        if scheduler is not None:
            merged_results.update(getattr(scheduler, "results", {}) or {})
        merged_results.update(self._results)
        return {
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "task_id": request.task_id,
            "status": self._state.current_status.value,
            "stop_reason": self._gate.stop_reason or None,
            "progress": self._gate.progress_snapshot(),
            "task_request": request.model_dump(mode="json"),
            "plan_version": self._plan_version,
            "replan_count": self._replan_count,
            "result_rework_count": self._result_rework_count,
            "plan_steps_merged": self._state.plan_steps_merged,
            "merged_subtask_ids": sorted(self._state.merged_subtask_ids),
            "subtasks": [spec.model_dump(mode="json") for spec in self._subtasks],
            "pending_specs": [
                spec.model_dump(mode="json") for spec in self._pending_specs
            ],
            "results": {
                key: result.model_dump(mode="json")
                for key, result in merged_results.items()
            },
            "plan_history": [
                [spec.model_dump(mode="json") for spec in plan]
                for plan in self._plan_history
            ],
            "agent_failure_history": list(self._agent_failure_history),
            "grants": [grant.model_dump(mode="json") for grant in self._grants],
            "plan_grants": [
                grant.model_dump(mode="json") for grant in self._plan_grants
            ],
            "plan_grant_map": {
                key: grant.model_dump(mode="json")
                for key, grant in self._plan_grant_map.items()
            },
            "plan_grant_proposals": [
                grant.model_dump(mode="json")
                for grant in self._plan_grant_proposals
            ],
            "last_review": (
                self._last_review.model_dump(mode="json")
                if self._last_review is not None
                else None
            ),
            "human_escalation": (
                self._human_escalation.model_dump(mode="json")
                if self._human_escalation is not None
                else None
            ),
            "archive_results": [
                {
                    "plan_version": version,
                    "subtask_id": subtask_id,
                    "result": result.model_dump(mode="json"),
                }
                for (version, subtask_id), result in self._archive_results.items()
            ],
            "seen_replacement_requests": sorted(
                self._seen_replacement_requests
            ),
            "seen_issue_fingerprints": [
                [role, index, list(fingerprints)]
                for role, index, fingerprints in self._seen_issue_fingerprints
            ],
        }

    def restore_checkpoint(self, payload: dict[str, Any]) -> None:
        """从检查点恢复状态（用于 CLI 断点续跑）。"""
        if not isinstance(payload, dict):
            raise WorkflowError("检查点必须是对象")
        try:
            request = TaskRequest.model_validate(payload.get("task_request"))
            status = TaskStatus(str(payload.get("status") or ""))
        except Exception as exc:
            raise WorkflowError(f"检查点任务信息非法：{exc}") from exc
        self._task_request = request
        self._state.current_status = status
        self._plan_version = int(payload.get("plan_version") or 0)
        self._replan_count = int(payload.get("replan_count") or 0)
        self._result_rework_count = int(
            payload.get("result_rework_count") or 0
        )
        self._state.plan_steps_merged = int(payload.get("plan_steps_merged") or 0)
        self._state.merged_subtask_ids = {
            str(item) for item in payload.get("merged_subtask_ids") or []
        }
        self._subtasks = [
            SubtaskSpec.model_validate(item)
            for item in payload.get("subtasks") or []
        ]
        self._pending_specs = [
            SubtaskSpec.model_validate(item)
            for item in payload.get("pending_specs") or []
        ]
        self._results = {
            str(key): ExecutionResult.model_validate(item)
            for key, item in (payload.get("results") or {}).items()
        }
        self._plan_history = [
            [SubtaskSpec.model_validate(spec) for spec in plan]
            for plan in payload.get("plan_history") or []
        ]
        self._agent_failure_history = [
            str(item) for item in payload.get("agent_failure_history") or []
        ]
        self._grants = [
            ToolGrant.model_validate(item) for item in payload.get("grants") or []
        ]
        self._plan_grants = [
            ToolGrant.model_validate(item)
            for item in payload.get("plan_grants") or []
        ]
        self._plan_grant_map = {
            str(key): ToolGrant.model_validate(item)
            for key, item in (payload.get("plan_grant_map") or {}).items()
        }
        self._plan_grant_proposals = [
            ToolGrant.model_validate(item)
            for item in payload.get("plan_grant_proposals") or []
        ]
        review = payload.get("last_review")
        self._last_review = (
            ReviewDecision.model_validate(review) if review else None
        )
        escalation = payload.get("human_escalation")
        self._human_escalation = (
            HumanEscalationRequest.model_validate(escalation)
            if escalation
            else None
        )
        self._archive_results = {
            (int(item["plan_version"]), str(item["subtask_id"])): (
                ExecutionResult.model_validate(item["result"])
            )
            for item in payload.get("archive_results") or []
        }
        self._seen_replacement_requests = {
            str(item) for item in payload.get("seen_replacement_requests") or []
        }
        self._seen_issue_fingerprints = {
            (str(role), int(index), tuple(str(x) for x in fingerprints))
            for role, index, fingerprints in (
                payload.get("seen_issue_fingerprints") or []
            )
        }
        self._gate.reset()
        self._state._advance_phase(status)
        self._restored_from_checkpoint = True
        self._events.log_event(
            EVENT_CHECKPOINT_RESUMED,
            {
                "task_id": request.task_id,
                "agent_id": DEFAULT_SUPERVISOR_ID,
                "status": status.value,
                "plan_version": self._plan_version,
                "subtasks": len(self._subtasks),
            },
        )

    def _persist_checkpoint(self, reason: str) -> None:
        """保存人工兜底检查点；失败只记录，不影响升级主流程。"""
        request = self._task_request
        if request is None:
            return
        try:
            path = save_checkpoint(
                self._context.sandbox_dir,
                request.task_id,
                self.export_checkpoint(),
            )
        except (CheckpointError, OSError, ValueError) as exc:
            logger.warning("保存检查点失败：%s", exc)
            return
        self._events.log_event(
            EVENT_CHECKPOINT_SAVED,
            {
                "task_id": request.task_id,
                "agent_id": DEFAULT_SUPERVISOR_ID,
                "reason": reason,
                "file": path.name,
            },
        )

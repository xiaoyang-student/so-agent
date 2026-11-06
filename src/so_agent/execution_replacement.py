"""失败升级与替代生命周期（ExecutionCoordinator 混入）。

失败结算、替代状态链通知接收（requested → executing → completed /
failed）、分解问题上报与重规划申请，以及失败证据合并。
"""

from __future__ import annotations

from typing import Any

from so_agent.models import (
    ExecutionResult,
    ReplacementRequest,
    TaskDecompositionIssue,
    TaskStatus,
)
from so_agent.runtime.events import (
    EVENT_REPLACEMENT_COMPLETED,
    EVENT_REPLACEMENT_EXECUTING,
    EVENT_REPLACEMENT_FAILED,
    EVENT_REPLACEMENT_REQUESTED,
)
from so_agent.runtime.registry import DEFAULT_SUPERVISOR_ID


class ReplacementMixin:
    def _settle_failure_states(self, batch: dict[str, ExecutionResult]) -> None:
        """结算批次中尚未结算的分解问题与失败历史（任务 34 后职责收敛）。

        替代状态链（replacement_requested → replacement_executing →
        executing / failed）已由调度器在真实生命周期时点经
        ``handle_replacement_lifecycle`` 实时落账：本方法不再事后重建
        该状态链，也不重放累积的替换申请列表；此处仅处理尚未结算的
        分解问题——追加到 ``_agent_failure_history``（供重规划参考与
        人工升级材料组装）。
        """
        subtask_ids = set(batch)
        for issue in self._scheduler.decomposition_issues:
            if issue.subtask_id not in subtask_ids:
                continue
            fingerprint = (
                issue.subtask_id,
                issue.plan_version,
                tuple(issue.failure_reasons),
            )
            if fingerprint in self._seen_issue_fingerprints:
                continue
            self._seen_issue_fingerprints.add(fingerprint)
            self._agent_failure_history.append(self._format_issue(issue))

    async def handle_replacement_lifecycle(
        self, event: str, payload: dict[str, Any]
    ) -> None:
        """调度器替代流程真实时点通知的受控接收端。

        编排器把本方法注入 SubtaskDAGScheduler（``set_lifecycle_notifier``）；
        调度器在替代请求创建后、替代 Agent 创建成功后、替代执行返回后
        （含创建失败/取消/尝试耗尽的终止路径）回调本方法。全局 FSM 迁移
        与相关事件写入经 ``_replacement_lifecycle_lock`` 串行化：并发
        子任务的替代流程同时回调时迁移逐一生效，时间线与实际执行一致。

        载荷仅含控制信息（task_id/subtask_id/request_id/agent_id/
        attempt_count/failure_category/非敏感 summary），不含业务产出；
        替换申请书仍经主管路由上报，星型通信不变。

        Args:
            event: 生命周期事件名（replacement_requested /
                replacement_executing / replacement_completed /
                replacement_failed）；未知事件名忽略（防御性）。
            payload: 控制信息载荷（由调度器组装）。
        """
        handlers = {
            EVENT_REPLACEMENT_REQUESTED: self._handle_lifecycle_requested,
            EVENT_REPLACEMENT_EXECUTING: self._handle_lifecycle_executing,
            EVENT_REPLACEMENT_COMPLETED: self._handle_lifecycle_completed,
            EVENT_REPLACEMENT_FAILED: self._handle_lifecycle_failed,
        }
        handler = handlers.get(event)
        if handler is None:
            return
        async with self._replacement_lifecycle_lock:
            await handler(payload)

    def _find_replacement_request(
        self, request_id: str
    ) -> ReplacementRequest | None:
        """按 request_id 从调度器累积列表反查完整替换申请。

        申请书对象由调度器持有；生命周期通知只携带控制信息，主管路由
        上报（留痕）所需的完整申请书在此按 request_id 检索复用。
        """
        for item in reversed(self._scheduler.replacement_requests):
            if item.request_id == request_id:
                return item
        return None

    async def _handle_lifecycle_requested(self, payload: dict[str, Any]) -> None:
        """requested：替代请求已创建、替代 Agent 尚未创建。

        对外效果与旧 ``_handle_replacement`` 一致：REPLACEMENT_REQUESTED
        迁移 + 申请书经主管路由上报（留痕）+ 失败历史追加；按 request_id
        幂等去重，同一请求不重复结算（防旧累积列表重放）。
        """
        request_id = str(payload.get("request_id") or "")
        subtask_id = str(payload.get("subtask_id") or "")
        if not request_id or request_id in self._seen_replacement_requests:
            return
        self._seen_replacement_requests.add(request_id)
        request = self._find_replacement_request(request_id)
        attempt_count = payload.get("attempt_count")
        if isinstance(attempt_count, int) and attempt_count > 0:
            requested_reason = (
                f"子 Agent 尝试耗尽（{attempt_count} 次），发起替换申请"
            )
        else:
            requested_reason = "子 Agent 尝试耗尽，发起替换申请"
        self._transition(
            TaskStatus.REPLACEMENT_REQUESTED,
            reason=requested_reason,
            subtask_id=subtask_id,
            request_id=request_id,
        )
        requester_id = (
            str(payload.get("requester_agent_id") or "")
            or (request.requester_agent_id if request is not None else "")
            or DEFAULT_SUPERVISOR_ID
        )
        self._route_message(
            requester_id,
            {
                "message_type": "replacement_request",
                "task_id": self._task_request.task_id,
                "subtask_id": subtask_id,
                "request": (
                    request.model_dump(mode="json")
                    if request is not None
                    else {"request_id": request_id, "subtask_id": subtask_id}
                ),
                "verdict": "approved",
                "decided_by": DEFAULT_SUPERVISOR_ID,
            },
        )
        self._agent_failure_history.append(
            f"替换申请 {request_id}（子任务 {subtask_id}）："
            + (
                request.failure_summary
                if request is not None
                else "失败摘要未随通知携带"
            )
        )

    async def _handle_lifecycle_executing(self, payload: dict[str, Any]) -> None:
        """executing：替代 Agent 创建成功、尚未实际尝试。"""
        subtask_id = str(payload.get("subtask_id") or "")
        request_id = str(payload.get("request_id") or "")
        self._transition(
            TaskStatus.REPLACEMENT_EXECUTING,
            reason="主管核验失败证据后批准替换，替代 Agent 开始执行",
            subtask_id=subtask_id,
            request_id=request_id,
        )
        self._events.log_event(
            EVENT_REPLACEMENT_EXECUTING,
            {
                "task_id": payload.get("task_id"),
                "subtask_id": subtask_id,
                "request_id": request_id,
                "agent_id": payload.get("agent_id"),
            },
        )

    async def _handle_lifecycle_completed(self, payload: dict[str, Any]) -> None:
        """completed：替代执行成功返回，全局状态回到 executing。"""
        subtask_id = str(payload.get("subtask_id") or "")
        request_id = str(payload.get("request_id") or "")
        self._transition(
            TaskStatus.EXECUTING,
            reason="替代 Agent 执行成功，恢复常规执行",
            subtask_id=subtask_id,
            request_id=request_id,
        )
        self._events.log_event(
            EVENT_REPLACEMENT_COMPLETED,
            {
                "task_id": payload.get("task_id"),
                "subtask_id": subtask_id,
                "request_id": request_id,
                "agent_id": payload.get("agent_id"),
                "attempt_count": payload.get("attempt_count"),
                "summary": payload.get("summary"),
            },
        )

    async def _handle_lifecycle_failed(self, payload: dict[str, Any]) -> None:
        """failed：替代执行终止（尝试耗尽/取消/创建失败）。

        不迁移全局状态：保持既有失败恢复语义——分解问题随后由
        ``_settle_failure_states`` 结算并触发 decomposition_issue 状态
        链；终止通知仅留痕，不增加替代次数、不触发二级替代。
        """
        self._events.log_event(
            EVENT_REPLACEMENT_FAILED,
            {
                "task_id": payload.get("task_id"),
                "subtask_id": payload.get("subtask_id"),
                "request_id": payload.get("request_id"),
                "agent_id": payload.get("agent_id"),
                "attempt_count": payload.get("attempt_count"),
                "failure_category": payload.get("failure_category"),
                "summary": payload.get("summary"),
            },
        )

    async def _handle_decomposition_issue(
        self, issue: TaskDecompositionIssue
    ) -> bool:
        """处理任务分解问题：上报主管并申请重规划（消耗拆分修改额度）。

        状态链：``decomposition_issue → replanning``；额度不足时返回
        False（由调用方转入人工兜底）。

        Args:
            issue: 本次失败升级的诊断问题（含失败 Agent 与原因证据）。

        Returns:
            True 表示已获准重规划（额度已消耗）；False 表示额度耗尽。
        """
        self._transition(
            TaskStatus.DECOMPOSITION_ISSUE,
            reason="子任务执行持续失败，判定任务分解存在问题",
            subtask_id=issue.subtask_id,
            failed_agent_ids=list(issue.failed_agent_ids),
            recommendation=issue.recommendation,
        )
        sender: str | None = None
        for agent_id in issue.failed_agent_ids:
            if self._registry.get(agent_id) is not None:
                sender = agent_id
                break
        if sender is None:
            latest = self._find_latest_agent(issue.subtask_id)
            sender = latest.agent_id if latest is not None else None
        if sender is not None:
            self._route_message(
                sender,
                {
                    "message_type": "task_decomposition_issue",
                    "task_id": self._task_request.task_id,
                    "subtask_id": issue.subtask_id,
                    "issue": issue.model_dump(mode="json"),
                },
            )
        if not self._try_consume_replan_quota():
            return False
        self._transition(
            TaskStatus.REPLANNING,
            reason="主管受理任务分解问题，重新拆解任务",
            subtask_id=issue.subtask_id,
        )
        return True

    def _collect_failure_issue(
        self, failed_ids: list[str]
    ) -> TaskDecompositionIssue:
        """为失败批次组装任务分解问题（合并调度器上报的失败证据）。

        优先合并本批次（当前计划版本）相关的调度器问题记录（失败 Agent、
        原因与建议）；调度器未上报（如调度结构异常）时依据执行结果兜底
        构造，保证失败升级流程始终有诊断输入。

        Args:
            failed_ids: 本批次中失败的子任务编号列表。

        Returns:
            TaskDecompositionIssue：供主管重规划参考的诊断记录。
        """
        failed_set = set(failed_ids)
        relevant = [
            issue
            for issue in self._scheduler.decomposition_issues
            if issue.subtask_id in failed_set
            and issue.plan_version == self._plan_version
        ]
        if relevant:
            merged = relevant[-1].model_copy(deep=True)
            merged.task_id = self._task_request.task_id
            merged.failed_agent_ids = list(
                dict.fromkeys(
                    agent_id
                    for issue in relevant
                    for agent_id in issue.failed_agent_ids
                )
            )
            reasons: list[str] = []
            for issue in relevant:
                reasons.extend(
                    f"[{issue.subtask_id}] {reason}"
                    for reason in issue.failure_reasons
                )
            merged.failure_reasons = list(dict.fromkeys(reasons))
            merged.attempted_approaches = list(
                dict.fromkeys(
                    step
                    for issue in relevant
                    for step in issue.attempted_approaches
                )
            )
            covered = {issue.subtask_id for issue in relevant}
            for subtask_id in failed_ids:
                if subtask_id in covered:
                    continue
                result = self._results.get(subtask_id)
                merged.failure_reasons.append(
                    f"[{subtask_id}] "
                    f"{(result.error if result is not None else None) or '未提供错误信息'}"
                )
            return merged
        # 兜底：调度器未上报问题（如调度结构异常）时按批次结果构造
        fallback_reasons: list[str] = []
        for subtask_id in failed_ids:
            result = self._results.get(subtask_id)
            fallback_reasons.append(
                f"[{subtask_id}] "
                f"{(result.error if result is not None else None) or '未提供错误信息'}"
            )
        return TaskDecompositionIssue(
            task_id=self._task_request.task_id,
            subtask_id=failed_ids[0],
            plan_version=self._plan_version,
            failure_reasons=fallback_reasons,
            attempted_approaches=["由控制层依据执行批次结果兜底构造"],
            recommendation=(
                "检查子任务规格、依赖关系与工具授权是否合理，"
                "必要时调整拆分方式或更换执行角色。"
            ),
        )

    @staticmethod
    def _format_issue(issue: TaskDecompositionIssue) -> str:
        """把分解问题格式化为单行失败历史条目。"""
        reasons = "；".join(issue.failure_reasons) or "未提供原因"
        text = f"[计划 v{issue.plan_version}] 子任务 {issue.subtask_id}：{reasons}"
        if issue.recommendation:
            text += f"（建议：{issue.recommendation}）"
        return text

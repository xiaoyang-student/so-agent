"""汇总协调器（AggregationCoordinator）：结果汇总、终审与返工。

自原 ``WorkflowEngine``（workflow.py）拆分的汇总/结果评审/返工职责
集合，方法体保持逐字平移：

- 结果汇总：``_aggregate_results``（调用主管 Agent 生成面向用户的最终
  答案；单子任务透传 / 短输出本地拼接等快速路径；调用失败回退为本地
  确定性摘要）/ ``_successful_results`` / ``_fallback_summary``；
- 结果评审：``_final_review``（评审 Agent，stage=result_review）；
- 返工：``_resolve_rework_targets``（按结构化标识精确解析返工目标，
  刻意不做自由文本子串匹配）/ ``_build_rework_specs``（构造返工子任务
  清单，依赖裁剪到返工集合内）。

跨协调器协作：主管/评审 Agent 调用经 WorkflowDeps 通道由薄编排器
实现；Agent 引用解析委托共享 WorkflowState（协调器之间不直接互相
引用）。
"""

from __future__ import annotations

from typing import Any

from so_agent.models import (
    ExecutionResult,
    ReviewDecision,
    SubtaskSpec,
    TaskStatus,
)
from so_agent.orchestrator import AggregationOutput
from so_agent.tool_packages.review_agent.agent import ReviewInput, ReviewStage
from so_agent.workflow_support import BaseCoordinator


class AggregationCoordinator(BaseCoordinator):
    """汇总协调器：结果汇总、最终评审与返工目标解析。"""

    # ------------------------------------------------------------------
    # 汇总（aggregating）
    # ------------------------------------------------------------------
    async def _aggregate_results(self) -> str:
        """汇总阶段：调用主管 Agent 生成面向用户的最终答案。

        输入为当前计划的全部成功 ExecutionResult（严格 JSON）；主管调用
        失败或返回空文本时回退为本地确定性摘要（不阻断任务收尾）。

        Returns:
            最终结果文本（主管汇总或本地兜底摘要）。
        """
        self._transition(TaskStatus.AGGREGATING)
        successful = self._successful_results()

        # 单子任务直接透传结果，跳过 LLM 汇总
        if len(successful) == 1:
            return successful[0].output or self._fallback_summary(successful)

        # 双子任务且输出都够短时本地拼接，跳过 LLM 汇总
        if len(successful) <= 2 and all(
            len(r.output or "") < 500 for r in successful
        ):
            return self._fallback_summary(successful)

        payload: dict[str, Any] = {
            "phase": "aggregating",
            "task_id": self._task_request.task_id,
            "task_request": self._task_request.model_dump(mode="json"),
            "plan_version": self._plan_version,
            "subtasks": [spec.model_dump(mode="json") for spec in self._subtasks],
            "execution_results": [
                item.model_dump(mode="json") for item in successful
            ],
        }
        try:
            output = await self._run_supervisor(payload, AggregationOutput)
        except Exception as exc:
            return self._fallback_summary(
                successful,
                reason=f"主管 Agent 汇总不可用：{type(exc).__name__}: {exc}",
            )
        text = (output.final_result or "").strip()
        if text:
            return text
        return self._fallback_summary(successful, reason="主管 Agent 未返回汇总文本")

    def _successful_results(self) -> list[ExecutionResult]:
        """返回当前计划中成功的执行结果（按计划顺序、去重、深拷贝）。"""
        seen: set[str] = set()
        results: list[ExecutionResult] = []
        for spec in self._subtasks:
            if spec.subtask_id in seen:
                continue
            seen.add(spec.subtask_id)
            item = self._results.get(spec.subtask_id)
            if item is not None and item.success:
                results.append(item.model_copy(deep=True))
        return results

    def _fallback_summary(
        self, results: list[ExecutionResult], *, reason: str = ""
    ) -> str:
        """本地兜底汇总：以确定性文本列出各子任务产出摘要。"""
        lines = [
            f"任务 {self._task_request.task_id} 共完成 {len(results)} 个子任务："
        ]
        for item in results:
            brief = (item.output or "").strip().replace("\n", " ")
            if len(brief) > 200:
                brief = brief[:200] + "…"
            lines.append(f"- [{item.subtask_id}] {brief or '（无文本产出）'}")
        if reason:
            lines.append(f"（汇总说明：{reason}）")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # 最终评审（result_review）
    # ------------------------------------------------------------------
    async def _final_review(self) -> ReviewDecision:
        """最终评审：调用评审 Agent（stage=result_review）校验整体结果。"""
        self._transition(TaskStatus.RESULT_REVIEW)
        decision = await self._call_review_agent(
            ReviewInput(
                stage=ReviewStage.RESULT_REVIEW,
                task_request=self._task_request,
                subtasks=[spec.model_copy(deep=True) for spec in self._subtasks],
                execution_results=self._successful_results(),
                plan_version=self._plan_version,
            )
        )
        decision.stage = "result_review"
        self._last_review = decision
        return decision

    # ------------------------------------------------------------------
    # 返工（rework）
    # ------------------------------------------------------------------
    def _resolve_rework_targets(self, decision: ReviewDecision) -> set[str]:
        """解析结果评审指定的返工目标（subtask_id 集合）。

        ``retry_target`` 的语义是单个 subtask_id 或 agent_id（见
        ``ReviewDecision.retry_target``），因此按**精确标识**解析：先当
        subtask_id 精确匹配；未命中再当 agent_id（``{task_id}:{
        subtask_id}:...``）解析出 subtask_id。

        刻意**不做子串包含匹配**：自由文本（issues / required_fixes）中的
        短标识（如 ``"1"``）会误命中 ``"11"``、``"1 处"`` 等，导致无关子任务
        被返工、正确结果被覆盖。全部未命中时回退为"失败结果兜底"（返工
        所有失败子任务）；仍为空则返回空集合（由调用方转入人工兜底）。
        """
        plan_id_set = {spec.subtask_id for spec in self._subtasks}
        targets: set[str] = set()
        retry = (decision.retry_target or "").strip()
        if retry:
            if retry in plan_id_set:
                targets.add(retry)
            else:
                parsed = self._parse_agent_ref(retry)
                if parsed in plan_id_set:
                    targets.add(parsed)
        if not targets:
            targets.update(
                sid
                for sid, item in self._results.items()
                if sid in plan_id_set and not item.success
            )
        return targets

    def _build_rework_specs(self, rework_ids: set[str]) -> list[SubtaskSpec]:
        """构造返工子任务清单：仅保留返工子任务，把依赖裁剪到返工集合内。

        依赖裁剪保证返工批次仍是合法 DAG；未返工的依赖视为已满足
        （其成功结果保留在 ``self._results`` 中，聚合时仍会被纳入）。

        增量补充：返工子任务若在 ``self._results`` 中留有历史产出
        （output 长度 > 20），连同最近一次评审（``self._last_review``）
        的 ``required_fixes`` 一并注入 ``instructions``，让返工 Agent 在
        已有成果上补充修正，不必从头重做。
        """
        specs: list[SubtaskSpec] = []
        for spec in self._subtasks:
            if spec.subtask_id not in rework_ids:
                continue
            clone = spec.model_copy(deep=True)
            clone.dependencies = [
                dep for dep in clone.dependencies if dep in rework_ids
            ]
            existing = self._results.get(spec.subtask_id)
            if existing and existing.output and len(existing.output) > 20:
                clone.instructions += (
                    f"\n\n[已有产出] 以下是你之前的产出，"
                    f"评审认为方向正确但需要补充/修正，"
                    f"请在此基础上改进，不需要从头重做：\n{existing.output[:3000]}\n"
                )
            if self._last_review and self._last_review.required_fixes:
                clone.instructions += (
                    f"\n[评审要求] {self._last_review.required_fixes}\n"
                )
            specs.append(clone)
        return specs

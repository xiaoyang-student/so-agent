"""provisional 交付与结果评审策略（WorkflowEngine 混入）。

主流水线在交付阶段调用这些方法：判断是否可先行交付、执行后台评审、
失败时转人工兜底，以及结果评审与确定性覆盖的风险判定。
"""

from __future__ import annotations

import asyncio

from so_agent.models import TaskStatus
from so_agent.orchestrator import OrchestratorOutput
from so_agent.runtime.budget import TokenBudgetExceeded
from so_agent.runtime.events import (
    EVENT_PROVISIONAL_ESCALATED,
    EVENT_PROVISIONAL_REVIEW_FAILED,
    EVENT_PROVISIONAL_REVIEW_PASSED,
    EVENT_PROVISIONAL_REVISED,
)
from so_agent.runtime.logging_setup import get_logger
from so_agent.runtime.registry import DEFAULT_SUPERVISOR_ID
from so_agent.state_machine import PipelineStoppedError, WorkflowCancelledError

logger = get_logger("workflow.provisional")

# 简单任务快速通道的产出充分性阈值：全部成功子任务的 output 非空且
# 长度严格大于该值时才允许跳过汇总与结果评审，防止空白/过短产出
# 借快速通道绕过质量把关
_SIMPLE_FAST_PATH_MIN_OUTPUT_CHARS: int = 50


class ProvisionalMixin:
    @property
    def provisional_review_pending(self) -> bool:
        """后台 provisional 评审是否仍在进行。"""
        task = self._provisional_task
        return task is not None and not task.done()

    async def wait_for_provisional(self) -> OrchestratorOutput | None:
        """等待后台 provisional 评审结束并返回最终输出。

        未启用 provisional 交付或未创建后台任务时返回 None。等待方被
        取消不会取消后台评审（``asyncio.shield``），后台异常被吞并后
        返回最后一次输出，由调用方按状态处理。
        """
        task = self._provisional_task
        if task is None:
            return None
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            if not task.done():
                raise
        except Exception:  # 后台异常已写入事件与输出，这里只负责收口
            pass
        return self._provisional_output

    def _start_provisional_review_if_pending(
        self, output: OrchestratorOutput
    ) -> None:
        """在流水线收尾后为 provisional 输出启动后台评审任务。"""
        if output.status is not TaskStatus.COMPLETED_PROVISIONAL:
            return
        if self._provisional_task is not None:
            return
        delivered = output.final_result or self._provisional_delivered_text
        self._provisional_delivered_text = delivered
        self._provisional_output = output
        task_id = output.task_id[:8] if output.task_id else "task"
        self._provisional_task = asyncio.create_task(
            self._run_provisional_review(delivered),
            name=f"provisional-review-{task_id}",
        )

    def _should_run_result_review(self) -> tuple[bool, str]:
        """按 ``result_review_mode`` 决定是否执行语义结果评审。

        - ``always``（默认）：恒评审，保持既有行为；
        - ``off``：只做确定性验收（write_scope / 工具证据 / 沙箱执行），
          跳过语义评审；
        - ``risk_based``：以下任一条件命中即阻塞评审——存在验收标准、
          存在写入副作用（``write_scope`` 非空）、有重规划/返工历史、
          确定性检查覆盖不足；全部未命中且结果成功、产出非空时跳过。

        Returns:
            ``(是否评审, 原因)``；原因写入 ``review_skipped`` 事件供审计。
        """
        mode = str(
            getattr(self._settings, "result_review_mode", "always") or "always"
        ).strip().lower()
        if mode == "off":
            return False, "disabled"
        if mode != "risk_based":
            return True, "always"
        if (
            self._task_request is not None
            and self._task_request.acceptance_criteria
        ):
            return True, "acceptance_criteria_present"
        if any(spec.write_scope for spec in self._subtasks):
            return True, "side_effects_present"
        if self._replan_count > 0 or self._result_rework_count > 0:
            return True, "history_failures"
        if not self._verification_coverage_complete():
            return True, "verification_coverage_insufficient"
        successful = self._successful_results()
        if not successful:
            return True, "no_successful_results"
        if any(not (item.output or "").strip() for item in successful):
            return True, "empty_output"
        if any(not item.success for item in self._results.values()):
            return True, "failed_results"
        return False, "risk_based_low_risk"

    def _verification_coverage_complete(self) -> bool:
        """判断确定性检查是否覆盖全部需要验证的产出。

        需要验证 = 存在验收标准或存在写入声明；此时每个子任务都必须
        声明 ``verification`` 规则且全部成功。无验收标准且无写入的纯
        只读任务没有可覆盖对象，视为覆盖完整。
        """
        needs_verification = bool(
            self._task_request is not None
            and self._task_request.acceptance_criteria
        ) or any(spec.write_scope for spec in self._subtasks)
        if not needs_verification:
            return True
        for spec in self._subtasks:
            if not spec.verification:
                return False
            result = self._results.get(spec.subtask_id)
            if result is None or not result.success:
                return False
        return True

    def _should_deliver_provisionally(self) -> bool:
        """是否先交付结果、后台评审（默认关闭）。

        仅当配置开启且计划属于低风险无副作用场景时启用：任务无验收
        标准、无写入声明、无重规划/返工历史、确定性检查覆盖完整，且
        全部子任务成功、产出非空。任何外部副作用或历史失败信号都
        保持同步阻塞评审，避免先交付后纠正。
        """
        if not bool(
            getattr(self._settings, "result_review_async_provisional", False)
        ):
            return False
        if self._task_request is None:
            return False
        if self._task_request.acceptance_criteria:
            return False
        if any(spec.write_scope for spec in self._subtasks):
            return False
        if self._replan_count > 0 or self._result_rework_count > 0:
            return False
        if not self._verification_coverage_complete():
            return False
        if not self._results or any(
            not item.success for item in self._results.values()
        ):
            return False
        successful = self._successful_results()
        return bool(successful) and all(
            (item.output or "").strip() for item in successful
        )

    async def _run_provisional_review(self, delivered_text: str) -> None:
        """后台评审 provisional 结果，失败时自动返工最多 1 次。

        主流水线在交付后已停止并回收授权；本方法先重置停止门再继续
        执行，评审通过则转 ``completed``；未通过时只返工被指出的子
        任务，成功后重新汇总并在输出中保留旧结果；返工耗尽或无法
        定位返工目标时转人工兜底。所有分支都写入对应 provisional
        审计事件，最终输出写入 ``_provisional_output``。
        """
        task_id = self._task_request.task_id if self._task_request else ""
        try:
            self._gate.reset()
            self._check_cancel()
            self._transition(
                TaskStatus.RESULT_REVIEW, reason="异步结果评审"
            )
            decision = await self._final_review(transition=False)
            self._check_cancel()
            if decision.passed:
                self._events.log_event(
                    EVENT_PROVISIONAL_REVIEW_PASSED,
                    {
                        "task_id": task_id,
                        "agent_id": DEFAULT_SUPERVISOR_ID,
                        "summary": decision.summary,
                    },
                )
                self._transition(
                    TaskStatus.COMPLETED, reason="异步结果评审通过"
                )
                self._provisional_output = self._build_output(
                    TaskStatus.COMPLETED,
                    final_result=delivered_text,
                    review=decision,
                )
                return

            self._events.log_event(
                EVENT_PROVISIONAL_REVIEW_FAILED,
                {
                    "task_id": task_id,
                    "agent_id": DEFAULT_SUPERVISOR_ID,
                    "issues": decision.issues[:20],
                    "required_fixes": decision.required_fixes[:20],
                },
            )
            rework_ids = self._resolve_rework_targets(decision)
            if not rework_ids:
                self._provisional_output = await self._escalate_provisional(
                    "异步评审未通过且无法定位返工目标", delivered_text
                )
                return
            if self._result_rework_count >= 1:
                self._provisional_output = await self._escalate_provisional(
                    "异步评审自动返工次数耗尽仍未通过", delivered_text
                )
                return

            self._result_rework_count += 1
            rework_specs = self._build_rework_specs(rework_ids)
            if not rework_specs:
                self._provisional_output = await self._escalate_provisional(
                    "异步评审返工目标为空", delivered_text
                )
                return
            for subtask_id in rework_ids:
                self._results.pop(subtask_id, None)
            self._pending_specs = rework_specs
            self._check_cancel()
            batch = await self._execute_subtasks(rework_specs)
            failed = [
                subtask_id
                for subtask_id, item in batch.items()
                if not item.success
            ]
            if failed:
                self._provisional_output = await self._escalate_provisional(
                    "异步评审返工执行失败："
                    + "、".join(sorted(failed)),
                    delivered_text,
                )
                return

            revised_text = await self._aggregate_results(transition=False)
            # 返工后必须再次评审：未通过则升级人工，绝不把未验证的
            # 修订结果当作最终结果交付
            confirm = await self._final_review(transition=False)
            self._check_cancel()
            if not confirm.passed:
                self._provisional_output = await self._escalate_provisional(
                    "异步评审返工后仍未通过", delivered_text
                )
                return
            self._transition(
                TaskStatus.COMPLETED, reason="异步评审返工完成"
            )
            self._events.log_event(
                EVENT_PROVISIONAL_REVISED,
                {
                    "task_id": task_id,
                    "agent_id": DEFAULT_SUPERVISOR_ID,
                    "rework_subtasks": sorted(rework_ids),
                    "previous_result_chars": len(delivered_text),
                    "revised_result_chars": len(revised_text),
                },
            )
            self._provisional_output = self._build_output(
                TaskStatus.COMPLETED,
                final_result=revised_text,
                review=confirm,
                previous_result=delivered_text,
                notice=(
                    "后台评审发现问题并已完成修订，结果已更新；"
                    "旧版本保留在 previous_result 中。"
                ),
            )
        except asyncio.CancelledError:
            raise
        except WorkflowCancelledError:
            self._transition(
                TaskStatus.CANCELLED, reason="异步评审期间任务被取消"
            )
            self._provisional_output = self._build_output(
                TaskStatus.CANCELLED, final_result="任务已取消"
            )
        except PipelineStoppedError:
            self._transition(
                TaskStatus.CANCELLED, reason="异步评审期间 pipeline 已停止"
            )
            self._provisional_output = self._build_output(
                TaskStatus.CANCELLED, final_result="任务已取消"
            )
        except TokenBudgetExceeded as exc:
            # 预算熔断是 BaseException：必须显式收口，否则后台任务会
            # 带着未处理异常结束，provisional 结果也不会被标记为失败
            self._transition(
                TaskStatus.FAILED,
                reason=f"token_budget_exceeded: {exc.violation.message}",
            )
            self._provisional_output = self._build_output(
                TaskStatus.FAILED,
                final_result=f"任务因 token 预算熔断终止：{exc.violation.message}",
                previous_result=delivered_text,
                notice="后台评审期间触发 token 预算熔断，已交付结果未经最终确认。",
            )
        except Exception as exc:
            logger.warning(
                "异步评审异常，转人工兜底：%s: %s",
                type(exc).__name__,
                exc,
            )
            try:
                self._provisional_output = await self._escalate_provisional(
                    f"异步评审异常：{type(exc).__name__}: {exc}",
                    delivered_text,
                )
            except Exception:
                logger.exception("异步评审异常后的人工兜底也失败")

    async def _escalate_provisional(
        self, reason: str, delivered_text: str
    ) -> OrchestratorOutput:
        """把 provisional 评审失败转人工兜底，并保留已交付结果。"""
        output = await self._escalate_to_human(reason)
        self._events.log_event(
            EVENT_PROVISIONAL_ESCALATED,
            {
                "task_id": output.task_id,
                "agent_id": DEFAULT_SUPERVISOR_ID,
                "reason": reason,
            },
        )
        return output.model_copy(
            update={
                "previous_result": delivered_text,
                "notice": (
                    "后台评审未通过，任务已转人工处理；"
                    "此前交付的结果保留在 previous_result 中。"
                ),
            }
        )

    def _simple_outputs_sufficient(self) -> bool:
        """快速通道产出充分性校验：全部成功子任务的产出非空且足够长。

        阈值见 ``_SIMPLE_FAST_PATH_MIN_OUTPUT_CHARS``（严格大于）；任一
        成功子任务产出缺失、空白或过短即判定不充分，任务转走完整
        汇总与结果评审，防止内容为空的"成功"借快速通道放行。
        """
        successful = self._successful_results()
        if not successful:
            return False
        return all(
            len((item.output or "").strip()) > _SIMPLE_FAST_PATH_MIN_OUTPUT_CHARS
            for item in successful
        )

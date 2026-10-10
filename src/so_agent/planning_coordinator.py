"""规划协调器（PlanningCoordinator）：拆解、计划评审与工具授权。

自原 ``WorkflowEngine``（workflow.py）拆分的规划/评审/授权职责集合，
方法体保持逐字平移：

- 任务接收：``_create_task``（生成 task_id、固化任务请求并记录事件）；
- 拆解与两阶段计划评审循环：``_plan_and_review``（主管 Agent 拆解 →
  基础校验 → 生成授权候选（proposal，不注册网关）→ 计划评审 +
  授权评审；两路评审均通过后才真实签发 ToolGrant，评审不通过或拆解
  非法时消耗"任务拆分修改额度"重试）；网络异常（超时/连接失败）按
  独立网络重试预算处理，不消耗重规划额度；重规划前统一清理活跃授权；
- 拆解辅助：``_request_plan`` / ``_build_planning_payload`` /
  ``_extract_plan`` / ``_adopt_plan`` / ``_print_plan_summary`` /
  ``_is_simple_plan``（快速通道后置判断）；
- 授权：``_propose_grants``（评审用纯候选，不注册网关）/
  ``_issue_grants``（按子任务槽位真实签发，只覆盖已批准可分配工具；
  两路评审通过后或执行窗口重开/工具补发时调用）；
- 评审：``_review_plan``（计划评审 + 授权评审并行合并）/
  ``_combine_reviews``；
- 结果复用（任务 35，显式声明制）：``_archive_completed_results``
  （重规划替换前把旧版本结果归档到 ``archive_results``，保证来源
  可查证）/ ``_reuse_candidates_payload``（重规划载荷的可复用候选
  最小充分信息与失败级联范围）/ ``_validate_reuse_declarations``
  （对主管 ``reuse_from`` 声明的安全校验器：来源存在且成功、非失败
  下游闭包、指纹成分兼容（依赖经显式映射等价，禁止文本相似度）、
  依赖同时复用、artifacts 在沙箱内且存在、工具范围不扩大、单一
  消费；通过复制结果并记录 ``result_reused``，失败记录
  ``reuse_rejected`` 并回退正常 pending 队列）/ ``_downstream_closure``
  （失败级联闭包）/ ``_record_plan_version``（带指纹索引的计划版本
  快照入库）/ ``_completed_results_summary``（重规划载荷的已完成
  结果摘要）。仅凭 fingerprint 的自动复用决策路径已删除：无
  ``reuse_from`` 声明时绝不复用，即使 id、名称与指纹完全相同；
  LLM 负责语义判断，代码只校验安全硬约束；
- 治理额度：``_try_consume_replan_quota``；
- 文本工具：``_split_items``（多行/分号分隔条目解析）。

跨协调器协作：``_run_supervisor`` / ``_call_review_agent`` 经
WorkflowDeps 通道调用薄编排器绑定的方法；执行协调器在本协调器之外
通过同一通道回调 ``_plan_and_review`` / ``_issue_grants`` /
``_try_consume_replan_quota``（协调器之间不直接互相引用）。
"""

from __future__ import annotations

import asyncio
import re
import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import openai

from so_agent.fingerprint import spec_fingerprint, specs_fingerprint_equivalent
from so_agent.models import (
    ExecutionResult,
    PlanVersion,
    ReviewDecision,
    SubtaskSpec,
    TaskDecompositionIssue,
    TaskRequest,
    TaskStatus,
    ToolGrant,
    _utc_now,
)
from so_agent.orchestrator import OrchestratorInput, PlanningOutput
from so_agent.runtime.events import (
    EVENT_REPLAN,
    EVENT_RESULT_REUSED,
    EVENT_REUSE_REJECTED,
    EVENT_TASK_CREATED,
)
from so_agent.runtime.registry import DEFAULT_SUPERVISOR_ID
from so_agent.state_machine import SupervisorNetworkError, WorkflowError
from so_agent.tool_packages.review_agent.agent import ReviewInput, ReviewStage
from so_agent.workflow_support import BaseCoordinator

# 主管 Agent 网络异常类型：请求超时与连接失败。区别于计划结构问题——
# 网络失败不计入"任务拆分修改额度"，按独立预算重试（见 _plan_and_review）
_SUPERVISOR_NETWORK_ERRORS: tuple[type[Exception], ...] = (
    openai.APITimeoutError,
    openai.APIConnectionError,
)

# 网络重试预算（独立于任务拆分修改额度）：主管 Agent 调用连续网络失败
# （超时 / 连接错误）超过该次数时转入人工兜底（网络连续失败）
_MAX_NETWORK_RETRIES: int = 3


class PlanningCoordinator(BaseCoordinator):
    """规划协调器：任务接收、拆解、计划评审与工具授权。"""

    # ------------------------------------------------------------------
    # 任务接收
    # ------------------------------------------------------------------
    def _create_task(
        self, objective: str, constraints: str, acceptance_criteria: str
    ) -> None:
        """接收任务：生成 task_id（UUID）、固化任务请求并记录事件。

        Raises:
            WorkflowError: objective 为空时。
        """
        objective_text = (objective or "").strip()
        if not objective_text:
            raise WorkflowError("任务目标（objective）不能为空")
        constraints_list = self._split_items(constraints)
        criteria_list = self._split_items(acceptance_criteria)

        original_parts = [objective_text]
        if constraints_list:
            original_parts.append("[约束条件] " + "；".join(constraints_list))
        if criteria_list:
            original_parts.append("[验收标准] " + "；".join(criteria_list))

        self._task_request = TaskRequest(
            task_id=uuid.uuid4().hex,
            objective=objective_text,
            constraints=constraints_list,
            acceptance_criteria=criteria_list,
            original_input="\n".join(original_parts),
        )
        self._events.log_event(
            EVENT_TASK_CREATED,
            {
                "task_id": self._task_request.task_id,
                "agent_id": DEFAULT_SUPERVISOR_ID,
                "project_name": self._context.project_name,
                "request": self._task_request.model_dump(mode="json"),
            },
        )

    @staticmethod
    def _split_items(text: str) -> list[str]:
        """把多行/分号分隔的文本拆分为条目列表（去空、去重、保序）。"""
        if not isinstance(text, str) or not text.strip():
            return []
        items: list[str] = []
        for chunk in re.split(r"[\n\r;；]+", text):
            item = chunk.strip()
            if item and item not in items:
                items.append(item)
        return items

    # ------------------------------------------------------------------
    # 规划与评审（planning / replanning / plan_review）
    # ------------------------------------------------------------------
    async def _plan_and_review(
        self,
        *,
        issue: TaskDecompositionIssue | None = None,
        feedback: ReviewDecision | None = None,
    ) -> bool:
        """拆解任务并完成两阶段计划评审（计划评审 + 授权评审）。

        循环内部：调用主管 Agent 拆解 → 基础校验 → 生成授权候选
        （proposal）→ 计划评审 + 授权评审；两路评审均通过后才真实
        签发 ToolGrant（被拒计划在网关中没有任何签发记录）。评审不
        通过（或拆解结果非法）时消耗一次"任务拆分修改额度"并重新
        拆解，直到通过或额度耗尽。
        拆解评审重试包含在三次任务拆分修改额度内。重规划（plan_version
        > 0）进入本方法时先统一幂等清理上一版本的活跃授权，并把当前
        版本结果归档到 ``archive_results``（候选载荷构造发生在主管
        返回新计划之前，归档必须先行）。

        网络异常（主管 Agent 调用超时 / 连接失败）与计划结构问题分开
        处理：按独立网络重试预算（``_MAX_NETWORK_RETRIES``）重试，
        不消耗拆分修改额度；网络连续失败耗尽预算时抛出
        ``SupervisorNetworkError``（由调用方转入人工兜底）。

        Args:
            issue: 触发本次重规划的分解问题（首轮为 None）。
            feedback: 上一次评审的不通过结论（首轮为 None）。

        Returns:
            True 表示当前计划（``self._subtasks``）已通过评审；
            False 表示拆分修改额度耗尽，应转入人工兜底。

        Raises:
            SupervisorNetworkError: 主管 Agent 网络连续失败且重试
                预算耗尽时（不消耗重规划额度）。
        """
        # 显式复用制（任务 35）：重规划进入时先把当前版本结果（含失败）
        # 归档到 archive_results——_request_plan 构造可复用候选载荷发生
        # 在主管返回新计划之前，归档必须先行，否则首次重规划的候选载荷
        # 查无来源。幂等（同版本同 subtask_id 深拷贝覆盖）；拆解重试
        # 循环不会重复累计。首次规划（plan_version == 0）无历史可归档。
        if self._plan_version > 0:
            self._archive_completed_results()
        network_retry_count = 0
        while True:
            # 停止门（任务 45）：重规划/拆解重试循环头拒绝新动作
            #（PipelineStoppedError 穿透宽捕获，不消耗重规划额度）
            self._gate_check("replan")
            is_replan = self._plan_version > 0
            if is_replan:
                # 重规划前统一清理：撤销上一版本计划残留的活跃授权
                # （批次后回收的兜底；幂等，重复调用不产生重复事件）
                self._revoke_active_grants(reason="lifecycle_cleanup:replanning")
            self._transition(TaskStatus.REPLANNING if is_replan else TaskStatus.PLANNING)
            self._check_cancel()

            version_before_attempt = self._plan_version
            try:
                subtasks, problems = await self._request_plan(
                    issue=issue, feedback=feedback
                )
            except _SUPERVISOR_NETWORK_ERRORS as exc:
                # 网络异常（超时/连接失败）：不计入重规划额度，按独立预算重试
                network_retry_count += 1
                error_name = type(exc).__name__
                if network_retry_count > _MAX_NETWORK_RETRIES:
                    raise SupervisorNetworkError(
                        f"主管 Agent 网络连续失败（{network_retry_count} 次，"
                        f"最后异常：{error_name}）"
                    ) from exc
                print(
                    f"[so-agent] 网络请求失败（{error_name}），正在重试...",
                    file=sys.stderr, flush=True,
                )
                print(
                    f"[so-agent] 网络重试 {network_retry_count}/{_MAX_NETWORK_RETRIES}...",
                    file=sys.stderr, flush=True,
                )
                continue

            # 迟到结果隔离（任务 45）：主管返回后、采纳新计划前检查——
            # 等待期间已停止则不采纳（不 clear results、不推进版本）
            self._gate_check("adopt_plan")
            if not problems:
                # 显式复用制（任务 35）：采纳新计划并清空 results；只有
                # 通过安全校验的 reuse_from 声明才会把旧结果复制回
                # self._results。无声明时绝不复用——即使 id、名称与
                # fingerprint 完全相同（LLM 负责语义判断，代码只校验
                # 声明；仅凭指纹的自动复用路径已删除）。旧版本结果已在
                # 方法入口归档（候选载荷与校验器经 archive_results
                # 查证来源），此处不再重复归档。
                self._plan_version += 1
                adopted = self._adopt_plan(subtasks)
                self._subtasks = adopted
                # 只保留显式声明且校验通过的复用结果：失败结果与未被
                # 声明复用的旧结果不得跨版本残留（subtask_id 可能被
                # 复用但语义不同，避免污染聚合/诊断）。
                self._results.clear()
                reused = self._validate_reuse_declarations()
                self._results.update(reused)
                self._plan_history.append(
                    [spec.model_copy(deep=True) for spec in self._subtasks]
                )
                # 授权候选（proposal）：计划评审的授权输入能看到"预计
                # 授权工具"，但候选不注册网关、无有效签发记录；真实
                # 签发推迟到两路评审均通过后（见下方 decision.passed）
                self._propose_grants()
                # --- 输出拆解结果详情与复用统计 ---
                self._print_plan_summary()
                if reused:
                    print(
                        f"[so-agent] 复用 {len(reused)} 个子任务的既有成功结果"
                        f"（{', '.join(sorted(reused))}），跳过重新执行",
                        file=sys.stderr, flush=True,
                    )
                self._record_plan_version(issue)
                if is_replan:
                    self._events.log_event(
                        EVENT_REPLAN,
                        {
                            "task_id": self._task_request.task_id,
                            "agent_id": DEFAULT_SUPERVISOR_ID,
                            "plan_version": self._plan_version,
                            "replan_count": self._replan_count,
                            "subtask_ids": [s.subtask_id for s in self._subtasks],
                            "issue": issue.model_dump(mode="json") if issue else None,
                        },
                    )
                decision = await self._review_plan()
                if decision.passed:
                    # 停止门（任务 45）：评审等待期间已停止则不签发授权
                    #（所有出口撤销授权不变，此处拒绝新增签发）
                    self._gate_check("issue_grants")
                    # 两路评审（计划评审 + 授权评审）均通过：此刻才真实
                    # 签发 ToolGrant（被拒计划在网关无任何签发记录）
                    self._issue_grants()
                    print(
                        f"[so-agent] 计划评审通过（版本 {self._plan_version}）",
                        file=sys.stderr, flush=True,
                    )
                    return True
                print(
                    f"[so-agent] 计划评审未通过：{decision.summary or '；'.join(decision.issues) or '未说明原因'}",
                    file=sys.stderr, flush=True,
                )
                feedback = decision
                issue = None
            else:
                feedback = ReviewDecision(
                    stage="plan_review",
                    passed=False,
                    issues=problems,
                    required_fixes=["修正拆解输出的结构问题后重新提交完整子任务列表"],
                    summary="拆解结果未通过基础校验（未进入评审阶段）",
                )
                self._last_review = feedback

            if not self._try_consume_replan_quota():
                return False

            # 计划未被采纳（plan_version 不变）→ 同状态重试：状态迁移
            # 去重逻辑不会触发进度提示，这里补充"第 N 次重规划"提示
            if self._plan_version == version_before_attempt:
                max_attempts = int(self._settings.max_replan_attempts)
                print(
                    f"[so-agent] 第 {self._replan_count} 次重规划"
                    f"（拆分修改额度 {self._replan_count}/{max_attempts}）...",
                    file=sys.stderr,
                    flush=True,
                )

    def _print_plan_summary(self) -> None:
        """打印当前计划的拆解结果详情（版本号 + 子任务清单，stderr）。"""
        print(
            f"[so-agent] 拆解完成（版本 {self._plan_version}），"
            f"共 {len(self._subtasks)} 个子任务：",
            file=sys.stderr, flush=True,
        )
        for spec in self._subtasks:
            tools_str = ", ".join(spec.allowed_tools) if spec.allowed_tools else "无"
            print(
                f"  - [{spec.subtask_id}] {spec.title}\n"
                f"    角色: {spec.role} | 工具: {tools_str}",
                file=sys.stderr, flush=True,
            )

    async def _request_plan(
        self,
        *,
        issue: TaskDecompositionIssue | None,
        feedback: ReviewDecision | None,
    ) -> tuple[list[SubtaskSpec], list[str]]:
        """调用主管 Agent 生成/修订拆分方案，并提取校验后的子任务列表。

        网络异常（超时/连接失败）直接抛出，由 ``_plan_and_review`` 按
        独立网络重试预算处理；其余异常（调用/解析失败）仍转为计划
        问题清单（消耗重规划额度）。

        Returns:
            ``(子任务列表, 问题清单)``；问题清单非空表示计划非法
            （应重新拆解），此时子任务列表为空。

        Raises:
            openai.APITimeoutError: 主管 Agent 请求超时。
            openai.APIConnectionError: 主管 Agent 连接失败。
        """
        payload = self._build_planning_payload(issue=issue, feedback=feedback)
        try:
            output = await self._run_supervisor(payload, PlanningOutput)
        except _SUPERVISOR_NETWORK_ERRORS:
            raise  # 网络异常：交由 _plan_and_review 按独立网络预算重试
        except Exception as exc:
            return [], [f"主管 Agent 调用/解析失败：{type(exc).__name__}: {exc}"]
        return self._extract_plan(output)

    def _build_planning_payload(
        self,
        *,
        issue: TaskDecompositionIssue | None,
        feedback: ReviewDecision | None,
    ) -> dict[str, Any]:
        """构造拆解/重规划阶段的 JSON 输入（含可复用候选信息）。

        在标准 OrchestratorInput 契约与上一版计划快照之外，重规划时
        附带：

        - ``completed_results``：已成功子任务的结果摘要（供主管参考
          调整计划）；
        - ``reuse_candidates``（任务 35）：可复用候选的最小充分信息
          （旧 plan version、subtask_id、spec 语义字段、success、输出
          摘要、evidence、artifacts、失败诊断），供主管显式填写
          ``reuse_from``；首次规划不提供（无历史可复用）；
        - ``failure_cascade_scope``：失败级联范围（失败子任务及其全部
          下游，这些旧结果已被失败污染，不可声明复用）。

        结果复用决策由主管的显式 ``reuse_from`` 声明驱动，代码只负责
        安全校验（见 ``_validate_reuse_declarations``）。
        """
        orchestrator_input = OrchestratorInput(
            objective=self._task_request.objective,
            constraints=list(self._task_request.constraints),
            acceptance_criteria=list(self._task_request.acceptance_criteria),
        )
        payload: dict[str, Any] = {
            "phase": "replanning" if self._plan_version > 0 else "planning",
            "task_id": self._task_request.task_id,
            "orchestrator_input": orchestrator_input.model_dump(mode="json"),
            "plan_version": self._plan_version + 1,
            "replan_count": self._replan_count,
            "max_replan_attempts": int(self._settings.max_replan_attempts),
            "available_tools": self._tool_registry.list_assignable_tools(
                approved_only=True
            ),
            "previous_plan": [spec.model_dump(mode="json") for spec in self._subtasks],
        }
        if issue is not None:
            payload["decomposition_issue"] = issue.model_dump(mode="json")
        if feedback is not None:
            payload["review_feedback"] = feedback.model_dump(mode="json")
        # 重规划场景：附带已成功子任务的结果摘要供主管参考
        completed_results = self._completed_results_summary()
        if completed_results:
            payload["completed_results"] = completed_results
        # 重规划场景：可复用候选最小充分信息与失败级联范围（任务 35，
        # 供主管显式声明 reuse_from）；首次规划不提供候选
        if self._plan_version > 0:
            candidates, cascade_scope = self._reuse_candidates_payload()
            if candidates:
                payload["reuse_candidates"] = candidates
            if cascade_scope:
                payload["failure_cascade_scope"] = cascade_scope
        return payload

    def _extract_plan(
        self, output: PlanningOutput
    ) -> tuple[list[SubtaskSpec], list[str]]:
        """从规划输出（PlanningOutput 精简契约）提取子任务列表并做基础校验。

        校验项：subtasks 非空、subtask_id 非空且唯一。未通过校验时返回
        问题清单（触发重新拆解）。

        Returns:
            ``(子任务列表, 问题清单)``。
        """
        if not output.subtasks:
            return [], ["主管 Agent 输出缺少 subtasks（拆分方案为空）"]

        specs = [spec.model_copy(deep=True) for spec in output.subtasks]
        problems: list[str] = []
        seen: set[str] = set()
        duplicates: set[str] = set()
        for spec in specs:
            subtask_id = (spec.subtask_id or "").strip()
            if not subtask_id:
                problems.append("存在 subtask_id 为空的子任务")
                continue
            if subtask_id in seen:
                duplicates.add(subtask_id)
            seen.add(subtask_id)
        if duplicates:
            problems.append(f"存在重复的 subtask_id：{sorted(duplicates)}")
        return specs, problems

    def _adopt_plan(self, subtasks: list[SubtaskSpec]) -> list[SubtaskSpec]:
        """采纳拆分方案：规范化字段并裁剪非法引用。

        - ``subtask_id`` 去除首尾空白；
        - ``plan_version`` 强制为当前计划版本号；
        - ``allowed_tools`` 裁剪主管专属工具（assignable=False，严禁分配），
          其余名称去重保序保留：尚未注册工具名作为"工具缺口"留给执行
          阶段的缺口流程（创建 → 评审 → 注册 → 授权）处理；
        - ``dependencies`` 移除不存在或自引用的 subtask_id（避免调度器
          因依赖非法而拒绝整个计划）。
        """
        valid_ids = {(spec.subtask_id or "").strip() for spec in subtasks}
        adopted: list[SubtaskSpec] = []
        for spec in subtasks:
            spec.subtask_id = (spec.subtask_id or "").strip()
            spec.plan_version = self._plan_version
            cleaned_tools: list[str] = []
            for tool in spec.allowed_tools:
                name = tool.strip() if isinstance(tool, str) else ""
                if not name or name in cleaned_tools:
                    continue
                entry = self._tool_registry.get_tool(name)
                if entry is not None and not entry.get("assignable", False):
                    continue  # 主管专属工具：严禁分配给子 Agent
                cleaned_tools.append(name)
            spec.allowed_tools = cleaned_tools
            spec.dependencies = [
                dep
                for dep in spec.dependencies
                if dep in valid_ids and dep != spec.subtask_id
            ]
            adopted.append(spec)
        return adopted

    # ------------------------------------------------------------------
    # 结果复用（任务 35：主管显式 reuse_from 声明 + 安全校验器）
    # ------------------------------------------------------------------
    def _archive_completed_results(self) -> None:
        """把当前版本的执行结果归档到 ``archive_results``。

        归档键为 ``(当前 plan_version, subtask_id)``，深拷贝固化成功与
        失败结果。``self._results`` 在重规划采纳时会被清空，历史来源
        只能经此归档查证——归档独立于 ``results`` 存活，InMemory 与
        SQLite 场景行为一致（不依赖 ResultStore 读回，旧库无需迁移）。
        幂等：同版本重复归档按 subtask_id 覆盖。

        调用时机：``_plan_and_review`` 入口（重规划进入、请求新计划
        之前）——候选载荷的构造发生在主管返回新计划之前，归档必须
        先行，否则首次重规划的候选载荷查无来源；采纳新计划的
        ``if not problems`` 块不再重复归档。
        """
        for subtask_id, result in self._results.items():
            self._archive_results[(self._plan_version, subtask_id)] = (
                result.model_copy(deep=True)
            )

    def _reuse_candidates_payload(
        self,
    ) -> tuple[list[dict[str, Any]], list[str]]:
        """构造重规划输入的可复用候选载荷（最小充分信息）。

        候选条目覆盖上一版计划中**已有结果**（成功或失败）的子任务，
        每条含：旧 plan version、subtask_id、spec 语义字段（title /
        instructions / role / dependencies / allowed_tools /
        expected_output）、success、输出摘要（截断）、evidence、
        artifacts 与失败诊断（error，仅失败项）。失败项列入载荷是为
        了向主管完整呈现失败语境；只有 ``success=true`` 的条目才可能
        通过复用校验（校验器强制）。

        调用时机：主管拆解请求前（``self._subtasks`` 仍是旧计划、
        ``self._plan_version`` 仍是旧版本号）。

        Returns:
            ``(候选条目列表, 失败级联范围)``；级联范围为上一版失败
            子任务及其全部下游的 subtask_id 列表（含失败项自身）；
            无历史版本时两者均为空。
        """
        if self._plan_version < 1 or len(self._plan_history) < self._plan_version:
            return [], []
        previous_version = self._plan_version
        previous_specs = self._plan_history[previous_version - 1]
        failed_ids = {
            subtask_id
            for (version, subtask_id), result in self._archive_results.items()
            if version == previous_version and not result.success
        }
        cascade_scope = self._downstream_closure(failed_ids, previous_specs)

        candidates: list[dict[str, Any]] = []
        for spec in previous_specs:
            result = self._archive_results.get((previous_version, spec.subtask_id))
            if result is None:
                continue  # 未执行到：无结果可复用，也不构成失败语境
            entry: dict[str, Any] = {
                "plan_version": previous_version,
                "subtask_id": spec.subtask_id,
                "title": spec.title,
                "instructions": spec.instructions,
                "role": spec.role,
                "dependencies": list(spec.dependencies),
                "allowed_tools": list(spec.allowed_tools),
                "expected_output": spec.expected_output,
                "success": result.success,
                "output_summary": (result.output or "")[:500],
                "evidence": list(result.evidence)[:10],
                "artifacts": list(result.artifacts),
            }
            if not result.success:
                entry["failure_diagnosis"] = (result.error or "")[:500]
            candidates.append(entry)
        return candidates, sorted(cascade_scope)

    def _validate_reuse_declarations(self) -> dict[str, ExecutionResult]:
        """对新计划的显式 ``reuse_from`` 声明逐项安全校验。

        核心原则：LLM（主管）负责判断旧结果在新计划语境下是否可
        复用；代码不凭同名或 fingerprint 自动作语义决定，只校验声明
        是否满足安全硬约束。无声明时绝不复用，即使 id、名称与指纹
        完全相同。

        校验矩阵（全部通过才放行）：

        1. 来源存在：``reuse_from`` 必须是上一被采纳计划版本中存在
           且有归档结果的 subtask_id（``source_not_found``）；
        2. 来源成功：归档结果 ``success=True``（``source_not_successful``）；
        3. 非污染：来源不在上一版失败子任务的下游级联闭包内
           （``source_poisoned_by_failure``）；
        4. 指纹成分兼容：instructions / role / expected_output 完全
           一致，依赖集合经本轮合法复用声明形成的 id 映射后集合等价
           （``fingerprint_incompatible``；禁止文本相似度放宽）；
        5. 依赖成立：新任务依赖的旧对应者若留有成功结果，则该依赖
           必须同时复用通过（否则旧结果建立在被丢弃的旧成功结果
           之上，依赖链断裂）；旧对应无成功结果时允许其重新执行
           （``dependency_not_satisfied``）；
        6. artifacts 可信：本地路径必须位于沙箱范围内且无路径穿越、
           文件仍然存在；URL 等非本地引用跳过本地校验，纯文本伪装
           的产物路径无法通过存在性校验（``artifact_missing`` /
           ``artifact_outside_sandbox`` / ``artifact_path_traversal``）；
        7. 工具范围不扩大：新 spec ``allowed_tools`` 的 manifest 稳定
           标识集合必须 ⊆ 旧 spec（``tool_scope_expanded``）；
        8. 单一消费：同一旧结果不可被两个新任务重复引用
           （``duplicate_reference``，先完全通过者先消费）。

        校验通过：复制 ``ExecutionResult`` 到新 subtask_id（metadata
        写入 ``reused_from`` / ``reused_from_plan_version`` 来源标记，
        随序列化进入汇总与最终评审输入，评审不因复用而跳过），记录
        ``result_reused`` 事件。校验失败：记录 ``reuse_rejected`` 事件
        （机器可读 reason 码，不携带完整输出），新任务保持无结果
        状态 → 执行协调器的 pending 过滤自动将其放回正常执行队列
        （不静默成功）。

        固定点迭代：依赖映射与依赖成立校验依赖“同时复用”的其它
        声明，按“每轮用当前通过集重校验全部未通过声明”迭代至收敛
        （依赖链逐层解析：无依赖改名的一轮内先通过）。

        Returns:
            ``{新 subtask_id: 复制后的 ExecutionResult}``；无声明或
            全部被拒时为空字典。
        """
        declarations = [
            spec
            for spec in self._subtasks
            if spec.reuse_from and str(spec.reuse_from).strip()
        ]
        if not declarations:
            # 无显式声明：绝不复用（即使 id、名称与指纹完全相同）
            return {}

        task_id = self._task_request.task_id
        previous_version = self._plan_version - 1
        if 1 <= previous_version <= len(self._plan_history):
            previous_specs = self._plan_history[previous_version - 1]
        else:
            previous_specs = []
        previous_spec_map = {spec.subtask_id: spec for spec in previous_specs}

        # 上一版失败子任务及其下游污染闭包（按上一版依赖图计算）
        failed_ids = {
            subtask_id
            for (version, subtask_id), result in self._archive_results.items()
            if version == previous_version and not result.success
        }
        poisoned = self._downstream_closure(failed_ids, previous_specs)

        def _log_rejected(spec: SubtaskSpec, reason: str) -> None:
            # 机器可读 reason 码；不携带完整输出，防泄露
            self._events.log_event(
                EVENT_REUSE_REJECTED,
                {
                    "task_id": task_id,
                    "agent_id": DEFAULT_SUPERVISOR_ID,
                    "plan_version": self._plan_version,
                    "subtask_id": spec.subtask_id,
                    "source_subtask_id": str(spec.reuse_from).strip(),
                    "reason": reason,
                },
            )
            print(
                f"[so-agent] 复用声明被拒：[{spec.subtask_id}] ← "
                f"[{str(spec.reuse_from).strip()}]（原因 {reason}），"
                f"回退重新执行",
                file=sys.stderr,
                flush=True,
            )

        # 静态校验（不依赖其它声明）：来源/成功/污染/artifacts/工具范围
        pending: list[tuple[SubtaskSpec, SubtaskSpec, ExecutionResult]] = []
        for spec in declarations:
            source_id = str(spec.reuse_from).strip()
            old_spec = previous_spec_map.get(source_id)
            old_result = self._archive_results.get((previous_version, source_id))
            if old_spec is None or old_result is None:
                _log_rejected(spec, "source_not_found")
                continue
            if not old_result.success:
                _log_rejected(spec, "source_not_successful")
                continue
            if source_id in poisoned:
                _log_rejected(spec, "source_poisoned_by_failure")
                continue
            artifact_problem = self._validate_reuse_artifacts(old_result)
            if artifact_problem is not None:
                _log_rejected(spec, artifact_problem)
                continue
            if not self._reuse_tool_scope_compatible(spec, old_spec):
                _log_rejected(spec, "tool_scope_expanded")
                continue
            pending.append((spec, old_spec, old_result))

        # 固定点迭代：指纹兼容（依赖映射来自当前通过集）+ 依赖成立 +
        # 单一消费（先完全通过者先消费来源；重复引用检查后置，收敛后
        # 按"指纹 → 依赖 → 重复引用"优先级诊断根本原因）
        passed: dict[str, tuple[str, ExecutionResult]] = {}
        while True:
            id_map = {source: new_id for new_id, (source, _) in passed.items()}
            progressed = False
            for item in list(pending):
                spec, old_spec, old_result = item
                if not specs_fingerprint_equivalent(spec, old_spec, id_map):
                    continue  # 映射可能随其它声明通过而扩展，下一轮再试
                problem = self._reuse_dependencies_satisfied(
                    spec, id_map, previous_version, passed
                )
                if problem is not None:
                    continue  # 依赖的声明可能后续轮次通过，下一轮再试
                source_id = str(spec.reuse_from).strip()
                if source_id in {source for source, _ in passed.values()}:
                    continue  # 来源已被其它声明消费：收敛后按重复引用拒绝
                passed[spec.subtask_id] = (source_id, old_result)
                pending.remove(item)
                progressed = True
            if not progressed:
                break

        # 收敛后剩余声明：最终诊断并拒绝（机器可读 reason；先报根本
        # 原因——指纹不兼容 / 依赖不满足，均不成立时才报重复引用）
        final_id_map = {source: new_id for new_id, (source, _) in passed.items()}
        for spec, old_spec, old_result in pending:
            if not specs_fingerprint_equivalent(spec, old_spec, final_id_map):
                reason = "fingerprint_incompatible"
            else:
                reason = self._reuse_dependencies_satisfied(
                    spec, final_id_map, previous_version, passed
                ) or "duplicate_reference"
            _log_rejected(spec, reason)

        # 通过校验：复制结果到新 subtask_id（来源标记写入 metadata，
        # 随序列化进入汇总与最终评审输入）并记录 result_reused 事件
        reused: dict[str, ExecutionResult] = {}
        for new_id, (source_id, old_result) in passed.items():
            metadata = dict(old_result.metadata or {})
            metadata["reused_from"] = source_id
            metadata["reused_from_plan_version"] = previous_version
            reused[new_id] = old_result.model_copy(
                update={"subtask_id": new_id, "metadata": metadata}
            )
            self._events.log_event(
                EVENT_RESULT_REUSED,
                {
                    "task_id": task_id,
                    "agent_id": DEFAULT_SUPERVISOR_ID,
                    "plan_version": self._plan_version,
                    "subtask_id": new_id,
                    "source_subtask_id": source_id,
                    "source_plan_version": previous_version,
                },
            )
        return reused

    def _validate_reuse_artifacts(self, result: ExecutionResult) -> str | None:
        """校验复用结果的 artifacts（返回 None 表示通过，否则 reason 码）。

        规则（任务 35 安全硬约束）：

        - URL 等非本地引用（含 ``://``）不是本地产物路径，跳过本地
          路径校验（非本地证据不能伪装为产物路径也无需路径校验）；
        - 其余条目视为本地产物路径：相对路径按沙箱内相对路径解释、
          绝对路径直接解析，解析结果必须位于沙箱范围内（越界且含
          ``..`` 为 ``artifact_path_traversal``，越界无 ``..`` 为
          ``artifact_outside_sandbox``），且文件必须仍然存在
          （``artifact_missing``；纯文本等非路径内容伪装的产物路径
          无法通过存在性校验，同样落入此 reason）。
        """
        sandbox = Path(self._context.sandbox_dir).resolve()
        for artifact in result.artifacts:
            if not isinstance(artifact, str) or not artifact.strip():
                continue
            text = artifact.strip()
            if "://" in text:
                continue  # URL 等非本地引用：不作本地路径校验
            raw = Path(text)
            resolved = (raw if raw.is_absolute() else sandbox / raw).resolve()
            traversal = ".." in raw.parts
            try:
                inside = resolved.is_relative_to(sandbox)
            except (TypeError, ValueError):
                inside = False
            if not inside:
                return (
                    "artifact_path_traversal"
                    if traversal
                    else "artifact_outside_sandbox"
                )
            if not resolved.exists():
                return "artifact_missing"
        return None

    def _tool_stable_ids(self, names: list[str]) -> set[str]:
        """把工具名集合映射为 manifest 稳定标识集合。

        动态生成工具（已注册）用 ``generated:{tool_id}`` 标识——跨版本
        改名后仍可经 manifest 稳定标识对齐；其余工具（内置/预置/
        未注册缺口）用 ``tool:{name}`` 名字标识（名字本身即稳定
        标识，未注册名不产生稳定 id，只能按名字比对）。
        """
        stable: set[str] = set()
        for name in names:
            entry = self._tool_registry.get_tool(name)
            tool_id = entry.get("tool_id") if entry else None
            if tool_id:
                stable.add(f"generated:{tool_id}")
            else:
                stable.add(f"tool:{name}")
        return stable

    def _reuse_tool_scope_compatible(
        self, new_spec: SubtaskSpec, old_spec: SubtaskSpec
    ) -> bool:
        """新 spec 的工具范围不得扩大旧 spec（按 manifest 稳定标识）。"""
        return self._tool_stable_ids(new_spec.allowed_tools) <= self._tool_stable_ids(
            old_spec.allowed_tools
        )

    def _reuse_dependencies_satisfied(
        self,
        new_spec: SubtaskSpec,
        id_map: dict[str, str],
        previous_version: int,
        passed: dict[str, tuple[str, ExecutionResult]],
    ) -> str | None:
        """校验复用任务的新依赖是否成立（返回 None 表示满足）。

        指纹兼容已保证新依赖集合 == 旧依赖集合经 ``id_map`` 映射。
        对每个新依赖 ``dep``：找到其旧对应 ``old_dep``（同名直通或
        ``id_map`` 中的改名目标）；若上一版 ``old_dep`` 留有**成功**
        结果，则该结果只有经“同时复用”保留（``dep`` 在通过集中）才
        仍然成立——否则 ``dep`` 将重新执行，复用的旧结果建立在被
        丢弃的旧成功结果之上，依赖链断裂。旧对应无成功结果（失败
        /未执行）时 ``dep`` 本来就会（重新）执行，“仍会在执行前
        完成”，允许。
        """
        reverse_map = {new: old for old, new in id_map.items()}
        passed_new_ids = set(passed.keys())
        for dep in new_spec.dependencies:
            old_dep = reverse_map.get(dep, dep)
            old_dep_result = self._archive_results.get(
                (previous_version, old_dep)
            )
            if old_dep_result is not None and old_dep_result.success:
                if dep not in passed_new_ids:
                    return "dependency_not_satisfied"
        return None

    def _downstream_closure(
        self, seeds: set[str], specs: list[SubtaskSpec] | None = None
    ) -> set[str]:
        """计算给定计划中种子子任务的全部下游子任务（含种子自身）。

        依据 ``specs``（默认当前计划 ``self._subtasks``）的依赖边沿
        扇出方向做 BFS：失败子任务的下游结果受级联影响不可信，不得
        跨版本复用（任务 35 校验器显式传入上一版计划计算污染闭包）。
        """
        effective = self._subtasks if specs is None else specs
        dependents: dict[str, list[str]] = {}
        for spec in effective:
            for dep in spec.dependencies:
                dependents.setdefault(dep, []).append(spec.subtask_id)
        closure = set(seeds)
        stack = list(seeds)
        while stack:
            current = stack.pop()
            for child in dependents.get(current, []):
                if child not in closure:
                    closure.add(child)
                    stack.append(child)
        return closure

    def _record_plan_version(self, issue: TaskDecompositionIssue | None) -> None:
        """把当前采纳的计划版本快照（含 spec 指纹索引）写入结果存储。

        指纹索引供跨版本的结果复用审计与持久化实现（SQLite）复用；
        存储未配置（依赖容器未注入）时静默跳过。
        """
        result_store = self._result_store
        if result_store is None:
            return
        plan_version = PlanVersion(
            version=self._plan_version,
            specs=[spec.model_copy(deep=True) for spec in self._subtasks],
            fingerprints={
                spec.subtask_id: spec_fingerprint(spec) for spec in self._subtasks
            },
            trigger_issue=issue.model_copy(deep=True) if issue is not None else None,
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        result_store.save_plan_version(plan_version)

    def _completed_results_summary(self) -> list[dict[str, Any]]:
        """汇总当前计划下已成功子任务的结果摘要（重规划载荷参考用）。

        LLM（主管）参考摘要调整计划；结果复用由主管的显式
        ``reuse_from`` 声明驱动（候选信息见 ``_reuse_candidates_payload``），
        代码只做安全校验（``_validate_reuse_declarations``）；失败结果
        不进入摘要（避免误导，失败语境由候选载荷与级联范围提供）。
        """
        summary: list[dict[str, Any]] = []
        for spec in self._subtasks:
            result = self._results.get(spec.subtask_id)
            if result is None or not result.success:
                continue
            summary.append(
                {
                    "subtask_id": spec.subtask_id,
                    "title": spec.title,
                    "fingerprint": spec_fingerprint(spec),
                    "output": result.output,
                    "artifacts": list(result.artifacts),
                }
            )
        return summary

    # ------------------------------------------------------------------
    # 工具授权（候选 proposal 与真实签发分离）
    # ------------------------------------------------------------------
    def _propose_grants(self) -> list[ToolGrant]:
        """为当前计划生成工具授权候选（纯数据，不注册授权网关）。

        与 ``_issue_grants`` 的差别：候选仅作为两阶段计划评审中授权
        评审的输入（评审需要看到"预计授权工具"），不调用 ``issue_grant``、
        不写入网关登记表；grant_id 统一以 ``proposal-`` 前缀标识——
        即使候选凭证被传入工具包装器，``get_grant`` 也查无登记记录，
        三元组校验与 grant_id 绑定校验都会拦截。真实授权只在计划
        评审与授权评审均通过后由 ``_issue_grants`` 签发。本方法同时
        清空本版本的 ``plan_grants`` / ``plan_grant_map``（被拒计划不
        得残留可供工具包装器使用的有效签发记录）。

        动态工具（未注册 / 未批准）不进候选白名单（与 ``_issue_grants``
        同一资格过滤）：计划阶段仅在子任务 allowed_tools 中作"工具
        缺口提醒"，执行阶段创建成功后再补发真实授权。

        Returns:
            本版本新生成的 ToolGrant 候选列表。
        """
        task_id = self._task_request.task_id
        self._plan_grants = []
        self._plan_grant_map = {}
        proposals: list[ToolGrant] = []
        for spec in self._subtasks:
            eligible = [
                name
                for name in spec.allowed_tools
                if self._tool_registry.is_approved(name)
                and self._tool_registry.is_assignable(name)
            ]
            if not eligible:
                continue
            proposals.append(
                ToolGrant(
                    grant_id=f"proposal-{uuid.uuid4().hex[:12]}",
                    task_id=task_id,
                    agent_id=f"{task_id}:{spec.subtask_id}:grant-slot",
                    allowed_tools=eligible,
                    issued_by=DEFAULT_SUPERVISOR_ID,
                    expires_at=None,
                )
            )
        self._plan_grant_proposals = proposals
        return list(proposals)

    def _issue_grants(self) -> list[ToolGrant]:
        """为当前计划的子任务签发工具授权（无工具的子任务不签发空授权）。

        调用时机：两路计划评审（计划评审 + 授权评审）均通过后才真实
        签发（评审输入只看到 ``_propose_grants`` 生成的纯候选）；执行
        协调器在返工窗口重开（``_ensure_batch_grants_active``）与工具
        缺口解决（``_resolve_tool_gaps``）时经 deps 通道补发。

        只对"已注册且已批准"的可分配工具签发：工具缺口（未注册 / 未批准）
        不进白名单，待执行阶段缺口流程解决后由本方法补发（重新签发时
        覆盖本版本计划的映射，旧授权不主动撤销）。

        复用任务过滤（任务 35 / 31 协调）：复用任务的候选授权已随
        ``_propose_grants`` 参与最小化计划评审（评审可见性保留），
        但其结果已直接并入 ``self._results`` 且不再执行——本方法在
        计划评审通过后、真实签发前跳过它们，避免为已复用任务留下
        无用途的活跃 grant；真实授权只供实际 pending 任务使用。

        每份授权固定携带 ``expires_at``（当前时间 + ``max_pipeline_timeout``）
        作为授权生命周期的第二道防线：即使执行协调器的批次后回收（主动
        撤销）因异常路径遗漏，授权也会随流水线超时上限自动失效，不会
        永久有效。

        授权在规划阶段按"子任务槽位"签发（agent_id 形如
        ``{task_id}:{subtask_id}:grant-slot`` 的挂载点标识）；实际执行
        Agent（原始/替代/动态）只能在该授权范围内获得工具注入，且授权
        不可由子 Agent 转授（issued_by 固定为主管标识）。

        Returns:
            本版本新签发的 ToolGrant 列表。
        """
        task_id = self._task_request.task_id
        self._plan_grants = []
        self._plan_grant_map = {}
        expires_at = _utc_now() + timedelta(
            seconds=float(self._settings.max_pipeline_timeout)
        )
        for spec in self._subtasks:
            # 复用任务（结果已并入 _results 且带来源标记）不重新执行：
            # 不为其真实签发授权（候选已随 _propose_grants 参与计划评审，
            # 评审可见性保留）。注意失败待重试任务不在此列：其结果
            # metadata 无来源标记，返工窗口重开与工具缺口补发的签发
            # 不受影响。
            existing = self._results.get(spec.subtask_id)
            if existing is not None and existing.metadata.get("reused_from"):
                continue
            eligible = [
                name
                for name in spec.allowed_tools
                if self._tool_registry.is_approved(name)
                and self._tool_registry.is_assignable(name)
            ]
            if not eligible:
                continue
            grant = self._gateway.issue_grant(
                task_id=task_id,
                agent_id=f"{task_id}:{spec.subtask_id}:grant-slot",
                allowed_tools=eligible,
                issued_by=DEFAULT_SUPERVISOR_ID,
                expires_at=expires_at,
            )
            self._plan_grants.append(grant)
            self._plan_grant_map[spec.subtask_id] = grant
        self._grants.extend(self._plan_grants)
        return list(self._plan_grants)

    # ------------------------------------------------------------------
    # 计划评审（计划评审 + 授权评审并行合并）
    # ------------------------------------------------------------------
    async def _review_plan(self) -> ReviewDecision:
        """计划评审 + 授权评审：并行调用评审 Agent 后合并结论。

        两份 ReviewInput 均附带 ``available_tools``（当前已批准可分配的
        工具名称清单，与规划载荷同源），供评审 Agent 核对子任务声明的
        工具是否越出环境能力范围。授权评审的 ``tool_grants`` 传入的是
        纯候选（``_plan_grant_proposals``，不注册网关）——评审能看到
        预计授权工具，但候选不具备任何放行能力；真实签发在合并结论
        ``passed=True`` 之后由 ``_plan_and_review`` 执行。
        """
        self._transition(TaskStatus.PLAN_REVIEW)
        task_request = self._task_request
        subtask_snapshot = [spec.model_copy(deep=True) for spec in self._subtasks]
        available_tools = [
            item["name"]
            for item in self._tool_registry.list_assignable_tools(approved_only=True)
        ]

        plan_decision, authorization_decision = await asyncio.gather(
            self._call_review_agent(
                ReviewInput(
                    stage=ReviewStage.PLAN_REVIEW,
                    task_request=task_request,
                    subtasks=subtask_snapshot,
                    available_tools=list(available_tools),
                    plan_version=self._plan_version,
                )
            ),
            self._call_review_agent(
                ReviewInput(
                    stage=ReviewStage.AUTHORIZATION_REVIEW,
                    task_request=task_request,
                    subtasks=subtask_snapshot,
                    tool_grants=list(self._plan_grant_proposals),
                    available_tools=list(available_tools),
                    plan_version=self._plan_version,
                )
            ),
        )
        combined = self._combine_reviews(plan_decision, authorization_decision)
        self._last_review = combined
        return combined

    @staticmethod
    def _combine_reviews(
        plan: ReviewDecision, authorization: ReviewDecision
    ) -> ReviewDecision:
        """合并计划评审与授权评审的结论（两者均通过才算通过）。

        任务 32 契约：合并只读取两路合法 ``decision.passed``，不做任何
        issues / required_fixes 文本二次推断；上游 ``_call_review_agent``
        已保证输入结论满足字段契约（passed=false 时 required_fixes
        非空），因此合并后契约自动自洽：任一路未通过即合并未通过，
        未通过方的 required_fixes 已非空。非阻断建议（suggestions）
        按序合并，不参与阻断判定。
        """
        return ReviewDecision(
            stage="plan_review",
            passed=plan.passed and authorization.passed,
            issues=list(plan.issues) + list(authorization.issues),
            suggestions=list(plan.suggestions) + list(authorization.suggestions),
            required_fixes=list(plan.required_fixes) + list(authorization.required_fixes),
            retry_target="; ".join(
                t for t in (plan.retry_target, authorization.retry_target) if t
            ) or None,
            summary=" | ".join(
                text for text in (plan.summary, authorization.summary) if text
            ),
        )

    # ------------------------------------------------------------------
    # 治理额度与快速通道判定
    # ------------------------------------------------------------------
    def _try_consume_replan_quota(self) -> bool:
        """尝试消耗一次任务拆分修改额度。

        Returns:
            True 表示额度充足并已消耗；False 表示已耗尽（应人工兜底）。
        """
        if self._replan_count >= int(self._settings.max_replan_attempts):
            return False
        self._replan_count += 1
        return True

    def _is_simple_plan(self) -> bool:
        """规划后确认：子任务数 <= 2 且无复杂依赖（快速通道后置判断）。"""
        if len(self._subtasks) > 2:
            return False
        return all(not spec.dependencies for spec in self._subtasks)

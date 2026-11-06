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

能力分组：结果复用（planning_reuse.PlanningReuseMixin）、工具授权
（planning_grants.PlanningGrantMixin）与计划评审
（planning_review.PlanningReviewMixin）位于独立模块；本模块保留任务
接收、拆解循环与治理额度。
"""

from __future__ import annotations

import re
import uuid
from typing import Any

import openai

from so_agent.models import (
    ReviewDecision,
    SubtaskSpec,
    TaskDecompositionIssue,
    TaskRequest,
    TaskStatus,
    VerificationRule,
)
from so_agent.orchestrator import OrchestratorInput, PlanningOutput
from so_agent.planning_grants import PlanningGrantMixin
from so_agent.planning_reuse import PlanningReuseMixin
from so_agent.planning_review import PlanningReviewMixin
from so_agent.runtime.events import (
    EVENT_PARALLELISM_DECIDED,
    EVENT_PLAN_STEPS_MERGED,
    EVENT_PLAN_TOOL_SCOPE_REPAIRED,
    EVENT_REPLAN,
    EVENT_TASK_CREATED,
    EVENT_VERIFICATION_RULE_REJECTED,
)
from so_agent.runtime.logging_setup import get_logger
from so_agent.runtime.parallelism import (
    ParallelismDecision,
    RuntimeBreaker,
    decide_parallelism,
)
from so_agent.runtime.registry import DEFAULT_SUPERVISOR_ID
from so_agent.state_machine import SupervisorNetworkError, WorkflowError
from so_agent.workflow_support import BaseCoordinator

logger = get_logger("planning")

# 主管 Agent 网络异常类型：请求超时与连接失败。区别于计划结构问题——
# 网络失败不计入"任务拆分修改额度"，按独立预算重试（见 _plan_and_review）
_SUPERVISOR_NETWORK_ERRORS: tuple[type[Exception], ...] = (
    openai.APITimeoutError,
    openai.APIConnectionError,
)

# 网络重试预算（独立于任务拆分修改额度）：主管 Agent 调用连续网络失败
# （超时 / 连接错误）超过该次数时转入人工兜底（网络连续失败）
_MAX_NETWORK_RETRIES: int = 3

# 工具意图推断：只匹配明确要求执行/写入/读取的动作，避免把“运行结果”
# 这类名词性表述误判为需要执行工具。
_TOOL_INTENT_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("run_in_sandbox", re.compile(r"run_in_sandbox")),
    ("run_in_sandbox", re.compile(r"python\s+[^\s，。;；]+\.py", re.IGNORECASE)),
    (
        "run_in_sandbox",
        re.compile(r"(执行|运行|跑)\s*(一下|一次)?\s*(脚本|代码|程序)"),
    ),
    (
        "run_in_sandbox",
        re.compile(
            r"\b(run|execute)\b[^\n。]{0,24}\b(script|code|program|python)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "write_file",
        re.compile(r"(创建|写入|生成|保存)\s*[^\n。]{0,16}\.(py|md|json|csv|txt)"),
    ),
    (
        "read_file",
        re.compile(r"读取\s*[^\n。]{0,16}\.(py|md|json|csv|txt|log)"),
    ),
)


# 计划内步骤合并的判定常量：只有下游声明写报告类文件、上游确实写盘、
# 且下游指令显式引用上游产物时才合并。
_REPORT_SUFFIXES: tuple[str, ...] = (".md", ".markdown")
_WRITE_TOOL_NAMES: frozenset[str] = frozenset(
    {"write_file", "write_and_run", "append_file"}
)
# 执行器族：role 决定子任务由哪类 Agent 执行，跨族合并会改变路由，
# 因此只在同族内合并（与 runtime.scheduler._ROLE_TYPE_MAP 的分组一致）
_EXECUTOR_FAMILIES: dict[str, str] = {
    "code": "code_agent",
    "code_agent": "code_agent",
    "review": "review_agent",
    "review_agent": "review_agent",
    "creator": "subagent_creator",
    "subagent_creator": "subagent_creator",
}


def _executor_family(role: str) -> str:
    """把 role 归一为执行器族；未知 role 一律按动态子 Agent 处理。"""
    return _EXECUTOR_FAMILIES.get((role or "").strip().lower(), "dynamic")


def _path_basename(entry: str) -> str:
    """取沙箱内相对路径的文件名部分（兼容 Windows 分隔符）。"""
    return (entry or "").strip().replace("\\", "/").rstrip("/").split("/")[-1]


def _is_report_path(entry: str) -> bool:
    """判断写入路径是否为报告类文档。"""
    name = _path_basename(entry).lower()
    return any(name.endswith(suffix) for suffix in _REPORT_SUFFIXES)


def _merged_unique(first: list[str], second: list[str]) -> list[str]:
    """合并两个名称列表：去空、去重、保持先出现顺序。"""
    merged: list[str] = []
    for name in [*first, *second]:
        if name and name not in merged:
            merged.append(name)
    return merged


def compact_available_tools(
    tools: list[dict[str, Any]], description_limit: int
) -> list[dict[str, Any]]:
    """压缩主管规划载荷中的工具清单。

    清单里的工具均已批准且可分配，``source`` / ``approved`` 对拆解决策
    没有增量信息；描述只用于判断是否适合分配，无需完整正文。
    ``description_limit <= 0`` 表示不截断描述。
    """
    compact: list[dict[str, Any]] = []
    for item in tools:
        name = str(item.get("name") or "").strip()
        if not name:
            continue
        description = str(item.get("description") or "").strip()
        if 0 < description_limit < len(description):
            description = description[:description_limit].rstrip() + "…"
        compact.append({"name": name, "description": description})
    return compact


def _verification_rule_problem(rule: VerificationRule) -> str | None:
    """返回规则无法客观判定的原因；规则可执行时返回 None。"""
    if rule.kind in ("file_exists", "file_non_empty"):
        return None if rule.path else "missing_path"
    if rule.kind == "text_contains":
        if not rule.text:
            return "missing_text"
        if not rule.path:
            return "text_contains_without_path"
        return None
    if rule.kind in ("json_field", "numeric_match"):
        if not rule.field:
            return "missing_field"
        if rule.kind == "numeric_match" and rule.value is None:
            return "missing_value"
        return None
    return None


def _verification_rule_summary(rule: VerificationRule) -> dict[str, Any]:
    """规则的最小审计摘要：文本截断，不把长内容写进事件日志。"""
    text = (rule.text or "").strip()
    if len(text) > 80:
        text = text[:80] + "…"
    return {
        "kind": rule.kind,
        "path": rule.path,
        "field": rule.field,
        "text": text or None,
    }


class PlanningCoordinator(
    PlanningReuseMixin,
    PlanningGrantMixin,
    PlanningReviewMixin,
    BaseCoordinator,
):
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
                logger.warning("网络请求失败（%s），正在重试...", error_name)
                logger.info(
                    "网络重试 %s/%s...", network_retry_count, _MAX_NETWORK_RETRIES
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
                    logger.info(
                        "复用 %s 个子任务的既有成功结果（%s），跳过重新执行",
                        len(reused),
                        ", ".join(sorted(reused)),
                    )
                self._record_plan_version(issue)
                # 并行度决策（代码判定，模型不可干预）：在评审前留痕，
                # 使"为什么没有并行"成为可审计事实
                self._decide_parallelism()
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
                # 结构硬校验先行：非法结构不浪费模型评审调用，
                # 直接作为未通过结论进入重新拆解
                structural = self._validate_plan_structure()
                if structural is not None:
                    decision = structural
                elif self._should_skip_plan_review():
                    decision = self._deterministic_plan_review()
                else:
                    decision = await self._review_plan()
                if decision.passed:
                    # 停止门（任务 45）：评审等待期间已停止则不签发授权
                    #（所有出口撤销授权不变，此处拒绝新增签发）
                    self._gate_check("issue_grants")
                    # 两路评审（计划评审 + 授权评审）均通过：此刻才真实
                    # 签发 ToolGrant（被拒计划在网关无任何签发记录）
                    self._issue_grants()
                    logger.info("计划评审通过（版本 %s）", self._plan_version)
                    self._report_progress("plan_accepted", checkpoint=True)
                    return True
                logger.warning(
                    "计划评审未通过：%s",
                    decision.summary or "；".join(decision.issues) or "未说明原因",
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
                logger.info(
                    "第 %s 次重规划（拆分修改额度 %s/%s）...",
                    self._replan_count,
                    self._replan_count,
                    max_attempts,
                )

    def _print_plan_summary(self) -> None:
        """打印当前计划的拆解结果详情（版本号 + 子任务清单，stderr）。"""
        logger.info(
            "拆解完成（版本 %s），共 %s 个子任务：",
            self._plan_version,
            len(self._subtasks),
        )
        for spec in self._subtasks:
            tools_str = ", ".join(spec.allowed_tools) if spec.allowed_tools else "无"
            logger.info(
                "  - [%s] %s\n    角色: %s | 工具: %s",
                spec.subtask_id,
                spec.title,
                spec.role,
                tools_str,
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
        available_tools = self._tool_registry.list_assignable_tools(
            approved_only=True
        )
        if bool(getattr(self._settings, "supervisor_context_slimming", True)):
            available_tools = compact_available_tools(
                available_tools,
                int(
                    getattr(
                        self._settings,
                        "supervisor_tool_description_max_chars",
                        160,
                    )
                ),
            )
        payload: dict[str, Any] = {
            "phase": "replanning" if self._plan_version > 0 else "planning",
            "task_id": self._task_request.task_id,
            "orchestrator_input": orchestrator_input.model_dump(mode="json"),
            "plan_version": self._plan_version + 1,
            "replan_count": self._replan_count,
            "max_replan_attempts": int(self._settings.max_replan_attempts),
            "available_tools": available_tools,
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
        return self._compress_context(payload)

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
          因依赖非法而拒绝整个计划）；
        - 满足确定性条件的“产物步骤 → 报告步骤”合并为一个子任务
          （见 ``_coalesce_report_steps``），其余计划保持原样；
        - ``verification`` 丢弃无法客观判定或缺少必填字段的规则并留痕
          （见 ``_lint_verification_rules``），再为写入路径补齐最小规则。
        """
        valid_ids = {(spec.subtask_id or "").strip() for spec in subtasks}
        normalized: list[SubtaskSpec] = []
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
            normalized.append(spec)

        adopted = self._coalesce_report_steps(normalized)
        for spec in adopted:
            self._lint_tool_scope(spec)
            self._lint_verification_rules(spec)
            self._ensure_write_verification(spec)
        return adopted

    def _coalesce_report_steps(
        self, specs: list[SubtaskSpec]
    ) -> list[SubtaskSpec]:
        """把“产生产物 → 基于产物写报告”的单依赖两步合并为一次执行。

        报告步骤单独调用时必须重新读取上一步的产物，跨调用重复读取
        是模型输入 token 的主要来源之一。只合并确定性可判定的组合：
        下游仅依赖上游一项、上游确实写盘且其产物被下游指令显式引用、
        两步由同一执行器族承担；任何条件不满足都保持原计划交给评审。
        合并数受 ``plan_merge_report_max_pairs`` 限制（0 表示关闭）。
        """
        self._state.plan_steps_merged = 0
        self._state.merged_subtask_ids = set()
        budget = int(getattr(self._settings, "plan_merge_report_max_pairs", 0) or 0)
        if budget <= 0 or len(specs) < 2:
            return specs

        by_id = {spec.subtask_id: spec for spec in specs}
        # 被其它子任务声明复用的 id 不能消失：合并会让 reuse_from 指向
        # 不存在的来源，校验器只能拒绝并重跑，代价高于少合并一次。
        reuse_targets = {spec.reuse_from for spec in specs if spec.reuse_from}
        absorbed: dict[str, str] = {}
        for consumer in specs:
            if len(absorbed) >= budget:
                break
            if consumer.subtask_id in absorbed:
                continue
            if consumer.subtask_id in reuse_targets:
                continue
            deps = [dep for dep in consumer.dependencies if dep in by_id]
            if len(deps) != 1:
                continue
            producer = by_id[deps[0]]
            if producer is consumer or producer.subtask_id in absorbed:
                continue
            if not self._mergeable_report_pair(producer, consumer):
                continue
            self._merge_report_pair(producer, consumer)
            absorbed[consumer.subtask_id] = producer.subtask_id

        if not absorbed:
            return specs

        for spec in specs:
            if spec.subtask_id in absorbed:
                continue
            rewritten: list[str] = []
            for dep in spec.dependencies:
                target = absorbed.get(dep, dep)
                if target not in rewritten:
                    rewritten.append(target)
            spec.dependencies = rewritten
        self._state.plan_steps_merged = len(absorbed)
        return [spec for spec in specs if spec.subtask_id not in absorbed]

    def _mergeable_report_pair(
        self, producer: SubtaskSpec, consumer: SubtaskSpec
    ) -> bool:
        """判断“上游产物 + 下游报告”是否可安全合并（纯语义约束）。"""
        if producer.reuse_from or consumer.reuse_from:
            return False
        if not producer.write_scope:
            return False
        if not any(_is_report_path(entry) for entry in consumer.write_scope):
            return False
        if not self._declares_write(producer) or not self._declares_write(consumer):
            return False
        if _executor_family(producer.role) != _executor_family(consumer.role):
            return False
        text = f"{consumer.instructions}\n{consumer.expected_output}".lower()
        return any(
            _path_basename(entry).lower() in text
            for entry in producer.write_scope
            if _path_basename(entry)
        )

    @staticmethod
    def _declares_write(spec: SubtaskSpec) -> bool:
        """allowed_tools 为空表示未声明（按标准工具处理），否则必须含写入工具。"""
        if not spec.allowed_tools:
            return True
        return any(name in _WRITE_TOOL_NAMES for name in spec.allowed_tools)

    def _merge_report_pair(
        self, producer: SubtaskSpec, consumer: SubtaskSpec
    ) -> None:
        """把下游报告步骤并入上游：保留上游 id/位置，改写指令与范围。"""
        artifacts = [entry for entry in producer.write_scope if entry]
        producer.title = f"{producer.title}；{consumer.title}"
        producer.instructions = (
            f"{producer.instructions}\n\n"
            f"【同一次执行内继续】{consumer.instructions}\n"
            f"（{'、'.join(artifacts)} 已由本次调用前面的步骤生成，"
            "可直接使用或按需读取，无需重新生成）"
        )
        producer.allowed_tools = _merged_unique(
            producer.allowed_tools, consumer.allowed_tools
        )
        producer.write_scope = _merged_unique(
            producer.write_scope, consumer.write_scope
        )
        if consumer.expected_output.strip():
            producer.expected_output = (
                f"{producer.expected_output}\n{consumer.expected_output}"
            ).strip()
        producer.verification = producer.verification + consumer.verification
        producer.max_attempts = max(producer.max_attempts, consumer.max_attempts)
        self._state.merged_subtask_ids.add(producer.subtask_id)
        self._events.log_event(
            EVENT_PLAN_STEPS_MERGED,
            {
                "task_id": self._task_request.task_id,
                "agent_id": DEFAULT_SUPERVISOR_ID,
                "plan_version": self._plan_version,
                "merged_subtask_id": producer.subtask_id,
                "absorbed_subtask_id": consumer.subtask_id,
                "artifacts": artifacts,
            },
        )

    def _lint_tool_scope(self, spec: SubtaskSpec) -> None:
        """补齐 instructions 明确要求、但 allowed_tools 漏写的标准内置工具。

        计划若写了“运行脚本 / 执行代码 / python xxx.py”却没给
        ``run_in_sandbox``，执行期再补授权会被评审判为越权并触发重试；
        在采纳阶段确定性补齐（只补标准内置工具，不补动态/外部工具）并
        留痕，比让计划带缺陷进入执行更便宜。``allowed_tools`` 为空的
        计划不做补齐：空列表本身表示“未声明”，路由按标准工具处理。
        """
        if not bool(getattr(self._settings, "plan_tool_scope_lint", True)):
            return
        if not spec.allowed_tools:
            return
        text = f"{spec.instructions}\n{spec.expected_output}"
        missing: list[str] = []
        for tool, pattern in _TOOL_INTENT_PATTERNS:
            if tool in spec.allowed_tools or tool in missing:
                continue
            if pattern.search(text):
                missing.append(tool)
        if not missing:
            return
        spec.allowed_tools = list(spec.allowed_tools) + missing
        self._events.log_event(
            EVENT_PLAN_TOOL_SCOPE_REPAIRED,
            {
                "task_id": self._task_request.task_id,
                "agent_id": DEFAULT_SUPERVISOR_ID,
                "plan_version": self._plan_version,
                "subtask_id": spec.subtask_id,
                "added_tools": missing,
            },
        )

    def _lint_verification_rules(self, spec: SubtaskSpec) -> None:
        """采纳计划时清理无法客观判定的验收规则并留痕。

        无 ``path`` 的 ``text_contains`` 只能匹配模型自述措辞：产物正确
        但说法不同就会误判失败并触发修复循环。此类规则以及缺少必填字
        段的规则在采纳阶段丢弃，计划其余部分保持不变。
        """
        kept: list[VerificationRule] = []
        rejected: list[dict[str, Any]] = []
        seen: set[tuple[Any, ...]] = set()
        for rule in spec.verification:
            problem = _verification_rule_problem(rule)
            if problem is not None:
                rejected.append(
                    {"rule": _verification_rule_summary(rule), "reason": problem}
                )
                continue
            marker = (
                rule.kind,
                rule.path,
                rule.field,
                repr(rule.equals),
                rule.value,
                rule.text,
            )
            if marker in seen:
                continue
            seen.add(marker)
            kept.append(rule)
        spec.verification = kept
        if not rejected:
            return
        self._events.log_event(
            EVENT_VERIFICATION_RULE_REJECTED,
            {
                "task_id": self._task_request.task_id,
                "agent_id": DEFAULT_SUPERVISOR_ID,
                "plan_version": self._plan_version,
                "subtask_id": spec.subtask_id,
                "rejected": rejected,
            },
        )

    @staticmethod
    def _ensure_write_verification(spec: SubtaskSpec) -> None:
        """为声明写入的子任务补齐最小机检验收规则。

        规划器漏写 `verification` 时，控制层至少保证声明的写入路径
        真实落地且非空；这层兜底与 `write_scope` 后验互补，不替代
        主管针对内容追加的 `text_contains` 等规则。
        """
        if not spec.write_scope:
            return
        existing = {
            (rule.kind, rule.path)
            for rule in spec.verification
            if rule.path is not None
        }
        for raw_entry in spec.write_scope:
            entry = (raw_entry or "").strip().replace("\\", "/")
            if (
                not entry
                or entry.startswith("/")
                or ".." in entry.split("/")
                or any(marker in entry for marker in ("*", "?", "["))
            ):
                continue
            if ("file_exists", entry) not in existing:
                spec.verification.append(
                    VerificationRule(
                        kind="file_exists",
                        path=entry,
                        description="控制层自动补齐：写入路径必须存在",
                    )
                )
                existing.add(("file_exists", entry))
            basename = entry.rsplit("/", 1)[-1]
            looks_like_file = "." in basename and not basename.startswith(".")
            if looks_like_file and ("file_non_empty", entry) not in existing:
                spec.verification.append(
                    VerificationRule(
                        kind="file_non_empty",
                        path=entry,
                        description="控制层自动补齐：写入文件必须非空",
                    )
                )
                existing.add(("file_non_empty", entry))

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
        """规划后确认：子任务数 <= 2 且无复杂依赖（快速通道后置判断）。

        步骤合并会把原本依赖串联的两步折成一个无依赖子任务，但合并不
        降低验收/汇总要求，因此合并过的计划一律不走简单计划快速通道。
        """
        if self._state.plan_steps_merged > 0:
            return False
        if len(self._subtasks) > 2:
            return False
        return all(not spec.dependencies for spec in self._subtasks)

    def _decide_parallelism(self) -> ParallelismDecision:
        """按计划形状做确定性并行度决策，并为本版本建立新的熔断器。

        决策只依赖子任务依赖关系与工具声明，不读取模型自由文本；
        每次采纳新计划都重置熔断器——重规划意味着任务结构已变化，
        旧版本的失败压力不应继续限制新版本。
        """
        decision = decide_parallelism(
            self._subtasks,
            max_concurrency=int(self._settings.max_concurrency),
            objective=self._settings.optimization_objective,
            net_gain_threshold=float(
                self._settings.parallelism_net_gain_threshold
            ),
            min_speedup=float(self._settings.parallelism_min_speedup),
        )
        self._parallelism_decision = decision
        self._runtime_breaker = RuntimeBreaker()
        self._events.log_event(
            EVENT_PARALLELISM_DECIDED,
            {
                "task_id": self._task_request.task_id,
                "agent_id": DEFAULT_SUPERVISOR_ID,
                "plan_version": self._plan_version,
                **decision.as_event_payload(),
            },
        )
        logger.info(
            "并行度决策（版本 %s，目标 %s）：%s，并发 %s（%s）",
            self._plan_version,
            decision.objective,
            decision.mode,
            decision.concurrency,
            decision.reason,
        )
        return decision

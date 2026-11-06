"""计划评审与授权评审（PlanningCoordinator 的评审能力）。

从 planning_coordinator 拆分（行为不变）：简单计划的确定性评审、
计划/授权两路并行评审与结论合并。
"""

from __future__ import annotations

import asyncio

from so_agent.models import ReviewDecision, TaskStatus
from so_agent.runtime.events import (
    EVENT_PLAN_VALIDATION_FAILED,
    EVENT_REVIEW_SKIPPED,
)
from so_agent.runtime.logging_setup import get_logger
from so_agent.runtime.registry import DEFAULT_SUPERVISOR_ID
from so_agent.tool_packages.review_agent.agent import ReviewInput, ReviewStage

logger = get_logger("planning")

# 与评审 Agent 的授权检查保持同一口径：这些工具名过于宽泛/高风险，
# 命中时不得由确定性校验替代模型授权评审。
_SUSPICIOUS_TOOL_NAMES: frozenset[str] = frozenset(
    {"*", "all", "any", "any_tool", "shell", "exec", "sudo"}
)


def _cyclic_subtask_ids(specs: list) -> list[str]:
    """用 Kahn 拓扑排序找出无法解析的 subtask_id（含环成员）。"""
    ids = [
        (spec.subtask_id or "").strip()
        for spec in specs
        if (spec.subtask_id or "").strip()
    ]
    id_set = set(ids)
    indegree = {sid: 0 for sid in ids}
    dependents: dict[str, list[str]] = {sid: [] for sid in ids}
    for spec in specs:
        sid = (spec.subtask_id or "").strip()
        if not sid:
            continue
        for dep in spec.dependencies:
            if dep in id_set and dep != sid:
                indegree[sid] += 1
                dependents[dep].append(sid)
    queue = [sid for sid in ids if indegree[sid] == 0]
    resolved: set[str] = set()
    while queue:
        current = queue.pop(0)
        resolved.add(current)
        for nxt in dependents[current]:
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                queue.append(nxt)
    return sorted(sid for sid in ids if sid not in resolved)


def _unsafe_write_entry(entry: str) -> bool:
    """判断写入声明是否为沙箱外路径（绝对路径或包含 ``..``）。"""
    text = (entry or "").strip().replace("\\", "/")
    if not text:
        return False
    if text.startswith("/") or (len(text) >= 2 and text[1] == ":"):
        return True
    return ".." in text.split("/")


class PlanningReviewMixin:
    """计划评审、授权评审与确定性快速评审。"""

    def _validate_plan_structure(self) -> ReviewDecision | None:
        """在模型评审前做计划结构的确定性校验。

        只拦截会破坏执行安全与调度正确性的硬结构问题：空/重复
        subtask_id、自依赖、依赖缺失、依赖成环、越界写入路径
        （绝对路径或含 ``..``）。语义质量（粒度、验收覆盖）仍交给
        模型评审；结构合法时返回 None。
        """
        problems: list[str] = []
        required_fixes: list[str] = []
        ids = [(spec.subtask_id or "").strip() for spec in self._subtasks]
        if not self._subtasks:
            problems.append("子任务清单为空")
            required_fixes.append("重新拆解并输出至少一个子任务")
        empty_positions = [index for index, sid in enumerate(ids) if not sid]
        if empty_positions:
            problems.append(f"存在 subtask_id 为空的子任务（位置 {empty_positions}）")
            required_fixes.append("为每个子任务填写唯一且非空的 subtask_id")
        id_set = {sid for sid in ids if sid}
        duplicates = sorted({sid for sid in ids if sid and ids.count(sid) > 1})
        if duplicates:
            problems.append(f"存在重复的 subtask_id：{duplicates}")
            required_fixes.append("合并或重命名重复的 subtask_id")
        self_dependencies = sorted(
            (spec.subtask_id or "").strip()
            for spec in self._subtasks
            if (spec.subtask_id or "").strip()
            and (spec.subtask_id or "").strip() in spec.dependencies
        )
        if self_dependencies:
            problems.append(f"存在自依赖的 subtask_id：{self_dependencies}")
            required_fixes.append("移除子任务对自身的依赖")
        missing_dependencies = [
            {
                "subtask_id": (spec.subtask_id or "").strip(),
                "missing": sorted(dep for dep in spec.dependencies if dep not in id_set),
            }
            for spec in self._subtasks
            if any(dep not in id_set for dep in spec.dependencies)
        ]
        if missing_dependencies:
            problems.append(
                "存在依赖不存在的子任务：" + repr(missing_dependencies)
            )
            required_fixes.append("修正依赖的 subtask_id 或补充缺失子任务")
        cyclic_ids = _cyclic_subtask_ids(self._subtasks)
        if cyclic_ids:
            problems.append(f"依赖图存在环：{cyclic_ids}")
            required_fixes.append("调整依赖关系以消除环")
        unsafe_writes = sorted(
            {
                entry
                for spec in self._subtasks
                for entry in spec.write_scope
                if _unsafe_write_entry(entry)
            }
        )
        if unsafe_writes:
            problems.append(f"存在越界写入路径（绝对路径或包含 ..）：{unsafe_writes}")
            required_fixes.append("写入范围只能使用沙箱内的相对路径")
        if not problems:
            return None

        decision = ReviewDecision(
            stage="plan_review",
            passed=False,
            issues=problems,
            required_fixes=required_fixes,
            summary="计划结构确定性校验未通过（未进入模型评审）",
        )
        self._last_review = decision
        self._events.log_event(
            EVENT_PLAN_VALIDATION_FAILED,
            {
                "task_id": self._task_request.task_id,
                "agent_id": DEFAULT_SUPERVISOR_ID,
                "plan_version": self._plan_version,
                "problems": problems,
            },
        )
        logger.warning("计划结构确定性校验未通过：%s", "；".join(problems))
        return decision

    def _authorization_review_is_deterministic(self) -> bool:
        """授权候选能否由确定性校验替代模型授权评审。

        条件（全部满足才替代）：``authorization_review_mode=risk_based``
        且每份候选都满足——非空、工具名不宽泛、工具都在计划声明范围内、
        都已批准且可分配、且都在对应子任务自己的声明范围内。候选由
        控制层从声明机械派生，满足这些条件时"是否超额授权"无需模型判断；
        工具必要性的语义判断由计划评审负责。
        """
        mode = str(
            getattr(self._settings, "authorization_review_mode", "risk_based")
            or "risk_based"
        ).strip().lower()
        if mode != "risk_based":
            return False
        proposals = list(getattr(self, "_plan_grant_proposals", None) or [])
        if not proposals:
            return False
        specs = {
            (spec.subtask_id or "").strip(): spec for spec in self._subtasks
        }
        declared = {
            tool.strip()
            for spec in self._subtasks
            for tool in spec.allowed_tools
            if tool.strip()
        }
        for grant in proposals:
            tools = [tool.strip() for tool in grant.allowed_tools if tool.strip()]
            if not tools or len(tools) != len(grant.allowed_tools):
                return False
            if any(tool.lower() in _SUSPICIOUS_TOOL_NAMES for tool in tools):
                return False
            if any(tool not in declared for tool in tools):
                return False
            if any(
                not self._tool_registry.is_approved(tool)
                or not self._tool_registry.is_assignable(tool)
                for tool in tools
            ):
                return False
            slot = grant.agent_id.split(":")[-2] if ":" in grant.agent_id else ""
            spec = specs.get(slot)
            if spec is None:
                return False
            declared_here = {
                tool.strip() for tool in spec.allowed_tools if tool.strip()
            }
            if any(tool not in declared_here for tool in tools):
                return False
        return True

    def _deterministic_authorization_review(self) -> ReviewDecision:
        """记录确定性授权评审结论并跳过模型授权评审调用。"""
        decision = ReviewDecision(
            stage="plan_review",
            passed=True,
            summary=(
                "授权候选通过确定性最小化校验（均为对应子任务声明工具的子集、"
                "已批准且可分配、无空授权与宽泛工具名），跳过模型授权评审"
            ),
        )
        self._events.log_event(
            EVENT_REVIEW_SKIPPED,
            {
                "task_id": self._task_request.task_id,
                "agent_id": DEFAULT_SUPERVISOR_ID,
                "stage": "authorization_review",
                "plan_version": self._plan_version,
                "reason": "authorization_deterministic",
            },
        )
        logger.info("授权候选通过确定性最小化校验，跳过模型授权评审")
        return decision

    def _should_skip_plan_review(self) -> bool:
        """简单计划可用确定性校验替代 LLM 计划评审。

        仅在同时满足以下条件时跳过：首次规划（无重规划历史）、子任务
        不超过 2 个且无依赖、所有工具均为已批准可分配工具（无动态工具
        缺口）、所有写入子任务都有可机检验收规则、写入路径无重叠。
        任一条件不满足仍进入模型评审，避免弱化安全边界。
        """
        if not bool(
            getattr(self._settings, "skip_plan_review_for_simple", True)
        ):
            return False
        # 合并后的计划在外形上与简单计划相同，但原计划是两步依赖串联，
        # 不能因此失去模型评审（见 PlanningCoordinator._is_simple_plan）
        if self._state.plan_steps_merged > 0:
            return False
        if self._replan_count > 0 or self._plan_version > 1:
            return False
        if not self._subtasks or len(self._subtasks) > 2:
            return False
        if any(spec.dependencies for spec in self._subtasks):
            return False
        for spec in self._subtasks:
            if spec.write_scope and not spec.verification:
                return False
            for tool in spec.allowed_tools:
                if not (
                    self._tool_registry.is_approved(tool)
                    and self._tool_registry.is_assignable(tool)
                ):
                    return False
        declared: set[str] = set()
        for spec in self._subtasks:
            for entry in spec.write_scope:
                normalized = (entry or "").strip().replace("\\", "/")
                if not normalized or any(
                    marker in normalized for marker in ("*", "?", "[")
                ):
                    continue
                if normalized in declared:
                    return False
                declared.add(normalized)
        return True

    def _deterministic_plan_review(self) -> ReviewDecision:
        """对简单计划出具确定性通过结论，并记录跳过原因。"""
        decision = ReviewDecision(
            stage="plan_review",
            passed=True,
            summary=(
                "简单计划通过确定性校验（规模、依赖、工具授权与验收规则），"
                "跳过模型计划评审"
            ),
        )
        self._last_review = decision
        self._events.log_event(
            EVENT_REVIEW_SKIPPED,
            {
                "task_id": self._task_request.task_id,
                "agent_id": DEFAULT_SUPERVISOR_ID,
                "stage": "plan_review",
                "plan_version": self._plan_version,
                "reason": "simple_plan_deterministic",
            },
        )
        logger.info("简单计划跳过模型计划评审（确定性校验通过）")
        return decision

    async def _review_plan(self) -> ReviewDecision:
        """计划评审与授权评审（按风险分级）。

        无授权候选时只做计划评审；候选满足确定性最小化条件
        （``authorization_review_mode=risk_based``）时同样只做计划评审，
        授权结论由控制层给出；其余情况并行调用计划评审与授权评审后
        合并结论。

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

        plan_input = ReviewInput(
            stage=ReviewStage.PLAN_REVIEW,
            task_request=task_request,
            subtasks=subtask_snapshot,
            available_tools=list(available_tools),
            plan_version=self._plan_version,
        )
        # 无授权候选时授权评审没有可审对象：确定性跳过，省一次模型
        # 调用；计划评审仍可见 available_tools 并负责越权检查
        if not self._plan_grant_proposals:
            self._events.log_event(
                EVENT_REVIEW_SKIPPED,
                {
                    "task_id": task_request.task_id,
                    "agent_id": DEFAULT_SUPERVISOR_ID,
                    "stage": "authorization_review",
                    "plan_version": self._plan_version,
                    "reason": "no_grant_proposals",
                },
            )
            decision = await self._call_review_agent(plan_input)
            self._last_review = decision
            return decision

        # 授权评审分级：候选满足确定性最小化条件时只做计划评审，
        # 授权结论由控制层确定性给出（工具必要性语义判断归计划评审）
        if self._authorization_review_is_deterministic():
            self._deterministic_authorization_review()
            decision = await self._call_review_agent(plan_input)
            self._last_review = decision
            return decision

        plan_decision, authorization_decision = await asyncio.gather(
            self._call_review_agent(plan_input),
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

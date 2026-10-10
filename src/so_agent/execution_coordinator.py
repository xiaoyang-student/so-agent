"""执行协调器（ExecutionCoordinator）：调度执行、失败升级与工具缺口。

自原 ``WorkflowEngine``（workflow.py）拆分的执行/监控/替换/工具缺口
职责集合，方法体保持逐字平移：

- 调度执行：``_run_execution_cycle``（执行循环：调度 → 失败则触发
  分解问题重规划 → 直到全部成功）/ ``_execute_subtasks``（把一个批次
  交由 SubtaskDAGScheduler 调度并把过程映射到状态机）/
  ``_execute_subtask_attempt``（调度器的单次执行契约实现，任务 69
  起主管调用异常穿透上报，不再吞转为结果文本）；
- 结果解析：``_parse_execution_result``（主管回传文本 →
  ExecutionResult 的宽容解析薄封装）；
- 授权生命周期：``_revoke_batch_grants``（批次执行完成后统一回收
  子任务槽位授权，成功/失败路径均覆盖）/ ``_ensure_batch_grants_active``
  （返工批次重入执行时重新签发已回收的授权）；
- 失败升级：``_settle_failure_states``（结算尚未结算的分解问题与
  失败历史，替代状态链不再事后重建）/
  ``handle_replacement_lifecycle``（调度器替代流程真实时点通知的
  受控接收端：全局 FSM 迁移与相关事件写入经 asyncio.Lock 串行化，
  requested → executing → completed / failed 顺序与实际执行一致）/
  ``_handle_decomposition_issue``（分解问题上报与重规划申请）/
  ``_collect_failure_issue``（失败证据合并）/ ``_format_issue``；
- 工具缺口：``_resolve_tool_gaps``（检测 → 创建 → 评审 → 注册 →
  补发授权）/ ``_create_and_approve_tool``（评审不通过时含 1 次
  迭代修复机会）/ ``_verify_generated_tool``（沙箱逐用例语义比对
  验收）/ ``_invoke_code_agent_for_tool``（创建/按评审反馈修正）/
  ``_sync_generated_registry`` / ``_is_tool_ready`` /
  ``_ensure_mcp_client``。

跨协调器协作：失败升级后的重规划（``_plan_and_review``）、工具就绪
后的授权补发（``_issue_grants``）与重规划额度判定
（``_try_consume_replan_quota``）经 WorkflowDeps 通道回调规划协调器；
主管/评审 Agent 调用经同一通道由薄编排器实现（协调器之间不直接
互相引用）。
"""

from __future__ import annotations

import asyncio
import json
import math
import re
import sys
from datetime import timezone
from typing import Any

from pydantic import ValidationError

import so_agent.tool_packages.code_agent.agent as code_agent_agent_mod
from so_agent.json_utils import extract_json_object
from so_agent.mcp.client import MCPClient, MCPClientError
from so_agent.mcp.protocol import extract_result_text
from so_agent.orchestrator import ExecutionOutput
from so_agent.models import (
    AgentRecord,
    ExecutionResult,
    ReplacementRequest,
    SubtaskSpec,
    TaskDecompositionIssue,
    TaskStatus,
    _utc_now,
)
from so_agent.runtime.events import (
    EVENT_REPLACEMENT_COMPLETED,
    EVENT_REPLACEMENT_EXECUTING,
    EVENT_REPLACEMENT_FAILED,
    EVENT_REPLACEMENT_REQUESTED,
    EVENT_TOOL_CALLED,
)
from so_agent.runtime.generated_tool_runner import (
    GeneratedToolExecutionError,
    run_generated_tool_in_sandbox,
)
from so_agent.runtime.registry import DEFAULT_SUPERVISOR_ID
from so_agent.runtime.sandbox import SandboxRunner
from so_agent.runtime.scheduler import SchedulerError
from so_agent.runtime.tool_registry import ToolRegistryError
from so_agent.state_machine import (
    WorkflowError,
    WorkflowState,
    _coerce_structured_output,
)
from so_agent.tool_packages.code_agent.agent import (
    CodeAction,
    CodeAgentInput,
    CodeAgentOutput,
)
from so_agent.tool_packages.review_agent.agent import ReviewInput, ReviewStage
from so_agent.workflow_support import BaseCoordinator, WorkflowDeps

# 动态生成工具名称规范（与代码 Agent 侧保持一致）
_TOOL_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,63}$")

# 沙箱验收的数值语义比对容差（浮点运算的机器误差在该量级内视为一致）
_SEMANTIC_TOLERANCE = 1e-6


def _semantic_matches(actual: Any, expected: Any) -> bool:
    """沙箱验收的语义比对：实际输出与期望值语义一致即视为匹配。

    相比精确 JSON 字符串比对（``json.dumps(sort_keys=True)`` 全等）
    放宽三点：

    - 数值类型经 ``math.isclose`` 以 1e-6 容差比较（bool 除外：bool
      与数值显式区分，避免 ``1 == True`` 的跨型误判）；
    - 字符串去除首尾空白后比较；
    - dict 以期望值为键集基准：实际值多出的额外字段忽略，期望中的
      每个键须存在且值递归匹配；list 长度一致并逐元素递归匹配。

    Args:
        actual: 工具实际返回值（沙箱运行器已强制 JSON 可解析）。
        expected: manifest 声明的期望输出。

    Returns:
        True 表示语义一致（该验收项通过）。
    """
    if isinstance(expected, bool) or isinstance(actual, bool):
        # bool 是 int 子类：与数值显式区分，须同型同值
        return (
            isinstance(expected, bool)
            and isinstance(actual, bool)
            and actual is expected
        )
    if isinstance(expected, (int, float)):
        return isinstance(actual, (int, float)) and math.isclose(
            actual,
            expected,
            rel_tol=_SEMANTIC_TOLERANCE,
            abs_tol=_SEMANTIC_TOLERANCE,
        )
    if isinstance(expected, str):
        return isinstance(actual, str) and actual.strip() == expected.strip()
    if expected is None:
        return actual is None
    if isinstance(expected, dict):
        # 以期望值为键集基准：实际值多出的额外字段忽略
        return isinstance(actual, dict) and all(
            key in actual and _semantic_matches(actual[key], value)
            for key, value in expected.items()
        )
    if isinstance(expected, list):
        return (
            isinstance(actual, list)
            and len(actual) == len(expected)
            and all(
                _semantic_matches(actual_item, expected_item)
                for actual_item, expected_item in zip(actual, expected)
            )
        )
    return actual == expected

# 未完成标记关键词：降级解析路径的 output/error 中出现这些词时，
# 强制判定为失败（宁可重试不可放行“待执行”的空壳结果）
_PENDING_KEYWORDS: tuple[str, ...] = ("待执行", "未完成", "pending", "not started")

# 降级路径 output 有效性下限：短于该长度的“成功”产出视为空壳/无效
_MIN_VALID_OUTPUT_LENGTH: int = 20


class ExecutionCoordinator(BaseCoordinator):
    """执行协调器：子任务调度执行、失败升级与工具缺口流程。"""

    def __init__(self, state: WorkflowState, deps: WorkflowDeps) -> None:
        super().__init__(state, deps)
        # 替代生命周期通知的串行化锁（任务 34）：并发子任务的替代流程
        # 同时回调时，全局 FSM 迁移与相关事件写入在此锁内串行执行，
        # 避免并发替代导致非法状态迁移
        self._replacement_lifecycle_lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # 单次子任务执行（SubtaskDAGScheduler 的 executor 契约实现）
    # ------------------------------------------------------------------
    async def _execute_subtask_attempt(
        self, spec: SubtaskSpec, agent_record: AgentRecord
    ) -> ExecutionResult:
        """单次子任务执行（SubtaskDAGScheduler 的 executor 契约实现）。

        通过主管 Agent 路由执行：构造 ``phase="execute_subtask"`` 的 JSON
        载荷（含任务请求、子任务规格、Agent 记录、尝试序号与该子任务的
        ToolGrant），由主管 Agent 选择对应工具完成执行，并要求其把工具
        返回的 ExecutionResult JSON 填入 ``final_result`` 回传。

        Args:
            spec: 子任务规格（同一计划版本内）。
            agent_record: 本次执行使用的 Agent 记录（原始或替代 Agent）。

        Returns:
            ExecutionResult：解析成功的结果；主管输出无法解析为契约时
            返回 ``success=False`` 的结果（由调度器按失败尝试计数并
            推进升级流程）。

        Raises:
            Exception: 主管调用抛出的控制层异常（超时 / 网络 / 限流 /
            权限等）原样向上穿透（任务 69 信任边界）。此类异常是控制层
            可信证据，必须由调度器按异常对象高信任分类
            （classify_exception：异常类名 / HTTP 状态码），不得在本层
            吞转为结果文本——那会落入调度器的低信任自由文本分类，
            丢失真实系统失败类型，还给予子 Agent 在 error 文本里伪造
            “429 / permission denied” 等关键词冒充系统失败的通道。
            asyncio.CancelledError / PipelineStoppedError 属
            BaseException，天然穿透，不受影响。
        """
        print(
            f"[so-agent] 开始执行子任务 [{spec.subtask_id}] {spec.title}"
            f"（角色: {spec.role}，尝试: {agent_record.attempt_count}）",
            file=sys.stderr, flush=True,
        )
        grant = self._plan_grant_map.get(spec.subtask_id)
        payload: dict[str, Any] = {
            "phase": "execute_subtask",
            "task_id": self._task_request.task_id,
            "task_request": self._task_request.model_dump(mode="json"),
            "subtask": spec.model_dump(mode="json"),
            "agent": agent_record.model_dump(mode="json"),
            "attempt": agent_record.attempt_count,
            "grant": grant.model_dump(mode="json") if grant is not None else None,
        }
        try:
            output = await self._run_supervisor(payload, ExecutionOutput)
        except Exception as exc:
            # 控制层异常穿透（任务 69）：打印审计日志后原样上抛，由调度
            # 器统一捕获并按异常对象高信任分类——本层不再吞转为
            # “主管 Agent 调用失败：…” 的结果文本（那会让真实系统失败
            # 降级为低信任文本分类，丢失恢复方向所需的结构化证据）
            print(
                f"[so-agent] 子任务 [{spec.subtask_id}] 执行失败：{type(exc).__name__}",
                file=sys.stderr, flush=True,
            )
            raise
        result = self._parse_execution_result(output.final_result, spec)
        status = "成功" if result.success else f"失败：{result.error or '未知'}"
        print(
            f"[so-agent] 子任务 [{spec.subtask_id}] 执行完成 → {status}",
            file=sys.stderr, flush=True,
        )
        return result

    @staticmethod
    def _parse_execution_result(
        final_result: str, spec: SubtaskSpec
    ) -> ExecutionResult:
        """把主管 Agent 回传的文本解析为 ExecutionResult（宽容解析薄封装）。

        解析委托 ``_coerce_structured_output``（纯文本 JSON → 契约模型，
        含围栏剥离与 JSON 修复链路），成功后把 ``subtask_id`` 归一化为
        当前子任务编号；仅在本类特有的降级场景下做字段抢救（从内嵌
        JSON 对象提取可用字段），避免整体判失败导致无效重试。全部失败
        时返回 ``success=False`` 的结果（保留原始文本摘要作为错误信息）。

        Args:
            final_result: 主管 Agent 输出的 final_result 文本。
            spec: 对应当前执行批次的子任务规格。
        """
        text = (final_result or "").strip()
        if text:
            # 快速路径：复用统一宽容解析链路（契约完全匹配时直接采用）
            try:
                parsed = _coerce_structured_output(
                    text, ExecutionResult, label="子任务执行结果"
                )
                if parsed.subtask_id != spec.subtask_id:
                    parsed = parsed.model_copy(
                        update={"subtask_id": spec.subtask_id}
                    )
                return parsed
            except WorkflowError:
                pass
            # 降级路径：从内嵌 JSON 候选抢救可用字段（ExecutionResult 特有）
            candidate = extract_json_object(text)
            if candidate is not None:
                raw_success = candidate.get("success")
                raw_output = (
                    candidate.get("output")
                    or candidate.get("final_result")
                    or json.dumps(candidate, ensure_ascii=False)
                )
                raw_error = candidate.get("error")
                raw_evidence = candidate.get("evidence")
                raw_artifacts = candidate.get("artifacts")
                inferred_success = (
                    raw_success
                    if isinstance(raw_success, bool)
                    else "error" not in candidate or not candidate.get("error")
                )
                output_text = (
                    raw_output
                    if isinstance(raw_output, str)
                    else json.dumps(raw_output, ensure_ascii=False, default=str)
                )
                error_text = raw_error if isinstance(raw_error, str) else None
                # 防御一（output 有效性）：降级判定为成功但产出为空/过短时，
                # 翻转为失败——空壳 JSON 不得混入最终交付物
                if inferred_success and len(output_text.strip()) < _MIN_VALID_OUTPUT_LENGTH:
                    inferred_success = False
                    error_text = "输出内容过短，疑似无效结果"
                # 防御二（关键词）：output/error 携带“待执行/未完成/pending/
                # not started”等未完成标记时强制判失败（大小写不敏感）
                if inferred_success:
                    blob = f"{output_text}\n{error_text or ''}".lower()
                    hit = next(
                        (kw for kw in _PENDING_KEYWORDS if kw in blob), None
                    )
                    if hit is not None:
                        inferred_success = False
                        error_text = (
                            error_text
                            or f"输出包含未完成标记（{hit}），疑似无效结果"
                        )
                return ExecutionResult(
                    subtask_id=spec.subtask_id,
                    success=inferred_success,
                    output=output_text,
                    error=error_text,
                    evidence=raw_evidence if isinstance(raw_evidence, list) else [],
                    artifacts=raw_artifacts if isinstance(raw_artifacts, list) else [],
                )
        summary = text[:200] + ("…" if len(text) > 200 else "")
        return ExecutionResult(
            subtask_id=spec.subtask_id,
            success=False,
            error=f"未能从主管 Agent 输出解析出 ExecutionResult JSON；原文：{summary!r}",
        )

    # ------------------------------------------------------------------
    # 执行循环与批次调度
    # ------------------------------------------------------------------
    async def _run_execution_cycle(self) -> dict[str, ExecutionResult] | None:
        """执行循环：调度执行 → 失败则触发分解问题重规划 → 直到全部成功。

        每一轮完成一个批次的调度（含调度器内部的尝试/替换流程）；出现
        失败批次时组装任务分解问题、上报主管、消耗一次拆分修改额度并
        重新拆解；额度耗尽时返回 None（由调用方转入人工兜底）。

        重规划采纳后，计划协调器已把通过安全校验的显式复用结果
        （reuse_from 声明，含来源标记）并入 ``self._results``；此处
        把执行队列过滤为"无复用结果的子任务"，被拒声明的新任务因
        无结果自动回退 pending 重新执行，复用任务不进入执行队列。

        Returns:
            当前计划全部成功时的执行结果（含复用结果）；额度耗尽时为 None。
        """
        while True:
            # 停止门（任务 45）：重规划循环头拒绝新动作（失败升级的
            # 重规划入口，PipelineStoppedError 穿透宽捕获向上传播）
            self._gate_check("replan")
            batch = await self._execute_subtasks(list(self._pending_specs))
            failed = [
                subtask_id for subtask_id, item in batch.items() if not item.success
            ]
            if not failed:
                return {**self._results, **batch}
            issue = self._collect_failure_issue(failed)
            if not await self._handle_decomposition_issue(issue):
                return None
            if not await self._plan_and_review(issue=issue):
                return None
            # 重规划采纳后：执行队列仅保留无复用结果的子任务
            self._pending_specs = [
                spec
                for spec in self._subtasks
                if spec.subtask_id not in self._results
            ]
            # 清理调度器残留状态，防止跨版本指纹污染：通知去重集合与
            # 调度器累积列表同步清理，避免旧版本的替换申请/分解问题在
            # 新版本中被重新回放（任务 34 后替代状态链由真实时点通知
            # 落账，去重集合与源列表必须保持一致）
            self._seen_replacement_requests.clear()
            self._seen_issue_fingerprints.clear()
            self._scheduler.replacement_requests.clear()
            self._scheduler.decomposition_issues.clear()

    async def _execute_subtasks(
        self, specs: list[SubtaskSpec]
    ) -> dict[str, ExecutionResult]:
        """通过 SubtaskDAGScheduler 调度一个子任务批次，并把过程映射到状态机。

        调度前先解决工具缺口（``_resolve_tool_gaps``：创建 → 评审 →
        注册 → 授权）；调度器内部完成"策略驱动重试（失败分类 / 指纹
        去重 / 幂等退避 / 替换限制）→ 一次替换请求 → 替代 Agent
        （无再申请资格）→ 分解问题上报"。替代状态链
        （replacement_requested → replacement_executing → executing /
        failed）由调度器在真实生命周期时点经
        ``handle_replacement_lifecycle`` 实时落账（任务 34，不再事后
        重建）；返回后由 ``_settle_failure_states`` 仅结算分解问题与
        失败历史，并把结果写入 ``self._results``。

        Args:
            specs: 待执行批次（当前计划全量或结果评审返工子集）。

        Returns:
            ``{subtask_id: ExecutionResult}``（仅包含本批次子任务）；
            工具缺口未解决的子任务直接标记为失败；调度器拒绝调度
            （结构异常）时转换为全失败批次，交由失败升级流程处理。
        """
        # 停止门（任务 45）：批次派发前拒绝新动作（含工具缺口解决与
        # 授权重签发），停止后不再派发任何子任务
        self._gate_check("dispatch")
        if not specs:
            return {}
        gap_failures = await self._resolve_tool_gaps(specs)
        executable = [
            spec for spec in specs if spec.subtask_id not in gap_failures
        ]
        if not executable:
            self._results.update(gap_failures)
            return dict(gap_failures)
        self._transition(
            TaskStatus.EXECUTING,
            reason="调度子任务批次执行",
            subtask_ids=[spec.subtask_id for spec in executable],
        )
        # 授权生命周期闭环（执行窗口开始）：上一窗口回收的槽位授权在
        # 返工批次重入执行时重新签发，保证本窗口授权新鲜有效
        self._ensure_batch_grants_active(executable)
        try:
            batch = await self._scheduler.schedule(
                self._task_request.task_id, executable
            )
        except SchedulerError as exc:
            batch = {
                spec.subtask_id: ExecutionResult(
                    subtask_id=spec.subtask_id,
                    success=False,
                    error=f"调度失败：{exc}",
                )
                for spec in executable
            }
        # 授权生命周期闭环（执行窗口结束）：批次已定型（成功/耗尽尝试
        # 失败，含替代 Agent 完成），统一回收本批次子任务的槽位授权，
        # 避免授权跨执行窗口残留（成功与失败路径均覆盖）
        self._revoke_batch_grants(executable)
        merged: dict[str, ExecutionResult] = {**gap_failures, **batch}
        self._settle_failure_states(merged)
        self._results.update(merged)
        return merged

    # ------------------------------------------------------------------
    # 授权生命周期（批次后回收 + 返工重签发）
    # ------------------------------------------------------------------
    def _grant_slot_agent_id(self, subtask_id: str) -> str:
        """返回子任务的授权槽位标识（与规划阶段 _issue_grants 同一命名规则）。"""
        return f"{self._task_request.task_id}:{subtask_id}:grant-slot"

    def _revoke_batch_grants(self, specs: list[SubtaskSpec]) -> None:
        """回收本批次子任务的槽位授权（授权生命周期闭环）。

        子任务执行完成后（成功与失败路径统一，含替代 Agent 完成后的
        定型批次），按槽位标识（``{task_id}:{subtask_id}:grant-slot``）
        批量撤销授权：该子任务后续工具调用立即失去授权依据，防止授权
        跨执行窗口残留（签发时的 ``expires_at`` 为第二道防线）。撤销
        对无授权子任务为无操作（无可回收，静默忽略）；撤销原因固定为
        ``execution_window_closed``（审计可区分批次回收与任务级清理）。
        """
        for spec in specs:
            self._gateway.revoke_grants_for_agent(
                self._grant_slot_agent_id(spec.subtask_id),
                reason="execution_window_closed",
            )

    def _ensure_batch_grants_active(self, specs: list[SubtaskSpec]) -> None:
        """保证本批次子任务的槽位授权处于有效状态（执行窗口开始前）。

        授权生命周期闭环的对称面：批次执行完成后槽位授权已被统一回收
        （``_revoke_batch_grants``）；结果评审返工批次会再次进入执行
        路径，此时上一窗口的授权已撤销——检测到失效授权（不存在/已
        撤销/已过期）时重新签发（``_issue_grants`` 经 deps 通道覆盖本
        版本计划的授权映射），保证每个执行窗口的授权新鲜有效，避免
        返工批次因授权缺失而必然失败。
        """
        for spec in specs:
            grant = self._plan_grant_map.get(spec.subtask_id)
            if grant is None:
                continue  # 无授权子任务（allowed_tools 未含已就绪工具）
            current = self._gateway.get_grant(grant.grant_id)
            expires_at = current.expires_at if current is not None else None
            if expires_at is not None and expires_at.tzinfo is None:
                expires_at = expires_at.replace(tzinfo=timezone.utc)
            expired = expires_at is not None and expires_at <= _utc_now()
            if current is None or current.revoked or expired:
                self._issue_grants()  # 重签发：覆盖本版本授权映射
                return

    # ------------------------------------------------------------------
    # 工具缺口流程（创建 → 评审 → 验收 → 注册 → 补发授权）
    # ------------------------------------------------------------------
    async def _resolve_tool_gaps(
        self, specs: list[SubtaskSpec]
    ) -> dict[str, ExecutionResult]:
        """检测并解决子任务的工具缺口（未注册或未批准的 allowed_tools）。

        处理顺序：与磁盘候选清单对齐 → 逐缺口工具经 MCP 调用代码 Agent
        创建候选 → 评审 Agent 审批（tool_review）→ 注册进工具注册表
        （含 MCP Server）→ 对已就绪工具补发本版本计划的 ToolGrant；任一
        环节失败时把相关子任务标记为失败（附带 ``tool_gap:<tool>``
        证据），交由失败升级流程处理。

        Returns:
            ``{subtask_id: ExecutionResult}``：缺口未解决的子任务失败结果；
            所有工具均已就绪时为空字典。
        """
        self._check_cancel()
        missing: dict[str, list[str]] = {}
        for spec in specs:
            gaps = [
                name for name in spec.allowed_tools if not self._is_tool_ready(name)
            ]
            if gaps:
                missing[spec.subtask_id] = gaps
        if not missing:
            return {}

        # 与磁盘候选清单对齐：规划阶段可能已由代码 Agent 产出候选
        sync_error = self._sync_generated_registry()
        missing = {
            subtask_id: [name for name in gaps if not self._is_tool_ready(name)]
            for subtask_id, gaps in missing.items()
        }
        missing = {key: value for key, value in missing.items() if value}
        if not missing:
            self._issue_grants()  # 同步后工具已就绪：补发授权
            return {}

        failures: dict[str, list[str]] = {}
        resolved_any = False
        for tool_name in sorted({name for gaps in missing.values() for name in gaps}):
            reason = await self._create_and_approve_tool(tool_name)
            if reason is None:
                resolved_any = True
                continue
            for subtask_id, gaps in missing.items():
                if tool_name in gaps:
                    failures.setdefault(subtask_id, []).append(
                        f"{tool_name}（{reason}）"
                    )
        if resolved_any:
            self._issue_grants()  # 对已就绪工具补发授权（覆盖本版本映射）
        if not failures:
            return {}
        results: dict[str, ExecutionResult] = {}
        for subtask_id, reasons in failures.items():
            detail = "；".join(reasons)
            if sync_error:
                detail += f"；候选清单同步异常：{sync_error}"
            results[subtask_id] = ExecutionResult(
                subtask_id=subtask_id,
                success=False,
                error=f"工具缺口未解决：{detail}",
                evidence=[f"tool_gap:{name}" for name in missing[subtask_id]],
            )
        return results

    def _is_tool_ready(self, tool_name: str) -> bool:
        """工具是否已注册且已批准（可进入子 Agent 授权候选集合）。"""
        return self._tool_registry.is_approved(tool_name)

    def _sync_generated_registry(self) -> str | None:
        """把代码 Agent 落盘的生成工具清单同步进工具注册表。

        Returns:
            None 表示同步成功（或清单尚不存在）；否则返回错误说明。
        """
        registry_path = code_agent_agent_mod.GENERATED_REGISTRY_PATH
        try:
            if registry_path.is_file():
                self._tool_registry.load_generated_registry(registry_path)
        except ToolRegistryError as exc:
            return str(exc)
        return None

    def _ensure_mcp_client(self) -> MCPClient:
        """返回 MCP 客户端：优先使用注入实例，否则从注册表已挂载的 Server 构建。"""
        if self._mcp_client is not None:
            return self._mcp_client
        client = MCPClient()
        for name in self._tool_registry.list_tool_names():
            server = self._tool_registry.get_mcp_server(name)
            if server is not None:
                client.connect(server)
        self._mcp_client = client
        return client

    async def _create_and_approve_tool(self, tool_name: str) -> str | None:
        """创建一个缺口工具候选并送评审，通过后注册进工具注册表。

        流程：记录缺口事件 → （候选不存在时）经 MCP 调用代码 Agent 创建
        → 评审 Agent 审批（tool_review）→ 注册（含 MCP Server）并写回
        registry.json。评审不通过时有 1 次迭代修复机会：把评审反馈传回
        代码 Agent 修正候选，修正后重新走沙箱验收 + 评审；第 2 次评审
        仍不通过才真正标记失败。

        Args:
            tool_name: 缺口工具名称（子任务 allowed_tools 中的未就绪名称）。

        Returns:
            None 表示工具已注册且 approved；否则返回失败原因。
        """
        task_id = self._task_request.task_id
        base_event: dict[str, Any] = {
            "task_id": task_id,
            "agent_id": DEFAULT_SUPERVISOR_ID,
            "tool_name": tool_name,
            "plan_version": self._plan_version,
        }
        self._events.log_event(
            EVENT_TOOL_CALLED, {**base_event, "action": "gap_detected"}
        )
        if not _TOOL_NAME_PATTERN.match(tool_name):
            reason = "工具名称不符合 snake_case 规范，无法自动创建"
            self._events.log_event(
                EVENT_TOOL_CALLED,
                {**base_event, "action": "gap_unresolved", "reason": reason},
            )
            return reason

        manifest = self._tool_registry.get_manifest(tool_name)
        if manifest is None:
            create_error = await self._invoke_code_agent_for_tool(tool_name)
            if create_error is not None:
                self._events.log_event(
                    EVENT_TOOL_CALLED,
                    {**base_event, "action": "gap_unresolved", "reason": create_error},
                )
                return create_error
            self._sync_generated_registry()
            manifest = self._tool_registry.get_manifest(tool_name)
            if manifest is None:
                reason = "代码 Agent 未按约定产出工具候选清单"
                self._events.log_event(
                    EVENT_TOOL_CALLED,
                    {**base_event, "action": "gap_unresolved", "reason": reason},
                )
                return reason
            self._events.log_event(
                EVENT_TOOL_CALLED, {**base_event, "action": "candidate_created"}
            )

        # 评审 + 沙箱验收循环：评审不通过时允许 1 次迭代修复——评审反馈
        # 传回代码 Agent 修正候选，修正后重新走沙箱验收 + 评审；第 2 次
        # 评审仍不通过才真正标记失败
        review_attempt = 0
        while True:
            if manifest.review_status != "approved":
                review_attempt += 1
                decision = await self._call_review_agent(
                    ReviewInput(
                        stage=ReviewStage.TOOL_REVIEW,
                        task_request=self._task_request,
                        generated_tool=manifest,
                        plan_version=self._plan_version,
                    )
                )
                if not decision.passed:
                    detail = (
                        "；".join(decision.issues)
                        or decision.summary
                        or "未说明原因"
                    )
                    if review_attempt >= 2:
                        # 第 2 次评审仍未通过：真正标记失败
                        reason = f"工具评审未通过：{detail}"
                        self._events.log_event(
                            EVENT_TOOL_CALLED,
                            {
                                **base_event,
                                "action": "gap_unresolved",
                                "reason": reason,
                            },
                        )
                        return reason
                    # 首次未通过：把评审反馈传回代码 Agent 迭代修正一次
                    fix_error = await self._invoke_code_agent_for_tool(
                        tool_name, feedback=detail
                    )
                    if fix_error is not None:
                        reason = f"工具评审未通过且修正调用失败：{detail}；{fix_error}"
                        self._events.log_event(
                            EVENT_TOOL_CALLED,
                            {
                                **base_event,
                                "action": "gap_unresolved",
                                "reason": reason,
                            },
                        )
                        return reason
                    self._sync_generated_registry()
                    refreshed = self._tool_registry.get_manifest(tool_name)
                    if refreshed is None:
                        reason = f"工具评审未通过且修正后未产出候选清单：{detail}"
                        self._events.log_event(
                            EVENT_TOOL_CALLED,
                            {
                                **base_event,
                                "action": "gap_unresolved",
                                "reason": reason,
                            },
                        )
                        return reason
                    manifest = refreshed
                    continue  # 修正后重新走沙箱验收 + 评审
                manifest = manifest.model_copy(update={"review_status": "approved"})

            # 沙箱运行验收：逐 test_cases 试跑比对正确答案，通过才注册
            if not manifest.verified:
                manifest, verify_error = await self._verify_generated_tool(manifest)
                if verify_error is not None:
                    self._events.log_event(
                        EVENT_TOOL_CALLED,
                        {
                            **base_event,
                            "action": "gap_unresolved",
                            "reason": verify_error,
                        },
                    )
                    return verify_error
            break

        try:
            self._tool_registry.register_generated_tool(manifest)
            self._tool_registry.save_generated_registry(
                code_agent_agent_mod.GENERATED_REGISTRY_PATH
            )
        except ToolRegistryError as exc:
            reason = f"工具注册失败：{exc}"
            self._events.log_event(
                EVENT_TOOL_CALLED,
                {**base_event, "action": "gap_unresolved", "reason": reason},
            )
            return reason
        self._events.log_event(
            EVENT_TOOL_CALLED, {**base_event, "action": "registered"}
        )
        return None

    async def _verify_generated_tool(
        self, manifest: "GeneratedToolManifest"
    ) -> "tuple[GeneratedToolManifest, str | None]":
        """沙箱运行验收：逐 test_cases 试跑并与期望输出语义比对（正确答案校验）。

        生成工具在沙箱子进程运行（与正式调用一致），把实际返回值与
        ``expected_outputs`` 语义比对（``_semantic_matches``：数值容差、
        字符串去首尾空白、忽略实际值多出的额外字段）；返回值须为 JSON
        （沙箱运行器已强制）。全部用例通过才置 ``verified=True``。

        Returns:
            ``(manifest, None)`` 表示验收通过；``(manifest, 失败原因)`` 表示
            未通过（工具应被拒绝注册并回代码 Agent 返工）。
        """
        test_cases = list(manifest.test_cases or [])
        expected = list(manifest.expected_outputs or [])
        base = {
            "task_id": self._task_request.task_id,
            "agent_id": DEFAULT_SUPERVISOR_ID,
            "tool_name": manifest.tool_name,
            "plan_version": self._plan_version,
        }
        if not test_cases:
            # 无测试用例：无法比对正确答案，保守跳过运行验证（结构/评审已通过）
            self._events.log_event(
                EVENT_TOOL_CALLED,
                {**base, "action": "tool_verify_skipped", "reason": "未提供 test_cases"},
            )
            return manifest.model_copy(update={"verified": True}), None
        if len(expected) < len(test_cases):
            return manifest, (
                f"验收测试用例与期望输出数量不匹配：{len(test_cases)} 用例 vs "
                f"{len(expected)} 期望"
            )
        sandbox = SandboxRunner(context=self._context, settings=self._settings)
        entry = code_agent_agent_mod.GENERATED_TOOLS_DIR / manifest.entry_file
        for index, args in enumerate(test_cases):
            try:
                result = await run_generated_tool_in_sandbox(
                    sandbox,
                    entry,
                    manifest.tool_name,
                    dict(args),
                    task_id=self._task_request.task_id,
                    run_id=f"verify-{index}",
                )
            except GeneratedToolExecutionError as exc:
                return manifest, f"验收用例 #{index} 执行失败：{exc}"
            if not _semantic_matches(result, expected[index]):
                return manifest, (
                    f"验收用例 #{index} 输出不符：实际 {result!r} ≠ 期望 {expected[index]!r}"
                )
        self._events.log_event(
            EVENT_TOOL_CALLED,
            {**base, "action": "tool_verified", "test_count": len(test_cases)},
        )
        return manifest.model_copy(update={"verified": True}), None

    async def _invoke_code_agent_for_tool(
        self, tool_name: str, *, feedback: str | None = None
    ) -> str | None:
        """经 MCP 调用代码 Agent 创建/修正工具候选（write_generated_tool）。

        Args:
            tool_name: 缺口工具名称（子任务 allowed_tools 中的未就绪名称）。
            feedback: 评审反馈；提供时进入修正模式（要求代码 Agent 按反馈
                修正既有候选后重写），None 表示首次创建。

        Returns:
            None 表示代码 Agent 成功完成创建/修正动作；否则返回失败原因。
        """
        client = self._ensure_mcp_client()
        if feedback:
            task_description = (
                f"修正动态工具 {tool_name}：该工具候选未通过评审，评审反馈："
                f"{feedback}。请按反馈修正实现后重新调用 write_generated_tool "
                f"写入 generated_tools（工具名必须为 {tool_name}）；入口函数须为 "
                f"run(**kwargs)，其返回值必须是 JSON 可序列化（dict/list/标量，"
                f"硬约束）。manifest_data 需含 description、input_schema、"
                f"output_schema、created_by_task={self._task_request.task_id}，"
                f"以及 test_cases（验收测试用例，每项为 run() 的关键字参数对象）"
                f"与 expected_outputs（与之一一对应的期望返回值，JSON 可序列化）"
                f"——注册前将在沙箱逐用例运行比对，全部匹配才通过验收。"
            )
            action = CodeAction.MODIFY
        else:
            task_description = (
                f"创建动态工具 {tool_name}：实现其完整功能并调用 "
                f"write_generated_tool 写入 generated_tools（工具名必须为 "
                f"{tool_name}）；入口函数须为 run(**kwargs)，其返回值必须是 "
                f"JSON 可序列化（dict/list/标量，硬约束）。manifest_data 需含 "
                f"description、input_schema、output_schema、created_by_task="
                f"{self._task_request.task_id}，以及 test_cases（验收测试用例，"
                f"每项为 run() 的关键字参数对象）与 expected_outputs（与之一一 "
                f"对应的期望返回值，JSON 可序列化）——注册前将在沙箱逐用例运行 "
                f"比对，全部匹配才通过验收。"
            )
            action = CodeAction.CREATE
        payload = CodeAgentInput(
            task_description=task_description,
            action=action,
        )
        try:
            response = await client.call_tool(
                "code_agent", "code_agent", payload.model_dump(mode="json")
            )
        except MCPClientError as exc:
            return f"代码 Agent MCP 通道不可用：{exc}"
        if response.error is not None:
            return (
                f"代码 Agent 调用失败：[{response.error.code}] "
                f"{response.error.message}"
            )
        text = extract_result_text(response)
        output: CodeAgentOutput | None = None
        try:
            output = CodeAgentOutput.model_validate_json(text)
        except ValidationError:
            candidate = extract_json_object(text)
            if candidate is not None:
                try:
                    output = CodeAgentOutput.model_validate(candidate)
                except ValidationError:
                    output = None
        if output is None:
            summary = text[:200] + ("…" if len(text) > 200 else "")
            return f"代码 Agent 输出无法解析为 CodeAgentOutput：{summary!r}"
        if not output.success:
            return (
                "代码 Agent 创建工具失败："
                f"{output.error or output.summary or '未说明原因'}"
            )
        return None

    # ------------------------------------------------------------------
    # 失败升级（replacement / decomposition）
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # 替代生命周期通知接收端（任务 34：真实时点落账）
    # ------------------------------------------------------------------
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

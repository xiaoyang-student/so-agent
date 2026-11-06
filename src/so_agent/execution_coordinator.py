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

为控制单文件体量，两个职责域已拆为混入模块：工具缺口生命周期
（:mod:`so_agent.execution_tool_gap`）与失败升级/替代生命周期
（:mod:`so_agent.execution_replacement`）；本模块保留单次执行、
结果解析、执行循环与授权生命周期。
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from datetime import timezone
from pathlib import Path
from typing import Any

from so_agent.json_utils import extract_json_object
from so_agent.mcp.protocol import METHOD_TOOLS_CALL, MCPRequest, extract_result_text
from so_agent.orchestrator import ExecutionOutput
from so_agent.models import (
    AgentRecord,
    ExecutionResult,
    SubtaskSpec,
    TaskStatus,
    _utc_now,
)
from so_agent.runtime.events import (
    EVENT_ACTION_REPAIR,
    EVENT_TOOL_CALLED,
    EVENT_TOOL_EVIDENCE_MISSING,
    EVENT_VERIFICATION_FAILED,
    EVENT_WRITE_SCOPE_UNMET,
)
from so_agent.runtime.logging_setup import get_logger
from so_agent.runtime.script_cache import (
    record_verified_scripts,
    stage_cached_scripts,
)
from so_agent.turn_budget import model_turn_override
from so_agent.runtime.usage import usage_scope
from so_agent.runtime.verification import evaluate_verification_rules
from so_agent.runtime.write_scope import (
    GLOB_CHARS as _WRITE_SCOPE_GLOB_CHARS,
    missing_scope_entries,
    normalize_entry as _normalize_write_scope_entry,
)
from so_agent.runtime.routing import EXECUTOR_CODE_AGENT, resolve_executor
from so_agent.runtime.scheduler import SchedulerError
from so_agent.state_machine import (
    WorkflowError,
    WorkflowState,
    _coerce_structured_output,
)
from so_agent.execution_replacement import ReplacementMixin
from so_agent.execution_tool_gap import (
    ToolGapMixin,
    _SEMANTIC_TOLERANCE,
    _semantic_matches,
)
from so_agent.workflow_support import BaseCoordinator, WorkflowDeps

logger = get_logger("execution")

# 动态生成工具名称规范（与代码 Agent 侧保持一致）

# 沙箱验收的数值语义比对容差（浮点运算的机器误差在该量级内视为一致）



# 未完成标记关键词：降级解析路径的 output/error 中出现这些词时，
# 强制判定为失败（宁可重试不可放行“待执行”的空壳结果）
_PENDING_KEYWORDS: tuple[str, ...] = ("待执行", "未完成", "pending", "not started")

# 降级路径 output 有效性下限：短于该长度的“成功”产出视为空壳/无效
_MIN_VALID_OUTPUT_LENGTH: int = 20

def missing_write_scope(
    root: Path, scope: list[str], *, since: float | None = None
) -> list[str]:
    """返回 write_scope 中未落地的条目（要求非空；``since`` 限定本次运行）。"""
    return missing_scope_entries(root, scope, since=since)


class ExecutionCoordinator(ToolGapMixin, ReplacementMixin, BaseCoordinator):
    """执行协调器：子任务调度执行、失败升级与工具缺口流程。"""

    def __init__(self, state: WorkflowState, deps: WorkflowDeps) -> None:
        super().__init__(state, deps)
        # 替代生命周期通知的串行化锁（任务 34）：并发子任务的替代流程
        # 同时回调时，全局 FSM 迁移与相关事件写入在此锁内串行执行，
        # 避免并发替代导致非法状态迁移
        self._replacement_lifecycle_lock = asyncio.Lock()
        # 写作用域新鲜度基准：本次运行开始时间。按尝试计会让合法的幂等
        # 重试（文件在上一轮已写好）被误判为未产出。
        self._run_started_at = time.time()

    # ------------------------------------------------------------------
    # 单次子任务执行（SubtaskDAGScheduler 的 executor 契约实现）
    # ------------------------------------------------------------------
    async def _execute_subtask_attempt(
        self, spec: SubtaskSpec, agent_record: AgentRecord
    ) -> ExecutionResult:
        """单次子任务执行（SubtaskDAGScheduler 的 executor 契约实现）。

        用量归因作用域在本层设置：主管与嵌套 code_agent 的模型调用共享
        同一异步上下文，因此本次尝试的全部 token 精确记到当前子任务，
        并发子任务之间互不串账（子任务硬预算依赖该归因）。

        执行路由由 ``resolve_executor`` 确定性决定：标准内置工具可完成
        时直接调用 ``code_agent``（跳过主管一层，省去一次模型往返）；
        声明了非标准工具时仍由主管 Agent 选择工具完成执行。

        Args:
            spec: 子任务规格（同一计划版本内）。
            agent_record: 本次执行使用的 Agent 记录（原始或替代 Agent）。

        Returns:
            ExecutionResult：解析成功的结果；输出无法解析为契约时返回
            ``success=False`` 的结果（由调度器按失败尝试计数并推进升级
            流程）。

        Raises:
            Exception: 控制层异常（超时 / 网络 / 限流 / 权限等）原样向
            上穿透（任务 69 信任边界）：必须由调度器按异常对象高信任
            分类（classify_exception），不得在本层吞转为结果文本。
            asyncio.CancelledError / PipelineStoppedError 属
            BaseException，天然穿透，不受影响。
        """
        with usage_scope(
            self._task_request.task_id,
            spec.subtask_id,
            agent_record.attempt_count,
        ):
            return await self._execute_subtask_attempt_scoped(spec, agent_record)

    async def _execute_subtask_attempt_scoped(
        self, spec: SubtaskSpec, agent_record: AgentRecord
    ) -> ExecutionResult:
        """在用量归因作用域内执行一次尝试并应用动作级修复。"""
        logger.info(
            "开始执行子任务 [%s] %s（角色: %s，尝试: %s）",
            spec.subtask_id,
            spec.title,
            spec.role,
            agent_record.attempt_count,
        )
        sandbox_root = Path(self._context.sandbox_dir)
        staged = stage_cached_scripts(
            sandbox_root,
            self._task_request.objective,
            task_id=self._task_request.task_id,
            settings=self._settings,
            events=self._events,
        )
        result, declared_success = await self._dispatch_and_enforce(
            spec, agent_record, staged_scripts=staged
        )
        # 动作级修复：仅在“自述成功却被确定性验收翻转”时触发——这类
        # 失败通常只差补齐产物/完成真实调用，同一个尝试内定向重跑一次
        # 比整轮业务重试便宜；真正的执行失败仍交给调度器的恢复策略
        if declared_success and not result.success:
            result = await self._attempt_action_repair(spec, agent_record, result)
        if result.success:
            record_verified_scripts(
                sandbox_root,
                self._task_request.objective,
                [
                    *result.artifacts,
                    *(
                        entry
                        for entry in spec.write_scope
                        if str(entry).lower().endswith(".py")
                    ),
                ],
                task_id=self._task_request.task_id,
                settings=self._settings,
                events=self._events,
            )
        status = "成功" if result.success else f"失败：{result.error or '未知'}"
        logger.info("子任务 [%s] 执行完成 → %s", spec.subtask_id, status)
        return result

    async def _dispatch_and_enforce(
        self,
        spec: SubtaskSpec,
        agent_record: AgentRecord,
        *,
        staged_scripts: list[dict[str, str]] | None = None,
    ) -> tuple[ExecutionResult, bool]:
        """按确定性路由执行一次，并应用全部控制层验收。

        Returns:
            (验收后的结果, Agent 自述是否成功)。后者用于区分真正的
            执行失败与“自述成功但被验收翻转”，只有后者值得动作级修复。
        """
        final_text, route = await self._dispatch_subtask_output(
            spec, agent_record, staged_scripts=staged_scripts
        )
        if route == "direct_code_agent":
            parsed = self._parse_code_agent_result(final_text, spec)
        else:
            parsed = self._parse_execution_result(final_text, spec)
        declared_success = parsed.success
        result = self._enforce_write_scope(
            spec, parsed, since=self._run_started_at
        )
        result = self._enforce_tool_evidence(
            spec, result, agent_record.attempt_count
        )
        result = self._enforce_verification(spec, result)
        return result, declared_success

    async def _dispatch_subtask_output(
        self,
        spec: SubtaskSpec,
        agent_record: AgentRecord,
        *,
        staged_scripts: list[dict[str, str]] | None = None,
    ) -> tuple[str, str]:
        """执行一次子任务并返回（输出文本, 实际路由）。

        路由为 ``direct_code_agent`` 时直接调用 code_agent MCP 工具；
        直接调用不可用（Server 未挂载）时回退主管路径，行为与改造前一致。
        """
        override = self._merged_turn_budget(spec)
        if override is None:
            return await self._dispatch_subtask_routed(
                spec, agent_record, staged_scripts=staged_scripts
            )
        with model_turn_override(override):
            return await self._dispatch_subtask_routed(
                spec, agent_record, staged_scripts=staged_scripts
            )

    def _merged_turn_budget(self, spec: SubtaskSpec) -> int | None:
        """返回合并子任务的回合上限（未合并或余量为 0 时为 None）。"""
        bonus = int(getattr(self._settings, "plan_merge_turn_bonus", 0) or 0)
        if bonus <= 0:
            return None
        if spec.subtask_id not in self._state.merged_subtask_ids:
            return None
        return int(self._settings.max_model_turns) + bonus

    async def _dispatch_subtask_routed(
        self,
        spec: SubtaskSpec,
        agent_record: AgentRecord,
        *,
        staged_scripts: list[dict[str, str]] | None = None,
    ) -> tuple[str, str]:
        """按确定性路由执行一次子任务（不含回合上限作用域）。"""
        routing = resolve_executor(spec.allowed_tools)
        if (
            bool(getattr(self._settings, "subtask_direct_code_agent", True))
            and routing["tool"] == EXECUTOR_CODE_AGENT
        ):
            direct = await self._dispatch_code_agent(
                spec, agent_record, staged_scripts=staged_scripts
            )
            if direct is not None:
                return direct, "direct_code_agent"
        payload = self._build_supervisor_payload(spec, agent_record)
        try:
            output = await self._run_supervisor(payload, ExecutionOutput)
        except Exception as exc:
            # 控制层异常穿透（任务 69）：打印审计日志后原样上抛，由调度
            # 器统一捕获并按异常对象高信任分类——本层不再吞转为
            # “主管 Agent 调用失败：…” 的结果文本（那会让真实系统失败
            # 降级为低信任文本分类，丢失恢复方向所需的结构化证据）
            logger.warning(
                "子任务 [%s] 执行失败：%s", spec.subtask_id, type(exc).__name__
            )
            raise
        return output.final_result, "supervisor"

    def _build_supervisor_payload(
        self, spec: SubtaskSpec, agent_record: AgentRecord
    ) -> dict[str, Any]:
        """构造主管执行载荷（非标准工具路径 / 直派回退路径共用）。"""
        grant = self._plan_grant_map.get(spec.subtask_id)
        # 载荷瘦身（阶段2）：任务目标/约束/验收标准是执行所需的最小
        # 上下文；AgentRecord 与 ToolGrant 只保留主管路由与转交授权所需
        # 的字段（时间戳、状态、内部统计不参与决策，只会放大固定开销）。
        payload: dict[str, Any] = {
            "phase": "execute_subtask",
            "task_id": self._task_request.task_id,
            "task_request": {
                "task_id": self._task_request.task_id,
                "objective": self._task_request.objective,
                "constraints": list(self._task_request.constraints),
                "acceptance_criteria": list(self._task_request.acceptance_criteria),
            },
            "subtask": spec.model_dump(mode="json"),
            "agent": {
                "agent_id": agent_record.agent_id,
                "agent_type": agent_record.agent_type,
                "attempt_count": agent_record.attempt_count,
                "allowed_tools": list(agent_record.allowed_tools),
                "can_request_replacement": agent_record.can_request_replacement,
                "replacement_of": agent_record.replacement_of,
            },
            "attempt": agent_record.attempt_count,
            "grant": (
                {
                    "grant_id": grant.grant_id,
                    "task_id": grant.task_id,
                    "agent_id": grant.agent_id,
                    "allowed_tools": list(grant.allowed_tools),
                    "issued_by": grant.issued_by,
                }
                if grant is not None
                else None
            ),
            # 确定性执行器路由：标准内置工具可完成时不创建动态子 Agent
            "routing": resolve_executor(spec.allowed_tools),
        }
        return payload

    async def _dispatch_code_agent(
        self,
        spec: SubtaskSpec,
        agent_record: AgentRecord,
        *,
        staged_scripts: list[dict[str, str]] | None = None,
    ) -> str | None:
        """直接调用 code_agent MCP 工具（标准工具子任务跳过主管一层）。

        绕过主管意味着控制的确定性更高、也少一次模型往返，但工具证据
        事件必须由本层补齐：嵌套 Agent 的流式运行不携带事件的审计上下文，
        调度器的确定性工具证据验收依赖这里写入的 invoke_completed。

        Returns:
            工具的 JSON 文本输出；Server 未挂载时返回 None（回退主管路径）。
        """
        get_server = getattr(self._tool_registry, "get_mcp_server", None)
        server = get_server("code_agent") if callable(get_server) else None
        if server is None:
            logger.warning(
                "子任务 [%s] 直派路由不可用（code_agent Server 未挂载），回退主管路径",
                spec.subtask_id,
            )
            return None
        arguments: dict[str, Any] = {
            "task_description": self._code_agent_task_description(
                spec, staged_scripts
            ),
            "action": "auto",
        }
        file_path = self._single_write_scope_file(spec)
        if file_path is not None:
            arguments["file_path"] = file_path
        if spec.write_scope:
            # 写时校验：code_agent 在工具层直接拒绝越界路径，而不是等到
            # 事后验收再翻转失败
            arguments["write_scope"] = list(spec.write_scope)
        base_event: dict[str, Any] = {
            "task_id": self._task_request.task_id,
            "subtask_id": spec.subtask_id,
            "attempt": agent_record.attempt_count,
            "agent": agent_record.agent_id,
            "tool_name": "code_agent",
            "route": "direct_code_agent",
            # 证据配对键：与主管路径的 call_id 语义一致，
            # 缺失会让 invoke_started/completed 无法按调用配对
            "call_id": f"call_{uuid.uuid4().hex[:24]}",
        }
        # 合并步骤的回合上限抬高属于控制面决策，写入事件便于事后核对
        turn_budget = self._merged_turn_budget(spec)
        if turn_budget is not None:
            base_event["max_turns"] = turn_budget
        self._events.log_event(
            EVENT_TOOL_CALLED,
            {
                **base_event,
                "action": "invoke_started",
                "arguments_preview": json.dumps(arguments, ensure_ascii=False)[:800],
            },
        )
        started = time.monotonic()
        text = ""
        error_message: str | None = None
        try:
            response = await server.handle_request(
                MCPRequest(
                    method=METHOD_TOOLS_CALL,
                    params={"name": "code_agent", "arguments": arguments},
                    id=f"direct-{spec.subtask_id}-{agent_record.attempt_count}",
                )
            )
            if response.error is not None:
                error_message = f"[{response.error.code}] {response.error.message}"
            else:
                text = extract_result_text(response)
        except Exception as exc:
            error_message = f"{type(exc).__name__}: {exc}"
        duration = round(max(0.0, time.monotonic() - started), 3)
        self._events.log_event(
            EVENT_TOOL_CALLED,
            {
                **base_event,
                "action": (
                    "invoke_completed" if error_message is None else "invoke_failed"
                ),
                "duration": duration,
                "result_ok": error_message is None,
                "output_preview": (text or error_message or "")[:800],
            },
        )
        if error_message is not None:
            # 以异常形式上抛：调度器按异常对象做高信任分类（超时 / 网络
            # / 限流等真实系统失败不会被降级为低信任结果文本）
            raise RuntimeError(f"code_agent 直派调用失败：{error_message}")
        return text

    def _code_agent_task_description(
        self,
        spec: SubtaskSpec,
        staged_scripts: list[dict[str, str]] | None = None,
    ) -> str:
        """把子任务规格压缩为 code_agent 的任务描述。

        只传当前子任务的指令与约束。任务总目标会重复列出全部交付物，
        直派时会让执行器越界重做其他子任务，既放大输出也拖慢时延。
        """
        parts: list[str] = [spec.title, spec.instructions]
        if staged_scripts:
            entries = "、".join(
                f"{item['path']}（sha256 {item['sha256'][:12]}）"
                for item in staged_scripts
                if item.get("path")
            )
            if entries:
                parts.append(
                    "可复用脚本（控制层已放入沙箱并校验哈希）："
                    f"{entries}。若输入格式与本次数据一致，先 read_file 核对后"
                    "直接运行或做最小修改，不要重写整份脚本；不一致则忽略。"
                )
        if spec.expected_output:
            parts.append(f"期望产出：{spec.expected_output}")
        if spec.write_scope:
            parts.append("允许写入：" + "、".join(spec.write_scope))
        if spec.allowed_tools:
            # 授权范围是硬约束：不在列表内的工具（尤其是 run_in_sandbox）
            # 一旦使用，结果评审会判为越权并触发整轮重试
            parts.append("本子任务仅授权工具：" + "、".join(spec.allowed_tools))
        constraints = list(getattr(self._task_request, "constraints", []) or [])
        if constraints:
            parts.append("执行约束：" + "；".join(constraints))
        return "\n".join(part.strip() for part in parts if part and part.strip())

    @staticmethod
    def _single_write_scope_file(spec: SubtaskSpec) -> str | None:
        """返回唯一的具象写入路径（含通配符或多条目时返回 None）。"""
        entries = [
            entry
            for entry in (
                _normalize_write_scope_entry(item) for item in spec.write_scope
            )
            if entry is not None
        ]
        if len(entries) != 1:
            return None
        if any(char in entries[0] for char in _WRITE_SCOPE_GLOB_CHARS):
            return None
        return entries[0]

    @staticmethod
    def _parse_code_agent_result(
        final_text: str, spec: SubtaskSpec
    ) -> ExecutionResult:
        """把 code_agent 的 CodeAgentOutput JSON 转换为 ExecutionResult。"""
        candidate = extract_json_object(final_text or "")
        if not isinstance(candidate, dict):
            summary = (final_text or "").strip()
            return ExecutionResult(
                subtask_id=spec.subtask_id,
                success=False,
                error=(
                    "未能从 code_agent 输出解析出结构化 JSON；原文："
                    f"{summary[:200]!r}"
                ),
            )
        success = candidate.get("success")
        success = success if isinstance(success, bool) else False
        error = candidate.get("error")
        error_text = error if isinstance(error, str) and error.strip() else None
        execution = candidate.get("execution_result")
        # 沙箱执行失败是硬事实：不得因模型自述 success 而放行
        if (
            success
            and isinstance(execution, dict)
            and execution.get("success") is False
        ):
            success = False
            exec_error = execution.get("error")
            error_text = (
                exec_error
                if isinstance(exec_error, str) and exec_error.strip()
                else "沙箱执行失败"
            )
        action_performed = candidate.get("action_performed")
        summary_text = candidate.get("summary")
        output_text = next(
            (
                text.strip()
                for text in (summary_text, action_performed)
                if isinstance(text, str) and text.strip()
            ),
            "",
        )
        if not output_text:
            output_text = json.dumps(candidate, ensure_ascii=False)
        file_path = candidate.get("file_path")
        artifacts = (
            [file_path]
            if isinstance(file_path, str) and file_path.strip()
            else []
        )
        evidence: list[str] = []
        declared_evidence = candidate.get("evidence")
        if isinstance(declared_evidence, list):
            evidence.extend(
                item.strip()
                for item in declared_evidence
                if isinstance(item, str) and item.strip()
            )
        if isinstance(action_performed, str) and action_performed.strip():
            evidence.append(action_performed.strip())
        metadata: dict[str, Any] = {"route": "direct_code_agent"}
        if isinstance(execution, dict):
            status = execution.get("status")
            if isinstance(status, str):
                evidence.append(f"execution:{status}")
            # 脚本打印的结构化结果常在这里；无 path 的 json_field /
            # numeric_match 验收规则需要它作为可机检证据
            exec_output = execution.get("output")
            if isinstance(exec_output, str) and exec_output.strip():
                metadata["execution_output"] = exec_output
            # run_in_sandbox 把退出码放在 execution_result.metadata 内；
            # 不提升到顶层时 exit_code 验收规则会判定“缺少证据”并无限重试
            exec_meta = execution.get("metadata")
            if isinstance(exec_meta, dict) and "exit_code" in exec_meta:
                metadata["exit_code"] = exec_meta["exit_code"]
        return ExecutionResult(
            subtask_id=spec.subtask_id,
            success=success,
            output=output_text,
            error=(
                None
                if success
                else (error_text or "code_agent 未完成本次动作")
            ),
            evidence=evidence,
            artifacts=artifacts,
            metadata=metadata,
        )

    async def _attempt_action_repair(
        self,
        spec: SubtaskSpec,
        agent_record: AgentRecord,
        failed: ExecutionResult,
    ) -> ExecutionResult:
        """在同一尝试内做一次定向动作修复（仅修正被翻转的验收点）。

        不占额外业务尝试额度：修复轮沿用当前 attempt 序号，只附加
        针对性指令；修复仍失败时把两次失败原因合并返回，交调度器按
        既有恢复策略处理（同因抑制 / 重规划）。
        """
        limit = int(
            getattr(self._settings, "subtask_action_repair_max_attempts", 0) or 0
        )
        if limit <= 0:
            return failed
        repair_spec = spec.model_copy(deep=True)
        repair_spec.instructions += (
            "\n\n[动作级修复] 上一次执行自述成功，但未通过确定性验收："
            f"{failed.error or '未知原因'}。\n"
            "本次只修正上述问题：补齐声明的产物、完成真实的文件写入或"
            "沙箱执行，使验收条件成立；不要重做已经完成的部分，也不要在"
            "未产出时声称成功。\n"
        )
        repaired, _ = await self._dispatch_and_enforce(repair_spec, agent_record)
        self._events.log_event(
            EVENT_ACTION_REPAIR,
            {
                "task_id": self._task_request.task_id,
                "subtask_id": spec.subtask_id,
                "attempt": agent_record.attempt_count,
                "first_failure": (failed.error or "")[:300],
                "repair_success": repaired.success,
                "repair_failure": (repaired.error or "")[:300],
            },
        )
        logger.info(
            "子任务 [%s] 动作级修复 → %s",
            spec.subtask_id,
            "成功" if repaired.success else "仍未通过",
        )
        if repaired.success:
            return repaired
        return repaired.model_copy(
            update={
                "error": (
                    f"{failed.error or '验收未通过'}；动作级修复后仍未通过："
                    f"{repaired.error or '未知原因'}"
                )
            }
        )

    def _subtask_budget_violation(self, subtask_id: str) -> str | None:
        """子任务 token 硬预算结算（调度器每次尝试前调用）。"""
        limit = int(getattr(self._settings, "subtask_max_tokens", 0) or 0)
        if limit <= 0:
            return None
        budget = getattr(self._context, "token_budget", None)
        if budget is None or not hasattr(budget, "tokens_for_subtask"):
            return None
        observed = int(budget.tokens_for_subtask(subtask_id))
        if observed <= limit:
            return None
        return (
            f"子任务 {subtask_id} token 用量 {observed} 超过上限 {limit}，"
            "已触发子任务预算熔断"
        )

    def _enforce_write_scope(
        self, spec: SubtaskSpec, result: ExecutionResult, *, since: float | None = None
    ) -> ExecutionResult:
        """确定性验收 write_scope：自述成功但声明文件未落地时翻转为失败。

        子 Agent 的成功标记是低信任文本，控制层以文件系统事实为准：
        声明的每个写入路径必须在沙箱根目录下存在且非空（文件、目录内
        至少一个非空文件，或通配符至少匹配一个非空文件）；传入
        ``since`` 时还要求产出不早于本次运行，避免上一进程的残留文件
        冒充本轮结果。缺失时返回失败结果，使调度器按既有重试 / 重规划
        链路处理，而不是把空壳成功带进汇总与结果评审。
        """
        if not result.success or not spec.write_scope:
            return result
        root = Path(self._context.sandbox_dir).expanduser().resolve()
        missing = missing_write_scope(root, spec.write_scope, since=since)
        if not missing:
            return result
        self._events.log_event(
            EVENT_WRITE_SCOPE_UNMET,
            {
                "task_id": self._task_request.task_id,
                "subtask_id": spec.subtask_id,
                "declared": list(spec.write_scope),
                "missing": missing,
            },
        )
        logger.warning(
            "子任务 [%s] 自述成功但写入范围未产出：%s",
            spec.subtask_id,
            "、".join(missing),
        )
        return result.model_copy(
            update={
                "success": False,
                "error": (
                    f"写入范围未产出：{'、'.join(missing)}；"
                    "子任务自述成功但声明的文件在沙箱中不存在"
                ),
            }
        )

    def _enforce_tool_evidence(
        self, spec: SubtaskSpec, result: ExecutionResult, attempt: int
    ) -> ExecutionResult:
        """确定性验收工具证据：声明产出却没有完成的工具调用时翻转失败。

        仅约束“结果声明了文件/产物”的成功子任务：这类产出不可能在
        零工具调用的情况下真实产生，因而本次尝试必须存在至少一条
        ``invoke_completed`` 的工具生命周期事件（按 subtask_id 与
        attempt 精确过滤，避免复用上一轮尝试或并发子任务的证据）。
        """
        if not result.success:
            return result
        if not getattr(self._settings, "subtask_tool_evidence_required", True):
            return result
        declared = list(spec.write_scope or [])
        artifacts = list(result.artifacts or [])
        if not declared and not artifacts:
            return result
        task_id = self._task_request.task_id
        try:
            records = self._events.get_events(
                task_id=task_id, event_type=EVENT_TOOL_CALLED
            )
        except Exception:  # 审计通道不可用时不以证据缺失阻断任务
            logger.warning(
                "子任务 [%s] 工具证据查询失败，跳过证据验收", spec.subtask_id
            )
            return result
        for record in records:
            payload = record.get("payload") or {}
            if payload.get("subtask_id") != spec.subtask_id:
                continue
            if payload.get("attempt") != attempt:
                continue
            if payload.get("action") != "invoke_completed":
                continue
            if payload.get("result_ok") is False:
                continue
            return result
        self._events.log_event(
            EVENT_TOOL_EVIDENCE_MISSING,
            {
                "task_id": task_id,
                "subtask_id": spec.subtask_id,
                "attempt": attempt,
                "declared": declared,
                "artifacts": artifacts[:20],
            },
        )
        logger.warning(
            "子任务 [%s] 缺少本次尝试的工具调用证据（attempt=%s）",
            spec.subtask_id,
            attempt,
        )
        return result.model_copy(
            update={
                "success": False,
                "error": (
                    "缺少工具调用证据：结果声明了产出，但本次尝试没有"
                    "任何完成的工具调用"
                ),
            }
        )

    def _enforce_verification(
        self, spec: SubtaskSpec, result: ExecutionResult
    ) -> ExecutionResult:
        """确定性验收：运行规划器声明的可机检规则，任一失败即翻转。

        规则只读沙箱文件与本次执行结果；没有声明的子任务不受影响。
        失败原因逐条写入审计事件，便于返工与人工排查。
        """
        if not result.success or not spec.verification:
            return result
        root = Path(self._context.sandbox_dir).expanduser().resolve()
        try:
            failures = evaluate_verification_rules(spec, result, root)
        except Exception as exc:  # 规则求值异常按验收失败处理（fail-closed）
            failures = [f"verification 规则求值异常：{type(exc).__name__}: {exc}"]
        if not failures:
            return result
        self._events.log_event(
            EVENT_VERIFICATION_FAILED,
            {
                "task_id": self._task_request.task_id,
                "subtask_id": spec.subtask_id,
                "failures": failures[:20],
            },
        )
        logger.warning(
            "子任务 [%s] 自述成功但可机检验收未通过：%s",
            spec.subtask_id,
            "；".join(failures[:5]),
        )
        return result.model_copy(
            update={
                "success": False,
                "error": "可机检验收未通过：" + "；".join(failures),
            }
        )

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
            # 预算检查点：批次开始前结算已累计用量（工作流引擎注入）
            self._enforce_token_budget()
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
        decision = self._parallelism_decision
        try:
            batch = await self._scheduler.schedule(
                self._task_request.task_id,
                executable,
                concurrency=(
                    decision.concurrency if decision is not None else None
                ),
                breaker=self._runtime_breaker,
                # 续跑/返工批次可能只含未完成项，其依赖已在上批完成
                prior_results=dict(self._results),
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
        self._report_progress(
            f"batch_completed:{len(merged)}", checkpoint=True
        )
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

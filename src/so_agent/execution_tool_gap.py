"""工具缺口生命周期（ExecutionCoordinator 混入）。

检测未就绪的 allowed_tools → 代码 Agent 创建候选 → 安全准入 →
沙箱逐用例验收 → 评审 Agent 审批 → 注册并补发授权。
"""

from __future__ import annotations

import math
import re
from typing import Any

from pydantic import ValidationError

import so_agent.tool_packages.code_agent.agent as code_agent_agent_mod
from so_agent.json_utils import extract_json_object
from so_agent.mcp.client import MCPClient, MCPClientError
from so_agent.mcp.protocol import extract_result_text
from so_agent.models import ExecutionResult, SubtaskSpec
from so_agent.runtime.events import EVENT_TOOL_CALLED
from so_agent.runtime.generated_tool_runner import (
    GeneratedToolExecutionError,
    run_generated_tool_in_sandbox,
)
from so_agent.runtime.logging_setup import get_logger
from so_agent.runtime.registry import DEFAULT_SUPERVISOR_ID
from so_agent.runtime.sandbox import create_sandbox
from so_agent.runtime.tool_registry import ToolRegistryError
from so_agent.runtime.tool_safety import assess_tool_source
from so_agent.runtime.tool_store import stage_tool_for_sandbox
from so_agent.tool_packages.code_agent.agent import (
    CodeAction,
    CodeAgentInput,
    CodeAgentOutput,
)
from so_agent.tool_packages.review_agent.agent import ReviewInput, ReviewStage

logger = get_logger("execution.tool_gap")

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


class ToolGapMixin:
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
        """工具是否已注册、已批准，且生成工具已通过安全准入与沙箱验收。"""
        return self._tool_registry.is_approved(
            tool_name
        ) and self._tool_registry.is_safety_cleared(tool_name)

    def _sync_generated_registry(self) -> str | None:
        """把代码 Agent 落盘的生成工具清单同步进工具注册表。

        Returns:
            None 表示同步成功（或清单尚不存在）；否则返回错误说明。
        """
        registry_path = code_agent_agent_mod.resolve_generated_registry_path(
            self._context, self._settings
        )
        try:
            if registry_path.is_file():
                self._tool_registry.load_generated_registry(registry_path)
        except ToolRegistryError as exc:
            return str(exc)
        return None

    def _ensure_tool_safety(
        self, tool_name: str, manifest
    ) -> str | None:
        """对候选工具执行静态安全检查并写回结论。

        Returns:
            None 表示通过或此前已通过；否则返回拒绝原因。
        """
        if manifest.safety_status == "passed":
            return None
        tools_dir = code_agent_agent_mod.resolve_generated_tools_dir(
            self._context, self._settings
        )
        source_path = tools_dir / manifest.entry_file
        try:
            source = source_path.read_text(encoding="utf-8")
        except OSError as exc:
            updated = manifest.model_copy(
                update={
                    "safety_status": "failed",
                    "safety_issues": [f"无法读取工具源码：{exc}"],
                }
            )
            self._persist_manifest(updated)
            return f"工具安全检查未通过：无法读取源码 {source_path}"
        assessment = assess_tool_source(source, tool_name=tool_name)
        updated = manifest.model_copy(
            update={
                "safety_status": "passed" if assessment.passed else "failed",
                "safety_issues": assessment.issues + assessment.warnings,
            }
        )
        self._persist_manifest(updated)
        if assessment.passed:
            return None
        return "工具安全检查未通过：" + "；".join(assessment.issues)

    def _persist_manifest(self, manifest) -> None:
        """把清单更新写回注册表与 registry.json（尽力而为）。"""
        try:
            self._tool_registry.register_generated_tool(manifest)
            self._tool_registry.save_generated_registry(
                code_agent_agent_mod.resolve_generated_registry_path(
                    self._context, self._settings
                )
            )
        except ToolRegistryError as exc:
            logger.warning("工具清单写回失败（%s）：%s", manifest.tool_name, exc)

    def _ensure_mcp_client(self) -> MCPClient:
        """返回 MCP 客户端，并补齐注册表中新增的 Server。

        MCP 客户端会被缓存：工具缺口流程在注册新生成工具前后都会复用
        同一实例，因此每次调用都必须把注册表中尚未连接的 Server 补上，
        否则同任务内刚注册的工具不可调用。
        """
        client = self._mcp_client if self._mcp_client is not None else MCPClient()
        for name in self._tool_registry.list_tool_names():
            if client.get_server(name) is not None:
                continue
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

        # 安全准入：静态检查不通过（网络/进程/动态执行/破坏性操作）直接拒绝
        safety_error = self._ensure_tool_safety(tool_name, manifest)
        if safety_error is not None:
            self._events.log_event(
                EVENT_TOOL_CALLED,
                {
                    **base_event,
                    "action": "gap_unresolved",
                    "reason": safety_error,
                },
            )
            return safety_error
        manifest = self._tool_registry.get_manifest(tool_name) or manifest

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
                code_agent_agent_mod.resolve_generated_registry_path(
                    self._context, self._settings
                )
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
            if getattr(self._settings, "tool_require_test_cases", True):
                reason = (
                    "验收用例缺失：生成工具必须提供 test_cases 与 "
                    "expected_outputs 才能准入"
                )
                self._events.log_event(
                    EVENT_TOOL_CALLED,
                    {**base, "action": "tool_verify_skipped", "reason": reason},
                )
                return manifest, reason
            # 显式关闭强制用例（受控调试）：保守跳过运行验证
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
        sandbox = create_sandbox(context=self._context, settings=self._settings)
        entry = (
            code_agent_agent_mod.resolve_generated_tools_dir(
                self._context, self._settings
            )
            / manifest.entry_file
        )
        execution_entry = await stage_tool_for_sandbox(sandbox, entry)
        for index, args in enumerate(test_cases):
            try:
                result = await run_generated_tool_in_sandbox(
                    sandbox,
                    execution_entry,
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

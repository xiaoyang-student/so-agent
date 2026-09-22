"""工作流状态机（WorkflowEngine）。

本模块是整个框架的控制层核心，实现完整的任务生命周期管理：

- 状态流转严格遵循蓝图第四节：
  ``received → planning → plan_review → executing → aggregating →
  result_review → completed``；执行失败分支：
  ``executing → replacement_requested → replacement_executing →
  decomposition_issue → replanning → plan_review``（循环）；
  终态：``completed / human_required / failed / cancelled``。
- 两阶段评审：计划评审（含授权最小化评审）与结果评审，均由评审
  Agent 给出 ``ReviewDecision`` 结构化结论；
- 失败升级：单 Agent 最多三次尝试（由调度器执行）、一次替代申请、
  替代 Agent 无再申请资格、替代仍失败触发任务拆分复查、主管最多
  三次拆分修改，耗尽后进入人工兜底（首版仅接口与状态）；
- 全程 JSON 通信：对主管 Agent / 评审 Agent 的一切调用输入输出均为
  Pydantic 模型序列化；替换申请与分解问题经 SupervisorRouter 留痕；
- 状态迁移一律通过 EventLogger 记录（``status_transition`` 事件），
  供审计与人工升级材料组装。

调用示例::

    context = ProjectContext(sandbox_dir=Path("sandbox"), project_name="demo")
    engine = WorkflowEngine(context)
    output = await engine.run("为指定目录实现一个 CLI 工具")
    print(output.model_dump_json(indent=2))

说明：
- 调度器（TaskScheduler）内部完成"DAG 调度 + 三次尝试 + 一次替换请求 +
  替代执行 + 分解问题上报"；本引擎在调度完成后把该过程映射到状态机
  状态链（replacement_requested → replacement_executing → …），并负责
  拆分修改额度的消耗与人工兜底；
- 首版中"主管检查失败证据后同意替换"由控制层自动批准（证据随
  ReplacementRequest 全程留痕），不额外消耗模型调用。
"""

from __future__ import annotations

import asyncio
import json
import re
import sys
import uuid
from typing import Any, TypeVar

import openai
from agents import Agent, Model, Runner
from pydantic import BaseModel, ValidationError

import so_agent.tool_packages.code_agent.agent as code_agent_agent_mod
from so_agent.config import Settings
from so_agent.context import ProjectContext
from so_agent.json_utils import extract_json_object
from so_agent.models import (
    AgentRecord,
    ExecutionResult,
    HumanEscalationRequest,
    ReplacementRequest,
    ReviewDecision,
    SubtaskSpec,
    TaskDecompositionIssue,
    TaskRequest,
    TaskStatus,
    ToolGrant,
)
from so_agent.mcp.client import MCPClient, MCPClientError
from so_agent.mcp.protocol import extract_result_text
from so_agent.orchestrator import (
    AggregationOutput,
    ExecutionOutput,
    OrchestratorInput,
    OrchestratorOutput,
    PlanningOutput,
    create_orchestrator,
)
from so_agent.runtime.events import (
    EVENT_AGENT_CREATED,
    EVENT_AGENT_RELEASED,
    EVENT_DECOMPOSITION_ISSUE,
    EVENT_HUMAN_ESCALATION,
    EVENT_REPLAN,
    EVENT_REPLACEMENT_REQUESTED,
    EVENT_STATUS_TRANSITION,
    EVENT_TASK_CREATED,
    EVENT_TASK_DISPATCHED,
    EVENT_TOOL_CALLED,
    EVENT_TOOL_GRANTED,
    EVENT_TOOL_REVOKED,
    EventLogger,
)
from so_agent.runtime.permissions import PermissionGateway
from so_agent.runtime.registry import DEFAULT_SUPERVISOR_ID, AgentRegistry
from so_agent.runtime.scheduler import SchedulerError, TaskScheduler
from so_agent.runtime.supervisor_router import CommunicationError, SupervisorRouter
from so_agent.runtime.tool_registry import ToolRegistry, ToolRegistryError
from so_agent.tool_packages.code_agent.agent import (
    CodeAction,
    CodeAgentInput,
    CodeAgentOutput,
)
from so_agent.tool_packages.review_agent.agent import (
    ReviewInput,
    ReviewStage,
    create_review_agent,
)

# 主管 Agent 的三个固定工具入口：作为路由的合法接收方；子任务的
# allowed_tools 允许出现未注册的工具名（作为"工具缺口"），由执行阶段的
# 缺口流程（创建 → 评审 → 注册 → 授权）解决
_TOOL_UNIVERSE: tuple[str, ...] = ("code_agent", "subagent_creator", "review_agent")

# 主管 Agent 网络异常类型：请求超时与连接失败。区别于计划结构问题——
# 网络失败不计入“任务拆分修改额度”，按独立预算重试（见 _plan_and_review）
_SUPERVISOR_NETWORK_ERRORS: tuple[type[Exception], ...] = (
    openai.APITimeoutError,
    openai.APIConnectionError,
)

# 网络重试预算（独立于任务拆分修改额度）：主管 Agent 调用连续网络失败
# （超时 / 连接错误）超过该次数时转入人工兜底（网络连续失败）
_MAX_NETWORK_RETRIES: int = 3

# 动态生成工具名称规范（与代码 Agent 侧保持一致）
_TOOL_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,63}$")

# 终态集合：到达后不允许继续迁移
_TERMINAL_STATUSES: frozenset[TaskStatus] = frozenset(
    {
        TaskStatus.COMPLETED,
        TaskStatus.HUMAN_REQUIRED,
        TaskStatus.FAILED,
        TaskStatus.CANCELLED,
    }
)

# 各执行阶段的进度提示（打印到 stderr，供用户实时观察工作流进展）
_STATUS_PROGRESS_MESSAGES: dict[TaskStatus, str] = {
    TaskStatus.PLANNING: "正在拆解任务...",
    TaskStatus.REPLANNING: "正在重新拆解任务...",
    TaskStatus.PLAN_REVIEW: "正在评审计划...",
    TaskStatus.EXECUTING: "正在执行子任务...",
    TaskStatus.REPLACEMENT_REQUESTED: "正在申请替代 Agent...",
    TaskStatus.REPLACEMENT_EXECUTING: "正在执行替代 Agent...",
    TaskStatus.AGGREGATING: "正在汇总结果...",
    TaskStatus.RESULT_REVIEW: "正在评审结果...",
}

# 失败/中断类终态：迁移原因作为错误信息一并打印到 stderr
_ERROR_STATUSES: frozenset[TaskStatus] = frozenset(
    {
        TaskStatus.FAILED,
        TaskStatus.CANCELLED,
        TaskStatus.HUMAN_REQUIRED,
    }
)


def _log_progress(status: TaskStatus, reason: str = "") -> None:
    """把状态迁移的进度提示或错误信息打印到标准错误（stderr）。"""
    message = _STATUS_PROGRESS_MESSAGES.get(status)
    if message:
        print(f"[so-agent] {message}", file=sys.stderr, flush=True)
    if status in _ERROR_STATUSES:
        detail = (reason or "").strip() or "未提供原因"
        print(
            f"[so-agent] 任务进入 {status.value} 状态：{detail}",
            file=sys.stderr,
            flush=True,
        )

# 计入"工具调用证据"的事件类型（人工升级材料组装用）
_EVIDENCE_EVENT_TYPES: frozenset[str] = frozenset(
    {
        EVENT_TASK_DISPATCHED,
        EVENT_TOOL_GRANTED,
        EVENT_TOOL_REVOKED,
        EVENT_AGENT_CREATED,
        EVENT_AGENT_RELEASED,
        EVENT_REPLACEMENT_REQUESTED,
        EVENT_DECOMPOSITION_ISSUE,
    }
)


class WorkflowError(Exception):
    """工作流领域错误：非法输入、主管/评审输出无法解析等。"""


class WorkflowCancelledError(WorkflowError):
    """协作式取消：任务在检查点处被请求取消。"""


class SupervisorNetworkError(WorkflowError):
    """主管 Agent 网络连续失败（超时 / 连接错误）耗尽网络重试预算。

    此类错误由 ``_plan_and_review`` 在独立网络重试预算（
    ``_MAX_NETWORK_RETRIES``）耗尽后抛出，由 ``_run_pipeline`` 捕获并
    转入人工兜底（原因“网络连续失败”），不消耗重规划额度。
    """


class HumanEscalationHandler:
    """人工兜底处理器接口。首版只定义接口和状态，不实现人工处理系统。"""

    def __init__(self) -> None:
        """初始化处理器（首版仅保留已受理的升级请求记录）。"""
        self._requests: list[HumanEscalationRequest] = []

    async def escalate(self, request: HumanEscalationRequest) -> None:
        """首版仅记录事件并设置状态，不实际通知人工。

        Args:
            request: 人工升级材料包（原始任务、历代计划、失败历史与证据）。
        """
        self._requests.append(request)

    async def check_resolution(self, task_id: str) -> bool:
        """首版始终返回 False（人工处理系统尚未实现）。"""
        return False

    @property
    def pending_requests(self) -> list[HumanEscalationRequest]:
        """返回已受理的升级请求列表（浅拷贝，供审计与测试）。"""
        return list(self._requests)


class WorkflowEngine:
    """任务生命周期状态机引擎（控制层）。

    职责：
    - 驱动完整流程：拆解（主管 Agent）→ 计划评审（评审 Agent）→ 调度
      执行（TaskScheduler）→ 失败升级（替换/重规划）→ 汇总（主管 Agent）
      → 结果评审（评审 Agent）→ 终态；
    - 管理治理额度：三次任务拆分修改（含评审重试与分解问题重规划）、
      结果评审返工次数上限；
    - 记录全过程事件并保留全部历史（计划版本、失败历史、工具证据）。
    """

    def __init__(
        self,
        context: ProjectContext,
        *,
        settings: Settings | None = None,
        orchestrator: Agent[Any] | None = None,
        review_agent: Agent[Any] | None = None,
        registry: AgentRegistry | None = None,
        gateway: PermissionGateway | None = None,
        events: EventLogger | None = None,
        router: SupervisorRouter | None = None,
        scheduler: TaskScheduler | None = None,
        escalation_handler: HumanEscalationHandler | None = None,
        supervisor_model: str | Model | None = None,
        tool_registry: ToolRegistry | None = None,
        mcp_client: MCPClient | None = None,
    ) -> None:
        """初始化工作流引擎。

        Args:
            context: 项目共享上下文（沙箱、注册表镜像、事件日志、配置来源）。
            settings: 生效配置；未提供时取 ``context.config``。
            orchestrator: 主管 Agent 覆盖（测试可注入假 Agent）；未提供时
                按 context 调用 ``create_orchestrator`` 创建。
            review_agent: 评审 Agent 覆盖（测试可注入假 Agent）；未提供时
                按 context 调用 ``create_review_agent`` 创建。
            registry: Agent 注册表；未提供时新建并镜像到 context。
            gateway: 工具授权网关；未提供时新建。
            events: 事件记录器；未提供时新建（写入 ``context.event_log``）。
            router: 主管路由器；未提供时新建（通信边界校验与留痕）。
            scheduler: 任务调度器；未提供时按本引擎的执行器新建。
            escalation_handler: 人工兜底处理器；未提供时使用默认占位实现。
            supervisor_model: 主管 Agent 的模型覆盖（透传 create_orchestrator）。
            tool_registry: 工具注册表（MCP 注册中心）；未提供时新建默认实例
                （生成工具目录与 code_agent 工具包对齐）。授权校验与工具
                缺口流程均以该注册表为准，并透传给主管 Agent 装配。
            mcp_client: MCP 客户端；未提供时按需从注册表已挂载的 MCP
                Server 构建（工具缺口流程经它统一调用代码 Agent）。
        """
        self._context = context
        self._settings = settings or context.config
        self._events = events if events is not None else EventLogger(context=context)
        self._registry = (
            registry
            if registry is not None
            else AgentRegistry(context=context, settings=self._settings)
        )
        self._tool_registry = (
            tool_registry
            if tool_registry is not None
            else ToolRegistry(
                generated_tools_dir=code_agent_agent_mod.GENERATED_TOOLS_DIR
            )
        )
        self._mcp_client = mcp_client
        self._gateway = (
            gateway
            if gateway is not None
            else PermissionGateway(
                context=context,
                settings=self._settings,
                events=self._events,
                tool_registry=self._tool_registry,
            )
        )
        self._router = (
            router
            if router is not None
            else SupervisorRouter(
                registry=self._registry,
                events=self._events,
                extra_recipients=_TOOL_UNIVERSE,
            )
        )
        self._escalation_handler = (
            escalation_handler
            if escalation_handler is not None
            else HumanEscalationHandler()
        )
        self._orchestrator = (
            orchestrator
            if orchestrator is not None
            else create_orchestrator(
                context,
                settings=self._settings,
                model=supervisor_model,
                registry=self._registry,
                tool_registry=self._tool_registry,
                mcp_client=self._mcp_client,
            )
        )
        self._review_agent = (
            review_agent
            if review_agent is not None
            else create_review_agent(context, settings=self._settings)
        )
        self._scheduler = (
            scheduler
            if scheduler is not None
            else TaskScheduler(
                self._execute_subtask_attempt,
                context=context,
                settings=self._settings,
                registry=self._registry,
                events=self._events,
            )
        )

        # ---- 状态跟踪（任务级） ----
        self._current_status: TaskStatus = TaskStatus.RECEIVED
        self._plan_version: int = 0
        self._replan_count: int = 0
        self._result_rework_count: int = 0
        self._task_request: TaskRequest | None = None
        self._subtasks: list[SubtaskSpec] = []
        self._pending_specs: list[SubtaskSpec] = []
        self._results: dict[str, ExecutionResult] = {}
        self._plan_history: list[list[SubtaskSpec]] = []
        self._agent_failure_history: list[str] = []
        self._grants: list[ToolGrant] = []
        self._plan_grants: list[ToolGrant] = []
        self._plan_grant_map: dict[str, ToolGrant] = {}
        self._last_review: ReviewDecision | None = None
        self._human_escalation: HumanEscalationRequest | None = None
        self._cancel_requested: bool = False
        # 已处理的替换申请（request_id 去重，避免跨批次重复重建状态链）
        self._seen_replacement_requests: set[str] = set()
        # 已写入失败历史的分解问题指纹（subtask_id, plan_version, reasons）
        self._seen_issue_fingerprints: set[tuple[str, int, tuple[str, ...]]] = set()

    # ------------------------------------------------------------------
    # 只读状态（供外部审计与测试）
    # ------------------------------------------------------------------
    @property
    def current_status(self) -> TaskStatus:
        """返回当前任务状态。"""
        return self._current_status

    @property
    def plan_version(self) -> int:
        """返回当前计划版本号（从 1 开始；尚无计划时为 0）。"""
        return self._plan_version

    @property
    def replan_count(self) -> int:
        """返回已消耗的任务拆分修改次数。"""
        return self._replan_count

    @property
    def task_request(self) -> TaskRequest | None:
        """返回当前任务请求；未接收任务时为 None。"""
        return self._task_request

    @property
    def subtasks(self) -> list[SubtaskSpec]:
        """返回当前计划版本的子任务列表（深拷贝）。"""
        return [spec.model_copy(deep=True) for spec in self._subtasks]

    @property
    def plan_history(self) -> list[list[SubtaskSpec]]:
        """返回历代计划快照（深拷贝）。"""
        return [
            [spec.model_copy(deep=True) for spec in plan] for plan in self._plan_history
        ]

    def cancel(self) -> None:
        """请求取消当前任务（协作式：在下一个检查点生效）。"""
        self._cancel_requested = True

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------
    async def run(
        self,
        objective: str,
        constraints: str = "",
        acceptance_criteria: str = "",
    ) -> OrchestratorOutput:
        """执行一个完整任务的生命周期，直至终态。

        Args:
            objective: 任务目标（期望达成什么）。
            constraints: 约束条件文本；支持换行或分号分隔多条（可选）。
            acceptance_criteria: 验收标准文本；支持换行或分号分隔多条（可选）。

        Returns:
            OrchestratorOutput：包含 task_id、终态 status、final_result、
            plan_versions（历代计划快照）、agent_history（Agent 生命周期）、
            review_summary（最后一次评审结论摘要），以及人工兜底时的
            human_escalation 材料包。

        说明：任何未捕获的系统错误都会把任务置为 ``failed`` 终态并返回
        结构化输出；``asyncio.CancelledError`` 会先记录 ``cancelled``
        状态再向上抛出（遵循 asyncio 取消语义）。
        """
        try:
            return await self._run_pipeline(objective, constraints, acceptance_criteria)
        except asyncio.CancelledError:
            self._transition(TaskStatus.CANCELLED, reason="任务被外部取消（CancelledError）")
            raise
        except WorkflowCancelledError:
            self._transition(TaskStatus.CANCELLED, reason="任务被取消（协作式）")
            return self._build_output(TaskStatus.CANCELLED, final_result="任务已取消")
        except Exception as exc:  # 不可恢复的系统错误：置为 failed 终态
            self._transition(TaskStatus.FAILED, reason=f"{type(exc).__name__}: {exc}")
            return self._build_output(
                TaskStatus.FAILED,
                final_result=f"任务失败（系统错误）：{type(exc).__name__}: {exc}",
            )

    # ------------------------------------------------------------------
    # 完整流水线
    # ------------------------------------------------------------------
    async def _run_pipeline(
        self, objective: str, constraints: str, acceptance_criteria: str
    ) -> OrchestratorOutput:
        """完整流水线：接收 → 拆解/评审 → 执行/失败升级 → 汇总/终审 → 终态。"""
        # 1) 接收任务 → received
        self._create_task(objective, constraints, acceptance_criteria)

        # 2) 拆解 + 计划评审（含重拆循环；额度耗尽转人工兜底）
        try:
            plan_accepted = await self._plan_and_review()
        except SupervisorNetworkError as exc:
            return await self._escalate_to_human(f"网络连续失败：{exc}")
        if not plan_accepted:
            return await self._escalate_to_human("计划评审连续未通过：任务拆分修改额度耗尽")

        self._pending_specs = list(self._subtasks)

        # 3) 执行/汇总/终审循环（含返工）
        while True:
            self._check_cancel()

            # 3.1) 执行直至无失败（失败自动重规划；额度耗尽返回 None）
            try:
                execution = await self._run_execution_cycle()
            except SupervisorNetworkError as exc:
                return await self._escalate_to_human(f"网络连续失败：{exc}")
            if execution is None:
                return await self._escalate_to_human("执行阶段持续失败：任务拆分修改额度耗尽")

            # 3.2) 汇总 → aggregating
            final_text = await self._aggregate_results()

            # 3.3) 最终评审 → result_review
            decision = await self._final_review()
            if decision.passed:
                print(
                    "[so-agent] 结果评审通过，任务完成",
                    file=sys.stderr, flush=True,
                )
                self._transition(TaskStatus.COMPLETED)
                return self._build_output(
                    TaskStatus.COMPLETED, final_result=final_text, review=decision
                )

            # 3.4) 结果评审不通过：仅返工被指出的子任务
            rework_ids = self._resolve_rework_targets(decision)
            if not rework_ids:
                return await self._escalate_to_human("结果评审未通过且无法定位返工目标")
            if self._result_rework_count >= int(self._settings.max_replan_attempts):
                return await self._escalate_to_human("结果评审返工次数耗尽仍未通过")
            self._result_rework_count += 1
            rework_specs = self._build_rework_specs(rework_ids)
            if not rework_specs:
                return await self._escalate_to_human("结果评审未通过且返工目标为空")
            for subtask_id in rework_ids:
                self._results.pop(subtask_id, None)
            self._pending_specs = rework_specs

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

        循环内部：调用主管 Agent 拆解 → 基础校验 → 签发 ToolGrant →
        评审 Agent 评审；评审不通过（或拆解结果非法）时消耗一次
        "任务拆分修改额度"并重新拆解，直到通过或额度耗尽。
        拆解评审重试包含在三次任务拆分修改额度内。

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
        network_retry_count = 0
        while True:
            is_replan = self._plan_version > 0
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

            if not problems:
                self._plan_version += 1
                self._subtasks = self._adopt_plan(subtasks)
                self._plan_history.append(
                    [spec.model_copy(deep=True) for spec in self._subtasks]
                )
                self._issue_grants()
                # --- 输出拆解结果详情 ---
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
            # 去重逻辑不会触发进度提示，这里补充“第 N 次重规划”提示
            if self._plan_version == version_before_attempt:
                max_attempts = int(self._settings.max_replan_attempts)
                print(
                    f"[so-agent] 第 {self._replan_count} 次重规划"
                    f"（拆分修改额度 {self._replan_count}/{max_attempts}）...",
                    file=sys.stderr,
                    flush=True,
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
        """构造拆解/重规划阶段的 JSON 输入（含标准 OrchestratorInput 契约）。"""
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

    def _issue_grants(self) -> list[ToolGrant]:
        """为当前计划的子任务签发工具授权（无工具的子任务不签发空授权）。

        只对"已注册且已批准"的可分配工具签发：工具缺口（未注册 / 未批准）
        不进白名单，待执行阶段缺口流程解决后由本方法补发（重新签发时
        覆盖本版本计划的映射，旧授权不主动撤销）。

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
        for spec in self._subtasks:
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
            )
            self._plan_grants.append(grant)
            self._plan_grant_map[spec.subtask_id] = grant
        self._grants.extend(self._plan_grants)
        return list(self._plan_grants)

    async def _review_plan(self) -> ReviewDecision:
        """计划评审 + 授权评审：两轮调用评审 Agent 后合并结论。"""
        self._transition(TaskStatus.PLAN_REVIEW)
        task_request = self._task_request
        subtask_snapshot = [spec.model_copy(deep=True) for spec in self._subtasks]

        plan_decision = await self._call_review_agent(
            ReviewInput(
                stage=ReviewStage.PLAN_REVIEW,
                task_request=task_request,
                subtasks=subtask_snapshot,
                plan_version=self._plan_version,
            )
        )
        authorization_decision = await self._call_review_agent(
            ReviewInput(
                stage=ReviewStage.AUTHORIZATION_REVIEW,
                task_request=task_request,
                subtasks=subtask_snapshot,
                tool_grants=list(self._plan_grants),
                plan_version=self._plan_version,
            )
        )
        combined = self._combine_reviews(plan_decision, authorization_decision)
        self._last_review = combined
        return combined

    @staticmethod
    def _combine_reviews(
        plan: ReviewDecision, authorization: ReviewDecision
    ) -> ReviewDecision:
        """合并计划评审与授权评审的结论（两者均通过才算通过）。"""
        return ReviewDecision(
            stage="plan_review",
            passed=plan.passed and authorization.passed,
            issues=list(plan.issues) + list(authorization.issues),
            required_fixes=list(plan.required_fixes) + list(authorization.required_fixes),
            retry_target=plan.retry_target or authorization.retry_target,
            summary=" | ".join(
                text for text in (plan.summary, authorization.summary) if text
            ),
        )

    def _try_consume_replan_quota(self) -> bool:
        """尝试消耗一次任务拆分修改额度。

        Returns:
            True 表示额度充足并已消耗；False 表示已耗尽（应人工兜底）。
        """
        if self._replan_count >= int(self._settings.max_replan_attempts):
            return False
        self._replan_count += 1
        return True

    def _check_cancel(self) -> None:
        """协作式取消检查点：收到取消请求时抛出 WorkflowCancelledError。

        Raises:
            WorkflowCancelledError: 已通过 ``cancel()`` 请求取消时。
        """
        if self._cancel_requested:
            raise WorkflowCancelledError("任务已收到取消请求")

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
    # 状态迁移（一律通过 EventLogger 记录 status_transition 事件）
    # ------------------------------------------------------------------
    def _transition(
        self, new_status: TaskStatus, *, reason: str = "", **extra: Any
    ) -> None:
        """状态迁移：更新当前状态并记录 ``status_transition`` 事件。

        状态实际发生变化时，同步把阶段进度提示（或失败/取消类状态的
        错误原因）打印到标准错误，便于用户实时观察工作流进展。

        Args:
            new_status: 目标状态。
            reason: 迁移原因（中文说明，写入事件 payload）。
            **extra: 附加到事件 payload 的 JSON 兼容字段（如 subtask_id）。
        """
        previous = self._current_status
        self._current_status = new_status
        if new_status == previous:
            return
        _log_progress(new_status, reason)
        payload: dict[str, Any] = {
            "task_id": self._task_request.task_id if self._task_request else None,
            "agent_id": DEFAULT_SUPERVISOR_ID,
            "from_status": previous.value,
            "to_status": new_status.value,
            "plan_version": self._plan_version,
            "replan_count": self._replan_count,
        }
        if reason:
            payload["reason"] = reason
        payload.update(extra)
        self._events.log_event(EVENT_STATUS_TRANSITION, payload)

    # ------------------------------------------------------------------
    # 执行（executing）与失败升级（replacement / decomposition）
    # ------------------------------------------------------------------
    async def _execute_subtask_attempt(
        self, spec: SubtaskSpec, agent_record: AgentRecord
    ) -> ExecutionResult:
        """单次子任务执行（TaskScheduler 的 executor 契约实现）。

        通过主管 Agent 路由执行：构造 ``phase="execute_subtask"`` 的 JSON
        载荷（含任务请求、子任务规格、Agent 记录、尝试序号与该子任务的
        ToolGrant），由主管 Agent 选择对应工具完成执行，并要求其把工具
        返回的 ExecutionResult JSON 填入 ``final_result`` 回传。

        Args:
            spec: 子任务规格（同一计划版本内）。
            agent_record: 本次执行使用的 Agent 记录（原始或替代 Agent）。

        Returns:
            ExecutionResult：解析成功的结果；主管 Agent 调用失败或输出
            无法解析时返回 ``success=False`` 的结果（由调度器按失败尝试
            计数并推进升级流程）。
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
            print(
                f"[so-agent] 子任务 [{spec.subtask_id}] 执行失败：{type(exc).__name__}",
                file=sys.stderr, flush=True,
            )
            return ExecutionResult(
                subtask_id=spec.subtask_id,
                success=False,
                error=f"主管 Agent 调用失败：{type(exc).__name__}: {exc}",
            )
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
        """把主管 Agent 回传的文本解析为 ExecutionResult（宽容解析）。

        依次尝试：整体 JSON 解析 → 文本内嵌 JSON 对象提取；解析成功后把
        ``subtask_id`` 归一化为当前子任务编号。全部失败时返回
        ``success=False`` 的结果（保留原始文本摘要作为错误信息）。

        Args:
            final_result: 主管 Agent 输出的 final_result 文本。
            spec: 对应当前执行批次的子任务规格。
        """
        text = (final_result or "").strip()
        candidate: dict[str, Any] | None = None
        if text:
            try:
                parsed = json.loads(text)
                if isinstance(parsed, dict):
                    candidate = parsed
            except json.JSONDecodeError:
                candidate = extract_json_object(text)
        if candidate is not None:
            candidate["subtask_id"] = spec.subtask_id
            try:
                return ExecutionResult.model_validate(candidate)
            except ValidationError:
                pass
        summary = text[:200] + ("…" if len(text) > 200 else "")
        return ExecutionResult(
            subtask_id=spec.subtask_id,
            success=False,
            error=f"未能从主管 Agent 输出解析出 ExecutionResult JSON；原文：{summary!r}",
        )

    async def _run_execution_cycle(self) -> dict[str, ExecutionResult] | None:
        """执行循环：调度执行 → 失败则触发分解问题重规划 → 直到全部成功。

        每一轮完成一个批次的调度（含调度器内部的尝试/替换流程）；出现
        失败批次时组装任务分解问题、上报主管、消耗一次拆分修改额度并
        重新拆解；额度耗尽时返回 None（由调用方转入人工兜底）。

        Returns:
            当前计划全部成功时的执行结果；额度耗尽时为 None。
        """
        while True:
            batch = await self._execute_subtasks(list(self._pending_specs))
            failed = [
                subtask_id for subtask_id, item in batch.items() if not item.success
            ]
            if not failed:
                return batch
            issue = self._collect_failure_issue(failed)
            if not await self._handle_decomposition_issue(issue):
                return None
            if not await self._plan_and_review(issue=issue):
                return None
            self._pending_specs = list(self._subtasks)

    async def _execute_subtasks(
        self, specs: list[SubtaskSpec]
    ) -> dict[str, ExecutionResult]:
        """通过 TaskScheduler 调度一个子任务批次，并把过程映射到状态机。

        调度前先解决工具缺口（``_resolve_tool_gaps``：创建 → 评审 →
        注册 → 授权）；调度器内部完成"三次尝试 → 一次替换请求 → 替代
        Agent（无再申请资格）→ 分解问题上报"；返回后由
        ``_settle_failure_states`` 依据替换请求与分解问题重建状态链
        （replacement_requested → replacement_executing → executing），
        并把结果写入 ``self._results``。

        Args:
            specs: 待执行批次（当前计划全量或结果评审返工子集）。

        Returns:
            ``{subtask_id: ExecutionResult}``（仅包含本批次子任务）；
            工具缺口未解决的子任务直接标记为失败；调度器拒绝调度
            （结构异常）时转换为全失败批次，交由失败升级流程处理。
        """
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
        merged: dict[str, ExecutionResult] = {**gap_failures, **batch}
        self._settle_failure_states(merged)
        self._results.update(merged)
        return merged

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
        registry.json。

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

        if manifest.review_status != "approved":
            decision = await self._call_review_agent(
                ReviewInput(
                    stage=ReviewStage.TOOL_REVIEW,
                    task_request=self._task_request,
                    generated_tool=manifest,
                    plan_version=self._plan_version,
                )
            )
            if not decision.passed:
                detail = "；".join(decision.issues) or decision.summary or "未说明原因"
                reason = f"工具评审未通过：{detail}"
                self._events.log_event(
                    EVENT_TOOL_CALLED,
                    {**base_event, "action": "gap_unresolved", "reason": reason},
                )
                return reason
            manifest = manifest.model_copy(update={"review_status": "approved"})

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

    async def _invoke_code_agent_for_tool(self, tool_name: str) -> str | None:
        """经 MCP 调用代码 Agent 创建工具候选（write_generated_tool）。

        Returns:
            None 表示代码 Agent 成功完成创建动作；否则返回失败原因。
        """
        client = self._ensure_mcp_client()
        payload = CodeAgentInput(
            task_description=(
                f"创建动态工具 {tool_name}：实现其完整功能并调用 "
                f"write_generated_tool 写入 generated_tools（工具名必须为 "
                f"{tool_name}）；manifest_data 需包含 description、"
                f"input_schema、output_schema 与 created_by_task="
                f"{self._task_request.task_id}。"
            ),
            action=CodeAction.CREATE,
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

    def _settle_failure_states(self, batch: dict[str, ExecutionResult]) -> None:
        """依据调度产物重建失败升级状态链，并记录失败历史。

        - 新出现的替换申请：``replacement_requested → replacement_executing``
          （主管核验失败证据后批准），替代成功时回到 ``executing``；
        - 新出现的分解问题：追加到 ``_agent_failure_history``（供重规划
          参考与人工升级材料组装）。
        """
        subtask_ids = set(batch)
        for request in self._scheduler.replacement_requests:
            if request.request_id in self._seen_replacement_requests:
                continue
            if request.subtask_id not in subtask_ids:
                continue
            requester = self._registry.get(request.requester_agent_id)
            if self._handle_replacement(request.subtask_id, requester) is None:
                continue
            item = batch.get(request.subtask_id)
            if item is not None and item.success:
                self._transition(
                    TaskStatus.EXECUTING,
                    reason="替代 Agent 执行成功，恢复常规执行",
                    subtask_id=request.subtask_id,
                )
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

    def _handle_replacement(
        self, subtask_id: str, agent_record: AgentRecord | None
    ) -> ReplacementRequest | None:
        """处理一次替换申请：重建状态链并把申请书上报主管（留痕）。

        首版替换批准策略：失败证据随申请完整留痕，控制层核验申请材料
        完整性后自动批准（等价于主管批准）；替代 Agent 由调度器以
        ``can_request_replacement=False`` 执行，故同一子任务最多出现
        一次该状态链。

        Args:
            subtask_id: 发起替换的子任务编号。
            agent_record: 发起申请的原始 Agent 记录（可为 None，此时以
                申请书内的 requester_agent_id 作为留痕主体）。

        Returns:
            本次处理的 ReplacementRequest；无待处理申请时为 None。
        """
        request: ReplacementRequest | None = None
        for item in reversed(self._scheduler.replacement_requests):
            if (
                item.subtask_id == subtask_id
                and item.request_id not in self._seen_replacement_requests
            ):
                request = item
                break
        if request is None:
            return None
        self._seen_replacement_requests.add(request.request_id)
        self._transition(
            TaskStatus.REPLACEMENT_REQUESTED,
            reason="子 Agent 三次尝试失败，发起替换申请",
            subtask_id=subtask_id,
            request_id=request.request_id,
        )
        requester_id = (
            agent_record.agent_id
            if agent_record is not None
            else request.requester_agent_id
        )
        self._route_message(
            requester_id,
            {
                "message_type": "replacement_request",
                "task_id": self._task_request.task_id,
                "subtask_id": subtask_id,
                "request": request.model_dump(mode="json"),
                "verdict": "approved",
                "decided_by": DEFAULT_SUPERVISOR_ID,
            },
        )
        self._agent_failure_history.append(
            f"替换申请 {request.request_id}（子任务 {subtask_id}）："
            f"{request.failure_summary}"
        )
        self._transition(
            TaskStatus.REPLACEMENT_EXECUTING,
            reason="主管核验失败证据后批准替换，替代 Agent 开始执行",
            subtask_id=subtask_id,
            request_id=request.request_id,
        )
        return request

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

    # ------------------------------------------------------------------
    # 汇总（aggregating）与最终评审（result_review）
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

    def _resolve_rework_targets(self, decision: ReviewDecision) -> set[str]:
        """解析结果评审指定的返工目标（subtask_id 集合）。

        匹配顺序：``retry_target`` 精确匹配 → ``retry_target`` 包含匹配 →
        issues / required_fixes 文本包含匹配 → 失败结果兜底；全部未命中
        时返回空集合（由调用方转入人工兜底）。
        """
        plan_ids = [spec.subtask_id for spec in self._subtasks]
        plan_id_set = set(plan_ids)
        targets: set[str] = set()
        retry = (decision.retry_target or "").strip()
        if retry in plan_id_set:
            targets.add(retry)
        elif retry:
            targets.update(sid for sid in plan_ids if sid and sid in retry)
        if not targets:
            haystack = "\n".join(
                list(decision.issues) + list(decision.required_fixes)
            )
            targets.update(sid for sid in plan_ids if sid and sid in haystack)
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
        """
        specs: list[SubtaskSpec] = []
        for spec in self._subtasks:
            if spec.subtask_id not in rework_ids:
                continue
            clone = spec.model_copy(deep=True)
            clone.dependencies = [
                dep for dep in clone.dependencies if dep in rework_ids
            ]
            specs.append(clone)
        return specs

    # ------------------------------------------------------------------
    # 人工兜底（human_required）
    # ------------------------------------------------------------------
    async def _escalate_to_human(self, reason: str) -> OrchestratorOutput:
        """人工兜底：组装升级材料包、交接处理器并把任务置为 human_required。

        首版 HumanEscalationHandler 仅记录请求，不实际通知人工；材料包
        含原始任务、历代计划、失败历史、工具调用证据与最终评审结论。

        Args:
            reason: 触发人工介入的原因说明。

        Returns:
            OrchestratorOutput：human_required 终态的结构化输出。
        """
        self._transition(TaskStatus.HUMAN_REQUIRED, reason=reason)
        request = HumanEscalationRequest(
            task_id=self._task_request.task_id,
            original_request=self._task_request.model_copy(deep=True),
            plan_history=self.plan_history,
            agent_failure_history=list(self._agent_failure_history),
            tool_call_evidence=self._collect_tool_evidence(),
            final_review=(
                self._last_review.model_copy(deep=True)
                if self._last_review is not None
                else None
            ),
        )
        self._human_escalation = request
        self._events.log_event(
            EVENT_HUMAN_ESCALATION,
            {
                "task_id": self._task_request.task_id,
                "agent_id": DEFAULT_SUPERVISOR_ID,
                "reason": reason,
                "request": request.model_dump(mode="json"),
            },
        )
        await self._escalation_handler.escalate(request)
        return self._build_output(
            TaskStatus.HUMAN_REQUIRED,
            final_result=f"任务需要人工介入：{reason}",
            review=self._last_review,
        )

    def _collect_tool_evidence(self) -> list[str]:
        """从事件日志提取工具调用 / Agent 生命周期证据（JSON 文本行）。

        仅保留本任务相关事件（无 task_id 的事件一并保留），供人工升级
        材料组装与审计复核。
        """
        task_id = self._task_request.task_id if self._task_request else None
        evidence: list[str] = []
        for record in list(self._context.event_log):
            if not isinstance(record, dict):
                continue
            if record.get("event_type") not in _EVIDENCE_EVENT_TYPES:
                continue
            record_task = record.get("task_id")
            if task_id is not None and record_task not in (None, task_id):
                continue
            evidence.append(json.dumps(record, ensure_ascii=False, default=str))
        return evidence

    # ------------------------------------------------------------------
    # 输出组装（严格 JSON 通信的最终载体）
    # ------------------------------------------------------------------
    def _build_output(
        self,
        status: TaskStatus,
        *,
        final_result: str = "",
        review: ReviewDecision | None = None,
    ) -> OrchestratorOutput:
        """构造统一的结构化输出（OrchestratorOutput）。

        Args:
            status: 终态（或过程中的最终状态）。
            final_result: 最终结果文本。
            review: 评审结论覆盖；未提供时使用最近一次评审结论。
        """
        effective_review = review if review is not None else self._last_review
        return OrchestratorOutput(
            task_id=self._task_request.task_id if self._task_request else "",
            status=status,
            final_result=final_result,
            plan_versions=self.plan_history,
            agent_history=self._agent_history(),
            review_summary=self._format_review(effective_review),
            human_escalation=(
                self._human_escalation.model_copy(deep=True)
                if self._human_escalation is not None
                else None
            ),
        )

    def _agent_history(self) -> list[AgentRecord]:
        """返回本任务的 Agent 生命周期记录（含已释放，供审计与替换溯源）。"""
        if self._task_request is None:
            return []
        return [
            record.model_copy(deep=True)
            for record in self._registry.list_by_task(self._task_request.task_id)
        ]

    @staticmethod
    def _format_review(decision: ReviewDecision | None) -> str:
        """把评审结论格式化为单行摘要文本。"""
        if decision is None:
            return ""
        verdict = "通过" if decision.passed else "未通过"
        parts = [f"[{decision.stage}] {verdict}"]
        if decision.summary:
            parts.append(decision.summary)
        if decision.issues:
            parts.append("问题：" + "；".join(decision.issues))
        if decision.required_fixes:
            parts.append("修复项：" + "；".join(decision.required_fixes))
        if decision.retry_target:
            parts.append(f"重试目标：{decision.retry_target}")
        return " | ".join(parts)

    # ------------------------------------------------------------------
    # 注册表与通信辅助
    # ------------------------------------------------------------------
    def _find_latest_agent(
        self, subtask_id: str, *, replacement: bool | None = None
    ) -> AgentRecord | None:
        """在注册表中查找某子任务最近的 Agent 记录。

        Args:
            subtask_id: 子任务编号。
            replacement: True 仅查替代 Agent；False 仅查原始 Agent；
                None 不限（按创建时间取最新）。
        """
        if self._task_request is None:
            return None
        latest: AgentRecord | None = None
        for record in self._registry.list_by_task(self._task_request.task_id):
            if self._subtask_id_of(record) != subtask_id:
                continue
            is_replacement = record.replacement_of is not None
            if replacement is not None and is_replacement != replacement:
                continue
            latest = record
        return latest

    def _subtask_id_of(self, record: AgentRecord) -> str:
        """从 agent_id 解析所属子任务编号。

        调度器生成的格式为 ``{task_id}:{subtask_id}:{kind}-{hex}``；
        无法解析时返回空字符串。
        """
        raw = record.agent_id or ""
        if self._task_request is None:
            return ""
        prefix = f"{self._task_request.task_id}:"
        if not raw.startswith(prefix):
            return ""
        return raw[len(prefix):].split(":", 1)[0]

    def _route_message(self, sender_id: str, message: dict[str, Any]) -> bool:
        """经 SupervisorRouter 向主管上报一条 JSON 消息（失败不阻断流程）。

        Returns:
            True 表示消息已通过通信边界校验并留痕；False 表示被拒绝
            （拒绝原因已由路由器写入 message_rejected 事件）。
        """
        try:
            self._router.route_message(
                sender_id, self._router.supervisor_id, message
            )
        except CommunicationError:
            return False
        return True

    # ------------------------------------------------------------------
    # 主管 / 评审 Agent 调用（严格 JSON 输入输出）
    # ------------------------------------------------------------------
    async def _run_supervisor(
        self, payload: dict[str, Any], output_model: type[_StructuredOutputT]
    ) -> _StructuredOutputT:
        """调用主管 Agent：JSON 输入 → 按阶段契约手动解析的结构化输出。

        主管 Agent 以纯文本返回（output_type=None，兼容忽略
        response_format 的千问等模型服务）：此处剥离 Markdown 代码围栏
        并按 ``output_model`` 阶段性精简契约（PlanningOutput /
        ExecutionOutput / AggregationOutput）手动校验，每个阶段只
        要求当前需要的字段，大幅降低输出 token 与耗时；对模型实例 /
        字典 / JSON 文本形态做宽容解析。

        Args:
            payload: 含 ``phase`` 阶段标记的 JSON 输入载荷。
            output_model: 该阶段的输出契约模型。

        Raises:
            WorkflowError: 输出无法解析为 output_model 时。
        """
        message = json.dumps(payload, ensure_ascii=False, default=str)
        result = await Runner.run(
            self._orchestrator,
            input=message,
            max_turns=int(self._settings.max_model_turns),
        )
        return _coerce_structured_output(
            result.final_output, output_model, label="主管 Agent"
        )

    async def _call_review_agent(self, review_input: ReviewInput) -> ReviewDecision:
        """调用评审 Agent：JSON 输入 → ReviewDecision 结构化输出。

        异常或输出不可解析时返回保守的"未通过"结论（宁可拦截不可放行，
        防止错误结果被放行）；``stage`` 统一归一化为 plan_review /
        result_review（授权评审在结论中映射为 plan_review）。
        """
        expected_stage: str = (
            "result_review"
            if review_input.stage == ReviewStage.RESULT_REVIEW
            else "plan_review"
        )
        try:
            message = review_input.model_dump_json()
            result = await Runner.run(
                self._review_agent,
                input=message,
                max_turns=int(self._settings.max_model_turns),
            )
            decision = self._coerce_review_decision(result.final_output)
        except Exception as exc:
            return ReviewDecision(
                stage=expected_stage,  # type: ignore[arg-type]
                passed=False,
                issues=[f"评审 Agent 调用/解析失败：{type(exc).__name__}: {exc}"],
                required_fixes=["修复评审链路后重新提交评审"],
                summary="评审不可用，保守判定为未通过",
            )
        decision.stage = expected_stage  # type: ignore[assignment]
        return decision

    @staticmethod
    def _coerce_review_decision(output: Any) -> ReviewDecision:
        """把评审 Agent 输出规范化为 ReviewDecision（宽容解析，含围栏剥离）。

        评审 Agent 同样以纯文本返回（output_type=None），复用
        ``_coerce_structured_output`` 的四形态宽容解析。

        Raises:
            WorkflowError: 输出无法解析为 ReviewDecision 时。
        """
        return _coerce_structured_output(output, ReviewDecision, label="评审 Agent")


# ---------------------------------------------------------------------------
# 模块级工具：结构化输出宽容解析（纯文本返回 + 围栏剥离 + Pydantic 校验）
# ---------------------------------------------------------------------------
_StructuredOutputT = TypeVar("_StructuredOutputT", bound=BaseModel)


def _coerce_structured_output(
    output: Any, model_cls: type[_StructuredOutputT], *, label: str
) -> _StructuredOutputT:
    """把 Agent 原始输出宽容解析为目标契约模型（Pydantic）。

    所有 Agent 以纯文本返回（output_type=None，兼容忽略
    response_format=json_schema 的千问等模型服务）；依次尝试：

    1. 目标模型实例（测试桩 / 已校验结果直接透传）；
    2. 其他 Pydantic 模型实例：序列化后按目标契约重校验（字段兼容即可
       通过，如旧全量形态经宽容归一化映射到精简契约）；
    3. 字典：直接按目标契约校验；
    4. JSON 文本：剥离 Markdown 围栏后整体/贪心提取再校验。

    Args:
        output: Runner 返回的 final_output（任意形态）。
        model_cls: 目标契约模型（PlanningOutput / ExecutionOutput /
            AggregationOutput / ReviewDecision 等）。
        label: 错误消息中的主体名称（如“主管 Agent”）。

    Raises:
        WorkflowError: 全部形态均无法解析为目标契约时。
    """
    if isinstance(output, model_cls):
        return output
    if isinstance(output, BaseModel):
        try:
            return model_cls.model_validate(output.model_dump(mode="json"))
        except ValidationError as exc:
            raise WorkflowError(
                f"{label} 输出（{type(output).__name__}）不符合 "
                f"{model_cls.__name__} 契约：{exc}"
            ) from exc
    if isinstance(output, dict):
        try:
            return model_cls.model_validate(output)
        except ValidationError as exc:
            raise WorkflowError(
                f"{label} 输出字典不符合 {model_cls.__name__} 契约：{exc}"
            ) from exc
    if isinstance(output, str):
        obj = extract_json_object(output)
        if obj is not None:
            try:
                return model_cls.model_validate(obj)
            except ValidationError as exc:
                raise WorkflowError(
                    f"{label} 输出 JSON 不符合 {model_cls.__name__} 契约：{exc}"
                ) from exc
    raise WorkflowError(
        f"{label} 输出无法解析为 {model_cls.__name__}（{type(output).__name__}）"
    )

"""工作流薄编排器（WorkflowEngine）。

本模块是整个框架的控制层入口，自 v0.2 起由"God Object"拆分为
**薄编排器 + 三个协调器 + 共享状态底座**的协调器架构：

- :mod:`so_agent.state_machine`：声明式状态机（TaskStateMachine，
  13 阶段影子视图）与共享可变状态（WorkflowState）及工作流领域
  异常、结构化输出宽容解析（本模块对其 re-export）；
- :mod:`so_agent.planning_coordinator`：任务接收、拆解、两阶段计划
  评审与工具授权（PlanningCoordinator）；
- :mod:`so_agent.execution_coordinator`：子任务调度执行、失败升级
  （替换/分解问题）与工具缺口流程（ExecutionCoordinator）；
- :mod:`so_agent.aggregation_coordinator`：结果汇总、结果评审与
  返工（AggregationCoordinator）；
- :mod:`so_agent.workflow_support`：协调器依赖容器（WorkflowDeps）、
  协调器基类（BaseCoordinator）与人工兜底处理器
  （HumanEscalationHandler，本模块对其 re-export）；
- 编排器自身的职责方法按领域拆为混入模块：provisional 交付与结果
  评审策略（:mod:`so_agent.workflow_provisional`）、检查点导出/恢复
  （:mod:`so_agent.workflow_checkpoint`）、人工兜底与指导恢复
  （:mod:`so_agent.workflow_escalation`）、主管/评审 Agent 调用与
  任务级预算（:mod:`so_agent.workflow_supervisor`），本模块保留主
  循环、装配与输出组装。

职责分工与不变式：

- 编排器只负责主循环（``run`` / ``_run_pipeline``）、人工兜底与输出
  组装，按状态把动作分派给协调器；协调器之间不直接互相引用，一律
  通过共享 WorkflowState 交换状态、经 WorkflowDeps 通道回调
  （主管/评审调用绑定在本模块方法上，保持 ``workflow_mod.Runner``
  的模块级 monkeypatch 语义）；
- 状态流转仍严格遵循蓝图：``received → planning → plan_review →
  executing → aggregating → result_review → completed``；执行失败
  分支：``executing → replacement_requested → replacement_executing →
  decomposition_issue → replanning → plan_review``（循环）；终态：
  ``completed / human_required / failed / cancelled``；
- 两阶段评审：计划评审（含授权最小化评审）与结果评审，均由评审
  Agent 给出 ``ReviewDecision`` 结构化结论；
- 失败升级：调度器按确定性失败恢复策略（runtime/failure.py）
  驱动重试——失败分类 + 稳定指纹去重，同因连续失败达阈值即抑制
  重试、provider 类故障禁止替换；单 Agent 尝试次数以
  ``max_agent_attempts`` 为上限、真实次数由策略决定；一次替代
  申请、替代 Agent 无再申请资格、替代仍失败触发任务拆分复查、
  主管最多三次拆分修改，耗尽后进入人工兜底（首版仅接口与状态）；
- 全程 JSON 通信：对主管 Agent / 评审 Agent 的一切调用输入输出均为
  Pydantic 模型序列化；状态迁移一律通过 EventLogger 记录
  （``status_transition`` 事件），供审计与人工升级材料组装。

向后兼容（Facade 兼容代理）：

- 历史私有状态属性（``_task_request`` / ``_subtasks`` / ``_results``
  等）经由 :class:`_WorkflowStateField` 描述符读写代理到共享
  WorkflowState——外部直接赋值（如 ``engine._results = {...}``）的
  语义与拆分前完全一致；
- 历史私有方法（``_plan_and_review`` / ``_verify_generated_tool`` /
  ``_resolve_rework_targets`` 等）经 ``__getattr__`` 按声明的路由表
  代理到对应协调器的同名方法。

调用示例::

    context = ProjectContext(sandbox_dir=Path("sandbox"), project_name="demo")
    engine = WorkflowEngine(context)
    output = await engine.run("为指定目录实现一个 CLI 工具")
    print(output.model_dump_json(indent=2))

说明：调度器（SubtaskDAGScheduler）内部完成"DAG 调度 + 策略驱动重试
（失败分类 / 指纹去重 / 幂等退避 / 替换限制）+ 一次替换请求 +
替代执行 + 分解问题上报"；替代状态链
（replacement_requested → replacement_executing → …）由调度器在
真实生命周期时点经受控回调（execution_coordinator.
handle_replacement_lifecycle，asyncio.Lock 串行化，任务 34）实时
映射到状态机——不再事后重建；编排器在装配时注入该回调，并负责
拆分修改额度的消耗与人工兜底。首版中"主管检查失败证据后同意替换"
由控制层自动批准（证据随 ReplacementRequest 全程留痕），不额外
消耗模型调用。
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from agents import Agent, Model, Runner

import so_agent.tool_packages.code_agent.agent as code_agent_agent_mod
from so_agent.aggregation_coordinator import AggregationCoordinator
from so_agent.config import Settings
from so_agent.context import ProjectContext
from so_agent.execution_coordinator import ExecutionCoordinator
from so_agent.mcp.client import MCPClient
from so_agent.mcp.directory import MCPDirectoryManager
from so_agent.runtime.logging_setup import get_logger
from so_agent.runtime.tool_lifecycle import sweep_generated_tools
from so_agent.runtime.tool_store import resolve_archive_dir, resolve_mcp_servers_dir
from so_agent.runtime.tool_store_migration import migrate_legacy_generated_tools
from so_agent.models import (
    AgentRecord,
    ExecutionResult,
    HumanEscalationRequest,
    ReviewDecision,
    TaskStatus,
)
from so_agent.orchestrator import OrchestratorOutput, create_orchestrator
from so_agent.planning_coordinator import PlanningCoordinator
from so_agent.result_store import InMemoryResultStore, ResultStore
from so_agent.runtime.events import (
    EVENT_PROVISIONAL_DELIVERED,
    EVENT_REVIEW_SKIPPED,
    EventLogger,
)
from so_agent.runtime.budget import TokenBudgetExceeded, TokenBudgetTracker
from so_agent.runtime.audit import build_audit_report
from so_agent.runtime.notifications import build_escalation_notifier
from so_agent.runtime.permissions import PermissionGateway
from so_agent.runtime.registry import DEFAULT_SUPERVISOR_ID, AgentRegistry
from so_agent.runtime.scheduler import SubtaskDAGScheduler
from so_agent.runtime.supervisor_router import SupervisorRouter
from so_agent.runtime.tool_registry import ToolRegistry
from so_agent.runtime.telemetry import async_span_scope, set_span_attributes
from so_agent.state_machine import (
    HumanGuidanceError,
    PipelineDeadlineExceeded,
    PipelineStoppedError,
    ReviewContractError,
    SupervisorNetworkError,
    TaskStateMachine,
    WorkflowCancelledError,
    WorkflowError,
    WorkflowState,
    _coerce_structured_output,
)
from so_agent.tool_packages.review_agent.agent import create_review_agent
from so_agent.workflow_support import (
    HumanEscalationHandler,
    PipelineGate,
    WorkflowDeps,
)

from so_agent.workflow_checkpoint import CheckpointMixin
from so_agent.workflow_escalation import EscalationMixin
from so_agent.workflow_provisional import ProvisionalMixin
from so_agent.workflow_supervisor import SupervisorMixin

logger = get_logger("workflow")

__all__ = [
    "AggregationCoordinator",
    "ExecutionCoordinator",
    "HumanEscalationHandler",
    "HumanGuidanceError",
    "PlanningCoordinator",
    "ReviewContractError",
    "PipelineGate",
    "PipelineDeadlineExceeded",
    "PipelineStoppedError",
    "SupervisorNetworkError",
    "TaskStateMachine",
    "WorkflowCancelledError",
    "WorkflowDeps",
    "WorkflowEngine",
    "WorkflowError",
    "WorkflowState",
    "_coerce_structured_output",
]

# 主管 Agent 的三个固定工具入口：作为路由的合法接收方；子任务的
# allowed_tools 允许出现未注册的工具名（作为"工具缺口"），由执行
# 阶段的缺口流程（创建 → 评审 → 注册 → 授权）解决
_TOOL_UNIVERSE: tuple[str, ...] = ("code_agent", "subagent_creator", "review_agent")

# Facade 兼容代理路由表：历史私有方法名 → 引擎上的协调器属性。
# __getattr__ 据此把外部对原 WorkflowEngine 私有方法的访问代理到
# 对应协调器的同名方法（协调器之间不互相引用，此表仅编排器可见）。
_FACADE_METHOD_ROUTES: dict[str, str] = {
    # ---- PlanningCoordinator（规划 / 评审 / 授权） ----
    "_create_task": "_planning_coordinator",
    "_plan_and_review": "_planning_coordinator",
    "_print_plan_summary": "_planning_coordinator",
    "_request_plan": "_planning_coordinator",
    "_build_planning_payload": "_planning_coordinator",
    "_extract_plan": "_planning_coordinator",
    "_adopt_plan": "_planning_coordinator",
    "_issue_grants": "_planning_coordinator",
    "_propose_grants": "_planning_coordinator",
    "_revoke_active_grants": "_planning_coordinator",
    "_review_plan": "_planning_coordinator",
    "_validate_plan_structure": "_planning_coordinator",
    "_authorization_review_is_deterministic": "_planning_coordinator",
    "_deterministic_authorization_review": "_planning_coordinator",
    "_combine_reviews": "_planning_coordinator",
    "_try_consume_replan_quota": "_planning_coordinator",
    "_is_simple_plan": "_planning_coordinator",
    "_decide_parallelism": "_planning_coordinator",
    "_split_items": "_planning_coordinator",
    "_archive_completed_results": "_planning_coordinator",
    "_reuse_candidates_payload": "_planning_coordinator",
    "_validate_reuse_declarations": "_planning_coordinator",
    "_downstream_closure": "_planning_coordinator",
    "_record_plan_version": "_planning_coordinator",
    "_completed_results_summary": "_planning_coordinator",
    # ---- ExecutionCoordinator（执行 / 监控 / 替换 / 工具缺口） ----
    "_execute_subtask_attempt": "_execution_coordinator",
    "_parse_execution_result": "_execution_coordinator",
    "_run_execution_cycle": "_execution_coordinator",
    "_execute_subtasks": "_execution_coordinator",
    "_resolve_tool_gaps": "_execution_coordinator",
    "_is_tool_ready": "_execution_coordinator",
    "_sync_generated_registry": "_execution_coordinator",
    "_ensure_mcp_client": "_execution_coordinator",
    "_create_and_approve_tool": "_execution_coordinator",
    "_verify_generated_tool": "_execution_coordinator",
    "_invoke_code_agent_for_tool": "_execution_coordinator",
    "_settle_failure_states": "_execution_coordinator",
    "_handle_decomposition_issue": "_execution_coordinator",
    "_collect_failure_issue": "_execution_coordinator",
    "_format_issue": "_execution_coordinator",
    # ---- AggregationCoordinator（汇总 / 结果评审 / 返工） ----
    "_aggregate_results": "_aggregation_coordinator",
    "_successful_results": "_aggregation_coordinator",
    "_fallback_summary": "_aggregation_coordinator",
    "_final_review": "_aggregation_coordinator",
    "_resolve_rework_targets": "_aggregation_coordinator",
    "_build_rework_specs": "_aggregation_coordinator",
}


class _WorkflowStateField:
    """引擎历史私有状态属性的读写代理（Facade 兼容层）。

    读写直接转发到共享 :class:`WorkflowState` 的同名字段（去下划线），
    使外部直接赋值（如 ``engine._results = {...}``，既有测试在用）的
    语义与拆分前完全一致。
    """

    def __init__(self, state_attr: str) -> None:
        self._state_attr = state_attr

    def __get__(self, instance: Any, owner: type | None = None) -> Any:
        if instance is None:
            return self
        return getattr(instance._state, self._state_attr)

    def __set__(self, instance: Any, value: Any) -> None:
        setattr(instance._state, self._state_attr, value)


class WorkflowEngine(
    ProvisionalMixin,
    CheckpointMixin,
    EscalationMixin,
    SupervisorMixin,
):
    """任务生命周期编排器（控制层薄入口）。

    职责：
    - 驱动完整流程：拆解（主管 Agent）→ 计划评审（评审 Agent）→ 调度
      执行（SubtaskDAGScheduler）→ 失败升级（替换/重规划）→ 汇总（主管
      Agent）→ 结果评审（评审 Agent）→ 终态；具体动作由规划 / 执行 /
      汇总三个协调器完成；
    - 管理治理额度：三次任务拆分修改（含评审重试与分解问题重规划）、
      结果评审返工次数上限（额度判定在规划协调器与主循环中）；
    - 记录全过程事件并保留全部历史（计划版本、失败历史、工具证据），
      并在收尾组装统一的结构化输出（OrchestratorOutput）。
    """

    # ------------------------------------------------------------------
    # Facade 兼容层：历史私有状态属性 → 共享 WorkflowState（可读可写）
    # ------------------------------------------------------------------
    _current_status = _WorkflowStateField("current_status")
    _plan_version = _WorkflowStateField("plan_version")
    _replan_count = _WorkflowStateField("replan_count")
    _result_rework_count = _WorkflowStateField("result_rework_count")
    _task_request = _WorkflowStateField("task_request")
    _subtasks = _WorkflowStateField("subtasks")
    _pending_specs = _WorkflowStateField("pending_specs")
    _results = _WorkflowStateField("results")
    _plan_history = _WorkflowStateField("plan_history")
    _agent_failure_history = _WorkflowStateField("agent_failure_history")
    _grants = _WorkflowStateField("grants")
    _plan_grants = _WorkflowStateField("plan_grants")
    _plan_grant_map = _WorkflowStateField("plan_grant_map")
    _plan_grant_proposals = _WorkflowStateField("plan_grant_proposals")
    _last_review = _WorkflowStateField("last_review")
    _human_escalation = _WorkflowStateField("human_escalation")
    _cancel_requested = _WorkflowStateField("cancel_requested")
    _seen_replacement_requests = _WorkflowStateField("seen_replacement_requests")
    _seen_issue_fingerprints = _WorkflowStateField("seen_issue_fingerprints")
    _archive_results = _WorkflowStateField("archive_results")
    _parallelism_decision = _WorkflowStateField("parallelism_decision")
    _runtime_breaker = _WorkflowStateField("runtime_breaker")

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
        log_file: str | Path | None = None,
        router: SupervisorRouter | None = None,
        scheduler: SubtaskDAGScheduler | None = None,
        escalation_handler: HumanEscalationHandler | None = None,
        supervisor_model: str | Model | None = None,
        tool_registry: ToolRegistry | None = None,
        mcp_client: MCPClient | None = None,
        result_store: ResultStore | None = None,
    ) -> None:
        """初始化工作流编排器（装配共享状态、依赖与三个协调器）。

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
            scheduler: 任务调度器；未提供时按执行协调器的单次执行方法新建。
            escalation_handler: 人工兜底处理器；未提供时使用默认占位实现。
            supervisor_model: 主管 Agent 的模型覆盖（透传 create_orchestrator）。
            tool_registry: 工具注册表（MCP 注册中心）；未提供时新建默认实例。
            mcp_client: MCP 客户端；未提供时按需从注册表已挂载的 MCP
                Server 构建（工具缺口流程经它统一调用代码 Agent）。
            result_store: 执行结果存储（跨版本结果复用与计划版本快照）；
                未提供时使用 InMemoryResultStore。
        """
        self._context = context
        self._settings = settings or context.config
        self._events = (
            events
            if events is not None
            else EventLogger(context=context, log_file=log_file)
        )
        # 共享事件记录器：工具包内部运行嵌套 Agent 时经它上报 token 用量，
        # 保证主管/评审之外的调用同样进入同一事件流
        context.event_logger = self._events
        self._registry = (
            registry
            if registry is not None
            else AgentRegistry(context=context, settings=self._settings)
        )
        self._tool_registry = (
            tool_registry
            if tool_registry is not None
            else ToolRegistry(
                generated_tools_dir=code_agent_agent_mod.resolve_generated_tools_dir(
                    context, self._settings
                )
            )
        )
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
            else HumanEscalationHandler(
                notifier=build_escalation_notifier(self._settings)
            )
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
                mcp_client=mcp_client,
                gateway=self._gateway,
                events=self._events,
            )
        )
        self._review_agent = (
            review_agent
            if review_agent is not None
            else create_review_agent(
                context,
                settings=self._settings,
                with_tools=not bool(
                    getattr(self._settings, "review_precomputed_findings", True)
                ),
            )
        )
        self._result_store = (
            result_store if result_store is not None else InMemoryResultStore()
        )
        self._mcp_directory = MCPDirectoryManager(
            resolve_mcp_servers_dir(self._settings)
        )

        # ---- 共享状态 + 协调器装配（协调器之间经 state/deps 解耦） ----
        # pipeline 停止门（任务 45）：超时 / 取消 / 人工终止后原子进入
        # stopping，全部 dispatch / retry / replacement / replan / 计划
        # 采纳 / 授权签发入口先检查，停止后拒绝一切新动作
        self._gate = PipelineGate()
        # 嵌套 Agent（code_agent / review_agent / 动态子 Agent）的流事件
        # 经 context 反向续时 pipeline 看门狗；sink 在调用时读取，因此
        # 即使 orchestrator 先于 gate 构造也能生效。
        self._context.activity_sink = self._gate.record_progress
        self._state = WorkflowState(events=self._events)
        # 仅由 restore_checkpoint 置位：允许在非人工兜底状态下续跑
        # 未完成子任务（如外部取消留下的 executing 检查点）。
        self._restored_from_checkpoint = False
        self._deps = WorkflowDeps(
            context=context,
            settings=self._settings,
            events=self._events,
            registry=self._registry,
            gateway=self._gateway,
            router=self._router,
            tool_registry=self._tool_registry,
            escalation_handler=self._escalation_handler,
            orchestrator=self._orchestrator,
            review_agent=self._review_agent,
            mcp_client=mcp_client,
            result_store=self._result_store,
            gate=self._gate,
        )
        self._planning_coordinator = PlanningCoordinator(self._state, self._deps)
        self._execution_coordinator = ExecutionCoordinator(self._state, self._deps)
        self._aggregation_coordinator = AggregationCoordinator(self._state, self._deps)
        self._scheduler = (
            scheduler
            if scheduler is not None
            else SubtaskDAGScheduler(
                self._execution_coordinator._execute_subtask_attempt,
                context=context,
                settings=self._settings,
                registry=self._registry,
                events=self._events,
                gate=self._gate,
            )
        )
        self._deps.scheduler = self._scheduler
        # 替代生命周期通知通道（任务 34）：调度器在真实生命周期时点
        # 回调执行协调器，由其在锁内串行化全局 FSM 迁移与事件写入；
        # 鸭子类型注入以兼容测试用的简化调度器（无该方法时直接跳过）
        attach_notifier = getattr(
            self._scheduler, "set_lifecycle_notifier", None
        )
        if callable(attach_notifier):
            attach_notifier(
                self._execution_coordinator.handle_replacement_lifecycle
            )
        # 停止门鸭子附加（任务 45）：外部注入的调度器（含测试替身）
        # 具备 set_gate 时同样接入门控
        attach_gate = getattr(self._scheduler, "set_gate", None)
        if callable(attach_gate):
            attach_gate(self._gate)
        # 子任务级进展通知（任务 70）：每个子任务定型即重置无进展
        # 看门狗并落检查点，超时后无需从头重跑
        attach_progress = getattr(self._scheduler, "set_progress_notifier", None)
        if callable(attach_progress):
            attach_progress(self._handle_subtask_progress)
        # token 预算检查（任务级）：调度器每轮派发前结算累计用量，
        # 超限时拒绝新子任务并由 run() 收敛为 failed 终态
        attach_budget = getattr(self._scheduler, "set_budget_check", None)
        if callable(attach_budget):
            attach_budget(self._enforce_token_budget)
        # 子任务硬预算（token）：每次尝试前按精确归因结算，超限即停止
        # 该子任务的重试并由失败升级链路转入重规划
        attach_subtask_budget = getattr(
            self._scheduler, "set_subtask_budget_check", None
        )
        if callable(attach_subtask_budget):
            attach_subtask_budget(
                self._execution_coordinator._subtask_budget_violation
            )
        # 跨协调器调用通道：绑定规划协调器方法（执行协调器经 deps 回调）
        self._deps.plan_and_review = self._planning_coordinator._plan_and_review
        self._deps.issue_grants = self._planning_coordinator._issue_grants
        self._deps.consume_replan_quota = (
            self._planning_coordinator._try_consume_replan_quota
        )
        # 主管/评审调用通道：绑定编排器方法（保持 workflow_mod.Runner 打补丁语义）
        self._deps.run_supervisor = self._run_supervisor
        self._deps.call_review_agent = self._call_review_agent
        self._deps.record_model_usage = self._record_model_usage
        self._deps.enforce_token_budget = self._enforce_token_budget
        # 协调器进展上报时可落检查点（超时/取消后从最近完成点续跑）
        self._deps.persist_checkpoint = self._persist_checkpoint
        # 生命周期清理原因提示：特定退出路径（如 pipeline 超时）在结构化
        # 终态（human_required）之外提供更细的清理原因；run() 的最外层
        # finally 据此填充 tool_revoked 审计事件的 reason
        self._lifecycle_exit_reason: str = ""
        # Token 预算追踪器：任务创建后激活并挂到 context，默认只累加不拦截
        self._token_budget: TokenBudgetTracker | None = None
        # 异步 provisional 评审：先交付后评审的后台任务与最终输出
        self._provisional_task: asyncio.Task[None] | None = None
        self._provisional_output: OrchestratorOutput | None = None
        self._provisional_delivered_text: str = ""
        # pipeline 计时基准（任务 33 嵌套修复）：_run_pipeline 启动时记录
        # monotonic 起点；_run_supervisor 据此把主管绝对总预算截断到
        # pipeline 剩余预算（未启动流水线时为 None，使用完整预算）。
        self._pipeline_started_at: float | None = None

    # ------------------------------------------------------------------
    # Facade 兼容代理（历史私有方法 → 协调器同名方法）
    # ------------------------------------------------------------------
    def __getattr__(self, name: str) -> Any:
        """按路由表把历史私有方法访问代理到对应协调器（兼容层）。

        仅在常规属性查找失败后调用；未登记的名称按标准语义抛出
        AttributeError。
        """
        try:
            coordinator_attr = _FACADE_METHOD_ROUTES[name]
        except KeyError:
            raise AttributeError(
                f"{type(self).__name__} 没有属性 {name!r}"
            ) from None
        return getattr(getattr(self, coordinator_attr), name)

    @property
    def _mcp_client(self) -> MCPClient | None:
        """MCP 客户端（可由执行协调器惰性填充，实际存储于依赖容器）。"""
        return self._deps.mcp_client

    @_mcp_client.setter
    def _mcp_client(self, value: MCPClient | None) -> None:
        self._deps.mcp_client = value

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
    def task_request(self) -> Any:
        """返回当前任务请求；未接收任务时为 None。"""
        return self._task_request

    @property
    def subtasks(self) -> list:
        """返回当前计划版本的子任务列表（深拷贝）。"""
        return [spec.model_copy(deep=True) for spec in self._subtasks]

    @property
    def plan_history(self) -> list:
        """返回历代计划快照（深拷贝）。"""
        return [
            [spec.model_copy(deep=True) for spec in plan] for plan in self._plan_history
        ]

    @property
    def results(self) -> dict[str, ExecutionResult]:
        """返回当前计划版本的子任务结果（深拷贝，只读审计用）。"""
        return {
            subtask_id: result.model_copy(deep=True)
            for subtask_id, result in self._results.items()
        }

    @property
    def human_escalation(self) -> HumanEscalationRequest | None:
        """返回当前人工兜底材料包（深拷贝）；未升级时为 None。"""
        if self._human_escalation is None:
            return None
        return self._human_escalation.model_copy(deep=True)

    def token_budget_snapshot(self) -> dict[str, Any]:
        """返回本任务 token 预算累计快照（默认仅统计，不拦截）。"""
        tracker = self._token_budget
        if tracker is None:
            return {}
        return tracker.snapshot()



    def cancel(self) -> None:
        """请求取消当前任务（协作式：在下一个检查点生效）。

        任务 45：同时原子停止 pipeline 门，后续所有 dispatch / retry /
        replacement / replan 入口立即拒绝新动作。
        """
        self._cancel_requested = True
        self._gate.request_stop("cancelled")

    # ------------------------------------------------------------------
    # 状态迁移与取消检查（委托共享 WorkflowState）
    # ------------------------------------------------------------------
    def _transition(
        self, new_status: TaskStatus, *, reason: str = "", **extra: Any
    ) -> None:
        """状态迁移：委托共享状态（含影子状态机推进与事件留痕）。"""
        self._state.transition(new_status, reason=reason, **extra)

    def _check_cancel(self) -> None:
        """协作式取消检查点：委托共享状态。

        Raises:
            WorkflowCancelledError: 已通过 ``cancel()`` 请求取消时。
        """
        self._state.check_cancel()

    def _parse_agent_ref(self, agent_id: str) -> str:
        """从 Agent 引用解析所属子任务编号（委托共享状态）。"""
        return self._state.parse_agent_ref(agent_id)

    def _subtask_id_of(self, record: AgentRecord) -> str:
        """从 agent_id 解析所属子任务编号（委托共享状态）。"""
        return self._state.subtask_id_of(record)

    def _find_latest_agent(
        self, subtask_id: str, *, replacement: bool | None = None
    ) -> AgentRecord | None:
        """在注册表中查找某子任务最近的 Agent 记录（委托执行协调器基类实现）。"""
        return self._execution_coordinator._find_latest_agent(
            subtask_id, replacement=replacement
        )

    def _route_message(self, sender_id: str, message: dict[str, Any]) -> bool:
        """经 SupervisorRouter 向主管上报一条 JSON 消息（委托执行协调器基类实现）。"""
        return self._execution_coordinator._route_message(sender_id, message)

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

        生命周期收口：无论正常完成、人工升级（评审失败 / 结果评审
        不通过 / 网络失败）、pipeline 超时、取消还是未捕获异常，
        最外层 ``finally`` 都会统一幂等撤销本任务当前活跃授权
        （``tool_revoked`` 事件携带 ``lifecycle_cleanup:<原因>``）；
        重复清理不产生重复事件、不抛错，也不掩盖原始异常/返回值。
        """
        cleanup_reason = "unhandled_exception"
        # 任务 45：同一引擎实例复用时重置停止门（generation 保留递增）
        self._gate.reset()
        self._provisional_task = None
        self._provisional_output = None
        self._provisional_delivered_text = ""
        output: OrchestratorOutput | None = None
        try:
            async with async_span_scope(
                "so_agent.workflow",
                {
                    "task.objective_chars": len(objective),
                    "task.constraints_chars": len(constraints),
                    "task.acceptance_chars": len(acceptance_criteria),
                },
            ) as span:
                output = await self._run_pipeline(
                    objective, constraints, acceptance_criteria
                )
                set_span_attributes(
                    span,
                    {
                        "task.id": output.task_id,
                        "task.status": getattr(
                            output.status, "value", str(output.status)
                        ),
                    },
                )
            cleanup_reason = getattr(output.status, "value", str(output.status))
        except asyncio.CancelledError:
            cleanup_reason = "cancelled"
            self._gate.request_stop("cancelled")
            # 取消也保留最近检查点：已完成的子任务不应因外部取消被丢弃
            self._persist_checkpoint("cancelled")
            self._transition(TaskStatus.CANCELLED, reason="任务被外部取消（CancelledError）")
            raise
        except WorkflowCancelledError:
            cleanup_reason = "cancelled"
            self._gate.request_stop("cancelled")
            self._persist_checkpoint("cancelled")
            self._transition(TaskStatus.CANCELLED, reason="任务被取消（协作式）")
            output = self._build_output(
                TaskStatus.CANCELLED, final_result="任务已取消"
            )
        except PipelineStoppedError as exc:
            # 停止门拒绝新动作（BaseException 穿透各协调器宽捕获）：
            # 映射为 cancelled 终态，不触发重试 / 重规划
            cleanup_reason = "cancelled"
            self._transition(TaskStatus.CANCELLED, reason=str(exc))
            output = self._build_output(
                TaskStatus.CANCELLED, final_result="任务已取消"
            )
        except TokenBudgetExceeded as exc:
            # 预算熔断（BaseException）：不可重试，直接失败并保留结构化原因
            cleanup_reason = "token_budget_exceeded"
            self._transition(
                TaskStatus.FAILED,
                reason=f"token_budget_exceeded: {exc.violation.message}",
            )
            output = self._build_output(
                TaskStatus.FAILED,
                final_result=f"任务因 token 预算熔断终止：{exc.violation.message}",
            )
        except Exception as exc:  # 不可恢复的系统错误：置为 failed 终态
            cleanup_reason = "unhandled_exception"
            self._transition(TaskStatus.FAILED, reason=f"{type(exc).__name__}: {exc}")
            output = self._build_output(
                TaskStatus.FAILED,
                final_result=f"任务失败（系统错误）：{type(exc).__name__}: {exc}",
            )
        finally:
            # 统一幂等清理（所有退出路径）：撤销本任务当前活跃授权；
            # 同时兜底原子停止门（任何退出后 gate 均拒绝新动作）
            exit_reason = self._lifecycle_exit_reason or cleanup_reason
            self._lifecycle_exit_reason = ""
            self._gate.request_stop(f"lifecycle_exit:{exit_reason}")
            self._revoke_active_grants(
                reason=f"lifecycle_cleanup:{exit_reason}"
            )
        # 生命周期清理完成后再启动后台评审：避免 run 收尾的停止门
        # 与后台返工批次的派发互相竞争
        if output is None:
            raise WorkflowError("任务执行未产生输出")
        self._start_provisional_review_if_pending(output)
        return output


    # ------------------------------------------------------------------
    # 完整流水线（主循环：按状态分派到各协调器）
    # ------------------------------------------------------------------
    async def _run_pipeline(
        self, objective: str, constraints: str, acceptance_criteria: str
    ) -> OrchestratorOutput:
        """完整流水线：活动看门狗为主控，总时长默认不设上限。

        - ``pipeline_no_progress_timeout``：连续该秒数没有任何活动
          （SDK 流事件、工具生命周期、子任务完成、阶段推进、检查点
          落盘）才判定卡死并取消；正常长任务只要持续产出就不会被
          总时长误杀。
        - ``max_pipeline_timeout``：绝对兜底上限，仅在部署方显式开启
          （>0）时生效；<=0（默认）表示不设置。

        两种超时都先取消进行中的工作并携带已完成的部分结果转人工
        兜底；子任务级检查点保证续跑时只执行未完成项。
        """
        timeout = float(self._settings.max_pipeline_timeout)
        no_progress = float(
            getattr(self._settings, "pipeline_no_progress_timeout", 0.0) or 0.0
        )
        started_at = time.monotonic()
        self._pipeline_started_at = started_at
        # 绝对 deadline 同步进停止门；<=0 表示不设绝对上限
        self._gate.set_deadline(
            started_at + timeout if timeout > 0 else None
        )
        self._gate.record_progress("pipeline_start")

        main_task = asyncio.create_task(
            self._run_pipeline_main(objective, constraints, acceptance_criteria)
        )
        watchdog_task: asyncio.Task[str] | None = None
        if no_progress > 0:
            watchdog_task = asyncio.create_task(
                self._no_progress_watchdog(no_progress)
            )
        wait_set: set[asyncio.Task[Any]] = {main_task}
        if watchdog_task is not None:
            wait_set.add(watchdog_task)
        try:
            done, _ = await asyncio.wait(
                wait_set,
                timeout=timeout if timeout > 0 else None,
                return_when=asyncio.FIRST_COMPLETED,
            )
        except asyncio.CancelledError:
            self._gate.request_stop("cancelled")
            await self._cancel_pipeline_tasks(main_task, watchdog_task)
            raise

        if not done:
            # 绝对兜底到期
            self._gate.request_stop("pipeline_timeout")
            await self._cancel_pipeline_tasks(main_task, watchdog_task)
            return await self._escalate_on_pipeline_timeout(timeout)

        if main_task in done:
            if watchdog_task is not None and not watchdog_task.done():
                await self._cancel_pipeline_tasks(watchdog_task)
            try:
                return main_task.result()
            except PipelineDeadlineExceeded as exc:
                # 绝对 deadline 截断了下游调用预算：与 wait 超时同一
                # 兜底语义（携带已完成部分结果转人工），不消费任何
                # 业务失败额度、不覆盖首次停止原因前的进展快照
                logger.warning("流水线绝对 deadline 已耗尽：%s", exc)
                self._gate.request_stop("pipeline_timeout")
                return await self._escalate_on_pipeline_timeout(timeout)

        # 无进展看门狗先触发：取消进行中的阶段后优雅降级
        reason = "no_progress_timeout"
        try:
            reason = str(watchdog_task.result())  # type: ignore[union-attr]
        except Exception:
            pass
        self._gate.request_stop(reason)
        await self._cancel_pipeline_tasks(main_task)
        return await self._escalate_on_pipeline_stall(reason, no_progress)

    async def _no_progress_watchdog(self, window: float) -> str:
        """无进展看门狗：连续 ``window`` 秒无进展即返回停止原因。"""
        interval = min(max(window / 10.0, 0.05), 5.0)
        while True:
            await asyncio.sleep(interval)
            idle = self._gate.seconds_since_progress()
            if idle >= window:
                reason = f"no_progress_timeout:{window:g}s"
                self._gate.request_stop(reason)
                return reason

    @staticmethod
    async def _cancel_pipeline_tasks(
        *tasks: asyncio.Task[Any] | None,
    ) -> None:
        """取消并等待流水线内部任务收尾（不吞业务结果，异常按取消处理）。"""
        pending = [task for task in tasks if task is not None and not task.done()]
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

    def _handle_subtask_progress(self, subtask_id: str, result: Any) -> None:
        """子任务定型回调：重置看门狗并落检查点（超时后续跑依据）。"""
        self._gate.record_progress(f"subtask:{subtask_id}")
        self._persist_checkpoint(f"subtask:{subtask_id}")




    def _migrate_legacy_tool_store(self) -> None:
        """把旧 ``{sandbox}/generated_tools`` 一次性复制进工具库（失败不阻断任务）。"""
        if not bool(getattr(self._settings, "tool_store_migrate_legacy", True)):
            return
        configured = str(
            getattr(self._settings, "generated_tools_dir", "generated_tools")
            or ""
        ).strip()
        if configured not in ("", "generated_tools"):
            # 显式配置的旧目录仍是当前生效目录，无需迁移
            return
        try:
            report = migrate_legacy_generated_tools(
                Path(self._context.sandbox_dir) / "generated_tools",
                code_agent_agent_mod.resolve_generated_tools_dir(
                    self._context, self._settings
                ),
            )
        except Exception as exc:
            logger.warning("旧工具库迁移失败：%s", exc)
            return
        if report.migrated:
            logger.info("旧生成工具已迁移到工具库：%s", report.migrated)
        if report.errors:
            logger.warning("旧生成工具迁移存在问题：%s", report.errors)

    def _maintain_tool_store(
        self, active_tools: Iterable[str] = ()
    ) -> None:
        """按配置的间隔清理过期生成工具（失败只记录，不影响任务）。

        ``active_tools`` 为当前计划引用到的工具名，维护时永不清理。
        """
        try:
            report = sweep_generated_tools(
                tools_dir=code_agent_agent_mod.resolve_generated_tools_dir(
                    self._context, self._settings
                ),
                registry_path=code_agent_agent_mod.resolve_generated_registry_path(
                    self._context, self._settings
                ),
                archive_dir=resolve_archive_dir(self._settings),
                settings=self._settings,
                candidate_ttl_days=int(
                    getattr(self._settings, "tool_candidate_ttl_days", 7)
                ),
                approved_ttl_days=int(
                    getattr(self._settings, "tool_approved_ttl_days", 30)
                ),
                deprecation_grace_days=int(
                    getattr(self._settings, "tool_deprecation_grace_days", 7)
                ),
                archive_retention_days=int(
                    getattr(self._settings, "tool_archive_retention_days", 90)
                ),
                force=False,
                min_interval_minutes=int(
                    getattr(self._settings, "tool_sweep_interval_minutes", 1440)
                ),
                active_tools=frozenset(active_tools),
            )
        except Exception as exc:  # 生命周期维护不得阻断任务
            logger.warning("工具生命周期维护失败：%s", exc)
            return
        if report.archived or report.deprecated or report.errors:
            logger.info(
                "工具生命周期维护：弃用=%s 归档=%s 错误=%s",
                report.deprecated,
                report.archived,
                report.errors,
            )

    async def _refresh_external_tools(self) -> None:
        """按需刷新外部 MCP 目录并注册工具（目录异常不影响任务）。"""
        if not bool(getattr(self._settings, "mcp_directory_enabled", True)):
            return
        try:
            report = await self._mcp_directory.refresh(self._tool_registry)
        except Exception as exc:
            logger.warning("外部 MCP 目录刷新失败：%s", exc)
            return
        if report.registered:
            logger.info("外部 MCP 工具已注册：%s", report.registered)
        if report.errors:
            logger.warning("外部 MCP 目录存在问题：%s", report.errors)

    async def _run_pipeline_main(
        self, objective: str, constraints: str, acceptance_criteria: str
    ) -> OrchestratorOutput:
        """完整流水线主循环：接收 → 拆解/评审 → 执行/失败升级 → 汇总/终审 → 终态。"""
        # 1) 接收任务 → received
        self._create_task(objective, constraints, acceptance_criteria)
        # 旧工具库迁移：沙箱内历史 generated_tools 复制到主控侧 ToolStore
        self._migrate_legacy_tool_store()
        # 外部 MCP 目录：任务开始前按目录签名刷新，复制进来的 Server 此时生效
        await self._refresh_external_tools()
        # Token 预算追踪（默认启用，800k/400k）：同一任务内所有模型调用共用
        self._activate_token_budget()

        # 2) 先走完整流程拆解（含评审）
        try:
            plan_accepted = await self._plan_and_review()
        except SupervisorNetworkError as exc:
            return await self._escalate_to_human(f"网络连续失败：{exc}")
        if not plan_accepted:
            return await self._escalate_to_human("计划评审连续未通过：任务拆分修改额度耗尽")

        # 3) 工具库维护：计划中引用的工具受保护，其余按 TTL 清理
        active_tool_names = {
            name for spec in self._subtasks for name in spec.allowed_tools
        }
        self._maintain_tool_store(active_tool_names)

        # 4) 后置判断：规划结果是否为简单任务
        is_simple = self._is_simple_plan()

        self._pending_specs = list(self._subtasks)

        # 5) 执行/汇总/终审循环（含返工）
        return await self._run_result_loop(is_simple=is_simple)

    async def _run_result_loop(self, *, is_simple: bool) -> OrchestratorOutput:
        """执行 → 汇总 → 终审循环（含结果评审返工与人工兜底升级）。

        由首次流水线与人工指导恢复共用：调用前必须已具备当前计划与
        待执行批次（``_pending_specs``），本方法只负责执行后的闭环。
        """
        while True:
            # 生命周期维护：主动清理已过期授权（仅标记撤销，不续期、
            # 不扩大授权——续签只发生在新执行窗口开始时的批次重签发）
            self._gateway.purge_expired()
            self._check_cancel()
            # 停止门（任务 45）：每个循环周期开始前拒绝新动作
            self._gate.check("cycle")
            # 预算检查点：上一周期（含嵌套 Agent）的用量在此统一结算
            self._enforce_token_budget()

            # 4.1) 执行直至无失败（失败自动重规划；额度耗尽返回 None）
            try:
                execution = await self._run_execution_cycle()
            except SupervisorNetworkError as exc:
                return await self._escalate_to_human(f"网络连续失败：{exc}")
            if execution is None:
                return await self._escalate_to_human("执行阶段持续失败：任务拆分修改额度耗尽")

            # 4.2) 简单任务快速通道：全部成功且产出充分时跳过汇总与
            # 结果评审；产出缺失/过短或配置关闭时仍走完整链路
            all_success = all(r.success for r in self._results.values())
            if (
                is_simple
                and all_success
                and bool(self._settings.skip_result_review_for_simple)
                and self._simple_outputs_sufficient()
            ):
                final_text = self._fallback_summary(self._successful_results())
                logger.info("简单任务全部成功且产出充分，跳过汇总与评审（快速通道）")
                self._transition(TaskStatus.COMPLETED)
                return self._build_output(
                    TaskStatus.COMPLETED, final_result=final_text
                )

            # 4.3) 汇总 → aggregating（结果评审按配置分级、可选并行）
            run_review, review_reason = self._should_run_result_review()
            if not run_review:
                self._events.log_event(
                    EVENT_REVIEW_SKIPPED,
                    {
                        "task_id": self._task_request.task_id,
                        "stage": "result_review",
                        "reason": review_reason,
                    },
                )
                final_text = await self._aggregate_results()
                self._transition(TaskStatus.COMPLETED)
                return self._build_output(
                    TaskStatus.COMPLETED, final_result=final_text
                )

            # 4.4) 异步 provisional：低风险任务先交付，评审在后台继续
            if self._should_deliver_provisionally():
                final_text = await self._aggregate_results()
                self._transition(
                    TaskStatus.COMPLETED_PROVISIONAL,
                    reason="低风险结果先行交付，后台评审中",
                )
                self._events.log_event(
                    EVENT_PROVISIONAL_DELIVERED,
                    {
                        "task_id": self._task_request.task_id,
                        "agent_id": DEFAULT_SUPERVISOR_ID,
                        "plan_version": self._plan_version,
                        "subtask_ids": [
                            spec.subtask_id for spec in self._subtasks
                        ],
                    },
                )
                self._provisional_delivered_text = final_text
                output = self._build_output(
                    TaskStatus.COMPLETED_PROVISIONAL,
                    final_result=final_text,
                    provisional=True,
                    notice=(
                        "结果已先行交付，后台评审正在进行；"
                        "如发现问题将自动修订并通知，旧版本会保留。"
                    ),
                )
                self._provisional_output = output
                return output

            # 4.5) 最终评审 → result_review：无论是否有验收标准一律执行
            #（对照 objective 校验整体结果，无验收标准不再跳过）
            if bool(
                getattr(
                    self._settings,
                    "result_review_parallel_with_aggregation",
                    False,
                )
            ):
                self._transition(TaskStatus.AGGREGATING)
                final_text, decision = await asyncio.gather(
                    self._aggregate_results(transition=False),
                    self._final_review(transition=False),
                )
                self._transition(TaskStatus.RESULT_REVIEW)
            else:
                final_text = await self._aggregate_results()
                decision = await self._final_review()
            if decision.passed:
                logger.info("结果评审通过，任务完成")
                self._transition(TaskStatus.COMPLETED)
                return self._build_output(
                    TaskStatus.COMPLETED, final_result=final_text, review=decision
                )

            # 4.5) 结果评审不通过：仅返工被指出的子任务
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


    # ------------------------------------------------------------------
    # 输出组装（严格 JSON 通信的最终载体）
    # ------------------------------------------------------------------
    def _build_output(
        self,
        status: TaskStatus,
        *,
        final_result: str = "",
        review: ReviewDecision | None = None,
        provisional: bool = False,
        previous_result: str = "",
        notice: str = "",
    ) -> OrchestratorOutput:
        """构造统一的结构化输出（OrchestratorOutput）。

        Args:
            status: 终态（或过程中的最终状态）。
            final_result: 最终结果文本。
            review: 评审结论覆盖；未提供时使用最近一次评审结论。
            provisional: 是否为"已交付、后台评审中"的初步结果。
            previous_result: 后台返工前已交付的旧结果（修订时保留）。
            notice: 面向用户的交付/修订提示。
        """
        effective_review = review if review is not None else self._last_review
        task_id = self._task_request.task_id if self._task_request else ""
        plan_versions = self.plan_history
        agent_history = self._agent_history()
        review_summary = self._format_review(effective_review)
        audit = build_audit_report(
            events=self._events.events,
            task_id=task_id,
            status=status,
            final_result=final_result,
            review_summary=review_summary,
            plan_versions=plan_versions,
            agent_history=agent_history,
            results=self._results,
        )
        return OrchestratorOutput(
            task_id=task_id,
            status=status,
            final_result=final_result,
            plan_versions=plan_versions,
            agent_history=agent_history,
            review_summary=review_summary,
            provisional=provisional,
            previous_result=previous_result,
            notice=notice,
            human_escalation=(
                self._human_escalation.model_copy(deep=True)
                if self._human_escalation is not None
                else None
            ),
            audit=audit,
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
    # 主管 / 评审 Agent 调用（严格 JSON 输入输出；Runner 保持模块级可替换）
    # ------------------------------------------------------------------












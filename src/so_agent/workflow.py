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
  （HumanEscalationHandler，本模块对其 re-export）。

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
import json
import sys
import time
from pathlib import Path
from typing import Any

from agents import Agent, Model, Runner

import so_agent.tool_packages.code_agent.agent as code_agent_agent_mod
from so_agent.aggregation_coordinator import AggregationCoordinator
from so_agent.config import Settings
from so_agent.context import ProjectContext
from so_agent.execution_coordinator import ExecutionCoordinator
from so_agent.mcp.client import MCPClient
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
    EVENT_AGENT_CREATED,
    EVENT_AGENT_RELEASED,
    EVENT_DECOMPOSITION_ISSUE,
    EVENT_FAILURE_CLASSIFIED,
    EVENT_HUMAN_ESCALATION,
    EVENT_RECOVERY_DECIDED,
    EVENT_REPLACEMENT_REQUESTED,
    EVENT_RETRY_SUPPRESSED,
    EVENT_REVIEW_AGENT_OUTPUT,
    EVENT_REVIEW_CONTRACT_ERROR,
    EVENT_REVIEW_CONTRACT_REPAIR,
    EVENT_REVIEW_PARSE_ERROR,
    EVENT_TASK_DISPATCHED,
    EVENT_TOOL_GRANTED,
    EVENT_TOOL_REVOKED,
    EventLogger,
)
from so_agent.runtime.permissions import PermissionGateway
from so_agent.runtime.registry import DEFAULT_SUPERVISOR_ID, AgentRegistry
from so_agent.runtime.scheduler import SubtaskDAGScheduler
from so_agent.runtime.supervisor_router import SupervisorRouter
from so_agent.runtime.tool_registry import ToolRegistry
from so_agent.state_machine import (
    PipelineStoppedError,
    ReviewContractError,
    SupervisorNetworkError,
    TaskStateMachine,
    WorkflowCancelledError,
    WorkflowError,
    WorkflowState,
    _StructuredOutputT,
    _coerce_structured_output,
    _review_contract_contradiction,
)
from so_agent.tool_packages.review_agent.agent import ReviewStage, create_review_agent
from so_agent.workflow_support import (
    HumanEscalationHandler,
    PipelineGate,
    WorkflowDeps,
    make_tool_timeout_resolver,
    run_agent_streamed,
)

__all__ = [
    "AggregationCoordinator",
    "ExecutionCoordinator",
    "HumanEscalationHandler",
    "PlanningCoordinator",
    "ReviewContractError",
    "PipelineGate",
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

# 简单任务快速通道的产出充分性阈值：全部成功子任务的 output 非空且
# 长度严格大于该值时才允许跳过汇总与结果评审，防止空白/过短产出
# 借快速通道绕过质量把关
_SIMPLE_FAST_PATH_MIN_OUTPUT_CHARS: int = 50

# 契约修复调用的严格字段模板（任务 32）：随修复指令下发给评审 Agent，
# 要求只修复字段一致性（passed 与 required_fixes/suggestions 的组合），
# 禁止重新评审或改变评审事实
_REVIEW_CONTRACT_TEMPLATE: dict[str, Any] = {
    "stage": "plan_review 或 result_review（与本次评审阶段一致）",
    "passed": True,
    "issues": [],
    "suggestions": [],
    "required_fixes": [],
    "retry_target": None,
    "summary": "",
}

# 契约修复消息中携带的原始输出截断上限（避免把完整敏感载荷无限写日志
# /无限回传模型；正常评审结论远小于该阈值，仅防御异常超长输出）
_REVIEW_CONTRACT_REPAIR_MAX_ORIGINAL_CHARS: int = 4000

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
        EVENT_FAILURE_CLASSIFIED,
        EVENT_RECOVERY_DECIDED,
        EVENT_RETRY_SUPPRESSED,
    }
)

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
    "_combine_reviews": "_planning_coordinator",
    "_try_consume_replan_quota": "_planning_coordinator",
    "_is_simple_plan": "_planning_coordinator",
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


class WorkflowEngine:
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
                mcp_client=mcp_client,
                gateway=self._gateway,
            )
        )
        self._review_agent = (
            review_agent
            if review_agent is not None
            else create_review_agent(context, settings=self._settings)
        )
        self._result_store = (
            result_store if result_store is not None else InMemoryResultStore()
        )

        # ---- 共享状态 + 协调器装配（协调器之间经 state/deps 解耦） ----
        # pipeline 停止门（任务 45）：超时 / 取消 / 人工终止后原子进入
        # stopping，全部 dispatch / retry / replacement / replan / 计划
        # 采纳 / 授权签发入口先检查，停止后拒绝一切新动作
        self._gate = PipelineGate()
        self._state = WorkflowState(events=self._events)
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
        # 生命周期清理原因提示：特定退出路径（如 pipeline 超时）在结构化
        # 终态（human_required）之外提供更细的清理原因；run() 的最外层
        # finally 据此填充 tool_revoked 审计事件的 reason
        self._lifecycle_exit_reason: str = ""
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
        try:
            output = await self._run_pipeline(
                objective, constraints, acceptance_criteria
            )
            cleanup_reason = getattr(output.status, "value", str(output.status))
            return output
        except asyncio.CancelledError:
            cleanup_reason = "cancelled"
            self._gate.request_stop("cancelled")
            self._transition(TaskStatus.CANCELLED, reason="任务被外部取消（CancelledError）")
            raise
        except WorkflowCancelledError:
            cleanup_reason = "cancelled"
            self._gate.request_stop("cancelled")
            self._transition(TaskStatus.CANCELLED, reason="任务被取消（协作式）")
            return self._build_output(TaskStatus.CANCELLED, final_result="任务已取消")
        except PipelineStoppedError as exc:
            # 停止门拒绝新动作（BaseException 穿透各协调器宽捕获）：
            # 映射为 cancelled 终态，不触发重试 / 重规划
            cleanup_reason = "cancelled"
            self._transition(TaskStatus.CANCELLED, reason=str(exc))
            return self._build_output(TaskStatus.CANCELLED, final_result="任务已取消")
        except Exception as exc:  # 不可恢复的系统错误：置为 failed 终态
            cleanup_reason = "unhandled_exception"
            self._transition(TaskStatus.FAILED, reason=f"{type(exc).__name__}: {exc}")
            return self._build_output(
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

    # ------------------------------------------------------------------
    # 完整流水线（主循环：按状态分派到各协调器）
    # ------------------------------------------------------------------
    async def _run_pipeline(
        self, objective: str, constraints: str, acceptance_criteria: str
    ) -> OrchestratorOutput:
        """完整流水线（含整体超时保护）。

        整条流水线受 ``max_pipeline_timeout`` 保护：超时后取消进行中的
        阶段并优雅降级——携带已完成的部分结果转入人工兑底，而不是直接
        崩溃或静默丢弃已完成的工作。启动时记录 monotonic 计时基准
        （``_pipeline_started_at``），供 ``_run_supervisor`` 把主管绝对
        总预算截断到 pipeline 剩余预算。
        """
        timeout = float(self._settings.max_pipeline_timeout)
        self._pipeline_started_at = time.monotonic()
        # 任务 45：统一绝对 deadline 同步进停止门，剩余时间计算一律经
        # gate.remaining()（永不为负）
        self._gate.set_deadline(self._pipeline_started_at + timeout)
        try:
            return await asyncio.wait_for(
                self._run_pipeline_main(
                    objective, constraints, acceptance_criteria
                ),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            # 原子进入 stopping：后续所有新动作入口立即拒绝（任务 45）
            self._gate.request_stop("pipeline_timeout")
            return await self._escalate_on_pipeline_timeout(timeout)

    def _authoritative_success_results(self) -> list[ExecutionResult]:
        """权威成功结果：跨版本归档与当前/调度器结果的合并视图（深拷贝）。

        任务 45：pipeline 超时的完成计数依据。重规划采纳会清空
        ``self._results``，仅依赖当前计划的成功结果会低估真实完成量；
        合并顺序：``archive_results`` 按 (plan_version, subtask_id) 升序
        遍历（新版本覆盖旧版本）→ 当前 ``self._results`` 覆盖 → 调度器
        清理期落账的部分结果覆盖（不依赖被重规划修改的临时 pending
        集合），最后仅保留 success 项（部分结果同样准确）。
        """
        merged: dict[str, ExecutionResult] = {}
        archive = self._archive_results
        for version, subtask_id in sorted(
            archive.keys(), key=lambda key: (key[0], key[1])
        ):
            merged[subtask_id] = archive[(version, subtask_id)]

        def _merge(incoming: dict[str, ExecutionResult]) -> None:
            # 成功结果总是覆盖（最新成功优先）；失败结果不抹掉已有的成功
            # 记录：归档成功是已持久化的事实，当前版本重试 / 重规划失败
            # 不得低估真实完成量（任务 45：部分结果必须准确）
            for sid, result in incoming.items():
                existing = merged.get(sid)
                if result.success or existing is None or not existing.success:
                    merged[sid] = result

        _merge(self._results)
        scheduler = self._scheduler
        if scheduler is not None:
            _merge(getattr(scheduler, "results", {}) or {})
        return [
            item.model_copy(deep=True)
            for item in merged.values()
            if item.success
        ]

    async def _escalate_on_pipeline_timeout(
        self, timeout: float
    ) -> OrchestratorOutput:
        """流水线整体超时：携带已完成的部分结果优雅降级（转人工兑底）。

        与直接失败不同，此处保留权威成功结果（归档 + 当前 + 调度器
        落账，任务 45）的子任务产出，以本地确定性摘要并入 final_result，
        避免超时把已完成的工作一并丢弃；同时走标准人工兑底链路
        （状态迁移 + 升级材料包留痕）。
        """
        self._lifecycle_exit_reason = "pipeline_timeout"
        completed = self._authoritative_success_results()
        total = max(len(self._subtasks), len(completed))
        reason = (
            f"流水线整体超时（上限 {timeout:.0f} 秒）："
            f"已完成 {len(completed)}/{total} 个子任务"
        )
        output = await self._escalate_to_human(reason)
        if completed:
            partial = self._fallback_summary(completed)
            output = output.model_copy(
                update={
                    "final_result": (
                        f"{output.final_result}\n已完成的部分结果：\n{partial}"
                    )
                }
            )
        return output

    async def _run_pipeline_main(
        self, objective: str, constraints: str, acceptance_criteria: str
    ) -> OrchestratorOutput:
        """完整流水线主循环：接收 → 拆解/评审 → 执行/失败升级 → 汇总/终审 → 终态。"""
        # 1) 接收任务 → received
        self._create_task(objective, constraints, acceptance_criteria)

        # 2) 先走完整流程拆解（含评审）
        try:
            plan_accepted = await self._plan_and_review()
        except SupervisorNetworkError as exc:
            return await self._escalate_to_human(f"网络连续失败：{exc}")
        if not plan_accepted:
            return await self._escalate_to_human("计划评审连续未通过：任务拆分修改额度耗尽")

        # 3) 后置判断：规划结果是否为简单任务
        is_simple = self._is_simple_plan()

        self._pending_specs = list(self._subtasks)

        # 4) 执行/汇总/终审循环（含返工）
        while True:
            # 生命周期维护：主动清理已过期授权（仅标记撤销，不续期、
            # 不扩大授权——续签只发生在新执行窗口开始时的批次重签发）
            self._gateway.purge_expired()
            self._check_cancel()
            # 停止门（任务 45）：每个循环周期开始前拒绝新动作
            self._gate.check("cycle")

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
                print(
                    "[so-agent] 简单任务全部成功且产出充分，跳过汇总与评审（快速通道）",
                    file=sys.stderr, flush=True,
                )
                self._transition(TaskStatus.COMPLETED)
                return self._build_output(
                    TaskStatus.COMPLETED, final_result=final_text
                )

            # 4.3) 汇总 → aggregating
            final_text = await self._aggregate_results()

            # 4.4) 最终评审 → result_review：无论是否有验收标准一律执行
            #（对照 objective 校验整体结果，无验收标准不再跳过）
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
        for record in list(self._events.events):
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
    # 主管 / 评审 Agent 调用（严格 JSON 输入输出；Runner 保持模块级可替换）
    # ------------------------------------------------------------------
    def _record_model_usage(self, result: Any, agent_label: str) -> None:
        """记录一次模型调用的 token 用量（写入事件日志，随日志文件落盘）。"""
        try:
            input_tok = output_tok = total_tok = 0
            requests = 0
            for resp in getattr(result, "raw_responses", None) or []:
                usage = getattr(resp, "usage", None)
                if usage is None:
                    continue
                requests += 1
                input_tok += getattr(usage, "input_tokens", 0) or 0
                output_tok += getattr(usage, "output_tokens", 0) or 0
                total_tok += getattr(usage, "total_tokens", 0) or 0
            self._events.log_event(
                "token_usage",
                {
                    "agent": agent_label,
                    "requests": requests,
                    "input_tokens": input_tok,
                    "output_tokens": output_tok,
                    "total_tokens": total_tok,
                },
            )
        except Exception:  # 用量记录失败不得影响主流程
            return

    def _supervisor_budget(self) -> float:
        """主管单次调用的绝对总预算（秒）：不短于嵌套子 Agent 工具窗口，
        且不超过 pipeline 剩余预算。

        - 基准取 ``supervisor_total_timeout``（配置校验器保证其不短于
          ``subagent_tool_execution_timeout``，父级总预算不会先于嵌套
          子 Agent 工具窗口触发，消除嵌套调用的外层提前误杀）；
        - pipeline 已启动（``_run_pipeline`` 已设置门 deadline）时，再与
          ``max_pipeline_timeout`` 的剩余预算取较小值，保证 2400 秒
          流水线总上限仍是最高优先级硬约束；剩余时间一律经停止门
          ``remaining()`` 计算（永不为负，任务 45）；
        - 直接调用 ``_run_supervisor``（未启动流水线）时使用完整基准，
          保持既有测试与降级路径语义；子 Agent 自身仍是
          ``agent_total_timeout`` 硬上限，不被父级预算无上限延长。
        """
        budget = float(self._settings.supervisor_total_timeout)
        remaining = self._gate.remaining()
        if remaining is not None:
            budget = min(budget, remaining)
        return budget

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
        # 分阶段流式超时（任务 33）：首事件窗口（覆盖 SDK 冷启动与模型
        # 首包等待）/ 模型空闲窗口 / 工具执行窗口三类阶段窗口均被主管
        # 绝对总预算截断——只要 SDK 原生事件持续产出就不会被空闲超时
        # 误杀；工具阶段中的嵌套子 Agent 执行工具（subagent_creator /
        # execute_subagent，按 ToolCallItem 内容识别）使用独立的
        # subagent_tool_execution_timeout（默认 610s，覆盖子 Agent
        # 600s 总预算与清理宽限），不会被普通工具窗口 180s 提前误杀；
        # 任一超时先取消底层流再抛 asyncio.TimeoutError（消息携带
        # phase），由调用方按各自策略优雅降级（执行尝试记为失败、
        # 汇总回退本地摘要、重规划转结构性问题），不会直接崩溃主流程；
        # 超时前写入 timeout_audit 审计事件；runner=Runner 保持
        # workflow_mod.Runner 的模块级 monkeypatch 语义（测试可整体
        # 替换假 Runner）
        #
        # 主管绝对总预算（任务 33 嵌套修复）：取
        # supervisor_total_timeout（配置校验器保证不短于嵌套子 Agent
        # 工具窗口），且不超过 pipeline 剩余预算——2400s 流水线总上限
        # 仍是最高优先级硬约束；子 Agent 自身仍是 agent_total_timeout
        # （600s）硬上限，不被父级预算无限延长。
        supervisor_budget = self._supervisor_budget()
        # 停止门预检 + 最小执行窗口（任务 45）：停止后拒绝新调用
        #（穿透宽捕获）；剩余预算 <= 0 时不调用模型、不进入流式等待，
        # 直接抛出可审计的预算超时（与 wait_for 超时同一下降级语义，
        # 由调用方按既有策略优雅降级）
        self._gate.check("supervisor_call")
        if supervisor_budget <= 0.0:
            raise asyncio.TimeoutError(
                f"pipeline 剩余预算不足（剩余 {supervisor_budget:g}s），"
                "跳过本次主管调用"
            )
        gate_generation_before = self._gate.generation
        task_request = self._task_request
        result = await run_agent_streamed(
            self._orchestrator,
            input=message,
            max_turns=int(self._settings.max_model_turns),
            first_event_timeout=float(self._settings.first_event_timeout),
            model_idle_timeout=float(self._settings.model_idle_timeout),
            tool_execution_timeout=float(
                self._settings.tool_execution_timeout
            ),
            subagent_tool_execution_timeout=float(
                self._settings.subagent_tool_execution_timeout
            ),
            agent_total_timeout=supervisor_budget,
            runner=Runner,
            events=self._events,
            # 工具阶段窗口按实际 tool_name 的超时 profile 解析
            #（任务 45：code_agent / subagent / sandbox / standard），
            # 总预算仍被 agent_total 与 pipeline deadline 截断
            tool_timeout_resolver=make_tool_timeout_resolver(
                self._settings, self._tool_registry
            ),
            audit_context={
                "agent": "supervisor",
                "pipeline_phase": (
                    payload.get("phase") if isinstance(payload, dict) else None
                ),
                "task_id": (
                    task_request.task_id if task_request is not None else None
                ),
            },
        )
        # 迟到结果隔离（任务 45）：调用返回时门已停止或代数变化（等待
        # 期间发生取消 / deadline 到期 / 人工终止），本结果视为迟到——
        # 不写 WorkflowState、不记用量、不触发重试或授权签发，抛出
        # 穿透宽捕获的停止信号由 run() 映射为 cancelled 终态
        if (
            self._gate.stopping
            or self._gate.generation != gate_generation_before
        ):
            raise PipelineStoppedError(
                f"主管调用返回时 pipeline 已停止"
                f"（{self._gate.stop_reason}），结果按迟到隔离"
            )
        self._record_model_usage(result, "supervisor")
        return _coerce_structured_output(
            result.final_output, output_model, label="主管 Agent"
        )

    async def _call_review_agent(self, review_input: Any) -> ReviewDecision:
        """调用评审 Agent：JSON 输入 → 契约校验后的 ReviewDecision。

        任务 32 确定化契约：评审放行只依据解析出的合法 ReviewDecision
        的 ``passed`` 布尔字段，绝不从评审文本语气或"通过/未通过"等
        关键词推断通过与否；结构矛盾先修复契约，无法修复时 fail closed。

        处理链路（全部 fail closed，宁可拦截不可放行）：

        1. 调用 + 宽容解析（``_coerce_review_decision``）：提取出的
           JSON/对象缺失显式可解析 ``passed`` 字段时抛
           ``ReviewContractError`` → 保守未通过 +
           ``review_contract_error`` 事件；调用超时/完全无法解析
           （``WorkflowError``）→ 保守未通过 + ``review_parse_error``
           事件；其余意外异常向上传播不吞掉；
        2. 解析成功后执行字段契约校验（``_review_contract_contradiction``）：
           ``passed=true`` 时 ``required_fixes`` 必须为空、``passed=false``
           时必须非空（``suggestions`` 非空不阻断）；
        3. 契约矛盾时仅进行一次契约修复调用（只修字段一致性，禁止重新
           评审/改变评审事实）；修复结果再次解析与校验；
        4. 修复后仍矛盾、缺失显式 passed 或解析失败 → fail closed 的
           ReviewDecision（``required_fixes`` 写明评审契约无效）+
           ``review_contract_error`` 事件。

        ``review_agent_output`` 事件保留 stage / passed / raw_output_preview
        并新增 ``contract_valid``；原始输出预览截断至 1000 字符。``stage``
        统一归一化为 plan_review / result_review（授权评审在结论中映射为
        plan_review）。
        """
        expected_stage: str = (
            "result_review"
            if review_input.stage == ReviewStage.RESULT_REVIEW
            else "plan_review"
        )
        result: Any = None
        try:
            message = review_input.model_dump_json()
            result = await Runner.run(
                self._review_agent,
                input=message,
                max_turns=int(self._settings.max_model_turns),
            )
            self._record_model_usage(result, "review_agent")
            decision = self._coerce_review_decision(result.final_output)
        except ReviewContractError as exc:
            # 输出缺失显式可解析 passed 字段：无法构造合法评审决定，
            # fail closed（不猜测结论），契约错误事件留痕
            raw_output = str(result.final_output) if result else "N/A"
            self._events.log_event(
                EVENT_REVIEW_CONTRACT_ERROR,
                {
                    "stage": expected_stage,
                    "error": f"{type(exc).__name__}: {exc}",
                    "raw_output_preview": raw_output[:500],
                },
            )
            decision = ReviewDecision(
                stage=expected_stage,
                passed=False,
                issues=[f"评审契约无效：{exc}"],
                required_fixes=[
                    "评审契约无效：输出缺失显式 passed 字段，"
                    "须修复评审输出契约后重新提交评审"
                ],
                summary="评审契约无效（缺失显式 passed 字段），保守判定为未通过",
            )
            self._log_review_agent_output(
                expected_stage, decision, raw_output, contract_valid=False
            )
            return decision
        except (WorkflowError, asyncio.TimeoutError) as exc:
            self._events.log_event(
                EVENT_REVIEW_PARSE_ERROR,
                {
                    "stage": expected_stage,
                    "error": f"{type(exc).__name__}: {exc}",
                    "raw_output_preview": (
                        str(result.final_output)[:500] if result else "N/A"
                    ),
                },
            )
            return ReviewDecision(
                stage=expected_stage,
                passed=False,
                issues=[f"评审 Agent 调用/解析失败：{type(exc).__name__}: {exc}"],
                required_fixes=["修复评审链路后重新提交评审"],
                summary="评审不可用，保守判定为未通过",
            )

        # 解析成功后执行字段契约校验；矛盾时仅一次契约修复调用
        contradiction = _review_contract_contradiction(decision)
        contract_valid = contradiction is None
        if contradiction is not None:
            decision, contract_valid = await self._repair_review_contract(
                expected_stage, review_input, result.final_output, contradiction
            )
        decision.stage = expected_stage
        # 成功解析后记录评审原始输出（截断避免日志膨胀）
        raw_output = str(result.final_output) if result.final_output is not None else ""
        self._log_review_agent_output(
            expected_stage, decision, raw_output, contract_valid=contract_valid
        )
        return decision

    def _log_review_agent_output(
        self,
        stage: str,
        decision: ReviewDecision,
        raw_output: str,
        *,
        contract_valid: bool,
    ) -> None:
        """记录 ``review_agent_output`` 审计事件（含契约有效性标记）。"""
        self._events.log_event(
            EVENT_REVIEW_AGENT_OUTPUT,
            {
                "stage": stage,
                "passed": decision.passed,
                "contract_valid": contract_valid,
                "raw_output_preview": raw_output[:1000],
            },
        )

    async def _repair_review_contract(
        self,
        expected_stage: str,
        review_input: Any,
        raw_output: Any,
        contradiction: str,
    ) -> tuple[ReviewDecision, bool]:
        """对契约矛盾的评审输出执行一次（且仅一次）契约修复调用。

        修复输入 = 原始评审输入 JSON + ``contract_repair`` 附加字段
        （原始输出、矛盾说明、严格字段模板与修复指令），要求评审
        Agent 只修复字段一致性：调整 passed 与 required_fixes /
        suggestions 的组合使契约自洽，禁止重新评审或改变评审事实。
        修复输出再次解析并做同样的契约校验；仍矛盾、缺失显式
        passed 或解析失败时 fail closed（保守未通过）。

        审计：``review_contract_repair`` 事件记录 attempt（固定 1）、
        原始矛盾与最终状态（repaired / failed），修复输出预览截断至
        500 字符；最终 fail closed 时另记 ``review_contract_error``。

        Args:
            expected_stage: 归一化后的评审阶段（plan_review / result_review）。
            review_input: 原始评审输入（ReviewInput，修复消息保留其全部
                字段供评审 Agent 定位上下文）。
            raw_output: 原始评审输出（final_output，任意形态）。
            contradiction: ``_review_contract_contradiction`` 给出的矛盾说明。

        Returns:
            ``(decision, contract_valid)``：修复成功时为修复后的合法
            结论（True）；失败时为 fail closed 结论（False）。
        """
        repair_message = self._build_contract_repair_message(
            review_input, raw_output, contradiction
        )
        repair_result: Any = None
        repair_error = ""
        decision: ReviewDecision | None = None
        try:
            repair_result = await Runner.run(
                self._review_agent,
                input=repair_message,
                max_turns=int(self._settings.max_model_turns),
            )
            self._record_model_usage(repair_result, "review_agent")
            repaired = self._coerce_review_decision(repair_result.final_output)
            residual = _review_contract_contradiction(repaired)
            if residual is not None:
                repair_error = f"修复输出仍违反契约：{residual}"
            else:
                decision = repaired
        except ReviewContractError as exc:
            repair_error = f"修复输出缺失显式 passed 字段：{exc}"
        except (WorkflowError, asyncio.TimeoutError) as exc:
            repair_error = f"修复调用/解析失败：{type(exc).__name__}: {exc}"

        final_status = "repaired" if decision is not None else "failed"
        self._events.log_event(
            EVENT_REVIEW_CONTRACT_REPAIR,
            {
                "stage": expected_stage,
                "attempt": 1,
                "contradiction": contradiction,
                "final_status": final_status,
                "repair_error": repair_error or None,
                "repair_output_preview": (
                    str(repair_result.final_output)[:500]
                    if repair_result is not None
                    else "N/A"
                ),
            },
        )
        if decision is not None:
            return decision, True

        self._events.log_event(
            EVENT_REVIEW_CONTRACT_ERROR,
            {
                "stage": expected_stage,
                "contradiction": contradiction,
                "repair_error": repair_error,
                "raw_output_preview": (
                    str(raw_output)[:500] if raw_output is not None else "N/A"
                ),
            },
        )
        return (
            ReviewDecision(
                stage=expected_stage,
                passed=False,
                issues=[f"评审契约无效：{contradiction}；{repair_error}"],
                required_fixes=[
                    "评审契约无效：须修复评审输出的字段契约后重新提交评审"
                ],
                summary="评审契约无效（契约修复未成功），保守判定为未通过",
            ),
            False,
        )

    @staticmethod
    def _build_contract_repair_message(
        review_input: Any, raw_output: Any, contradiction: str
    ) -> str:
        """构造契约修复消息：原始评审输入 JSON + contract_repair 附加字段。

        消息保留 ReviewInput 全部字段（评审 Agent 据此定位上下文；
        Pydantic 校验会忽略附加字段，既有调用链/测试桩兼容），附加的
        ``contract_repair`` 字段携带修复指令、矛盾说明、原始输出
        （超长截断至 ``_REVIEW_CONTRACT_REPAIR_MAX_ORIGINAL_CHARS``）
        与严格字段模板。
        """
        original_text = str(raw_output) if raw_output is not None else ""
        if len(original_text) > _REVIEW_CONTRACT_REPAIR_MAX_ORIGINAL_CHARS:
            original_text = (
                original_text[:_REVIEW_CONTRACT_REPAIR_MAX_ORIGINAL_CHARS]
                + "…（已截断）"
            )
        payload = json.loads(review_input.model_dump_json())
        payload["contract_repair"] = {
            "instruction": (
                "你上一次输出的评审结论 JSON 存在字段契约矛盾。"
                "请只修复字段一致性，使输出满足 strict_template 的字段约束："
                "passed=true 时 required_fixes 必须为空数组；"
                "passed=false 时 required_fixes 必须至少包含一条修复项；"
                "非阻断的改进建议一律写入 suggestions 数组。"
                "禁止重新评审、禁止改变评审事实：不得修改 issues 的内容、"
                "不得翻转 passed 的评审结论语义，只做上述字段组合的"
                "一致性调整。只输出修复后的 ReviewDecision JSON"
                "（不包裹代码围栏、不附加任何解释文字）。"
            ),
            "contradiction": contradiction,
            "original_output": original_text,
            "strict_template": _REVIEW_CONTRACT_TEMPLATE,
        }
        return json.dumps(payload, ensure_ascii=False)

    @staticmethod
    def _coerce_review_decision(output: Any) -> ReviewDecision:
        """把评审 Agent 输出规范化为 ReviewDecision（宽容解析，含围栏剥离）。

        评审 Agent 同样以纯文本返回（output_type=None），复用
        ``_coerce_structured_output`` 的四形态宽容解析；仅接受显式
        可解析 ``passed`` 字段（缺失时抛 ``ReviewContractError``），
        不做任何文本关键词结论推断。

        Raises:
            WorkflowError: 输出无法解析为 ReviewDecision 时。
            ReviewContractError: 提取出 JSON/对象但缺失显式可解析
                passed 字段时。
        """
        return _coerce_structured_output(output, ReviewDecision, label="评审 Agent")

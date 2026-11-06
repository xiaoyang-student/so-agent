"""协调器公共支撑：依赖容器、停止门与协调器基类。

薄编排器（WorkflowEngine）与三个协调器（Planning / Execution /
Aggregation）之间的解耦层：

- ``PipelineGate``：pipeline 级停止门、绝对 deadline 与进展看门狗；
- ``WorkflowDeps``：一次性收集基础设施依赖（项目上下文、配置、事件、
  Agent 注册表、授权网关、主管路由器、调度器、工具注册表、主管/评审
  Agent、MCP 客户端）。主管与评审 Agent 的调用通道（run_supervisor /
  call_review_agent / record_model_usage）绑定薄编排器（WorkflowEngine）
  的方法后注入——``Runner`` 的模块级 monkeypatch 语义（测试在 workflow
  模块上打补丁）因此保持不变；跨协调器调用（plan_and_review /
  issue_grants / consume_replan_quota）由编排器绑定 PlanningCoordinator
  后注入，协调器之间不直接互相引用；
- ``BaseCoordinator``：协调器基类。原 WorkflowEngine 的公共辅助方法
  （状态迁移、取消检查、主管路由、Agent 记录查找/解析）与历史私有
  状态属性（``_task_request`` / ``_subtasks`` / ``_results`` 等，代理到
  共享 WorkflowState）在此集中定义，使平移的方法体保持逐字不变；
- ``HumanEscalationHandler``（workflow_human）与 ``run_agent_streamed``
  （workflow_streaming）已拆分到独立模块；本模块继续 re-export，既有
  导入路径与 monkeypatch 目标保持不变。
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from agents import Agent, Runner

from so_agent.config import Settings
from so_agent.context import ProjectContext
from so_agent.mcp.client import MCPClient
from so_agent.models import (
    AgentRecord,
    ReviewDecision,
    ToolGrant,
)
from so_agent.runtime.events import (
    EVENT_CONTEXT_COMPRESSED,
    EventLogger,
)
from so_agent.runtime.context_compression import (
    DEFAULT_PROTECTED_KEYS,
    compress_payload,
)
from so_agent.runtime.permissions import PermissionGateway
from so_agent.runtime.registry import AgentRegistry
from so_agent.runtime.scheduler import SubtaskDAGScheduler
from so_agent.runtime.supervisor_router import CommunicationError, SupervisorRouter
from so_agent.runtime.tool_registry import (
    ToolRegistry,
    ToolTimeoutProfile,
    timeout_profile_for_tool,
)
from so_agent.result_store import ResultStore
from so_agent.state_machine import (
    PipelineStoppedError,
    TaskStatus,
    WorkflowState,
)
from so_agent.workflow_human import HumanEscalationHandler
from so_agent.workflow_streaming import run_agent_streamed

# 保留 Runner 绑定：测试通过 monkeypatch 替换
# ``so_agent.workflow_support.Runner``，流式实现运行时动态读取本属性。


class PipelineGate:
    """Pipeline 级停止门、绝对 deadline 与进展看门狗（原子门控）。

    超时 / 取消 / 人工终止后原子进入 stopping；所有 dispatch、retry、
    replacement、replan、计划采纳、授权签发入口先经 :meth:`check`，
    停止后拒绝一切新动作。方法均为同步（无 await 点），事件循环内
    天然原子；``generation`` 在每次停止请求时递增，供迟到结果隔离
    （调用前后 generation 变化即视为迟到结果）。

    进展语义（任务 80）：协调器在真实工作推进时调用
    :meth:`record_progress`（子任务完成、阶段推进、检查点落盘），
    主管与嵌套 Agent 的 SDK 流事件也经 ``activity_hook`` 调用它。
    ``pipeline_no_progress_timeout`` 以“距上次活动的秒数”判定停滞，
    而不是限制总时长——正常长任务只要持续产出事件或推进阶段就不会
    被误杀；``max_pipeline_timeout`` 仅在部署方显式开启时作为绝对
    兜底（默认关闭）。

    同一引擎实例复用（连续 ``run``）时在任务开始处调用 :meth:`reset`。
    """

    def __init__(self) -> None:
        """初始化：未停止、无 deadline、无进展时间戳。"""
        self._deadline: float | None = None
        self._stopping: bool = False
        self._stop_reason: str = ""
        self._generation: int = 0
        self._last_progress_at: float | None = None
        self._progress_label: str = ""
        self._progress_count: int = 0

    @property
    def stopping(self) -> bool:
        """是否已请求停止（原子门控位）。"""
        return self._stopping

    @property
    def stop_reason(self) -> str:
        """首次停止请求的原因（幂等保留，不随后续请求覆盖）。"""
        return self._stop_reason

    @property
    def generation(self) -> int:
        """门控代数：每次停止请求 / 重置递增（迟到结果检测依据）。"""
        return self._generation

    @property
    def progress_count(self) -> int:
        """本 pipeline 已记录的进展次数（只增不减，供审计）。"""
        return self._progress_count

    def set_deadline(self, deadline: float | None) -> None:
        """设置（或清除）本 pipeline 的绝对 deadline（monotonic 时刻）。"""
        self._deadline = deadline

    def record_progress(self, label: str = "progress") -> None:
        """记录一次真实工作推进（重置无进展窗口）。"""
        self._last_progress_at = time.monotonic()
        self._progress_label = (label or "progress").strip() or "progress"
        self._progress_count += 1

    def seconds_since_progress(self) -> float:
        """距上次进展的秒数；尚未记录进展时按 0 处理（任务刚启动）。"""
        if self._last_progress_at is None:
            return 0.0
        return max(0.0, time.monotonic() - self._last_progress_at)

    def progress_snapshot(self) -> dict[str, Any]:
        """返回进展快照（供超时诊断与审计）。"""
        return {
            "count": self._progress_count,
            "label": self._progress_label,
            "seconds_since_progress": round(self.seconds_since_progress(), 3),
        }

    def remaining(self) -> float | None:
        """返回剩余预算（秒）；永不为负，未设 deadline 时返回 None。

        剩余时间计算的统一入口：任何调用方都不得自行计算
        ``deadline - now``（可能为负），一律使用本方法。
        """
        if self._deadline is None:
            return None
        return max(0.0, self._deadline - time.monotonic())

    def request_stop(self, reason: str) -> None:
        """原子请求停止（幂等）：首次请求记录原因并递增 generation。"""
        if self._stopping:
            return
        self._stopping = True
        self._stop_reason = (reason or "stopped").strip() or "stopped"
        self._generation += 1

    def reset(self) -> None:
        """重置停止状态（同一引擎实例开始新任务时复用；generation 保留递增）。"""
        self._stopping = False
        self._stop_reason = ""
        self._deadline = None
        self._last_progress_at = None
        self._progress_label = ""
        self._progress_count = 0
        self._generation += 1

    def check(self, action: str = "action") -> None:
        """停止检查点：已请求停止时抛出可穿透宽捕获的停止信号。

        Raises:
            PipelineStoppedError: 已请求停止时（继承 BaseException，
                穿透各协调器的 ``except Exception`` 宽捕获）。
        """
        if self._stopping:
            raise PipelineStoppedError(
                f"pipeline 已停止（{self._stop_reason}），拒绝新动作：{action}"
            )


def prune_empty(value: Any) -> Any:
    """递归剔除空字段（None / 空串 / 空容器），保留 False 与 0 等语义值。

    载荷瘦身只做无损裁剪：模型侧不区分"字段缺失"与"字段为空"，剔除
    空值可稳定减少固定开销；布尔 False、数字 0、空字符串列表等业务
    语义值不在此列（空串与空容器本身不含信息）。
    """

    def is_empty(item: Any) -> bool:
        return item is None or (
            isinstance(item, (str, list, dict, tuple, set)) and len(item) == 0
        )

    if isinstance(value, dict):
        pruned: dict[str, Any] = {}
        for key, raw in value.items():
            cleaned = prune_empty(raw)
            if not is_empty(cleaned):
                pruned[key] = cleaned
        return pruned
    if isinstance(value, list):
        items: list[Any] = []
        for raw in value:
            cleaned = prune_empty(raw)
            if not is_empty(cleaned):
                items.append(cleaned)
        return items
    return value


def estimate_tokens(text: str) -> int:
    """粗略估算文本 token 数（CJK 按 1 字/token，其余按 3 字符/token）。"""
    if not text:
        return 0
    cjk = sum(1 for char in text if "\u4e00" <= char <= "\u9fff")
    return int(cjk + (len(text) - cjk) / 3)


def make_tool_timeout_resolver(
    settings: Settings,
    tool_registry: ToolRegistry | None = None,
) -> Callable[[str | None], float]:
    """构造按超时 profile 解析工具阶段等待窗口的 resolver（任务 45）。

    workflow_support 在工具阶段按实际 ``tool_name`` 经本 resolver 取
    窗口：code_agent → code_agent_tool_execution_timeout（默认 0，不设
    父级硬截止）、subagent_creator / execute_subagent →
    subagent_tool_execution_timeout（默认 0）、run_in_sandbox / 生成
    工具 → max(tool_execution, sandbox_timeout)（任一为 0 则关闭）、其余 →
    tool_execution_timeout。返回 <=0 表示该工具阶段不设父级硬截止；
    显式开启总预算时仍会被剩余总预算截断（profile 绝不延长总预算）。

    Args:
        settings: 生效配置（窗口来源）。
        tool_registry: 工具注册表；提供时经
            ``ToolRegistry.get_timeout_profile`` 判定（额外识别生成
            工具 → SANDBOX），否则按模块默认映射判定。
    """

    def resolve(tool_name: str | None) -> float:
        if tool_registry is not None:
            profile = tool_registry.get_timeout_profile(tool_name)
        else:
            profile = timeout_profile_for_tool(tool_name)
        if profile is ToolTimeoutProfile.CODE_AGENT:
            return float(settings.code_agent_tool_execution_timeout)
        if profile is ToolTimeoutProfile.SUBAGENT:
            return float(settings.subagent_tool_execution_timeout)
        if profile is ToolTimeoutProfile.SANDBOX:
            tool_window = float(settings.tool_execution_timeout)
            sandbox_window = float(settings.sandbox_timeout)
            # 任一窗口为 0 均表示关闭该限制：max() 会把 0 当成有限值，
            # 从而错误地回落到另一个窗口，必须显式短路为不限。
            if tool_window <= 0 or sandbox_window <= 0:
                return 0.0
            return max(tool_window, sandbox_window)
        return float(settings.tool_execution_timeout)

    return resolve



@dataclass
class WorkflowDeps:
    """协调器共享依赖容器（基础设施 + 编排器注入的调用通道）。

    Attributes:
        context: 项目共享上下文（沙箱、注册表镜像、事件日志、配置来源）。
        settings: 生效配置。
        events: 事件记录器。
        registry: Agent 注册表。
        gateway: 工具授权网关。
        router: 主管路由器（通信边界校验与留痕）。
        tool_registry: 工具注册表（MCP 注册中心）。
        escalation_handler: 人工兜底处理器。
        orchestrator: 主管 Agent 实例（测试可注入假 Agent）。
        review_agent: 评审 Agent 实例（测试可注入假 Agent）。
        mcp_client: MCP 客户端；None 表示按需从注册表已挂载的 Server 构建
            （由执行协调器的 ``_ensure_mcp_client`` 惰性填充）。
        result_store: 执行结果/计划版本存储（默认内存实现）；规划协调器
            采纳计划时把带指纹索引的 PlanVersion 快照写入该存储。
        scheduler: 任务调度器；编排器在构造执行协调器后填充。
        gate: Pipeline 停止门（任务 45）；编排器构造后注入，各协调器
            经基类 ``_gate_check`` 在新动作入口检查。None 时检查为空操作
            （兼容直接构造协调器的既有测试）。
        run_supervisor: 主管 Agent 调用通道（绑定 WorkflowEngine._run_supervisor）。
        call_review_agent: 评审 Agent 调用通道（绑定 WorkflowEngine._call_review_agent）。
        record_model_usage: 模型用量记录通道（绑定 WorkflowEngine 同名方法）。
        plan_and_review: 规划协调器的拆解-评审循环（失败升级重规划入口）。
        issue_grants: 规划协调器的授权签发（工具缺口解决后补发入口）。
        consume_replan_quota: 规划协调器的重规划额度消耗判定。
    """

    context: ProjectContext
    settings: Settings
    events: EventLogger
    registry: AgentRegistry
    gateway: PermissionGateway
    router: SupervisorRouter
    tool_registry: ToolRegistry
    escalation_handler: HumanEscalationHandler
    orchestrator: Agent[Any]
    review_agent: Agent[Any]
    mcp_client: MCPClient | None = None
    result_store: ResultStore | None = None
    scheduler: SubtaskDAGScheduler | None = None
    gate: PipelineGate | None = None
    run_supervisor: Callable[..., Awaitable[Any]] | None = None
    call_review_agent: Callable[[Any], Awaitable[ReviewDecision]] | None = None
    record_model_usage: Callable[[Any, str], None] | None = None
    enforce_token_budget: Callable[[], None] | None = None
    plan_and_review: Callable[..., Awaitable[bool]] | None = None
    issue_grants: Callable[[], list[ToolGrant]] | None = None
    consume_replan_quota: Callable[[], bool] | None = None
    persist_checkpoint: Callable[[str], None] | None = None


class BaseCoordinator:
    """协调器基类：注入共享状态与依赖，集中承载公共辅助方法。

    历史私有状态属性（``_task_request`` / ``_subtasks`` / ``_results``
    等）以 property 形式读写代理到共享 :class:`WorkflowState`；主管 /
    评审调用与跨协调器调用经 :class:`WorkflowDeps` 通道。因此从原
    WorkflowEngine 平移的方法体保持逐字不变。
    """

    def __init__(self, state: WorkflowState, deps: WorkflowDeps) -> None:
        """初始化协调器。

        Args:
            state: 共享任务状态（与薄编排器及其他协调器同一实例）。
            deps: 共享依赖容器（基础设施对象与编排器注入的调用通道）。
        """
        self._state = state
        self._deps = deps

    # ------------------------------------------------------------------
    # 依赖代理（与原 WorkflowEngine 的属性同名同义）
    # ------------------------------------------------------------------
    @property
    def _context(self) -> ProjectContext:
        return self._deps.context

    @property
    def _settings(self) -> Settings:
        return self._deps.settings

    @property
    def _events(self) -> EventLogger:
        return self._deps.events

    @property
    def _registry(self) -> AgentRegistry:
        return self._deps.registry

    @property
    def _gateway(self) -> PermissionGateway:
        return self._deps.gateway

    @property
    def _router(self) -> SupervisorRouter:
        return self._deps.router

    @property
    def _scheduler(self) -> SubtaskDAGScheduler | None:
        return self._deps.scheduler

    @property
    def _gate(self) -> PipelineGate | None:
        """Pipeline 停止门（未装配时为 None，门控检查为空操作）。"""
        return self._deps.gate

    def _gate_check(self, action: str) -> None:
        """pipeline 停止门检查点：已请求停止时拒绝新动作（穿透宽捕获）。

        供 dispatch / retry / replacement / replan / 计划采纳 / 授权
        签发等新动作入口在执行前调用。

        Raises:
            PipelineStoppedError: 停止门已请求停止时（继承
                BaseException，穿透 ``except Exception`` 宽捕获）。
        """
        gate = self._deps.gate
        if gate is not None:
            gate.check(action)

    def _enforce_token_budget(self) -> None:
        """结算 token 预算（由工作流引擎注入；独立构造协调器时为空操作）。"""
        checker = self._deps.enforce_token_budget
        if callable(checker):
            checker()

    def _report_progress(self, label: str, *, checkpoint: bool = False) -> None:
        """上报一次真实工作推进，必要时落检查点。

        进展是“无进展看门狗”的唯一续时依据：正常长任务只要持续完成
        子任务 / 推进阶段就不会被固定总时长误杀。``checkpoint=True``
        时同时落盘，保证超时或取消后可以从最近完成点续跑。
        """
        gate = self._deps.gate
        if gate is not None:
            gate.record_progress(label)
        if not checkpoint:
            return
        persist = self._deps.persist_checkpoint
        if callable(persist):
            persist(f"progress:{label}")

    def _compress_context(self, payload: Any) -> Any:
        """按配置对 LLM 载荷做确定性压缩；发生压缩时记录事件。"""
        settings = self._settings
        if not getattr(settings, "context_compression_enabled", True):
            return payload
        compressed, report = compress_payload(
            payload,
            max_total_chars=int(
                getattr(settings, "context_compression_max_chars", 0) or 0
            ),
            per_string_chars=int(
                getattr(settings, "context_compression_result_chars", 0) or 0
            ),
            protected_keys=(
                DEFAULT_PROTECTED_KEYS
                if getattr(
                    settings, "context_compression_protect_critical", True
                )
                else None
            ),
        )
        if report.changed:
            self._events.log_event(
                EVENT_CONTEXT_COMPRESSED,
                {
                    "task_id": (
                        self._task_request.task_id
                        if self._task_request is not None
                        else None
                    ),
                    "original_chars": report.original_chars,
                    "compressed_chars": report.compressed_chars,
                    "fields": report.paths[:20],
                },
            )
        return compressed

    @property
    def _tool_registry(self) -> ToolRegistry:
        return self._deps.tool_registry

    @property
    def _escalation_handler(self) -> HumanEscalationHandler:
        return self._deps.escalation_handler

    @property
    def _mcp_client(self) -> MCPClient | None:
        return self._deps.mcp_client

    @_mcp_client.setter
    def _mcp_client(self, value: MCPClient | None) -> None:
        self._deps.mcp_client = value

    @property
    def _result_store(self) -> ResultStore:
        """结果存储（计划版本快照与执行记录持久化，供跨版本结果复用）。"""
        return self._deps.result_store  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # 编排器注入的调用通道（主管/评审 + 跨协调器）
    # ------------------------------------------------------------------
    @property
    def _run_supervisor(self) -> Callable[..., Awaitable[Any]]:
        return self._deps.run_supervisor  # type: ignore[return-value]

    @property
    def _call_review_agent(self) -> Callable[[Any], Awaitable[ReviewDecision]]:
        return self._deps.call_review_agent  # type: ignore[return-value]

    @property
    def _record_model_usage(self) -> Callable[[Any, str], None]:
        return self._deps.record_model_usage  # type: ignore[return-value]

    @property
    def _plan_and_review(self) -> Callable[..., Awaitable[bool]]:
        """拆解-评审循环：PlanningCoordinator 覆写为真实方法（跨协调器经 deps）。"""
        return self._deps.plan_and_review  # type: ignore[return-value]

    @property
    def _issue_grants(self) -> Callable[[], list[ToolGrant]]:
        """授权签发：PlanningCoordinator 覆写为真实方法（跨协调器经 deps）。"""
        return self._deps.issue_grants  # type: ignore[return-value]

    @property
    def _try_consume_replan_quota(self) -> Callable[[], bool]:
        """重规划额度判定：PlanningCoordinator 覆写为真实方法。"""
        return self._deps.consume_replan_quota  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # 共享状态代理（历史私有属性名 → WorkflowState 字段）
    # ------------------------------------------------------------------
    @property
    def _current_status(self) -> TaskStatus:
        return self._state.current_status

    @_current_status.setter
    def _current_status(self, value: TaskStatus) -> None:
        self._state.current_status = value

    @property
    def _plan_version(self) -> int:
        return self._state.plan_version

    @_plan_version.setter
    def _plan_version(self, value: int) -> None:
        self._state.plan_version = value

    @property
    def _replan_count(self) -> int:
        return self._state.replan_count

    @_replan_count.setter
    def _replan_count(self, value: int) -> None:
        self._state.replan_count = value

    @property
    def _result_rework_count(self) -> int:
        return self._state.result_rework_count

    @_result_rework_count.setter
    def _result_rework_count(self, value: int) -> None:
        self._state.result_rework_count = value

    @property
    def _task_request(self):
        return self._state.task_request

    @_task_request.setter
    def _task_request(self, value) -> None:
        self._state.task_request = value

    @property
    def _subtasks(self) -> list:
        return self._state.subtasks

    @_subtasks.setter
    def _subtasks(self, value: list) -> None:
        self._state.subtasks = value

    @property
    def _pending_specs(self) -> list:
        return self._state.pending_specs

    @_pending_specs.setter
    def _pending_specs(self, value: list) -> None:
        self._state.pending_specs = value

    @property
    def _results(self) -> dict:
        return self._state.results

    @_results.setter
    def _results(self, value: dict) -> None:
        self._state.results = value

    @property
    def _plan_history(self) -> list:
        return self._state.plan_history

    @_plan_history.setter
    def _plan_history(self, value: list) -> None:
        self._state.plan_history = value

    @property
    def _agent_failure_history(self) -> list:
        return self._state.agent_failure_history

    @_agent_failure_history.setter
    def _agent_failure_history(self, value: list) -> None:
        self._state.agent_failure_history = value

    @property
    def _grants(self) -> list:
        return self._state.grants

    @_grants.setter
    def _grants(self, value: list) -> None:
        self._state.grants = value

    @property
    def _plan_grants(self) -> list:
        return self._state.plan_grants

    @_plan_grants.setter
    def _plan_grants(self, value: list) -> None:
        self._state.plan_grants = value

    @property
    def _plan_grant_map(self) -> dict:
        return self._state.plan_grant_map

    @_plan_grant_map.setter
    def _plan_grant_map(self, value: dict) -> None:
        self._state.plan_grant_map = value

    @property
    def _plan_grant_proposals(self) -> list:
        return self._state.plan_grant_proposals

    @_plan_grant_proposals.setter
    def _plan_grant_proposals(self, value: list) -> None:
        self._state.plan_grant_proposals = value

    @property
    def _last_review(self):
        return self._state.last_review

    @_last_review.setter
    def _last_review(self, value) -> None:
        self._state.last_review = value

    @property
    def _human_escalation(self):
        return self._state.human_escalation

    @_human_escalation.setter
    def _human_escalation(self, value) -> None:
        self._state.human_escalation = value

    @property
    def _cancel_requested(self) -> bool:
        return self._state.cancel_requested

    @_cancel_requested.setter
    def _cancel_requested(self, value: bool) -> None:
        self._state.cancel_requested = value

    @property
    def _seen_replacement_requests(self) -> set:
        return self._state.seen_replacement_requests

    @_seen_replacement_requests.setter
    def _seen_replacement_requests(self, value: set) -> None:
        self._state.seen_replacement_requests = value

    @property
    def _seen_issue_fingerprints(self) -> set:
        return self._state.seen_issue_fingerprints

    @_seen_issue_fingerprints.setter
    def _seen_issue_fingerprints(self, value: set) -> None:
        self._state.seen_issue_fingerprints = value

    @property
    def _archive_results(self) -> dict:
        """跨计划版本结果归档（``{(plan_version, subtask_id): ExecutionResult}``）。"""
        return self._state.archive_results

    @_archive_results.setter
    def _archive_results(self, value: dict) -> None:
        self._state.archive_results = value

    @property
    def _parallelism_decision(self):
        """规划阶段的确定性并行度决策（未计算时为 None）。"""
        return self._state.parallelism_decision

    @_parallelism_decision.setter
    def _parallelism_decision(self, value) -> None:
        self._state.parallelism_decision = value

    @property
    def _runtime_breaker(self):
        """调度期运行时熔断器（G4）；每个计划版本独立实例。"""
        return self._state.runtime_breaker

    @_runtime_breaker.setter
    def _runtime_breaker(self, value) -> None:
        self._state.runtime_breaker = value

    # ------------------------------------------------------------------
    # 公共辅助方法（自原 WorkflowEngine 平移）
    # ------------------------------------------------------------------
    def _transition(
        self, new_status: TaskStatus, *, reason: str = "", **extra: Any
    ) -> None:
        """状态迁移：委托共享 WorkflowState（含影子状态机推进与事件留痕）。"""
        self._state.transition(new_status, reason=reason, **extra)

    def _check_cancel(self) -> None:
        """协作式取消检查点：委托共享 WorkflowState。

        Raises:
            WorkflowCancelledError: 已通过 ``cancel()`` 请求取消时。
        """
        self._state.check_cancel()

    def _find_latest_agent(
        self, subtask_id: str, *, replacement: bool | None = None
    ) -> AgentRecord | None:
        """在注册表中查找某子任务最近的 Agent 记录。

        Args:
            subtask_id: 子任务编号。
            replacement: True 仅查替代 Agent；False 仅查原始 Agent；
                None 不限（按创建时间取最新）。
        """
        if self._state.task_request is None:
            return None
        latest: AgentRecord | None = None
        for record in self._registry.list_by_task(self._state.task_request.task_id):
            if self._subtask_id_of(record) != subtask_id:
                continue
            is_replacement = record.replacement_of is not None
            if replacement is not None and is_replacement != replacement:
                continue
            latest = record
        return latest

    def _subtask_id_of(self, record: AgentRecord) -> str:
        """从 agent_id 解析所属子任务编号（委托共享状态）。"""
        return self._state.subtask_id_of(record)

    def _parse_agent_ref(self, agent_id: str) -> str:
        """从 Agent 引用解析所属子任务编号（委托共享状态）。"""
        return self._state.parse_agent_ref(agent_id)

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

    def _revoke_active_grants(
        self, *, reason: str = "lifecycle_cleanup"
    ) -> list[str]:
        """统一幂等清理：撤销本任务当前全部活跃授权（生命周期收口）。

        供所有退出路径调用（正常完成 / 结果评审失败转人工 / 人工升级 /
        pipeline 超时 / 取消 / 未捕获异常 / 重规划前）：按任务维度撤销
        未撤销授权，原因写入 ``tool_revoked`` 审计事件（如
        ``lifecycle_cleanup:pipeline_timeout``）。

        幂等性：重复调用不产生重复事件、不抛错；任务尚未接收
        （task_request 为空）或无活跃授权时为空操作。本方法常在
        finally 路径执行，内部绝不向上抛错（避免掩盖原始异常）。

        Returns:
            本次被撤销的 grant_id 列表（此前已撤销的不计入）。
        """
        task_request = self._state.task_request
        if task_request is None:
            return []
        try:
            return self._gateway.revoke_grants_for_task(
                task_request.task_id, reason=reason
            )
        except Exception:  # 清理失败不得掩盖原始异常/返回值
            return []


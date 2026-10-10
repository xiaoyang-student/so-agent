"""协调器公共支撑：依赖容器、协调器基类与人工兜底处理器。

薄编排器（WorkflowEngine）与三个协调器（Planning / Execution /
Aggregation）之间的解耦层：

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
- ``HumanEscalationHandler``：人工兜底处理器接口（首版仅记录请求）；
- ``run_agent_streamed``：流式运行辅助（分阶段空闲超时 + 绝对总
  预算，任务 33），主管与子 Agent 调用共用；工具阶段按工具项内容
  （``tool_name`` / ``call_id``）区分普通工具与嵌套子 Agent 执行
  工具，并取所有 pending 工具窗口的安全上限（嵌套窗口默认派生 =
  agent_total + 10 秒，修复父级工具窗口误杀嵌套子 Agent）；
  ``runner`` 可注入以保持调用方模块级 ``Runner`` 的 monkeypatch
  语义；超时前可写 ``timeout_audit`` 审计事件。
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, NoReturn

from agents import Agent, Runner
from agents.stream_events import (
    AgentUpdatedStreamEvent,
    RawResponsesStreamEvent,
    RunItemStreamEvent,
)

from so_agent.config import Settings
from so_agent.context import ProjectContext
from so_agent.mcp.client import MCPClient
from so_agent.models import (
    AgentRecord,
    HumanEscalationRequest,
    ReviewDecision,
    ToolGrant,
)
from so_agent.runtime.events import EVENT_TIMEOUT_AUDIT, EventLogger
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


# SDK 原生流事件中标识工具阶段的语义事件名（openai-agents 0.22.3
# RunItemStreamEvent.name 的 Literal 枚举子集，经 SDK 源码核查）：
# - tool_called / tool_search_called：工具调用项在工具真正执行前推
#   送（先于 execute_tools_and_side_effects）；
# - tool_output / tool_search_output_created：工具执行完成后推送。
_TOOL_START_EVENT_NAMES: frozenset[str] = frozenset(
    {"tool_called", "tool_search_called"}
)
_TOOL_END_EVENT_NAMES: frozenset[str] = frozenset(
    {"tool_output", "tool_search_output_created"}
)

# 分阶段流式超时默认值（与 Settings 同名字段默认一致；显式传参优先）
_DEFAULT_FIRST_EVENT_TIMEOUT: float = 180.0
_DEFAULT_MODEL_IDLE_TIMEOUT: float = 90.0
_DEFAULT_TOOL_EXECUTION_TIMEOUT: float = 180.0
_DEFAULT_AGENT_TOTAL_TIMEOUT: float = 600.0

# 嵌套子 Agent 执行工具的识别集合（任务 33 嵌套修复）：判定依据为
# 真实 RunItemStreamEvent（tool_called / tool_output）中工具项的
# ``tool_name`` 公共属性（SDK ToolCallItem.tool_name / call_id），
# 不解析任何文本输出：
# - "subagent_creator"：主管工具列表中嵌套子 Agent 创建/执行入口的
#   实际 FunctionTool 名称（orchestrator.SUPERVISOR_TOOL_NAMES 经
#   mcp.adapter.mcp_server_to_agent_tool 包装，name=工具名）；
# - "execute_subagent"：创建器内部的子 Agent 执行工具名（as_tool /
#   直接挂载等形态可能直接出现在调用方流中），防御性识别。
_SUBAGENT_TOOL_NAMES: frozenset[str] = frozenset(
    {"subagent_creator", "execute_subagent"}
)

# 工具类别（写入超时审计的 tool_phase_category 字段）
_SUBAGENT_TOOL_CATEGORY: str = "subagent"
_STANDARD_TOOL_CATEGORY: str = "standard"

# 嵌套子 Agent 工具窗口的清理宽限（秒）：未显式传参时派生
# subagent_tool_execution_timeout = agent_total_timeout + 本宽限
# （默认 600 + 10 = 610），覆盖子 Agent 总预算与取消 / 审计收尾。
_SUBAGENT_TOOL_CLEANUP_GRACE: float = 10.0

# 流式迭代器关闭宽限（秒，任务 45）：超时/取消收尾时等待底层
# iterator.aclose() 完成的上限；超过后不再等待，迟到结果由调用方的
# 停止门（PipelineGate）隔离，不再写状态/触发重试/签发授权。
_DEFAULT_STREAM_CLEANUP_GRACE: float = 2.0


async def _aclose_quietly(iterator: Any, grace: float) -> None:
    """尽力关闭流式迭代器（有限宽限），吞并关闭期异常但不吞取消。

    任务 45：``run_agent_streamed`` 在超时 / 取消收尾时调用。SDK 的
    async generator 自带 ``aclose``；关闭在宽限内未完成（底层 SDK
    未及时取消）时放弃等待，由调用方的停止门隔离迟到结果。
    只吞并 ``Exception``（关闭期业务异常），``CancelledError``
    继续向上传播（asyncio 取消语义）。
    """
    aclose = getattr(iterator, "aclose", None)
    if not callable(aclose):
        return
    try:
        await asyncio.wait_for(aclose(), timeout=max(0.0, float(grace)))
    except asyncio.CancelledError:
        raise
    except Exception:
        return


class PipelineGate:
    """Pipeline 级停止门与绝对 deadline（原子门控，任务 45）。

    超时 / 取消 / 人工终止后原子进入 stopping；所有 dispatch、retry、
    replacement、replan、计划采纳、授权签发入口先经 :meth:`check`，
    停止后拒绝一切新动作。方法均为同步（无 await 点），事件循环内
    天然原子；``generation`` 在每次停止请求时递增，供迟到结果隔离
    （调用前后 generation 变化即视为迟到结果）。

    同一引擎实例复用（连续 ``run``）时在任务开始处调用 :meth:`reset`。
    """

    def __init__(self) -> None:
        """初始化：未停止、无 deadline。"""
        self._deadline: float | None = None
        self._stopping: bool = False
        self._stop_reason: str = ""
        self._generation: int = 0

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

    def set_deadline(self, deadline: float | None) -> None:
        """设置（或清除）本 pipeline 的绝对 deadline（monotonic 时刻）。"""
        self._deadline = deadline

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


def make_tool_timeout_resolver(
    settings: Settings,
    tool_registry: ToolRegistry | None = None,
) -> Callable[[str | None], float]:
    """构造按超时 profile 解析工具阶段等待窗口的 resolver（任务 45）。

    workflow_support 在工具阶段按实际 ``tool_name`` 经本 resolver 取
    窗口：code_agent → code_agent_tool_execution_timeout（默认派生
    610s）、subagent_creator / execute_subagent →
    subagent_tool_execution_timeout、run_in_sandbox / 生成工具 →
    max(tool_execution, sandbox_timeout)、其余 → tool_execution_
    timeout。返回窗口只影响工具阶段等待，运行时仍会被 agent_total
    与 pipeline deadline 截断（总预算绝不被 profile 延长）。

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
            return max(
                float(settings.tool_execution_timeout),
                float(settings.sandbox_timeout),
            )
        return float(settings.tool_execution_timeout)

    return resolve


@dataclass(frozen=True)
class _PendingToolCall:
    """一次尚未收到输出的工具调用（工具阶段窗口选择的配对单元）。

    Attributes:
        call_id: SDK 工具项的 call_id（与 tool_output 精确配对；缺失时
            为 None）。
        name: 由 ``ToolCallItem.tool_name`` 提取的工具名（可能为 None）。
        category: "subagent"（嵌套子 Agent 执行工具）或 "standard"。
        window: 该工具所需的活动窗口（秒）。
    """

    call_id: str | None
    name: str | None
    category: str
    window: float


def _extract_tool_name(item: Any) -> str | None:
    """从工具调用项提取工具名（优先 SDK ``ToolCallItem.tool_name``）。"""
    candidate = getattr(item, "tool_name", None)
    if isinstance(candidate, str) and candidate:
        return candidate
    raw = getattr(item, "raw_item", None)
    if isinstance(raw, dict):
        candidate = raw.get("name")
    else:
        candidate = getattr(raw, "name", None)
    return candidate if isinstance(candidate, str) and candidate else None


def _extract_call_id(item: Any) -> str | None:
    """从工具项提取 call_id（ToolCallItem / ToolCallOutputItem 公共属性）。"""
    candidate = getattr(item, "call_id", None)
    if candidate is None:
        raw = getattr(item, "raw_item", None)
        if isinstance(raw, dict):
            candidate = raw.get("call_id") or raw.get("id")
        else:
            candidate = getattr(raw, "call_id", None) or getattr(raw, "id", None)
    return str(candidate) if candidate is not None else None


async def run_agent_streamed(
    agent: Agent[Any],
    input: Any,
    max_turns: int,
    idle_timeout: float | None = None,
    total_timeout: float | None = None,
    *,
    first_event_timeout: float | None = None,
    model_idle_timeout: float | None = None,
    tool_execution_timeout: float | None = None,
    subagent_tool_execution_timeout: float | None = None,
    agent_total_timeout: float | None = None,
    runner: Any = None,
    events: EventLogger | None = None,
    audit_context: dict[str, Any] | None = None,
    tool_timeout_resolver: Callable[[str | None], float] | None = None,
    cleanup_grace_timeout: float | None = None,
) -> Any:
    """以流式方式运行 Agent：分阶段空闲超时 + 绝对总预算（任务 33）。

    替代 ``asyncio.wait_for(Runner.run(...), timeout=固定值)`` 的固定
    超时保护，按 SDK 原生流事件的真实语义分阶段计时：

    - ``first_event`` 阶段（首个流事件到达前，覆盖 SDK 冷启动、MCP
      初始化与模型首包等待）使用 ``first_event_timeout``；
    - ``model`` 阶段（模型产出流事件期间）使用 ``model_idle_timeout``：
      连续该秒数无任何 SDK 原生流事件才判定卡死；不接受 LLM 自行
      输出的文本心跳作为续时依据（只有 SDK 原生流事件能重置计时）；
    - ``tool_execution`` 阶段（语义 ``tool_called`` 事件到达后、对应
      ``tool_output`` 事件到达前）使用所有 pending 工具所需窗口的安全
      上限：普通工具为 ``tool_execution_timeout``，嵌套子 Agent 执行
      工具（按工具项 ``tool_name`` 识别 ``subagent_creator`` /
      ``execute_subagent``）为 ``subagent_tool_execution_timeout``
      （默认派生 = agent_total_timeout + 10 秒，覆盖子 Agent 总预算与
      清理宽限，避免父级工具窗口提前误杀复杂子任务）；按 ``call_id``
      精确配对了结，无法配对时保守移除窗口最小的 pending 项（宁可保留
      大窗口，不提前收缩）；全部 ``tool_output`` 到达后恢复模型窗口；
    - 所有阶段窗口均被 ``agent_total_timeout`` 的绝对截止时刻截断：
      每次等待取“阶段窗口”与“剩余总预算”的较小值，即使事件持续
      产出也不会超过总预算；总预算检查点位于每次取事件之前。
    未知（非 SDK 类型或未识别名称）事件不切换阶段、不放大窗口，
    仅作为活动信号重置当前阶段的空闲计时，并记入超时审计的调试
    上下文。

    超时收尾统一为：先 ``streamed.cancel()``，再抛携带 phase 信息的
    ``asyncio.TimeoutError``（消息含 ``<phase> idle timeout`` 或
    ``total timeout`` 前缀），调用方既有的
    ``except asyncio.TimeoutError`` 降级语义不变。

    可选的 ``events`` / ``audit_context`` 用于超时审计：任一超时抛出
    前写入 ``timeout_audit`` 事件（phase、配置窗口、实际等待窗口、
    elapsed、总预算、嵌套子 Agent 窗口、事件计数、最近事件种类、
    pending 工具名与采用的工具类别、未知事件名单与调用方上下文）；
    审计只记录工具名与类别等元数据，不记录工具参数等敏感输入；
    审计写入失败不影响超时本身。

    向后兼容（deprecated 参数，任务 33）：
    - ``idle_timeout`` 显式给定时映射到 ``first_event_timeout`` 与
      ``model_idle_timeout``（旧单一空闲窗口语义；工具执行阶段保持
      独立的新窗口，修复旧的固定空闲超时误杀工具执行的问题）；
    - ``total_timeout`` 显式给定时映射到 ``agent_total_timeout``。

    Args:
        agent: 待运行的 Agent 实例。
        input: 传给 Agent 的输入（JSON 文本）。
        max_turns: 最大模型回合数。
        idle_timeout: （deprecated）旧空闲超时；显式给出时映射到
            first_event / model 窗口。
        total_timeout: （deprecated）旧兜底总超时；显式给出时映射到
            agent_total_timeout。
        first_event_timeout: 首事件等待窗口（秒）；None 取默认 180。
        model_idle_timeout: 模型阶段空闲窗口（秒）；None 取默认 90。
        tool_execution_timeout: 工具执行窗口（秒）；None 取默认 180。
        subagent_tool_execution_timeout: 嵌套子 Agent 执行工具的工具
            阶段窗口（秒）；None 时派生为 agent_total_timeout + 10 秒
            （默认 610，覆盖子 Agent 总预算与取消/审计收尾宽限）；
            普通工具仍使用 tool_execution_timeout。
        agent_total_timeout: 绝对总预算（秒）；None 取默认 600。
        runner: Runner 类（run_streamed / run 的持有者）；None 时使用
            ``agents.Runner``。
        events: 可选事件记录器；提供时在超时抛出前写入 timeout_audit。
        audit_context: 可选审计上下文（如 agent / task_id /
            subtask_id / pipeline_phase）；原样并入超时审计 payload，
            禁止放入敏感输入。
        tool_timeout_resolver: 可选的工具阶段窗口解析器（任务 45）：
            提供时按实际 ``tool_name`` 经超时 profile（code_agent /
            subagent / sandbox / standard）解析工具阶段等待窗口，
            不再把长时代码 Agent 固定按普通工具窗口处理；未提供时
            保持二分类回退（嵌套子 Agent 窗口 / 普通工具窗口）。
            返回窗口仍被 agent_total 绝对总预算截断，绝不延长总预算。
        cleanup_grace_timeout: 超时/取消收尾时等待底层流式迭代器
            ``aclose()`` 完成的宽限（秒）；None 取默认 2 秒。宽限耗尽
            后不再等待（迟到结果由调用方停止门隔离）。

    Returns:
        运行完成的 RunResultStreaming（含 final_output / raw_responses /
        new_items / last_agent 等属性，与 RunResult 兼容）。

    Raises:
        asyncio.TimeoutError: 阶段空闲超时或总预算超时；抛出前已调用
            ``streamed.cancel()`` 并在有限宽限内尽力关闭流式迭代器；
            消息携带 phase 与耗时上下文。
    """
    # 1) 五层预算解析：新参数 > 旧参数（deprecated 映射）> 默认值
    resolved_total = (
        float(agent_total_timeout)
        if agent_total_timeout is not None
        else float(total_timeout)
        if total_timeout is not None
        else _DEFAULT_AGENT_TOTAL_TIMEOUT
    )
    resolved_first = (
        float(first_event_timeout)
        if first_event_timeout is not None
        else float(idle_timeout)
        if idle_timeout is not None
        else _DEFAULT_FIRST_EVENT_TIMEOUT
    )
    resolved_model = (
        float(model_idle_timeout)
        if model_idle_timeout is not None
        else float(idle_timeout)
        if idle_timeout is not None
        else _DEFAULT_MODEL_IDLE_TIMEOUT
    )
    resolved_tool = (
        float(tool_execution_timeout)
        if tool_execution_timeout is not None
        else _DEFAULT_TOOL_EXECUTION_TIMEOUT
    )
    resolved_subagent_tool = (
        float(subagent_tool_execution_timeout)
        if subagent_tool_execution_timeout is not None
        else resolved_total + _SUBAGENT_TOOL_CLEANUP_GRACE
    )
    resolved_cleanup_grace = (
        float(cleanup_grace_timeout)
        if cleanup_grace_timeout is not None
        else _DEFAULT_STREAM_CLEANUP_GRACE
    )

    active_runner = runner if runner is not None else Runner
    streamed = active_runner.run_streamed(agent, input=input, max_turns=max_turns)
    started = time.monotonic()
    deadline = started + resolved_total
    iterator = streamed.stream_events()

    # 2) 阶段跟踪状态（仅由 SDK 原生流事件与工具阶段事件驱动）
    phase = "first_event"
    pending_tools: list[_PendingToolCall] = []
    events_seen = 0
    last_event_kind = "none"
    unknown_event_kinds: set[str] = set()

    def _tool_phase_window() -> float:
        """工具阶段窗口：取所有 pending 工具所需窗口的安全上限。"""
        if not pending_tools:
            return resolved_tool
        return max(pending.window for pending in pending_tools)

    def _tool_phase_category() -> str | None:
        """当前工具阶段实际采用的类别（按最大窗口对应的 pending 判定）。"""
        if phase != "tool_execution" or not pending_tools:
            return None
        window = _tool_phase_window()
        if any(
            pending.category == _SUBAGENT_TOOL_CATEGORY
            and pending.window == window
            for pending in pending_tools
        ):
            return _SUBAGENT_TOOL_CATEGORY
        return _STANDARD_TOOL_CATEGORY

    def _phase_window() -> float:
        """当前阶段的活动窗口（秒）。"""
        if phase == "first_event":
            return resolved_first
        if phase == "tool_execution":
            return _tool_phase_window()
        return resolved_model

    def _elapsed() -> float:
        """自 run_streamed 起点的累计耗时（秒）。"""
        return time.monotonic() - started

    def _settle_tool_call(call_id: str | None) -> None:
        """工具结束：按 call_id 精确配对了结 pending 项。

        无法精确配对（call_id 缺失或未命中）时，保守移除当前窗口最小
        的 pending 项——宁可把大窗口（嵌套子 Agent）保留到确认完成，
        也不提前收缩窗口造成误杀。
        """
        if call_id is not None:
            for index, pending in enumerate(pending_tools):
                if pending.call_id == call_id:
                    pending_tools.pop(index)
                    return
        if not pending_tools:
            return
        smallest = min(
            range(len(pending_tools)),
            key=lambda index: pending_tools[index].window,
        )
        pending_tools.pop(smallest)

    def _emit_timeout_audit(kind: str, configured: float, window: float) -> None:
        """写入超时审计事件（失败静默，不得掩盖超时本身）。"""
        if events is None:
            return
        payload: dict[str, Any] = dict(audit_context or {})
        payload.update(
            {
                "timeout_kind": kind,  # phase_idle | agent_total
                "phase": phase,
                "configured_timeout": configured,
                "wait_window": round(window, 3),
                "elapsed": round(_elapsed(), 3),
                "agent_total_timeout": resolved_total,
                "subagent_tool_execution_timeout": resolved_subagent_tool,
                "events_seen": events_seen,
                "last_event_kind": last_event_kind,
                "pending_tool_calls": len(pending_tools),
                "pending_tool_names": sorted(
                    {pending.name for pending in pending_tools if pending.name}
                )[:5],
                "tool_phase_category": _tool_phase_category(),
                "unknown_event_kinds": sorted(unknown_event_kinds)[:5],
            }
        )
        try:
            events.log_event(EVENT_TIMEOUT_AUDIT, payload)
        except Exception:  # 审计写入失败不得影响主流程
            return

    def _track(event: Any) -> None:
        """按 SDK 原生事件更新阶段状态（未知事件不切换阶段）。"""
        nonlocal phase, last_event_kind
        if isinstance(event, RawResponsesStreamEvent):
            # 模型原始 chunk：模型阶段活动；模型继续推进意味着上一工具
            # 阶段已结束（SDK 时序下不会交错，保守恢复模型窗口）
            last_event_kind = "model_chunk"
            pending_tools.clear()
            phase = "model"
            return
        if isinstance(event, RunItemStreamEvent):
            name = str(getattr(event, "name", ""))
            last_event_kind = f"item:{name}"
            if name in _TOOL_START_EVENT_NAMES:
                # 工具名与配对键取自 SDK 工具项（ToolCallItem 公共属性
                # tool_name / call_id），不解析任何文本输出；窗口按超时
                # profile（任务 45：code_agent/subagent/sandbox/
                # standard）经 resolver 解析，未提供 resolver 时回退
                # 二分类窗口（嵌套子 Agent / 普通工具）
                item = getattr(event, "item", None)
                tool_name = _extract_tool_name(item)
                is_subagent = tool_name in _SUBAGENT_TOOL_NAMES
                if tool_timeout_resolver is not None:
                    window = max(0.0, float(tool_timeout_resolver(tool_name)))
                elif is_subagent:
                    window = resolved_subagent_tool
                else:
                    window = resolved_tool
                pending_tools.append(
                    _PendingToolCall(
                        call_id=_extract_call_id(item),
                        name=tool_name,
                        category=(
                            _SUBAGENT_TOOL_CATEGORY
                            if is_subagent
                            else _STANDARD_TOOL_CATEGORY
                        ),
                        window=window,
                    )
                )
                phase = "tool_execution"
                return
            if name in _TOOL_END_EVENT_NAMES:
                _settle_tool_call(_extract_call_id(getattr(event, "item", None)))
                phase = "tool_execution" if pending_tools else "model"
                return
            if phase == "first_event":
                phase = "model"
            return
        if isinstance(event, AgentUpdatedStreamEvent):
            last_event_kind = "agent_updated"
            if phase == "first_event":
                phase = "model"
            return
        # 未知（含非 SDK 类型）事件：不切换阶段、不放大窗口，仅记录
        # 调试上下文供超时审计
        kind = f"unknown:{type(event).__name__}"
        last_event_kind = kind
        unknown_event_kinds.add(kind)
        if phase == "first_event":
            phase = "model"

    async def _cancel_and_raise_idle(configured: float, window: float) -> NoReturn:
        """阶段空闲超时：取消底层运行并尽力关闭流，再抛携带 phase 信息的超时。"""
        category = _tool_phase_category()
        category_suffix = (
            f", category={category}" if category is not None else ""
        )
        streamed.cancel()
        # 任务 45：取消后在有限宽限内尽力关闭流式迭代器（迟到结果由
        # 调用方停止门隔离）；宽限耗尽或关闭异常不影响超时抛出
        await _aclose_quietly(iterator, resolved_cleanup_grace)
        _emit_timeout_audit("phase_idle", configured, window)
        raise asyncio.TimeoutError(
            f"{phase} idle timeout: no SDK stream events for {window:g}s "
            f"(configured={configured:g}s, elapsed={_elapsed():g}s"
            f"{category_suffix})"
        )

    async def _cancel_and_raise_total(window: float) -> NoReturn:
        """绝对总预算超时：取消底层运行并尽力关闭流，再抛携带 phase 信息的超时。"""
        streamed.cancel()
        await _aclose_quietly(iterator, resolved_cleanup_grace)
        _emit_timeout_audit("agent_total", resolved_total, window)
        raise asyncio.TimeoutError(
            f"total timeout: agent budget {resolved_total:g}s exceeded "
            f"(phase={phase}, elapsed={_elapsed():g}s)"
        )

    # 3) 逐事件消费：每次等待取“阶段窗口”与“剩余总预算”的较小值
    #    （remaining 检查先行：<=0 时立即超时收尾，窗口恒为正，不会把
    #    负剩余时间传给 wait_for）
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            await _cancel_and_raise_total(0.0)
        configured = _phase_window()
        window = min(configured, remaining)
        try:
            event = await asyncio.wait_for(iterator.__anext__(), timeout=window)
        except StopAsyncIteration:
            break
        except asyncio.TimeoutError:
            if window < configured:
                # 等待窗口被总预算截断：按总超时判定，保证总预算即使
                # 在较大阶段窗口下也能按时触发
                await _cancel_and_raise_total(window)
            await _cancel_and_raise_idle(configured, window)
        events_seen += 1
        _track(event)
    return streamed


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
    plan_and_review: Callable[..., Awaitable[bool]] | None = None
    issue_grants: Callable[[], list[ToolGrant]] | None = None
    consume_replan_quota: Callable[[], bool] | None = None


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


"""流式 Agent 运行辅助（从 workflow_support 拆分，行为不变）。

``run_agent_streamed`` 以 SDK 原生流事件做分阶段存活检测：首事件窗口、
模型空闲窗口与工具执行窗口；默认不设总时长上限（任务 80）。``runner``
为 None 时经 ``_default_runner`` 动态读取 ``workflow_support.Runner``，
保持测试对 ``so_agent.workflow_support.Runner`` 的 monkeypatch 语义。
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from typing import Any, Callable, NoReturn

from agents import Agent
from agents.stream_events import (
    AgentUpdatedStreamEvent,
    RawResponsesStreamEvent,
    RunItemStreamEvent,
)

from so_agent.runtime.events import (
    EVENT_TIMEOUT_AUDIT,
    EVENT_TOOL_CALLED,
    EventLogger,
)


def _default_runner() -> Any:
    """读取 workflow_support 当前的 Runner（保持模块级 monkeypatch 语义）。"""
    from so_agent import workflow_support as workflow_support_module

    return workflow_support_module.Runner


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
# 默认不设单次调用总预算：正常任务可以持续任意时长，只靠阶段空闲
# 窗口检测断流/卡死（任务 80）。
_DEFAULT_AGENT_TOTAL_TIMEOUT: float = 0.0

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

# 嵌套子 Agent 工具窗口的清理宽限（秒）：未显式传参且总预算为正时
# 派生 subagent_tool_execution_timeout = agent_total_timeout + 本宽限；
# 总预算 <=0（默认不设）时派生为 0。
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


@dataclass(frozen=True)
class _PendingToolCall:
    """一次尚未收到输出的工具调用（工具阶段窗口选择的配对单元）。

    Attributes:
        call_id: SDK 工具项的 call_id（与 tool_output 精确配对；缺失时
            为 None）。
        name: 由 ``ToolCallItem.tool_name`` 提取的工具名（可能为 None）。
        category: "subagent"（嵌套子 Agent 执行工具）或 "standard"。
        window: 该工具所需的活动窗口（秒）。
        started: 工具开始时刻（monotonic 秒，用于计算调用耗时）。
    """

    call_id: str | None
    name: str | None
    category: str
    window: float
    started: float = 0.0


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


def _preview_value(value: Any, limit: int) -> str:
    """把工具参数/输出压缩为单行短文本（仅用于控制台展示）。"""
    if value is None:
        return ""
    if not isinstance(value, str):
        try:
            value = json.dumps(value, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            value = str(value)
    text = " ".join(value.split())
    return text if len(text) <= limit else text[:limit] + "…"


def _extract_tool_arguments(item: Any) -> str:
    """从工具调用项提取参数摘要（FunctionTool 的 raw_item.arguments）。"""
    raw = getattr(item, "raw_item", None)
    arguments: Any = None
    if isinstance(raw, dict):
        arguments = raw.get("arguments")
    else:
        arguments = getattr(raw, "arguments", None)
        if arguments is None:
            arguments = getattr(raw, "input", None)
    return _preview_value(arguments, 240)


def _extract_tool_output(item: Any) -> tuple[str, bool | None]:
    """从工具输出项提取结果摘要与成功标志（能判定时）。

    仅当输出是带 ``success`` 布尔字段或 ``status=error`` 的 JSON 对象时
    返回确定结果；其余情况返回 None，界面按“已完成”展示。
    """
    raw = getattr(item, "raw_item", None)
    output: Any = getattr(item, "output", None)
    if output is None and isinstance(raw, dict):
        output = raw.get("output")
    if output is None and not isinstance(raw, dict):
        output = getattr(raw, "output", None)
    preview = _preview_value(output, 240)
    ok: bool | None = None
    if isinstance(output, str):
        try:
            parsed = json.loads(output)
        except (TypeError, ValueError):
            parsed = None
        if isinstance(parsed, dict):
            if isinstance(parsed.get("success"), bool):
                ok = parsed["success"]
            elif parsed.get("status") == "error":
                ok = False
    elif isinstance(output, dict):
        if isinstance(output.get("success"), bool):
            ok = output["success"]
        elif output.get("status") == "error":
            ok = False
    return preview, ok


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
    activity_hook: Callable[[str], None] | None = None,
    activity_interval: float = 5.0,
    run_config: Any = None,
    progress_state: dict[str, Any] | None = None,
    session: Any = None,
) -> Any:
    """以流式方式运行 Agent：分阶段存活检测 + 可选绝对总预算。

    超时语义（任务 80）：**正常任务不设总时长上限**。默认
    ``agent_total_timeout=0`` 表示不设绝对 deadline，合法长调用可以
    持续任意时长；只保留以下“存活/卡死”检测：

    - ``first_event`` 阶段（首个流事件到达前，覆盖 SDK 冷启动与模型
      首包等待）使用 ``first_event_timeout``；
    - ``model`` 阶段（模型产出流事件期间）使用 ``model_idle_timeout``：
      连续该秒数无任何 SDK 原生流事件才判定断流/卡死；不接受 LLM
      自行输出的文本心跳作为续时依据；
    - ``tool_execution`` 阶段（语义 ``tool_called`` 事件到达后、对应
      ``tool_output`` 事件到达前）使用所有 pending 工具所需窗口的安全
      上限：普通工具为 ``tool_execution_timeout``，嵌套子 Agent /
      code_agent 为对应长窗口；窗口 <=0 表示该阶段不设父级硬截止
      （由子 Agent 自身的存活检测负责，任务 80）；
    - 任一阶段窗口 <=0 表示关闭该窗口；总预算 <=0 表示不设绝对
      deadline。只有显式开启总预算时，阶段窗口才会被剩余总预算截断。

    ``activity_hook``：每个 SDK 原生流事件到达时（按
    ``activity_interval`` 秒节流）回调一次，供 pipeline 活动看门狗
    续时——正常长调用只要持续产出事件就不会被“无进展”误判卡死；
    回调异常被静默吞并，不影响执行。

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
        first_event_timeout: 首事件等待窗口（秒）；None 取默认 180；
            <=0 表示关闭该窗口。
        model_idle_timeout: 模型阶段空闲窗口（秒）；None 取默认 90；
            <=0 表示关闭该窗口。
        tool_execution_timeout: 普通工具执行窗口（秒）；None 取默认
            180；<=0 表示关闭该窗口。
        subagent_tool_execution_timeout: 嵌套子 Agent 执行工具的工具
            阶段窗口（秒）；None 时在总预算为正的情况下派生为
            agent_total_timeout + 10 秒，总预算 <=0 时派生为 0
            （不设父级硬截止）。
        agent_total_timeout: 绝对总预算（秒）；None 取默认 0（不设）；
            <=0 表示不设绝对 deadline。
        runner: Runner 类（run_streamed / run 的持有者）；None 时使用
            ``agents.Runner``。
        events: 可选事件记录器；提供时在超时抛出前写入 timeout_audit。
        audit_context: 可选审计上下文（如 agent / task_id /
            subtask_id / pipeline_phase）；原样并入超时审计 payload，
            禁止放入敏感输入。
        tool_timeout_resolver: 可选的工具阶段窗口解析器（任务 45）：
            提供时按实际 ``tool_name`` 经超时 profile（code_agent /
            subagent / sandbox / standard）解析工具阶段等待窗口；
            返回 <=0 表示该工具阶段不设父级硬截止。
        cleanup_grace_timeout: 超时/取消收尾时等待底层流式迭代器
            ``aclose()`` 完成的宽限（秒）；None 取默认 2 秒。
        activity_hook: 可选活动回调（任何 SDK 流事件到达时调用，
            用于 pipeline 看门狗续时）；None 表示不回调。
        activity_interval: 活动回调最小间隔（秒），默认 5 秒；避免
            高频事件造成额外开销。
        progress_state: 可选的可变进度累加器：工具结束时写入成功/失败
            次数、成功工具名（有界）与最近一次成功输出摘要，供调用方
            在回合上限中断后判断"本轮是否真的做了事"。`None` 表示不记录。
        session: 可选 SDK 会话对象，透传给 Runner。提供时同一会话的
            历史由 SDK 自动合并到 input，调用方后续只需传入增量输入
            即可续跑，无需重建完整载荷。

    Returns:
        运行完成的 RunResultStreaming（含 final_output / raw_responses /
        new_items / last_agent 等属性，与 RunResult 兼容）。

    Raises:
        asyncio.TimeoutError: 阶段空闲超时或显式开启的总预算超时；
            抛出前已调用 ``streamed.cancel()`` 并在有限宽限内尽力
            关闭流式迭代器；消息携带 phase 与耗时上下文。
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
        else (
            resolved_total + _SUBAGENT_TOOL_CLEANUP_GRACE
            if resolved_total > 0
            else 0.0
        )
    )
    resolved_cleanup_grace = (
        float(cleanup_grace_timeout)
        if cleanup_grace_timeout is not None
        else _DEFAULT_STREAM_CLEANUP_GRACE
    )

    active_runner = runner if runner is not None else _default_runner()
    run_kwargs: dict[str, Any] = {"max_turns": max_turns}
    if run_config is not None:
        run_kwargs["run_config"] = run_config
    if session is not None:
        run_kwargs["session"] = session
    streamed = active_runner.run_streamed(agent, input=input, **run_kwargs)
    started = time.monotonic()
    # 总预算 <=0：不设绝对 deadline，只靠阶段空闲窗口检测断流/卡死
    deadline = started + resolved_total if resolved_total > 0 else None
    iterator = streamed.stream_events()

    # 2) 阶段跟踪状态（仅由 SDK 原生流事件与工具阶段事件驱动）
    phase = "first_event"
    pending_tools: list[_PendingToolCall] = []
    events_seen = 0
    last_event_kind = "none"
    unknown_event_kinds: set[str] = set()
    last_activity_at = started - max(0.0, float(activity_interval))

    def _emit_activity() -> None:
        """按最小间隔回调活动信号（pipeline 看门狗续时）。"""
        nonlocal last_activity_at
        if activity_hook is None:
            return
        now = time.monotonic()
        if now - last_activity_at < max(0.0, float(activity_interval)):
            return
        last_activity_at = now
        try:
            activity_hook(f"stream:{phase}")
        except Exception:  # 活动上报失败不得影响执行
            return

    def _tool_phase_window() -> float:
        """工具阶段窗口：取所有 pending 工具所需窗口的安全上限。

        任一 pending 工具窗口 <=0 表示该工具不设父级硬截止（如默认关闭
        的嵌套子 Agent / code_agent 窗口）：此时整个工具阶段同样不设
        硬截止，否则取最大值会把 0（不限）误当成有限窗口并误杀长任务。
        """
        if not pending_tools:
            return resolved_tool
        windows = [pending.window for pending in pending_tools]
        if any(window <= 0 for window in windows):
            return 0.0
        return max(windows)

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

    def _settle_tool_call(
        call_id: str | None, tool_name: str | None = None
    ) -> _PendingToolCall | None:
        """工具结束：按 call_id 精确配对了结 pending 项并返回该项。

        无法精确配对（call_id 缺失或未命中）时按工具名匹配；仍无法匹配
        时保守移除当前窗口最小的 pending 项——宁可把大窗口（嵌套子
        Agent）保留到确认完成，也不提前收缩窗口造成误杀。
        """
        if call_id is not None:
            for index, pending in enumerate(pending_tools):
                if pending.call_id == call_id:
                    return pending_tools.pop(index)
        if tool_name:
            for index, pending in enumerate(pending_tools):
                if pending.name == tool_name:
                    return pending_tools.pop(index)
        if not pending_tools:
            return None
        # 无法配对时优先移除有限窗口中最小的项；window<=0（不限）视为
        # 最大窗口，保留到确认完成，避免把长任务误当成可提前收缩的项。
        smallest = min(
            range(len(pending_tools)),
            key=lambda index: (
                pending_tools[index].window <= 0,
                pending_tools[index].window,
            ),
        )
        return pending_tools.pop(smallest)

    def _log_tool_start(
        tool_name: str | None, call_id: str | None, item: Any
    ) -> None:
        """记录工具调用开始（参数摘要）；写入失败不影响主流程。"""
        if events is None:
            return
        payload: dict[str, Any] = dict(audit_context or {})
        payload.update(
            {
                "action": "invoke_started",
                "tool_name": tool_name,
                "call_id": call_id,
                "arguments_preview": _extract_tool_arguments(item),
            }
        )
        try:
            events.log_event(EVENT_TOOL_CALLED, payload)
        except Exception:  # 审计写入失败不得影响执行
            return

    def _log_tool_end(
        pending: _PendingToolCall | None, call_id: str | None, item: Any
    ) -> None:
        """记录工具调用结束（结果摘要与耗时）；写入失败不影响主流程。"""
        output_preview, result_ok = _extract_tool_output(item)
        tool_name = (pending.name if pending is not None else None) or _extract_tool_name(
            item
        )
        if progress_state is not None:
            # 进度只服务"是否值得续轮"的判定，不参与业务结论；失败与成功
            # 分开计数，避免把失败的工具调用误当成有效产出。
            try:
                if result_ok is False:
                    progress_state["tool_failures"] = (
                        int(progress_state.get("tool_failures", 0)) + 1
                    )
                else:
                    progress_state["tool_successes"] = (
                        int(progress_state.get("tool_successes", 0)) + 1
                    )
                    names = progress_state.get("successful_tools")
                    if not isinstance(names, list):
                        names = []
                        progress_state["successful_tools"] = names
                    if tool_name and tool_name not in names:
                        names.append(tool_name)
                        del names[8:]
                    progress_state["last_success_preview"] = output_preview[:200]
            except Exception:  # 进度记录失败不得影响工具结果
                pass
        if events is None:
            return
        duration = (
            round(max(0.0, time.monotonic() - pending.started), 3)
            if pending is not None and pending.started
            else 0.0
        )
        action = "invoke_failed" if result_ok is False else "invoke_completed"
        payload: dict[str, Any] = dict(audit_context or {})
        payload.update(
            {
                "action": action,
                "tool_name": tool_name,
                "call_id": call_id,
                "arguments_preview": (
                    _extract_tool_arguments(item) if pending is None else ""
                ),
                "output_preview": output_preview,
                "duration": duration,
                "result_ok": result_ok,
            }
        )
        try:
            events.log_event(EVENT_TOOL_CALLED, payload)
        except Exception:  # 审计写入失败不得影响执行
            return

    def _finalize_pending_tools(reason: str) -> None:
        """为仍未收到输出的工具调用补写失败结束事件（收口 running 状态）。

        工具抛错时 SDK 会在执行阶段直接抛出，不会推送对应的
        ``tool_output`` 事件；若只依赖开始/结束配对，这些调用会在
        审计与 Web 路线图中永久停留在 ``running``。流式运行进入下一
        模型回合、正常结束、异常、超时或取消时统一在此收口：剩余
        pending 项写入 ``invoke_failed``（附耗时与原因），保证一次
        工具调用最多只有一个终态事件。
        """
        while pending_tools:
            pending = pending_tools.pop(0)
            if events is None:
                continue
            duration = (
                round(max(0.0, time.monotonic() - pending.started), 3)
                if pending.started
                else 0.0
            )
            payload: dict[str, Any] = dict(audit_context or {})
            payload.update(
                {
                    "action": "invoke_failed",
                    "tool_name": pending.name,
                    "call_id": pending.call_id,
                    "arguments_preview": "",
                    "output_preview": f"工具未返回输出（{reason}）",
                    "duration": duration,
                    "result_ok": False,
                    "unsettled": True,
                }
            )
            try:
                events.log_event(EVENT_TOOL_CALLED, payload)
            except Exception:  # 审计写入失败不得影响收口
                continue

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
        # 任一 SDK 流事件都代表“仍在活动”：为 pipeline 看门狗续时
        _emit_activity()
        if isinstance(event, RawResponsesStreamEvent):
            # 模型原始 chunk：模型阶段活动；模型继续推进意味着上一工具
            # 阶段已结束（SDK 时序下不会交错，保守恢复模型窗口）。
            # 仍有 pending 说明对应 tool_output 缺失，先按失败收口，
            # 避免工具卡片永久停留在 running。
            last_event_kind = "model_chunk"
            _finalize_pending_tools("进入下一模型回合")
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
                call_id = _extract_call_id(item)
                is_subagent = tool_name in _SUBAGENT_TOOL_NAMES
                if tool_timeout_resolver is not None:
                    window = max(0.0, float(tool_timeout_resolver(tool_name)))
                elif is_subagent:
                    window = resolved_subagent_tool
                else:
                    window = resolved_tool
                pending_tools.append(
                    _PendingToolCall(
                        call_id=call_id,
                        name=tool_name,
                        category=(
                            _SUBAGENT_TOOL_CATEGORY
                            if is_subagent
                            else _STANDARD_TOOL_CATEGORY
                        ),
                        window=window,
                        started=time.monotonic(),
                    )
                )
                _log_tool_start(tool_name, call_id, item)
                phase = "tool_execution"
                return
            if name in _TOOL_END_EVENT_NAMES:
                item = getattr(event, "item", None)
                call_id = _extract_call_id(item)
                settled = _settle_tool_call(call_id, _extract_tool_name(item))
                _log_tool_end(settled, call_id, item)
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

    # 3) 逐事件消费：阶段窗口 <=0 表示关闭该窗口；总预算 <=0 表示
    #    不设 deadline。只有显式开启总预算时，等待窗口才会被剩余总
    #    预算截断；未设 deadline 且当前阶段窗口关闭时，等待完全由
    #    “下一个 SDK 事件”驱动（正常任务不因总时长被终止）。
    finish_reason = "stream_ended"
    try:
        while True:
            remaining = (
                deadline - time.monotonic() if deadline is not None else None
            )
            if remaining is not None and remaining <= 0:
                await _cancel_and_raise_total(0.0)
            configured = _phase_window()
            clipped_by_total = False
            if remaining is not None:
                if configured <= 0 or remaining < configured:
                    window: float | None = remaining
                    clipped_by_total = True
                else:
                    window = configured
            else:
                window = configured if configured > 0 else None
            try:
                if window is None:
                    event = await iterator.__anext__()
                else:
                    event = await asyncio.wait_for(
                        iterator.__anext__(), timeout=window
                    )
            except StopAsyncIteration:
                break
            except asyncio.TimeoutError:
                if clipped_by_total:
                    # 等待窗口被总预算截断：按总超时判定
                    await _cancel_and_raise_total(window or 0.0)
                await _cancel_and_raise_idle(max(0.0, configured), window or 0.0)
            events_seen += 1
            _track(event)
    except asyncio.CancelledError:
        finish_reason = "cancelled"
        raise
    except asyncio.TimeoutError:
        finish_reason = "timeout"
        raise
    except BaseException as exc:
        finish_reason = f"error:{type(exc).__name__}"
        # 失败运行已产生的模型响应仍在 streamed 内部状态里：挂到异常上，
        # 供调用方补记这一段 token（否则失败尝试的消耗完全不入账）。
        try:
            setattr(exc, "partial_result", streamed)
        except Exception:  # 异常不可写属性时忽略
            pass
        raise
    finally:
        # 工具抛错 / 超时 / 取消时 SDK 不会推送 tool_output；在此统一
        # 写终态，保证一次调用最多只有一个结束事件且不会停留在 running
        _finalize_pending_tools(finish_reason)
    return streamed


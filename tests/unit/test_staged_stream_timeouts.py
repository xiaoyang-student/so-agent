"""单元测试：分阶段流式超时（任务 33）。

对照任务验收清单覆盖场景：

1. 首包延迟未超过窗口：首事件在 ``first_event_timeout`` 内到达 → 正常返回；
2. 首包超时：首事件前挂起 → ``first_event idle timeout``（先 cancel 再抛）；
3. 持续 token 不误杀：模型原生事件间隔 < ``model_idle_timeout`` → 全部消费；
4. 模型空闲超时：模型事件产出后停止 → ``model idle timeout``；
5. 工具执行窗口：``tool_called`` 后挂起，较小的模型窗口不误杀、
   较大的工具窗口才超时（修复旧固定空闲窗口误杀工具执行）；
6. 工具返回后恢复模型窗口：``tool_output`` 到达后按模型窗口计时；
7. 持续事件仍被 ``agent_total_timeout`` 绝对截断（含截断工具窗口）；
8. 每个超时路径都调用 ``streamed.cancel()``；
9. scheduler 外层不早杀：默认外层超时 = ``supervisor_total_timeout`` + 宽限，
   且外层硬兜底仍在；
10. 旧配置兼容：``idle_timeout`` / ``total_timeout`` 映射到新阶段窗口，
    且旧空闲窗口不污染工具执行阶段；
11. 权限生命周期不回归：超时链路不改动 ToolGrant / 网关状态；
12. 审计：``timeout_audit`` 字段完整、合并调用方上下文、不写敏感输入、
    审计写入失败不掩盖超时本身；
13. 嵌套子 Agent 工具窗口（任务 33 修订）：按真实 ToolCallItem 内容的
    ``tool_name`` 识别 ``subagent_creator`` / ``execute_subagent``，工具
    阶段使用 ``subagent_tool_execution_timeout``（默认派生 =
    agent_total + 10 秒），超过普通工具窗口但低于子 Agent 总预算的运行
    不被父级误杀；
14. 普通工具仍按 ``tool_execution_timeout`` 超时；并行 pending 取最大
    窗口，仍被主管绝对 deadline 截断；call_id 精确配对（无法配对时保守
    保留大窗口）；
15. 嵌套子 Agent 的硬上限：子 Agent 自身 ``agent_total_timeout`` 与父级
    （主管）绝对总预算分别兜底，大窗口不会突破总预算；
16. 父级总预算受 pipeline 剩余预算约束（``_supervisor_budget``）；
17. 嵌套配置兼容与不变量（旧字段映射、派生窗口、保底校验）——
    详见 ``test_config.py::TestDeprecatedTimeoutAliases``。

时间参数大量使用「缩放值」（如 0.05s 对应 90s 语义）以保持测试秒级运行，
但阶段归属、窗口选择与消息内容按真实语义断言。
"""

from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from agents.items import ToolCallItem, ToolCallOutputItem
from agents.stream_events import (
    AgentUpdatedStreamEvent,
    RawResponsesStreamEvent,
    RunItemStreamEvent,
)

from so_agent.config import Settings
from so_agent.models import AgentRecord, ExecutionResult, SubtaskSpec
from so_agent.runtime.events import EVENT_TIMEOUT_AUDIT, EventLogger
from so_agent.runtime.permissions import PermissionGateway
from so_agent.runtime.registry import DEFAULT_SUPERVISOR_ID
from so_agent.runtime.scheduler import (
    _ATTEMPT_TIMEOUT_CLEANUP_GRACE,
    SubtaskDAGScheduler,
)
from so_agent.workflow import WorkflowEngine
from so_agent.workflow_support import PipelineGate, run_agent_streamed

# ---------------------------------------------------------------------------
# 测试替身与事件构造
# ---------------------------------------------------------------------------


def raw_chunk(label: str = "chunk") -> RawResponsesStreamEvent:
    """构造 SDK 原生模型流事件（RawResponsesStreamEvent）。"""
    return RawResponsesStreamEvent(data={"label": label})


class _ToolItemAgentStub:
    """ToolCallItem / ToolCallOutputItem 的 agent 替身。

    SDK ``RunItemBase.__post_init__`` 会对 agent 建 weakref；
    普通 ``object()`` 不支持弱引用，这里使用支持 weakref 的类实例。
    """


def item_event(
    name: str,
    *,
    tool_name: str | None = None,
    call_id: str | None = None,
) -> RunItemStreamEvent:
    """构造 SDK 语义运行项事件（RunItemStreamEvent.name 为真实 Literal 取值）。

    工具阶段事件携带真实 SDK 工具项：``tool_called`` 用 ``ToolCallItem``、
    ``tool_output`` 用 ``ToolCallOutputItem``——工具名与配对键均走 SDK
    公共属性（``tool_name`` / ``call_id``），与生产识别路径完全一致，
    不依赖任何文本输出猜测。
    """
    if name in {"tool_called", "tool_search_called"}:
        raw: dict = {}
        if tool_name is not None:
            raw["name"] = tool_name
        if call_id is not None:
            raw["call_id"] = call_id
        item: object = ToolCallItem(agent=_ToolItemAgentStub(), raw_item=raw)
    elif name in {"tool_output", "tool_search_output_created"}:
        raw = {"call_id": call_id} if call_id is not None else {}
        item = ToolCallOutputItem(
            agent=_ToolItemAgentStub(), raw_item=raw, output="ok"
        )
    else:
        item = object()
    return RunItemStreamEvent(name=name, item=item)


def subagent_event(
    call_id: str, tool_name: str = "subagent_creator"
) -> RunItemStreamEvent:
    """嵌套子 Agent 执行工具的 tool_called 事件（真实 ToolCallItem）。"""
    return item_event("tool_called", tool_name=tool_name, call_id=call_id)


def tool_output_event(call_id: str | None = None) -> RunItemStreamEvent:
    """工具输出事件（真实 ToolCallOutputItem）。"""
    return item_event("tool_output", call_id=call_id)


def agent_updated() -> AgentUpdatedStreamEvent:
    """构造 Agent 切换流事件。"""
    return AgentUpdatedStreamEvent(new_agent=object())


class ScriptedStreamedResult:
    """假 RunResultStreaming：按脚本逐步产出事件、等待或永久挂起。

    脚本步骤：
    - ``("event", ev)``：产出事件 ev；
    - ``("sleep", sec)``：等待 sec 秒（模拟 token 间隔 / 工具执行时长）；
    - ``("repeat", ev, gap)``：无限循环产出事件（模拟永不结束的流）；
    - ``("hang",)``：永久挂起（模拟卡死：不再产生任何事件）。
    """

    def __init__(self, script) -> None:
        self._script = list(script)
        self.final_output = "scripted-output"
        self.raw_responses: list = []
        self.new_items: list = []
        self.cancelled = False
        self.consumed = 0

    def cancel(self, mode: str = "immediate") -> None:
        self.cancelled = True

    async def stream_events(self):
        for step in self._script:
            kind = step[0]
            if kind == "sleep":
                await asyncio.sleep(step[1])
            elif kind == "event":
                self.consumed += 1
                yield step[1]
            elif kind == "repeat":
                while True:
                    await asyncio.sleep(step[2])
                    self.consumed += 1
                    yield step[1]
            else:  # ("hang",)
                await asyncio.Event().wait()
                return


class FakeRunner:
    """假 Runner：run_streamed 返回预置的假流式结果并记录调用参数。"""

    def __init__(self, streamed: ScriptedStreamedResult) -> None:
        self.streamed = streamed
        self.calls: list[dict] = []

    def run_streamed(self, agent, input=None, *, max_turns=None, **kwargs):
        self.calls.append({"agent": agent, "input": input, "max_turns": max_turns})
        return self.streamed


AGENT = object()


def _run(streamed: ScriptedStreamedResult, **kwargs):
    """以假 Runner 调用 run_agent_streamed（保留关键字透传）。"""
    runner = FakeRunner(streamed)
    return run_agent_streamed(AGENT, input="x", max_turns=1, runner=runner, **kwargs)


# ---------------------------------------------------------------------------
# 1) 首事件阶段
# ---------------------------------------------------------------------------


class TestFirstEventPhase:
    """首事件阶段：首个 SDK 流事件前使用 first_event_timeout。"""

    async def test_first_packet_within_window_completes(self):
        # 首包延迟 0.3s < first_event 0.5s → 正常返回，全部事件消费
        streamed = ScriptedStreamedResult(
            [("sleep", 0.3), ("event", raw_chunk()), ("event", raw_chunk("b"))]
        )
        result = await _run(
            streamed, first_event_timeout=0.5, agent_total_timeout=5.0
        )
        assert result is streamed
        assert streamed.consumed == 2
        assert streamed.cancelled is False

    async def test_first_packet_timeout_raised(self):
        # 首事件前挂起超过 first_event 0.05s → first_event idle timeout
        streamed = ScriptedStreamedResult([("hang",)])
        with pytest.raises(asyncio.TimeoutError, match=r"first_event idle timeout") as ei:
            await _run(
                streamed, first_event_timeout=0.05, agent_total_timeout=5.0
            )
        assert "configured=0.05" in str(ei.value)
        assert streamed.cancelled is True  # 先 cancel 再抛

    async def test_default_windows_do_not_kill_fast_first_packet(self):
        # 不传任何新参数：默认 first_event=180s，首包 0.05s 远未超时
        streamed = ScriptedStreamedResult(
            [("sleep", 0.05), ("event", raw_chunk())]
        )
        result = await _run(streamed, agent_total_timeout=5.0)
        assert result is streamed
        assert streamed.cancelled is False


# ---------------------------------------------------------------------------
# 2) 模型阶段
# ---------------------------------------------------------------------------


class TestModelIdlePhase:
    """模型阶段：连续无 SDK 原生流事件达到 model_idle_timeout 才判定卡死。"""

    async def test_continuous_tokens_not_killed(self):
        # 10 个 token 间隔 0.02s，均 < model_idle 0.1s → 全部消费不误杀
        script = []
        for i in range(10):
            script.append(("sleep", 0.02))
            script.append(("event", raw_chunk(str(i))))
        streamed = ScriptedStreamedResult(script)
        result = await _run(
            streamed,
            model_idle_timeout=0.1,
            agent_total_timeout=5.0,
        )
        assert result is streamed
        assert streamed.consumed == 10
        assert streamed.cancelled is False

    async def test_model_idle_timeout_after_output(self):
        # 模型产出一个 chunk 后停止 → model idle timeout（configured=模型窗口）
        streamed = ScriptedStreamedResult([("event", raw_chunk()), ("hang",)])
        with pytest.raises(asyncio.TimeoutError, match=r"model idle timeout") as ei:
            await _run(
                streamed, model_idle_timeout=0.05, agent_total_timeout=5.0
            )
        assert "configured=0.05" in str(ei.value)
        assert streamed.cancelled is True

    async def test_agent_updated_leaves_first_event_phase(self):
        # AgentUpdatedStreamEvent 也是首事件（显式将阶段推进到 model）
        streamed = ScriptedStreamedResult([("event", agent_updated()), ("hang",)])
        with pytest.raises(asyncio.TimeoutError, match=r"model idle timeout") as ei:
            await _run(
                streamed,
                first_event_timeout=5.0,
                model_idle_timeout=0.05,
                agent_total_timeout=5.0,
            )
        assert "configured=0.05" in str(ei.value)

    async def test_unknown_event_is_activity_without_window_amplification(self):
        # 未知事件：作为活动信号并离开 first_event，但窗口按 model 计算，
        # 不得因未知事件无限续时
        streamed = ScriptedStreamedResult([("event", {"raw": "dict"}), ("hang",)])
        with pytest.raises(asyncio.TimeoutError, match=r"model idle timeout") as ei:
            await _run(
                streamed,
                first_event_timeout=5.0,
                model_idle_timeout=0.05,
                agent_total_timeout=5.0,
            )
        assert "configured=0.05" in str(ei.value)
        assert streamed.cancelled is True


# ---------------------------------------------------------------------------
# 3) 工具执行阶段
# ---------------------------------------------------------------------------


class TestToolExecutionPhase:
    """工具执行阶段：tool_called → tool_output 之间使用 tool_execution_timeout。"""

    async def test_tool_window_not_killed_by_smaller_model_window(self):
        # 核心修复：tool_called 后挂起，模型窗口（0.1s，对应 90s）不得误杀，
        # 工具窗口（0.5s，对应 180s）才超时。
        evs = EventLogger()
        streamed = ScriptedStreamedResult(
            [("event", raw_chunk()), ("event", item_event("tool_called")), ("hang",)]
        )
        started = time.monotonic()
        with pytest.raises(
            asyncio.TimeoutError, match=r"tool_execution idle timeout"
        ) as ei:
            await _run(
                streamed,
                model_idle_timeout=0.1,
                tool_execution_timeout=0.5,
                agent_total_timeout=5.0,
                events=evs,
                audit_context={"agent": "subagent", "subtask_id": "s-1"},
            )
        elapsed = time.monotonic() - started
        assert "configured=0.5" in str(ei.value)
        assert elapsed >= 0.4  # 未被 0.1s 的模型窗口提前误杀
        assert streamed.cancelled is True
        payload = evs.events[-1]["payload"]
        assert payload["phase"] == "tool_execution"
        assert payload["configured_timeout"] == 0.5
        assert payload["elapsed"] >= 0.4

    async def test_tool_output_restores_model_window(self):
        # tool_output 到达后恢复模型窗口：后续挂起按 model 窗口 0.1s 超时
        streamed = ScriptedStreamedResult(
            [
                ("event", raw_chunk()),
                ("event", item_event("tool_called")),
                ("sleep", 0.05),
                ("event", item_event("tool_output")),
                ("hang",),
            ]
        )
        with pytest.raises(asyncio.TimeoutError, match=r"model idle timeout") as ei:
            await _run(
                streamed,
                model_idle_timeout=0.1,
                tool_execution_timeout=0.5,
                agent_total_timeout=5.0,
            )
        # 若错误地停留在工具窗口，configured 会是 0.5
        assert "configured=0.1" in str(ei.value)
        assert streamed.cancelled is True

    async def test_parallel_tools_keep_tool_window_until_all_outputs(self):
        # 两个 tool_called（含 tool_search_called）并行：第一个 tool_output
        # 后仍有 pending → 仍在工具窗口；超时按工具窗口判定
        streamed = ScriptedStreamedResult(
            [
                ("event", raw_chunk()),
                ("event", item_event("tool_called")),
                ("event", item_event("tool_search_called")),
                ("sleep", 0.05),
                ("event", item_event("tool_output")),
                ("hang",),
            ]
        )
        with pytest.raises(
            asyncio.TimeoutError, match=r"tool_execution idle timeout"
        ) as ei:
            await _run(
                streamed,
                model_idle_timeout=0.1,
                tool_execution_timeout=0.5,
                agent_total_timeout=5.0,
            )
        assert "configured=0.5" in str(ei.value)

    async def test_tool_search_output_also_restores_model_window(self):
        # tool_search_output_created 也是工具结束事件
        streamed = ScriptedStreamedResult(
            [
                ("event", item_event("tool_search_called")),
                ("sleep", 0.05),
                ("event", item_event("tool_search_output_created")),
                ("hang",),
            ]
        )
        with pytest.raises(asyncio.TimeoutError, match=r"model idle timeout") as ei:
            await _run(
                streamed,
                model_idle_timeout=0.1,
                tool_execution_timeout=0.5,
                agent_total_timeout=5.0,
            )
        assert "configured=0.1" in str(ei.value)

    async def test_first_named_event_switches_tool_window(self):
        # 关键风险说明中的保守兼容：首个语义 tool_called 事件（即便发生在
        # 首事件阶段）立即切换工具窗口
        streamed = ScriptedStreamedResult([("event", item_event("tool_called")), ("hang",)])
        with pytest.raises(
            asyncio.TimeoutError, match=r"tool_execution idle timeout"
        ) as ei:
            await _run(
                streamed,
                first_event_timeout=5.0,
                model_idle_timeout=5.0,
                tool_execution_timeout=0.05,
                agent_total_timeout=5.0,
            )
        assert "configured=0.05" in str(ei.value)


# ---------------------------------------------------------------------------
# 4) agent_total_timeout 绝对总预算
# ---------------------------------------------------------------------------


class TestAgentTotalBudget:
    """绝对总预算：每次等待取「阶段窗口」与「剩余总预算」较小值。"""

    async def test_continuous_events_capped_by_total_budget(self):
        # 事件持续产出（间隔 0.02 < 模型窗口 1.0），但总预算 0.15s 到期
        streamed = ScriptedStreamedResult([("repeat", raw_chunk(), 0.02)])
        with pytest.raises(asyncio.TimeoutError, match=r"total timeout") as ei:
            await _run(
                streamed,
                model_idle_timeout=1.0,
                agent_total_timeout=0.15,
            )
        assert "agent budget 0.15s exceeded" in str(ei.value)
        assert streamed.consumed >= 3  # 事件确实在持续产出（排除空闲误判）
        assert streamed.cancelled is True

    async def test_total_budget_truncates_larger_tool_window(self):
        # 工具窗口 5.0s 远大于剩余总预算 0.15s：等待取较小值并走 total 判定
        streamed = ScriptedStreamedResult(
            [("event", item_event("tool_called")), ("hang",)]
        )
        with pytest.raises(asyncio.TimeoutError, match=r"total timeout") as ei:
            await _run(
                streamed,
                model_idle_timeout=5.0,
                tool_execution_timeout=5.0,
                agent_total_timeout=0.15,
            )
        message = str(ei.value)
        assert "agent budget 0.15s exceeded" in message
        assert "phase=tool_execution" in message
        assert streamed.cancelled is True

    async def test_total_budget_audit_kind(self):
        evs = EventLogger()
        streamed = ScriptedStreamedResult([("repeat", raw_chunk(), 0.02)])
        with pytest.raises(asyncio.TimeoutError, match=r"total timeout"):
            await _run(
                streamed,
                model_idle_timeout=1.0,
                agent_total_timeout=0.15,
                events=evs,
                audit_context={"agent": "supervisor", "task_id": "t-9"},
            )
        record = evs.events[-1]
        assert record["event_type"] == EVENT_TIMEOUT_AUDIT
        payload = record["payload"]
        assert payload["timeout_kind"] == "agent_total"
        assert payload["agent_total_timeout"] == 0.15
        assert payload["agent"] == "supervisor"
        assert payload["task_id"] == "t-9"


# ---------------------------------------------------------------------------
# 4.5) 嵌套子 Agent 工具窗口（任务 33 修订）
# ---------------------------------------------------------------------------


class TestNestedSubagentToolWindow:
    """嵌套子 Agent 执行工具：按 ToolCallItem.tool_name 识别并独立窗口。

    等比例缩放值（约 1/100）：tool_execution_timeout=0.3（对应 180s）、
    subagent_tool_execution_timeout=1.2（对应 610s 量级）、
    agent_total_timeout=4.0（对应 600s 量级）。
    """

    async def test_sdk_tool_item_exposes_names_used_for_recognition(self):
        # 识别证据：只依赖真实 SDK 工具项公共属性（非文本输出猜测）
        call = ToolCallItem(
            agent=_ToolItemAgentStub(),
            raw_item={"name": "subagent_creator", "call_id": "c-1"},
        )
        assert call.tool_name == "subagent_creator"
        assert call.call_id == "c-1"
        output = ToolCallOutputItem(
            agent=_ToolItemAgentStub(),
            raw_item={"call_id": "c-1"},
            output="ok",
        )
        assert output.call_id == "c-1"

    async def test_subagent_tool_past_standard_window_not_killed(self):
        # 核心修复：嵌套工具运行 0.6s（> 普通窗口 0.3，对应 >180s）后
        # 正常返回，低于子 Agent 总预算（4.0）→ 不得被父工具窗口误杀
        streamed = ScriptedStreamedResult(
            [
                ("event", raw_chunk()),
                ("event", subagent_event("c-1")),
                ("sleep", 0.6),
                ("event", tool_output_event("c-1")),
                ("event", raw_chunk()),
            ]
        )
        result = await _run(
            streamed,
            model_idle_timeout=1.5,
            tool_execution_timeout=0.3,
            subagent_tool_execution_timeout=1.2,
            agent_total_timeout=4.0,
        )
        assert result is streamed
        assert streamed.consumed == 4
        assert streamed.cancelled is False

    async def test_execute_subagent_name_also_uses_subagent_window(self):
        # 防御分支：execute_subagent（创建器内部工具名）同样按嵌套窗口处理
        streamed = ScriptedStreamedResult(
            [
                ("event", raw_chunk()),
                (
                    "event",
                    item_event(
                        "tool_called",
                        tool_name="execute_subagent",
                        call_id="c-1",
                    ),
                ),
                ("sleep", 0.6),
                ("event", tool_output_event("c-1")),
                ("event", raw_chunk()),
            ]
        )
        result = await _run(
            streamed,
            model_idle_timeout=1.5,
            tool_execution_timeout=0.3,
            subagent_tool_execution_timeout=1.2,
            agent_total_timeout=4.0,
        )
        assert result is streamed
        assert streamed.cancelled is False

    async def test_standard_tool_still_times_out_at_standard_window(self):
        # 对照组：普通工具（code_agent）超过 0.3s（对应 180s）仍超时，
        # 且不误用嵌套窗口 1.2
        streamed = ScriptedStreamedResult(
            [
                ("event", raw_chunk()),
                (
                    "event",
                    item_event(
                        "tool_called", tool_name="code_agent", call_id="c-1"
                    ),
                ),
                ("hang",),
            ]
        )
        started = time.monotonic()
        with pytest.raises(
            asyncio.TimeoutError, match=r"tool_execution idle timeout"
        ) as ei:
            await _run(
                streamed,
                model_idle_timeout=1.5,
                tool_execution_timeout=0.3,
                subagent_tool_execution_timeout=1.2,
                agent_total_timeout=4.0,
            )
        elapsed = time.monotonic() - started
        assert "configured=0.3" in str(ei.value)
        assert "category=standard" in str(ei.value)
        assert elapsed < 1.0  # 若误用嵌套窗口会在 1.2s 才超时
        assert streamed.cancelled is True

    async def test_parallel_tools_take_max_window_of_pending(self):
        # 并行工具取所有 pending 窗口的最大值：普通工具 0.6s 后返回
        # （> 0.3 普通窗口），嵌套 pending 保证窗口为 1.2，不误杀
        streamed = ScriptedStreamedResult(
            [
                ("event", raw_chunk()),
                (
                    "event",
                    item_event(
                        "tool_called", tool_name="code_agent", call_id="c-1"
                    ),
                ),
                ("event", subagent_event("c-2")),
                ("sleep", 0.6),
                ("event", tool_output_event("c-1")),
                ("sleep", 0.6),
                ("event", tool_output_event("c-2")),
                ("event", raw_chunk()),
            ]
        )
        result = await _run(
            streamed,
            model_idle_timeout=1.5,
            tool_execution_timeout=0.3,
            subagent_tool_execution_timeout=1.2,
            agent_total_timeout=5.0,
        )
        assert result is streamed
        assert streamed.cancelled is False

    async def test_exact_call_id_settlement_shrinks_window(self):
        # call_id 精确配对：嵌套工具（c-1）先返回后，窗口回落到剩余的
        # 普通工具（0.3）——悬挂后按 configured=0.3 超时
        streamed = ScriptedStreamedResult(
            [
                ("event", raw_chunk()),
                ("event", subagent_event("c-1")),
                (
                    "event",
                    item_event(
                        "tool_called", tool_name="code_agent", call_id="c-2"
                    ),
                ),
                ("sleep", 0.1),
                ("event", tool_output_event("c-1")),
                ("hang",),
            ]
        )
        with pytest.raises(
            asyncio.TimeoutError, match=r"tool_execution idle timeout"
        ) as ei:
            await _run(
                streamed,
                model_idle_timeout=1.5,
                tool_execution_timeout=0.3,
                subagent_tool_execution_timeout=1.2,
                agent_total_timeout=4.0,
            )
        assert "configured=0.3" in str(ei.value)
        assert "category=standard" in str(ei.value)

    async def test_unknown_call_id_keeps_large_window_conservatively(self):
        # 无法精确配对（输出 call_id 未知）：保守移除最小窗口（普通 0.3），
        # 保留嵌套大窗口 1.2——悬挂后按 configured=1.2 超时，不提前收缩
        streamed = ScriptedStreamedResult(
            [
                ("event", raw_chunk()),
                ("event", subagent_event("c-1")),
                (
                    "event",
                    item_event(
                        "tool_called", tool_name="code_agent", call_id="c-2"
                    ),
                ),
                ("sleep", 0.1),
                ("event", tool_output_event("c-99")),
                ("hang",),
            ]
        )
        with pytest.raises(
            asyncio.TimeoutError, match=r"tool_execution idle timeout"
        ) as ei:
            await _run(
                streamed,
                model_idle_timeout=1.5,
                tool_execution_timeout=0.3,
                subagent_tool_execution_timeout=1.2,
                agent_total_timeout=4.0,
            )
        assert "configured=1.2" in str(ei.value)
        assert "category=subagent" in str(ei.value)

    async def test_subagent_window_truncated_by_supervisor_total(self):
        # 父级硬上限：嵌套窗口 1.2 被主管绝对总预算 0.8 截断；
        # 悬挂在工具阶段 → total timeout（phase=tool_execution）
        streamed = ScriptedStreamedResult(
            [
                ("event", raw_chunk()),
                ("event", subagent_event("c-1")),
                ("hang",),
            ]
        )
        with pytest.raises(asyncio.TimeoutError, match=r"total timeout") as ei:
            await _run(
                streamed,
                model_idle_timeout=1.5,
                tool_execution_timeout=0.3,
                subagent_tool_execution_timeout=1.2,
                agent_total_timeout=0.8,
            )
        message = str(ei.value)
        assert "agent budget 0.8s exceeded" in message
        assert "phase=tool_execution" in message
        assert streamed.cancelled is True

    async def test_subagent_own_budget_stays_hard_cap(self):
        # 子级硬上限：子 Agent 自身运行（execute_subagent 路径）仍以
        # agent_total_timeout 为绝对 deadline——事件持续产出也被截断；
        # 派生嵌套窗口（0.2 + 10 = 10.2）不得放大子 Agent 自身预算
        streamed = ScriptedStreamedResult([("repeat", raw_chunk(), 0.02)])
        with pytest.raises(asyncio.TimeoutError, match=r"total timeout") as ei:
            await _run(
                streamed,
                model_idle_timeout=1.0,
                agent_total_timeout=0.2,
            )
        assert "agent budget 0.2s exceeded" in str(ei.value)
        assert streamed.cancelled is True

    async def test_nested_tool_idle_timeout_audit_fields(self):
        # 审计记录实际采用的工具阶段与类别（工具名/类别元数据，不含参数）
        evs = EventLogger()
        streamed = ScriptedStreamedResult(
            [
                ("event", raw_chunk()),
                ("event", subagent_event("c-1")),
                ("hang",),
            ]
        )
        with pytest.raises(
            asyncio.TimeoutError, match=r"tool_execution idle timeout"
        ):
            await _run(
                streamed,
                model_idle_timeout=1.5,
                tool_execution_timeout=0.3,
                subagent_tool_execution_timeout=1.2,
                agent_total_timeout=4.0,
                events=evs,
                audit_context={"agent": "supervisor", "task_id": "t-nest"},
            )
        payload = evs.events[-1]["payload"]
        assert payload["phase"] == "tool_execution"
        assert payload["tool_phase_category"] == "subagent"
        assert payload["pending_tool_calls"] == 1
        assert payload["pending_tool_names"] == ["subagent_creator"]
        assert payload["subagent_tool_execution_timeout"] == 1.2
        assert payload["configured_timeout"] == 1.2
        assert payload["agent"] == "supervisor"
        assert payload["task_id"] == "t-nest"


# ---------------------------------------------------------------------------
# 4.6) 主管总预算（pipeline 剩余预算截断）
# ---------------------------------------------------------------------------


class TestSupervisorBudget:
    """_supervisor_budget：min(supervisor_total_timeout, pipeline 剩余)。"""

    @staticmethod
    def _stub_object(settings: Settings, gate):
        # 任务 45：剩余时间统一经停止门 gate.remaining()（永不为负），
        # 替身与真实引擎形状一致（_gate 为 __init__ 必建属性）
        return SimpleNamespace(_settings=settings, _gate=gate)

    def test_full_budget_when_pipeline_not_started(self):
        # 未设 deadline：gate.remaining() 为 None → 完整主管总预算
        stub = self._stub_object(Settings(), PipelineGate())
        assert WorkflowEngine._supervisor_budget(stub) == 900.0

    def test_budget_never_exceeds_supervisor_total(self):
        stub = self._stub_object(Settings(max_pipeline_timeout=10_000.0), PipelineGate())
        assert WorkflowEngine._supervisor_budget(stub) == 900.0

    def test_budget_clamped_by_pipeline_remaining(self):
        # pipeline 剩余 ≈ 10s < 主管总预算 900s → 预算被截断到剩余
        gate = PipelineGate()
        gate.set_deadline(time.monotonic() + 10.0)
        settings = Settings(max_pipeline_timeout=100.0)
        stub = self._stub_object(settings, gate)
        budget = WorkflowEngine._supervisor_budget(stub)
        assert 9.0 <= budget < 10.0
        assert budget < float(settings.supervisor_total_timeout)

    def test_budget_covers_nested_subagent_window(self):
        # 不变量：agent_total < 嵌套子 Agent 工具窗口 ≤ 主管总预算
        settings = Settings()
        assert (
            settings.agent_total_timeout
            < settings.subagent_tool_execution_timeout
            <= settings.supervisor_total_timeout
        )


# ---------------------------------------------------------------------------
# 5) 超时审计
# ---------------------------------------------------------------------------


class TestTimeoutAudit:
    """timeout_audit 事件：字段完整、上下文合并、不泄露输入、失败不掩盖超时。"""

    async def test_phase_idle_audit_payload_fields(self):
        evs = EventLogger()
        streamed = ScriptedStreamedResult([("hang",)])
        with pytest.raises(asyncio.TimeoutError):
            await _run(
                streamed,
                first_event_timeout=0.05,
                agent_total_timeout=5.0,
                events=evs,
                audit_context={"agent": "subagent", "subtask_id": "sub-7"},
            )
        record = evs.events[-1]
        assert record["event_type"] == EVENT_TIMEOUT_AUDIT
        payload = record["payload"]
        # 必填字段：phase / configured_timeout / elapsed / agent 或 subtask 上下文
        assert payload["timeout_kind"] == "phase_idle"
        assert payload["phase"] == "first_event"
        assert payload["configured_timeout"] == 0.05
        assert payload["elapsed"] >= 0.05
        assert payload["wait_window"] == 0.05
        assert payload["agent_total_timeout"] == 5.0
        assert payload["agent"] == "subagent"
        assert payload["subtask_id"] == "sub-7"
        assert payload["events_seen"] == 0
        assert payload["last_event_kind"] == "none"
        assert payload["pending_tool_calls"] == 0
        assert payload["unknown_event_kinds"] == []

    async def test_unknown_event_kinds_recorded(self):
        evs = EventLogger()
        streamed = ScriptedStreamedResult(
            [("event", {"weird": 1}), ("event", {"weird": 2}), ("hang",)]
        )
        with pytest.raises(asyncio.TimeoutError):
            await _run(
                streamed,
                first_event_timeout=5.0,
                model_idle_timeout=0.05,
                agent_total_timeout=5.0,
                events=evs,
            )
        payload = evs.events[-1]["payload"]
        assert payload["unknown_event_kinds"] == ["unknown:dict"]
        assert payload["last_event_kind"] == "unknown:dict"
        assert payload["events_seen"] == 2

    async def test_audit_context_does_not_contain_sensitive_input(self):
        secret = "SECRET-PROMPT-XYZ-42"
        evs = EventLogger()
        streamed = ScriptedStreamedResult([("hang",)])
        runner = FakeRunner(streamed)
        with pytest.raises(asyncio.TimeoutError):
            await run_agent_streamed(
                AGENT,
                input=secret,
                max_turns=1,
                first_event_timeout=0.05,
                agent_total_timeout=5.0,
                runner=runner,
                events=evs,
                audit_context={"agent": "supervisor"},
            )
        blob = json.dumps(evs.events, ensure_ascii=False)
        assert secret not in blob  # 敏感输入不得完整写入日志

    async def test_audit_write_failure_does_not_mask_timeout(self):
        class FailingLogger:
            def log_event(self, *args, **kwargs):
                raise RuntimeError("audit backend down")

        streamed = ScriptedStreamedResult([("hang",)])
        with pytest.raises(asyncio.TimeoutError, match=r"first_event idle timeout"):
            await _run(
                streamed,
                first_event_timeout=0.05,
                agent_total_timeout=5.0,
                events=FailingLogger(),
            )
        assert streamed.cancelled is True

    async def test_no_timeout_no_audit_event(self):
        evs = EventLogger()
        streamed = ScriptedStreamedResult([("event", raw_chunk())])
        result = await _run(
            streamed, events=evs, agent_total_timeout=5.0, model_idle_timeout=1.0
        )
        assert result is streamed
        assert evs.events == []  # 正常完成不写审计


# ---------------------------------------------------------------------------
# 6) 旧配置兼容（run_agent_streamed 参数层）
# ---------------------------------------------------------------------------


class TestLegacyCompatibility:
    """deprecated 参数映射：idle_timeout / total_timeout → 新阶段窗口。"""

    async def test_legacy_idle_timeout_maps_first_event(self):
        streamed = ScriptedStreamedResult([("hang",)])
        with pytest.raises(asyncio.TimeoutError, match=r"first_event idle timeout") as ei:
            await _run(streamed, idle_timeout=0.05, total_timeout=5.0)
        assert "configured=0.05" in str(ei.value)

    async def test_legacy_idle_timeout_maps_model_phase(self):
        streamed = ScriptedStreamedResult([("event", raw_chunk()), ("hang",)])
        with pytest.raises(asyncio.TimeoutError, match=r"model idle timeout") as ei:
            await _run(streamed, idle_timeout=0.05, total_timeout=5.0)
        assert "configured=0.05" in str(ei.value)

    async def test_legacy_total_timeout_maps_agent_budget(self):
        streamed = ScriptedStreamedResult([("repeat", raw_chunk(), 0.02)])
        with pytest.raises(asyncio.TimeoutError, match=r"total timeout") as ei:
            await _run(streamed, idle_timeout=1.0, total_timeout=0.1)
        assert "agent budget 0.1s exceeded" in str(ei.value)

    async def test_new_params_override_legacy(self):
        # first_event_timeout 显式给出时优先于 idle_timeout
        streamed = ScriptedStreamedResult([("hang",)])
        with pytest.raises(asyncio.TimeoutError) as ei:
            await _run(
                streamed,
                idle_timeout=0.05,
                first_event_timeout=0.2,
                total_timeout=5.0,
            )
        assert "configured=0.2" in str(ei.value)

    async def test_legacy_idle_window_never_leaks_into_tool_phase(self):
        # 关键修复：旧空闲窗口（idle_timeout）不得作用于工具执行阶段——
        # 工具阶段使用显式 tool_execution_timeout（0.5），而非旧窗口 0.05
        streamed = ScriptedStreamedResult(
            [("event", item_event("tool_called")), ("hang",)]
        )
        started = time.monotonic()
        with pytest.raises(
            asyncio.TimeoutError, match=r"tool_execution idle timeout"
        ) as ei:
            await _run(
                streamed,
                idle_timeout=0.05,
                tool_execution_timeout=0.5,
                total_timeout=5.0,
            )
        elapsed = time.monotonic() - started
        assert "configured=0.5" in str(ei.value)
        assert elapsed >= 0.4  # 旧 0.05s 窗口若误用于工具阶段会在此前误杀

    async def test_tool_phase_uses_new_default_when_only_legacy_given(self):
        # 只给旧 idle_timeout 时，工具阶段应使用新默认值（180s），
        # 不继承旧窗口。用短总预算验证超时不来自旧 0.05 空闲窗口。
        streamed = ScriptedStreamedResult(
            [("event", item_event("tool_called")), ("hang",)]
        )
        with pytest.raises(asyncio.TimeoutError, match=r"total timeout"):
            await _run(streamed, idle_timeout=0.05, total_timeout=0.2)


# ---------------------------------------------------------------------------
# 7) scheduler 外层硬兜底
# ---------------------------------------------------------------------------


def _spec(subtask_id: str, *, max_attempts: int = 1) -> SubtaskSpec:
    return SubtaskSpec(
        subtask_id=subtask_id,
        title=subtask_id,
        instructions=f"do {subtask_id}",
        role="code",
        dependencies=[],
        allowed_tools=[],
        max_attempts=max_attempts,
    )


class TestSchedulerOuterBoundary:
    """外层 asyncio.wait_for = supervisor_total_timeout + 清理宽限（硬兜底）。"""

    def test_default_boundary_is_supervisor_budget_plus_grace(self):
        async def noop(task_spec, agent):
            return ExecutionResult(subtask_id=task_spec.subtask_id, success=True)

        settings = Settings()
        scheduler = SubtaskDAGScheduler(noop, settings=settings)
        resolved = scheduler._resolve_attempt_timeout()
        assert resolved == 900.0 + _ATTEMPT_TIMEOUT_CLEANUP_GRACE
        # 外层必须不短于主管总预算（旧实现为 agent_total + 宽限 610s，
        # 嵌套子 Agent 场景会先于主管 900s 总预算触发）
        assert resolved > settings.supervisor_total_timeout

    def test_boundary_tracks_custom_supervisor_budget(self):
        async def noop(task_spec, agent):
            return ExecutionResult(subtask_id=task_spec.subtask_id, success=True)

        scheduler = SubtaskDAGScheduler(
            noop, settings=Settings(supervisor_total_timeout=1234.0)
        )
        assert (
            scheduler._resolve_attempt_timeout()
            == 1234.0 + _ATTEMPT_TIMEOUT_CLEANUP_GRACE
        )

    def test_boundary_covers_large_nested_agent_budget(self):
        async def noop(task_spec, agent):
            return ExecutionResult(subtask_id=task_spec.subtask_id, success=True)

        settings = Settings(agent_total_timeout=1200.0)
        scheduler = SubtaskDAGScheduler(noop, settings=settings)
        # 嵌套窗口派生 = 1200 + 10；主管总预算保底提升到 1210；外层 = 1220
        assert settings.subagent_tool_execution_timeout == 1210.0
        assert settings.supervisor_total_timeout == 1210.0
        assert (
            scheduler._resolve_attempt_timeout()
            == 1210.0 + _ATTEMPT_TIMEOUT_CLEANUP_GRACE
        )

    def test_explicit_attempt_timeout_wins(self):
        async def noop(task_spec, agent):
            return ExecutionResult(subtask_id=task_spec.subtask_id, success=True)

        scheduler = SubtaskDAGScheduler(noop, attempt_timeout=42.0)
        assert scheduler._resolve_attempt_timeout() == 42.0

    async def test_executor_survives_past_inner_budget_within_grace(self):
        # 执行器耗时 0.4s > agent_total_timeout 0.2s，但远小于外层兜底
        # （主管总预算 900 + 10s 宽限）→ 外层不得提前杀死
        async def slow_executor(task_spec, agent):
            await asyncio.sleep(0.4)
            return ExecutionResult(
                subtask_id=task_spec.subtask_id, success=True, output="done"
            )

        scheduler = SubtaskDAGScheduler(
            slow_executor,
            settings=Settings(
                max_agent_attempts=1,
                max_replacement_requests=0,
                agent_total_timeout=0.2,
            ),
        )
        results = await scheduler.schedule("t1", [_spec("s1")])
        assert results["s1"].success is True

    async def test_outer_hard_fallback_still_enforced(self):
        # 外层硬兜底仍然存在：执行器超过显式 attempt_timeout 被终止为失败
        async def hang_executor(task_spec, agent):
            await asyncio.sleep(5)
            return ExecutionResult(subtask_id=task_spec.subtask_id, success=True)

        scheduler = SubtaskDAGScheduler(
            hang_executor,
            settings=Settings(max_agent_attempts=1, max_replacement_requests=0),
            attempt_timeout=0.15,
        )
        results = await scheduler.schedule("t1", [_spec("s1")])
        assert results["s1"].success is False
        assert "单次执行超时" in (results["s1"].error or "")


# ---------------------------------------------------------------------------
# 8) 权限生命周期不回归
# ---------------------------------------------------------------------------


class TestPermissionLifecycleUntouched:
    """超时链路不得影响授权有效期与授权状态（任务 31 边界）。"""

    async def test_timeout_does_not_touch_grant_lifecycle(self):
        gateway = PermissionGateway()
        expires = datetime.now(timezone.utc) + timedelta(seconds=3600)
        grant = gateway.issue_grant(
            "t-1",
            "agent-1",
            ["read_file"],
            issued_by=DEFAULT_SUPERVISOR_ID,
            expires_at=expires,
        )
        # 运行一次工具执行阶段超时（缩放窗口）
        streamed = ScriptedStreamedResult(
            [("event", item_event("tool_called")), ("hang",)]
        )
        with pytest.raises(asyncio.TimeoutError):
            await _run(
                streamed,
                tool_execution_timeout=0.05,
                agent_total_timeout=1.0,
            )
        assert streamed.cancelled is True
        # 授权生命周期不受超时影响：记录未变、未撤销、过期时间未延长、仍有效
        stored = gateway.get_grant(grant.grant_id)
        assert stored is grant
        assert stored.revoked is False
        assert stored.expires_at == expires
        assert gateway.verify_grant("agent-1", "read_file", "t-1") is True

    async def test_agent_total_timeout_not_applied_to_grant_expiry(self):
        # agent_total_timeout 是执行预算，不得被用作授权过期时间来源
        gateway = PermissionGateway()
        grant = gateway.issue_grant(
            "t-2",
            "agent-2",
            ["write_file"],
            issued_by=DEFAULT_SUPERVISOR_ID,
        )
        assert grant.expires_at is None  # 未显式指定 → 不过期
        streamed = ScriptedStreamedResult([("hang",)])
        with pytest.raises(asyncio.TimeoutError):
            await _run(
                streamed,
                first_event_timeout=0.05,
                agent_total_timeout=0.1,
            )
        stored = gateway.get_grant(grant.grant_id)
        assert stored is not None
        assert stored.expires_at is None
        assert stored.revoked is False

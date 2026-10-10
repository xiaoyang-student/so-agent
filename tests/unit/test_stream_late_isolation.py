"""单元测试：run_agent_streamed 流关闭与迟到结果隔离（任务 45）。

覆盖：

1. 超时收尾路径：先 ``streamed.cancel()``，再在有限 cleanup grace 内
   ``await iterator.aclose()``（生成器 finally 观察关闭），随后抛出
   携带 phase 的 ``asyncio.TimeoutError``（既有降级语义不变）；
2. 关闭挂起超过 ``cleanup_grace_timeout``：不再等待，原超时照常抛出
   （迟到结果由调用方停止门隔离）；
3. ``tool_timeout_resolver``：code_agent 工具使用长窗口，不再被普通
   工具窗口误杀；普通工具仍按普通窗口超时；
4. 长 profile 窗口仍被 ``agent_total_timeout`` 绝对截断——总预算绝不
   被 profile 延长（任务 45 不变量）。
"""

from __future__ import annotations

import asyncio

import pytest
from agents.items import ToolCallItem, ToolCallOutputItem
from agents.stream_events import RawResponsesStreamEvent, RunItemStreamEvent

from so_agent.config import Settings
from so_agent.workflow_support import make_tool_timeout_resolver, run_agent_streamed


def raw_chunk(label: str = "chunk") -> RawResponsesStreamEvent:
    """构造 SDK 原生模型流事件。"""
    return RawResponsesStreamEvent(data={"label": label})


class _AgentStub:
    """SDK RunItemBase 需要 weakref-able 的 agent 替身。"""


def item_event(
    name: str, *, tool_name: str | None = None, call_id: str | None = None
) -> RunItemStreamEvent:
    """构造 SDK 语义运行项事件（与生产识别路径一致：真实工具项）。"""
    if name in {"tool_called", "tool_search_called"}:
        raw: dict = {}
        if tool_name is not None:
            raw["name"] = tool_name
        if call_id is not None:
            raw["call_id"] = call_id
        item: object = ToolCallItem(agent=_AgentStub(), raw_item=raw)
    elif name in {"tool_output", "tool_search_output_created"}:
        raw = {"call_id": call_id} if call_id is not None else {}
        item = ToolCallOutputItem(agent=_AgentStub(), raw_item=raw, output="ok")
    else:
        item = object()
    return RunItemStreamEvent(name=name, item=item)


AGENT = object()


class _Runner:
    """假 Runner：返回预置的流式结果。"""

    def __init__(self, streamed) -> None:
        self._streamed = streamed

    def run_streamed(self, agent, input=None, *, max_turns=None, **kwargs):
        return self._streamed


class TrackingStreamed:
    """流式结果替身：脚本化产出事件；cancel 与 aclose 可观察。

    生成器 ``finally`` 块在 iterator.aclose() 时执行 → ``closed`` 置真。
    """

    def __init__(self, script) -> None:
        self._script = list(script)
        self.final_output = "done"
        self.raw_responses: list = []
        self.cancelled = False
        self.closed = False
        self.consumed = 0

    def cancel(self, mode: str = "immediate") -> None:
        self.cancelled = True

    async def stream_events(self):
        try:
            for step in self._script:
                kind = step[0]
                if kind == "sleep":
                    await asyncio.sleep(step[1])
                elif kind == "event":
                    self.consumed += 1
                    yield step[1]
                else:  # ("hang",)
                    await asyncio.Event().wait()
        finally:
            self.closed = True


class TestStreamCloseOnTimeout:
    """超时收尾：cancel → 有限宽限内 aclose → 抛 TimeoutError。"""

    async def test_aclose_completed_within_grace_on_total_timeout(self):
        streamed = TrackingStreamed([("event", raw_chunk()), ("hang",)])
        with pytest.raises(asyncio.TimeoutError, match="total timeout"):
            await run_agent_streamed(
                AGENT,
                input="x",
                max_turns=2,
                first_event_timeout=0.5,
                model_idle_timeout=5.0,  # 模型窗口大于总预算：由总预算触发
                agent_total_timeout=0.15,
                runner=_Runner(streamed),
            )
        assert streamed.cancelled is True  # 先 cancel
        assert streamed.closed is True  # 宽限内 aclose 完成
        assert streamed.consumed == 1  # 超时前消费的首个事件

    async def test_aclose_grace_expires_timeout_still_raised(self):
        """底层关闭挂起超过宽限：不再等待，原超时照常抛出（迟到隔离）。"""

        class SlowClose(TrackingStreamed):
            async def stream_events(self):
                try:
                    await asyncio.Event().wait()
                    yield {}
                finally:
                    try:
                        await asyncio.sleep(5.0)  # 关闭远超宽限
                    except asyncio.CancelledError:
                        pass

        streamed = SlowClose([("hang",)])
        with pytest.raises(asyncio.TimeoutError):
            await run_agent_streamed(
                AGENT,
                input="x",
                max_turns=2,
                first_event_timeout=0.05,
                agent_total_timeout=5.0,
                cleanup_grace_timeout=0.05,
                runner=_Runner(streamed),
            )
        assert streamed.cancelled is True

    async def test_aclose_called_on_phase_idle_timeout(self):
        """阶段空闲超时同样走 cancel → aclose 收尾。"""
        streamed = TrackingStreamed([("event", raw_chunk()), ("hang",)])
        with pytest.raises(asyncio.TimeoutError, match="model idle timeout"):
            await run_agent_streamed(
                AGENT,
                input="x",
                max_turns=2,
                first_event_timeout=0.5,
                model_idle_timeout=0.05,
                agent_total_timeout=5.0,
                runner=_Runner(streamed),
            )
        assert streamed.cancelled is True
        assert streamed.closed is True


class TestResolverToolWindows:
    """工具阶段窗口按 profile 解析（任务 45）。"""

    async def test_code_agent_long_window_not_killed_by_ordinary(self):
        """code_agent 执行 0.3s：普通窗口 0.05s 会误杀，profile 长窗口不杀。"""
        settings = Settings(
            _env_file=None,
            tool_execution_timeout=0.05,
            code_agent_tool_execution_timeout=5.0,
        )
        streamed = TrackingStreamed(
            [
                ("event", raw_chunk()),
                (
                    "event",
                    item_event("tool_called", tool_name="code_agent", call_id="c1"),
                ),
                ("sleep", 0.3),  # 超过普通窗口、远小于 code_agent 窗口
                ("event", item_event("tool_output", call_id="c1")),
                ("event", raw_chunk("final")),
            ]
        )
        result = await run_agent_streamed(
            AGENT,
            input="x",
            max_turns=2,
            first_event_timeout=0.5,
            model_idle_timeout=0.05,
            tool_execution_timeout=0.05,
            agent_total_timeout=5.0,
            runner=_Runner(streamed),
            tool_timeout_resolver=make_tool_timeout_resolver(settings),
        )
        assert result is streamed
        assert streamed.cancelled is False
        assert streamed.consumed == 4

    async def test_ordinary_tool_still_killed_by_short_window(self):
        """普通工具仍按 tool_execution_timeout 超时（profile 不放松普通工具）。"""
        settings = Settings(_env_file=None, tool_execution_timeout=0.05)
        streamed = TrackingStreamed(
            [
                ("event", raw_chunk()),
                (
                    "event",
                    item_event("tool_called", tool_name="read_file", call_id="c2"),
                ),
                ("hang",),
            ]
        )
        with pytest.raises(asyncio.TimeoutError, match="tool_execution idle timeout"):
            await run_agent_streamed(
                AGENT,
                input="x",
                max_turns=2,
                first_event_timeout=0.5,
                model_idle_timeout=0.05,
                tool_execution_timeout=0.05,
                agent_total_timeout=5.0,
                runner=_Runner(streamed),
                tool_timeout_resolver=make_tool_timeout_resolver(settings),
            )
        assert streamed.cancelled is True

    async def test_long_profile_window_still_truncated_by_agent_total(self):
        """长 profile 窗口被 agent_total 绝对截断：总预算绝不被 profile 延长。"""
        settings = Settings(
            _env_file=None,
            tool_execution_timeout=0.05,
            code_agent_tool_execution_timeout=50.0,  # 远大于总预算
        )
        streamed = TrackingStreamed(
            [
                ("event", raw_chunk()),
                (
                    "event",
                    item_event("tool_called", tool_name="code_agent", call_id="c3"),
                ),
                ("hang",),
            ]
        )
        with pytest.raises(asyncio.TimeoutError, match="total timeout"):
            await run_agent_streamed(
                AGENT,
                input="x",
                max_turns=2,
                first_event_timeout=0.5,
                model_idle_timeout=0.05,
                tool_execution_timeout=0.05,
                agent_total_timeout=0.2,
                runner=_Runner(streamed),
                tool_timeout_resolver=make_tool_timeout_resolver(settings),
            )
        assert streamed.cancelled is True

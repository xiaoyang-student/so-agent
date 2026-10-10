"""单元测试：流式运行辅助 ``run_agent_streamed``（Fix 1 + 任务 33 兼容）。

任务 33 起 ``run_agent_streamed`` 采用分阶段流式超时（首事件 / 模型空闲 /
工具执行 + Agent 绝对总预算）；本文件保留旧参数（``idle_timeout`` /
``total_timeout``）路径的回归验证：旧空闲窗口映射到 first_event 与 model
阶段窗口，旧总超时映射到 agent_total_timeout，超时消息仍包含
``idle timeout`` / ``total timeout`` 前缀，调用方降级语义不变。
分阶段新契约的专项用例见 ``test_staged_stream_timeouts.py``。

覆盖场景：
1. 正常流完成：stream_events 产出 3 个事件后结束 → 正常返回流式结果；
2. 持续产出不中断：事件间隔 < idle_timeout 且总时长 < total_timeout
   → 全部事件被消费，不触发任何超时（Fix 1 的核心收益）；
3. 空闲超时触发：stream_events 的 __anext__ 永不返回 → 映射后的
   first_event 窗口耗尽后抛 asyncio.TimeoutError（消息含 idle timeout）；
4. 兜底总超时触发：事件持续产出但累计超过 total_timeout → 抛
   asyncio.TimeoutError（消息含 total timeout）；
5. 超时收尾：两种超时路径都必须调用 streamed.cancel()；
6. runner 缺省回退：runner=None 时使用模块级 Runner（monkeypatch 语义）。
"""

from __future__ import annotations

import asyncio

import pytest

from so_agent.workflow_support import run_agent_streamed


class FakeStreamedResult:
    """假 RunResultStreaming：可控节奏的 stream_events 异步迭代器。

    Args:
        events: 有限事件列表（按序产出后迭代结束）。
        gap: 相邻事件之间的等待秒数（模拟 token 产出节奏）。
        hang: True 时首个 __anext__ 永不返回（模拟 LLM 卡死无事件）。
        infinite: True 时无限产出事件（模拟永不结束的流）。
    """

    def __init__(self, *, events=(), gap=0.0, hang=False, infinite=False):
        self._events = list(events)
        self._gap = gap
        self._hang = hang
        self._infinite = infinite
        self.final_output = "fake-final-output"
        self.raw_responses = []
        self.new_items = []
        self.cancelled = False
        self.consumed = 0

    def cancel(self, mode="immediate"):
        self.cancelled = True

    async def stream_events(self):
        if self._hang:
            await asyncio.Event().wait()  # 永不产出且永不返回
            return
        index = 0
        while self._infinite or index < len(self._events):
            await asyncio.sleep(self._gap)
            self.consumed += 1
            event = self._events[index % len(self._events)] if self._events else index
            yield event
            index += 1


class FakeRunner:
    """假 Runner：run_streamed 返回预置的假流式结果并记录调用参数。"""

    def __init__(self, streamed):
        self.streamed = streamed
        self.calls: list[dict] = []

    def run_streamed(self, agent, input=None, *, max_turns=None, **kwargs):
        self.calls.append({"agent": agent, "input": input, "max_turns": max_turns})
        return self.streamed


AGENT = object()


class TestNormalCompletion:
    """正常流完成与 runner 参数透传。"""

    async def test_three_events_then_complete(self):
        streamed = FakeStreamedResult(events=[{"i": 0}, {"i": 1}, {"i": 2}])
        runner = FakeRunner(streamed)

        result = await run_agent_streamed(
            AGENT,
            input="hello",
            max_turns=3,
            idle_timeout=1.0,
            total_timeout=5.0,
            runner=runner,
        )

        # 返回流式结果对象本体，final_output / new_items 等属性可用
        assert result is streamed
        assert result.final_output == "fake-final-output"
        assert result.new_items == []
        assert streamed.consumed == 3
        assert streamed.cancelled is False
        # 调用参数完整透传给底层 run_streamed
        assert runner.calls == [{"agent": AGENT, "input": "hello", "max_turns": 3}]

    async def test_continuous_events_not_interrupted(self):
        # 事件间隔 0.02s（< idle_timeout=0.5）且总时长 < total_timeout：
        # 持续产出 token 的流不应被空闲超时误判
        streamed = FakeStreamedResult(
            events=[{"i": i} for i in range(5)], gap=0.02
        )
        runner = FakeRunner(streamed)

        result = await run_agent_streamed(
            AGENT,
            input="x",
            max_turns=1,
            idle_timeout=0.5,
            total_timeout=5.0,
            runner=runner,
        )

        assert result is streamed
        assert streamed.consumed == 5
        assert streamed.cancelled is False


class TestIdleTimeout:
    """空闲超时：连续无新事件达到 idle_timeout 秒即判定卡死。"""

    async def test_idle_timeout_raises(self):
        streamed = FakeStreamedResult(hang=True)
        runner = FakeRunner(streamed)

        with pytest.raises(asyncio.TimeoutError, match="idle timeout"):
            await run_agent_streamed(
                AGENT,
                input="x",
                max_turns=1,
                idle_timeout=0.05,
                total_timeout=5.0,
                runner=runner,
            )

    async def test_cancel_called_on_idle_timeout(self):
        streamed = FakeStreamedResult(hang=True)
        runner = FakeRunner(streamed)

        with pytest.raises(asyncio.TimeoutError):
            await run_agent_streamed(
                AGENT,
                input="x",
                max_turns=1,
                idle_timeout=0.05,
                total_timeout=5.0,
                runner=runner,
            )

        # 超时收尾：底层流被取消，避免悬挂任务泄漏
        assert streamed.cancelled is True


class TestTotalTimeout:
    """兜底总超时：即使事件持续产出，也不超过 total_timeout。"""

    async def test_total_timeout_raises(self):
        # 无限产出事件（间隔 0.02s 远小于 idle 窗口），但累计超过总超时
        streamed = FakeStreamedResult(infinite=True, gap=0.02)
        runner = FakeRunner(streamed)

        with pytest.raises(asyncio.TimeoutError, match="total timeout"):
            await run_agent_streamed(
                AGENT,
                input="x",
                max_turns=1,
                idle_timeout=1.0,
                total_timeout=0.1,
                runner=runner,
            )

        # 事件确实在持续产出（排除"其实是空闲超时"的误判）
        assert streamed.consumed >= 1

    async def test_cancel_called_on_total_timeout(self):
        streamed = FakeStreamedResult(infinite=True, gap=0.02)
        runner = FakeRunner(streamed)

        with pytest.raises(asyncio.TimeoutError):
            await run_agent_streamed(
                AGENT,
                input="x",
                max_turns=1,
                idle_timeout=1.0,
                total_timeout=0.1,
                runner=runner,
            )

        assert streamed.cancelled is True


class TestDefaultRunnerFallback:
    """runner=None 时回退到模块级 Runner（保持 monkeypatch 语义）。"""

    async def test_none_uses_module_level_runner(self, monkeypatch):
        streamed = FakeStreamedResult(events=[1])
        seen: dict = {}

        class ModuleRunner:
            @classmethod
            def run_streamed(cls, agent, input=None, *, max_turns=None, **kwargs):
                seen["args"] = (agent, input, max_turns)
                return streamed

        monkeypatch.setattr("so_agent.workflow_support.Runner", ModuleRunner)

        result = await run_agent_streamed(
            AGENT,
            input="hi",
            max_turns=2,
            idle_timeout=1.0,
            total_timeout=5.0,
        )

        assert result is streamed
        assert seen["args"] == (AGENT, "hi", 2)

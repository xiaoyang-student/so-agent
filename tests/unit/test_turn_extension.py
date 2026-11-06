"""单元测试：code_agent 回合续期（``agent_to_mcp_server``）。

覆盖场景：
1. 会话续跑：首次运行已把历史写入 Session，续轮只发简短说明，不重发
   整份载荷，也不重复执行输入装饰器；
2. Session 无历史（或 SDK 不可用）时回退为重建完整载荷 + 续轮说明；
3. 无成功工具调用的空转不续轮，``MaxTurnsExceeded`` 原样抛给上层；
4. 续轮次数有界（``code_agent_turn_extension_max``）；
5. ``turn_extension=False`` 时不续轮（不创建会话）；
6. 失败运行的部分用量经异常上的 ``partial_result`` 补记，不漏账；
7. 每次续期写 ``turn_budget_extended`` 事件（含前后上限与续跑模式）。
"""

from __future__ import annotations

import json
from types import SimpleNamespace

from agents.exceptions import MaxTurnsExceeded
from agents.items import ToolCallOutputItem
from agents.stream_events import RunItemStreamEvent
from pydantic import BaseModel

from so_agent.config import Settings
from so_agent.mcp import adapter as adapter_mod
from so_agent.mcp.adapter import agent_to_mcp_server
from so_agent.mcp.protocol import METHOD_TOOLS_CALL, MCPRequest, extract_result_text
from so_agent.runtime.events import EVENT_TURN_BUDGET_EXTENDED


class DemoInput(BaseModel):
    task_description: str = ""
    value: int


class DemoOutput(BaseModel):
    doubled: int


class _AgentStub:
    """支持 weakref 的 agent 替身（SDK RunItem 构造需要）。"""


def tool_output_event(call_id: str = "c1", output: str = '{"success": true}') -> RunItemStreamEvent:
    """构造一次成功工具输出事件（推进进度计数）。"""
    item = ToolCallOutputItem(
        agent=_AgentStub(), raw_item={"call_id": call_id}, output=output
    )
    return RunItemStreamEvent(name="tool_output", item=item)


class _FakeSession:
    """内存会话替身：``fed`` 为真表示历史已落盘。"""

    def __init__(self) -> None:
        self.fed = False

    async def get_items(self) -> list[dict]:
        return [{"role": "user", "content": "历史"}] if self.fed else []


class _Streamed:
    """假流式结果：按序产出事件后正常结束，或产出事件后抛回合上限。"""

    def __init__(
        self,
        *,
        events=(),
        raise_limit: bool = False,
        final_output=None,
        persist_items: bool = True,
    ):
        self._events = list(events)
        self._raise_limit = raise_limit
        self.persist_items = persist_items
        self.final_output = final_output if final_output is not None else DemoOutput(doubled=42)
        self.raw_responses = []
        self._state = SimpleNamespace(
            _model_responses=[SimpleNamespace(usage=SimpleNamespace(
                input_tokens=100, output_tokens=10, total_tokens=110
            ))]
        )

    async def stream_events(self):
        for event in self._events:
            yield event
        if self._raise_limit:
            raise MaxTurnsExceeded("Max turns (8) exceeded")

    def cancel(self):
        return None


class _ScriptedRunner:
    """按脚本依次返回流式结果，记录参数并模拟 SDK 的会话落盘。"""

    def __init__(self, scripted, captured):
        self._scripted = list(scripted)
        self._captured = captured

    def run_streamed(self, agent, input=None, *, max_turns=None, **kwargs):
        self._captured.append(
            {"input": input, "max_turns": max_turns, "session": kwargs.get("session")}
        )
        streamed = self._scripted.pop(0)
        session = kwargs.get("session")
        if session is not None and streamed.persist_items:
            session.fed = True
        return streamed


def _install_runner(monkeypatch, scripted, captured, sessions=None):
    monkeypatch.setattr(adapter_mod, "Runner", _ScriptedRunner(scripted, captured))
    if sessions is None:
        return
    created: list[_FakeSession] = []

    def factory():
        session = _FakeSession()
        created.append(session)
        sessions.append(session)
        return session

    monkeypatch.setattr(adapter_mod, "_make_continuation_session", factory)


def _request() -> MCPRequest:
    return MCPRequest(
        method=METHOD_TOOLS_CALL,
        params={"name": "demo", "arguments": {"value": 7, "task_description": "演示任务"}},
        id=1,
    )


async def test_resumes_same_session_without_resending_payload(monkeypatch):
    captured: list[dict] = []
    sessions: list[_FakeSession] = []
    _install_runner(
        monkeypatch,
        [
            _Streamed(events=[tool_output_event()], raise_limit=True),
            _Streamed(final_output=DemoOutput(doubled=14)),
        ],
        captured,
        sessions=sessions,
    )
    decorations: list[dict] = []

    def decorator(payload):
        decorations.append(dict(payload))
        return {**payload, "sandbox_artifact_index": f"index-{len(decorations)}"}

    usage: list = []
    events: list[tuple[str, dict]] = []
    server = agent_to_mcp_server(
        object(),
        "demo",
        "d",
        DemoInput,
        DemoOutput,
        max_turns=8,
        settings=Settings(max_model_turns=8),
        usage_recorder=usage.append,
        input_decorator=decorator,
        event_sink=lambda event_type, payload: events.append((event_type, payload)),
        turn_extension=True,
    )

    response = await server.handle_request(_request())

    assert response.ok is True
    assert json.loads(extract_result_text(response)) == {"doubled": 14}
    assert [call["max_turns"] for call in captured] == [8, 14]
    # 两次运行复用同一个会话对象
    assert len(sessions) == 1
    assert captured[0]["session"] is sessions[0]
    assert captured[1]["session"] is sessions[0]
    # 续跑只发简短说明，不重发整份载荷，也不重复装饰输入
    continuation = captured[1]["input"]
    assert isinstance(continuation, str)
    assert "[续轮]" in continuation
    assert "对话历史与工具结果均已保留" in continuation
    assert "不要重复已完成的读取、写入或执行" in continuation
    assert "先用 read_file 确认内容" not in continuation
    assert "演示任务" not in continuation
    assert len(decorations) == 1
    # 失败尝试的部分用量与成功结果各自入账
    assert len(usage) == 2
    ext = [payload for kind, payload in events if kind == EVENT_TURN_BUDGET_EXTENDED]
    assert ext == [
        {
            "tool_name": "demo",
            "previous_max_turns": 8,
            "new_max_turns": 14,
            "extension_count": 1,
            "tool_successes": 1,
            "tool_failures": 0,
            "mode": "session",
        }
    ]


async def test_falls_back_to_full_payload_when_session_is_empty(monkeypatch):
    captured: list[dict] = []
    sessions: list[_FakeSession] = []
    _install_runner(
        monkeypatch,
        [
            _Streamed(
                events=[tool_output_event()], raise_limit=True, persist_items=False
            ),
            _Streamed(final_output=DemoOutput(doubled=14)),
        ],
        captured,
        sessions=sessions,
    )
    decorations: list[dict] = []

    def decorator(payload):
        decorations.append(dict(payload))
        return {**payload, "sandbox_artifact_index": f"index-{len(decorations)}"}

    events: list[tuple[str, dict]] = []
    server = agent_to_mcp_server(
        object(),
        "demo",
        "d",
        DemoInput,
        DemoOutput,
        max_turns=8,
        settings=Settings(max_model_turns=8),
        input_decorator=decorator,
        event_sink=lambda event_type, payload: events.append((event_type, payload)),
        turn_extension=True,
    )

    response = await server.handle_request(_request())

    assert response.ok is True
    retry_payload = json.loads(captured[1]["input"])
    assert retry_payload["value"] == 7
    assert retry_payload["task_description"].startswith("演示任务")
    assert "[续轮]" in retry_payload["task_description"]
    assert "不要从头重做" in retry_payload["task_description"]
    # 兜底续轮重新注入运行时信息，避免沿用中断前的旧产物索引
    assert retry_payload["sandbox_artifact_index"] == "index-2"
    assert len(decorations) == 2
    ext = [payload for kind, payload in events if kind == EVENT_TURN_BUDGET_EXTENDED]
    assert [payload["mode"] for payload in ext] == ["restart"]


async def test_no_extension_without_successful_tool_call(monkeypatch):
    captured: list[dict] = []
    _install_runner(
        monkeypatch,
        [
            _Streamed(events=[tool_output_event(output='{"status": "error"}')], raise_limit=True),
        ],
        captured,
        sessions=[],
    )
    usage: list = []
    server = agent_to_mcp_server(
        object(),
        "demo",
        "d",
        DemoInput,
        DemoOutput,
        max_turns=8,
        settings=Settings(max_model_turns=8),
        usage_recorder=usage.append,
        turn_extension=True,
    )

    response = await server.handle_request(_request())

    assert response.ok is False
    assert "MaxTurnsExceeded" in response.error.message
    assert len(captured) == 1
    # 失败尝试的用量仍补记
    assert len(usage) == 1


async def test_extension_is_bounded_by_max(monkeypatch):
    captured: list[dict] = []
    _install_runner(
        monkeypatch,
        [
            _Streamed(events=[tool_output_event()], raise_limit=True),
            _Streamed(events=[tool_output_event()], raise_limit=True),
        ],
        captured,
        sessions=[],
    )
    server = agent_to_mcp_server(
        object(),
        "demo",
        "d",
        DemoInput,
        DemoOutput,
        max_turns=8,
        settings=Settings(
            max_model_turns=8,
            code_agent_turn_extension_budget=4,
            code_agent_turn_extension_max=1,
        ),
        turn_extension=True,
    )

    response = await server.handle_request(_request())

    assert response.ok is False
    assert "MaxTurnsExceeded" in response.error.message
    assert [call["max_turns"] for call in captured] == [8, 12]


async def test_disabled_by_default(monkeypatch):
    captured: list[dict] = []
    sessions: list[_FakeSession] = []
    _install_runner(
        monkeypatch,
        [_Streamed(events=[tool_output_event()], raise_limit=True)],
        captured,
        sessions=sessions,
    )
    server = agent_to_mcp_server(
        object(), "demo", "d", DemoInput, DemoOutput, max_turns=8
    )

    response = await server.handle_request(_request())

    assert response.ok is False
    assert "MaxTurnsExceeded" in response.error.message
    assert len(captured) == 1
    # 未开启续轮时不创建会话
    assert sessions == []
    assert captured[0]["session"] is None

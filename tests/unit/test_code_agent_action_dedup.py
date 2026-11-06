"""单元测试：code_agent 写/执行重复动作回执。

覆盖：
1. 相同内容重复写（覆盖 / 追加）返回 ``unchanged``，不再写盘；
2. 内容变化时正常写入；
3. 相同代码且工作区未变化时返回上次结果（``replayed``），最多一次；
4. 第三次重复执行被拒绝（``duplicate_execution``），避免无界重放；
5. 工作区变化后允许重新执行；
6. ``code_agent_action_dedup=False`` 时不做去重，也不写事件。
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from agents.tool_context import ToolContext

from so_agent.config import Settings
from so_agent.context import ProjectContext
from so_agent.runtime.events import EVENT_ACTION_REPLAYED
from so_agent.tool_packages.code_agent.agent import (
    build_code_tools,
    reset_read_ledger,
)


class _RecordingLogger:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict]] = []

    def log_event(self, event_type: str, payload: dict) -> None:
        self.events.append((event_type, payload))


class _FakeSandbox:
    """记录调用次数的沙箱替身；不写文件，保持工作区指纹稳定。"""

    def __init__(self) -> None:
        self.calls: list[str] = []

    async def execute(self, code, task_id, run_id, *, workdir=None):
        self.calls.append(code)
        payload = {
            "subtask_id": task_id,
            "success": True,
            "output": f"run-{len(self.calls)}",
            "exit_code": 0,
            "duration": 0.01,
        }
        return SimpleNamespace(model_dump_json=lambda: json.dumps(payload))


def _context(tool_name: str) -> ToolContext:
    return ToolContext(
        context=None,
        tool_name=tool_name,
        tool_call_id="call-1",
        tool_arguments="{}",
    )


def _settings(tmp_path: Path, **overrides) -> Settings:
    return Settings(
        _env_file=None,
        tool_store_dir=str(tmp_path / "tools"),
        **overrides,
    )


def _tools(tmp_path: Path, *, logger=None, sandbox=None, **overrides):
    context = ProjectContext(sandbox_dir=tmp_path, project_name="dedup")
    if logger is not None:
        context.event_logger = logger
    return (
        build_code_tools(
            context,
            sandbox_runner=sandbox,
            settings=_settings(tmp_path, **overrides),
        ),
        context,
    )


@pytest.fixture(autouse=True)
def _ledger():
    reset_read_ledger()
    yield
    reset_read_ledger()


async def _write(tools, **arguments) -> dict:
    raw = await tools["write_file"].on_invoke_tool(
        _context("write_file"), json.dumps(arguments)
    )
    return json.loads(raw)


async def _run(tools, code: str = "print('hi')", task_id: str = "t1") -> dict:
    raw = await tools["run_in_sandbox"].on_invoke_tool(
        _context("run_in_sandbox"), json.dumps({"code": code, "task_id": task_id})
    )
    return json.loads(raw)


class TestWriteDedup:
    async def test_identical_overwrite_is_noop(self, tmp_path):
        logger = _RecordingLogger()
        tools, _ = _tools(tmp_path, logger=logger)

        first = await _write(tools, path="notes.md", content="alpha")
        before = (tmp_path / "notes.md").stat().st_mtime_ns
        second = await _write(tools, path="notes.md", content="alpha")

        assert first["status"] == "ok" and first["created"] is True
        assert second["status"] == "ok"
        assert second["unchanged"] is True
        assert second["total_bytes"] == len("alpha")
        assert (tmp_path / "notes.md").stat().st_mtime_ns == before
        assert [kind for kind, _ in logger.events] == [EVENT_ACTION_REPLAYED]
        assert logger.events[0][1]["tool"] == "write_file"

    async def test_changed_content_still_writes(self, tmp_path):
        tools, _ = _tools(tmp_path)
        await _write(tools, path="notes.md", content="alpha")

        changed = await _write(tools, path="notes.md", content="beta")

        assert changed.get("unchanged") is not True
        assert (tmp_path / "notes.md").read_text(encoding="utf-8") == "beta"

    async def test_repeated_append_is_noop(self, tmp_path):
        tools, _ = _tools(tmp_path)
        await _write(tools, path="log.txt", content="a")
        await _write(tools, path="log.txt", content="b", append=True)

        again = await _write(tools, path="log.txt", content="b", append=True)

        assert again["unchanged"] is True
        assert (tmp_path / "log.txt").read_text(encoding="utf-8") == "ab"

    async def test_disabled_keeps_writing(self, tmp_path):
        logger = _RecordingLogger()
        tools, _ = _tools(
            tmp_path, logger=logger, code_agent_action_dedup=False
        )

        await _write(tools, path="notes.md", content="alpha")
        second = await _write(tools, path="notes.md", content="alpha")

        assert second.get("unchanged") is not True
        assert logger.events == []


class TestRunDedup:
    async def test_identical_run_is_replayed_once(self, tmp_path):
        sandbox = _FakeSandbox()
        logger = _RecordingLogger()
        tools, _ = _tools(tmp_path, logger=logger, sandbox=sandbox)

        first = await _run(tools)
        second = await _run(tools)

        assert sandbox.calls == ["print('hi')"]
        assert first.get("replayed") is not True
        assert second["replayed"] is True
        assert second["success"] is True
        assert second["output"] == first["output"]
        assert [kind for kind, _ in logger.events] == [EVENT_ACTION_REPLAYED]
        assert logger.events[0][1]["tool"] == "run_in_sandbox"

    async def test_third_identical_run_is_rejected(self, tmp_path):
        sandbox = _FakeSandbox()
        tools, _ = _tools(tmp_path, sandbox=sandbox)

        await _run(tools)
        await _run(tools)
        third = await _run(tools)

        assert sandbox.calls == ["print('hi')"]
        assert third["status"] == "error"
        assert third["duplicate_execution"] is True
        assert "请修改代码" in third["error"]

    async def test_workspace_change_allows_rerun(self, tmp_path):
        sandbox = _FakeSandbox()
        tools, _ = _tools(tmp_path, sandbox=sandbox)

        await _run(tools)
        (tmp_path / "data.csv").write_text("x\n", encoding="utf-8")
        again = await _run(tools)

        assert sandbox.calls == ["print('hi')", "print('hi')"]
        assert again.get("replayed") is not True
        assert again["output"] == "run-2"

    async def test_different_code_or_task_runs_again(self, tmp_path):
        sandbox = _FakeSandbox()
        tools, _ = _tools(tmp_path, sandbox=sandbox)

        await _run(tools)
        other_code = await _run(tools, code="print('bye')")
        other_task = await _run(tools, task_id="t2")

        assert sandbox.calls == ["print('hi')", "print('bye')", "print('hi')"]
        assert other_code.get("replayed") is not True
        assert other_task.get("replayed") is not True

    async def test_disabled_executes_every_time(self, tmp_path):
        sandbox = _FakeSandbox()
        logger = _RecordingLogger()
        tools, _ = _tools(
            tmp_path,
            logger=logger,
            sandbox=sandbox,
            code_agent_action_dedup=False,
        )

        await _run(tools)
        again = await _run(tools)

        assert sandbox.calls == ["print('hi')", "print('hi')"]
        assert again.get("replayed") is not True
        assert logger.events == []

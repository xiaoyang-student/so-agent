"""code_agent 上下文瘦身：读取窗口、diff 上限与产物索引收敛。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from agents.tool_context import ToolContext

from so_agent.config import Settings
from so_agent.context import ProjectContext
from so_agent.runtime.events import EVENT_READ_COLLAPSED
from so_agent.tool_packages.code_agent.agent import (
    _CURRENT_WRITE_SCOPE,
    _READ_LEDGER,
    _make_input_decorator,
    build_code_tools,
    reset_read_ledger,
)


def _context() -> ToolContext:
    return ToolContext(
        context=None,
        tool_name="read_file",
        tool_call_id="call-1",
        tool_arguments="{}",
    )


def _settings(tmp_path: Path, **overrides) -> Settings:
    return Settings(
        _env_file=None,
        tool_store_dir=str(tmp_path / "tools"),
        **overrides,
    )


def _tools(tmp_path: Path, **overrides):
    return build_code_tools(
        ProjectContext(sandbox_dir=tmp_path, project_name="slim"),
        settings=_settings(tmp_path, **overrides),
    )


class TestReadWindow:
    async def test_default_window_returns_next_start(self, tmp_path):
        (tmp_path / "big.py").write_text(
            "\n".join(f"line-{index}" for index in range(1, 201)),
            encoding="utf-8",
        )
        tools = _tools(tmp_path, code_agent_read_max_lines=10)
        raw = await tools["read_file"].on_invoke_tool(
            _context(), json.dumps({"path": "big.py"})
        )
        data = json.loads(raw)
        assert data["status"] == "ok"
        assert data["start_line"] == 1
        assert data["end_line"] == 10
        assert data["total_lines"] == 200
        assert data["next_start"] == 11
        assert data["content"].splitlines() == [
            f"line-{index}" for index in range(1, 11)
        ]

    async def test_start_line_continues_window(self, tmp_path):
        (tmp_path / "big.py").write_text(
            "\n".join(f"line-{index}" for index in range(1, 31)),
            encoding="utf-8",
        )
        tools = _tools(tmp_path, code_agent_read_max_lines=5)
        raw = await tools["read_file"].on_invoke_tool(
            _context(), json.dumps({"path": "big.py", "start_line": 26})
        )
        data = json.loads(raw)
        assert data["start_line"] == 26
        assert data["end_line"] == 30
        assert data["next_start"] is None
        assert len(data["content"].splitlines()) == 5

    async def test_char_budget_stops_at_line_boundary(self, tmp_path):
        (tmp_path / "wide.txt").write_text(
            "\n".join("x" * 40 for _ in range(50)), encoding="utf-8"
        )
        tools = _tools(
            tmp_path,
            code_agent_read_max_lines=50,
            code_agent_read_max_chars=100,
        )
        raw = await tools["read_file"].on_invoke_tool(
            _context(), json.dumps({"path": "wide.txt"})
        )
        data = json.loads(raw)
        assert data["end_line"] < 50
        assert len(data["content"]) <= 120
        assert data["next_start"] == data["end_line"] + 1


class TestDiffBudget:
    async def test_write_file_caps_diff_lines(self, tmp_path):
        (tmp_path / "old.py").write_text(
            "\n".join(f"old-{index}" for index in range(100)),
            encoding="utf-8",
        )
        tools = _tools(tmp_path, code_agent_tool_diff_max_lines=5)
        raw = await tools["write_file"].on_invoke_tool(
            _context(),
            json.dumps(
                {
                    "path": "old.py",
                    "content": "\n".join(
                        f"new-{index}" for index in range(100)
                    ),
                }
            ),
        )
        data = json.loads(raw)
        assert data["status"] == "ok"
        assert len(data["diff"].splitlines()) <= 6
        assert "省略" in data["diff"]


class _RecordingLogger:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict]] = []

    def log_event(self, event_type: str, payload: dict) -> None:
        self.events.append((event_type, payload))


def _dedup_tools(tmp_path: Path, *, logger=None, **overrides):
    context = ProjectContext(sandbox_dir=tmp_path, project_name="slim")
    if logger is not None:
        context.event_logger = logger
    return build_code_tools(
        context, settings=_settings(tmp_path, **overrides)
    )


class TestReadDedup:
    @pytest.fixture(autouse=True)
    def _ledger(self):
        token = _READ_LEDGER.set(None)
        reset_read_ledger()
        yield
        _READ_LEDGER.reset(token)

    async def _read(self, tools, **arguments) -> dict:
        raw = await tools["read_file"].on_invoke_tool(
            _context(), json.dumps({"path": "sample.py", **arguments})
        )
        return json.loads(raw)

    async def test_repeat_same_window_returns_receipt(self, tmp_path):
        (tmp_path / "sample.py").write_text(
            "\n".join(f"line-{index}" for index in range(1, 21)),
            encoding="utf-8",
        )
        logger = _RecordingLogger()
        tools = _dedup_tools(tmp_path, logger=logger)

        first = await self._read(tools)
        second = await self._read(tools)

        assert first["content"].splitlines()[0] == "line-1"
        assert second["status"] == "ok"
        assert second["unchanged"] is True
        assert "content" not in second
        assert second["start_line"] == 1
        assert second["end_line"] == 20
        assert [(kind, payload["path"]) for kind, payload in logger.events] == [
            (EVENT_READ_COLLAPSED, "sample.py")
        ]

    async def test_changed_file_returns_content_again(self, tmp_path):
        target = tmp_path / "sample.py"
        target.write_text("alpha\nbeta\n", encoding="utf-8")
        tools = _dedup_tools(tmp_path)

        await self._read(tools)
        target.write_text("alpha\ngamma\n", encoding="utf-8")
        again = await self._read(tools)

        assert again.get("unchanged") is not True
        assert "gamma" in again["content"]

    async def test_force_bypasses_collapse(self, tmp_path):
        (tmp_path / "sample.py").write_text("alpha\n", encoding="utf-8")
        tools = _dedup_tools(tmp_path)

        await self._read(tools)
        forced = await self._read(tools, force=True)

        assert forced.get("unchanged") is not True
        assert forced["content"] == "alpha"

    async def test_subset_of_read_window_collapses(self, tmp_path):
        (tmp_path / "sample.py").write_text(
            "\n".join(f"line-{index}" for index in range(1, 31)),
            encoding="utf-8",
        )
        tools = _dedup_tools(tmp_path)

        await self._read(tools)
        subset = await self._read(tools, start_line=5, max_lines=3)

        assert subset["unchanged"] is True
        assert subset["start_line"] == 5
        assert subset["end_line"] == 7

    async def test_window_beyond_read_range_returns_content(self, tmp_path):
        (tmp_path / "sample.py").write_text(
            "\n".join(f"line-{index}" for index in range(1, 31)),
            encoding="utf-8",
        )
        tools = _dedup_tools(tmp_path)

        await self._read(tools, max_lines=5)
        wider = await self._read(tools, max_lines=10)

        assert wider.get("unchanged") is not True
        assert wider["end_line"] == 10

    async def test_no_ledger_never_collapses(self, tmp_path):
        (tmp_path / "sample.py").write_text("alpha\n", encoding="utf-8")
        tools = _dedup_tools(tmp_path)
        _READ_LEDGER.set(None)

        first = await self._read(tools)
        second = await self._read(tools)

        assert first["content"] == "alpha"
        assert second["content"] == "alpha"

    async def test_reset_between_runs_returns_content(self, tmp_path):
        (tmp_path / "sample.py").write_text("alpha\n", encoding="utf-8")
        tools = _dedup_tools(tmp_path)

        await self._read(tools)
        reset_read_ledger()
        again = await self._read(tools)

        assert again.get("unchanged") is not True
        assert again["content"] == "alpha"

    async def test_truncated_output_is_not_marked_read(self, tmp_path):
        (tmp_path / "sample.py").write_text(
            "\n".join("x" * 50 for _ in range(20)), encoding="utf-8"
        )
        tools = _dedup_tools(
            tmp_path,
            tool_output_max_chars=50,
            code_agent_read_max_chars=5000,
        )

        first = await self._read(tools)
        second = await self._read(tools)

        assert "已截断" in first["content"]
        assert len(first["content"]) < 200
        assert second.get("unchanged") is not True
        assert second["content"]

    async def test_disabled_by_settings(self, tmp_path):
        (tmp_path / "sample.py").write_text("alpha\n", encoding="utf-8")
        tools = _dedup_tools(tmp_path, code_agent_read_dedup=False)

        first = await self._read(tools)
        second = await self._read(tools)

        assert first.get("unchanged") is not True
        assert second.get("unchanged") is not True

    def test_input_decorator_resets_ledger(self, tmp_path):
        decorator = _make_input_decorator(
            ProjectContext(sandbox_dir=tmp_path, project_name="slim"),
            _settings(tmp_path),
        )
        _READ_LEDGER.set({"stale.py": {"sha256": "x", "ranges": [(1, 5)]}})

        decorator({"write_scope": []})

        assert _READ_LEDGER.get() == {}


class TestArtifactIndex:
    def test_index_is_scoped_and_capped(self, tmp_path):
        for name in ("a.txt", "b.txt", "c.txt", "target.py"):
            (tmp_path / name).write_text(name, encoding="utf-8")
        decorator = _make_input_decorator(
            ProjectContext(sandbox_dir=tmp_path, project_name="slim"),
            _settings(tmp_path, artifact_index_max_entries=2),
        )
        token = _CURRENT_WRITE_SCOPE.set(())
        try:
            decorated = decorator({"write_scope": ["target.py"]})
        finally:
            _CURRENT_WRITE_SCOPE.reset(token)
        assert decorated is not None
        index = decorated["sandbox_artifact_index"]
        assert "target.py" in index
        assert index.count("- ") <= 2

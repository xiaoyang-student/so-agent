"""code_agent 写入工具：单次写入上限、追加写入与写入+执行合并。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from agents.tool_context import ToolContext

from so_agent.config import Settings
from so_agent.context import ProjectContext
from so_agent.models import ExecutionResult
from so_agent.tool_packages.code_agent.agent import (
    _CURRENT_WRITE_SCOPE,
    _write_text_file,
    build_code_tools,
)


class _RecordingSandbox:
    backend_name = "recording"
    isolation_level = "none"

    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def execute(self, code, task_id, run_id, *, workdir=None):
        self.calls.append(
            {"code": code, "task_id": task_id, "run_id": run_id, "workdir": workdir}
        )
        return ExecutionResult(
            subtask_id=task_id,
            success=True,
            output="ran",
            metadata={"exit_code": 0},
        )


def _context(root: Path) -> ToolContext:
    return ToolContext(
        context=None,
        tool_name="write_file",
        tool_call_id="call-1",
        tool_arguments="{}",
    )


def _settings(tmp_path: Path, **overrides) -> Settings:
    return Settings(
        _env_file=None,
        tool_store_dir=str(tmp_path / "tools"),
        **overrides,
    )


class TestWriteTextFile:
    def test_creates_then_appends(self, tmp_path):
        first = _write_text_file(
            tmp_path, "out/a.py", "print('a')\n", append=False, max_bytes=0
        )
        assert first["created"] is True
        assert first["appended"] is False

        second = _write_text_file(
            tmp_path, "out/a.py", "print('b')\n", append=True, max_bytes=0
        )
        assert second["created"] is False
        assert second["appended"] is True
        assert (tmp_path / "out" / "a.py").read_text(encoding="utf-8") == (
            "print('a')\nprint('b')\n"
        )

    def test_over_limit_is_rejected_without_writing(self, tmp_path):
        with pytest.raises(ValueError, match="单次写入上限"):
            _write_text_file(
                tmp_path, "big.py", "0123456789", append=False, max_bytes=5
            )
        assert not (tmp_path / "big.py").exists()

    def test_append_counts_only_new_content(self, tmp_path):
        _write_text_file(tmp_path, "a.py", "123456", append=False, max_bytes=6)
        with pytest.raises(ValueError, match="单次写入上限"):
            _write_text_file(
                tmp_path, "a.py", "7890", append=True, max_bytes=3
            )


class TestWriteAndRun:
    async def test_single_call_writes_then_executes_full_file(self, tmp_path):
        sandbox = _RecordingSandbox()
        tools = build_code_tools(
            ProjectContext(sandbox_dir=tmp_path, project_name="write-run"),
            sandbox_runner=sandbox,
            settings=_settings(tmp_path),
        )
        payload = json.dumps(
            {"path": "run.py", "content": "print('hi')\n", "task_id": "t-1"}
        )
        raw = await tools["write_and_run"].on_invoke_tool(_context(tmp_path), payload)
        result = json.loads(raw)

        assert result["status"] == "ok"
        assert result["created"] is True
        assert result["execution_result"]["success"] is True
        assert (tmp_path / "run.py").read_text(encoding="utf-8") == "print('hi')\n"
        assert len(sandbox.calls) == 1
        assert sandbox.calls[0]["code"] == "print('hi')\n"
        assert sandbox.calls[0]["workdir"] == tmp_path

    async def test_append_then_run_executes_combined_content(self, tmp_path):
        (tmp_path / "run.py").write_text("line1\n", encoding="utf-8")
        sandbox = _RecordingSandbox()
        tools = build_code_tools(
            ProjectContext(sandbox_dir=tmp_path, project_name="write-run"),
            sandbox_runner=sandbox,
            settings=_settings(tmp_path),
        )
        payload = json.dumps(
            {
                "path": "run.py",
                "content": "line2\n",
                "task_id": "t-1",
                "append": True,
            }
        )
        result = json.loads(
            await tools["write_and_run"].on_invoke_tool(_context(tmp_path), payload)
        )
        assert result["appended"] is True
        assert sandbox.calls[0]["code"] == "line1\nline2\n"

    async def test_over_limit_write_is_not_executed(self, tmp_path):
        sandbox = _RecordingSandbox()
        tools = build_code_tools(
            ProjectContext(sandbox_dir=tmp_path, project_name="write-run"),
            sandbox_runner=sandbox,
            settings=_settings(tmp_path, code_agent_write_max_bytes=4),
        )
        payload = json.dumps(
            {"path": "run.py", "content": "print('too long')\n", "task_id": "t-1"}
        )
        result = json.loads(
            await tools["write_and_run"].on_invoke_tool(_context(tmp_path), payload)
        )
        assert result["status"] == "error"
        assert "单次写入上限" in result["error"]
        assert sandbox.calls == []
        assert not (tmp_path / "run.py").exists()

    async def test_write_file_reports_budget_error(self, tmp_path):
        tools = build_code_tools(
            ProjectContext(sandbox_dir=tmp_path, project_name="write-run"),
            sandbox_runner=_RecordingSandbox(),
            settings=_settings(tmp_path, code_agent_write_max_bytes=4),
        )
        payload = json.dumps({"path": "a.py", "content": "#012345"})
        result = json.loads(
            await tools["write_file"].on_invoke_tool(_context(tmp_path), payload)
        )
        assert result["status"] == "error"
        assert "单次写入上限" in result["error"]


class TestPythonSyntaxCheck:
    @staticmethod
    def _tools(tmp_path: Path) -> dict:
        return build_code_tools(
            ProjectContext(sandbox_dir=tmp_path, project_name="syntax"),
            sandbox_runner=_RecordingSandbox(),
            settings=_settings(tmp_path),
        )

    async def test_write_file_rejects_invalid_python_without_writing(
        self, tmp_path
    ):
        tools = self._tools(tmp_path)
        payload = json.dumps(
            {"path": "broken.py", "content": "def broken(:\n"}
        )

        result = json.loads(
            await tools["write_file"].on_invoke_tool(_context(tmp_path), payload)
        )

        assert result["status"] == "error"
        assert result["syntax_error"] is True
        assert result["line"] == 1
        assert "语法检查" in result["error"]
        assert not (tmp_path / "broken.py").exists()

    async def test_write_and_run_rejects_invalid_python_without_execution(
        self, tmp_path
    ):
        sandbox = _RecordingSandbox()
        tools = build_code_tools(
            ProjectContext(sandbox_dir=tmp_path, project_name="syntax"),
            sandbox_runner=sandbox,
            settings=_settings(tmp_path),
        )
        payload = json.dumps(
            {
                "path": "bad.py",
                "content": "print('x')\nif True\n",
                "task_id": "t-1",
            }
        )

        result = json.loads(
            await tools["write_and_run"].on_invoke_tool(
                _context(tmp_path), payload
            )
        )

        assert result["status"] == "error"
        assert result["syntax_error"] is True
        assert sandbox.calls == []
        assert not (tmp_path / "bad.py").exists()

    async def test_append_checks_combined_content(self, tmp_path):
        (tmp_path / "part.py").write_text("print('a')\n", encoding="utf-8")
        tools = self._tools(tmp_path)
        payload = json.dumps(
            {
                "path": "part.py",
                "content": "def broken(:\n",
                "append": True,
            }
        )

        result = json.loads(
            await tools["write_file"].on_invoke_tool(_context(tmp_path), payload)
        )

        assert result["status"] == "error"
        assert (tmp_path / "part.py").read_text(encoding="utf-8") == (
            "print('a')\n"
        )

    async def test_non_python_file_skips_syntax_check(self, tmp_path):
        tools = self._tools(tmp_path)
        payload = json.dumps(
            {"path": "notes.md", "content": "def broken(:\n"}
        )

        result = json.loads(
            await tools["write_file"].on_invoke_tool(_context(tmp_path), payload)
        )

        assert result["status"] == "ok"
        assert (tmp_path / "notes.md").is_file()


class TestWriteScopeEnforcement:
    @staticmethod
    def _tools(tmp_path: Path) -> dict:
        return build_code_tools(
            ProjectContext(sandbox_dir=tmp_path, project_name="scope"),
            sandbox_runner=_RecordingSandbox(),
            settings=_settings(tmp_path),
        )

    async def test_write_file_rejects_path_outside_scope(self, tmp_path):
        tools = self._tools(tmp_path)
        token = _CURRENT_WRITE_SCOPE.set(("out/report.md",))
        try:
            payload = json.dumps({"path": "other.md", "content": "x"})
            result = json.loads(
                await tools["write_file"].on_invoke_tool(_context(tmp_path), payload)
            )
        finally:
            _CURRENT_WRITE_SCOPE.reset(token)
        assert result["status"] == "error"
        assert "不在本子任务声明的写入范围内" in result["error"]
        assert not (tmp_path / "other.md").exists()

    async def test_write_file_allows_scope_directory_prefix(self, tmp_path):
        tools = self._tools(tmp_path)
        token = _CURRENT_WRITE_SCOPE.set(("out",))
        try:
            payload = json.dumps({"path": "out/report.md", "content": "ok"})
            result = json.loads(
                await tools["write_file"].on_invoke_tool(_context(tmp_path), payload)
            )
        finally:
            _CURRENT_WRITE_SCOPE.reset(token)
        assert result["status"] == "ok"
        assert (tmp_path / "out" / "report.md").read_text(encoding="utf-8") == "ok"

    async def test_write_and_run_rejected_without_execution(self, tmp_path):
        sandbox = _RecordingSandbox()
        tools = build_code_tools(
            ProjectContext(sandbox_dir=tmp_path, project_name="scope"),
            sandbox_runner=sandbox,
            settings=_settings(tmp_path),
        )
        token = _CURRENT_WRITE_SCOPE.set(("out/*.py",))
        try:
            payload = json.dumps(
                {"path": "run.py", "content": "print(1)\n", "task_id": "t-1"}
            )
            result = json.loads(
                await tools["write_and_run"].on_invoke_tool(
                    _context(tmp_path), payload
                )
            )
        finally:
            _CURRENT_WRITE_SCOPE.reset(token)
        assert result["status"] == "error"
        assert sandbox.calls == []
        assert not (tmp_path / "run.py").exists()

    async def test_write_and_run_allows_glob_match(self, tmp_path):
        sandbox = _RecordingSandbox()
        tools = build_code_tools(
            ProjectContext(sandbox_dir=tmp_path, project_name="scope"),
            sandbox_runner=sandbox,
            settings=_settings(tmp_path),
        )
        token = _CURRENT_WRITE_SCOPE.set(("out/*.py",))
        try:
            payload = json.dumps(
                {"path": "out/run.py", "content": "print(1)\n", "task_id": "t-1"}
            )
            result = json.loads(
                await tools["write_and_run"].on_invoke_tool(
                    _context(tmp_path), payload
                )
            )
        finally:
            _CURRENT_WRITE_SCOPE.reset(token)
        assert result["status"] == "ok"
        assert len(sandbox.calls) == 1

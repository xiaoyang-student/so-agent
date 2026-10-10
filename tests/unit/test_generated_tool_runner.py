"""单元测试：生成工具沙箱运行器 + 路径穿越防护。

覆盖：工具在沙箱子进程执行、返回值强制 JSON、正确答案比对基础、
入口缺失误报、以及生成工具入口的目录边界校验（防路径穿越）。
"""

from __future__ import annotations

import json

import pytest

from so_agent.mcp.adapter import MCPAdapterError, generated_tool_to_mcp_server
from so_agent.models import GeneratedToolManifest
from so_agent.runtime.generated_tool_runner import (
    GeneratedToolExecutionError,
    build_runner_code,
    parse_runner_output,
    run_generated_tool_in_sandbox,
)
from so_agent.runtime.sandbox import SandboxRunner

SENTINEL = "__SO_AGENT_TOOL_RESULT__"


def _write_tool(tmp_path, code: str):
    path = tmp_path / "demo_tool.py"
    path.write_text(code, encoding="utf-8")
    return path


def _manifest(name: str = "demo_tool", entry_file: str = "demo_tool.py"):
    return GeneratedToolManifest(
        tool_id="t-1",
        tool_name=name,
        entry_file=entry_file,
        created_by_task="task-1",
    )


class TestBuildRunnerCode:
    def test_generates_executable_source_with_sentinel(self, tmp_path):
        code = build_runner_code(tmp_path / "x.py", "x", {"a": 1})
        assert "import importlib.util" in code
        assert SENTINEL in code
        # 参数经 json 内嵌并还原，避免注入
        assert "json.loads" in code

    def test_argument_injection_is_neutralized(self, tmp_path):
        evil = {"x": "'); import os; os.system('echo pwned'); #"}
        code = build_runner_code(tmp_path / "x.py", "x", evil)
        # 恶意串被 repr/json 安全包裹为数据：runner 语法仍合法，不被当作代码执行
        compile(code, "<runner>", "exec")


class TestParseRunnerOutput:
    def test_parse_ok_result(self):
        out = f"noise\n{SENTINEL}{json.dumps({'ok': True, 'result': {'v': 1}})}"
        assert parse_runner_output(out) == {"v": 1}

    def test_parse_error_raises(self):
        out = f"{SENTINEL}{json.dumps({'ok': False, 'error': 'Boom'})}"
        with pytest.raises(GeneratedToolExecutionError):
            parse_runner_output(out)

    def test_missing_sentinel_raises(self):
        with pytest.raises(GeneratedToolExecutionError):
            parse_runner_output("no sentinel here")

    def test_invalid_json_raises(self):
        with pytest.raises(GeneratedToolExecutionError):
            parse_runner_output(f"{SENTINEL}not-json")


class TestRunInSandbox:
    async def test_runs_tool_and_returns_json(self, tmp_path):
        tool = _write_tool(tmp_path, "def run(x):\n    return {'double': x * 2}\n")
        sandbox = SandboxRunner(sandbox_dir=tmp_path / "sb")
        result = await run_generated_tool_in_sandbox(
            sandbox, tool, "demo_tool", {"x": 3}
        )
        assert result == {"double": 6}

    async def test_non_json_return_raises(self, tmp_path):
        tool = _write_tool(tmp_path, "def run():\n    return object()\n")
        sandbox = SandboxRunner(sandbox_dir=tmp_path / "sb")
        with pytest.raises(GeneratedToolExecutionError):
            await run_generated_tool_in_sandbox(sandbox, tool, "demo_tool", {})

    async def test_missing_entry_raises(self, tmp_path):
        tool = _write_tool(tmp_path, "VALUE = 1\n")
        sandbox = SandboxRunner(sandbox_dir=tmp_path / "sb")
        with pytest.raises(GeneratedToolExecutionError):
            await run_generated_tool_in_sandbox(sandbox, tool, "demo_tool", {})

    async def test_wrong_arguments_raises(self, tmp_path):
        tool = _write_tool(tmp_path, "def run(x):\n    return x\n")
        sandbox = SandboxRunner(sandbox_dir=tmp_path / "sb")
        with pytest.raises(GeneratedToolExecutionError):
            await run_generated_tool_in_sandbox(sandbox, tool, "demo_tool", {})


class TestPathTraversal:
    def test_out_of_bounds_entry_rejected(self, tmp_path):
        root = tmp_path / "gen"
        root.mkdir()
        outside = tmp_path / "evil.py"
        outside.write_text("def run():\n    return 1\n", encoding="utf-8")
        with pytest.raises(MCPAdapterError):
            generated_tool_to_mcp_server(
                outside, _manifest(entry_file="evil.py"), generated_tools_root=root
            )

    def test_in_bounds_entry_accepted(self, tmp_path):
        root = tmp_path / "gen"
        root.mkdir()
        inside = root / "demo_tool.py"
        inside.write_text("def run():\n    return 1\n", encoding="utf-8")
        server = generated_tool_to_mcp_server(
            inside, _manifest(), generated_tools_root=root
        )
        assert server.name == "demo_tool"

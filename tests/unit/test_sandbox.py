"""单元测试：验证型沙箱 SandboxRunner（真实子进程执行）。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from so_agent.config import Settings
from so_agent.runtime.sandbox import SandboxError, SandboxRunner

PRINT_HELLO = "print('hello sandbox')"
PRINT_BOTH = (
    "import sys\n"
    "print('to-stdout')\n"
    "print('to-stderr', file=sys.stderr)\n"
)
EXIT_NONZERO = (
    "import sys\n"
    "print('boom-message', file=sys.stderr)\n"
    "sys.exit(3)\n"
)
SLEEP_LONG = "import time\ntime.sleep(60)\n"


def make_runner(tmp_path: Path, *, timeout: float = 30.0, output_limit: int = 1_048_576) -> SandboxRunner:
    return SandboxRunner(
        sandbox_dir=tmp_path,
        settings=Settings(sandbox_timeout=timeout, sandbox_output_limit=output_limit),
    )


class TestBasicExecution:
    async def test_successful_run(self, tmp_path):
        runner = make_runner(tmp_path)
        result = await runner.execute(PRINT_HELLO, "t1", "run1")
        assert result.success is True
        assert result.error is None
        assert "hello sandbox" in result.output
        assert result.subtask_id == "run1"
        assert result.duration > 0

    async def test_code_written_and_executed_in_run_dir(self, tmp_path):
        runner = make_runner(tmp_path)
        await runner.execute(PRINT_HELLO, "t1", "run1")
        main_py = tmp_path / "task_t1" / "run_run1" / "main.py"
        assert main_py.is_file()
        assert main_py.read_text(encoding="utf-8") == PRINT_HELLO

    async def test_stdout_and_stderr_captured(self, tmp_path):
        runner = make_runner(tmp_path)
        result = await runner.execute(PRINT_BOTH, "t1", "run1")
        assert result.success is True
        assert "to-stdout" in result.output
        assert "to-stderr" not in result.output  # output 仅承载 stdout

    async def test_utf8_output(self, tmp_path):
        runner = make_runner(tmp_path)
        result = await runner.execute("print('中文输出✓')", "t1", "run1")
        assert result.success is True
        assert "中文输出✓" in result.output

    async def test_separate_runs_isolated(self, tmp_path):
        runner = make_runner(tmp_path)
        await runner.execute("print('one')", "t1", "run_a")
        await runner.execute("print('two')", "t1", "run_b")
        assert (tmp_path / "task_t1" / "run_run_a" / "main.py").is_file()
        assert (tmp_path / "task_t1" / "run_run_b" / "main.py").is_file()


class TestArtifactsAndMetadata:
    async def test_artifacts_written(self, tmp_path):
        runner = make_runner(tmp_path)
        result = await runner.execute(PRINT_HELLO, "t1", "run1")
        run_dir = tmp_path / "task_t1" / "run_run1"
        expected = [
            str(run_dir / "stdout.txt"),
            str(run_dir / "stderr.txt"),
            str(run_dir / "metadata.json"),
        ]
        assert result.artifacts == expected
        for path in expected:
            assert Path(path).is_file()

    async def test_stdout_file_content(self, tmp_path):
        runner = make_runner(tmp_path)
        await runner.execute(PRINT_HELLO, "t1", "run1")
        text = (tmp_path / "task_t1" / "run_run1" / "stdout.txt").read_text(
            encoding="utf-8"
        )
        assert "hello sandbox" in text

    async def test_metadata_fields(self, tmp_path):
        runner = make_runner(tmp_path)
        await runner.execute(PRINT_HELLO, "t1", "run1")
        metadata = json.loads(
            (tmp_path / "task_t1" / "run_run1" / "metadata.json").read_text(
                encoding="utf-8"
            )
        )
        assert metadata["task_id"] == "t1"
        assert metadata["run_id"] == "run1"
        assert metadata["exit_code"] == 0
        assert metadata["timed_out"] is False
        assert metadata["stdout_truncated"] is False
        assert metadata["stderr_truncated"] is False
        assert metadata["duration_seconds"] >= 0

    async def test_evidence_chain(self, tmp_path):
        runner = make_runner(tmp_path)
        result = await runner.execute(PRINT_HELLO, "t1", "run1")
        joined = "\n".join(result.evidence)
        assert "command:" in joined
        assert "exit_code: 0" in joined
        assert "timed_out: False" in joined
        assert "stdout.txt" in joined

    async def test_sandbox_root_property(self, tmp_path):
        runner = make_runner(tmp_path)
        assert runner.sandbox_root == tmp_path.resolve()


class TestNonZeroExit:
    async def test_failure_result(self, tmp_path):
        runner = make_runner(tmp_path)
        result = await runner.execute(EXIT_NONZERO, "t1", "run1")
        assert result.success is False
        assert result.error is not None
        assert result.error.startswith("进程退出码 3：")
        assert "boom-message" in result.error

    async def test_stderr_saved_to_file(self, tmp_path):
        runner = make_runner(tmp_path)
        await runner.execute(EXIT_NONZERO, "t1", "run1")
        text = (tmp_path / "task_t1" / "run_run1" / "stderr.txt").read_text(
            encoding="utf-8"
        )
        assert "boom-message" in text


class TestTimeout:
    async def test_timeout_kills_process(self, tmp_path):
        runner = make_runner(tmp_path, timeout=0.8)
        result = await runner.execute(SLEEP_LONG, "t1", "run1")
        assert result.success is False
        assert result.error is not None
        assert "执行超时" in result.error
        assert "已终止" in result.error

    async def test_per_call_timeout_override(self, tmp_path):
        runner = make_runner(tmp_path, timeout=30.0)
        result = await runner.execute(SLEEP_LONG, "t1", "run1", timeout=0.8)
        assert result.success is False
        assert "执行超时" in result.error

    async def test_timeout_recorded_in_metadata(self, tmp_path):
        runner = make_runner(tmp_path, timeout=0.8)
        await runner.execute(SLEEP_LONG, "t1", "run1")
        metadata = json.loads(
            (tmp_path / "task_t1" / "run_run1" / "metadata.json").read_text(
                encoding="utf-8"
            )
        )
        assert metadata["timed_out"] is True


class TestOutputTruncation:
    async def test_stdout_truncated_with_marker(self, tmp_path):
        runner = make_runner(tmp_path, output_limit=50)
        code = "print('x' * 200)"
        result = await runner.execute(code, "t1", "run1")
        assert "输出超过 sandbox_output_limit" in result.output
        assert "已截断" in result.output
        # 截断后保留 limit 字节（不含换行与标记）
        assert result.output.startswith("x" * 50)

    async def test_truncation_recorded_in_metadata(self, tmp_path):
        runner = make_runner(tmp_path, output_limit=50)
        await runner.execute("print('y' * 200)", "t1", "run1")
        metadata = json.loads(
            (tmp_path / "task_t1" / "run_run1" / "metadata.json").read_text(
                encoding="utf-8"
            )
        )
        assert metadata["stdout_truncated"] is True
        assert metadata["stdout_bytes"] > 50

    async def test_small_output_not_truncated(self, tmp_path):
        runner = make_runner(tmp_path, output_limit=10_000)
        result = await runner.execute(PRINT_HELLO, "t1", "run1")
        assert "已截断" not in result.output


class TestInputValidation:
    async def test_empty_code_rejected(self, tmp_path):
        runner = make_runner(tmp_path)
        with pytest.raises(SandboxError):
            await runner.execute("", "t1", "run1")
        with pytest.raises(SandboxError):
            await runner.execute("   ", "t1", "run1")

    @pytest.mark.parametrize(
        "bad_id",
        ["../evil", "a/b", "a\\b", "..", "x" * 65, "有中文", "a b"],
    )
    async def test_illegal_task_id_rejected(self, tmp_path, bad_id):
        runner = make_runner(tmp_path)
        with pytest.raises(SandboxError):
            await runner.execute(PRINT_HELLO, bad_id, "run1")

    @pytest.mark.parametrize("bad_id", ["../evil", "a/b", "", "y" * 100])
    async def test_illegal_run_id_rejected(self, tmp_path, bad_id):
        runner = make_runner(tmp_path)
        with pytest.raises(SandboxError):
            await runner.execute(PRINT_HELLO, "t1", bad_id)

    async def test_non_positive_timeout_rejected(self, tmp_path):
        runner = make_runner(tmp_path)
        with pytest.raises(SandboxError):
            await runner.execute(PRINT_HELLO, "t1", "run1", timeout=0)
        with pytest.raises(SandboxError):
            await runner.execute(PRINT_HELLO, "t1", "run1", timeout=-1.0)

    async def test_valid_ids_with_dash_and_underscore(self, tmp_path):
        runner = make_runner(tmp_path)
        result = await runner.execute(PRINT_HELLO, "task-1_A", "run-1_x")
        assert result.success is True
        assert result.subtask_id == "run-1_x"

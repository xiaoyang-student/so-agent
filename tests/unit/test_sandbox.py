"""单元测试：验证型沙箱 SandboxRunner（真实子进程执行）。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from so_agent.config import Settings
from so_agent.runtime.sandbox import (
    SandboxError,
    SandboxRunner,
    is_secret_env_name,
    scrub_env,
)

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
        # 隔离级别必须显式落盘：本地沙箱不是安全边界，不能靠调用方猜测
        assert metadata["backend"] == "local"
        assert metadata["isolation_level"] == "none"

    async def test_isolation_level_in_result(self, tmp_path):
        runner = make_runner(tmp_path)
        result = await runner.execute(PRINT_HELLO, "t1", "run1")
        assert result.metadata["backend"] == "local"
        assert result.metadata["isolation_level"] == "none"
        assert "isolation: none" in result.evidence

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

    async def test_timeout_kills_spawned_child_processes(self, tmp_path):
        runner = make_runner(tmp_path, timeout=0.8)
        code = (
            "import subprocess, sys, time\n"
            "subprocess.Popen([sys.executable, '-c', "
            "\"import time, pathlib; time.sleep(2); "
            "pathlib.Path('child_marker.txt').write_text('alive')\"])\n"
            "time.sleep(60)\n"
        )
        result = await runner.execute(code, "t1", "run1")
        assert result.success is False
        assert "执行超时" in result.error

        # 给幸存孙进程足够时间落盘；进程树终止后文件不应出现
        import asyncio

        await asyncio.sleep(2.5)
        run_dir = tmp_path / "task_t1" / "run_run1"
        assert not (run_dir / "child_marker.txt").exists()


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

    async def test_non_positive_timeout_means_unlimited(self, tmp_path):
        runner = make_runner(tmp_path)
        zero = await runner.execute(PRINT_HELLO, "t1", "run1", timeout=0)
        assert zero.success is True
        negative = await runner.execute(PRINT_HELLO, "t1", "run2", timeout=-1.0)
        assert negative.success is True

    async def test_valid_ids_with_dash_and_underscore(self, tmp_path):
        runner = make_runner(tmp_path)
        result = await runner.execute(PRINT_HELLO, "task-1_A", "run-1_x")
        assert result.success is True
        assert result.subtask_id == "run-1_x"


class TestEnvScrubbing:
    SECRET_CODE = (
        "import os\n"
        "print(os.environ.get('SO_AGENT_API_KEY', 'MISSING'))\n"
        "print(os.environ.get('GITHUB_TOKEN', 'MISSING'))\n"
        "print(os.environ.get('PLAIN_VISIBLE', 'MISSING'))\n"
    )

    def test_secret_names_detected(self):
        for name in (
            "SO_AGENT_API_KEY",
            "OPENAI_API_KEY",
            "MIMO_SECRET_KEY",
            "AWS_SECRET_ACCESS_KEY",
            "DB_PASSWORD",
            "GITHUB_TOKEN",
            "SIGNING_KEY",
            "AUTHORIZATION",
            "TOKEN",
            "KEY",
        ):
            assert is_secret_env_name(name), name

    def test_runtime_names_kept(self):
        for name in (
            "PATH",
            "SYSTEMROOT",
            "TEMP",
            "PYTHONPATH",
            "PYTHONIOENCODING",
            "MONKEY_MODE",
            "PLAIN_VISIBLE",
        ):
            assert not is_secret_env_name(name), name

    def test_scrub_env_strips_secrets_and_keeps_runtime(self):
        source = {
            "PATH": "/usr/bin",
            "SO_AGENT_API_KEY": "sk-test-secret",
            "GITHUB_TOKEN": "ghp-test",
            "DB_PASSWORD": "p@ss",
            "PLAIN_VISIBLE": "ok",
        }
        env, removed = scrub_env(source)
        assert "SO_AGENT_API_KEY" not in env
        assert "GITHUB_TOKEN" not in env
        assert "DB_PASSWORD" not in env
        assert "sk-test-secret" not in env.values()
        assert "ghp-test" not in env.values()
        assert env["PATH"] == "/usr/bin"
        assert env["PLAIN_VISIBLE"] == "ok"
        assert env["PYTHONIOENCODING"] == "utf-8"
        assert env["PYTHONUTF8"] == "1"
        assert removed == ["DB_PASSWORD", "GITHUB_TOKEN", "SO_AGENT_API_KEY"]
        # 不得修改调用方传入的映射
        assert source["SO_AGENT_API_KEY"] == "sk-test-secret"

    async def test_child_process_cannot_read_secret_env(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SO_AGENT_API_KEY", "sk-test-secret")
        monkeypatch.setenv("GITHUB_TOKEN", "ghp-test")
        monkeypatch.setenv("PLAIN_VISIBLE", "ok")
        runner = make_runner(tmp_path)
        result = await runner.execute(self.SECRET_CODE, "t1", "run1")
        assert result.success is True
        lines = result.output.splitlines()
        assert lines == ["MISSING", "MISSING", "ok"]
        metadata = json.loads(
            (tmp_path / "task_t1" / "run_run1" / "metadata.json").read_text(
                encoding="utf-8"
            )
        )
        assert metadata["env_scrubbed_count"] >= 2
        evidence = "\n".join(result.evidence)
        assert f"env_scrubbed: {metadata['env_scrubbed_count']}" in evidence


class TestWorkspaceContract:
    """脚本工作目录与文件工具共用沙箱根目录（路径一致性）。"""

    RELATIVE_WRITE = (
        "from pathlib import Path\n"
        "Path('sales.csv').write_text('region,sales\\nEast,10\\n', encoding='utf-8')\n"
        "print('wrote', Path('sales.csv').resolve())\n"
    )

    async def test_workdir_relative_paths_resolve_to_workspace(self, tmp_path):
        runner = make_runner(tmp_path)
        result = await runner.execute(
            self.RELATIVE_WRITE, "t1", "run1", workdir=tmp_path
        )
        assert result.success is True
        assert (tmp_path / "sales.csv").is_file()
        run_dir = tmp_path / "task_t1" / "run_run1"
        assert not (run_dir / "sales.csv").exists()
        # 运行留痕仍落在运行目录，不污染工作区
        assert (run_dir / "stdout.txt").is_file()
        assert (run_dir / "metadata.json").is_file()
        assert result.metadata["workdir"] == str(tmp_path.resolve())

    async def test_default_workdir_is_run_dir(self, tmp_path):
        runner = make_runner(tmp_path)
        result = await runner.execute(
            "from pathlib import Path\nprint(Path.cwd().name)\n", "t1", "run1"
        )
        assert result.success is True
        assert result.output.strip() == "run_run1"

    async def test_workdir_outside_root_rejected(self, tmp_path):
        runner = make_runner(tmp_path / "sandbox")
        with pytest.raises(SandboxError):
            await runner.execute(
                PRINT_HELLO, "t1", "run1", workdir=tmp_path
            )

    async def test_workdir_missing_rejected(self, tmp_path):
        runner = make_runner(tmp_path)
        with pytest.raises(SandboxError):
            await runner.execute(
                PRINT_HELLO, "t1", "run1", workdir=tmp_path / "missing"
            )

    async def test_workspace_root_on_sys_path(self, tmp_path):
        sandbox = tmp_path / "sandbox"
        sandbox.mkdir()
        (sandbox / "helper_module.py").write_text(
            "VALUE = 'from-workspace'\n", encoding="utf-8"
        )
        runner = make_runner(sandbox)
        result = await runner.execute(
            "import helper_module\nprint(helper_module.VALUE)\n",
            "t1",
            "run1",
            workdir=sandbox,
        )
        assert result.success is True
        assert result.output.strip() == "from-workspace"

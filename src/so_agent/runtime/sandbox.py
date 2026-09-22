"""验证型沙箱（SandboxRunner）。

职责与行为约定：
- 将子 Agent 提交的代码写入 ``sandbox/task_{task_id}/run_{run_id}/main.py``，
  以独立子进程执行（asyncio.create_subprocess_exec），工作目录限定在运行目录内；
- 标识符校验 + 路径包含校验双重防线：task_id / run_id 只允许字母、数字、
  下划线、连字符；解析后的运行目录必须位于沙箱根目录之内，杜绝路径穿越；
- 超时（sandbox_timeout，可按次覆盖）后强制终止进程；stdout / stderr 超过
  sandbox_output_limit 字节时截断并标注；
- 每次执行落盘 stdout.txt / stderr.txt / metadata.json，返回 ExecutionResult
  形态的结构化结果，artifacts 为上述产物路径，evidence 为执行证据链。

说明：本沙箱是“验证型”而非安全隔离容器——它保证执行过程可追踪、可审计、
可控时，不对抗恶意代码；真正的资源隔离属于后续演进方向。
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from so_agent.config import Settings, get_settings
from so_agent.context import ProjectContext
from so_agent.models import ExecutionResult

# task_id / run_id 允许的字符集（防路径穿越第一道防线）
_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

# error 字段中 stderr 摘要的最大长度（字符）
_ERROR_EXCERPT_LIMIT = 2000


class SandboxError(Exception):
    """沙箱领域错误：非法标识符、路径越界、空代码、非法超时等。"""


class SandboxRunner:
    """验证型代码沙箱（子进程执行实现）。"""

    def __init__(
        self,
        *,
        context: ProjectContext | None = None,
        sandbox_dir: str | Path | None = None,
        settings: Settings | None = None,
        python_executable: str | None = None,
    ) -> None:
        """初始化沙箱。

        Args:
            context: 可选共享上下文（沙箱根目录与配置的默认来源）。
            sandbox_dir: 沙箱根目录；优先级高于 ``context.sandbox_dir``；
                两者均未提供时回退到 ``<当前工作目录>/sandbox``。
            settings: 可选配置（超时、输出上限）；优先级为
                显式参数 > context.config > 全局配置。
            python_executable: 执行代码的解释器路径；默认使用当前解释器。
        """
        if sandbox_dir is not None:
            root = Path(sandbox_dir)
        elif context is not None:
            root = Path(context.sandbox_dir)
        else:
            root = Path.cwd() / "sandbox"
        self._root = root.expanduser().resolve()
        self._settings = settings or (
            context.config if context is not None else get_settings()
        )
        self._python = python_executable or sys.executable

    # ------------------------------------------------------------------
    # 属性
    # ------------------------------------------------------------------
    @property
    def sandbox_root(self) -> Path:
        """返回沙箱根目录（绝对路径）。"""
        return self._root

    @property
    def settings(self) -> Settings:
        """返回生效的配置对象。"""
        return self._settings

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------
    def _validate_identifier(self, value: str, label: str) -> None:
        """校验 task_id / run_id 合法性（字符集与长度）。

        Raises:
            SandboxError: 标识符非法时。
        """
        if not isinstance(value, str) or not _ID_PATTERN.match(value):
            raise SandboxError(
                f"{label} 非法：{value!r}；只允许字母/数字/下划线/连字符，长度 1-64"
            )

    def _resolve_run_dir(self, task_id: str, run_id: str) -> Path:
        """解析并校验运行目录（必须位于沙箱根目录之内）。

        Raises:
            SandboxError: 标识符非法或路径越界时。
        """
        self._validate_identifier(task_id, "task_id")
        self._validate_identifier(run_id, "run_id")
        run_dir = (self._root / f"task_{task_id}" / f"run_{run_id}").resolve()
        if not run_dir.is_relative_to(self._root):
            raise SandboxError(
                f"运行目录越界：{run_dir} 不在沙箱根目录 {self._root} 之内"
            )
        return run_dir

    def _decode_output(self, data: bytes, limit: int) -> tuple[str, bool]:
        """将字节输出解码为文本；超过 limit 字节时截断并标注。

        Returns:
            (文本, 是否发生截断)。limit <= 0 表示不限制。
        """
        limit = int(limit)
        truncated = limit > 0 and len(data) > limit
        if truncated:
            data = data[:limit]
        text = data.decode("utf-8", errors="replace")
        if truncated:
            text += f"\n...[输出超过 sandbox_output_limit（{limit} 字节），已截断]"
        return text, truncated

    # ------------------------------------------------------------------
    # 执行入口
    # ------------------------------------------------------------------
    async def execute(
        self,
        code: str,
        task_id: str,
        run_id: str,
        *,
        timeout: float | None = None,
    ) -> ExecutionResult:
        """在沙箱中执行一段 Python 代码并返回结构化执行结果。

        Args:
            code: 待执行的 Python 源码（写入运行目录下的 main.py）。
            task_id: 任务编号（用于目录隔离）。
            run_id: 本次运行编号（同一任务可多次运行）。
            timeout: 本次执行超时（秒）；默认取 ``settings.sandbox_timeout``。

        Returns:
            ExecutionResult：subtask_id 填 run_id（沙箱层不感知子任务语义）；
            success 为 退出码为 0 且未超时；artifacts 为 stdout.txt /
            stderr.txt / metadata.json 的路径；evidence 为执行证据链。

        Raises:
            SandboxError: 标识符非法、路径越界、代码为空、超时参数非法，
                或无法启动子进程时。
        """
        if not isinstance(code, str) or not code.strip():
            raise SandboxError("code 不能为空")

        effective_timeout = (
            float(timeout) if timeout is not None else float(self._settings.sandbox_timeout)
        )
        if effective_timeout <= 0:
            raise SandboxError(f"timeout 必须为正数，实际为 {effective_timeout}")

        output_limit = int(self._settings.sandbox_output_limit)
        run_dir = self._resolve_run_dir(task_id, run_id)
        run_dir.mkdir(parents=True, exist_ok=True)
        main_py = run_dir / "main.py"
        main_py.write_text(code, encoding="utf-8")

        command = [self._python, str(main_py)]
        env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}

        started = time.perf_counter()
        timed_out = False
        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                cwd=str(run_dir),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )
        except OSError as exc:
            raise SandboxError(f"无法启动沙箱子进程：{exc}") from exc

        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                process.communicate(), timeout=effective_timeout
            )
        except asyncio.TimeoutError:
            timed_out = True
            try:
                process.kill()
            except ProcessLookupError:
                pass
            stdout_bytes, stderr_bytes = await process.communicate()

        duration = time.perf_counter() - started
        exit_code = process.returncode if process.returncode is not None else -1
        raw_stdout_len = len(stdout_bytes)
        raw_stderr_len = len(stderr_bytes)

        stdout_text, stdout_truncated = self._decode_output(stdout_bytes, output_limit)
        stderr_text, stderr_truncated = self._decode_output(stderr_bytes, output_limit)

        success = (not timed_out) and exit_code == 0
        stdout_path = run_dir / "stdout.txt"
        stderr_path = run_dir / "stderr.txt"
        metadata_path = run_dir / "metadata.json"

        stdout_path.write_text(stdout_text, encoding="utf-8")
        stderr_path.write_text(stderr_text, encoding="utf-8")
        metadata = {
            "task_id": task_id,
            "run_id": run_id,
            "command": command,
            "cwd": str(run_dir),
            "code_file": str(main_py),
            "exit_code": exit_code,
            "duration_seconds": round(duration, 6),
            "timed_out": timed_out,
            "stdout_bytes": raw_stdout_len,
            "stderr_bytes": raw_stderr_len,
            "stdout_truncated": stdout_truncated,
            "stderr_truncated": stderr_truncated,
            "finished_at": datetime.now(timezone.utc).isoformat(),
        }
        metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        error: str | None = None
        if timed_out:
            error = f"执行超时（超过 {effective_timeout:g} 秒），进程已终止"
        elif exit_code != 0:
            excerpt = stderr_text.strip()
            if len(excerpt) > _ERROR_EXCERPT_LIMIT:
                excerpt = excerpt[:_ERROR_EXCERPT_LIMIT] + "…"
            error = (
                f"进程退出码 {exit_code}：{excerpt}"
                if excerpt
                else f"进程退出码 {exit_code}（无错误输出）"
            )

        evidence = [
            f"command: {' '.join(command)}",
            f"cwd: {run_dir}",
            f"exit_code: {exit_code}",
            f"duration: {duration:.3f}s",
            f"timed_out: {timed_out}",
            f"stdout: {stdout_path}",
            f"stderr: {stderr_path}",
        ]
        artifacts = [str(stdout_path), str(stderr_path), str(metadata_path)]

        return ExecutionResult(
            subtask_id=run_id,
            success=success,
            output=stdout_text,
            evidence=evidence,
            artifacts=artifacts,
            error=error,
            duration=duration,
        )

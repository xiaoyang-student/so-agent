"""容器沙箱后端（L2）。

复用 Docker / Podman 的命名空间、cgroup 与 seccomp 能力执行单次 Python
代码。默认策略：无网络、只读根文件系统、丢弃全部 capability、禁止提权、
限制 pids/内存/CPU，仅把本次运行目录挂载为可写工作区。

该后端不等于 microVM：容器与宿主共享内核，适合“基础安全隔离”，不适用于
高风险多租户或需要硬件级隔离的场景。云沙箱 / microVM 通过同一
``SandboxBackend`` 协议另行注册。
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from so_agent.config import Settings, get_settings
from so_agent.models import ExecutionResult
from so_agent.runtime.sandbox_common import (
    SandboxError,
    decode_output,
    resolve_run_dir,
)

_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_RUNTIME_CACHE: dict[str, tuple[float, "ContainerRuntime | None"]] = {}
_CACHE_TTL_SECONDS = 60.0


@dataclass(frozen=True)
class ContainerRuntime:
    """已探测可用的容器运行时。"""

    name: str
    executable: str
    version: str = ""


class ContainerSandboxError(SandboxError):
    """容器运行时不可用或容器启动失败。"""


def _run_probe(command: list[str], timeout: float = 8.0) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def _probe_runtime(name: str, executable: str) -> ContainerRuntime | None:
    try:
        if name == "docker":
            result = _run_probe(
                [executable, "info", "--format", "{{.ServerVersion}}"]
            )
        else:
            result = _run_probe(
                [executable, "info", "--format", "{{.Version.Version}}"]
            )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    version = (result.stdout or "").strip().splitlines()
    return ContainerRuntime(
        name=name,
        executable=executable,
        version=version[0] if version else "",
    )


def detect_container_runtime(
    preferred: str = "auto", *, refresh: bool = False
) -> ContainerRuntime | None:
    """探测 Docker / Podman。``preferred`` 为 auto/docker/podman。"""
    key = (preferred or "auto").strip().lower()
    now = time.monotonic()
    cached = _RUNTIME_CACHE.get(key)
    if cached is not None and not refresh and now - cached[0] < _CACHE_TTL_SECONDS:
        return cached[1]

    candidates = ["docker", "podman"] if key == "auto" else [key]
    runtime: ContainerRuntime | None = None
    for name in candidates:
        if name not in {"docker", "podman"}:
            continue
        executable = shutil.which(name)
        if not executable:
            continue
        runtime = _probe_runtime(name, executable)
        if runtime is not None:
            break
    _RUNTIME_CACHE[key] = (now, runtime)
    return runtime


def _container_name(task_id: str, run_id: str) -> str:
    raw = f"so-agent-{task_id[:24]}-{run_id[:24]}"
    return re.sub(r"[^A-Za-z0-9_.-]", "-", raw)


class ContainerSandbox:
    """Docker / Podman 容器沙箱。"""

    isolation_level = "container"

    def __init__(
        self,
        *,
        context: Any = None,
        sandbox_dir: str | Path | None = None,
        settings: Settings | None = None,
        python_executable: str | None = None,
        runtime: ContainerRuntime | None = None,
    ) -> None:
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
        self._runtime = runtime or detect_container_runtime()
        if self._runtime is None:
            raise ContainerSandboxError(
                "未检测到可用的 Docker/Podman 运行时；请先运行 "
                "`so-agent-sandbox setup` 安装并启动容器运行时"
            )
        self._image = (self._settings.sandbox_container_image or "").strip()
        self._python = (
            self._settings.sandbox_container_python or "python"
        ).strip()
        self._image_ready = False

    @property
    def sandbox_root(self) -> Path:
        return self._root

    @property
    def runtime(self) -> ContainerRuntime:
        assert self._runtime is not None
        return self._runtime

    def _resolve_run_dir(self, task_id: str, run_id: str) -> Path:
        return resolve_run_dir(self._root, task_id, run_id, _ID_PATTERN)

    def _resolve_workdir(self, workdir: str | Path | None, run_dir: Path) -> Path:
        """解析容器内工作区（缺省为运行目录，必须位于沙箱根目录内）。"""
        if workdir is None:
            return run_dir
        candidate = Path(workdir).expanduser()
        resolved = (
            candidate.resolve()
            if candidate.is_absolute()
            else (self._root / candidate).resolve()
        )
        if not resolved.is_relative_to(self._root):
            raise ContainerSandboxError(
                f"工作目录越界：{resolved} 不在沙箱根目录 {self._root} 之内"
            )
        if not resolved.is_dir():
            raise ContainerSandboxError(f"工作目录不存在：{resolved}")
        return resolved

    def _container_env(self) -> dict[str, str]:
        # 容器 CLI 自身需要 Docker/Podman 的连接配置；密钥不传入容器。
        env = dict(os.environ)
        for name in list(env):
            upper = name.upper()
            if any(
                marker in upper
                for marker in (
                    "API_KEY",
                    "SECRET",
                    "PASSWORD",
                    "CREDENTIAL",
                    "PRIVATE_KEY",
                    "ACCESS_KEY",
                    "AUTH_TOKEN",
                )
            ) or upper.endswith(("_TOKEN", "_KEY")):
                env.pop(name, None)
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUTF8"] = "1"
        return env

    async def _run_async(
        self, command: list[str], *, timeout: float
    ) -> tuple[int, str, str]:
        assert self._runtime is not None
        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except OSError as exc:
            raise ContainerSandboxError(
                f"无法启动容器运行时 {self.runtime.name}：{exc}"
            ) from exc
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(), timeout=timeout
            )
        except asyncio.TimeoutError:
            process.kill()
            await process.communicate()
            raise ContainerSandboxError(
                f"容器运行时命令超时（{timeout:g}s）：{' '.join(command[:3])}"
            ) from None
        return (
            process.returncode or 0,
            stdout.decode("utf-8", errors="replace"),
            stderr.decode("utf-8", errors="replace"),
        )

    async def _ensure_image(self) -> None:
        assert self._runtime is not None
        if self._image_ready:
            return
        if not self._image:
            raise ContainerSandboxError("sandbox_container_image 不能为空")
        code, _stdout, _stderr = await self._run_async(
            [self._runtime.executable, "image", "inspect", self._image],
            timeout=15,
        )
        if code == 0:
            self._image_ready = True
            return
        if not self._settings.sandbox_container_pull:
            raise ContainerSandboxError(
                f"沙箱镜像不存在且未启用自动拉取：{self._image}"
            )
        code, stdout, stderr = await self._run_async(
            [self._runtime.executable, "pull", self._image],
            timeout=600,
        )
        if code != 0:
            detail = (stderr or stdout or "").strip()[:1000]
            raise ContainerSandboxError(
                f"拉取沙箱镜像失败：{self._image}；{detail}"
            )
        self._image_ready = True

    def _build_command(
        self,
        *,
        task_id: str,
        run_id: str,
        run_dir: Path,
        workspace: Path,
    ) -> tuple[list[str], Path]:
        assert self._runtime is not None
        settings = self._settings
        name = _container_name(task_id, run_id)
        cidfile = run_dir / ".container_id"
        try:
            cidfile.unlink(missing_ok=True)
        except OSError:
            pass
        tmpfs_mb = max(16, int(settings.sandbox_container_tmpfs_mb))
        memory_mb = max(64, int(settings.sandbox_container_memory_mb))
        pids = max(16, int(settings.sandbox_container_pids_limit))
        cpus = max(0.1, float(settings.sandbox_container_cpus))
        network = settings.sandbox_container_network
        if network not in {"none", "bridge"}:
            network = "none"

        try:
            main_rel = (run_dir / "main.py").relative_to(workspace).as_posix()
        except ValueError:
            raise ContainerSandboxError(
                f"运行目录 {run_dir} 不在容器工作区 {workspace} 之内："
                "容器后端要求工作区包含运行目录"
            ) from None

        command = [
            self._runtime.executable,
            "run",
            "--rm",
            "--init",
            "--name",
            name,
            "--cidfile",
            str(cidfile),
            "--label",
            f"so-agent.task_id={task_id}",
            "--network",
            network,
            "--read-only",
            "--tmpfs",
            f"/tmp:rw,noexec,nosuid,size={tmpfs_mb}m",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--pids-limit",
            str(pids),
            "--memory",
            f"{memory_mb}m",
            "--memory-swap",
            f"{memory_mb}m",
            "--cpus",
            str(cpus),
            "--mount",
            f"type=bind,source={workspace},target=/work",
            "--workdir",
            "/work",
            "--env",
            "HOME=/tmp",
            "--env",
            "TMPDIR=/tmp",
            "--env",
            "PYTHONIOENCODING=utf-8",
            "--env",
            "PYTHONPATH=/work",
        ]
        user = (settings.sandbox_container_user or "").strip()
        if user:
            command.extend(["--user", user])
        elif os.name != "nt" and hasattr(os, "getuid"):
            command.extend(["--user", f"{os.getuid()}:{os.getgid()}"])
        command.extend(
            [self._image, self._python, f"/work/{main_rel}"]
        )
        return command, cidfile

    async def _kill_container(self, cidfile: Path, name: str) -> None:
        assert self._runtime is not None
        target = name
        try:
            if cidfile.is_file():
                value = cidfile.read_text(encoding="utf-8").strip()
                if value:
                    target = value
        except OSError:
            pass
        for verb in (("kill", target), ("rm", "-f", target)):
            try:
                process = await asyncio.create_subprocess_exec(
                    self._runtime.executable,
                    *verb,
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.DEVNULL,
                )
                await asyncio.wait_for(process.wait(), timeout=10)
            except (OSError, asyncio.TimeoutError):
                pass

    async def execute(
        self,
        code: str,
        task_id: str,
        run_id: str,
        *,
        timeout: float | None = None,
        workdir: str | Path | None = None,
    ) -> ExecutionResult:
        if not isinstance(code, str) or not code.strip():
            raise ContainerSandboxError("code 不能为空")
        effective_timeout = (
            float(timeout)
            if timeout is not None
            else float(self._settings.sandbox_timeout)
        )
        run_dir = self._resolve_run_dir(task_id, run_id)
        workspace = self._resolve_workdir(workdir, run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "main.py").write_text(code, encoding="utf-8")
        await self._ensure_image()
        command, cidfile = self._build_command(
            task_id=task_id,
            run_id=run_id,
            run_dir=run_dir,
            workspace=workspace,
        )
        started = time.perf_counter()
        timed_out = False
        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=self._container_env(),
            )
        except OSError as exc:
            raise ContainerSandboxError(
                f"无法启动容器运行时 {self.runtime.name}：{exc}"
            ) from exc

        try:
            if effective_timeout > 0:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    process.communicate(), timeout=effective_timeout
                )
            else:
                stdout_bytes, stderr_bytes = await process.communicate()
        except asyncio.TimeoutError:
            timed_out = True
            await self._kill_container(cidfile, _container_name(task_id, run_id))
            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    process.communicate(), timeout=10
                )
            except asyncio.TimeoutError:
                process.kill()
                stdout_bytes, stderr_bytes = await process.communicate()
        except asyncio.CancelledError:
            await self._kill_container(cidfile, _container_name(task_id, run_id))
            try:
                process.kill()
            except ProcessLookupError:
                pass
            await process.communicate()
            raise

        duration = time.perf_counter() - started
        exit_code = process.returncode if process.returncode is not None else -1
        output_limit = int(self._settings.sandbox_output_limit)
        stdout_text, stdout_truncated = decode_output(stdout_bytes, output_limit)
        stderr_text, stderr_truncated = decode_output(stderr_bytes, output_limit)
        success = (not timed_out) and exit_code == 0

        stdout_path = run_dir / "stdout.txt"
        stderr_path = run_dir / "stderr.txt"
        metadata_path = run_dir / "metadata.json"
        stdout_path.write_text(stdout_text, encoding="utf-8")
        stderr_path.write_text(stderr_text, encoding="utf-8")
        metadata = {
            "task_id": task_id,
            "run_id": run_id,
            "backend": self.runtime.name,
            "isolation_level": self.isolation_level,
            "image": self._image,
            "workspace": str(workspace),
            "run_dir": str(run_dir),
            "network": self._settings.sandbox_container_network,
            "memory_mb": self._settings.sandbox_container_memory_mb,
            "cpus": self._settings.sandbox_container_cpus,
            "pids_limit": self._settings.sandbox_container_pids_limit,
            "exit_code": exit_code,
            "duration_seconds": round(duration, 6),
            "timed_out": timed_out,
            "stdout_bytes": len(stdout_bytes),
            "stderr_bytes": len(stderr_bytes),
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
            error = f"容器执行超时（超过 {effective_timeout:g} 秒），容器已终止"
        elif exit_code != 0:
            excerpt = stderr_text.strip()
            if len(excerpt) > 2000:
                excerpt = excerpt[:2000] + "…"
            error = (
                f"容器退出码 {exit_code}：{excerpt}"
                if excerpt
                else f"容器退出码 {exit_code}（无错误输出）"
            )

        evidence = [
            f"backend: {self.runtime.name}",
            f"isolation: {self.isolation_level}",
            f"image: {self._image}",
            f"network: {self._settings.sandbox_container_network}",
            f"exit_code: {exit_code}",
            f"duration: {duration:.3f}s",
            f"timed_out: {timed_out}",
            f"stdout: {stdout_path}",
            f"stderr: {stderr_path}",
        ]
        return ExecutionResult(
            subtask_id=run_id,
            success=success,
            output=stdout_text,
            evidence=evidence,
            artifacts=[str(stdout_path), str(stderr_path), str(metadata_path)],
            error=error,
            duration=duration,
            metadata={
                "backend": self.runtime.name,
                "isolation_level": self.isolation_level,
                "exit_code": exit_code,
                "duration_seconds": round(duration, 6),
                "timed_out": timed_out,
                "stdout_truncated": stdout_truncated,
                "stderr_truncated": stderr_truncated,
                "workdir": "/work",
            },
        )


__all__ = [
    "ContainerRuntime",
    "ContainerSandbox",
    "ContainerSandboxError",
    "detect_container_runtime",
]

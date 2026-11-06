"""验证型沙箱（SandboxRunner）。

职责与行为约定：
- 将子 Agent 提交的代码写入 ``sandbox/task_{task_id}/run_{run_id}/main.py``，
  以独立子进程执行（asyncio.create_subprocess_exec）；默认工作目录为运行
  目录，调用方可通过 ``workdir`` 指定工作区（必须位于沙箱根目录内），
  使脚本中的相对路径与 ``read_file`` / ``write_file`` 的解析基准一致；
- 标识符校验 + 路径包含校验双重防线：task_id / run_id 只允许字母、数字、
  下划线、连字符；解析后的运行目录必须位于沙箱根目录之内，杜绝路径穿越；
- 超时（sandbox_timeout，可按次覆盖）或任务取消时终止整个进程树（POSIX
  独立进程组 killpg，Windows taskkill /T），避免子进程派生的孙进程继续存活；
  stdout / stderr 超过 sandbox_output_limit 字节时截断并标注；
- 子进程环境剔除疑似凭证变量（*_API_KEY / *SECRET* / *TOKEN 等），
  避免沙箱内代码读到模型密钥；保留 PATH / SYSTEMROOT 等启动必需变量；
- 每次执行落盘 stdout.txt / stderr.txt / metadata.json，返回 ExecutionResult
  形态的结构化结果，artifacts 为上述产物路径，evidence 为执行证据链。

说明：本沙箱是“验证型”而非安全隔离容器——它保证执行过程可追踪、可审计、
可控时，不对抗恶意代码；真正的资源隔离属于后续演进方向。
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import sys
import time
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol, runtime_checkable

from so_agent.config import Settings, get_settings
from so_agent.context import ProjectContext
from so_agent.models import ExecutionResult
from so_agent.runtime.process_tree import terminate_process_tree
from so_agent.runtime.sandbox_common import SandboxBackendError, SandboxError

# task_id / run_id 允许的字符集（防路径穿越第一道防线）
_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

logger = logging.getLogger(__name__)

# error 字段中 stderr 摘要的最大长度（字符）
_ERROR_EXCERPT_LIMIT = 2000

# 疑似凭证的环境变量名片段：命中的变量不传给沙箱子进程。
# 采用名字黑名单而非小范围白名单：Windows 上 PATH / SYSTEMROOT / TEMP 等
# 变量是解释器启动的硬依赖，白名单容易造成平台性启动失败。
_SECRET_ENV_MARKERS: tuple[str, ...] = (
    "API_KEY",
    "APIKEY",
    "API-KEY",
    "SECRET",
    "PASSWORD",
    "PASSWD",
    "CREDENTIAL",
    "PRIVATE_KEY",
    "ACCESS_KEY",
    "AUTH_TOKEN",
    "AUTHORIZATION",
    "BEARER",
)
_SECRET_ENV_SUFFIXES: tuple[str, ...] = ("_TOKEN", "_KEY")


def is_secret_env_name(name: str) -> bool:
    """判断环境变量名是否疑似凭证（大小写不敏感）。"""
    upper = name.upper()
    if upper in {"TOKEN", "KEY"}:
        return True
    if upper.endswith(_SECRET_ENV_SUFFIXES):
        return True
    return any(marker in upper for marker in _SECRET_ENV_MARKERS)


def scrub_env(environ: Mapping[str, str] | None = None) -> tuple[dict[str, str], list[str]]:
    """返回脱敏后的子进程环境及被剔除的变量名。

    保留运行必需变量（PATH / SYSTEMROOT / TEMP 等），剔除疑似凭证变量，
    并强制 UTF-8 输出。
    """
    source = dict(os.environ if environ is None else environ)
    removed = sorted(name for name in source if is_secret_env_name(name))
    env = {name: value for name, value in source.items() if name not in removed}
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    return env, removed


@runtime_checkable
class SandboxBackend(Protocol):
    """沙箱后端统一契约。

    消费方只依赖 ``execute``：输入一段 Python 源码与任务/运行标识，
    返回 ``ExecutionResult``。本地验证沙箱（``SandboxRunner``）是默认
    实现；后续云沙箱只需实现同一方法并通过
    ``register_sandbox_backend`` 注册，即可由配置切换。
    """

    async def execute(
        self,
        code: str,
        task_id: str,
        run_id: str,
        *,
        timeout: float | None = None,
        workdir: str | Path | None = None,
    ) -> ExecutionResult:
        """执行一段 Python 代码。

        Args:
            code: 源码（由后端决定落盘位置）。
            task_id: 任务编号（目录隔离）。
            run_id: 本次运行编号。
            timeout: 本次执行超时（秒）；<=0 表示不设单次上限。
            workdir: 脚本的工作目录（相对路径基准）；实现必须将其限制
                在沙箱根目录内。None 时沿用后端默认（运行目录）。
        """
        ...


# 后端名 → 构造工厂。工厂契约：接受 context / sandbox_dir / settings /
# python_executable 四个关键字参数，返回 SandboxBackend。
SandboxFactory = Callable[..., SandboxBackend]

_BACKEND_FACTORIES: dict[str, SandboxFactory] = {}


def register_sandbox_backend(name: str, factory: SandboxFactory) -> None:
    """注册沙箱后端（供云沙箱或测试替身注入）。

    Args:
        name: 后端名称（大小写不敏感，去掉首尾空白后保存）。
        factory: 工厂函数，接受 ``context`` / ``sandbox_dir`` /
            ``settings`` / ``python_executable`` 关键字参数。

    Raises:
        SandboxBackendError: 名称为空或工厂不可调用。
    """
    key = (name or "").strip().lower()
    if not key:
        raise SandboxBackendError("沙箱后端名称不能为空")
    if not callable(factory):
        raise SandboxBackendError(f"沙箱后端 {key!r} 的工厂不可调用")
    _BACKEND_FACTORIES[key] = factory


def available_sandbox_backends() -> tuple[str, ...]:
    """返回已注册的后端名称（排序后）。"""
    return tuple(sorted(_BACKEND_FACTORIES))


def _create_local_backend(
    *,
    context: ProjectContext | None = None,
    sandbox_dir: str | Path | None = None,
    settings: Settings | None = None,
    python_executable: str | None = None,
) -> SandboxBackend:
    return SandboxRunner(
        context=context,
        sandbox_dir=sandbox_dir,
        settings=settings,
        python_executable=python_executable,
    )


def _create_cloud_backend(**_kwargs: object) -> SandboxBackend:
    raise SandboxBackendError(
        "云沙箱后端（cloud）尚未提供实现：当前版本内置本地验证沙箱"
        "（local）与容器沙箱（docker/podman）。接入方式：调用 "
        "register_sandbox_backend('cloud', factory) 注册实现；配置项 "
        "sandbox_cloud_provider / sandbox_cloud_endpoint / "
        "sandbox_cloud_api_key_env / sandbox_cloud_template 已预留。"
    )


def _resolve_backend_settings(
    context: ProjectContext | None, settings: Settings | None
) -> Settings:
    return settings or (
        context.config if context is not None else get_settings()
    )


def _create_docker_backend(
    *,
    context: ProjectContext | None = None,
    sandbox_dir: str | Path | None = None,
    settings: Settings | None = None,
    python_executable: str | None = None,
) -> SandboxBackend:
    runtime = detect_container_runtime("docker")
    if runtime is None:
        raise SandboxBackendError(
            "Docker 后端不可用：未找到 docker 命令或 Docker daemon 未启动；"
            "请运行 `so-agent-sandbox doctor` 检查"
        )
    _ensure_linux_runtime(runtime)
    return ContainerSandbox(
        context=context,
        sandbox_dir=sandbox_dir,
        settings=_resolve_backend_settings(context, settings),
        python_executable=python_executable,
        runtime=runtime,
    )


def _ensure_linux_runtime(runtime: object) -> None:
    """默认沙箱镜像为 Linux 镜像：明确选用 Windows 容器运行时会被拒绝。"""
    if getattr(runtime, "os_type", "") == "windows":
        raise SandboxBackendError(
            f"{getattr(runtime, 'name', 'container')} 当前使用 Windows 容器，"
            "无法运行 Linux 沙箱镜像；请切换到 Linux 容器模式或改用 local 后端"
        )


def _create_podman_backend(
    *,
    context: ProjectContext | None = None,
    sandbox_dir: str | Path | None = None,
    settings: Settings | None = None,
    python_executable: str | None = None,
) -> SandboxBackend:
    runtime = detect_container_runtime("podman")
    if runtime is None:
        raise SandboxBackendError(
            "Podman 后端不可用：未找到 podman 命令或 Podman machine 未启动；"
            "请运行 `so-agent-sandbox doctor` 检查"
        )
    _ensure_linux_runtime(runtime)
    return ContainerSandbox(
        context=context,
        sandbox_dir=sandbox_dir,
        settings=_resolve_backend_settings(context, settings),
        python_executable=python_executable,
        runtime=runtime,
    )


def _create_auto_backend(
    *,
    context: ProjectContext | None = None,
    sandbox_dir: str | Path | None = None,
    settings: Settings | None = None,
    python_executable: str | None = None,
) -> SandboxBackend:
    resolved = _resolve_backend_settings(context, settings)
    runtime = detect_container_runtime("auto")
    windows_only = (
        runtime is not None and getattr(runtime, "os_type", "") == "windows"
    )
    if windows_only:
        # 默认镜像是 Linux 镜像；Windows 容器运行时保留给显式 backend 选择
        runtime = None
    if runtime is not None:
        return ContainerSandbox(
            context=context,
            sandbox_dir=sandbox_dir,
            settings=resolved,
            python_executable=python_executable,
            runtime=runtime,
        )
    detail = (
        "检测到的容器运行时仅支持 Windows 容器，无法运行 Linux 沙箱镜像"
        if windows_only
        else "未检测到 Docker/Podman"
    )
    if resolved.sandbox_require_isolation:
        raise SandboxBackendError(
            f"{detail}，且 sandbox_require_isolation=True："
            "拒绝回退到本地验证型沙箱。请配置 Linux 容器运行时后重试"
        )
    if not resolved.sandbox_allow_local_fallback:
        raise SandboxBackendError(
            f"{detail}，且 sandbox_allow_local_fallback=False：没有可用沙箱后端"
        )
    if windows_only:
        logger.warning(
            "检测到的容器运行时仅支持 Windows 容器，无法运行 Linux 沙箱镜像；"
            "回退到本地验证型沙箱（非安全隔离）"
        )
    else:
        logger.warning(
            "未检测到 Docker/Podman，回退到本地验证型沙箱（非安全隔离）；"
            "生产环境请安装容器运行时并设置 sandbox_require_isolation=True"
        )
    return SandboxRunner(
        context=context,
        sandbox_dir=sandbox_dir,
        settings=resolved,
        python_executable=python_executable,
    )


def create_sandbox(
    context: ProjectContext | None = None,
    *,
    sandbox_dir: str | Path | None = None,
    settings: Settings | None = None,
    backend: str | None = None,
    python_executable: str | None = None,
) -> SandboxBackend:
    """按配置构造沙箱后端（默认 auto：优先容器，按配置决定是否回退）。

    优先级：``backend`` 参数 > ``settings.sandbox_backend``。后端名称
    未注册时抛 ``SandboxBackendError``，绝不静默回退到本地实现——避免
    配置了云沙箱/容器沙箱却悄悄用验证沙箱执行。``auto`` 只在显式允许
    ``sandbox_allow_local_fallback`` 且未要求强制隔离时回退到 local。
    """
    resolved_settings = settings or (
        context.config if context is not None else get_settings()
    )
    name = (
        backend if backend is not None else resolved_settings.sandbox_backend
    ).strip().lower()
    factory = _BACKEND_FACTORIES.get(name)
    if factory is None:
        raise SandboxBackendError(
            f"未知沙箱后端 {name!r}；已注册后端："
            f"{', '.join(available_sandbox_backends())}"
        )
    return factory(
        context=context,
        sandbox_dir=sandbox_dir,
        settings=resolved_settings,
        python_executable=python_executable,
    )


class SandboxRunner:
    """验证型代码沙箱（子进程执行实现）。"""

    isolation_level = "none"
    backend_name = "local"

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

    def _resolve_workdir(self, workdir: str | Path | None, run_dir: Path) -> Path:
        """解析脚本工作目录（缺省为运行目录，且必须位于沙箱根目录内）。

        Raises:
            SandboxError: 工作目录越界或不存在时。
        """
        if workdir is None:
            return run_dir
        candidate = Path(workdir).expanduser()
        resolved = (
            candidate.resolve()
            if candidate.is_absolute()
            else (self._root / candidate).resolve()
        )
        if not resolved.is_relative_to(self._root):
            raise SandboxError(
                f"工作目录越界：{resolved} 不在沙箱根目录 {self._root} 之内"
            )
        if not resolved.is_dir():
            raise SandboxError(f"工作目录不存在：{resolved}")
        return resolved

    async def _terminate_process_tree(self, process: asyncio.subprocess.Process) -> None:
        """超时/取消时终止整个进程树（子进程可能再派生进程）。"""
        await terminate_process_tree(process)

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
        workdir: str | Path | None = None,
    ) -> ExecutionResult:
        """在沙箱中执行一段 Python 代码并返回结构化执行结果。

        Args:
            code: 待执行的 Python 源码（写入运行目录下的 main.py）。
            task_id: 任务编号（用于目录隔离）。
            run_id: 本次运行编号（同一任务可多次运行）。
            timeout: 本次执行超时（秒）；默认取 ``settings.sandbox_timeout``。
            workdir: 脚本工作目录（相对路径基准，必须在沙箱根目录内）；
                默认（None）为运行目录。代码 Agent 传入沙箱根目录，使脚本
                内的相对路径与 ``write_file`` / ``read_file`` 一致。

        Returns:
            ExecutionResult（ToolResponse 信封的领域扩展）：subtask_id 填
            run_id（沙箱层不感知子任务语义）；success 为 退出码为 0 且未超时；
            output 为 stdout 文本；失败时 error 携带 stderr 摘要或超时说明；
            artifacts 为 stdout.txt / stderr.txt / metadata.json 的路径；
            evidence 为执行证据链；metadata 携带 exit_code、duration_seconds、
            timed_out、stdout_truncated、stderr_truncated 等结构化字段。

        Raises:
            SandboxError: 标识符非法、路径越界、代码为空、工作目录非法，
                或无法启动子进程时。``timeout <= 0`` 表示不设单次命令上限
                （仍可被任务取消 / 上层活动看门狗终止）。
        """
        if not isinstance(code, str) or not code.strip():
            raise SandboxError("code 不能为空")

        effective_timeout = (
            float(timeout) if timeout is not None else float(self._settings.sandbox_timeout)
        )

        output_limit = int(self._settings.sandbox_output_limit)
        run_dir = self._resolve_run_dir(task_id, run_id)
        workspace = self._resolve_workdir(workdir, run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        main_py = run_dir / "main.py"
        main_py.write_text(code, encoding="utf-8")

        command = [self._python, str(main_py)]
        env, scrubbed_env = scrub_env()
        # 脚本所在目录（run_dir）由解释器自动加入 sys.path；工作区根目录
        # 需显式加入，保证跨子任务导入同项目模块时与文件工具基准一致。
        existing_path = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = (
            f"{workspace}{os.pathsep}{existing_path}"
            if existing_path
            else str(workspace)
        )

        started = time.perf_counter()
        timed_out = False
        spawn_kwargs: dict[str, object] = {}
        if os.name != "nt":
            spawn_kwargs["start_new_session"] = True
        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                cwd=str(workspace),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
                **spawn_kwargs,
            )
        except OSError as exc:
            raise SandboxError(f"无法启动沙箱子进程：{exc}") from exc

        try:
            if effective_timeout > 0:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    process.communicate(), timeout=effective_timeout
                )
            else:
                # <=0：不设单次命令上限；取消时仍走下方进程树终止路径
                stdout_bytes, stderr_bytes = await process.communicate()
        except asyncio.TimeoutError:
            timed_out = True
            await self._terminate_process_tree(process)
            stdout_bytes, stderr_bytes = await process.communicate()
        except asyncio.CancelledError:
            await self._terminate_process_tree(process)
            try:
                await process.communicate()
            except Exception:
                pass
            raise

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
            "backend": self.backend_name,
            "isolation_level": self.isolation_level,
            "command": command,
            "cwd": str(workspace),
            "run_dir": str(run_dir),
            "workdir_defaulted": workdir is None,
            "code_file": str(main_py),
            "exit_code": exit_code,
            "duration_seconds": round(duration, 6),
            "timed_out": timed_out,
            "stdout_bytes": raw_stdout_len,
            "stderr_bytes": raw_stderr_len,
            "stdout_truncated": stdout_truncated,
            "stderr_truncated": stderr_truncated,
            "env_scrubbed_count": len(scrubbed_env),
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
            f"backend: {self.backend_name}",
            f"isolation: {self.isolation_level}",
            f"command: {' '.join(command)}",
            f"cwd: {workspace}",
            f"run_dir: {run_dir}",
            f"exit_code: {exit_code}",
            f"duration: {duration:.3f}s",
            f"timed_out: {timed_out}",
            f"env_scrubbed: {len(scrubbed_env)}",
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
            # ToolResponse 统一信封的扩展元数据：与落盘 metadata.json
            # 同名同义，供调用方免解包读取关键诊断字段
            metadata={
                "backend": self.backend_name,
                "isolation_level": self.isolation_level,
                "exit_code": exit_code,
                "duration_seconds": round(duration, 6),
                "timed_out": timed_out,
                "stdout_truncated": stdout_truncated,
                "stderr_truncated": stderr_truncated,
                "workdir": str(workspace),
            },
        )


from so_agent.runtime.sandbox_container import (  # noqa: E402
    ContainerSandbox,
    detect_container_runtime,
)

register_sandbox_backend("local", _create_local_backend)
register_sandbox_backend("docker", _create_docker_backend)
register_sandbox_backend("podman", _create_podman_backend)
register_sandbox_backend("auto", _create_auto_backend)
# 预留云沙箱名称：未注册实现前 fail-closed，提示调用方注入。
register_sandbox_backend("cloud", _create_cloud_backend)

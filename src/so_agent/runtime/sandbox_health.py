"""沙箱能力探测与安全自检。

`doctor` 只做只读探测；`selftest` 会实际执行一段探针代码，验证
文件系统、网络、环境变量与进程边界是否符合预期。自检失败不会让应用
崩溃，但会阻止把结果标记为“已隔离”。
"""

from __future__ import annotations

import asyncio
import json
import os
import platform
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from so_agent.config import Settings, get_settings
from so_agent.runtime.sandbox import create_sandbox
from so_agent.runtime.sandbox_container import detect_container_runtime


@dataclass(frozen=True)
class BackendProbe:
    name: str
    available: bool
    isolation_level: str
    detail: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "available": self.available,
            "isolation_level": self.isolation_level,
            "detail": self.detail,
        }


@dataclass
class SandboxSelfTest:
    backend: str
    isolation_level: str
    passed: bool
    checks: dict[str, bool] = field(default_factory=dict)
    error: str | None = None
    output: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "isolation_level": self.isolation_level,
            "passed": self.passed,
            "checks": dict(self.checks),
            "error": self.error,
            "output": self.output,
        }


def probe_backends(settings: Settings | None = None) -> list[BackendProbe]:
    """探测当前机器可用的沙箱后端（只读）。"""
    resolved = settings or get_settings()
    probes: list[BackendProbe] = []
    for name in ("docker", "podman"):
        runtime = detect_container_runtime(name)
        probes.append(
            BackendProbe(
                name=name,
                available=runtime is not None,
                isolation_level="container",
                detail=(
                    f"{runtime.executable} {runtime.version}".strip()
                    if runtime is not None
                    else "未找到命令或守护进程未启动"
                ),
            )
        )
    probes.append(
        BackendProbe(
            name="local",
            available=True,
            isolation_level="none",
            detail="子进程验证沙箱；不是安全边界，仅限可信代码/开发调试",
        )
    )
    cloud_configured = bool(
        (resolved.sandbox_cloud_provider or "").strip()
        or (resolved.sandbox_cloud_endpoint or "").strip()
    )
    probes.append(
        BackendProbe(
            name="cloud",
            available=False,
            isolation_level="cloud",
            detail=(
                "已配置云沙箱参数，但尚未注册实现；请调用 "
                "register_sandbox_backend('cloud', factory) 注入"
                if cloud_configured
                else "未配置云沙箱；保留接口"
            ),
        )
    )
    return probes


def resolve_backend_name(settings: Settings | None = None) -> str:
    """返回当前配置最终会使用的后端名称。"""
    resolved = settings or get_settings()
    configured = (resolved.sandbox_backend or "auto").strip().lower()
    if configured != "auto":
        return configured
    runtime = detect_container_runtime("auto")
    if runtime is not None:
        return runtime.name
    if resolved.sandbox_require_isolation:
        return "none"
    if resolved.sandbox_allow_local_fallback:
        return "local"
    return "none"


def doctor_report(settings: Settings | None = None) -> dict[str, Any]:
    resolved = settings or get_settings()
    probes = probe_backends(resolved)
    return {
        "os": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
        },
        "configured_backend": resolved.sandbox_backend,
        "resolved_backend": resolve_backend_name(resolved),
        "require_isolation": resolved.sandbox_require_isolation,
        "allow_local_fallback": resolved.sandbox_allow_local_fallback,
        "container_image": resolved.sandbox_container_image,
        "container_network": resolved.sandbox_container_network,
        "backends": [probe.as_dict() for probe in probes],
        "wsl": shutil.which("wsl") is not None,
    }


def _normalize_backend(settings: Settings, backend: str | None) -> str:
    selected = (backend or settings.sandbox_backend or "auto").strip().lower()
    if selected == "auto":
        resolved = resolve_backend_name(settings)
        if resolved == "none":
            raise RuntimeError(
                "未检测到容器运行时，且当前配置禁止回退到本地沙箱；"
                "请先安装 Docker/Podman 或调整 sandbox_allow_local_fallback"
            )
        return resolved
    return selected


async def run_selftest_async(
    settings: Settings | None = None,
    *,
    backend: str | None = None,
    sandbox_dir: str | Path | None = None,
) -> SandboxSelfTest:
    """异步执行沙箱安全自检（见 :func:`run_selftest`）。"""
    resolved = settings or get_settings()
    root = Path(
        sandbox_dir or (Path.cwd() / "sandbox" / "selftest")
    ).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    token = os.urandom(8).hex()
    # 随机目录避免与容器内 /tmp 等既有路径重名，保证“工作目录外写入”
    # 探针只反映真实逃逸，而不是宿主沙箱目录本身的巧合。
    probe_dir = root / f"escape_probe_{token}"
    probe_dir.mkdir(parents=True, exist_ok=True)
    marker = probe_dir / "marker.txt"
    write_probe = probe_dir / "write_probe.txt"
    marker.write_text(token, encoding="utf-8")
    try:
        write_probe.unlink(missing_ok=True)
    except OSError:
        pass

    script = f'''
import json, os, socket
result = {{"read_escape": False, "write_escape": False, "network_escape": False, "env_escape": False}}
try:
    with open({json.dumps(str(marker))!s}, "r", encoding="utf-8") as handle:
        result["read_escape"] = handle.read().strip() == {json.dumps(token)!s}
except Exception:
    pass
try:
    with open({json.dumps(str(write_probe))!s}, "w", encoding="utf-8") as handle:
        handle.write("escape")
    result["write_escape"] = True
except Exception:
    pass
try:
    sock = socket.create_connection(("1.1.1.1", 443), timeout=1.5)
    sock.close()
    result["network_escape"] = True
except Exception:
    pass
result["env_escape"] = bool(os.environ.get("SO_AGENT_SELFTEST_SECRET"))
print("SO_AGENT_SELFTEST=" + json.dumps(result, sort_keys=True))
'''

    selected = _normalize_backend(resolved, backend)
    previous = os.environ.get("SO_AGENT_SELFTEST_SECRET")
    os.environ["SO_AGENT_SELFTEST_SECRET"] = token
    try:
        sandbox = create_sandbox(
            sandbox_dir=root,
            settings=resolved,
            backend=selected,
        )
        isolation = str(getattr(sandbox, "isolation_level", "unknown"))
        result = await sandbox.execute(script, "selftest", "run1", timeout=30)
    except Exception as exc:  # noqa: BLE001
        return SandboxSelfTest(
            backend=selected,
            isolation_level="unknown",
            passed=False,
            error=f"{type(exc).__name__}: {exc}",
        )
    finally:
        if previous is None:
            os.environ.pop("SO_AGENT_SELFTEST_SECRET", None)
        else:
            os.environ["SO_AGENT_SELFTEST_SECRET"] = previous
        shutil.rmtree(probe_dir, ignore_errors=True)

    checks: dict[str, bool] = {}
    for line in (result.output or "").splitlines():
        if line.startswith("SO_AGENT_SELFTEST="):
            try:
                checks = json.loads(line.split("=", 1)[1])
            except json.JSONDecodeError:
                checks = {}
            break
    escaped = any(bool(value) for value in checks.values())
    passed = bool(result.success) and bool(checks) and not escaped
    return SandboxSelfTest(
        backend=selected,
        isolation_level=isolation,
        passed=passed,
        checks={str(key): bool(value) for key, value in checks.items()},
        error=result.error,
        output=(result.output or "").strip()[:2000],
    )


def run_selftest(
    settings: Settings | None = None,
    *,
    backend: str | None = None,
    sandbox_dir: str | Path | None = None,
) -> SandboxSelfTest:
    """同步执行沙箱安全自检（CLI 入口）。

    在已有事件循环的进程中请改用 ``await run_selftest_async(...)``。
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(
            run_selftest_async(
                settings, backend=backend, sandbox_dir=sandbox_dir
            )
        )
    raise RuntimeError(
        "当前事件循环中不能调用同步版 run_selftest，请使用 run_selftest_async"
    )


__all__ = [
    "BackendProbe",
    "SandboxSelfTest",
    "doctor_report",
    "probe_backends",
    "resolve_backend_name",
    "run_selftest",
    "run_selftest_async",
]

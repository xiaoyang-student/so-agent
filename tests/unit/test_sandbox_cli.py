"""单元测试：沙箱 CLI、自动后端选择与容器安全参数（任务 101）。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from so_agent.config import Settings
from so_agent.runtime import sandbox as sandbox_module
from so_agent.runtime.sandbox import (
    SandboxBackendError,
    SandboxRunner,
    create_sandbox,
)
from so_agent.runtime.sandbox_container import ContainerRuntime, ContainerSandbox
from so_agent.runtime.sandbox_health import (
    doctor_report,
    probe_backends,
    resolve_backend_name,
    run_selftest,
)
from so_agent import sandbox_cli


def _settings(**overrides: object) -> Settings:
    base = {
        "sandbox_backend": "auto",
        "sandbox_allow_local_fallback": True,
        "sandbox_require_isolation": False,
    }
    base.update(overrides)
    return Settings(**base)  # type: ignore[arg-type]


def test_auto_falls_back_to_local_without_runtime(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(sandbox_module, "detect_container_runtime", lambda *a, **k: None)
    sandbox = create_sandbox(sandbox_dir=tmp_path, settings=_settings())
    assert isinstance(sandbox, SandboxRunner)
    assert sandbox.isolation_level == "none"


def test_auto_with_require_isolation_fails_closed(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(sandbox_module, "detect_container_runtime", lambda *a, **k: None)
    with pytest.raises(SandboxBackendError) as excinfo:
        create_sandbox(
            sandbox_dir=tmp_path,
            settings=_settings(sandbox_require_isolation=True),
        )
    assert "sandbox_require_isolation" in str(excinfo.value)


def test_auto_without_fallback_fails_closed(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(sandbox_module, "detect_container_runtime", lambda *a, **k: None)
    with pytest.raises(SandboxBackendError):
        create_sandbox(
            sandbox_dir=tmp_path,
            settings=_settings(sandbox_allow_local_fallback=False),
        )


def test_doctor_report_and_backend_resolution() -> None:
    settings = _settings(sandbox_backend="local")
    report = doctor_report(settings)
    assert report["configured_backend"] == "local"
    assert report["resolved_backend"] == "local"
    assert {item["name"] for item in report["backends"]} >= {"local", "docker", "podman", "cloud"}
    assert resolve_backend_name(settings) == "local"
    names = {probe.name for probe in probe_backends(settings)}
    assert "cloud" in names


def test_local_selftest_reports_escapes(tmp_path: Path) -> None:
    result = run_selftest(
        _settings(sandbox_backend="local"),
        sandbox_dir=tmp_path / "selftest",
    )
    assert result.backend == "local"
    assert result.isolation_level == "none"
    assert result.passed is False
    assert result.checks.get("read_escape") is True
    assert result.checks.get("write_escape") is True
    assert result.checks.get("env_escape") is False


def test_cli_doctor_json(capsys) -> None:
    code = sandbox_cli.main(["doctor", "--json"])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert "resolved_backend" in payload
    assert "backends" in payload


def test_cli_setup_without_runtime(monkeypatch, capsys) -> None:
    monkeypatch.setattr(sandbox_cli, "detect_container_runtime", lambda *a, **k: None)
    monkeypatch.setattr(sandbox_cli, "get_settings", lambda: _settings())
    code = sandbox_cli.main(["setup", "--json"])
    assert code == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["runtime"] is None
    assert payload["hints"]


def test_cli_setup_writes_only_missing_env(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("SO_AGENT_SANDBOX_BACKEND=docker\nOTHER=1\n", encoding="utf-8")
    runtime = ContainerRuntime(name="docker", executable="docker", version="27.0")
    monkeypatch.setattr(
        sandbox_cli, "detect_container_runtime", lambda *a, **k: runtime
    )
    monkeypatch.setattr(sandbox_cli, "get_settings", lambda: _settings())
    monkeypatch.setattr(sandbox_cli, "_image_present", lambda *a, **k: True)

    code = sandbox_cli.main(
        [
            "setup",
            "--json",
            "--write-env",
            str(env_file),
            "--require-isolation",
        ]
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["image_present"] is True
    assert "SO_AGENT_SANDBOX_BACKEND" not in payload["env_written"]

    content = env_file.read_text(encoding="utf-8")
    assert "SO_AGENT_SANDBOX_BACKEND=docker" in content
    assert "SO_AGENT_SANDBOX_REQUIRE_ISOLATION=true" in content
    assert "OTHER=1" in content


def test_container_command_has_security_flags(tmp_path: Path) -> None:
    runtime = ContainerRuntime(name="docker", executable="docker", version="27.0")
    sandbox = ContainerSandbox(
        sandbox_dir=tmp_path,
        settings=Settings(sandbox_container_image="python:3.12-slim"),
        runtime=runtime,
    )
    run_dir = tmp_path / "task_t1" / "run_r1"
    run_dir.mkdir(parents=True)
    command, cidfile = sandbox._build_command(
        task_id="t1", run_id="r1", run_dir=run_dir, workspace=tmp_path
    )
    joined = " ".join(command)

    assert command[:2] == ["docker", "run"]
    assert "--rm" in command
    assert "--read-only" in command
    assert "--privileged" not in command
    assert ["--network", "none"] == command[
        command.index("--network") : command.index("--network") + 2
    ]
    assert "--cap-drop" in command
    assert command[command.index("--cap-drop") + 1] == "ALL"
    assert "--security-opt" in command
    assert command[command.index("--security-opt") + 1] == "no-new-privileges"
    assert "--pids-limit" in command
    assert "--memory" in command
    # 工作区（沙箱根目录）整体挂载为 /work：脚本相对路径与文件工具一致
    assert f"source={tmp_path}" in joined
    assert "target=/work" in joined
    assert command[-3:] == [
        "python:3.12-slim",
        "python",
        "/work/task_t1/run_r1/main.py",
    ]
    assert str(cidfile).endswith(".container_id")

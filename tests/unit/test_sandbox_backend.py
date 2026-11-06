"""单元测试：沙箱后端契约、工厂与配置切换（任务 32）。"""

from __future__ import annotations

from pathlib import Path

import pytest

from so_agent.config import Settings
from so_agent.models import ExecutionResult
from so_agent.runtime.sandbox import (
    SandboxBackend,
    SandboxBackendError,
    SandboxRunner,
    _BACKEND_FACTORIES,
    available_sandbox_backends,
    create_sandbox,
    register_sandbox_backend,
)


class _FakeBackend:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs

    async def execute(
        self,
        code: str,
        task_id: str,
        run_id: str,
        *,
        timeout: float | None = None,
        workdir: str | Path | None = None,
    ) -> ExecutionResult:
        return ExecutionResult(
            subtask_id=run_id,
            success=True,
            output=f"{task_id}:{code}",
        )


@pytest.fixture(autouse=True)
def _restore_backend_registry():
    snapshot = dict(_BACKEND_FACTORIES)
    yield
    _BACKEND_FACTORIES.clear()
    _BACKEND_FACTORIES.update(snapshot)


def test_default_backend_is_local_runner(tmp_path: Path) -> None:
    sandbox = create_sandbox(sandbox_dir=tmp_path, settings=Settings())
    assert isinstance(sandbox, SandboxRunner)
    assert isinstance(sandbox, SandboxBackend)
    assert sandbox.sandbox_root == tmp_path.resolve()


def test_settings_select_backend(tmp_path: Path) -> None:
    register_sandbox_backend("fake-test", _FakeBackend)
    sandbox = create_sandbox(
        sandbox_dir=tmp_path,
        settings=Settings(sandbox_backend=" FAKE-Test "),
    )
    assert isinstance(sandbox, _FakeBackend)
    assert sandbox.kwargs["sandbox_dir"] == tmp_path


def test_explicit_backend_overrides_settings(tmp_path: Path) -> None:
    register_sandbox_backend("fake-test", _FakeBackend)
    sandbox = create_sandbox(
        sandbox_dir=tmp_path,
        settings=Settings(sandbox_backend="local"),
        backend="fake-test",
    )
    assert isinstance(sandbox, _FakeBackend)


def test_unknown_backend_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(SandboxBackendError) as excinfo:
        create_sandbox(
            sandbox_dir=tmp_path,
            settings=Settings(sandbox_backend="no-such-backend"),
        )
    message = str(excinfo.value)
    assert "no-such-backend" in message
    assert "local" in message


def test_cloud_backend_reserved_but_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(SandboxBackendError) as excinfo:
        create_sandbox(
            sandbox_dir=tmp_path,
            settings=Settings(sandbox_backend="cloud"),
        )
    assert "cloud" in str(excinfo.value)
    assert "尚未提供实现" in str(excinfo.value)


def test_register_rejects_empty_name_and_non_callable() -> None:
    with pytest.raises(SandboxBackendError):
        register_sandbox_backend("   ", _FakeBackend)
    with pytest.raises(SandboxBackendError):
        register_sandbox_backend("bad", object())  # type: ignore[arg-type]


def test_available_backends_include_local_and_cloud() -> None:
    names = available_sandbox_backends()
    assert "local" in names
    assert "cloud" in names


async def test_fake_backend_is_usable_through_contract(tmp_path: Path) -> None:
    register_sandbox_backend("fake-test", _FakeBackend)
    sandbox = create_sandbox(
        sandbox_dir=tmp_path,
        settings=Settings(sandbox_backend="fake-test"),
    )
    result = await sandbox.execute("print('x')", "t1", "run1")
    assert result.success is True
    assert result.subtask_id == "run1"
    assert result.output == "t1:print('x')"

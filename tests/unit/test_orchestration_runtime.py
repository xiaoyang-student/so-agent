from __future__ import annotations

import importlib.util

import pytest

from so_agent.models import TaskStatus
from so_agent.orchestrator import OrchestratorOutput
from so_agent.runtime.orchestration_runtime import (
    NativeRuntime,
    OrchestrationRuntime,
    OrchestrationRuntimeError,
    create_orchestration_runtime,
    ensure_runtime_available,
)


class FakeEngine:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, str]] = []
        self.cancelled = False
        self.provisional = OrchestratorOutput(
            task_id="provisional", status=TaskStatus.COMPLETED, final_result="p"
        )

    async def run(
        self, objective: str, constraints: str = "", acceptance_criteria: str = ""
    ) -> OrchestratorOutput:
        self.calls.append((objective, constraints, acceptance_criteria))
        return OrchestratorOutput(
            task_id="task-1", status=TaskStatus.COMPLETED, final_result="done"
        )

    def cancel(self) -> None:
        self.cancelled = True

    async def wait_for_provisional(self) -> OrchestratorOutput | None:
        return self.provisional


@pytest.mark.asyncio
async def test_native_runtime_delegates_lifecycle() -> None:
    engine = FakeEngine()
    runtime = create_orchestration_runtime(engine)

    assert isinstance(runtime, OrchestrationRuntime)
    assert isinstance(runtime, NativeRuntime)
    output = await runtime.run("目标", constraints="c", acceptance_criteria="a")
    assert output.task_id == "task-1"
    assert engine.calls == [("目标", "c", "a")]

    runtime.cancel()
    assert engine.cancelled is True
    assert await runtime.wait_for_provisional() is engine.provisional


@pytest.mark.asyncio
async def test_langgraph_runtime_is_optional_adapter() -> None:
    if importlib.util.find_spec("langgraph") is None:
        pytest.skip("langgraph 未安装")
    from langgraph.checkpoint.memory import MemorySaver

    engine = FakeEngine()
    runtime = create_orchestration_runtime(
        engine, backend="langgraph", checkpointer=MemorySaver(), thread_id="t-1"
    )
    assert runtime.name == "langgraph"

    output = await runtime.run("目标", "c", "a")
    assert output.task_id == "task-1"
    assert engine.calls == [("目标", "c", "a")]

    runtime.cancel()
    assert engine.cancelled is True


def test_unknown_runtime_name_fails_closed() -> None:
    with pytest.raises(OrchestrationRuntimeError):
        create_orchestration_runtime(FakeEngine(), backend="unknown")
    ensure_runtime_available("native")
    if importlib.util.find_spec("langgraph") is None:
        # 可选依赖未安装时必须启动期失败，而不是运行时才暴露
        with pytest.raises(OrchestrationRuntimeError):
            ensure_runtime_available("langgraph")
    else:
        ensure_runtime_available("langgraph")

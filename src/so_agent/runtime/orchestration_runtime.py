"""编排运行时边界：原生实现默认，LangGraph 仅作为可选外层适配器。

``WorkflowEngine`` 仍是唯一的编排内核：分阶段状态机、计划/执行/汇总
协调器、人工兜底与检查点语义都不变。本模块只定义调用边界，便于
需要 LangGraph 基础设施的团队按其偏好托管「一次任务」的调用与取消，
而不是把内核重写为图：

- :class:`OrchestrationRuntime`：``run`` / ``cancel`` /
  ``wait_for_provisional`` 三个生命周期方法；
- :class:`NativeRuntime`：直接委托 ``WorkflowEngine``（默认）；
- :class:`LangGraphRuntime`：惰性导入 ``langgraph``，用单节点状态图
  调用同一个引擎；未安装时构造失败，内核不引入硬依赖。

配置项 ``orchestration_runtime=native|langgraph`` 选择实现。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable
from uuid import uuid4

from so_agent.orchestrator import OrchestratorOutput


class OrchestrationRuntimeError(Exception):
    """编排运行时不可用或名称未知。"""


@runtime_checkable
class OrchestrationRuntime(Protocol):
    """编排生命周期边界（不暴露内核内部状态）。"""

    name: str

    async def run(
        self,
        objective: str,
        constraints: str = "",
        acceptance_criteria: str = "",
    ) -> OrchestratorOutput:
        ...

    def cancel(self) -> None:
        ...

    async def wait_for_provisional(self) -> OrchestratorOutput | None:
        ...


@dataclass
class NativeRuntime:
    """原生运行时：所有语义由 ``WorkflowEngine`` 提供。"""

    engine: Any
    name: str = "native"

    async def run(
        self,
        objective: str,
        constraints: str = "",
        acceptance_criteria: str = "",
    ) -> OrchestratorOutput:
        return await self.engine.run(objective, constraints, acceptance_criteria)

    def cancel(self) -> None:
        self.engine.cancel()

    async def wait_for_provisional(self) -> OrchestratorOutput | None:
        return await self.engine.wait_for_provisional()


class LangGraphRuntime:
    """可选的 LangGraph 外层适配器（任务粒度，不改变内核分阶段语义）。

    ``checkpointer`` 与 ``thread_id`` 原样交给 LangGraph；未提供
    ``thread_id`` 时为每次运行时生成一个。状态图只有一个节点，节点内
    调用原生引擎，因此 HITL、重试与检查点仍由内核控制。
    """

    name = "langgraph"

    def __init__(
        self,
        engine: Any,
        *,
        checkpointer: Any = None,
        thread_id: str | None = None,
    ) -> None:
        try:
            from langgraph.graph import END, START, StateGraph
        except ImportError as exc:  # pragma: no cover - 取决于安装环境
            raise OrchestrationRuntimeError(
                "已选择 LangGraph 运行时，但未安装 langgraph；"
                "请安装或改回 orchestration_runtime=native"
            ) from exc
        from typing import TypedDict

        class TaskState(TypedDict, total=False):
            objective: str
            constraints: str
            acceptance_criteria: str
            output: OrchestratorOutput

        async def execute(state: TaskState) -> dict[str, Any]:
            output = await self.engine.run(
                state["objective"],
                state.get("constraints", ""),
                state.get("acceptance_criteria", ""),
            )
            return {"output": output}

        graph = StateGraph(TaskState)
        graph.add_node("so_agent", execute)
        graph.add_edge(START, "so_agent")
        graph.add_edge("so_agent", END)
        self.engine = engine
        self._thread_id = thread_id or f"so-agent-{uuid4()}"
        self._graph = graph.compile(checkpointer=checkpointer)

    async def run(
        self,
        objective: str,
        constraints: str = "",
        acceptance_criteria: str = "",
    ) -> OrchestratorOutput:
        config = {"configurable": {"thread_id": self._thread_id}}
        result = await self._graph.ainvoke(
            {
                "objective": objective,
                "constraints": constraints,
                "acceptance_criteria": acceptance_criteria,
            },
            config,
        )
        return result["output"]

    def cancel(self) -> None:
        self.engine.cancel()

    async def wait_for_provisional(self) -> OrchestratorOutput | None:
        return await self.engine.wait_for_provisional()


def create_orchestration_runtime(
    engine: Any,
    settings: Any = None,
    *,
    backend: str | None = None,
    checkpointer: Any = None,
    thread_id: str | None = None,
) -> OrchestrationRuntime:
    """按配置构造编排运行时；未知名称与缺失依赖均 fail-closed。"""
    name = str(
        backend
        if backend is not None
        else getattr(settings, "orchestration_runtime", "native")
    ).strip().lower()
    if name == "native":
        return NativeRuntime(engine)
    if name == "langgraph":
        return LangGraphRuntime(
            engine, checkpointer=checkpointer, thread_id=thread_id
        )
    raise OrchestrationRuntimeError(
        f"未知编排运行时 {name!r}；可选：native, langgraph"
    )


def ensure_runtime_available(name: str) -> None:
    """启动期校验：仅检查依赖是否可用，不构造引擎。"""
    if name.strip().lower() != "langgraph":
        return
    try:
        from langgraph.graph import StateGraph  # noqa: F401
    except ImportError as exc:
        raise OrchestrationRuntimeError(
            "已选择 LangGraph 运行时，但未安装 langgraph；"
            "请安装或改回 orchestration_runtime=native"
        ) from exc

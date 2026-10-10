"""基准基座共享假件：假 Runner、剧本主管、评审脚本与引擎组装。

从既有集成测试的成熟模式（tests/integration/test_workflow.py、
test_replan_reuse.py、test_workflow_timeout.py）提取归一，消除重复
定义；行为语义与原实现一致：

- 主管调用经 ``run_streamed``（run_agent_streamed 的流式入口），评审
  调用仍走 ``run``，两者共用按 Agent 身份分派的假 Runner；
- supervisor 为同步或异步 callable（``payload: dict -> OrchestratorOutput``），
  异步 callable 用于模拟挂起（超时场景）；
- reviews 为 callable（``ReviewInput -> ReviewDecision``）。
"""

from __future__ import annotations

import inspect
import json
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Callable

import so_agent.workflow as workflow_mod
from so_agent.config import Settings
from so_agent.context import ProjectContext
from so_agent.models import (
    ExecutionResult,
    ReviewDecision,
    SubtaskSpec,
    TaskStatus,
)
from so_agent.orchestrator import OrchestratorOutput
from so_agent.result_store import SQLiteResultStore
from so_agent.runtime.events import EVENT_STATUS_TRANSITION
from so_agent.tool_packages.review_agent.agent import ReviewInput, ReviewStage
from so_agent.workflow import WorkflowEngine

# 主管 / 评审假件回调类型别名
SupervisorFn = Callable[[dict], Any]
ReviewFn = Callable[[ReviewInput], ReviewDecision]


class FakeRunResult:
    """假 RunResult：只保留 final_output，供控制层结构与文本解析。"""

    def __init__(self, final_output: Any) -> None:
        self.final_output = final_output


class FakeStreamedResult:
    """假 RunResultStreaming：首次取流式事件时求值协程工厂并回填输出。

    正常路径下取到假事件后迭代结束（模拟"流正常结束"）；supervisor
    抛出的异常（网络失败等）经 stream_events 原样透传；``cancel`` 记录
    超时收尾路径。
    """

    def __init__(self, run_call: Callable[[], Any]) -> None:
        self._run_call = run_call
        self.final_output: Any = None
        self.raw_responses: list[Any] = []
        self.cancelled = False

    def cancel(self, mode: str = "immediate") -> None:
        self.cancelled = True

    async def stream_events(self):
        result = await self._run_call()
        self.final_output = result.final_output
        self.raw_responses = getattr(result, "raw_responses", None) or []
        yield {"type": "fake_event"}


def install_fake_runner(
    monkeypatch,
    *,
    orchestrator: Any,
    review_agent: Any,
    supervisor: SupervisorFn,
    reviews: ReviewFn,
) -> None:
    """把 workflow 模块内的 Runner 替换为按 Agent 身份分派的假 Runner。"""

    class FakeRunner:
        @classmethod
        async def run(cls, starting_agent, input=None, *, max_turns=None, **kwargs):
            if starting_agent is review_agent:
                review_input = ReviewInput.model_validate_json(input)
                return FakeRunResult(reviews(review_input))
            payload = json.loads(input)
            outcome = supervisor(payload)
            if inspect.isawaitable(outcome):
                outcome = await outcome
            return FakeRunResult(outcome)

        @classmethod
        def run_streamed(cls, starting_agent, input=None, *, max_turns=None, **kwargs):
            return FakeStreamedResult(
                lambda: cls.run(
                    starting_agent, input=input, max_turns=max_turns, **kwargs
                )
            )

    monkeypatch.setattr(workflow_mod, "Runner", FakeRunner)


def make_spec(
    subtask_id: str,
    *,
    instructions: str | None = None,
    dependencies: list[str] | None = None,
    tools: list[str] | None = None,
    reuse_from: str | None = None,
) -> SubtaskSpec:
    """构造基准子任务规格。"""
    return SubtaskSpec(
        subtask_id=subtask_id,
        title=f"子任务 {subtask_id}",
        instructions=instructions if instructions is not None else f"完成子任务 {subtask_id}",
        role="code",
        dependencies=list(dependencies or []),
        allowed_tools=list(tools or []),
        max_attempts=3,
        reuse_from=reuse_from,
    )


@dataclass
class EngineHarness:
    """引擎组装产物：上下文、引擎、配置与可选的 SQLite 结果存储。"""

    context: ProjectContext
    engine: WorkflowEngine
    settings: Settings
    orchestrator: Any
    review_agent: Any
    result_store: SQLiteResultStore | None = None


def build_engine(
    tmp_path,
    monkeypatch,
    *,
    supervisor: SupervisorFn,
    reviews: ReviewFn,
    settings_overrides: dict[str, Any] | None = None,
    use_sqlite_store: bool = True,
) -> EngineHarness:
    """组装注入假主管/评审 Agent 的工作流引擎（默认挂 SQLite 存储）。"""
    context = ProjectContext(sandbox_dir=tmp_path, project_name="baseline")
    settings = Settings(_env_file=None, **(settings_overrides or {}))
    orchestrator = SimpleNamespace(name="fake-supervisor")
    review_agent = SimpleNamespace(name="fake-reviewer")
    install_fake_runner(
        monkeypatch,
        orchestrator=orchestrator,
        review_agent=review_agent,
        supervisor=supervisor,
        reviews=reviews,
    )
    store = (
        SQLiteResultStore(str(tmp_path / "result_store.db"))
        if use_sqlite_store
        else None
    )
    engine = WorkflowEngine(
        context,
        settings=settings,
        orchestrator=orchestrator,
        review_agent=review_agent,
        result_store=store,
    )
    return EngineHarness(
        context=context,
        engine=engine,
        settings=settings,
        orchestrator=orchestrator,
        review_agent=review_agent,
        result_store=store,
    )


def status_transitions(context: ProjectContext) -> list[tuple[str, str]]:
    """提取状态迁移序列 ``[(from_status, to_status), ...]``。"""
    return [
        (record["payload"]["from_status"], record["payload"]["to_status"])
        for record in context.event_log
        if record["event_type"] == EVENT_STATUS_TRANSITION
    ]


class ScriptedSupervisor:
    """按计划轮次返回不同拆分方案的剧本主管（合并既有两种假主管能力）。

    - ``plans``：按 planning/replanning 调用轮次取用的计划剧本；
    - ``fail_ids`` / ``fail_version``：指定计划版本的指定子任务持续
      返回业务失败（模拟质量失败）；
    - ``aggregate_summary``：汇总阶段返回文本；None 表示返回空文本
      （触发本地确定性摘要兜底）；
    - 执行输出文本形如 ``s1#1``（第 N 次执行的产出），便于断言复用来源。

    记录 ``payloads``（全部阶段载荷）、``exec_sequence``（执行子任务
    顺序）与 ``exec_counts``（各子任务执行次数，含失败尝试）。
    """

    def __init__(
        self,
        *,
        plans: list[list[SubtaskSpec]],
        fail_version: int = 1,
        fail_ids: set[str] | None = None,
        aggregate_summary: str | None = "聚合最终结果",
    ) -> None:
        self.plans = [list(plan) for plan in plans]
        self.fail_version = fail_version
        self.fail_ids = set(fail_ids or set())
        self.aggregate_summary = aggregate_summary
        self.payloads: list[dict] = []
        self.exec_sequence: list[str] = []
        self.exec_counts: dict[str, int] = {}

    @property
    def phases(self) -> list[str]:
        return [payload.get("phase") for payload in self.payloads]

    def _plan_index(self) -> int:
        plan_calls = sum(
            1
            for item in self.payloads
            if item.get("phase") in ("planning", "replanning")
        )
        return min(plan_calls - 1, len(self.plans) - 1)

    def __call__(self, payload: dict) -> OrchestratorOutput:
        self.payloads.append(payload)
        phase = payload.get("phase")
        if phase in ("planning", "replanning"):
            index = self._plan_index()
            plan = [spec.model_copy(deep=True) for spec in self.plans[index]]
            return OrchestratorOutput(
                task_id=str(payload.get("task_id") or "stub-task"),
                status=(
                    TaskStatus.REPLANNING
                    if phase == "replanning"
                    else TaskStatus.PLANNING
                ),
                plan_versions=[plan],
            )
        if phase == "execute_subtask":
            return self._execution_output(payload)
        if phase == "aggregating":
            return OrchestratorOutput(
                task_id=str(payload.get("task_id") or "stub-task"),
                status=TaskStatus.AGGREGATING,
                final_result=self.aggregate_summary or "",
            )
        raise AssertionError(f"未预期的阶段载荷：{phase}")

    def _execution_output(self, payload: dict) -> OrchestratorOutput:
        spec = payload["subtask"]
        subtask_id = spec["subtask_id"]
        self.exec_sequence.append(subtask_id)
        self.exec_counts[subtask_id] = self.exec_counts.get(subtask_id, 0) + 1
        if subtask_id in self.fail_ids and spec["plan_version"] == self.fail_version:
            result = ExecutionResult(
                subtask_id=subtask_id,
                success=False,
                error=f"{subtask_id} 持续失败（模拟）",
            )
        else:
            result = ExecutionResult(
                subtask_id=subtask_id,
                success=True,
                output=f"{subtask_id}#{self.exec_counts[subtask_id]}",
            )
        return OrchestratorOutput(
            task_id=str(payload.get("task_id") or "stub-task"),
            status=TaskStatus.EXECUTING,
            final_result=result.model_dump_json(),
        )


class ReviewScript:
    """评审脚本：计划/授权评审按批次，结果评审按序列（默认全部通过）。"""

    def __init__(
        self,
        plan_batch_results=None,
        *,
        result_results: bool | list[bool] = True,
        result_retry_target: str | None = "s1",
    ) -> None:
        self.plan_batch_results = list(plan_batch_results) or [True]
        if isinstance(result_results, bool):
            result_results = [result_results]
        self.result_results = list(result_results)
        self.result_retry_target = result_retry_target
        self.calls: list[ReviewInput] = []
        self.decisions: list[ReviewDecision] = []
        self._result_index = 0

    def __call__(self, review_input: ReviewInput) -> ReviewDecision:
        self.calls.append(review_input)
        if review_input.stage == ReviewStage.RESULT_REVIEW:
            passed = self.result_results[
                min(self._result_index, len(self.result_results) - 1)
            ]
            self._result_index += 1
            decision = ReviewDecision(
                stage="result_review",
                passed=passed,
                issues=[] if passed else ["整体结果不满足验收标准"],
                required_fixes=[] if passed else ["修复整体结果后重新提交"],
                retry_target=None if passed else self.result_retry_target,
                summary="结果评审（模拟）",
            )
        else:
            plan_calls = sum(
                1
                for call in self.calls
                if call.stage != ReviewStage.RESULT_REVIEW
            )
            index = min((plan_calls - 1) // 2, len(self.plan_batch_results) - 1)
            passed = self.plan_batch_results[index]
            decision = ReviewDecision(
                stage="plan_review",
                passed=passed,
                issues=[] if passed else ["子任务拆分粒度过粗"],
                required_fixes=[] if passed else ["重新拆分并补充依赖"],
                summary=f"计划评审（模拟，第 {index + 1} 批）",
            )
        self.decisions.append(decision)
        return decision

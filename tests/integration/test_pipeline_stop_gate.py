"""集成测试：pipeline 停止门端到端行为（任务 45）。

覆盖任务 45 验收点（引擎级）：

1. pipeline deadline 到期：优雅降级为 human_required，停止门原子进入
   stopping，执行链零新增 dispatch（依赖链后续子任务不再发起）；
2. ``_authoritative_success_results``：跨版本归档（新版本覆盖旧版本）
   → 当前结果 → 调度器清理期落账的合并视图，只保留 success（部分
   结果同样准确，不依赖被重规划修改的临时 pending 集合）；
3. 迟到结果隔离：主管调用等待期间外部取消 → 返回结果视为迟到，不写
   状态、不记用量、不触发后续动作，任务映射为 cancelled 终态；
4. 授权最终清零：任何出口（含取消）的 ``finally`` 幂等撤销全部授权。
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

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
from so_agent.workflow import WorkflowEngine


def make_spec(subtask_id: str, *, deps: list[str] | None = None) -> SubtaskSpec:
    return SubtaskSpec(
        subtask_id=subtask_id,
        title=f"模拟子任务 {subtask_id}",
        instructions=f"完成子任务 {subtask_id}",
        role="code",
        allowed_tools=["read_file"],
        dependencies=list(deps or []),
        max_attempts=3,
    )


class FakeRunResult:
    """假 RunResult：只保留 final_output。"""

    def __init__(self, final_output):
        self.final_output = final_output


class FakeStreamedResult:
    """假 RunResultStreaming：handler 求值后产出单个占位流事件。"""

    def __init__(self, run_call):
        self._run_call = run_call
        self.final_output = None
        self.raw_responses = []
        self.cancelled = False

    def cancel(self, mode="immediate"):
        self.cancelled = True

    async def stream_events(self):
        result = await self._run_call()
        self.final_output = result.final_output
        self.raw_responses = getattr(result, "raw_responses", None) or []
        yield {"type": "fake_event"}


def build_engine(tmp_path, monkeypatch, *, settings, handler) -> WorkflowEngine:
    """组装引擎：按身份分派的假 Runner（handler(is_review, input)）。"""
    context = ProjectContext(sandbox_dir=tmp_path, project_name="stop-gate-test")
    orchestrator = SimpleNamespace(name="fake-supervisor")
    review_agent = SimpleNamespace(name="fake-reviewer")

    class FakeRunner:
        @classmethod
        async def run(cls, starting_agent, input=None, *, max_turns=None, **kwargs):
            output = await handler(starting_agent is review_agent, input or "")
            return FakeRunResult(output)

        @classmethod
        def run_streamed(cls, starting_agent, input=None, *, max_turns=None, **kwargs):
            return FakeStreamedResult(
                lambda: cls.run(
                    starting_agent, input=input, max_turns=max_turns, **kwargs
                )
            )

    monkeypatch.setattr(workflow_mod, "Runner", FakeRunner)
    return WorkflowEngine(
        context,
        settings=settings,
        orchestrator=orchestrator,
        review_agent=review_agent,
    )


class TestPipelineTimeoutStopsEverything:
    """pipeline deadline 到期：停止门原子 stopping，零新增 dispatch。"""

    async def test_timeout_blocks_dispatch_of_dependent_subtask(
        self, tmp_path, monkeypatch
    ):
        """依赖链 s1 → s2：s1 执行挂起 → deadline 到期 → s2 零 dispatch。"""
        dispatch_inputs: list[str] = []
        holder: dict = {}

        async def handler(is_review: bool, raw_input: str):
            if is_review:
                return ReviewDecision(
                    stage="plan_review", passed=True, summary="计划评审（模拟）"
                )
            payload = json.loads(raw_input)
            phase = payload.get("phase")
            if phase in ("planning", "replanning"):
                specs = [make_spec("s1"), make_spec("s2", deps=["s1"])]
                return OrchestratorOutput(
                    task_id=str(payload.get("task_id") or "stub-task"),
                    status=TaskStatus.PLANNING,
                    plan_versions=[[s.model_copy(deep=True) for s in specs]],
                )
            if phase == "execute_subtask":
                # 执行阶段的子 Agent 调用：记录 dispatch 输入并挂起
                dispatch_inputs.append(raw_input)
                holder["stopping_at_dispatch"] = holder["engine"]._gate.stopping
                await asyncio.sleep(30)
            await asyncio.sleep(30)

        settings = Settings(_env_file=None, max_pipeline_timeout=0.3)
        engine = build_engine(tmp_path, monkeypatch, settings=settings, handler=handler)
        holder["engine"] = engine

        output = await engine.run("deadline 后零新增 dispatch")

        # 优雅降级：人工兜底终态
        assert output.status is TaskStatus.HUMAN_REQUIRED
        # 停止门已原子进入 stopping（pipeline_timeout 首因保留）
        assert engine._gate.stopping is True
        assert engine._gate.stop_reason == "pipeline_timeout"
        # 剩余预算永不为负
        assert engine._gate.remaining() == 0.0
        # 执行链零新增 dispatch：只发生了 s1 的执行调用，s2 从未发起
        assert dispatch_inputs, "s1 应已进入执行链"
        assert not any('"s2"' in item for item in dispatch_inputs)
        # 超时后调度器无残留运行任务、停止后无新落账
        assert engine._scheduler.results.get("s2") is None
        count_after = len(dispatch_inputs)
        await asyncio.sleep(0.1)
        assert len(dispatch_inputs) == count_after  # 停止后零新增

    async def test_timeout_count_uses_authoritative_results(self, tmp_path, monkeypatch):
        """超时完成计数来自权威合并视图（"1/2" 场景回归）。"""
        settings = Settings(_env_file=None, max_pipeline_timeout=0.1)
        engine = build_engine(
            tmp_path, monkeypatch, settings=settings, handler=_hang_forever
        )
        engine._subtasks = [make_spec("s_done"), make_spec("s_hang")]
        engine._results["s_done"] = ExecutionResult(
            subtask_id="s_done", success=True, output="已完成的部分产出"
        )

        output = await engine.run("权威计数")

        assert output.status is TaskStatus.HUMAN_REQUIRED
        assert "1/2" in output.final_result


async def _hang_forever(is_review: bool, raw_input: str):
    """所有阶段长时间挂起，等待外层 wait_for 超时取消。"""
    await asyncio.sleep(30)


class TestAuthoritativeResults:
    """权威成功结果合并视图（``_authoritative_success_results``）。"""

    async def test_merge_archive_current_and_scheduler_results(self, tmp_path, monkeypatch):
        settings = Settings(_env_file=None)
        engine = build_engine(
            tmp_path, monkeypatch, settings=settings, handler=_hang_forever
        )
        # 跨版本归档：v2 覆盖 v1（同 subtask_id 重规划后的新结果）
        engine._archive_results[("v1", "s1")] = ExecutionResult(
            subtask_id="s1", success=True, output="old-v1"
        )
        engine._archive_results[("v2", "s1")] = ExecutionResult(
            subtask_id="s1", success=True, output="new-v2"
        )
        # 当前计划版本：s1 已失败（不得以失败状态计入）、s3 成功
        engine._results["s1"] = ExecutionResult(
            subtask_id="s1", success=False, error="当前版本失败"
        )
        engine._results["s3"] = ExecutionResult(
            subtask_id="s3", success=True, output="current"
        )
        # 调度器清理期落账：s2 部分结果
        engine._scheduler.results["s2"] = ExecutionResult(
            subtask_id="s2", success=True, output="cleanup-partial"
        )

        completed = engine._authoritative_success_results()
        by_id = {item.subtask_id: item for item in completed}

        assert set(by_id) == {"s1", "s2", "s3"}
        assert by_id["s1"].output == "new-v2"  # 新版本覆盖旧版本
        assert by_id["s2"].output == "cleanup-partial"  # 清理期落账计入
        assert by_id["s3"].output == "current"
        # 深拷贝：外部修改不影响引擎内部状态
        by_id["s3"].output = "mutated"
        assert engine._results["s3"].output == "current"


class TestLateResultIsolation:
    """迟到结果隔离：等待期间取消 → 结果不得写状态/触发后续动作。"""

    async def test_cancel_during_supervisor_call_maps_to_cancelled(
        self, tmp_path, monkeypatch
    ):
        """主管返回前外部取消：返回结果按迟到隔离，任务为 cancelled 终态。"""
        holder: dict = {}

        async def handler(is_review: bool, raw_input: str):
            if is_review:
                return ReviewDecision(
                    stage="plan_review", passed=True, summary="计划评审（模拟）"
                )
            payload = json.loads(raw_input)
            if payload.get("phase") in ("planning", "replanning"):
                # 模拟等待期间外部取消（主管返回前原子停止）
                holder["engine"].cancel()
                # 用本任务真实 task_id 预置一条活跃授权：
                # 取消出口的 finally 按任务维度撤销，必须将其清零
                holder["engine"]._gateway.issue_grant(
                    task_id=str(payload.get("task_id")),
                    agent_id="a-pre",
                    allowed_tools=["read_file"],
                    issued_by=holder["engine"]._gateway.supervisor_id,
                )
                holder["grant_issued"] = True
                specs = [make_spec("s1")]
                return OrchestratorOutput(
                    task_id=str(payload.get("task_id") or "stub-task"),
                    status=TaskStatus.PLANNING,
                    plan_versions=[[s.model_copy(deep=True) for s in specs]],
                )
            await asyncio.sleep(30)

        settings = Settings(_env_file=None)
        engine = build_engine(tmp_path, monkeypatch, settings=settings, handler=handler)
        holder["engine"] = engine

        output = await engine.run("迟到结果隔离")

        # 迟到结果被隔离：任务映射为 cancelled（而非继续执行计划）
        assert output.status is TaskStatus.CANCELLED
        assert output.final_result == "任务已取消"
        # 前置确证：主管调用期间确有活跃授权被签发（防空断言）
        assert holder.get("grant_issued") is True
        assert any(
            not grant.revoked for grant in engine._gateway.grants.values()
        ) is False
        # 无子任务被创建/执行（计划未落地）
        assert output.agent_history == [] or all(
            getattr(record, "status", "") in {"cancelled"}
            for record in output.agent_history
        )
        # 授权最终清零：所有出口统一幂等撤销（任务 45 不变量）
        assert all(grant.revoked for grant in engine._gateway.grants.values())

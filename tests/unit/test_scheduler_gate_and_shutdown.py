"""单元测试：调度器停止门集成与 asyncio task 所有权收口（任务 45）。

覆盖任务 45 验收点（调度器层面）：

- 父协程取消：``schedule`` 的 ``finally`` 取消并 await 全部运行中任务，
  孤儿任务清零（``_shutdown_running``）；
- 停止门 stopping：循环头拒绝新 dispatch（``PipelineStoppedError``
  穿透宽捕获向上传播），运行中任务抛出的停止信号不落业务结果；
- deadline 到期 / 剩余预算耗尽：不调用执行器、不发 dispatch 事件，
  返回可审计的预算失败结果（剩余时间永不为负）；
- 清理期真实异常落账为可审计失败（不吞业务异常）；清理期被取消的
  任务不落业务结果；清理期恰好正常完成的任务结果保留。

另含：``Task.cancelling`` 跨版本兼容（任务 59，项目声明
Python >=3.10）——取消防御不得直接假设 ``asyncio.Task.cancelling``
存在：3.11+ 保持严格取消恢复；缺少该 API（3.10）时不抛
``AttributeError``，业务异常按 3.11 同类路径（cancelling()==0）
降级为失败结果。
"""

from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace

import pytest

from so_agent.models import AgentRecord, ExecutionResult, SubtaskSpec
from so_agent.runtime.scheduler import (
    SubtaskDAGScheduler,
    _task_holds_pending_cancel,
)
from so_agent.state_machine import PipelineStoppedError
from so_agent.workflow_support import PipelineGate


def spec(
    subtask_id: str,
    *,
    deps: list[str] | None = None,
    max_attempts: int = 3,
) -> SubtaskSpec:
    return SubtaskSpec(
        subtask_id=subtask_id,
        title=subtask_id,
        instructions=f"do {subtask_id}",
        role="code",
        dependencies=deps or [],
        allowed_tools=[],
        max_attempts=max_attempts,
    )


def _ok(spec_: SubtaskSpec) -> ExecutionResult:
    return ExecutionResult(subtask_id=spec_.subtask_id, success=True, output="ok")


async def success_executor_noop(
    task_spec: SubtaskSpec, agent: AgentRecord
) -> ExecutionResult:
    """占位执行器（直测 _shutdown_running 时不参与）。"""
    return _ok(task_spec)


class TestParentCancellation:
    """父协程取消：调度器拥有全部 asyncio tasks 并负责清理。"""

    async def test_cancel_awaits_all_running_tasks(self):
        """取消 schedule 任务后：全部运行中子任务被取消并等待收尾，无孤儿。"""
        entered = asyncio.Event()

        async def executor(task_spec: SubtaskSpec, agent: AgentRecord) -> ExecutionResult:
            entered.set()
            await asyncio.sleep(30)  # 挂起，等待被调度器清理取消
            return _ok(task_spec)

        scheduler = SubtaskDAGScheduler(executor)
        schedule_task = asyncio.create_task(
            scheduler.schedule("t1", [spec("a"), spec("b")])
        )
        await entered.wait()
        baseline = {
            task
            for task in asyncio.all_tasks()
            if task is not asyncio.current_task()
        } - {schedule_task}

        schedule_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await schedule_task

        await asyncio.sleep(0.05)  # 让清理期任务真正收尾
        leftover = {
            task
            for task in asyncio.all_tasks()
            if task is not asyncio.current_task()
        } - baseline
        assert leftover == set()  # 孤儿任务清零

    async def test_cancelled_subtasks_not_recorded_as_results(self):
        """清理期被取消的子任务：不落业务结果。"""

        async def executor(task_spec: SubtaskSpec, agent: AgentRecord) -> ExecutionResult:
            await asyncio.sleep(30)
            return _ok(task_spec)

        scheduler = SubtaskDAGScheduler(executor)
        schedule_task = asyncio.create_task(scheduler.schedule("t1", [spec("a")]))
        await asyncio.sleep(0.05)
        schedule_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await schedule_task
        assert "a" not in scheduler.results

    async def test_cleanup_period_real_exception_recorded(self):
        """清理期结算：任务以真实异常（非取消）收尾时落账为可审计失败，
        绝不吞掉（直测 ``_shutdown_running`` 的结算语义）。"""

        async def failing() -> ExecutionResult:
            await asyncio.sleep(0)  # 让任务启动后拋异常
            raise ValueError("底层资源损坏")

        scheduler = SubtaskDAGScheduler(success_executor_noop)
        task: asyncio.Task[ExecutionResult] = asyncio.create_task(failing())
        await asyncio.sleep(0.05)  # 任务已以 ValueError 完成
        results: dict[str, ExecutionResult] = {}
        running: dict[asyncio.Task[ExecutionResult], str] = {task: "a"}
        await scheduler._shutdown_running(running, results)
        assert running == set() or not running  # 所有权收口：集合清空
        result = results.get("a")
        assert result is not None
        assert result.success is False
        assert "清理期终止" in (result.error or "")
        assert "ValueError" in (result.error or "")

    async def test_swallowed_cancellation_restored_no_retry(self):
        """执行器吞掉取消并转换为普通异常（Python 3.12+ wait_for 直接展开
        coroutine 时真实可达）：调度器必须恢复取消语义，绝不降级为失败
        结果触发重试（否则停止/取消后仍会发起新尝试）。"""
        calls: list[str] = []

        async def executor(task_spec: SubtaskSpec, agent: AgentRecord) -> ExecutionResult:
            calls.append(task_spec.subtask_id)
            try:
                await asyncio.sleep(30)
            except asyncio.CancelledError:
                raise ValueError("清理期转为业务异常")  # 违反取消契约
            return _ok(task_spec)

        scheduler = SubtaskDAGScheduler(executor)
        schedule_task = asyncio.create_task(scheduler.schedule("t1", [spec("a")]))
        await asyncio.sleep(0.05)
        schedule_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await schedule_task
        assert calls == ["a"]  # 无重试：取消后零新增尝试
        assert "a" not in scheduler.results  # 取消不落业务结果

    async def test_subtask_completing_during_cleanup_kept(self):
        """清理期恰好正常完成（吞并取消后返回）的子任务：结果保留，不丢已完成工作。"""

        async def executor(task_spec: SubtaskSpec, agent: AgentRecord) -> ExecutionResult:
            try:
                await asyncio.sleep(30)
            except asyncio.CancelledError:
                await asyncio.sleep(0.01)  # 收尾工作（清理阶段完成）
                return ExecutionResult(
                    subtask_id=task_spec.subtask_id, success=True, output="saved"
                )
            return _ok(task_spec)

        scheduler = SubtaskDAGScheduler(executor)
        schedule_task = asyncio.create_task(scheduler.schedule("t1", [spec("a")]))
        await asyncio.sleep(0.05)
        schedule_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await schedule_task
        result = scheduler.results.get("a")
        assert result is not None
        assert result.success is True
        assert result.output == "saved"


class TestStopGateDispatch:
    """停止门 stopping：拒绝一切新 dispatch，信号穿透宽捕获。"""

    async def test_stopping_blocks_new_dispatch(self):
        """首个子任务执行期间外部停止：依赖链后续子任务不再 dispatch。"""
        gate = PipelineGate()
        calls: list[str] = []

        async def executor(task_spec: SubtaskSpec, agent: AgentRecord) -> ExecutionResult:
            calls.append(task_spec.subtask_id)
            if task_spec.subtask_id == "a":
                gate.request_stop("external_stop")
            return ExecutionResult(
                subtask_id=task_spec.subtask_id, success=False, error="boom"
            )

        scheduler = SubtaskDAGScheduler(executor, gate=gate)
        with pytest.raises(PipelineStoppedError):
            await scheduler.schedule("t1", [spec("a"), spec("b", deps=["a"])])
        assert calls == ["a"]  # b 未被 dispatch（零新增）

    async def test_pipeline_stopped_from_running_task_not_recorded(self):
        """运行中任务抛 PipelineStoppedError：不落业务结果，直接传播。"""
        gate = PipelineGate()

        async def executor(task_spec: SubtaskSpec, agent: AgentRecord) -> ExecutionResult:
            gate.request_stop("late_result")
            raise PipelineStoppedError("结果按迟到隔离")

        scheduler = SubtaskDAGScheduler(executor, gate=gate)
        with pytest.raises(PipelineStoppedError):
            await scheduler.schedule("t1", [spec("a")])
        assert scheduler.results == {}  # 不落业务结果、不转重试

    async def test_no_gate_schedule_unaffected(self):
        """未注入停止门：调度行为与旧版完全一致（向后兼容）。"""
        calls: list[str] = []

        async def executor(task_spec: SubtaskSpec, agent: AgentRecord) -> ExecutionResult:
            calls.append(task_spec.subtask_id)
            return _ok(task_spec)

        scheduler = SubtaskDAGScheduler(executor)
        results = await scheduler.schedule("t1", [spec("a"), spec("b")])
        assert set(results) == {"a", "b"}
        assert all(result.success for result in results.values())
        assert set(calls) == {"a", "b"}


class TestBudgetExhaustion:
    """剩余预算耗尽（remaining <= 0，永不为负）：不调用执行器，可审计失败。"""

    async def test_expired_deadline_auditable_failure_zero_dispatch(self):
        """deadline 已过：循环内尝试被拒，返回可审计失败，零 dispatch。"""
        gate = PipelineGate()
        gate.set_deadline(time.monotonic() - 1.0)  # 剩余 0（max(0, ...) 截断）
        assert gate.remaining() == 0.0
        calls: list[str] = []

        async def executor(task_spec: SubtaskSpec, agent: AgentRecord) -> ExecutionResult:
            calls.append(task_spec.subtask_id)
            return _ok(task_spec)

        scheduler = SubtaskDAGScheduler(executor, gate=gate)
        results = await scheduler.schedule("t1", [spec("a", max_attempts=2)])
        assert calls == []  # 零 dispatch
        assert results["a"].success is False
        assert "剩余预算不足" in (results["a"].error or "")

    async def test_midway_expiry_blocks_retry_and_replacement(self):
        """首次尝试期间 deadline 到期：重试与替换均不再发起，可审计失败。"""
        gate = PipelineGate()
        calls: list[str] = []

        async def executor(task_spec: SubtaskSpec, agent: AgentRecord) -> ExecutionResult:
            calls.append(task_spec.subtask_id)
            gate.set_deadline(time.monotonic() - 0.001)  # 执行期间 deadline 到期
            return ExecutionResult(
                subtask_id=task_spec.subtask_id, success=False, error="boom"
            )

        scheduler = SubtaskDAGScheduler(executor, gate=gate)
        results = await scheduler.schedule("t1", [spec("a", max_attempts=3)])
        assert calls == ["a"]  # 只有第一次真实执行
        assert results["a"].success is False
        assert "剩余预算不足" in (results["a"].error or "")

    async def test_stopping_deadline_blocks_dependent_dispatch(self):
        """stopping + deadline 均拒绝：后续依赖子任务零 dispatch。"""
        gate = PipelineGate()
        gate.set_deadline(time.monotonic() + 60.0)
        calls: list[str] = []

        async def executor(task_spec: SubtaskSpec, agent: AgentRecord) -> ExecutionResult:
            calls.append(task_spec.subtask_id)
            return _ok(task_spec)

        scheduler = SubtaskDAGScheduler(executor, gate=gate)
        gate.request_stop("human_terminate")
        with pytest.raises(PipelineStoppedError):
            await scheduler.schedule("t1", [spec("a"), spec("b", deps=["a"])])
        assert calls == []  # 人工终止后零新增 dispatch


async def _run_failing_attempt(
    monkeypatch: pytest.MonkeyPatch, current_task: object
) -> tuple[ExecutionResult, list[str], object]:
    """在指定伪 current_task 下直跑 _run_attempts，执行器抛业务异常。

    monkeypatch ``asyncio.current_task``：_run_attempts 的取消防御只在
    except Exception 分支读取 current_task 的 cancelling 状态。
    返回 ``(结果, 失败摘要, 恢复判定)`` 三元组（任务 46 起为 3 元组）。
    """

    async def executor(task_spec: SubtaskSpec, agent: AgentRecord) -> ExecutionResult:
        raise ValueError("清理期业务异常")

    scheduler = SubtaskDAGScheduler(executor)
    monkeypatch.setattr(asyncio, "current_task", lambda: current_task)
    subtask = spec("a", max_attempts=1)
    record = scheduler._new_agent_record(
        "t59",
        subtask,
        can_request_replacement=False,
        replacement_of=None,
    )
    return await scheduler._run_attempts("t59", subtask, record, max_attempts=1)


class TestTaskHoldsPendingCancelCompat:
    """``Task.cancelling`` 跨版本兼容 helper（任务 59，Python >=3.10）。

    ``asyncio.Task.cancelling`` 自 Python 3.11 才存在；取消防御不得
    直接假设该 API 存在，否则 3.10 运行时会抛 ``AttributeError``
    破坏异常降级路径。
    """

    def test_none_task_returns_false(self):
        assert _task_holds_pending_cancel(None) is False

    def test_task_without_cancelling_attribute_returns_false(self):
        """模拟 Python 3.10 的 task（无 cancelling 属性）：getattr 回退
        路径不抛 AttributeError，返回 False。"""
        legacy_task = SimpleNamespace()  # 无 cancelling 属性
        assert _task_holds_pending_cancel(legacy_task) is False

    def test_task_with_cancelling_zero_returns_false(self):
        """3.11+ 无未消费取消请求：返回 False（业务异常正常降级）。"""
        modern_task = SimpleNamespace(cancelling=lambda: 0)
        assert _task_holds_pending_cancel(modern_task) is False

    def test_task_with_cancelling_positive_returns_true(self):
        """3.11+ 持有未消费取消请求：返回 True（严格取消恢复，任务 45
        语义不变）。"""
        modern_task = SimpleNamespace(cancelling=lambda: 2)
        assert _task_holds_pending_cancel(modern_task) is True

    async def test_real_task_without_pending_cancel_returns_false(self):
        """真实 asyncio task（3.11+ 运行时原生 cancelling）：正常运行中
        无未消费取消请求 → False。"""
        assert _task_holds_pending_cancel(asyncio.current_task()) is False

    async def test_real_task_pending_cancel_returns_true(self):
        """真实取消请求未消费时（except CancelledError 块内）→ True。

        这是任务 45 取消防御依赖的原生语义回归。"""
        probe: dict[str, bool] = {}

        async def child() -> None:
            try:
                await asyncio.sleep(30)
            except asyncio.CancelledError:
                probe["pending"] = _task_holds_pending_cancel(
                    asyncio.current_task()
                )
                raise

        task = asyncio.create_task(child())
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert probe["pending"] is True


class TestRunAttemptsCancellingCompat:
    """_run_attempts 异常分支的 cancelling 兼容集成路径（任务 59）。"""

    async def test_legacy_task_without_cancelling_no_attribute_error(
        self, monkeypatch
    ):
        """current_task 无 cancelling 属性（模拟 Python 3.10）：不抛
        AttributeError，业务异常正常降级为失败结果（可审计、可重试）。"""
        legacy_task = SimpleNamespace()  # 无 cancelling 属性
        result, failures, _ = await _run_failing_attempt(monkeypatch, legacy_task)
        assert result.success is False
        assert "ValueError" in (result.error or "")
        assert failures and "ValueError" in failures[0]

    async def test_modern_task_pending_cancel_restores_cancellation(
        self, monkeypatch
    ):
        """current_task.cancelling() > 0（3.11+ 严格路径）：执行器抛
        业务异常时恢复 CancelledError 向上传播，绝不降级为失败结果。"""
        modern_task = SimpleNamespace(cancelling=lambda: 1)
        with pytest.raises(asyncio.CancelledError):
            await _run_failing_attempt(monkeypatch, modern_task)

    async def test_modern_task_zero_cancel_degrades_to_failure(
        self, monkeypatch
    ):
        """current_task.cancelling() == 0：与 3.10 兼容路径行为一致，
        业务异常降级为失败结果（跨版本语义对齐）。"""
        modern_task = SimpleNamespace(cancelling=lambda: 0)
        result, failures, _ = await _run_failing_attempt(monkeypatch, modern_task)
        assert result.success is False
        assert "ValueError" in (result.error or "")
        assert failures and "ValueError" in failures[0]

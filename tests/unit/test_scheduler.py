"""单元测试：子任务 DAG 调度器 TaskScheduler。"""

from __future__ import annotations

import asyncio

import pytest

from so_agent.config import Settings
from so_agent.context import ProjectContext
from so_agent.models import AgentRecord, ExecutionResult, SubtaskSpec
from so_agent.runtime.events import (
    EVENT_AGENT_CREATED,
    EVENT_AGENT_RELEASED,
    EVENT_DECOMPOSITION_ISSUE,
    EVENT_REPLACEMENT_REQUESTED,
    EVENT_TASK_DISPATCHED,
    EventLogger,
)
from so_agent.runtime.registry import AgentRegistry
from so_agent.runtime.scheduler import SchedulerError, TaskScheduler


def spec(
    subtask_id: str,
    *,
    deps: list[str] | None = None,
    role: str = "code",
    max_attempts: int = 3,
    allowed_tools: list[str] | None = None,
) -> SubtaskSpec:
    return SubtaskSpec(
        subtask_id=subtask_id,
        title=subtask_id,
        instructions=f"do {subtask_id}",
        role=role,
        dependencies=deps or [],
        allowed_tools=allowed_tools or [],
        max_attempts=max_attempts,
    )


def failing_executor(calls: list):
    async def executor(task_spec: SubtaskSpec, agent: AgentRecord) -> ExecutionResult:
        calls.append((task_spec.subtask_id, agent.agent_id, agent.can_request_replacement))
        return ExecutionResult(
            subtask_id=task_spec.subtask_id, success=False, error="boom"
        )

    return executor


def success_executor(calls: list):
    async def executor(task_spec: SubtaskSpec, agent: AgentRecord) -> ExecutionResult:
        calls.append((task_spec.subtask_id, agent.agent_id, agent.can_request_replacement))
        return ExecutionResult(
            subtask_id=task_spec.subtask_id, success=True, output=f"ok-{task_spec.subtask_id}"
        )

    return executor


class TestDagValidation:
    async def test_empty_subtask_list_rejected(self):
        scheduler = TaskScheduler(failing_executor([]))
        with pytest.raises(SchedulerError):
            await scheduler.schedule("t1", [])

    async def test_blank_task_id_rejected(self):
        scheduler = TaskScheduler(failing_executor([]))
        with pytest.raises(SchedulerError):
            await scheduler.schedule("", [spec("s1")])

    async def test_duplicate_ids_rejected(self):
        scheduler = TaskScheduler(failing_executor([]))
        with pytest.raises(SchedulerError):
            await scheduler.schedule("t1", [spec("s1"), spec("s1")])

    async def test_missing_dependency_rejected(self):
        scheduler = TaskScheduler(failing_executor([]))
        with pytest.raises(SchedulerError):
            await scheduler.schedule("t1", [spec("s1", deps=["ghost"])])

    async def test_self_dependency_rejected(self):
        scheduler = TaskScheduler(failing_executor([]))
        with pytest.raises(SchedulerError):
            await scheduler.schedule("t1", [spec("s1", deps=["s1"])])

    async def test_cycle_rejected(self):
        scheduler = TaskScheduler(failing_executor([]))
        with pytest.raises(SchedulerError):
            await scheduler.schedule(
                "t1", [spec("s1", deps=["s2"]), spec("s2", deps=["s1"])]
            )

    def test_topological_order_chain(self):
        scheduler = TaskScheduler(failing_executor([]))
        order = scheduler.topological_order(
            [spec("s3", deps=["s2"]), spec("s2", deps=["s1"]), spec("s1")]
        )
        assert order.index("s1") < order.index("s2") < order.index("s3")

    def test_topological_order_preserves_relative_order(self):
        scheduler = TaskScheduler(failing_executor([]))
        order = scheduler.topological_order([spec("a"), spec("b"), spec("c")])
        assert order == ["a", "b", "c"]


class TestDependencyScheduling:
    async def test_chain_executes_in_dependency_order(self):
        order: list[str] = []

        async def executor(task_spec: SubtaskSpec, agent: AgentRecord) -> ExecutionResult:
            order.append(task_spec.subtask_id)
            return ExecutionResult(subtask_id=task_spec.subtask_id, success=True)

        scheduler = TaskScheduler(executor)
        results = await scheduler.schedule(
            "t1", [spec("s3", deps=["s2"]), spec("s2", deps=["s1"]), spec("s1")]
        )
        assert order == ["s1", "s2", "s3"]
        assert all(result.success for result in results.values())

    async def test_failed_dependency_skips_dependent(self):
        scheduler = TaskScheduler(failing_executor([]), settings=Settings(max_replacement_requests=0))
        results = await scheduler.schedule(
            "t1", [spec("s1"), spec("s2", deps=["s1"])]
        )
        assert results["s1"].success is False
        assert results["s2"].success is False
        assert "跳过" in results["s2"].error
        assert "s1" in results["s2"].error

    async def test_independent_subtasks_all_executed(self):
        calls: list = []
        scheduler = TaskScheduler(success_executor(calls))
        results = await scheduler.schedule("t1", [spec("a"), spec("b"), spec("c")])
        assert set(results) == {"a", "b", "c"}
        assert all(result.success for result in results.values())
        assert {call[0] for call in calls} == {"a", "b", "c"}

    async def test_results_mirror_scheduler_state(self):
        scheduler = TaskScheduler(success_executor([]))
        results = await scheduler.schedule("t1", [spec("s1")])
        assert scheduler.results is results


class TestConcurrencyLimit:
    async def test_peak_concurrency_respects_limit(self):
        state = {"active": 0, "peak": 0}

        async def executor(task_spec: SubtaskSpec, agent: AgentRecord) -> ExecutionResult:
            state["active"] += 1
            state["peak"] = max(state["peak"], state["active"])
            await asyncio.sleep(0.05)
            state["active"] -= 1
            return ExecutionResult(subtask_id=task_spec.subtask_id, success=True)

        scheduler = TaskScheduler(executor, settings=Settings(max_concurrency=2))
        await scheduler.schedule("t1", [spec(f"s{i}") for i in range(4)])
        assert state["peak"] <= 2
        assert state["peak"] >= 2  # 并发确实生效

    async def test_single_concurrency_serializes(self):
        state = {"active": 0, "peak": 0}

        async def executor(task_spec: SubtaskSpec, agent: AgentRecord) -> ExecutionResult:
            state["active"] += 1
            state["peak"] = max(state["peak"], state["active"])
            await asyncio.sleep(0.02)
            state["active"] -= 1
            return ExecutionResult(subtask_id=task_spec.subtask_id, success=True)

        scheduler = TaskScheduler(executor, settings=Settings(max_concurrency=1))
        await scheduler.schedule("t1", [spec(f"s{i}") for i in range(3)])
        assert state["peak"] == 1


class TestAttemptsAndReplacement:
    async def test_attempt_count_capped_at_three(self):
        calls: list = []
        scheduler = TaskScheduler(
            failing_executor(calls),
            settings=Settings(max_agent_attempts=3, max_replacement_requests=0),
        )
        await scheduler.schedule("t1", [spec("s1")])
        assert len(calls) == 3  # 原始 Agent 最多尝试 3 次

    async def test_spec_max_attempts_smaller_wins(self):
        calls: list = []
        scheduler = TaskScheduler(
            failing_executor(calls),
            settings=Settings(max_agent_attempts=3, max_replacement_requests=0),
        )
        await scheduler.schedule("t1", [spec("s1", max_attempts=1)])
        assert len(calls) == 1

    async def test_replacement_flow_after_primary_exhaustion(self):
        primary_calls: list = []
        replacement_calls: list = []
        scheduler = TaskScheduler(
            failing_executor(primary_calls),
            replacement_executor=success_executor(replacement_calls),
            settings=Settings(max_agent_attempts=2, max_replacement_requests=1),
        )
        results = await scheduler.schedule("t1", [spec("s1")])
        assert len(primary_calls) == 2  # 原始 Agent 2 次失败
        assert len(replacement_calls) == 1  # 替代 Agent 1 次成功
        assert results["s1"].success is True
        assert len(scheduler.replacement_requests) == 1

    async def test_replacement_agent_has_no_replacement_right(self):
        primary_calls: list = []
        replacement_calls: list = []
        scheduler = TaskScheduler(
            failing_executor(primary_calls),
            replacement_executor=failing_executor(replacement_calls),
            settings=Settings(max_agent_attempts=1, max_replacement_requests=1),
        )
        await scheduler.schedule("t1", [spec("s1")])
        # 主 Agent 可申请替换；替代 Agent 固定无替换权
        assert primary_calls[0][2] is True
        assert replacement_calls[0][2] is False
        assert scheduler.decomposition_issues[0].failed_agent_ids[0] == primary_calls[0][1]
        assert replacement_calls[0][1] in scheduler.decomposition_issues[0].failed_agent_ids

    async def test_decomposition_issue_after_replacement_failure(self):
        scheduler = TaskScheduler(
            failing_executor([]),
            settings=Settings(max_agent_attempts=1, max_replacement_requests=1),
        )
        results = await scheduler.schedule("t1", [spec("s1")])
        assert results["s1"].success is False
        assert len(scheduler.decomposition_issues) == 1
        issue = scheduler.decomposition_issues[0]
        assert issue.task_id == "t1"
        assert issue.subtask_id == "s1"
        assert len(issue.failed_agent_ids) == 2  # 原始 + 替代
        assert issue.failure_reasons
        assert issue.attempted_approaches
        assert issue.recommendation

    async def test_no_replacement_when_quota_zero(self):
        scheduler = TaskScheduler(
            failing_executor([]),
            settings=Settings(max_agent_attempts=1, max_replacement_requests=0),
        )
        await scheduler.schedule("t1", [spec("s1")])
        assert scheduler.replacement_requests == []
        assert len(scheduler.decomposition_issues) == 1
        assert len(scheduler.decomposition_issues[0].failed_agent_ids) == 1

    async def test_replacement_request_payload(self):
        scheduler = TaskScheduler(
            failing_executor([]),
            settings=Settings(max_agent_attempts=1, max_replacement_requests=1),
        )
        await scheduler.schedule("t1", [spec("s1", allowed_tools=["write_file"])])
        request = scheduler.replacement_requests[0]
        assert request.task_id == "t1"
        assert request.subtask_id == "s1"
        assert request.request_count == 1
        assert request.requested_tools == ["write_file"]
        assert request.failure_summary
        assert request.attempt_evidence

    async def test_replacement_quota_is_per_task_cumulative(self):
        """两次调度共享替换额度：第二次无法再发起替换。"""
        scheduler = TaskScheduler(
            failing_executor([]),
            settings=Settings(max_agent_attempts=1, max_replacement_requests=1),
        )
        await scheduler.schedule("t1", [spec("s1")])
        assert len(scheduler.replacement_requests) == 1
        await scheduler.schedule("t1", [spec("s2")])
        assert len(scheduler.replacement_requests) == 1  # 额度已用完
        assert len(scheduler.decomposition_issues) == 2


class TestTimeout:
    async def test_attempt_timeout_recorded_as_failure(self):
        async def executor(task_spec: SubtaskSpec, agent: AgentRecord) -> ExecutionResult:
            await asyncio.sleep(5)
            return ExecutionResult(subtask_id=task_spec.subtask_id, success=True)

        scheduler = TaskScheduler(
            executor,
            settings=Settings(max_agent_attempts=1, max_replacement_requests=0),
            attempt_timeout=0.2,
        )
        results = await scheduler.schedule("t1", [spec("s1", max_attempts=1)])
        assert results["s1"].success is False
        assert "单次执行超时" in results["s1"].error


class TestExecutorRobustness:
    async def test_executor_exception_recorded(self):
        async def executor(task_spec: SubtaskSpec, agent: AgentRecord) -> ExecutionResult:
            raise RuntimeError("executor exploded")

        scheduler = TaskScheduler(
            executor,
            settings=Settings(max_agent_attempts=1, max_replacement_requests=0),
        )
        results = await scheduler.schedule("t1", [spec("s1")])
        assert results["s1"].success is False
        assert "执行异常" in results["s1"].error
        assert "RuntimeError" in results["s1"].error

    async def test_invalid_executor_return_recorded(self):
        async def executor(task_spec: SubtaskSpec, agent: AgentRecord):
            return "not-a-result"

        scheduler = TaskScheduler(
            executor,
            settings=Settings(max_agent_attempts=1, max_replacement_requests=0),
        )
        results = await scheduler.schedule("t1", [spec("s1")])
        assert results["s1"].success is False
        assert "非法结果" in results["s1"].error


class TestEventsAndRegistry:
    async def test_lifecycle_events_logged(self, tmp_path):
        context = ProjectContext(sandbox_dir=tmp_path, project_name="scheduler-test")
        events = EventLogger(context=context)
        scheduler = TaskScheduler(
            success_executor([]),
            context=context,
            settings=Settings(),
            events=events,
        )
        await scheduler.schedule("t1", [spec("s1")])
        types = [record["event_type"] for record in context.event_log]
        assert EVENT_AGENT_CREATED in types
        assert EVENT_TASK_DISPATCHED in types
        assert EVENT_AGENT_RELEASED in types

    async def test_failure_chain_events_logged(self, tmp_path):
        context = ProjectContext(sandbox_dir=tmp_path, project_name="scheduler-test")
        events = EventLogger(context=context)
        scheduler = TaskScheduler(
            failing_executor([]),
            context=context,
            settings=Settings(max_agent_attempts=1, max_replacement_requests=1),
            events=events,
        )
        await scheduler.schedule("t1", [spec("s1")])
        types = [record["event_type"] for record in context.event_log]
        assert EVENT_REPLACEMENT_REQUESTED in types
        assert EVENT_DECOMPOSITION_ISSUE in types
        dispatched = events.get_events(event_type=EVENT_TASK_DISPATCHED)
        assert [record["payload"]["attempt"] for record in dispatched] == [1, 1]

    async def test_registry_registration_and_release(self, tmp_path):
        context = ProjectContext(sandbox_dir=tmp_path, project_name="scheduler-test")
        registry = AgentRegistry(context=context, settings=Settings(max_dynamic_agents=4))
        scheduler = TaskScheduler(
            success_executor([]), context=context, registry=registry
        )
        await scheduler.schedule("t1", [spec("s1")])
        records = registry.list_by_task("t1")
        assert len(records) == 1
        assert records[0].status == "completed"
        assert registry.active_count() == 0  # 执行完成后释放配额

    async def test_dynamic_agent_limit_blocks_registration(self, tmp_path):
        context = ProjectContext(sandbox_dir=tmp_path, project_name="scheduler-test")
        registry = AgentRegistry(context=context, settings=Settings(max_dynamic_agents=1))

        async def executor(task_spec: SubtaskSpec, agent: AgentRecord) -> ExecutionResult:
            await asyncio.sleep(0.05)  # 让 s1 注册后保持活跃，s2 注册必被拒
            return ExecutionResult(subtask_id=task_spec.subtask_id, success=True)

        scheduler = TaskScheduler(
            executor,
            context=context,
            settings=Settings(max_dynamic_agents=1, max_concurrency=2),
            registry=registry,
        )
        results = await scheduler.schedule("t1", [spec("s1"), spec("s2")])
        assert results["s1"].success is True
        assert results["s2"].success is False
        assert "已达上限" in results["s2"].error

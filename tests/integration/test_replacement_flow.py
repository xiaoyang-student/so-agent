"""集成测试：子 Agent 失败 → 替换流程（模拟执行器，不消耗真实 API）。

模拟场景（无真实模型调用）：
1. 原始子 Agent 三次尝试全部失败 → 生成 ReplacementRequest；
2. 替代 Agent 固定 ``can_request_replacement=False``，仍失败 → 生成
   TaskDecompositionIssue（驱动重规划）；
3. 每 Agent 至多发起一次替换申请，额度耗尽后重复申请被拒绝。
"""

from __future__ import annotations

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
from so_agent.runtime.scheduler import TaskScheduler


def make_spec(subtask_id: str = "s1", **kwargs) -> SubtaskSpec:
    defaults = dict(
        subtask_id=subtask_id,
        title="构建报表",
        instructions="生成周报表",
        role="code",
        allowed_tools=["write_file"],
        max_attempts=3,
    )
    defaults.update(kwargs)
    return SubtaskSpec(**defaults)


class RecordingExecutor:
    """可编程模拟执行器：按调用次数/轮次返回结果，并记录收到的 AgentRecord。"""

    def __init__(self, outcomes: list[bool] | bool = False):
        self._outcomes = outcomes if isinstance(outcomes, list) else None
        self._fixed = None if self._outcomes is not None else outcomes
        self.calls: list[AgentRecord] = []

    async def __call__(self, spec: SubtaskSpec, agent: AgentRecord) -> ExecutionResult:
        index = len(self.calls)
        self.calls.append(agent)
        if self._outcomes is not None:
            success = self._outcomes[index] if index < len(self._outcomes) else self._outcomes[-1]
        else:
            success = bool(self._fixed)
        if success:
            return ExecutionResult(
                subtask_id=spec.subtask_id, success=True, output="ok"
            )
        return ExecutionResult(
            subtask_id=spec.subtask_id,
            success=False,
            error=f"第 {agent.attempt_count} 次尝试失败：模拟错误",
            evidence=[f"evidence-{index}"],
        )


def build_env(tmp_path, **settings_kwargs):
    context = ProjectContext(sandbox_dir=tmp_path, project_name="replacement-flow")
    settings = Settings(
        max_agent_attempts=settings_kwargs.pop("max_agent_attempts", 3),
        max_replacement_requests=settings_kwargs.pop("max_replacement_requests", 1),
        **settings_kwargs,
    )
    registry = AgentRegistry(context=context, settings=settings)
    events = EventLogger(context=context)
    return context, settings, registry, events


class TestPrimaryExhaustionProducesReplacementRequest:
    async def test_three_failures_then_replacement_request(self, tmp_path):
        context, settings, registry, events = build_env(tmp_path)
        primary = RecordingExecutor(False)
        replacement = RecordingExecutor(True)
        scheduler = TaskScheduler(
            primary,
            replacement_executor=replacement,
            context=context,
            settings=settings,
            registry=registry,
            events=events,
        )

        results = await scheduler.schedule("t1", [make_spec()])

        # 原始 Agent 三次尝试全部失败后才申请替换
        assert len(primary.calls) == 3
        assert len(scheduler.replacement_requests) == 1
        request = scheduler.replacement_requests[0]
        assert request.request_count == 1
        assert request.requester_agent_id == primary.calls[0].agent_id
        assert request.requested_tools == ["write_file"]
        assert request.failure_summary
        # 替代 Agent 一次成功
        assert len(replacement.calls) == 1
        assert results["s1"].success is True
        assert scheduler.decomposition_issues == []

    async def test_request_evidence_accumulated(self, tmp_path):
        context, settings, registry, events = build_env(tmp_path)
        scheduler = TaskScheduler(
            RecordingExecutor(False),
            replacement_executor=RecordingExecutor(False),
            context=context,
            settings=settings,
            registry=registry,
            events=events,
        )
        await scheduler.schedule("t1", [make_spec()])
        request = scheduler.replacement_requests[0]
        assert any("第 1 次尝试失败" in item for item in request.attempt_evidence)
        assert any("第 3 次尝试失败" in item for item in request.attempt_evidence)


class TestReplacementAgentConstraints:
    async def test_replacement_agent_cannot_request_replacement(self, tmp_path):
        context, settings, registry, events = build_env(
            tmp_path, max_replacement_requests=2
        )
        primary = RecordingExecutor(False)
        replacement = RecordingExecutor(False)
        scheduler = TaskScheduler(
            primary,
            replacement_executor=replacement,
            context=context,
            settings=settings,
            registry=registry,
            events=events,
        )

        await scheduler.schedule("t1", [make_spec()])

        assert primary.calls[0].can_request_replacement is True
        assert all(
            agent.can_request_replacement is False for agent in replacement.calls
        )
        # 即使全局额度为 2，替代 Agent 也不具备替换权 → 仅 1 条替换申请
        assert len(scheduler.replacement_requests) == 1

    async def test_replacement_agent_failure_produces_decomposition_issue(self, tmp_path):
        context, settings, registry, events = build_env(tmp_path)
        primary = RecordingExecutor(False)
        replacement = RecordingExecutor(False)
        scheduler = TaskScheduler(
            primary,
            replacement_executor=replacement,
            context=context,
            settings=settings,
            registry=registry,
            events=events,
        )

        results = await scheduler.schedule("t1", [make_spec()])

        assert results["s1"].success is False
        assert len(scheduler.decomposition_issues) == 1
        issue = scheduler.decomposition_issues[0]
        assert issue.subtask_id == "s1"
        assert primary.calls[0].agent_id in issue.failed_agent_ids
        assert replacement.calls[0].agent_id in issue.failed_agent_ids
        assert any("can_request_replacement=False" in item for item in issue.attempted_approaches)

    async def test_registry_tracks_replacement_lineage(self, tmp_path):
        context, settings, registry, events = build_env(tmp_path)
        primary = RecordingExecutor(False)
        replacement = RecordingExecutor(False)
        scheduler = TaskScheduler(
            primary,
            replacement_executor=replacement,
            context=context,
            settings=settings,
            registry=registry,
            events=events,
        )

        await scheduler.schedule("t1", [make_spec()])

        records = registry.list_by_task("t1")
        assert len(records) == 2
        primary_record = registry.get(primary.calls[0].agent_id)
        replacement_record = registry.get(replacement.calls[0].agent_id)
        assert primary_record.replacement_requested is True
        assert primary_record.status == "failed"
        assert replacement_record.replacement_of == primary.calls[0].agent_id
        assert replacement_record.status == "failed"
        # 两个 Agent 均已释放，配额归还
        assert registry.active_count() == 0


class TestDuplicateRequestRejected:
    async def test_second_replacement_blocked_after_quota_used(self, tmp_path):
        context, settings, registry, events = build_env(tmp_path)
        scheduler = TaskScheduler(
            RecordingExecutor(False),
            replacement_executor=RecordingExecutor(False),
            context=context,
            settings=settings,
            registry=registry,
            events=events,
        )

        await scheduler.schedule("t1", [make_spec("s1")])
        assert len(scheduler.replacement_requests) == 1

        # 同一任务再次执行另一个子任务：额度已耗尽，重复申请被拒绝
        await scheduler.schedule("t1", [make_spec("s2")])
        assert len(scheduler.replacement_requests) == 1
        assert len(scheduler.decomposition_issues) == 2

        second_issue = scheduler.decomposition_issues[1]
        assert any("替换额度已用尽" in reason for reason in second_issue.failure_reasons)

    async def test_replacement_request_count_increments_only_when_allowed(self, tmp_path):
        context, settings, registry, events = build_env(tmp_path, max_replacement_requests=2)
        scheduler = TaskScheduler(
            RecordingExecutor(False),
            replacement_executor=RecordingExecutor(False),
            context=context,
            settings=settings,
            registry=registry,
            events=events,
        )

        await scheduler.schedule("t1", [make_spec("s1")])
        await scheduler.schedule("t1", [make_spec("s2")])
        assert [r.request_count for r in scheduler.replacement_requests] == [1, 2]


class TestFullEventChain:
    async def test_event_sequence_recorded(self, tmp_path):
        context, settings, registry, events = build_env(tmp_path)
        primary = RecordingExecutor(False)
        replacement = RecordingExecutor(False)
        scheduler = TaskScheduler(
            primary,
            replacement_executor=replacement,
            context=context,
            settings=settings,
            registry=registry,
            events=events,
        )

        await scheduler.schedule("t1", [make_spec()])

        types = [record["event_type"] for record in context.event_log]
        expected_subsequence = [
            EVENT_AGENT_CREATED,
            EVENT_TASK_DISPATCHED,
            EVENT_TASK_DISPATCHED,
            EVENT_TASK_DISPATCHED,
            EVENT_AGENT_RELEASED,
            EVENT_REPLACEMENT_REQUESTED,
            EVENT_AGENT_CREATED,
            EVENT_TASK_DISPATCHED,
            EVENT_TASK_DISPATCHED,
            EVENT_TASK_DISPATCHED,
            EVENT_AGENT_RELEASED,
            EVENT_DECOMPOSITION_ISSUE,
        ]
        # 断言事件类型按流程顺序出现（保留彼此相对顺序）
        cursor = 0
        for event_type in types:
            if cursor < len(expected_subsequence) and event_type == expected_subsequence[cursor]:
                cursor += 1
        assert cursor == len(expected_subsequence), f"事件序列不完整：{types}"

    async def test_dispatched_events_track_attempts(self, tmp_path):
        context, settings, registry, events = build_env(tmp_path)
        scheduler = TaskScheduler(
            RecordingExecutor(False),
            replacement_executor=RecordingExecutor(False),
            context=context,
            settings=settings,
            registry=registry,
            events=events,
        )
        await scheduler.schedule("t1", [make_spec()])

        dispatched = events.get_events(event_type=EVENT_TASK_DISPATCHED)
        attempts = [(r["payload"]["agent_id"], r["payload"]["attempt"]) for r in dispatched]
        assert len(attempts) == 6  # 原始 3 次 + 替代 3 次
        assert [attempt for _, attempt in attempts] == [1, 2, 3, 1, 2, 3]

    async def test_replacement_requested_event_payload(self, tmp_path):
        context, settings, registry, events = build_env(tmp_path)
        scheduler = TaskScheduler(
            RecordingExecutor(False),
            replacement_executor=RecordingExecutor(False),
            context=context,
            settings=settings,
            registry=registry,
            events=events,
        )
        await scheduler.schedule("t1", [make_spec()])

        record = events.get_events(event_type=EVENT_REPLACEMENT_REQUESTED)[0]
        assert record["payload"]["subtask_id"] == "s1"
        assert record["payload"]["request"]["request_count"] == 1

    async def test_decomposition_issue_event_payload(self, tmp_path):
        context, settings, registry, events = build_env(tmp_path)
        scheduler = TaskScheduler(
            RecordingExecutor(False),
            replacement_executor=RecordingExecutor(False),
            context=context,
            settings=settings,
            registry=registry,
            events=events,
        )
        await scheduler.schedule("t1", [make_spec()])

        record = events.get_events(event_type=EVENT_DECOMPOSITION_ISSUE)[0]
        issue = record["payload"]["issue"]
        assert issue["task_id"] == "t1"
        assert issue["subtask_id"] == "s1"
        assert len(issue["failed_agent_ids"]) == 2

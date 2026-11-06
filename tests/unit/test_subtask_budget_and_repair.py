"""单元测试：子任务硬预算、用量归因、直派执行与动作级修复。

覆盖三条控制面改造：

- token 归因：``usage_scope`` 把主管与嵌套 Agent 的用量精确记到
  当前子任务，``TokenBudgetTracker`` 跨尝试累计；
- 子任务硬预算：调度器每次尝试前结算 token 与真实耗时，超限即停止
  新尝试并以不可重试失败收敛（由上层重规划）；
- 直派与动作级修复：标准工具子任务直接调用 code_agent，跳过主管
  一层；自述成功被确定性验收翻转时在同一尝试内定向修复一次。
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

from so_agent.config import Settings
from so_agent.execution_coordinator import ExecutionCoordinator
from so_agent.mcp.protocol import MCPResponse, text_content
from so_agent.models import (
    AgentRecord,
    ExecutionResult,
    SubtaskSpec,
    TaskRequest,
)
from so_agent.runtime.budget import TokenBudgetTracker
from so_agent.runtime.events import (
    EVENT_ACTION_REPAIR,
    EVENT_TOOL_CALLED,
    EVENT_TOKEN_USAGE,
    EventLogger,
)
from so_agent.runtime.scheduler import SubtaskDAGScheduler
from so_agent.runtime.usage import log_model_usage, usage_scope
from so_agent.state_machine import WorkflowState
from so_agent.turn_budget import current_model_turn_override


class _UsageFake:
    def __init__(self, total: int) -> None:
        self.input_tokens = total
        self.output_tokens = 0
        self.total_tokens = total


class _ResultFake:
    def __init__(self, total: int) -> None:
        self.raw_responses = [SimpleNamespace(usage=_UsageFake(total))]


class TestUsageAttribution:
    def test_usage_scope_records_subtask_tokens(self):
        events = EventLogger()
        budget = TokenBudgetTracker(task_id="t1")
        with usage_scope("t1", "s1", 1):
            log_model_usage(events, _ResultFake(120), "code_agent", budget=budget)
        payload = events.get_events(event_type=EVENT_TOKEN_USAGE)[0]["payload"]
        assert payload["subtask_id"] == "s1"
        assert payload["attempt"] == 1
        assert payload["task_id"] == "t1"
        assert payload["total_tokens"] == 120
        assert budget.tokens_for_subtask("s1") == 120
        assert budget.snapshot()["per_subtask"] == {"s1": 120}

    def test_untracked_usage_does_not_create_subtask_entry(self):
        events = EventLogger()
        budget = TokenBudgetTracker(task_id="t1")
        log_model_usage(events, _ResultFake(10), "supervisor", budget=budget)
        assert "subtask_id" not in events.get_events(event_type=EVENT_TOKEN_USAGE)[0][
            "payload"
        ]
        assert budget.snapshot()["per_subtask"] == {}

    def test_observe_subtask_accumulates_across_attempts(self):
        budget = TokenBudgetTracker(task_id="t1")
        usage = {"total_tokens": 40}
        budget.observe_subtask("s1", usage)
        budget.observe_subtask("s1", usage)
        budget.observe_subtask("s2", usage)
        assert budget.tokens_for_subtask("s1") == 80
        assert budget.tokens_for_subtask("s2") == 40


def _spec(subtask_id: str = "s1", **overrides) -> SubtaskSpec:
    params = dict(
        subtask_id=subtask_id,
        title="t",
        instructions="i",
        role="code",
        allowed_tools=[],
        max_attempts=3,
    )
    params.update(overrides)
    return SubtaskSpec(**params)


class TestSchedulerSubtaskBudget:
    async def test_token_budget_violation_stops_retries(self):
        calls: list[int] = []
        checks: list[str] = []

        async def executor(spec: SubtaskSpec, agent: AgentRecord) -> ExecutionResult:
            calls.append(agent.attempt_count)
            return ExecutionResult(
                subtask_id=spec.subtask_id, success=False, error="模拟失败"
            )

        scheduler = SubtaskDAGScheduler(executor, events=EventLogger())

        def check(subtask_id: str) -> str | None:
            checks.append(subtask_id)
            return "子任务 token 超限" if len(checks) > 1 else None

        scheduler.set_subtask_budget_check(check)
        results = await scheduler.schedule("t1", [_spec()])

        assert calls == [1]
        result = results["s1"]
        assert result.success is False
        assert result.metadata["failure_type"] == "budget"
        assert result.metadata["recovery_action"] == "stop"
        assert scheduler.decomposition_issues

    async def test_wall_clock_envelope_stops_retries(self):
        calls: list[int] = []

        async def slow_executor(
            spec: SubtaskSpec, agent: AgentRecord
        ) -> ExecutionResult:
            calls.append(agent.attempt_count)
            await asyncio.sleep(0.2)
            return ExecutionResult(
                subtask_id=spec.subtask_id, success=False, error="模拟失败"
            )

        settings = Settings(_env_file=None, subtask_max_wall_seconds=0.05)
        scheduler = SubtaskDAGScheduler(
            slow_executor, events=EventLogger(), settings=settings
        )
        results = await scheduler.schedule("t1", [_spec()])

        assert calls == [1]
        result = results["s1"]
        assert result.success is False
        assert result.metadata["recovery_action"] == "stop"
        assert result.metadata["failure_type"] == "timeout"


class _FakeCodeAgentServer:
    """假 code_agent MCP Server：按脚本返回 CodeAgentOutput JSON。"""

    def __init__(self, replies) -> None:
        self.requests: list[dict] = []
        self._replies = replies

    async def handle_request(self, request) -> MCPResponse:
        self.requests.append(request.params)
        payload = self._replies(len(self.requests))
        return MCPResponse(
            id=request.id,
            result=text_content(json.dumps(payload, ensure_ascii=False)),
        )


class _ToolRegistryStub:
    def __init__(self, server) -> None:
        self._server = server

    def get_mcp_server(self, name: str):
        return self._server if name == "code_agent" else None


def _make_coordinator(tmp_path, server, *, events=None):
    events = events or EventLogger()
    state = WorkflowState(events=events)
    state.task_request = TaskRequest(task_id="t1", objective="生成报告")
    deps = SimpleNamespace(
        context=SimpleNamespace(sandbox_dir=tmp_path),
        events=events,
        settings=Settings(_env_file=None, subtask_action_repair_max_attempts=1),
        tool_registry=_ToolRegistryStub(server),
    )

    async def fail_supervisor(payload, output_model):
        raise AssertionError("直派路由不应调用主管 Agent")

    deps.run_supervisor = fail_supervisor
    coordinator = ExecutionCoordinator(state, deps)
    return coordinator, events


def _agent() -> AgentRecord:
    return AgentRecord(
        agent_id="t1:s1:primary-1",
        agent_type="dynamic",
        parent_task_id="t1",
        attempt_count=1,
    )


class TestMergedTurnBudgetScope:
    """合并子任务的回合上限覆盖只在其派发作用域内生效。"""

    def _coordinator(self, tmp_path, overrides: list[int | None]):
        class _TurnRecordingServer(_FakeCodeAgentServer):
            async def handle_request(self, request):
                overrides.append(current_model_turn_override())
                return await super().handle_request(request)

        server = _TurnRecordingServer(
            lambda n: {
                "success": True,
                "action_performed": "create",
                "summary": "ok",
            }
        )
        return _make_coordinator(tmp_path, server)

    async def test_merged_subtask_dispatch_raises_turn_budget(self, tmp_path):
        overrides: list[int | None] = []
        coordinator, _ = self._coordinator(tmp_path, overrides)
        coordinator._state.merged_subtask_ids = {"s1"}

        await coordinator._dispatch_subtask_output(_spec(), _agent())

        assert overrides == [
            int(coordinator._settings.max_model_turns)
            + int(coordinator._settings.plan_merge_turn_bonus)
        ]

    async def test_unmerged_subtask_dispatch_keeps_default(self, tmp_path):
        overrides: list[int | None] = []
        coordinator, _ = self._coordinator(tmp_path, overrides)

        await coordinator._dispatch_subtask_output(_spec(), _agent())

        assert overrides == [None]


class TestDirectCodeAgentDispatch:
    async def test_standard_tools_dispatch_directly(self, tmp_path):
        server = _FakeCodeAgentServer(
            lambda n: {
                "success": True,
                "action_performed": "create",
                "file_path": None,
                "summary": "已完成只读核对并给出结论",
                "error": None,
            }
        )
        coordinator, events = _make_coordinator(tmp_path, server)
        spec = _spec()
        result = await coordinator._execute_subtask_attempt(spec, _agent())

        assert result.success is True
        assert result.metadata["route"] == "direct_code_agent"
        assert server.requests[0]["arguments"]["action"] == "auto"
        # 直派必须自行补齐工具证据事件（调度器的确定性验收依赖它）
        actions = [
            record["payload"]["action"]
            for record in events.get_events(event_type=EVENT_TOOL_CALLED)
        ]
        assert "invoke_started" in actions
        assert "invoke_completed" in actions
        call_ids = {
            record["payload"].get("call_id")
            for record in events.get_events(event_type=EVENT_TOOL_CALLED)
        }
        assert len(call_ids) == 1 and None not in call_ids

    async def test_execution_output_is_kept_for_verification(self, tmp_path):
        server = _FakeCodeAgentServer(
            lambda n: {
                "success": True,
                "action_performed": "run",
                "execution_result": {
                    "success": True,
                    "output": '{"consistent": true}',
                },
                "summary": "运行完成",
            }
        )
        coordinator, _ = _make_coordinator(tmp_path, server)
        text = await coordinator._dispatch_code_agent(_spec(), _agent())
        result = coordinator._parse_code_agent_result(text, _spec())
        assert result.metadata["execution_output"] == '{"consistent": true}'

    async def test_write_scope_is_forwarded(self, tmp_path):
        server = _FakeCodeAgentServer(
            lambda n: {"success": True, "action_performed": "create", "summary": "ok"}
        )
        coordinator, _ = _make_coordinator(tmp_path, server)
        await coordinator._dispatch_code_agent(
            _spec(write_scope=["out/report.md"]), _agent()
        )
        assert server.requests[0]["arguments"]["write_scope"] == ["out/report.md"]

    async def test_empty_write_scope_is_not_sent(self, tmp_path):
        server = _FakeCodeAgentServer(
            lambda n: {"success": True, "action_performed": "review", "summary": "ok"}
        )
        coordinator, _ = _make_coordinator(tmp_path, server)
        await coordinator._dispatch_code_agent(_spec(), _agent())
        assert "write_scope" not in server.requests[0]["arguments"]

    async def test_declared_evidence_is_parsed(self, tmp_path):
        server = _FakeCodeAgentServer(
            lambda n: {
                "success": True,
                "action_performed": "execute",
                "file_path": "metrics.json",
                "evidence": ["python analyze.py -> exit 0", "  ", 42],
                "summary": "运行完成",
            }
        )
        coordinator, _ = _make_coordinator(tmp_path, server)
        text = await coordinator._dispatch_code_agent(_spec(), _agent())
        result = coordinator._parse_code_agent_result(text, _spec())
        assert result.evidence == [
            "python analyze.py -> exit 0",
            "execute",
        ]
        assert result.artifacts == ["metrics.json"]

    async def test_sandbox_exit_code_is_lifted_for_verification(self, tmp_path):
        server = _FakeCodeAgentServer(
            lambda n: {
                "success": True,
                "action_performed": "execute",
                "execution_result": {
                    "success": True,
                    "output": "done",
                    "metadata": {"exit_code": 0, "timed_out": False},
                },
                "summary": "运行完成",
            }
        )
        coordinator, _ = _make_coordinator(tmp_path, server)
        text = await coordinator._dispatch_code_agent(_spec(), _agent())
        result = coordinator._parse_code_agent_result(text, _spec())
        assert result.metadata["exit_code"] == 0

    async def test_task_description_stays_scoped_to_subtask(self, tmp_path):
        server = _FakeCodeAgentServer(
            lambda n: {
                "success": True,
                "action_performed": "create",
                "file_path": None,
                "summary": "完成",
                "error": None,
            }
        )
        coordinator, _ = _make_coordinator(tmp_path, server)
        coordinator._task_request = TaskRequest(
            task_id="t1",
            objective="总目标：\n1. 生成 a.csv\n2. 生成 b.csv\n3. 写报告",
            constraints=["只允许在沙箱目录内读写"],
            acceptance_criteria=["a.csv 与 b.csv 都存在且报告非空"],
        )
        spec = _spec(title="生成 a.csv", instructions="只生成 a.csv")
        await coordinator._execute_subtask_attempt(spec, _agent())

        description = server.requests[0]["arguments"]["task_description"]
        assert "只生成 a.csv" in description
        assert "只允许在沙箱目录内读写" in description
        assert "生成 b.csv" not in description
        assert "b.csv 都存在" not in description

    async def test_non_standard_tools_keep_supervisor_route(self, tmp_path):
        calls: list[dict] = []

        async def supervisor(payload, output_model):
            calls.append(payload)
            return SimpleNamespace(
                final_result=ExecutionResult(
                    subtask_id="s1", success=True, output="主管路径完成"
                ).model_dump_json()
            )

        events = EventLogger()
        state = WorkflowState(events=events)
        state.task_request = TaskRequest(task_id="t1", objective="demo")
        deps = SimpleNamespace(
            context=SimpleNamespace(sandbox_dir=tmp_path),
            events=events,
            settings=Settings(_env_file=None),
            tool_registry=_ToolRegistryStub(
                _FakeCodeAgentServer(lambda n: {"success": True})
            ),
            run_supervisor=supervisor,
        )
        coordinator = ExecutionCoordinator(state, deps)
        spec = _spec(allowed_tools=["generate_chart"])
        result = await coordinator._execute_subtask_attempt(spec, _agent())

        assert result.success is True
        assert len(calls) == 1


class TestActionLevelRepair:
    async def test_flipped_success_repaired_once(self, tmp_path):
        def replies(n: int) -> dict:
            if n == 1:
                # 首次自述成功但未真正落盘（write_scope 将翻转）
                return {
                    "success": True,
                    "action_performed": "create",
                    "file_path": "report.md",
                    "summary": "已生成报告",
                }
            (tmp_path / "report.md").write_text("done", encoding="utf-8")
            return {
                "success": True,
                "action_performed": "create",
                "file_path": "report.md",
                "summary": "已补齐报告文件",
            }

        server = _FakeCodeAgentServer(replies)
        coordinator, events = _make_coordinator(tmp_path, server)
        spec = _spec(write_scope=["report.md"])
        result = await coordinator._execute_subtask_attempt(spec, _agent())

        assert result.success is True
        assert len(server.requests) == 2
        repair = events.get_events(event_type=EVENT_ACTION_REPAIR)
        assert repair[0]["payload"]["repair_success"] is True

    async def test_repair_failure_returns_combined_error(self, tmp_path):
        server = _FakeCodeAgentServer(
            lambda n: {
                "success": True,
                "action_performed": "create",
                "file_path": "report.md",
                "summary": "已生成报告",
            }
        )
        coordinator, events = _make_coordinator(tmp_path, server)
        spec = _spec(write_scope=["report.md"])
        result = await coordinator._execute_subtask_attempt(spec, _agent())

        assert result.success is False
        assert "动作级修复后仍未通过" in result.error
        assert len(server.requests) == 2


class TestSubtaskBudgetViolation:
    def test_violation_reported_after_limit(self, tmp_path):
        events = EventLogger()
        state = WorkflowState(events=events)
        state.task_request = TaskRequest(task_id="t1", objective="demo")
        budget = TokenBudgetTracker(task_id="t1")
        deps = SimpleNamespace(
            context=SimpleNamespace(sandbox_dir=tmp_path, token_budget=budget),
            events=events,
            settings=Settings(_env_file=None, subtask_max_tokens=100),
            tool_registry=_ToolRegistryStub(None),
        )
        coordinator = ExecutionCoordinator(state, deps)
        budget.observe_subtask("s1", {"total_tokens": 60})
        assert coordinator._subtask_budget_violation("s1") is None
        budget.observe_subtask("s1", {"total_tokens": 60})
        reason = coordinator._subtask_budget_violation("s1")
        assert reason is not None and "超过上限 100" in reason

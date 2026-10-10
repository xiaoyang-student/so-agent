"""集成测试：替代 Agent 生命周期真实时点通知（任务 34）。

修复目标：旧实现中调度器内部已创建并执行完替代 Agent，协调器才在
批次结算（``_settle_failure_states``）时事后补记 replacement_requested /
replacement_executing，导致状态时间线与实际执行相反。修复后调度器在
真实生命周期时点经受控回调（编排器注入执行协调器接收端，
asyncio.Lock 串行化全局 FSM 迁移）实时落账。

覆盖：

- 时序：requested 早于替代 Agent 创建（agent_created）；executing 早于
  替代 Agent 首个 task_dispatched；completed / failed 晚于替代执行结果；
- 终止路径：尝试耗尽 / 执行异常 / 超时 / 取消均产生顺序正确的
  replacement_failed（failure_category 区分），不增加替代次数、
  不触发二级替代；通知回调抛错不影响调度主流程；
- 并发：两个子任务同时替代时全局 FSM 迁移全部合法、影子状态机不冻结、
  每个 request_id 的生命周期顺序独立；
- 不变量：替代链深度一（替代 Agent 无再申请资格）、策略驱动重试
  （同因连续失败即抑制，指纹互异时最多三次；不可重试类即时终止）、
  ``_settle_failure_states`` 不重放历史替换申请、生命周期事件不泄露
  业务产出、引擎装配注入与鸭子类型调度器兼容。

模拟执行器（ScriptedExecutor）替代主管 Agent 调用，不消耗真实 API。
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from so_agent.config import Settings
from so_agent.context import ProjectContext
from so_agent.execution_coordinator import ExecutionCoordinator
from so_agent.models import (
    AgentRecord,
    ExecutionResult,
    SubtaskSpec,
    TaskRequest,
    TaskStatus,
)
from so_agent.planning_coordinator import PlanningCoordinator
from so_agent.runtime.events import (
    EVENT_AGENT_CREATED,
    EVENT_MESSAGE_ROUTED,
    EVENT_REPLACEMENT_COMPLETED,
    EVENT_REPLACEMENT_EXECUTING,
    EVENT_REPLACEMENT_FAILED,
    EVENT_REPLACEMENT_REQUESTED,
    EVENT_STATUS_TRANSITION,
    EVENT_TASK_DISPATCHED,
    EventLogger,
)
from so_agent.runtime.permissions import PermissionGateway
from so_agent.runtime.registry import AgentRegistry
from so_agent.runtime.scheduler import SubtaskDAGScheduler
from so_agent.runtime.supervisor_router import SupervisorRouter
from so_agent.runtime.tool_registry import ToolRegistry
from so_agent.state_machine import WorkflowState
from so_agent.workflow import WorkflowEngine
from so_agent.workflow_support import HumanEscalationHandler, WorkflowDeps

TASK_ID = "t1"

# 并发替代场景下允许出现的全局状态迁移（received 为协调器级初态）
_ALLOWED_TRANSITIONS: set[tuple[str, str]] = {
    ("received", "executing"),
    ("executing", "replacement_requested"),
    ("replacement_requested", "replacement_executing"),
    ("replacement_executing", "executing"),
    ("replacement_executing", "decomposition_issue"),
}


def make_spec(subtask_id: str = "s1", **kwargs) -> SubtaskSpec:
    defaults = dict(
        subtask_id=subtask_id,
        title=f"子任务 {subtask_id}",
        instructions=f"完成 {subtask_id}",
        role="code",
        allowed_tools=[],  # 无工具需求 → 无工具缺口流程
        max_attempts=3,
    )
    defaults.update(kwargs)
    return SubtaskSpec(**defaults)


class ScriptedExecutor:
    """可编程模拟执行器：按调用序回放脚本并记录收到的 AgentRecord。

    脚本条目：
    - ``"ok"`` → 成功结果（output 取构造参数）；
    - ``"fail"`` → 失败结果（错误文本随尝试序号变化）；
    - BaseException 实例 → 抛出该异常（含 CancelledError）；
    - ExecutionResult 实例 → 原样返回。
    脚本耗尽后重复最后一项（max_attempts 耗尽路径天然覆盖）。
    """

    def __init__(
        self,
        script: list,
        *,
        output: str = "任务完成产出内容",
        delay: float = 0.0,
    ) -> None:
        self._script = list(script)
        self._output = output
        self._delay = delay
        self.calls: list[AgentRecord] = []

    async def __call__(self, spec: SubtaskSpec, agent: AgentRecord) -> ExecutionResult:
        self.calls.append(agent)
        if self._delay > 0:
            await asyncio.sleep(self._delay)
        index = min(len(self.calls) - 1, len(self._script) - 1)
        outcome = self._script[index]
        if isinstance(outcome, BaseException):
            raise outcome
        if outcome == "fail":
            return ExecutionResult(
                subtask_id=spec.subtask_id,
                success=False,
                error=f"第 {agent.attempt_count} 次尝试失败：模拟错误",
            )
        if outcome == "ok":
            return ExecutionResult(
                subtask_id=spec.subtask_id, success=True, output=self._output
            )
        return outcome


def build_env(tmp_path, **settings_kwargs):
    """装配共享状态 + 执行协调器（真实依赖，不启动工作流、无 LLM 调用）。"""
    resolved = Settings(_env_file=None, **settings_kwargs)
    context = ProjectContext(sandbox_dir=tmp_path, project_name="replacement-timing")
    events = EventLogger(context=context)
    registry = AgentRegistry(context=context, settings=resolved)
    tool_registry = ToolRegistry(generated_tools_dir=tmp_path / "generated_tools")
    gateway = PermissionGateway(
        context=context,
        settings=resolved,
        events=events,
        tool_registry=tool_registry,
    )
    deps = WorkflowDeps(
        context=context,
        settings=resolved,
        events=events,
        registry=registry,
        gateway=gateway,
        router=SupervisorRouter(registry=registry, events=events),
        tool_registry=tool_registry,
        escalation_handler=HumanEscalationHandler(),
        orchestrator=SimpleNamespace(name="stub-supervisor"),
        review_agent=SimpleNamespace(name="stub-reviewer"),
    )
    state = WorkflowState(events=events)
    planning = PlanningCoordinator(state, deps)
    execution = ExecutionCoordinator(state, deps)
    deps.issue_grants = planning._issue_grants
    state.task_request = TaskRequest(
        task_id=TASK_ID,
        objective="验证替代生命周期时序",
        constraints=[],
        acceptance_criteria=[],
        original_input="验证替代生命周期时序",
    )
    return SimpleNamespace(
        context=context,
        settings=resolved,
        events=events,
        registry=registry,
        gateway=gateway,
        deps=deps,
        state=state,
        execution=execution,
    )


async def run_batch(env, specs, primary, replacement, *, attempt_timeout=None):
    """挂接真实调度器（注入生命周期接收端）并执行一个批次。"""
    scheduler = SubtaskDAGScheduler(
        primary,
        replacement_executor=replacement,
        context=env.context,
        settings=env.settings,
        registry=env.registry,
        events=env.events,
        attempt_timeout=attempt_timeout,
        lifecycle_notifier=env.execution.handle_replacement_lifecycle,
    )
    env.deps.scheduler = scheduler
    return await env.execution._execute_subtasks(specs)


def indexed(context, event_type: str) -> list[tuple[int, dict]]:
    """返回 ``[(全局序号, 事件记录), ...]``（按发生顺序）。"""
    return [
        (i, r)
        for i, r in enumerate(context.event_log)
        if r["event_type"] == event_type
    ]


def count_type(context, event_type: str) -> int:
    return sum(1 for r in context.event_log if r["event_type"] == event_type)


def status_transitions(context) -> list[tuple[str, str]]:
    return [
        (r["payload"]["from_status"], r["payload"]["to_status"])
        for r in context.event_log
        if r["event_type"] == EVENT_STATUS_TRANSITION
    ]


# ===========================================================================
# 一、真实时点时序（成功路径）
# ===========================================================================
class TestRealtimeOrdering:
    async def test_requested_precedes_creation_and_dispatch(self, tmp_path):
        """requested 早于替代 Agent 创建；executing 早于替代首个 dispatch。"""
        env = build_env(tmp_path)
        primary = ScriptedExecutor(["fail", "fail", "fail"])
        replacement = ScriptedExecutor(["ok"])
        batch = await run_batch(env, [make_spec("s1")], primary, replacement)

        assert batch["s1"].success is True
        log = env.context.event_log

        def idx_of(event_type: str, position: int = 0) -> int:
            return indexed(env.context, event_type)[position][0]

        created = indexed(env.context, EVENT_AGENT_CREATED)
        assert len(created) == 2  # 原始 + 替代
        repl_agent_id = replacement.calls[0].agent_id
        dispatches = [
            (i, r)
            for i, r in indexed(env.context, EVENT_TASK_DISPATCHED)
            if r["payload"]["agent_id"] == repl_agent_id
        ]
        assert len(dispatches) == 1  # 替代一次成功

        # 核心时序：requested（调度器事件）→ requested 迁移 →
        # 替代创建 → executing 迁移 → executing 事件 →
        # 替代 dispatch → 替代完成 → completed 事件
        order = [
            idx_of(EVENT_REPLACEMENT_REQUESTED),
            next(
                i
                for i, r in indexed(env.context, EVENT_STATUS_TRANSITION)
                if r["payload"]["to_status"] == "replacement_requested"
            ),
            created[1][0],
            next(
                i
                for i, r in indexed(env.context, EVENT_STATUS_TRANSITION)
                if r["payload"]["to_status"] == "replacement_executing"
            ),
            idx_of(EVENT_REPLACEMENT_EXECUTING),
            dispatches[0][0],
            idx_of(EVENT_REPLACEMENT_COMPLETED),
        ]
        assert order == sorted(order), f"生命周期时序错乱：{order}"

        # 替代执行结果（dispatch / 成功）必然早于 completed 通知
        assert dispatches[0][0] < idx_of(EVENT_REPLACEMENT_COMPLETED)

        # 原始 Agent 同因两次尝试（第二次失败即同因抑制）后发起替换请求
        primary_dispatches = [
            i
            for i, r in indexed(env.context, EVENT_TASK_DISPATCHED)
            if r["payload"]["agent_id"] == primary.calls[0].agent_id
        ]
        assert len(primary_dispatches) == 2
        assert max(primary_dispatches) < idx_of(EVENT_REPLACEMENT_REQUESTED)

        # 全局状态链精确（批次开头的 executing + 替代三段 + 回到 executing）
        assert status_transitions(env.context) == [
            ("received", "executing"),
            ("executing", "replacement_requested"),
            ("replacement_requested", "replacement_executing"),
            ("replacement_executing", "executing"),
        ]

    async def test_approval_message_routed_with_full_request(self, tmp_path):
        """requested 落账时申请书仍经主管路由上报（星型通信不变）。"""
        env = build_env(tmp_path)
        scheduler = SubtaskDAGScheduler(
            ScriptedExecutor(["fail", "fail", "fail"]),
            replacement_executor=ScriptedExecutor(["ok"]),
            context=env.context,
            settings=env.settings,
            registry=env.registry,
            events=env.events,
            lifecycle_notifier=env.execution.handle_replacement_lifecycle,
        )
        env.deps.scheduler = scheduler
        await env.execution._execute_subtasks([make_spec("s1")])

        routed = indexed(env.context, EVENT_MESSAGE_ROUTED)
        assert len(routed) == 1
        message = routed[0][1]["payload"]["message"]
        assert message["message_type"] == "replacement_request"
        assert message["verdict"] == "approved"
        assert message["decided_by"] == "supervisor"
        assert message["subtask_id"] == "s1"
        assert message["request"]["request_id"] == (
            scheduler.replacement_requests[0].request_id
        )
        # 失败历史含替换申请条目（供重规划参考）
        history = env.state.agent_failure_history
        assert sum(1 for item in history if "替换申请" in item) == 1


# ===========================================================================
# 二、终止路径（失败 / 异常 / 超时 / 取消 / 注册拒绝）
# ===========================================================================
class TestFailureTermination:
    async def test_attempts_exhausted_notifies_failed_once(self, tmp_path):
        """替代同因两次尝试耗尽（第二次失败即抑制）：failed 通知一次
        （attempts_exhausted），无二级替代。"""
        env = build_env(tmp_path)
        primary = ScriptedExecutor(["fail", "fail", "fail"])
        replacement = ScriptedExecutor(["fail", "fail", "fail"])
        batch = await run_batch(env, [make_spec("s1")], primary, replacement)

        assert batch["s1"].success is False
        failed = indexed(env.context, EVENT_REPLACEMENT_FAILED)
        assert len(failed) == 1
        payload = failed[0][1]["payload"]
        assert payload["failure_category"] == "attempts_exhausted"
        assert payload["attempt_count"] == 2
        assert "模拟错误" in payload["summary"]
        assert count_type(env.context, EVENT_REPLACEMENT_COMPLETED) == 0
        # 替代链深度一：替代失败后无第二次替换申请；分解问题照常生成
        assert len(env.deps.scheduler.replacement_requests) == 1
        assert len(env.deps.scheduler.decomposition_issues) == 1
        # 状态链：失败通知不迁移全局状态（停留在 replacement_executing，
        # decomposition_issue 迁移由重规划循环的 _handle_decomposition_issue
        # 触发，不在 _execute_subtasks 范围内）
        assert status_transitions(env.context)[-1] == (
            "replacement_requested",
            "replacement_executing",
        )

    async def test_executor_exception_maps_to_attempts_exhausted(self, tmp_path):
        """替代执行器抛异常：分类为不可重试（internal）即时终止，
        summary 含异常类型（仍走 attempts_exhausted 失败通知）。"""
        env = build_env(tmp_path)
        primary = ScriptedExecutor(["fail", "fail", "fail"])
        replacement = ScriptedExecutor([ValueError("boom-exception")])
        batch = await run_batch(env, [make_spec("s1")], primary, replacement)

        assert batch["s1"].success is False
        failed = indexed(env.context, EVENT_REPLACEMENT_FAILED)
        assert len(failed) == 1
        payload = failed[0][1]["payload"]
        assert payload["failure_category"] == "attempts_exhausted"
        assert payload["attempt_count"] == 1
        assert "执行异常" in payload["summary"]
        assert "ValueError" in payload["summary"]
        assert len(env.deps.scheduler.replacement_requests) == 1

    async def test_attempt_timeout_maps_to_attempts_exhausted(self, tmp_path):
        """替代执行超时：wait_for 转失败尝试直至耗尽，summary 含超时说明。"""
        env = build_env(tmp_path)
        primary = ScriptedExecutor(["fail", "fail", "fail"])  # 快速失败不受超时影响
        replacement = ScriptedExecutor(["ok"], delay=0.5)  # 必然超过 0.05s 超时
        batch = await run_batch(
            env, [make_spec("s1")], primary, replacement, attempt_timeout=0.05
        )

        assert batch["s1"].success is False
        failed = indexed(env.context, EVENT_REPLACEMENT_FAILED)
        assert len(failed) == 1
        payload = failed[0][1]["payload"]
        assert payload["failure_category"] == "attempts_exhausted"
        assert "单次执行超时" in payload["summary"]
        assert count_type(env.context, EVENT_REPLACEMENT_COMPLETED) == 0
        assert len(env.deps.scheduler.replacement_requests) == 1

    async def test_cancellation_notifies_failed_and_propagates(self, tmp_path):
        """替代执行被取消：failed(cancelled) 通知后取消继续传播，无二级替代。"""
        env = build_env(tmp_path)
        primary = ScriptedExecutor(["fail", "fail", "fail"])
        replacement = ScriptedExecutor([asyncio.CancelledError()])
        with pytest.raises(asyncio.CancelledError):
            await run_batch(env, [make_spec("s1")], primary, replacement)

        failed = indexed(env.context, EVENT_REPLACEMENT_FAILED)
        assert len(failed) == 1
        payload = failed[0][1]["payload"]
        assert payload["failure_category"] == "cancelled"
        assert count_type(env.context, EVENT_REPLACEMENT_COMPLETED) == 0
        # 终止通知不增加替代次数、不触发二级替代
        assert len(env.deps.scheduler.replacement_requests) == 1
        assert len(env.deps.scheduler.decomposition_issues) == 0
        # 原始 Agent 同因两次尝试（第二次失败即抑制）
        assert len(primary.calls) == 2

    async def test_registration_rejected_payload_recorded(self, tmp_path):
        """创建失败（注册拒绝）通知：接收端完整留痕且不迁移全局状态。"""
        env = build_env(tmp_path)
        before = env.state.current_status
        payload = {
            "task_id": TASK_ID,
            "subtask_id": "s9",
            "request_id": "repl-test-0001",
            "agent_id": f"{TASK_ID}:s9:replacement-abc123",
            "attempt_count": 0,
            "failure_category": "agent_registration_rejected",
            "summary": "替代 Agent 注册被拒绝（动态 Agent 数量已达上限）",
        }
        await env.execution.handle_replacement_lifecycle(
            EVENT_REPLACEMENT_FAILED, payload
        )

        failed = indexed(env.context, EVENT_REPLACEMENT_FAILED)
        assert len(failed) == 1
        recorded = failed[0][1]["payload"]
        assert recorded["failure_category"] == "agent_registration_rejected"
        assert recorded["request_id"] == "repl-test-0001"
        assert recorded["attempt_count"] == 0
        # 终止通知不迁移全局状态（分解问题链路后续处理）
        assert env.state.current_status == before

    async def test_broken_notifier_does_not_break_scheduling(self, tmp_path):
        """通知回调抛错：调度主流程不受影响（替代照常完成）。"""
        env = build_env(tmp_path)

        async def bad_notifier(event: str, payload: dict) -> None:
            raise RuntimeError("通知端崩溃")

        scheduler = SubtaskDAGScheduler(
            ScriptedExecutor(["fail", "fail", "fail"]),
            replacement_executor=ScriptedExecutor(["ok"]),
            context=env.context,
            settings=env.settings,
            registry=env.registry,
            events=env.events,
            lifecycle_notifier=bad_notifier,
        )
        env.deps.scheduler = scheduler
        batch = await env.execution._execute_subtasks([make_spec("s1")])

        assert batch["s1"].success is True
        assert len(scheduler.replacement_requests) == 1
        # 接收端未被调用：无替代状态迁移
        assert all(
            to != "replacement_requested"
            for _, to in status_transitions(env.context)
        )


# ===========================================================================
# 三、并发替代：FSM 合法性与每请求顺序独立
# ===========================================================================
class TestConcurrentReplacements:
    async def test_two_concurrent_replacements_fsm_stays_legal(self, tmp_path):
        """两个子任务同时替代：全局迁移全部合法、影子图不冻结、每请求独立。"""
        env = build_env(
            tmp_path,
            max_concurrency=2,
            # 全局替换额度按 task_id 计数：并发两子任务均需申请资格
            max_replacement_requests=2,
        )
        primary = ScriptedExecutor(["fail", "fail", "fail"])
        replacement = ScriptedExecutor(["ok"])
        batch = await run_batch(
            env, [make_spec("s1"), make_spec("s2")], primary, replacement
        )

        assert batch["s1"].success is True
        assert batch["s2"].success is True
        assert len(primary.calls) == 4  # 每子任务同因两次尝试（抑制不变量）
        assert len(replacement.calls) == 2
        # 替代 Agent 固定无再申请资格（替代链深度一）
        assert all(
            agent.can_request_replacement is False for agent in replacement.calls
        )
        assert len(env.deps.scheduler.replacement_requests) == 2

        # 1) 全局 FSM：记录在案的每次迁移都在允许集合内（并发交错下
        #    同状态迁移按既有语义 no-op，不产生非法转换）
        transitions = status_transitions(env.context)
        assert transitions[0] == ("received", "executing")
        for pair in transitions[1:]:
            assert pair in {
                ("executing", "replacement_requested"),
                ("replacement_requested", "replacement_executing"),
                ("replacement_executing", "executing"),
            }, f"非法全局迁移：{pair}"
        assert transitions[-1][1] == "executing"
        # 影子状态机活性由专用确定性交错测试覆盖（下测）：协调器级
        # 简化环境影子图从 idle 起（真实流水线经规划/评审/授权阶段
        # 推进至 executing），不在此重复断言

        # 2) 每个 request_id 的生命周期顺序独立：
        #    调度器 requested 事件 → 协调器 executing 事件 → completed 事件
        request_events = {
            r["payload"]["request"]["request_id"]: i
            for i, r in indexed(env.context, EVENT_REPLACEMENT_REQUESTED)
        }
        assert len(request_events) == 2
        executing_marks = {
            r["payload"]["request_id"]: i
            for i, r in indexed(env.context, EVENT_REPLACEMENT_EXECUTING)
        }
        completed_marks = {
            r["payload"]["request_id"]: i
            for i, r in indexed(env.context, EVENT_REPLACEMENT_COMPLETED)
        }
        assert set(executing_marks) == set(request_events)
        assert set(completed_marks) == set(request_events)
        for request_id, requested_idx in request_events.items():
            assert requested_idx < executing_marks[request_id], request_id
            assert executing_marks[request_id] < completed_marks[request_id]
            assert completed_marks[request_id] < len(env.context.event_log)

        # 3) 每个请求的批准消息恰好一条（并发下去重不重放）
        routed = [
            r
            for _, r in indexed(env.context, EVENT_MESSAGE_ROUTED)
            if r["payload"]["message"].get("message_type") == "replacement_request"
        ]
        assert len(routed) == 2

    def test_shadow_machine_survives_interleaved_sequences(self):
        """影子状态机在两链并发交错的极端序列下保持可推进（不冻结）。

        极端交错：s1.requested 后、s1.executing 前，s2.requested 到达
        （同状态 no-op）；随后任一子任务的 executing / completed 均能
        正常推进影子图。确定性验证用户约束：不靠跳过状态校验，而是
        全局单值语义 + 同状态 no-op + 目标阶段幂等返回天然保证合法。
        """
        state = WorkflowState(events=None)
        # 真实流水线的影子图推进路径：plan → review_plan → grant → execute
        state.machine.send("plan")
        state.machine.send("review_plan")
        state.machine.send("grant")
        state.machine.send("execute")
        assert state.phase == "executing"

        state.transition(
            TaskStatus.REPLACEMENT_REQUESTED, subtask_id="s1", request_id="r1"
        )
        assert state.phase == "replacing_agent"
        # s2 的请求在 s1 执行前到达：权威同状态迁移 no-op，影子图不动
        state.transition(
            TaskStatus.REPLACEMENT_REQUESTED, subtask_id="s2", request_id="r2"
        )
        assert state.phase == "replacing_agent"
        # s1 替代开始：replacing_agent → executing 合法推进
        state.transition(
            TaskStatus.REPLACEMENT_EXECUTING, subtask_id="s1", request_id="r1"
        )
        assert state.phase == "executing"
        # s2 替代开始：同状态 no-op
        state.transition(
            TaskStatus.REPLACEMENT_EXECUTING, subtask_id="s2", request_id="r2"
        )
        assert state.phase == "executing"
        # s1 替代成功：全局回 executing（影子图已在目标阶段，幂等返回）
        state.transition(
            TaskStatus.EXECUTING, subtask_id="s1", request_id="r1"
        )
        assert state.phase == "executing"
        # s2 替代成功：同状态 no-op
        state.transition(
            TaskStatus.EXECUTING, subtask_id="s2", request_id="r2"
        )
        assert state.phase == "executing"


# ===========================================================================
# 四、不变量：不重放 / 不泄露 / 引擎装配
# ===========================================================================
class TestInvariants:
    async def test_settle_does_not_replay_replacement_history(self, tmp_path):
        """历史替换申请不重放：settle 收敛后重复结算与后续批次均无副作用。"""
        env = build_env(tmp_path)
        primary = ScriptedExecutor(["fail", "fail", "fail"])
        replacement = ScriptedExecutor(["ok"])
        batch = await run_batch(env, [make_spec("s1")], primary, replacement)

        baseline = {
            "executing": count_type(env.context, EVENT_REPLACEMENT_EXECUTING),
            "completed": count_type(env.context, EVENT_REPLACEMENT_COMPLETED),
            "transitions": len(status_transitions(env.context)),
            "routed": count_type(env.context, EVENT_MESSAGE_ROUTED),
        }
        assert baseline["completed"] == 1

        # 重复结算同一批次：无任何新替代状态迁移 / 事件 / 消息
        env.execution._settle_failure_states(batch)
        first_scheduler = env.deps.scheduler  # 第二批次会替换调度器引用
        # 后续普通批次（另一子任务直接成功）同样不重放旧请求
        plain = ScriptedExecutor(["ok"])
        await run_batch(
            env, [make_spec("s2")], plain, ScriptedExecutor(["ok"])
        )

        assert count_type(env.context, EVENT_REPLACEMENT_EXECUTING) == (
            baseline["executing"]
        )
        assert count_type(env.context, EVENT_REPLACEMENT_COMPLETED) == (
            baseline["completed"]
        )
        # 第二批次的批次开头 executing 为同状态迁移（既有语义 no-op），
        # 不产生任何新的事件记录
        assert len(status_transitions(env.context)) == baseline["transitions"]
        assert status_transitions(env.context)[-1] == (
            "replacement_executing",
            "executing",
        )
        replacement_routed = [
            r
            for r in env.context.event_log
            if r["event_type"] == EVENT_MESSAGE_ROUTED
            and r["payload"]["message"].get("message_type") == "replacement_request"
        ]
        assert len(replacement_routed) == 1
        # 去重集合与源列表保持一致：旧请求仍留档但不被再次消费
        assert env.execution._seen_replacement_requests == {
            first_scheduler.replacement_requests[0].request_id
        }
        assert len(first_scheduler.replacement_requests) == 1
        assert sum(
            1 for item in env.state.agent_failure_history if "替换申请" in item
        ) == 1

    async def test_lifecycle_events_leak_no_business_output(self, tmp_path):
        """生命周期事件仅携带控制信息，不泄露替代 Agent 业务产出。"""
        secret = "机密业务产出-TOP-SECRET-RESULT-CONTENT"
        env = build_env(tmp_path)
        primary = ScriptedExecutor(["fail", "fail", "fail"])
        replacement = ScriptedExecutor(["ok"], output=secret)
        await run_batch(env, [make_spec("s1")], primary, replacement)

        watched = (
            EVENT_REPLACEMENT_REQUESTED,
            EVENT_REPLACEMENT_EXECUTING,
            EVENT_REPLACEMENT_COMPLETED,
            EVENT_REPLACEMENT_FAILED,
            EVENT_STATUS_TRANSITION,
        )
        for event_type in watched:
            for _, record in indexed(env.context, event_type):
                blob = json.dumps(record["payload"], ensure_ascii=False)
                assert secret not in blob, f"{event_type} 泄露业务产出"
        # completed 通知载荷键集合有限（仅控制字段 + 非敏感摘要）
        completed = indexed(env.context, EVENT_REPLACEMENT_COMPLETED)[0][1]
        assert set(completed["payload"]) == {
            "task_id",
            "subtask_id",
            "request_id",
            "agent_id",
            "attempt_count",
            "summary",
        }
        assert secret not in completed["payload"]["summary"]

    async def test_engine_injects_lifecycle_notifier(self, tmp_path):
        """引擎装配：编排器把执行协调器接收端注入真实调度器。"""
        context = ProjectContext(sandbox_dir=tmp_path, project_name="engine-inject")
        engine = WorkflowEngine(
            context,
            settings=Settings(_env_file=None),
            orchestrator=SimpleNamespace(name="fake-supervisor"),
            review_agent=SimpleNamespace(name="fake-reviewer"),
        )
        # 绑定方法每次访问产生新对象，用相等比较（同 self + 同函数）
        assert engine._scheduler._lifecycle_notifier == (
            engine._execution_coordinator.handle_replacement_lifecycle
        )

    async def test_engine_accepts_scheduler_without_notifier(self, tmp_path):
        """鸭子类型兼容：无 set_lifecycle_notifier 的调度器装配直接跳过。"""

        class LegacyScheduler:
            def __init__(self) -> None:
                self.replacement_requests: list = []
                self.decomposition_issues: list = []

            async def schedule(self, task_id, specs):
                return {}

        context = ProjectContext(sandbox_dir=tmp_path, project_name="engine-legacy")
        engine = WorkflowEngine(
            context,
            settings=Settings(_env_file=None),
            orchestrator=SimpleNamespace(name="fake-supervisor"),
            review_agent=SimpleNamespace(name="fake-reviewer"),
            scheduler=LegacyScheduler(),
        )
        assert getattr(engine._scheduler, "set_lifecycle_notifier", None) is None
        assert engine._deps.scheduler is engine._scheduler

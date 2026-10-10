"""单元测试：授权生命周期加固（Task #31，proposal/签发分离 + 统一幂等清理）。

覆盖：

1. 授权候选（proposal）与真实签发分离：
   * ``_propose_grants`` 生成的候选不注册授权网关（get_grant 无登记、
     verify_grant 恒拒绝），被拒计划在网关中没有任何有效签发记录；
   * 授权评审输入携带候选（评审能看到"预计授权工具"）；
   * 两路评审（计划评审 + 授权评审）均通过后才真实签发 ToolGrant；
   * 候选凭证被传入工具包装器时被运行时校验拦截。
2. 统一幂等清理：
   * ``revoke_grants_for_task`` 按任务撤销并写入明确 reason；
   * ``_revoke_active_grants`` 重复调用无重复事件、无任务时空操作、
     网关异常时内部消化不外抛（finally 路径安全）。
3. 重规划生命周期：进入重规划先清理旧版本授权；通过后签发新版本授权；
   重规划被拒时无任何活跃授权残留。
4. 动态工具补发：未注册工具不进候选白名单；执行阶段创建成功后再补发
   真实授权（最小工具集合）。
5. 执行窗口回收理由（``execution_window_closed``）与返工窗口重签发回归。
6. 安全回归：候选凭证 / 过期授权 / 已撤销授权持续被拒绝。
"""

from __future__ import annotations

import json
from datetime import timedelta
from types import SimpleNamespace

from agents import FunctionTool
from agents.tool_context import ToolContext

import so_agent.tool_packages.code_agent.agent as code_agent_agent_mod
from so_agent.config import Settings
from so_agent.context import ProjectContext
from so_agent.execution_coordinator import ExecutionCoordinator
from so_agent.models import (
    ExecutionResult,
    GeneratedToolManifest,
    ReviewDecision,
    SubtaskSpec,
    TaskRequest,
    _utc_now,
)
from so_agent.orchestrator import PlanningOutput
from so_agent.planning_coordinator import PlanningCoordinator
from so_agent.runtime.events import (
    EVENT_TOOL_GRANTED,
    EVENT_TOOL_REVOKED,
    EventLogger,
)
from so_agent.runtime.permissions import PermissionGateway
from so_agent.runtime.registry import AgentRegistry, DEFAULT_SUPERVISOR_ID
from so_agent.runtime.supervisor_router import SupervisorRouter
from so_agent.runtime.tool_registry import ToolRegistry
from so_agent.state_machine import WorkflowState
from so_agent.tool_packages.review_agent.agent import ReviewInput, ReviewStage
from so_agent.tool_packages.subagent_creator.agent import (
    _wrap_tool_with_permission_check,
)
from so_agent.workflow_support import HumanEscalationHandler, WorkflowDeps

SUPERVISOR = DEFAULT_SUPERVISOR_ID
TASK_ID = "t1"
SLOT = f"{TASK_ID}:s1:grant-slot"  # 规划阶段按子任务槽位签发的授权主体


def make_spec(
    subtask_id: str = "s1", *, allowed_tools: tuple[str, ...] = ("read_file",)
) -> SubtaskSpec:
    return SubtaskSpec(
        subtask_id=subtask_id,
        title=f"子任务 {subtask_id}",
        instructions=f"完成 {subtask_id}",
        role="code",
        allowed_tools=list(allowed_tools),
        max_attempts=3,
    )


class FakeScheduler:
    """鸭子类型调度器：默认返回全部成功的批次结果。"""

    def __init__(self, builder=None):
        self._builder = builder
        self.calls: list[tuple[str, list[SubtaskSpec]]] = []
        self.replacement_requests: list = []
        self.decomposition_issues: list = []

    async def schedule(self, task_id: str, specs: list[SubtaskSpec]):
        self.calls.append((task_id, list(specs)))
        if self._builder is not None:
            return self._builder(specs)
        return {
            spec.subtask_id: ExecutionResult(
                subtask_id=spec.subtask_id, success=True, output="子任务完成产出内容"
            )
            for spec in specs
        }


def build_env(tmp_path, specs: list[SubtaskSpec], *, settings=None):
    """装配共享状态 + 规划/执行协调器（真实依赖，不启动工作流）。"""
    resolved_settings = settings or Settings(_env_file=None)
    context = ProjectContext(sandbox_dir=tmp_path, project_name="proposal-test")
    events = EventLogger(context=context)
    registry = AgentRegistry(context=context, settings=resolved_settings)
    tool_registry = ToolRegistry(generated_tools_dir=tmp_path / "generated_tools")
    gateway = PermissionGateway(
        context=context,
        settings=resolved_settings,
        events=events,
        tool_registry=tool_registry,
    )
    deps = WorkflowDeps(
        context=context,
        settings=resolved_settings,
        events=events,
        registry=registry,
        gateway=gateway,
        router=SupervisorRouter(registry=registry, events=events),
        tool_registry=tool_registry,
        escalation_handler=HumanEscalationHandler(),
        orchestrator=SimpleNamespace(name="stub-supervisor"),
        review_agent=SimpleNamespace(name="stub-reviewer"),
        scheduler=FakeScheduler(),
    )
    state = WorkflowState(events=events)
    planning = PlanningCoordinator(state, deps)
    execution = ExecutionCoordinator(state, deps)
    deps.issue_grants = planning._issue_grants
    state.task_request = TaskRequest(
        task_id=TASK_ID,
        objective="验证授权生命周期",
        constraints=[],
        acceptance_criteria=[],
        original_input="验证授权生命周期",
    )
    state.subtasks = list(specs)
    return SimpleNamespace(
        context=context,
        settings=resolved_settings,
        events=events,
        registry=registry,
        tool_registry=tool_registry,
        gateway=gateway,
        deps=deps,
        state=state,
        planning=planning,
        execution=execution,
        scheduler=deps.scheduler,
    )


def granted_events(events: EventLogger) -> list[dict]:
    return events.get_events(event_type=EVENT_TOOL_GRANTED)


def revoked_events(events: EventLogger) -> list[dict]:
    return events.get_events(event_type=EVENT_TOOL_REVOKED)


def make_plan_channel(specs: list[SubtaskSpec]):
    """构造主管 Agent 调用通道假函数（每次返回计划的深拷贝）。"""

    async def fake_run_supervisor(payload, output_type):
        assert output_type is PlanningOutput
        return PlanningOutput(
            subtasks=[spec.model_copy(deep=True) for spec in specs]
        )

    return fake_run_supervisor


def make_reviewer(
    *,
    plan_passed=True,
    authorization_passed=True,
    on_authorization=None,
):
    """构造评审 Agent 调用通道假函数。

    ``plan_passed`` / ``authorization_passed`` 支持布尔或布尔序列
    （序列按调用顺序弹出，耗尽后沿用最后一个值）。

    Returns:
        ``(fake_review, calls)``：假评审函数与其收到的 ReviewInput 列表。
    """
    plan_script = (
        list(plan_passed)
        if isinstance(plan_passed, (list, tuple))
        else [plan_passed]
    )
    auth_script = (
        list(authorization_passed)
        if isinstance(authorization_passed, (list, tuple))
        else [authorization_passed]
    )
    plan_index = 0
    auth_index = 0
    calls: list[ReviewInput] = []

    def _decision(stage: str, passed: bool) -> ReviewDecision:
        return ReviewDecision(
            stage=stage,
            passed=passed,
            issues=[] if passed else ["模拟评审不通过"],
            summary="模拟评审",
        )

    async def fake_review(review_input: ReviewInput) -> ReviewDecision:
        nonlocal plan_index, auth_index
        calls.append(review_input)
        if review_input.stage == ReviewStage.AUTHORIZATION_REVIEW:
            passed = auth_script[min(auth_index, len(auth_script) - 1)]
            auth_index += 1
            if on_authorization is not None:
                on_authorization(review_input)
            return _decision("authorization_review", passed)
        passed = plan_script[min(plan_index, len(plan_script) - 1)]
        plan_index += 1
        return _decision("plan_review", passed)

    return fake_review, calls


# ===========================================================================
# 一、proposal 与真实签发分离
# ===========================================================================
class TestProposalSeparation:
    def test_proposals_not_registered_in_gateway(self, tmp_path):
        """候选不注册网关：无登记记录、verify 拒绝、计划映射为空。"""
        env = build_env(tmp_path, [make_spec("s1")])

        proposals = env.planning._propose_grants()

        assert len(proposals) == 1
        proposal = proposals[0]
        assert proposal.grant_id.startswith("proposal-")
        assert proposal.expires_at is None
        assert proposal.allowed_tools == ["read_file"]
        # 网关中没有任何可供工具包装器使用的有效签发记录
        assert env.gateway.grants == {}
        assert env.gateway.get_grant(proposal.grant_id) is None
        assert env.gateway.verify_grant(SLOT, "read_file", TASK_ID) is False
        # 计划级映射被清空（被拒计划不得残留旧映射）
        assert env.state.plan_grant_map == {}
        assert env.state.plan_grants == []
        # 候选已登记到状态（供授权评审输入）
        assert env.state.plan_grant_proposals == proposals

    def test_propose_overrides_stale_plan_mapping(self, tmp_path):
        """propose 覆盖上一版计划的映射（真实签发记录不供新计划复用）。"""
        env = build_env(tmp_path, [make_spec("s1")])
        env.planning._issue_grants()
        assert env.state.plan_grant_map  # 旧版本映射（真实签发）

        env.planning._propose_grants()

        assert env.state.plan_grant_map == {}
        assert env.state.plan_grants == []
        assert env.state.plan_grant_proposals

    async def test_authorization_review_input_contains_proposals(
        self, tmp_path
    ):
        """授权评审输入携带候选（评审可见预计授权工具）；计划评审无授权字段。"""
        env = build_env(tmp_path, [make_spec("s1")])
        reviewer, calls = make_reviewer()
        env.deps.call_review_agent = reviewer
        env.planning._propose_grants()

        await env.planning._review_plan()

        auth_calls = [
            c for c in calls if c.stage == ReviewStage.AUTHORIZATION_REVIEW
        ]
        plan_calls = [c for c in calls if c.stage == ReviewStage.PLAN_REVIEW]
        assert len(auth_calls) == 1 and len(plan_calls) == 1
        assert auth_calls[0].tool_grants
        assert all(
            grant.grant_id.startswith("proposal-")
            for grant in auth_calls[0].tool_grants
        )
        assert plan_calls[0].tool_grants is None

    async def test_proposal_credential_blocked_by_tool_wrapper(self, tmp_path):
        """候选凭证被传入工具包装器：运行时校验直接拦截，工具不执行。"""
        env = build_env(tmp_path, [make_spec("s1")])
        proposals = env.planning._propose_grants()
        tool, invoked = make_probe_tool("read_file")
        wrapped = _wrap_tool_with_permission_check(
            tool,
            env.gateway,
            "s1",
            agent_id=SLOT,
            task_id=TASK_ID,
            grant_id=proposals[0].grant_id,
        )

        payload = await invoke(wrapped, {"path": "a.txt"})

        assert payload["status"] == "error"
        assert "拦截" in payload["error"]
        assert invoked == []


# ===========================================================================
# 二、两路评审通过后才真实签发
# ===========================================================================
class TestPlanReviewIssuanceFlow:
    async def test_rejected_content_review_issues_no_real_grant(self, tmp_path):
        """内容评审失败：网关无任何真实授权（被拒计划无签发记录）。"""
        env = build_env(
            tmp_path,
            [make_spec("s1")],
            settings=Settings(_env_file=None, max_replan_attempts=0),
        )
        reviewer, _ = make_reviewer(plan_passed=False)
        env.deps.run_supervisor = make_plan_channel([make_spec("s1")])
        env.deps.call_review_agent = reviewer

        result = await env.planning._plan_and_review()

        assert result is False
        assert granted_events(env.events) == []
        assert env.gateway.grants == {}
        assert env.gateway.verify_grant(SLOT, "read_file", TASK_ID) is False
        # 候选存在（评审流程已发生），但没有真实签发
        assert env.state.plan_grant_proposals
        assert all(
            proposal.grant_id.startswith("proposal-")
            for proposal in env.state.plan_grant_proposals
        )

    async def test_rejected_authorization_review_issues_no_real_grant(
        self, tmp_path
    ):
        """授权评审失败：内容评审即使通过也绝不真实签发。"""
        env = build_env(
            tmp_path,
            [make_spec("s1")],
            settings=Settings(_env_file=None, max_replan_attempts=0),
        )
        reviewer, calls = make_reviewer(
            plan_passed=True, authorization_passed=False
        )
        env.deps.run_supervisor = make_plan_channel([make_spec("s1")])
        env.deps.call_review_agent = reviewer

        result = await env.planning._plan_and_review()

        assert result is False
        assert granted_events(env.events) == []
        assert env.gateway.grants == {}
        assert env.gateway.verify_grant(SLOT, "read_file", TASK_ID) is False
        auth_calls = [
            c for c in calls if c.stage == ReviewStage.AUTHORIZATION_REVIEW
        ]
        assert auth_calls and all(
            grant.grant_id.startswith("proposal-")
            for call in auth_calls
            for grant in call.tool_grants
        )

    async def test_grants_issued_only_after_both_reviews_pass(self, tmp_path):
        """两路评审均通过后才签发：授权评审时刻网关无真实签发（时序证据）。"""
        env = build_env(tmp_path, [make_spec("s1")])
        snapshots: list[int] = []

        def on_authorization(review_input: ReviewInput) -> None:
            snapshots.append(len(granted_events(env.events)))

        reviewer, _ = make_reviewer(on_authorization=on_authorization)
        env.deps.run_supervisor = make_plan_channel([make_spec("s1")])
        env.deps.call_review_agent = reviewer

        result = await env.planning._plan_and_review()

        assert result is True
        # 授权评审发生的时刻：网关尚无任何 tool_granted 事件
        assert snapshots == [0]
        # 通过后：真实签发（grant_id 非候选前缀、expires_at 已设置、verify 通过）
        assert len(granted_events(env.events)) == 1
        real = env.state.plan_grant_map["s1"]
        assert not real.grant_id.startswith("proposal-")
        assert real.expires_at is not None
        assert env.gateway.verify_grant(SLOT, "read_file", TASK_ID) is True

    async def test_authorization_reject_then_pass_issues_once(self, tmp_path):
        """授权评审先拒后过：每次评审时刻均无真实签发，通过后恰好签发一次。"""
        env = build_env(
            tmp_path,
            [make_spec("s1")],
            settings=Settings(_env_file=None, max_replan_attempts=1),
        )
        snapshots: list[int] = []

        def on_authorization(review_input: ReviewInput) -> None:
            snapshots.append(len(granted_events(env.events)))

        reviewer, _ = make_reviewer(
            authorization_passed=[False, True],
            on_authorization=on_authorization,
        )
        env.deps.run_supervisor = make_plan_channel([make_spec("s1")])
        env.deps.call_review_agent = reviewer

        result = await env.planning._plan_and_review()

        assert result is True
        assert snapshots == [0, 0]  # 两次授权评审时刻均无真实签发
        assert len(granted_events(env.events)) == 1
        assert env.gateway.verify_grant(SLOT, "read_file", TASK_ID) is True


# ===========================================================================
# 三、统一幂等清理
# ===========================================================================
class TestGatewayTaskRevocation:
    def test_revoke_grants_for_task_scoped_and_reasoned(self, tmp_path):
        """按任务撤销：只影响目标任务，reason 写入审计事件，重复幂等。"""
        env = build_env(tmp_path, [make_spec("s1")])
        g1 = env.gateway.issue_grant(TASK_ID, SLOT, ["read_file"], SUPERVISOR)
        g2 = env.gateway.issue_grant(
            TASK_ID, f"{TASK_ID}:s2:grant-slot", ["read_file"], SUPERVISOR
        )
        other = env.gateway.issue_grant(
            "t2", "t2:s1:grant-slot", ["read_file"], SUPERVISOR
        )

        revoked = env.gateway.revoke_grants_for_task(
            TASK_ID, reason="lifecycle_cleanup:test"
        )

        assert sorted(revoked) == sorted([g1.grant_id, g2.grant_id])
        assert other.revoked is False
        reasons = {record["payload"]["reason"] for record in revoked_events(env.events)}
        assert reasons == {"lifecycle_cleanup:test"}
        # 重复调用：空操作、无新事件
        assert (
            env.gateway.revoke_grants_for_task(
                TASK_ID, reason="lifecycle_cleanup:test"
            )
            == []
        )
        assert len(revoked_events(env.events)) == 2


class TestRevokeActiveGrantsCleanup:
    def test_cleanup_revokes_all_and_reason_visible(self, tmp_path):
        """统一清理撤销本任务全部活跃授权，reason 可在审计事件中确认。"""
        env = build_env(tmp_path, [make_spec("s1"), make_spec("s2")])
        grants = env.planning._issue_grants()
        assert len(grants) == 2

        revoked = env.planning._revoke_active_grants(
            reason="lifecycle_cleanup:unit_test"
        )

        assert sorted(revoked) == sorted(g.grant_id for g in grants)
        records = revoked_events(env.events)
        assert len(records) == 2
        assert all(
            record["payload"]["reason"] == "lifecycle_cleanup:unit_test"
            for record in records
        )
        assert env.gateway.verify_grant(SLOT, "read_file", TASK_ID) is False

    def test_cleanup_idempotent_no_duplicate_events(self, tmp_path):
        """重复清理：第二次为空操作，不产生重复事件、不抛错。"""
        env = build_env(tmp_path, [make_spec("s1")])
        env.planning._issue_grants()
        first = env.planning._revoke_active_grants(
            reason="lifecycle_cleanup:first"
        )
        assert len(first) == 1

        second = env.planning._revoke_active_grants(
            reason="lifecycle_cleanup:second"
        )

        assert second == []
        assert len(revoked_events(env.events)) == 1

    def test_cleanup_noop_without_task_request(self, tmp_path):
        """任务尚未接收（task_request 为空）：空操作不抛错。"""
        env = build_env(tmp_path, [make_spec("s1")])
        env.state.task_request = None

        assert env.planning._revoke_active_grants(reason="lifecycle_cleanup:x") == []

    def test_cleanup_swallows_gateway_errors(self, tmp_path, monkeypatch):
        """网关异常时清理内部消化（finally 路径不得掩盖原始异常/返回值）。"""
        env = build_env(tmp_path, [make_spec("s1")])

        def boom(task_id, *, reason="revoked"):
            raise RuntimeError("gateway down")

        monkeypatch.setattr(env.gateway, "revoke_grants_for_task", boom)

        assert env.planning._revoke_active_grants(reason="lifecycle_cleanup:x") == []


# ===========================================================================
# 四、重规划生命周期
# ===========================================================================
class TestReplanLifecycle:
    async def test_replan_cleans_old_grants_before_new_issue(self, tmp_path):
        """进入重规划先清理旧授权；通过后签发新版本授权（时序正确）。"""
        env = build_env(tmp_path, [make_spec("s1")])
        old_grant = env.planning._issue_grants()[0]
        env.state.plan_version = 1  # 模拟已采纳第一版计划

        reviewer, _ = make_reviewer()
        env.deps.run_supervisor = make_plan_channel([make_spec("s1")])
        env.deps.call_review_agent = reviewer

        result = await env.planning._plan_and_review()

        assert result is True
        assert env.state.plan_version == 2
        # 旧授权被清理（原因明确）
        assert old_grant.revoked is True
        records = revoked_events(env.events)
        assert records[-1]["payload"]["reason"] == "lifecycle_cleanup:replanning"
        # 新授权已签发且有效（不同 grant_id）
        new_grant = env.state.plan_grant_map["s1"]
        assert new_grant.grant_id != old_grant.grant_id
        assert env.gateway.verify_grant(SLOT, "read_file", TASK_ID) is True
        assert len(granted_events(env.events)) == 2
        # 时序：清理事件先于新签发事件
        log = env.context.event_log
        last_revoked = max(
            index
            for index, record in enumerate(log)
            if record["event_type"] == EVENT_TOOL_REVOKED
        )
        last_granted = max(
            index
            for index, record in enumerate(log)
            if record["event_type"] == EVENT_TOOL_GRANTED
        )
        assert last_revoked < last_granted

    async def test_rejected_replan_leaves_no_active_grant(self, tmp_path):
        """重规划被拒：旧授权已清理、无新签发、无任何活跃授权残留。"""
        env = build_env(
            tmp_path,
            [make_spec("s1")],
            settings=Settings(_env_file=None, max_replan_attempts=0),
        )
        old_grant = env.planning._issue_grants()[0]
        env.state.plan_version = 1

        reviewer, _ = make_reviewer(plan_passed=False)
        env.deps.run_supervisor = make_plan_channel([make_spec("s1")])
        env.deps.call_review_agent = reviewer

        result = await env.planning._plan_and_review()

        assert result is False
        assert old_grant.revoked is True
        assert env.state.plan_grant_map == {}
        assert len(granted_events(env.events)) == 1  # 仅旧签发，无新签发
        assert env.gateway.verify_grant(SLOT, "read_file", TASK_ID) is False


# ===========================================================================
# 五、动态工具补发（执行阶段创建成功后再签发）
# ===========================================================================
class TestDynamicToolReissue:
    async def test_gap_tool_reissued_only_after_creation(
        self, tmp_path, monkeypatch
    ):
        """未注册工具不进候选；执行阶段创建成功后按最小集合补发真实授权。"""
        env = build_env(tmp_path, [make_spec("s1", allowed_tools=("dyn_tool",))])

        # 计划阶段：动态工具未就绪 → 候选为空（仅作缺口提醒）
        proposals = env.planning._propose_grants()
        assert proposals == []
        assert env.state.plan_grant_map == {}

        # 干净环境：候选清单路径指向临时目录（不存在）
        monkeypatch.setattr(
            code_agent_agent_mod,
            "GENERATED_REGISTRY_PATH",
            tmp_path / "generated_tools" / "registry.json",
        )

        async def fake_create(tool_name: str) -> None:
            env.tool_registry.register_generated_tool(
                GeneratedToolManifest(
                    tool_id="tool-dyn",
                    tool_name=tool_name,
                    entry_file=f"{tool_name}.py",
                    created_by_task=TASK_ID,
                    review_status="approved",
                    verified=True,
                )
            )
            return None

        monkeypatch.setattr(
            env.execution, "_create_and_approve_tool", fake_create
        )

        # 执行阶段：缺口解决后补发真实授权
        failures = await env.execution._resolve_tool_gaps(
            [make_spec("s1", allowed_tools=("dyn_tool",))]
        )

        assert failures == {}
        assert env.tool_registry.is_approved("dyn_tool") is True
        grant = env.state.plan_grant_map["s1"]
        assert grant.allowed_tools == ["dyn_tool"]  # 最小工具集合
        assert not grant.grant_id.startswith("proposal-")
        assert env.gateway.verify_grant(SLOT, "dyn_tool", TASK_ID) is True
        assert len(granted_events(env.events)) == 1


# ===========================================================================
# 六、执行窗口回收理由与返工重签发（回归）
# ===========================================================================
class TestExecutionWindowLifecycle:
    async def test_batch_revoke_reason_and_rework_reissue(self, tmp_path):
        """执行窗口结束回收（reason=execution_window_closed）；返工窗口重签发。"""
        env = build_env(tmp_path, [make_spec("s1")])
        first_grant = env.planning._issue_grants()[0]

        batch = await env.execution._execute_subtasks([make_spec("s1")])

        assert batch["s1"].success is True
        assert first_grant.revoked is True
        assert revoked_events(env.events)[-1]["payload"]["reason"] == (
            "execution_window_closed"
        )
        assert env.gateway.verify_grant(SLOT, "read_file", TASK_ID) is False

        # 返工窗口：授权在窗口开始前重签发（执行期间有效）
        batch2 = await env.execution._execute_subtasks([make_spec("s1")])

        assert batch2["s1"].success is True
        second_grant = env.state.plan_grant_map["s1"]
        assert second_grant.grant_id != first_grant.grant_id
        assert second_grant.revoked is True  # 窗口结束后回收
        assert len(granted_events(env.events)) == 2


# ===========================================================================
# 七、安全回归：伪造 / 过期 / 已撤销授权持续被拒绝
# ===========================================================================
class TestLifecycleSecurityRegression:
    def test_proposal_credential_never_verifies(self, tmp_path):
        """候选凭证（伪造形态）永远无法通过网关校验。"""
        env = build_env(tmp_path, [make_spec("s1")])
        proposals = env.planning._propose_grants()

        for proposal in proposals:
            assert (
                env.gateway.verify_grant(
                    proposal.agent_id, "read_file", proposal.task_id
                )
                is False
            )

    def test_expired_real_grant_rejected(self, tmp_path):
        """真实签发后过期的授权立即失效（第二道防线）。"""
        env = build_env(tmp_path, [make_spec("s1")])
        grant = env.planning._issue_grants()[0]
        grant.expires_at = _utc_now() - timedelta(seconds=1)

        assert env.gateway.verify_grant(SLOT, "read_file", TASK_ID) is False

    def test_revoked_grant_rejected_after_cleanup(self, tmp_path):
        """生命周期清理后授权立即失效。"""
        env = build_env(tmp_path, [make_spec("s1")])
        env.planning._issue_grants()

        env.planning._revoke_active_grants(reason="lifecycle_cleanup:regression")

        assert env.gateway.verify_grant(SLOT, "read_file", TASK_ID) is False


# ===========================================================================
# 工具包装器探针辅助（与授权生命周期回归共用）
# ===========================================================================
def make_ctx(tool_name: str, arguments: dict) -> ToolContext:
    return ToolContext(
        context=None,
        tool_name=tool_name,
        tool_call_id=f"call-{tool_name}-1",
        tool_arguments=json.dumps(arguments, ensure_ascii=False),
    )


def make_probe_tool(name: str = "read_file") -> tuple[FunctionTool, list[str]]:
    """构造带调用记录的探针 FunctionTool。"""
    invoked: list[str] = []

    async def _invoke(ctx, args_json: str) -> str:
        invoked.append(args_json)
        return json.dumps({"status": "ok", "tool": name}, ensure_ascii=False)

    tool = FunctionTool(
        name=name,
        description=f"探针工具 {name}",
        params_json_schema={
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
        on_invoke_tool=_invoke,
    )
    return tool, invoked


async def invoke(tool: FunctionTool, arguments: dict) -> dict:
    raw = await tool.on_invoke_tool(
        make_ctx(tool.name, arguments), json.dumps(arguments, ensure_ascii=False)
    )
    return json.loads(raw)

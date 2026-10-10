"""单元测试：授权生命周期与防伪加固（Task #26，A 组）。

覆盖三项安全加固：

- Fix A1 授权生命周期闭环：
  * 子任务批次执行完成后（成功 / 失败 / 调度异常路径）槽位授权自动撤销
    （``_revoke_batch_grants``）；
  * 返工 / 重入批次在窗口开始前重新签发已回收或已失效的授权
    （``_ensure_batch_grants_active``），保证执行窗口内授权新鲜有效；
  * 规划阶段 ``_issue_grants`` 签发的授权固定携带 ``expires_at``
    （= 签发时刻 + ``max_pipeline_timeout``）作为第二道防线。
- Fix A2 grant_id 绑定校验（``_wrap_tool_with_permission_check``）：
  * 正确 grant_id 放行；未登记 / 主体不符 / 不在绑定凭证白名单的
    grant_id 在三元组校验通过后仍被拦截；
  * create_subagent 集成路径：凭证在传递链路中被篡改（grant_id /
    agent_id）时运行时拦截。
- Fix A3 权限审计：
  * ``verify_grant`` 拒绝时记录 ``tool_denied`` 审计事件并细分拒绝原因
    （参数非法 / 无匹配授权 / 已撤销 / 已过期 / 工具不在白名单）；
  * 授权通过时不记录 ``tool_denied``。
"""

from __future__ import annotations

import json
from datetime import timedelta
from types import SimpleNamespace

import pytest
from agents import FunctionTool
from agents.tool_context import ToolContext

from so_agent.config import Settings
from so_agent.context import ProjectContext
from so_agent.execution_coordinator import ExecutionCoordinator
from so_agent.models import (
    ExecutionResult,
    SubtaskSpec,
    TaskRequest,
    ToolGrant,
    _utc_now,
)
from so_agent.planning_coordinator import PlanningCoordinator
from so_agent.runtime.events import (
    EVENT_TOOL_DENIED,
    EVENT_TOOL_GRANTED,
    EventLogger,
)
from so_agent.runtime.permissions import PermissionGateway
from so_agent.runtime.registry import AgentRegistry, DEFAULT_SUPERVISOR_ID
from so_agent.runtime.scheduler import SchedulerError
from so_agent.runtime.supervisor_router import SupervisorRouter
from so_agent.runtime.tool_registry import ToolRegistry
from so_agent.state_machine import WorkflowState
from so_agent.tool_packages.subagent_creator import agent as creator_mod
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
    """鸭子类型调度器：记录调用、按 builder 产出批次结果。"""

    def __init__(self, builder=None, error: Exception | None = None):
        self._builder = builder
        self._error = error
        self.calls: list[tuple[str, list[SubtaskSpec]]] = []
        self.replacement_requests: list = []
        self.decomposition_issues: list = []

    async def schedule(self, task_id: str, specs: list[SubtaskSpec]):
        self.calls.append((task_id, list(specs)))
        if self._error is not None:
            raise self._error
        if self._builder is not None:
            return self._builder(specs)
        return {
            spec.subtask_id: ExecutionResult(
                subtask_id=spec.subtask_id, success=True, output="子任务完成产出内容"
            )
            for spec in specs
        }


def build_env(tmp_path, specs: list[SubtaskSpec], *, scheduler=None, settings=None):
    """装配共享状态 + 规划/执行协调器（真实依赖，不启动工作流）。"""
    resolved_settings = settings or Settings(_env_file=None)
    context = ProjectContext(sandbox_dir=tmp_path, project_name="lifecycle-test")
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
        scheduler=scheduler if scheduler is not None else FakeScheduler(),
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


def denied_events(events: EventLogger) -> list[dict]:
    return events.get_events(event_type=EVENT_TOOL_DENIED)


# ===========================================================================
# 一、Fix A1：批次执行完成后自动撤销槽位授权
# ===========================================================================
class TestBatchGrantRevocation:
    async def test_grant_revoked_after_successful_batch(self, tmp_path):
        """成功路径：批次完成后槽位授权被自动撤销，verify_grant 立即失效。"""
        env = build_env(tmp_path, [make_spec("s1")])
        grant = env.planning._issue_grants()[0]
        assert env.gateway.verify_grant(SLOT, "read_file", TASK_ID) is True

        batch = await env.execution._execute_subtasks([make_spec("s1")])

        assert batch["s1"].success is True
        assert env.gateway.get_grant(grant.grant_id).revoked is True
        assert env.gateway.verify_grant(SLOT, "read_file", TASK_ID) is False

    async def test_grant_revoked_after_failed_batch(self, tmp_path):
        """失败路径：失败批次完成后同样回收授权（不残留跨窗口）。"""

        def failing_builder(specs):
            return {
                spec.subtask_id: ExecutionResult(
                    subtask_id=spec.subtask_id, success=False, error="模拟失败"
                )
                for spec in specs
            }

        scheduler = FakeScheduler(builder=failing_builder)
        env = build_env(tmp_path, [make_spec("s1")], scheduler=scheduler)
        grant = env.planning._issue_grants()[0]

        batch = await env.execution._execute_subtasks([make_spec("s1")])

        assert batch["s1"].success is False
        assert env.gateway.get_grant(grant.grant_id).revoked is True
        assert env.gateway.verify_grant(SLOT, "read_file", TASK_ID) is False

    async def test_grant_revoked_on_scheduler_error(self, tmp_path):
        """调度异常路径：调度器抛错时批次转全失败，授权仍被统一回收。"""
        scheduler = FakeScheduler(error=SchedulerError("调度器拒绝该批次"))
        env = build_env(tmp_path, [make_spec("s1")], scheduler=scheduler)
        grant = env.planning._issue_grants()[0]

        batch = await env.execution._execute_subtasks([make_spec("s1")])

        assert batch["s1"].success is False
        assert "调度失败" in batch["s1"].error
        assert env.gateway.get_grant(grant.grant_id).revoked is True

    async def test_rework_batch_reissues_revoked_grant(self, tmp_path):
        """返工重入：上一窗口回收的授权在下一窗口开始前自动重签发并有效。

        执行窗口内（调度器回调时刻）授权必须有效——包括返工批次。
        """
        observed: list[bool] = []

        def probing_builder(specs):
            for spec in specs:
                current = env.state.plan_grant_map.get(spec.subtask_id)
                observed.append(
                    current is not None
                    and env.gateway.verify_grant(
                        current.agent_id, "read_file", TASK_ID
                    )
                )
            return {
                spec.subtask_id: ExecutionResult(
                    subtask_id=spec.subtask_id, success=True, output="完成产出内容"
                )
                for spec in specs
            }

        scheduler = FakeScheduler(builder=probing_builder)
        env = build_env(tmp_path, [make_spec("s1")], scheduler=scheduler)
        first_grant = env.planning._issue_grants()[0]

        await env.execution._execute_subtasks([make_spec("s1")])  # 第一窗口
        assert first_grant.revoked is True

        await env.execution._execute_subtasks([make_spec("s1")])  # 返工窗口

        # 两个执行窗口内授权均有效；返工窗口的授权来自重签发
        assert observed == [True, True]
        second_grant = env.state.plan_grant_map["s1"]
        assert second_grant.grant_id != first_grant.grant_id
        assert second_grant.revoked is True  # 返工窗口结束后同样被回收
        assert len(granted_events(env.events)) == 2  # 两次签发事件

    async def test_expired_grant_reissued_at_window_start(self, tmp_path):
        """已在窗口开始前过期的授权同样触发重签发（第二道防线兜底）。"""
        env = build_env(tmp_path, [make_spec("s1")])
        grant = env.planning._issue_grants()[0]
        grant.expires_at = _utc_now() - timedelta(seconds=1)  # 模拟已过期

        await env.execution._execute_subtasks([make_spec("s1")])

        renewed = env.state.plan_grant_map["s1"]
        assert renewed.grant_id != grant.grant_id
        assert renewed.expires_at is not None
        assert renewed.expires_at > _utc_now()
        assert len(granted_events(env.events)) == 2

    async def test_fresh_grant_not_reissued_in_first_window(self, tmp_path):
        """首窗口授权新鲜有效时不重复签发（避免无效授权噪声）。"""
        env = build_env(tmp_path, [make_spec("s1")])
        env.planning._issue_grants()

        await env.execution._execute_subtasks([make_spec("s1")])

        assert len(granted_events(env.events)) == 1

    async def test_no_grant_subtask_untouched(self, tmp_path):
        """无工具子任务（不签发授权）不因回收逻辑报错。"""
        env = build_env(tmp_path, [make_spec("s1", allowed_tools=())])
        grants = env.planning._issue_grants()
        assert grants == []

        batch = await env.execution._execute_subtasks(
            [make_spec("s1", allowed_tools=())]
        )

        assert batch["s1"].success is True
        assert env.gateway.grants == {}


# ===========================================================================
# 二、Fix A1：规划阶段签发授权携带 expires_at（第二道防线）
# ===========================================================================
class TestIssueGrantsExpiry:
    def test_expires_at_set_from_pipeline_timeout(self, tmp_path):
        """expires_at = 签发时刻 + max_pipeline_timeout，且为 UTC aware。"""
        settings = Settings(_env_file=None, max_pipeline_timeout=600.0)
        env = build_env(tmp_path, [make_spec("s1")], settings=settings)

        grants = env.planning._issue_grants()

        assert len(grants) == 1
        grant = grants[0]
        assert grant.expires_at is not None
        assert grant.expires_at.tzinfo is not None
        remaining = grant.expires_at - _utc_now()
        assert 0 < remaining.total_seconds() <= 600.0

    def test_all_grants_carry_same_expiry_and_map(self, tmp_path):
        """同批签发共享同一过期时刻，且 plan_grant_map 中的授权同样携带。"""
        env = build_env(
            tmp_path,
            [make_spec("s1"), make_spec("s2"), make_spec("s3")],
        )

        grants = env.planning._issue_grants()

        assert len(grants) == 3
        assert len({grant.expires_at for grant in grants}) == 1
        for spec in env.state.subtasks:
            mapped = env.state.plan_grant_map[spec.subtask_id]
            assert mapped.expires_at == grants[0].expires_at

    def test_fresh_expiry_allows_verification(self, tmp_path):
        """刚签发的授权在过期时刻前可通过校验（未误伤正常窗口）。"""
        env = build_env(tmp_path, [make_spec("s1")])
        env.planning._issue_grants()

        assert env.gateway.verify_grant(SLOT, "read_file", TASK_ID) is True


# ===========================================================================
# 三、Fix A2：grant_id 绑定校验（包裹层）
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


@pytest.fixture
def gateway(tmp_path):
    context = ProjectContext(sandbox_dir=tmp_path, project_name="binding-test")
    events = EventLogger(context=context)
    return PermissionGateway(context=context, settings=Settings(), events=events)


class TestGrantIdBinding:
    def _wrap(
        self,
        tool: FunctionTool,
        gateway: PermissionGateway,
        *,
        grant_id: str | None,
        agent_id: str = SLOT,
        task_id: str = TASK_ID,
    ) -> FunctionTool:
        return _wrap_tool_with_permission_check(
            tool,
            gateway,
            "s1",
            agent_id=agent_id,
            task_id=task_id,
            grant_id=grant_id,
        )

    async def test_correct_grant_id_passes(self, gateway):
        """正确 grant_id：三元组与绑定记录一致 → 放行真实工具。"""
        grant = gateway.issue_grant(TASK_ID, SLOT, ["read_file"], SUPERVISOR)
        tool, invoked = make_probe_tool()
        wrapped = self._wrap(tool, gateway, grant_id=grant.grant_id)

        payload = await invoke(wrapped, {"path": "a.txt"})
        assert payload["status"] == "ok"
        assert len(invoked) == 1

    async def test_unknown_grant_id_blocked(self, gateway):
        """未登记的 grant_id：三元组校验可通过（网关另有有效授权），
        但绑定记录不存在 → 被绑定校验拦截。"""
        gateway.issue_grant(TASK_ID, SLOT, ["read_file"], SUPERVISOR)
        tool, invoked = make_probe_tool()
        wrapped = self._wrap(tool, gateway, grant_id="grant-forged")

        payload = await invoke(wrapped, {"path": "a.txt"})
        assert payload["status"] == "error"
        assert "grant_id 绑定校验拦截" in payload["error"]
        assert invoked == []

    async def test_grant_id_of_other_slot_blocked(self, gateway):
        """主体不符：绑定记录属于其它槽位 → 拦截。"""
        gateway.issue_grant(TASK_ID, SLOT, ["read_file"], SUPERVISOR)  # 使三元组通过
        other = gateway.issue_grant(
            TASK_ID, f"{TASK_ID}:s2:grant-slot", ["read_file"], SUPERVISOR
        )
        tool, invoked = make_probe_tool()
        wrapped = self._wrap(tool, gateway, grant_id=other.grant_id)

        payload = await invoke(wrapped, {"path": "a.txt"})
        assert payload["status"] == "error"
        assert "grant_id 绑定校验拦截" in payload["error"]
        assert invoked == []

    async def test_revoked_bound_grant_blocked_even_if_triplet_passes(
        self, gateway
    ):
        """绑定记录已撤销：另一有效授权使三元组通过，绑定校验仍拦截。"""
        revoked = gateway.issue_grant(TASK_ID, SLOT, ["read_file"], SUPERVISOR)
        gateway.issue_grant(TASK_ID, SLOT, ["read_file"], SUPERVISOR)  # 仍有效
        gateway.revoke_grant(revoked.grant_id)
        tool, invoked = make_probe_tool()
        wrapped = self._wrap(tool, gateway, grant_id=revoked.grant_id)

        payload = await invoke(wrapped, {"path": "a.txt"})
        assert payload["status"] == "error"
        assert "grant_id 绑定校验拦截" in payload["error"]
        assert invoked == []

    async def test_tool_not_in_bound_grant_whitelist_blocked(self, gateway):
        """工具越权：绑定记录白名单不含本次工具 → 拦截。"""
        gateway.issue_grant(TASK_ID, SLOT, ["read_file"], SUPERVISOR)  # 使三元组通过
        narrow = gateway.issue_grant(TASK_ID, SLOT, ["write_file"], SUPERVISOR)
        tool, invoked = make_probe_tool("read_file")
        wrapped = self._wrap(tool, gateway, grant_id=narrow.grant_id)

        payload = await invoke(wrapped, {"path": "a.txt"})
        assert payload["status"] == "error"
        assert "grant_id 绑定校验拦截" in payload["error"]
        assert invoked == []

    async def test_none_grant_id_keeps_legacy_behavior(self, gateway):
        """grant_id 为 None：保持旧行为（仅三元组校验，向后兼容）。"""
        gateway.issue_grant(TASK_ID, SLOT, ["read_file"], SUPERVISOR)
        tool, invoked = make_probe_tool()
        wrapped = self._wrap(tool, gateway, grant_id=None)

        payload = await invoke(wrapped, {"path": "a.txt"})
        assert payload["status"] == "ok"
        assert len(invoked) == 1


# ===========================================================================
# 四、Fix A2：create_subagent 集成路径（网关登记凭证 + 篡改拦截）
# ===========================================================================
class FakeRunResult:
    def __init__(self, final_output):
        self.final_output = final_output


class FakeStreamedResult:
    """假 RunResultStreaming：首次取流式事件时求值协程工厂并回填输出。

    ``execute_subagent`` 经 ``run_agent_streamed`` 驱动流式执行；正常
    路径下取到假事件后迭代结束，返回对象（final_output 已就绪）。
    """

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


def install_fake_runner(monkeypatch, captured_agents: list):
    """把 subagent_creator 模块内的 Runner 替换为记录 agent 实例的假实现。

    ``run_streamed`` 为 ``execute_subagent`` 的流式执行入口，与
    ``run`` 共用同一记录逻辑。
    """

    class FakeRunner:
        @classmethod
        async def run(cls, starting_agent, input=None, *, max_turns=None, **kwargs):
            captured_agents.append(starting_agent)
            return FakeRunResult(
                ExecutionResult(subtask_id="ignored", success=True, output="done")
            )

        @classmethod
        def run_streamed(cls, starting_agent, input=None, *, max_turns=None, **kwargs):
            return FakeStreamedResult(
                lambda: cls.run(
                    starting_agent, input=input, max_turns=max_turns, **kwargs
                )
            )

    monkeypatch.setattr(creator_mod, "Runner", FakeRunner)
    return FakeRunner


class TestCreateSubagentGrantIdBinding:
    @pytest.fixture
    def creator_env(self, tmp_path):
        context = ProjectContext(sandbox_dir=tmp_path, project_name="binding-int")
        settings = Settings(_env_file=None, max_dynamic_agents=8)
        events = EventLogger(context=context)
        registry = AgentRegistry(context=context, settings=settings)
        gateway = PermissionGateway(context=context, settings=settings, events=events)
        tools = creator_mod.build_creator_tools(
            context,
            settings=settings,
            registry=registry,
            subagent_model="test-model",
            gateway=gateway,
        )
        (tmp_path / "hello.txt").write_text("binding test", encoding="utf-8")
        return {
            "context": context,
            "tools": tools,
            "registry": registry,
            "gateway": gateway,
        }

    def _spec(self) -> SubtaskSpec:
        return SubtaskSpec(
            subtask_id="s1",
            title="演示子任务",
            instructions="读取沙箱文件",
            role="code",
            allowed_tools=["read_file"],
        )

    async def _create(self, tools, grant: ToolGrant) -> dict:
        raw = await tools["create_subagent"].on_invoke_tool(
            make_ctx("create_subagent", {}),
            json.dumps(
                {
                    "subtask_spec_json": self._spec().model_dump_json(),
                    "tool_grant_json": grant.model_dump_json(),
                }
            ),
        )
        return json.loads(raw)

    async def _execute(self, tools, agent_id: str) -> dict:
        raw = await tools["execute_subagent"].on_invoke_tool(
            make_ctx("execute_subagent", {}),
            json.dumps({"agent_id": agent_id, "input_data": "{}"}),
        )
        return json.loads(raw)

    async def test_registered_grant_id_passes(self, creator_env, monkeypatch):
        """网关登记的凭证：grant_id 绑定校验通过 → 真实 read_file 放行。"""
        captured: list = []
        install_fake_runner(monkeypatch, captured)
        grant = creator_env["gateway"].issue_grant(
            TASK_ID, SLOT, ["read_file"], SUPERVISOR
        )

        payload = await self._create(creator_env["tools"], grant)
        await self._execute(creator_env["tools"], payload["agent_id"])

        result = await invoke(captured[0].tools[0], {"path": "hello.txt"})
        assert result["status"] == "ok"
        assert result["content"] == "binding test"

    async def test_forged_grant_id_blocked_at_runtime(self, creator_env, monkeypatch):
        """grant_id 被篡改（未登记）：构造时注入放行，运行时不执行。"""
        captured: list = []
        install_fake_runner(monkeypatch, captured)
        grant = creator_env["gateway"].issue_grant(
            TASK_ID, SLOT, ["read_file"], SUPERVISOR
        )
        tampered = grant.model_copy(update={"grant_id": "grant-forged"})

        payload = await self._create(creator_env["tools"], tampered)
        assert payload["injected_tools"] == ["read_file"]
        await self._execute(creator_env["tools"], payload["agent_id"])

        result = await invoke(captured[0].tools[0], {"path": "hello.txt"})
        assert result["status"] == "error"
        assert "grant_id 绑定校验拦截" in result["error"]

    async def test_tampered_agent_id_blocked_at_runtime(self, creator_env, monkeypatch):
        """agent_id 被篡改（凭证主体非网关登记）：三元组校验即拦截。"""
        captured: list = []
        install_fake_runner(monkeypatch, captured)
        grant = creator_env["gateway"].issue_grant(
            TASK_ID, SLOT, ["read_file"], SUPERVISOR
        )
        tampered = grant.model_copy(
            update={"agent_id": f"{TASK_ID}:s999:grant-slot"}
        )

        payload = await self._create(creator_env["tools"], tampered)
        await self._execute(creator_env["tools"], payload["agent_id"])

        result = await invoke(captured[0].tools[0], {"path": "hello.txt"})
        assert result["status"] == "error"
        assert "运行时权限校验拦截" in result["error"]

    async def test_revoked_after_creation_blocked(self, creator_env, monkeypatch):
        """创建后撤销：绑定校验与三元组校验同时失效 → 拦截。"""
        captured: list = []
        install_fake_runner(monkeypatch, captured)
        grant = creator_env["gateway"].issue_grant(
            TASK_ID, SLOT, ["read_file"], SUPERVISOR
        )

        payload = await self._create(creator_env["tools"], grant)
        await self._execute(creator_env["tools"], payload["agent_id"])
        creator_env["gateway"].revoke_grant(grant.grant_id)

        result = await invoke(captured[0].tools[0], {"path": "hello.txt"})
        assert result["status"] == "error"


# ===========================================================================
# 五、Fix A3：tool_denied 审计事件（细分拒绝原因）
# ===========================================================================
class TestToolDeniedAudit:
    def test_no_matching_grant_denied_event(self, gateway, tmp_path):
        """无匹配授权：记录 tool_denied，reason 细分。"""
        assert gateway.verify_grant(SLOT, "read_file", TASK_ID) is False
        events = denied_events(gateway._events)
        assert len(events) == 1
        payload = events[0]["payload"]
        assert payload["agent_id"] == SLOT
        assert payload["tool_name"] == "read_file"
        assert payload["task_id"] == TASK_ID
        assert payload["reason"] == "无匹配授权"
        assert events[0]["task_id"] == TASK_ID  # 索引字段
        assert events[0]["agent_id"] == SLOT

    def test_revoked_grant_denied_event(self, gateway):
        """已撤销：reason=授权已撤销。"""
        grant = gateway.issue_grant(TASK_ID, SLOT, ["read_file"], SUPERVISOR)
        gateway.revoke_grant(grant.grant_id)

        assert gateway.verify_grant(SLOT, "read_file", TASK_ID) is False
        events = denied_events(gateway._events)
        assert len(events) == 1
        assert events[0]["payload"]["reason"] == "授权已撤销"

    def test_expired_grant_denied_event(self, gateway):
        """已过期：reason=授权已过期。"""
        gateway.issue_grant(
            TASK_ID,
            SLOT,
            ["read_file"],
            SUPERVISOR,
            expires_at=_utc_now() - timedelta(seconds=1),
        )

        assert gateway.verify_grant(SLOT, "read_file", TASK_ID) is False
        events = denied_events(gateway._events)
        assert len(events) == 1
        assert events[0]["payload"]["reason"] == "授权已过期"

    def test_tool_not_in_whitelist_denied_event(self, gateway):
        """工具不在白名单：reason=工具不在授权白名单。"""
        gateway.issue_grant(TASK_ID, SLOT, ["write_file"], SUPERVISOR)

        assert gateway.verify_grant(SLOT, "read_file", TASK_ID) is False
        events = denied_events(gateway._events)
        assert len(events) == 1
        assert events[0]["payload"]["reason"] == "工具不在授权白名单"

    def test_invalid_params_denied_event(self, gateway):
        """参数非法：reason 细分且不影响后续合法调用。"""
        assert gateway.verify_grant("", "read_file", TASK_ID) is False
        assert gateway.verify_grant(SLOT, "read_file", "") is False
        events = denied_events(gateway._events)
        assert len(events) == 2
        assert all("参数非法" in event["payload"]["reason"] for event in events)

    def test_success_records_no_denied_event(self, gateway):
        """校验通过：不记录 tool_denied。"""
        gateway.issue_grant(TASK_ID, SLOT, ["read_file"], SUPERVISOR)

        assert gateway.verify_grant(SLOT, "read_file", TASK_ID) is True
        assert denied_events(gateway._events) == []

    def test_reason_prefers_first_matching_grant(self, gateway):
        """同主体多授权：首个匹配授权的拒绝原因作为备注原因。"""
        first = gateway.issue_grant(TASK_ID, SLOT, ["write_file"], SUPERVISOR)
        second = gateway.issue_grant(TASK_ID, SLOT, ["run_in_sandbox"], SUPERVISOR)
        gateway.revoke_grant(second.grant_id)

        assert gateway.verify_grant(SLOT, "read_file", TASK_ID) is False
        events = denied_events(gateway._events)
        assert len(events) == 1
        # first 白名单不含 read_file（不撤销未过期）→ 命中“工具不在授权白名单”
        assert events[0]["payload"]["reason"] == "工具不在授权白名单"

    async def test_runtime_wrapper_denial_audited(self, gateway):
        """运行时包裹拦截时同样留下 tool_denied 审计痕迹。"""
        tool, invoked = make_probe_tool()
        wrapped = _wrap_tool_with_permission_check(
            tool, gateway, "s1", agent_id=SLOT, task_id=TASK_ID
        )

        payload = await invoke(wrapped, {"path": "a.txt"})
        assert payload["status"] == "error"
        assert invoked == []
        events = denied_events(gateway._events)
        assert len(events) == 1
        assert events[0]["payload"]["reason"] == "无匹配授权"

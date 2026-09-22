"""单元测试：工具授权网关 PermissionGateway 与子 Agent 工具注入边界。

覆盖主管专属工具约束（leader 补充要求）：
- code_agent / subagent_creator（及 review_agent）为「主管专属工具」，
  不可被分配给任何子 Agent：即使出现在 ToolGrant 白名单中，也会在
  动态子 Agent 的注入边界被拒绝（ignored_tools），实际 Agent.tools 不含；
- 动态子 Agent 默认看不到任何工具（空授权 → tools=[]）；
- 只有经评审注册的生成工具才进入可分配候选（ToolRegistry 审批门槛）。
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest
from agents.tool_context import ToolContext

from so_agent.config import Settings
from so_agent.context import ProjectContext
from so_agent.models import ExecutionResult, GeneratedToolManifest, SubtaskSpec, ToolGrant
from so_agent.runtime.events import (
    EVENT_TOOL_GRANTED,
    EVENT_TOOL_REVOKED,
    EventLogger,
)
from so_agent.runtime.permissions import (
    PermissionGateway,
    PermissionGatewayError,
)
from so_agent.runtime.registry import AgentRegistry, DEFAULT_SUPERVISOR_ID
from so_agent.runtime.tool_registry import ToolRegistry
from so_agent.tool_packages.subagent_creator import agent as creator_mod

UTC = timezone.utc

SUPERVISOR = DEFAULT_SUPERVISOR_ID
SUBAGENT = "t1:s1:dynamic-deadbeef"

# 主管专属工具（严禁分配给子 Agent）
EXCLUSIVE_TOOLS = ("code_agent", "subagent_creator", "review_agent")
# 动态工具池（可注入子 Agent）
POOL_TOOLS = ("read_file", "write_file", "run_in_sandbox")


@pytest.fixture
def context(tmp_path):
    return ProjectContext(
        sandbox_dir=tmp_path, project_name="permissions-test"
    )


@pytest.fixture
def events(context):
    return EventLogger(context=context)


@pytest.fixture
def gateway(context, events):
    return PermissionGateway(
        context=context, settings=Settings(), events=events
    )


def make_ctx(tool_name: str, arguments: dict) -> ToolContext:
    return ToolContext(
        context=None,
        tool_name=tool_name,
        tool_call_id=f"call-{tool_name}-1",
        tool_arguments=json.dumps(arguments, ensure_ascii=False),
    )


class FakeRunResult:
    def __init__(self, final_output):
        self.final_output = final_output


def install_fake_runner(monkeypatch, captured_agents: list):
    """把 subagent_creator 模块内的 Runner 替换为记录 agent 实例的假实现。"""

    class FakeRunner:
        @classmethod
        async def run(cls, starting_agent, input=None, *, max_turns=None, **kwargs):
            captured_agents.append(starting_agent)
            return FakeRunResult(
                ExecutionResult(subtask_id="ignored", success=True, output="done")
            )

    monkeypatch.setattr(creator_mod, "Runner", FakeRunner)
    return FakeRunner


# ===========================================================================
# 一、授权签发（只有主管可签发，子 Agent 不可转授）
# ===========================================================================
class TestIssueGrant:
    def test_supervisor_can_issue(self, gateway):
        grant = gateway.issue_grant("t1", SUBAGENT, ["write_file"], SUPERVISOR)
        assert grant.grant_id.startswith("grant-")
        assert grant.task_id == "t1"
        assert grant.agent_id == SUBAGENT
        assert grant.allowed_tools == ["write_file"]
        assert grant.issued_by == SUPERVISOR
        assert grant.revoked is False
        assert grant.expires_at is None

    def test_subagent_cannot_issue(self, gateway):
        """转授拒绝：非主管标识签发一律 PermissionError。"""
        with pytest.raises(PermissionError):
            gateway.issue_grant("t1", "a2", ["write_file"], issued_by=SUBAGENT)

    def test_subagent_cannot_grant_for_itself(self, gateway):
        with pytest.raises(PermissionError):
            gateway.issue_grant("t1", SUBAGENT, ["write_file"], issued_by=SUBAGENT)

    def test_empty_task_id_rejected(self, gateway):
        with pytest.raises(PermissionGatewayError):
            gateway.issue_grant("", SUBAGENT, ["write_file"], SUPERVISOR)

    def test_empty_agent_id_rejected(self, gateway):
        with pytest.raises(PermissionGatewayError):
            gateway.issue_grant("t1", "  ", ["write_file"], SUPERVISOR)

    def test_empty_whitelist_rejected(self, gateway):
        with pytest.raises(PermissionGatewayError):
            gateway.issue_grant("t1", SUBAGENT, [], SUPERVISOR)

    def test_whitespace_only_whitelist_rejected(self, gateway):
        with pytest.raises(PermissionGatewayError):
            gateway.issue_grant("t1", SUBAGENT, ["   ", ""], SUPERVISOR)

    def test_whitelist_deduped_trimmed_ordered(self, gateway):
        grant = gateway.issue_grant(
            "t1",
            SUBAGENT,
            [" write_file ", "read_file", "write_file", "", "read_file"],
            SUPERVISOR,
        )
        assert grant.allowed_tools == ["write_file", "read_file"]

    def test_naive_expires_at_treated_as_utc(self, gateway):
        naive = datetime(2030, 1, 1, 12, 0)
        grant = gateway.issue_grant(
            "t1", SUBAGENT, ["write_file"], SUPERVISOR, expires_at=naive
        )
        assert grant.expires_at is not None
        assert grant.expires_at.tzinfo is not None
        assert grant.expires_at.utcoffset().total_seconds() == 0

    def test_grant_mirrored_to_context(self, gateway, context):
        grant = gateway.issue_grant("t1", SUBAGENT, ["write_file"], SUPERVISOR)
        assert context.tool_grants[grant.grant_id] is grant

    def test_grant_event_logged(self, gateway, context):
        gateway.issue_grant("t1", SUBAGENT, ["write_file"], SUPERVISOR)
        granted = [
            record
            for record in context.event_log
            if record["event_type"] == EVENT_TOOL_GRANTED
        ]
        assert len(granted) == 1
        assert granted[0]["task_id"] == "t1"
        assert granted[0]["agent_id"] == SUBAGENT
        assert granted[0]["payload"]["grant"]["allowed_tools"] == ["write_file"]

    def test_get_grant_and_grants_view(self, gateway):
        grant = gateway.issue_grant("t1", SUBAGENT, ["write_file"], SUPERVISOR)
        assert gateway.get_grant(grant.grant_id) is grant
        assert gateway.get_grant("nope") is None
        snapshot = gateway.grants
        snapshot.clear()
        assert len(gateway.grants) == 1  # 浅拷贝，不影响内部登记


# ===========================================================================
# 二、五维校验（调用者 / 任务 / 工具 / 白名单 / 有效期）
# ===========================================================================
class TestVerifyGrant:
    @pytest.fixture
    def scoped_gateway(self, gateway):
        gateway.issue_grant("t1", SUBAGENT, ["write_file", "read_file"], SUPERVISOR)
        return gateway

    def test_authorized_call_passes(self, scoped_gateway):
        assert scoped_gateway.verify_grant(SUBAGENT, "write_file", "t1") is True
        assert scoped_gateway.verify_grant(SUBAGENT, "read_file", "t1") is True

    def test_wrong_agent_rejected(self, scoped_gateway):
        assert scoped_gateway.verify_grant("other-agent", "write_file", "t1") is False

    def test_wrong_task_rejected(self, scoped_gateway):
        assert scoped_gateway.verify_grant(SUBAGENT, "write_file", "t2") is False

    def test_tool_not_in_whitelist_rejected(self, scoped_gateway):
        assert scoped_gateway.verify_grant(SUBAGENT, "run_in_sandbox", "t1") is False
        assert scoped_gateway.verify_grant(SUBAGENT, "code_agent", "t1") is False

    def test_blank_inputs_rejected(self, scoped_gateway):
        assert scoped_gateway.verify_grant("", "write_file", "t1") is False
        assert scoped_gateway.verify_grant(SUBAGENT, "", "t1") is False
        assert scoped_gateway.verify_grant(SUBAGENT, "write_file", "") is False

    def test_revoked_grant_rejected_immediately(self, scoped_gateway):
        grant = scoped_gateway.grants.popitem()[1]
        scoped_gateway.revoke_grant(grant.grant_id)
        assert scoped_gateway.verify_grant(SUBAGENT, "write_file", "t1") is False

    def test_expired_grant_rejected(self, context, events):
        gateway = PermissionGateway(context=context, settings=Settings(), events=events)
        gateway.issue_grant(
            "t1",
            SUBAGENT,
            ["write_file"],
            SUPERVISOR,
            expires_at=datetime.now(UTC) - timedelta(seconds=1),
        )
        assert gateway.verify_grant(SUBAGENT, "write_file", "t1") is False

    def test_not_yet_expired_grant_passes(self, gateway):
        gateway.issue_grant(
            "t1",
            SUBAGENT,
            ["write_file"],
            SUPERVISOR,
            expires_at=datetime.now(UTC) + timedelta(hours=1),
        )
        assert gateway.verify_grant(SUBAGENT, "write_file", "t1") is True

    def test_any_matching_grant_among_multiple(self, gateway):
        gateway.issue_grant("t1", SUBAGENT, ["read_file"], SUPERVISOR)
        gateway.issue_grant("t1", SUBAGENT, ["write_file"], SUPERVISOR)
        assert gateway.verify_grant(SUBAGENT, "write_file", "t1") is True
        assert gateway.verify_grant(SUBAGENT, "read_file", "t1") is True
        assert gateway.verify_grant(SUBAGENT, "run_in_sandbox", "t1") is False


# ===========================================================================
# 三、撤销与过期清理
# ===========================================================================
class TestRevoke:
    def test_revoke_takes_effect_immediately(self, gateway):
        grant = gateway.issue_grant("t1", SUBAGENT, ["write_file"], SUPERVISOR)
        gateway.revoke_grant(grant.grant_id)
        assert grant.revoked is True
        assert gateway.verify_grant(SUBAGENT, "write_file", "t1") is False

    def test_revoke_idempotent(self, gateway):
        grant = gateway.issue_grant("t1", SUBAGENT, ["write_file"], SUPERVISOR)
        gateway.revoke_grant(grant.grant_id)
        gateway.revoke_grant(grant.grant_id)  # 重复撤销不抛异常
        assert grant.revoked is True

    def test_revoke_unknown_raises(self, gateway):
        with pytest.raises(PermissionGatewayError):
            gateway.revoke_grant("nope")

    def test_revoke_event_recorded(self, gateway, context):
        grant = gateway.issue_grant("t1", SUBAGENT, ["write_file"], SUPERVISOR)
        gateway.revoke_grant(grant.grant_id)
        revoked = [
            record
            for record in context.event_log
            if record["event_type"] == EVENT_TOOL_REVOKED
        ]
        assert len(revoked) == 1
        assert revoked[0]["payload"]["reason"] == "revoked"

    def test_revoke_grants_for_agent_batch(self, gateway):
        g1 = gateway.issue_grant("t1", SUBAGENT, ["write_file"], SUPERVISOR)
        g2 = gateway.issue_grant("t1", SUBAGENT, ["read_file"], SUPERVISOR)
        other = gateway.issue_grant("t1", "other-agent", ["read_file"], SUPERVISOR)
        revoked = gateway.revoke_grants_for_agent(SUBAGENT)
        assert sorted(revoked) == sorted([g1.grant_id, g2.grant_id])
        assert other.revoked is False
        assert gateway.verify_grant(SUBAGENT, "write_file", "t1") is False
        assert gateway.verify_grant("other-agent", "read_file", "t1") is True

    def test_context_and_gateway_share_grant_objects(self, gateway, context):
        grant = gateway.issue_grant("t1", SUBAGENT, ["write_file"], SUPERVISOR)
        gateway.revoke_grant(grant.grant_id)
        assert context.tool_grants[grant.grant_id].revoked is True


class TestPurgeExpired:
    def test_expired_marked_revoked(self, gateway):
        expired = gateway.issue_grant(
            "t1",
            SUBAGENT,
            ["write_file"],
            SUPERVISOR,
            expires_at=datetime(2025, 1, 1, tzinfo=UTC),
        )
        purged = gateway.purge_expired(now=datetime(2030, 1, 1, tzinfo=UTC))
        assert purged == [expired.grant_id]
        assert expired.revoked is True

    def test_future_and_never_expiring_kept(self, gateway):
        future = gateway.issue_grant(
            "t1",
            "a-future",
            ["write_file"],
            SUPERVISOR,
            expires_at=datetime(2035, 1, 1, tzinfo=UTC),
        )
        never = gateway.issue_grant("t1", "a-never", ["write_file"], SUPERVISOR)
        purged = gateway.purge_expired(now=datetime(2030, 1, 1, tzinfo=UTC))
        assert purged == []
        assert future.revoked is False
        assert never.revoked is False

    def test_boundary_equals_now_purged(self, gateway):
        moment = datetime(2030, 1, 1, tzinfo=UTC)
        grant = gateway.issue_grant(
            "t1", SUBAGENT, ["write_file"], SUPERVISOR, expires_at=moment
        )
        assert gateway.purge_expired(now=moment) == [grant.grant_id]

    def test_naive_now_treated_as_utc(self, gateway):
        grant = gateway.issue_grant(
            "t1",
            SUBAGENT,
            ["write_file"],
            SUPERVISOR,
            expires_at=datetime(2025, 1, 1, tzinfo=UTC),
        )
        purged = gateway.purge_expired(now=datetime(2030, 1, 1))  # naive
        assert purged == [grant.grant_id]

    def test_expired_reason_in_event(self, gateway, context):
        gateway.issue_grant(
            "t1",
            SUBAGENT,
            ["write_file"],
            SUPERVISOR,
            expires_at=datetime(2025, 1, 1, tzinfo=UTC),
        )
        gateway.purge_expired(now=datetime(2030, 1, 1, tzinfo=UTC))
        revoked = [
            record
            for record in context.event_log
            if record["event_type"] == EVENT_TOOL_REVOKED
        ]
        assert revoked[-1]["payload"]["reason"] == "expired"


# ===========================================================================
# 四、主管专属工具不可分配给子 Agent（leader 补充要求）
# ===========================================================================
class TestExclusiveToolsCannotReachSubagents:
    """在 create_subagent 的注入边界验证专属工具被拒绝。"""

    @pytest.fixture
    def creator_env(self, context, tmp_path):
        settings = Settings(max_dynamic_agents=8)
        registry = AgentRegistry(context=context, settings=settings)
        tools = creator_mod.build_creator_tools(
            context,
            settings=settings,
            registry=registry,
            subagent_model="test-model",
        )
        return {"tools": tools, "registry": registry, "settings": settings}

    async def _create(self, tools, allowed_tools, subtask_id="s1"):
        spec = SubtaskSpec(
            subtask_id=subtask_id,
            title="demo",
            instructions="do it",
            role="code",
            allowed_tools=list(allowed_tools),
        )
        grant = ToolGrant(
            grant_id="g1",
            task_id="t1",
            agent_id="slot",
            allowed_tools=list(allowed_tools),
            issued_by=SUPERVISOR,
        )
        raw = await tools["create_subagent"].on_invoke_tool(
            make_ctx("create_subagent", {}),
            json.dumps(
                {
                    "subtask_spec_json": spec.model_dump_json(),
                    "tool_grant_json": grant.model_dump_json(),
                }
            ),
        )
        return json.loads(raw)

    async def test_empty_grant_yields_no_tools(self, creator_env, monkeypatch):
        """子 Agent 默认看不到任何工具：空授权 → tools=[]。"""
        captured: list = []
        install_fake_runner(monkeypatch, captured)
        tools = creator_env["tools"]
        payload = await self._create(tools, [])
        assert payload["injected_tools"] == []
        assert payload["ignored_tools"] == []

        raw = await tools["execute_subagent"].on_invoke_tool(
            make_ctx("execute_subagent", {}),
            json.dumps({"agent_id": payload["agent_id"], "input_data": "{}"}),
        )
        result = json.loads(raw)
        assert result["success"] is True
        assert captured[0].tools == []

    async def test_exclusive_tools_rejected_at_injection(self, creator_env, monkeypatch):
        """尝试把 code_agent/subagent_creator/review_agent 授权给子 Agent → 全部被拒绝注入。"""
        captured: list = []
        install_fake_runner(monkeypatch, captured)
        tools = creator_env["tools"]
        payload = await self._create(tools, EXCLUSIVE_TOOLS)
        assert payload["injected_tools"] == []
        assert payload["ignored_tools"] == list(EXCLUSIVE_TOOLS)
        assert payload["allowed_tools"] == []  # 记录中也不沉淀专属工具

        await tools["execute_subagent"].on_invoke_tool(
            make_ctx("execute_subagent", {}),
            json.dumps({"agent_id": payload["agent_id"], "input_data": "{}"}),
        )
        assert captured[0].tools == []  # 专属工具确实没有进入子 Agent

    async def test_mixed_grant_only_pool_tools_injected(self, creator_env, monkeypatch):
        """混合白名单：仅动态工具池内的工具被注入，专属工具被拒绝。"""
        captured: list = []
        install_fake_runner(monkeypatch, captured)
        tools = creator_env["tools"]
        payload = await self._create(tools, ["write_file", "code_agent", "subagent_creator"])
        assert payload["injected_tools"] == ["write_file"]
        assert payload["ignored_tools"] == ["code_agent", "subagent_creator"]

        await tools["execute_subagent"].on_invoke_tool(
            make_ctx("execute_subagent", {}),
            json.dumps({"agent_id": payload["agent_id"], "input_data": "{}"}),
        )
        injected_names = [tool.name for tool in captured[0].tools]
        assert injected_names == ["write_file"]

    async def test_all_pool_tools_injected_when_granted(self, creator_env, monkeypatch):
        captured: list = []
        install_fake_runner(monkeypatch, captured)
        tools = creator_env["tools"]
        payload = await self._create(tools, list(POOL_TOOLS))
        assert payload["injected_tools"] == list(POOL_TOOLS)
        assert payload["ignored_tools"] == []

        await tools["execute_subagent"].on_invoke_tool(
            make_ctx("execute_subagent", {}),
            json.dumps({"agent_id": payload["agent_id"], "input_data": "{}"}),
        )
        assert sorted(tool.name for tool in captured[0].tools) == sorted(POOL_TOOLS)

    async def test_unregistered_tool_name_ignored(self, creator_env, monkeypatch):
        """未注册（非池内）名字同样无法进入子 Agent。"""
        install_fake_runner(monkeypatch, [])
        tools = creator_env["tools"]
        payload = await self._create(tools, ["write_file", "not_a_real_tool"])
        assert payload["injected_tools"] == ["write_file"]
        assert payload["ignored_tools"] == ["not_a_real_tool"]


# ===========================================================================
# 五、启用工具注册表后的签发校验（专属 / 未登记工具一律拒绝）
# ===========================================================================
class TestIssueGrantWithToolRegistry:
    """启用注册表后：issue_grant 只放行已登记的可分配工具。"""

    # 注册表中真正的专属工具（review_agent 可分配，不在此列）
    EXCLUSIVE = ("code_agent", "subagent_creator")

    @pytest.fixture
    def registry(self):
        return ToolRegistry()

    @pytest.fixture
    def gated_gateway(self, context, events, registry):
        return PermissionGateway(
            context=context,
            settings=Settings(),
            events=events,
            tool_registry=registry,
        )

    def test_exclusive_tool_grant_rejected(self, gated_gateway):
        for name in self.EXCLUSIVE:
            with pytest.raises(PermissionError):
                gated_gateway.issue_grant("t1", SUBAGENT, [name], SUPERVISOR)

    def test_unknown_tool_grant_rejected(self, gated_gateway):
        with pytest.raises(PermissionError):
            gated_gateway.issue_grant("t1", SUBAGENT, ["ghost_tool"], SUPERVISOR)

    def test_mixed_whitelist_rejected_as_whole(self, gated_gateway):
        """白名单混入专属工具 → 整体拒绝（不允许静默裁剪）。"""
        with pytest.raises(PermissionError):
            gated_gateway.issue_grant(
                "t1", SUBAGENT, ["write_file", "code_agent"], SUPERVISOR
            )

    def test_assignable_tools_grant_allowed(self, gated_gateway):
        grant = gated_gateway.issue_grant(
            "t1", SUBAGENT, ["write_file", "read_file", "run_in_sandbox"], SUPERVISOR
        )
        assert grant.allowed_tools == ["write_file", "read_file", "run_in_sandbox"]

    def test_approved_generated_tool_grant_allowed(self, gated_gateway, registry):
        registry.register_generated_tool(
            GeneratedToolManifest(
                tool_id="gt-1",
                tool_name="csv_report",
                entry_file="csv_report.py",
                created_by_task="t1",
                review_status="approved",  # type: ignore[arg-type]
            )
        )
        grant = gated_gateway.issue_grant("t1", SUBAGENT, ["csv_report"], SUPERVISOR)
        assert grant.allowed_tools == ["csv_report"]

    def test_get_assignable_tools_excludes_exclusive(self, gated_gateway):
        names = gated_gateway.get_assignable_tools()
        assert "code_agent" not in names
        assert "subagent_creator" not in names
        assert "review_agent" in names
        assert "write_file" in names

    def test_get_assignable_tools_without_registry_returns_empty(self, gateway):
        assert gateway.get_assignable_tools() == []


# ===========================================================================
# 六、动态生成工具的评审注册门槛（生成工具须 approved 才可放行）
# ===========================================================================
class TestGeneratedToolApprovalGate:
    def _manifest(self, status: str) -> GeneratedToolManifest:
        return GeneratedToolManifest(
            tool_id="gt-1",
            tool_name="report_writer",
            entry_file="report_writer.py",
            created_by_task="t1",
            review_status=status,  # type: ignore[arg-type]
        )

    def test_pending_generated_tool_not_approved(self):
        registry = ToolRegistry()
        registry.register_generated_tool(self._manifest("pending"))
        assert registry.is_approved("report_writer") is False
        assert "report_writer" not in registry.list_approved_tools()

    def test_rejected_generated_tool_not_approved(self):
        registry = ToolRegistry()
        registry.register_generated_tool(self._manifest("rejected"))
        assert registry.is_approved("report_writer") is False

    def test_approved_after_review_passes_gate(self):
        registry = ToolRegistry()
        registry.register_generated_tool(self._manifest("pending"))
        registry.register_generated_tool(self._manifest("approved"))  # 评审通过后更新
        assert registry.is_approved("report_writer") is True
        assert "report_writer" in registry.list_approved_tools()

    def test_unknown_tool_not_approved(self):
        registry = ToolRegistry()
        assert registry.is_approved("never_registered") is False

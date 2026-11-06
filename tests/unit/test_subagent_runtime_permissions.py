"""单元测试：动态子 Agent 运行时权限校验（Task #21）。

覆盖"构造时注入 + 运行时校验"双重防护的第二层
（``_wrap_tool_with_permission_check`` + PermissionGateway.verify_grant）：

- 包裹层单元行为：
  * 有权限（授权有效）→ 透传原始工具，真实执行并返回原结果；
  * 无权限（网关未签发授权）→ 拦截并返回明确错误 JSON，原工具不执行；
  * 权限过期（expires_at 已过）→ 运行时拦截；
  * 授权撤销 → 已包裹工具的后续调用立即失效；
  * 任务编号不匹配（授权绑定其他 task_id）→ 拦截；
  * 包裹保留工具元数据（name/description/schema），且不修改原始工具。
- create_subagent 集成路径：
  * 提供网关时注入工具被包裹：凭槽位授权放行真实工具调用（read_file）；
  * 凭证未在网关登记（伪造授权）→ 运行时被拦截；
  * 过期槽位授权 → 运行时被拦截；
  * 其他子任务槽位的授权不能放行本子任务（槽位身份严格匹配）；
  * 未提供网关时保持原行为（不包裹、不拦截，向后兼容）。

校验身份约定：运行时校验以授权凭证（ToolGrant）登记的槽位身份
（``grant.agent_id`` 形如 ``{task_id}:{subtask_id}:grant-slot``）与
``grant.task_id`` 为准——与规划阶段 PlanningCoordinator._issue_grants
签发的授权保持同一登记链；动态子 Agent 自身的 agent_id
（``...dynamic-{hex}``）不是网关登记的授权主体。
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest
from agents import FunctionTool
from agents.tool_context import ToolContext

from so_agent.config import Settings
from so_agent.context import ProjectContext
from so_agent.models import ExecutionResult, SubtaskSpec, ToolGrant
from so_agent.runtime.events import EventLogger
from so_agent.runtime.permissions import PermissionGateway
from so_agent.runtime.registry import AgentRegistry, DEFAULT_SUPERVISOR_ID
from so_agent.tool_packages.subagent_creator import agent as creator_mod
from so_agent.tool_packages.subagent_creator.agent import (
    _wrap_tool_with_permission_check,
)

UTC = timezone.utc
SUPERVISOR = DEFAULT_SUPERVISOR_ID
SLOT = "t1:s1:grant-slot"  # 规划阶段按子任务槽位签发授权的挂载点标识


@pytest.fixture
def context(tmp_path):
    return ProjectContext(sandbox_dir=tmp_path, project_name="runtime-gate-test")


@pytest.fixture
def events(context):
    return EventLogger(context=context)


@pytest.fixture
def gateway(context, events):
    return PermissionGateway(context=context, settings=Settings(), events=events)


def make_ctx(tool_name: str, arguments: dict) -> ToolContext:
    return ToolContext(
        context=None,
        tool_name=tool_name,
        tool_call_id=f"call-{tool_name}-1",
        tool_arguments=json.dumps(arguments, ensure_ascii=False),
    )


def make_probe_tool(name: str = "read_file") -> tuple[FunctionTool, list[str]]:
    """构造带调用记录的探针 FunctionTool（模拟注入池中的工具）。"""
    invoked: list[str] = []

    async def _invoke(ctx, args_json: str) -> str:
        invoked.append(args_json)
        return json.dumps(
            {"status": "ok", "tool": name, "args": args_json}, ensure_ascii=False
        )

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


async def invoke(tool: FunctionTool, arguments: dict) -> dict:
    raw = await tool.on_invoke_tool(
        make_ctx(tool.name, arguments), json.dumps(arguments, ensure_ascii=False)
    )
    return json.loads(raw)


# ===========================================================================
# 一、包裹层单元行为（_wrap_tool_with_permission_check）
# ===========================================================================
class TestWrapToolWithPermissionCheck:
    def _wrap(self, tool: FunctionTool, gateway: PermissionGateway) -> FunctionTool:
        return _wrap_tool_with_permission_check(
            tool, gateway, "s1", agent_id=SLOT, task_id="t1"
        )

    async def test_authorized_tool_executes(self, gateway):
        """有权限的工具正常执行：verify_grant 通过 → 透传原始工具。"""
        gateway.issue_grant("t1", SLOT, ["read_file"], SUPERVISOR)
        tool, invoked = make_probe_tool()
        wrapped = self._wrap(tool, gateway)

        payload = await invoke(wrapped, {"path": "a.txt"})
        assert payload["status"] == "ok"
        assert payload["tool"] == "read_file"
        assert invoked == [json.dumps({"path": "a.txt"}, ensure_ascii=False)]

    async def test_unauthorized_tool_blocked(self, gateway):
        """无权限的工具被拦截：网关未签发任何授权 → 返回错误，原工具不执行。"""
        tool, invoked = make_probe_tool()
        wrapped = self._wrap(tool, gateway)

        payload = await invoke(wrapped, {"path": "a.txt"})
        assert payload["status"] == "error"
        assert "运行时权限校验拦截" in payload["error"]
        assert "read_file" in payload["error"]
        assert invoked == []  # 原工具从未被执行

    async def test_expired_grant_blocked(self, gateway):
        """权限过期的工具被拦截：expires_at 已过 → verify_grant 拒绝。"""
        gateway.issue_grant(
            "t1",
            SLOT,
            ["read_file"],
            SUPERVISOR,
            expires_at=datetime.now(UTC) - timedelta(seconds=1),
        )
        tool, invoked = make_probe_tool()
        wrapped = self._wrap(tool, gateway)

        payload = await invoke(wrapped, {"path": "a.txt"})
        assert payload["status"] == "error"
        assert "运行时权限校验拦截" in payload["error"]
        assert invoked == []

    async def test_revoked_grant_blocked_immediately(self, gateway):
        """授权撤销后运行时立即失效：同一包裹工具先放行、撤销后即拦截。"""
        grant = gateway.issue_grant("t1", SLOT, ["read_file"], SUPERVISOR)
        tool, invoked = make_probe_tool()
        wrapped = self._wrap(tool, gateway)

        first = await invoke(wrapped, {"path": "a.txt"})
        assert first["status"] == "ok"

        gateway.revoke_grant(grant.grant_id)
        second = await invoke(wrapped, {"path": "a.txt"})
        assert second["status"] == "error"
        assert "运行时权限校验拦截" in second["error"]
        assert len(invoked) == 1  # 撤销后原工具未再执行

    async def test_grant_bound_to_other_task_blocked(self, gateway):
        """任务编号不匹配：授权绑定 t2，本调用按 t1 校验 → 拦截。"""
        gateway.issue_grant("t2", SLOT, ["read_file"], SUPERVISOR)
        tool, invoked = make_probe_tool()
        wrapped = self._wrap(tool, gateway)

        payload = await invoke(wrapped, {"path": "a.txt"})
        assert payload["status"] == "error"
        assert invoked == []

    async def test_tool_not_in_whitelist_blocked(self, gateway):
        """授权白名单不含该工具名（write_file 未获批）→ 拦截。"""
        gateway.issue_grant("t1", SLOT, ["read_file"], SUPERVISOR)
        tool, invoked = make_probe_tool("write_file")
        wrapped = _wrap_tool_with_permission_check(
            tool, gateway, "s1", agent_id=SLOT, task_id="t1"
        )

        payload = await invoke(wrapped, {"path": "a.txt", "content": "x"})
        assert payload["status"] == "error"
        assert "write_file" in payload["error"]
        assert invoked == []

    def test_wrapper_preserves_metadata_and_leaves_original(self, gateway):
        """包裹保留工具元数据，且不修改原始工具实例。"""
        tool, _ = make_probe_tool()
        original_invoke = tool.on_invoke_tool
        original_schema = json.dumps(tool.params_json_schema, sort_keys=True)

        wrapped = self._wrap(tool, gateway)

        assert wrapped.name == tool.name
        assert wrapped.description == tool.description
        assert wrapped.strict_json_schema == tool.strict_json_schema
        assert (
            json.dumps(wrapped.params_json_schema, sort_keys=True) == original_schema
        )
        assert tool.on_invoke_tool is original_invoke  # 原工具未被改动
        assert wrapped.on_invoke_tool is not original_invoke


# ===========================================================================
# 二、create_subagent 集成路径（构造时注入 + 运行时校验双重防护）
# ===========================================================================
class TestCreateSubagentRuntimeGate:
    @pytest.fixture
    def creator_env(self, context, events):
        settings = Settings(max_dynamic_agents=8)
        registry = AgentRegistry(context=context, settings=settings)
        gateway = PermissionGateway(
            context=context, settings=settings, events=events
        )
        tools = creator_mod.build_creator_tools(
            context,
            settings=settings,
            registry=registry,
            subagent_model="test-model",
            gateway=gateway,
        )
        return {
            "tools": tools,
            "registry": registry,
            "settings": settings,
            "gateway": gateway,
        }

    @pytest.fixture
    def creator_env_no_gateway(self, context):
        """未提供网关的旧式装配（向后兼容基准）。"""
        settings = Settings(max_dynamic_agents=8)
        registry = AgentRegistry(context=context, settings=settings)
        tools = creator_mod.build_creator_tools(
            context,
            settings=settings,
            registry=registry,
            subagent_model="test-model",
        )
        return {"tools": tools, "registry": registry, "settings": settings}

    @staticmethod
    def _spec(subtask_id: str = "s1") -> SubtaskSpec:
        return SubtaskSpec(
            subtask_id=subtask_id,
            title="演示子任务",
            instructions="读取沙箱文件",
            role="code",
            allowed_tools=["read_file"],
        )

    @staticmethod
    def _grant(agent_id: str = SLOT, allowed=("read_file",)) -> ToolGrant:
        return ToolGrant(
            grant_id="g-manual",
            task_id="t1",
            agent_id=agent_id,
            allowed_tools=list(allowed),
            issued_by=SUPERVISOR,
        )

    async def _create(self, tools, spec: SubtaskSpec, grant: ToolGrant) -> dict:
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

    async def _execute(self, tools, agent_id: str) -> dict:
        raw = await tools["execute_subagent"].on_invoke_tool(
            make_ctx("execute_subagent", {}),
            json.dumps({"agent_id": agent_id, "input_data": "{}"}),
        )
        return json.loads(raw)

    def _sandbox_file(self, context) -> None:
        (context.sandbox_dir / "hello.txt").write_text(
            "hello runtime gate", encoding="utf-8"
        )

    async def test_authorized_tool_passes_runtime_gate(
        self, creator_env, context, monkeypatch
    ):
        """有权限的工具正常执行：槽位授权经网关签发 → 真实 read_file 放行。"""
        captured: list = []
        install_fake_runner(monkeypatch, captured)
        self._sandbox_file(context)
        grant = creator_env["gateway"].issue_grant(
            "t1", SLOT, ["read_file"], SUPERVISOR
        )

        payload = await self._create(creator_env["tools"], self._spec(), grant)
        assert payload["injected_tools"] == ["read_file"]
        execution = await self._execute(creator_env["tools"], payload["agent_id"])
        assert execution["success"] is True

        sub_agent = captured[0]
        assert [tool.name for tool in sub_agent.tools] == ["read_file"]
        result = await invoke(sub_agent.tools[0], {"path": "hello.txt"})
        assert result["status"] == "ok"
        assert result["content"] == "hello runtime gate"  # 真实工具已执行

    async def test_fabricated_grant_blocked_at_runtime(
        self, creator_env, context, monkeypatch
    ):
        """无权限的工具被拦截：凭证未在网关登记（伪造授权）→ 调用返回错误。"""
        captured: list = []
        install_fake_runner(monkeypatch, captured)
        self._sandbox_file(context)

        payload = await self._create(
            creator_env["tools"], self._spec(), self._grant()
        )
        assert payload["injected_tools"] == ["read_file"]  # 构造时过滤仍放行
        await self._execute(creator_env["tools"], payload["agent_id"])

        result = await invoke(captured[0].tools[0], {"path": "hello.txt"})
        assert result["status"] == "error"
        assert "运行时权限校验拦截" in result["error"]
        assert "read_file" in result["error"]

    async def test_expired_slot_grant_blocked_at_runtime(
        self, creator_env, context, monkeypatch
    ):
        """权限过期的工具被拦截：槽位授权已过期 → 运行时拒绝。"""
        captured: list = []
        install_fake_runner(monkeypatch, captured)
        self._sandbox_file(context)
        grant = creator_env["gateway"].issue_grant(
            "t1",
            SLOT,
            ["read_file"],
            SUPERVISOR,
            expires_at=datetime.now(UTC) - timedelta(hours=1),
        )

        payload = await self._create(creator_env["tools"], self._spec(), grant)
        await self._execute(creator_env["tools"], payload["agent_id"])

        result = await invoke(captured[0].tools[0], {"path": "hello.txt"})
        assert result["status"] == "error"
        assert "运行时权限校验拦截" in result["error"]

    async def test_cross_subtask_slot_grant_not_honored(
        self, creator_env, context, monkeypatch
    ):
        """为 s2 槽位签发的授权不能放行 s1 的子 Agent（槽位身份严格匹配）。"""
        captured: list = []
        install_fake_runner(monkeypatch, captured)
        self._sandbox_file(context)
        creator_env["gateway"].issue_grant(
            "t1", "t1:s2:grant-slot", ["read_file"], SUPERVISOR
        )

        payload = await self._create(
            creator_env["tools"], self._spec(), self._grant()
        )
        await self._execute(creator_env["tools"], payload["agent_id"])

        result = await invoke(captured[0].tools[0], {"path": "hello.txt"})
        assert result["status"] == "error"
        assert "运行时权限校验拦截" in result["error"]

    async def test_revoked_slot_grant_blocks_after_creation(
        self, creator_env, context, monkeypatch
    ):
        """构造后撤销授权 → 运行时调用立即被拦截（不依赖重建 Agent）。"""
        captured: list = []
        install_fake_runner(monkeypatch, captured)
        self._sandbox_file(context)
        grant = creator_env["gateway"].issue_grant(
            "t1", SLOT, ["read_file"], SUPERVISOR
        )

        payload = await self._create(creator_env["tools"], self._spec(), grant)
        await self._execute(creator_env["tools"], payload["agent_id"])
        tool = captured[0].tools[0]

        ok = await invoke(tool, {"path": "hello.txt"})
        assert ok["status"] == "ok"

        creator_env["gateway"].revoke_grant(grant.grant_id)
        blocked = await invoke(tool, {"path": "hello.txt"})
        assert blocked["status"] == "error"
        assert "运行时权限校验拦截" in blocked["error"]

    async def test_no_gateway_keeps_raw_tools(
        self, creator_env_no_gateway, context, monkeypatch
    ):
        """未提供网关 → 不包裹、不拦截（向后兼容：仅构造时过滤）。"""
        captured: list = []
        install_fake_runner(monkeypatch, captured)
        self._sandbox_file(context)

        payload = await self._create(
            creator_env_no_gateway["tools"],
            self._spec(),
            self._grant(),  # 从未在网关登记（也无网关）
        )
        await self._execute(creator_env_no_gateway["tools"], payload["agent_id"])

        result = await invoke(captured[0].tools[0], {"path": "hello.txt"})
        assert result["status"] == "ok"
        assert result["content"] == "hello runtime gate"

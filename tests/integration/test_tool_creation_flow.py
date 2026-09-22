"""集成测试：工具缺口自动创建流程（MCP 全链路，不消耗真实 API）。

模拟场景（workflow 全链路 + 真实 MCP 适配器 + 假代码 Agent MCP Server）：

1. 主管拆解任务时为子任务声明尚不存在的工具（csv_report）→ 控制层在
   执行阶段检测到工具缺口，经 MCP 调用代码 Agent 创建候选（真实
   write_generated_tool：写源码 + registry.json + mcp_schema.json，
   review_status=pending）；
2. 评审 Agent 审批通过后由控制层注册为 approved（自动创建 MCPToolServer
   并写回清单）；只有 approved 工具才补发 ToolGrant 并交由子 Agent 执行；
3. 评审不通过或代码 Agent 创建失败时缺口未解决：子任务标记失败
   （tool_gap 证据），触发重规划；
4. 已批准工具在同一计划版本的返工中复用，不重复创建；
5. 创建/审批全过程以 tool_called 事件留痕（gap_detected / candidate_created
   / registered / gap_unresolved）。
"""

from __future__ import annotations

import json
from types import SimpleNamespace

from agents.tool_context import ToolContext

import so_agent.tool_packages.code_agent.agent as code_agent_mod
import so_agent.workflow as workflow_mod
from so_agent.config import Settings
from so_agent.context import ProjectContext
from so_agent.mcp.client import MCPClient
from so_agent.mcp.protocol import MCPToolSchema
from so_agent.mcp.server import MCPToolServer
from so_agent.models import ExecutionResult, ReviewDecision, SubtaskSpec, TaskStatus
from so_agent.orchestrator import OrchestratorOutput
from so_agent.runtime.events import EVENT_TOOL_CALLED
from so_agent.runtime.tool_registry import ToolRegistry
from so_agent.tool_packages.code_agent.agent import build_code_tools
from so_agent.tool_packages.review_agent.agent import ReviewInput, ReviewStage
from so_agent.workflow import WorkflowEngine

# 子任务需要、但初始并不存在的"工作工具"
WORK_TOOL = "csv_report"

# 代码 Agent 产出的候选工具源码（原样落盘，供断言）
TOOL_SOURCE = '''"""csv_report：把记录列表渲染为 CSV 报表文本。"""

from __future__ import annotations


def run(rows: list[dict[str, object]]) -> dict[str, object]:
    """把记录列表渲染为 CSV 文本（表头取首行的键）。"""
    if not rows:
        return {"status": "ok", "csv": ""}
    header = list(rows[0].keys())
    lines = [",".join(header)]
    for row in rows:
        lines.append(",".join(str(row.get(key, "")) for key in header))
    return {"status": "ok", "csv": "\\n".join(lines)}
'''


class FakeRunResult:
    """假 RunResult：只保留 final_output，供控制层结构与文本解析。"""

    def __init__(self, final_output):
        self.final_output = final_output


def install_fake_runner(
    monkeypatch, *, orchestrator, review_agent, supervisor, reviews
):
    """把 workflow 模块内的 Runner 替换为按 Agent 身份分派的假 Runner。"""

    class FakeRunner:
        @classmethod
        async def run(cls, starting_agent, input=None, *, max_turns=None, **kwargs):
            if starting_agent is review_agent:
                review_input = ReviewInput.model_validate_json(input)
                return FakeRunResult(await reviews(review_input))
            payload = json.loads(input)
            return FakeRunResult(await supervisor(payload))

    monkeypatch.setattr(workflow_mod, "Runner", FakeRunner)


class FakeCodeAgentServer:
    """假代码 Agent（MCP Server 包装）：真实调用 write_generated_tool 落盘候选。

    模拟"控制层经 MCP 调用代码 Agent"的真实通道：handler 记录调用参数，
    调用真实 ``write_generated_tool`` 写入候选（源码 + 清单 + MCP 契约），
    并按 CodeAgentOutput 形态返回 JSON 文本。
    """

    def __init__(self, context: ProjectContext, *, succeed: bool = True):
        self.calls: list[dict] = []
        self._tools = build_code_tools(context)
        self._succeed = succeed
        self._server = MCPToolServer(name="code_agent")
        self._server.register_tool(
            MCPToolSchema(
                name="code_agent",
                description="假代码 Agent（MCP 包装，集成测试用）",
                inputSchema={"type": "object", "properties": {}},
            ),
            self._handle,
        )

    @property
    def server(self) -> MCPToolServer:
        """返回包装好的 MCP Server（供工作流注入调用）。"""
        return self._server

    async def _handle(self, arguments: dict) -> str:
        """MCP handler：记录调用；成功时真实落盘候选并返回 CodeAgentOutput JSON。"""
        self.calls.append(dict(arguments))
        if not self._succeed:
            return json.dumps(
                {
                    "success": False,
                    "action_performed": "create",
                    "error": "模拟创建失败：无法生成工具源码",
                    "summary": "",
                },
                ensure_ascii=False,
            )
        write_arguments = json.dumps(
            {
                "tool_name": WORK_TOOL,
                "code": TOOL_SOURCE,
                "manifest_data": json.dumps(
                    {
                        "description": "把记录列表渲染为 CSV 报表文本",
                        "input_schema": {
                            "type": "object",
                            "properties": {"rows": {"type": "array"}},
                            "required": ["rows"],
                        },
                        "created_by_task": str(
                            arguments.get("task_id") or "unknown"
                        ),
                        "version": "0.1.0",
                    },
                    ensure_ascii=False,
                ),
            },
            ensure_ascii=False,
        )
        ctx = ToolContext(
            context=None,
            tool_name="write_generated_tool",
            tool_call_id="call-write-generated-1",
            tool_arguments=write_arguments,
        )
        raw = await self._tools["write_generated_tool"].on_invoke_tool(
            ctx, write_arguments
        )
        data = json.loads(raw)
        ok = data.get("status") == "ok"
        return json.dumps(
            {
                "success": ok,
                "action_performed": "create（写入生成工具候选）",
                "error": None if ok else data.get("error"),
                "summary": f"候选 {WORK_TOOL} 已写入 generated_tools",
            },
            ensure_ascii=False,
        )


class GapSupervisor:
    """模拟主管：各计划版本按脚本声明 allowed_tools（含未注册的工具缺口）。"""

    def __init__(self, *, plans: list[list[str]]):
        self.plan_tools = list(plans)
        self.payloads: list[dict] = []
        self.exec_payloads: dict[str, list[dict]] = {}

    @property
    def phases(self) -> list[str]:
        return [payload.get("phase") for payload in self.payloads]

    async def __call__(self, payload: dict) -> OrchestratorOutput:
        self.payloads.append(payload)
        phase = payload.get("phase")
        if phase in ("planning", "replanning"):
            return self._plan_output(payload)
        if phase == "execute_subtask":
            subtask_id = payload["subtask"]["subtask_id"]
            self.exec_payloads.setdefault(subtask_id, []).append(payload)
            return self._execution_output(payload)
        if phase == "aggregating":
            return OrchestratorOutput(
                task_id=str(payload.get("task_id") or "stub-task"),
                status=TaskStatus.AGGREGATING,
                final_result="含新注册工具产出的最终结果",
            )
        raise AssertionError(f"未预期的阶段载荷：{phase}")

    def _plan_output(self, payload: dict) -> OrchestratorOutput:
        plan_calls = sum(
            1 for item in self.payloads if item.get("phase") in ("planning", "replanning")
        )
        index = min(plan_calls - 1, len(self.plan_tools) - 1)
        spec = SubtaskSpec(
            subtask_id="s1",
            title="生成 CSV 报表模块",
            instructions=f"使用已注册的 {WORK_TOOL} 工具实现报表模块",
            role="code",
            allowed_tools=list(self.plan_tools[index]),
            max_attempts=3,
        )
        return OrchestratorOutput(
            task_id=str(payload.get("task_id") or "stub-task"),
            status=(
                TaskStatus.REPLANNING
                if payload.get("phase") == "replanning"
                else TaskStatus.PLANNING
            ),
            plan_versions=[[spec]],
        )

    def _execution_output(self, payload: dict) -> OrchestratorOutput:
        subtask_id = payload["subtask"]["subtask_id"]
        result = ExecutionResult(
            subtask_id=subtask_id, success=True, output=f"{WORK_TOOL} 模块产出完成"
        )
        return OrchestratorOutput(
            task_id=str(payload.get("task_id") or "stub-task"),
            status=TaskStatus.EXECUTING,
            final_result=result.model_dump_json(),
        )


class ReviewScript:
    """评审脚本：计划评审固定通过；工具评审与结果评审按序列返回。"""

    def __init__(self, *, tool_results=(True,), result_results=(True,)):
        self.tool_results = list(tool_results) or [True]
        self.result_results = list(result_results) or [True]
        self.calls: list[ReviewInput] = []
        self._tool_index = 0
        self._result_index = 0

    async def __call__(self, review_input: ReviewInput) -> ReviewDecision:
        self.calls.append(review_input)
        if review_input.stage == ReviewStage.TOOL_REVIEW:
            passed = self.tool_results[
                min(self._tool_index, len(self.tool_results) - 1)
            ]
            self._tool_index += 1
            return ReviewDecision(
                stage="plan_review",
                passed=passed,
                issues=[] if passed else ["候选工具输入输出契约不完整"],
                required_fixes=[] if passed else ["补充 input_schema 与错误处理"],
                summary="工具评审（模拟）",
            )
        if review_input.stage == ReviewStage.RESULT_REVIEW:
            passed = self.result_results[
                min(self._result_index, len(self.result_results) - 1)
            ]
            self._result_index += 1
            return ReviewDecision(
                stage="result_review",
                passed=passed,
                issues=[] if passed else ["整体结果不满足验收标准"],
                retry_target=None if passed else "s1",
                summary="结果评审（模拟）",
            )
        return ReviewDecision(stage="plan_review", passed=True, summary="计划评审（模拟）")


def build_env(
    tmp_path,
    monkeypatch,
    *,
    plans,
    code_succeed=True,
    tool_results=(True,),
    result_results=(True,),
):
    """组装环境：临时 generated_tools 目录 + 假代码 Agent MCP Server + 引擎。"""
    generated_dir = tmp_path / "generated_tools"
    registry_path = generated_dir / "registry.json"
    monkeypatch.setattr(code_agent_mod, "GENERATED_TOOLS_DIR", generated_dir)
    monkeypatch.setattr(code_agent_mod, "GENERATED_REGISTRY_PATH", registry_path)

    context = ProjectContext(sandbox_dir=tmp_path, project_name="tool-creation-flow")
    settings = Settings(_env_file=None)
    code_server = FakeCodeAgentServer(context, succeed=code_succeed)
    mcp_client = MCPClient()
    mcp_client.connect(code_server.server)
    registry = ToolRegistry(generated_tools_dir=generated_dir)
    supervisor = GapSupervisor(plans=plans)
    reviews = ReviewScript(tool_results=tool_results, result_results=result_results)
    orchestrator = SimpleNamespace(name="fake-supervisor")
    review_agent = SimpleNamespace(name="fake-reviewer")
    install_fake_runner(
        monkeypatch,
        orchestrator=orchestrator,
        review_agent=review_agent,
        supervisor=supervisor,
        reviews=reviews,
    )
    engine = WorkflowEngine(
        context,
        settings=settings,
        orchestrator=orchestrator,
        review_agent=review_agent,
        tool_registry=registry,
        mcp_client=mcp_client,
    )
    return SimpleNamespace(
        context=context,
        engine=engine,
        supervisor=supervisor,
        reviews=reviews,
        code_server=code_server,
        generated_dir=generated_dir,
        registry_path=registry_path,
        registry=registry,
    )


def tool_events(context) -> list[dict]:
    """提取本任务的全部 tool_called 事件载荷（按发生顺序）。"""
    return [
        record["payload"]
        for record in context.event_log
        if record["event_type"] == EVENT_TOOL_CALLED
    ]


def load_registry(generated_dir, registry_path) -> ToolRegistry:
    """从落盘清单重建工具注册表（校验持久化结果）。"""
    reloaded = ToolRegistry(generated_tools_dir=generated_dir)
    reloaded.load_generated_registry(registry_path)
    return reloaded


class TestToolGapCreationFlow:
    async def test_gap_created_reviewed_registered_then_authorized(
        self, tmp_path, monkeypatch
    ):
        env = build_env(
            tmp_path,
            monkeypatch,
            plans=[[WORK_TOOL]],
            result_results=[False, True],  # 一次返工，验证工具复用
        )

        output = await env.engine.run("基于记录渲染 CSV 报表")

        # 1) 缺口处理顺序：发现缺口 → 创建候选 → 注册（approved）
        assert [event["action"] for event in tool_events(env.context)] == [
            "gap_detected",
            "candidate_created",
            "registered",
        ]

        # 2) 经 MCP 调用代码 Agent 一次，携带任务编号
        assert len(env.code_server.calls) == 1
        assert output.task_id in env.code_server.calls[0]["task_description"]

        # 3) 候选真实落盘：源码 + MCP 契约（mcp_schema.json）+ 清单 approved
        assert (env.generated_dir / f"{WORK_TOOL}.py").is_file()
        mcp_schema_path = env.generated_dir / f"{WORK_TOOL}.mcp_schema.json"
        assert mcp_schema_path.is_file()
        assert json.loads(mcp_schema_path.read_text(encoding="utf-8"))["name"] == WORK_TOOL
        reloaded = load_registry(env.generated_dir, env.registry_path)
        assert reloaded.is_approved(WORK_TOOL) is True
        assert reloaded.get_manifest(WORK_TOOL).review_status == "approved"
        assert reloaded.get_mcp_server(WORK_TOOL) is not None
        # 引擎内注册表同样挂载了 MCP Server（含 MCPToolServer）
        assert env.registry.get_mcp_server(WORK_TOOL) is not None

        # 4) 授权补发：执行载荷的 grant 覆盖新注册工具
        first_exec = env.supervisor.exec_payloads["s1"][0]
        assert first_exec["subtask"]["allowed_tools"] == [WORK_TOOL]
        assert first_exec["grant"] is not None
        assert first_exec["grant"]["allowed_tools"] == [WORK_TOOL]
        assert first_exec["grant"]["issued_by"] == "supervisor"

        # 5) 结果评审返工复用工具：不重复创建、再次调度未重新签发缺口
        assert len(env.supervisor.exec_payloads["s1"]) == 2
        assert len(env.code_server.calls) == 1

        # 6) 工作流正常完成：工具创建流程与状态机协同
        assert output.status == TaskStatus.COMPLETED
        assert output.final_result == "含新注册工具产出的最终结果"
        assert env.supervisor.phases == [
            "planning",
            "execute_subtask",
            "aggregating",
            "execute_subtask",
            "aggregating",
        ]

    async def test_gap_review_rejected_marks_subtask_failed_and_replans(
        self, tmp_path, monkeypatch
    ):
        env = build_env(
            tmp_path,
            monkeypatch,
            plans=[[WORK_TOOL], []],  # 重规划后放弃缺口工具
            tool_results=[False],
        )

        output = await env.engine.run("基于记录渲染 CSV 报表")

        # 1) 评审不通过：候选已创建但缺口未解决（不注册、不授权）
        assert [event["action"] for event in tool_events(env.context)] == [
            "gap_detected",
            "candidate_created",
            "gap_unresolved",
        ]
        assert "工具评审未通过" in tool_events(env.context)[-1]["reason"]

        # 2) 候选保持 pending，未进入可放行集合
        reloaded = load_registry(env.generated_dir, env.registry_path)
        assert reloaded.is_approved(WORK_TOOL) is False
        assert reloaded.get_manifest(WORK_TOOL).review_status == "pending"

        # 3) 首版计划未调度执行（缺口失败在调度前拦截）→ 重规划后执行 v2
        assert env.supervisor.phases == [
            "planning",
            "replanning",
            "execute_subtask",
            "aggregating",
        ]
        v2_exec = env.supervisor.exec_payloads["s1"][0]
        assert v2_exec["subtask"]["plan_version"] == 2
        assert v2_exec["grant"] is None  # v2 无工具需求：不签发授权

        # 4) 重规划输入携带缺口诊断
        replan_payload = next(
            item for item in env.supervisor.payloads if item.get("phase") == "replanning"
        )
        issue_text = json.dumps(
            replan_payload["decomposition_issue"], ensure_ascii=False
        )
        assert "工具缺口未解决" in issue_text

        # 5) v2 完成后任务收尾
        assert output.status == TaskStatus.COMPLETED

    async def test_pending_candidate_reused_without_recreation(
        self, tmp_path, monkeypatch
    ):
        env = build_env(tmp_path, monkeypatch, plans=[[WORK_TOOL]])

        # 规划阶段主管先行创建候选（真实 write_generated_tool → pending）
        code_tools = build_code_tools(env.context)
        write_arguments = json.dumps(
            {
                "tool_name": WORK_TOOL,
                "code": TOOL_SOURCE,
                "manifest_data": json.dumps(
                    {"description": "把记录列表渲染为 CSV 报表文本"},
                    ensure_ascii=False,
                ),
            },
            ensure_ascii=False,
        )
        ctx = ToolContext(
            context=None,
            tool_name="write_generated_tool",
            tool_call_id="call-pre-created",
            tool_arguments=write_arguments,
        )
        raw = await code_tools["write_generated_tool"].on_invoke_tool(
            ctx, write_arguments
        )
        assert json.loads(raw)["status"] == "ok"

        output = await env.engine.run("基于记录渲染 CSV 报表")

        # 候选已存在：直接送评审（无 candidate_created），不重复调用代码 Agent
        assert [event["action"] for event in tool_events(env.context)] == [
            "gap_detected",
            "registered",
        ]
        assert env.code_server.calls == []
        reloaded = load_registry(env.generated_dir, env.registry_path)
        assert reloaded.is_approved(WORK_TOOL) is True
        exec_payload = env.supervisor.exec_payloads["s1"][0]
        assert exec_payload["grant"]["allowed_tools"] == [WORK_TOOL]
        assert output.status == TaskStatus.COMPLETED

    async def test_code_agent_creation_failure_marks_subtask_failed(
        self, tmp_path, monkeypatch
    ):
        env = build_env(
            tmp_path,
            monkeypatch,
            plans=[[WORK_TOOL], []],
            code_succeed=False,
        )

        output = await env.engine.run("基于记录渲染 CSV 报表")

        # 创建失败：代码 Agent 被调用一次且缺口未解决
        assert len(env.code_server.calls) == 1
        assert [event["action"] for event in tool_events(env.context)] == [
            "gap_detected",
            "gap_unresolved",
        ]
        assert "代码 Agent 创建工具失败" in tool_events(env.context)[-1]["reason"]

        # 无候选落盘；重规划后 v2 执行成功
        assert not (env.generated_dir / f"{WORK_TOOL}.py").exists()
        assert env.supervisor.phases == [
            "planning",
            "replanning",
            "execute_subtask",
            "aggregating",
        ]
        assert output.status == TaskStatus.COMPLETED

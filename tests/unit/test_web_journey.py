"""全链路快照测试：路由判断轨迹、分支选择、工具调用与动态 Agent。

全部直接构造任务记录与事件日志，不触发真实模型调用。验证：
1. 路由轨迹能区分规则 / 模型 / 升级 / 守卫降级，并标出选中分支；
2. 工具调用按 call_id 配对参数、结果与耗时，并归属子任务；
3. 动态 Agent 生命周期与缺口工具生成过程可被还原；
4. 流程图数据包含判断节点、四个分支、工作流阶段与子任务连线。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from so_agent.config import Settings
from so_agent.context import ProjectContext
from so_agent.models import (
    ClassificationResult,
    ExecutionResult,
    RouteStep,
    SubtaskSpec,
    TaskStatus,
    TaskType,
)
from so_agent.orchestrator import OrchestratorOutput
from so_agent.runtime.events import EventLogger
from so_agent.web.manager import TaskManager, WebTask


class _EngineStub:
    def __init__(self) -> None:
        self.plan_history = [
            [
                SubtaskSpec(
                    subtask_id="s1",
                    title="收集数据",
                    instructions="读取输入并整理",
                    role="analysis",
                    allowed_tools=["read_file", "report_tool"],
                ),
                SubtaskSpec(
                    subtask_id="s2",
                    title="生成报告",
                    instructions="写出报告",
                    role="code",
                    dependencies=["s1"],
                    allowed_tools=["write_file"],
                ),
            ]
        ]
        self.results = {
            "s1": ExecutionResult(
                subtask_id="s1", success=True, output="数据已整理", duration=1.2
            ),
            "s2": ExecutionResult(
                subtask_id="s2", success=True, output="报告已生成", duration=2.0
            ),
        }
        self.plan_version = 1
        self.replan_count = 0
        self.current_status = TaskStatus.COMPLETED


def _make_task(
    tmp_path: Path,
    classification: ClassificationResult,
) -> tuple[TaskManager, WebTask, EventLogger]:
    manager = TaskManager(tmp_path, max_concurrent=1)
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir(exist_ok=True)
    context = ProjectContext(
        sandbox_dir=sandbox,
        project_name="journey-test",
        config=Settings(),
    )
    task = WebTask(
        task_id="task-journey",
        objective="整理数据并生成报告",
        constraints="",
        acceptance_criteria="生成报告文件",
        sandbox_dir=sandbox,
        context=context,
        status="completed",
    )
    task.classification = classification
    task.output = OrchestratorOutput(
        task_id="engine-task",
        status=TaskStatus.COMPLETED,
        final_result="## 报告\n\n已完成",
    )
    manager._tasks[task.task_id] = task
    return manager, task, EventLogger(context=context)


def test_route_trace_distinguishes_method_and_branch(tmp_path):
    classification = ClassificationResult(
        task_type=TaskType.FULL_WORKFLOW,
        reasoning="需要多步骤协作",
        steps=[
            RouteStep(
                stage="rule_check",
                status="skipped",
                detail="存在约束或验收标准，规则预判不适用",
            ),
            RouteStep(
                stage="llm_classify",
                status="passed",
                detail="模型判定 full_workflow：需要多步骤协作",
            ),
        ],
    )
    manager, task, _ = _make_task(tmp_path, classification)

    route = manager._build_route(task)

    assert route["available"] is True
    assert route["method"] == "llm"
    assert route["method_label"] == "模型分类"
    assert route["effective_task_type"] == "full_workflow"
    selected = [item["type"] for item in route["branches"] if item["selected"]]
    assert selected == ["full_workflow"]
    assert all(
        item["status"] == "unselected"
        for item in route["branches"]
        if not item["selected"]
    )
    assert route["guard_fallback"] is False


def test_route_trace_upgrade_and_guard_fallback(tmp_path):
    classification = ClassificationResult(
        task_type=TaskType.DIRECT_ANSWER,
        reasoning="原判直答",
        steps=[
            RouteStep(stage="rule_check", status="miss", detail="未命中规则"),
            RouteStep(
                stage="llm_classify",
                status="passed",
                detail="模型判定 direct_answer",
            ),
            RouteStep(
                stage="tool_intent_upgrade",
                status="upgraded",
                detail="原分类 direct_answer；目标命中工具动作“创建”；升级为完整工作流",
            ),
            RouteStep(
                stage="output_guard",
                status="applied",
                detail="输出包含伪造工具调用标记，已降级为完整工作流",
            ),
        ],
    )
    manager, task, _ = _make_task(tmp_path, classification)

    route = manager._build_route(task)

    assert route["method"] == "upgrade"
    assert route["original_task_type"] == "direct_answer"
    assert route["classified_task_type"] == "direct_answer"
    assert route["effective_task_type"] == "full_workflow"
    assert route["guard_fallback"] is True


def test_journey_guard_fallback_records_initial_and_target_branch(tmp_path):
    classification = ClassificationResult(
        task_type=TaskType.DIRECT_ANSWER,
        reasoning="原判直答",
        steps=[
            RouteStep(stage="rule_check", status="miss", detail="未命中规则"),
            RouteStep(
                stage="llm_classify",
                status="passed",
                detail="模型判定 direct_answer",
            ),
            RouteStep(
                stage="output_guard",
                status="applied",
                detail="输出包含伪造工具调用标记，已降级为完整工作流",
            ),
        ],
    )
    manager, task, _ = _make_task(tmp_path, classification)

    snapshot = manager.snapshot(task.task_id)
    assert snapshot is not None
    nodes = {node["id"]: node for node in snapshot["journey"]["nodes"]}

    assert nodes["branch-direct_answer"]["status"] == "chosen"
    assert "守卫降级 → 完整工作流" in nodes["branch-direct_answer"]["chips"]
    guard_rows = {label: value for label, value in nodes["guard"]["rows"]}
    assert guard_rows == {
        "初始分支": "direct_answer",
        "降级后": "full_workflow",
    }
    edge_pairs = {
        (edge["from"], edge["to"]) for edge in snapshot["journey"]["edges"]
    }
    assert ("branch-direct_answer", "guard") in edge_pairs
    assert ("guard", "planning") in edge_pairs


def test_journey_direct_answer_shows_all_branches(tmp_path):
    classification = ClassificationResult(
        task_type=TaskType.DIRECT_ANSWER,
        reasoning="纯算术直接回答",
        steps=[RouteStep(stage="rule_check", status="hit", detail="命中纯数学表达式规则")],
    )
    manager, task, _ = _make_task(tmp_path, classification)
    task.output = OrchestratorOutput(
        task_id="engine-task", status=TaskStatus.COMPLETED, final_result="42"
    )

    snapshot = manager.snapshot(task.task_id)
    assert snapshot is not None
    journey = snapshot["journey"]
    ids = {node["id"] for node in journey["nodes"]}
    assert {"start", "rule", "branch-direct_answer", "path", "end"} <= ids
    assert "llm" not in ids
    branch_status = {
        node["id"]: node["status"]
        for node in journey["nodes"]
        if node["kind"] == "branch"
    }
    assert branch_status["branch-direct_answer"] == "chosen"
    assert branch_status["branch-full_workflow"] == "unselected"
    branch_edges = [edge for edge in journey["edges"] if edge["kind"] == "branch"]
    assert len(branch_edges) == 4
    chosen = next(
        edge for edge in branch_edges if edge["to"] == "branch-direct_answer"
    )
    assert chosen["status"] == "success"
    assert all(
        edge["dashed"] is True
        for edge in branch_edges
        if edge["to"] != "branch-direct_answer"
    )
    path = next(node for node in journey["nodes"] if node["id"] == "path")
    assert path["status"] == "done"


def test_journey_workflow_records_tools_agents_and_generated_tools(tmp_path):
    classification = ClassificationResult(
        task_type=TaskType.FULL_WORKFLOW,
        reasoning="需要读取文件并写出报告",
        steps=[
            RouteStep(stage="rule_check", status="skipped", detail="存在验收标准"),
            RouteStep(
                stage="llm_classify",
                status="passed",
                detail="模型判定 full_workflow",
            ),
        ],
    )
    manager, task, events = _make_task(tmp_path, classification)
    task.engine = _EngineStub()

    events.log_event(
        "task_dispatched",
        {
            "task_id": "engine-task",
            "subtask_id": "s1",
            "agent_id": "agent-s1",
            "attempt": 1,
            "max_attempts": 3,
        },
    )
    events.log_event(
        "agent_created",
        {
            "task_id": "engine-task",
            "agent_id": "agent-s1",
            "agent_type": "code_agent",
            "subtask_id": "s1",
            "plan_version": 1,
            "allowed_tools": ["read_file", "report_tool"],
            "injected_tools": ["read_file", "report_tool"],
            "ignored_tools": [],
            "replacement_of": None,
        },
    )
    events.log_event(
        "tool_called",
        {
            "agent": "subagent",
            "agent_id": "agent-s1",
            "subtask_id": "s1",
            "action": "invoke_started",
            "tool_name": "read_file",
            "call_id": "call-1",
            "arguments_preview": '{"path": "input.csv"}',
        },
    )
    events.log_event(
        "tool_called",
        {
            "agent": "subagent",
            "agent_id": "agent-s1",
            "subtask_id": "s1",
            "action": "invoke_completed",
            "tool_name": "read_file",
            "call_id": "call-1",
            "output_preview": '{"success": true, "rows": 3}',
            "duration": 0.42,
            "result_ok": True,
        },
    )
    events.log_event(
        "agent_released",
        {
            "task_id": "engine-task",
            "agent_id": "agent-s1",
            "status": "completed",
            "attempt_count": 1,
        },
    )
    for action, extra in [
        ("gap_detected", {}),
        ("candidate_created", {}),
        ("tool_verified", {"test_count": 2}),
        ("registered", {}),
    ]:
        events.log_event(
            "tool_called",
            {
                "task_id": "engine-task",
                "agent_id": "supervisor",
                "tool_name": "report_tool",
                "plan_version": 1,
                "action": action,
                **extra,
            },
        )

    snapshot = manager.snapshot(task.task_id)
    assert snapshot is not None

    tool_calls = snapshot["tool_calls"]
    assert len(tool_calls) == 1
    call = tool_calls[0]
    assert call["tool"] == "read_file"
    assert call["call_id"] == "call-1"
    assert call["subtask_id"] == "s1"
    assert call["status"] == "completed"
    assert call["duration"] == 0.42
    assert "input.csv" in call["arguments"]
    assert "rows" in call["output"]

    agents = snapshot["dynamic_agents"]
    assert len(agents) == 1
    assert agents[0]["agent_id"] == "agent-s1"
    assert agents[0]["subtask_id"] == "s1"
    assert agents[0]["status"] == "completed"
    assert agents[0]["injected_tools"] == ["read_file", "report_tool"]

    generated = snapshot["generated_tools"]
    assert len(generated) == 1
    assert generated[0]["tool_name"] == "report_tool"
    assert generated[0]["status"] == "registered"
    assert [step["action"] for step in generated[0]["steps"]] == [
        "gap_detected",
        "candidate_created",
        "tool_verified",
        "registered",
    ]

    journey = snapshot["journey"]
    ids = {node["id"] for node in journey["nodes"]}
    assert {"planning", "plan_review", "s1", "s2", "aggregation",
            "result_review", "end"} <= ids
    s1 = next(node for node in journey["nodes"] if node["id"] == "s1")
    s2 = next(node for node in journey["nodes"] if node["id"] == "s2")
    assert s1["status"] == "completed"
    assert [item["call_id"] for item in s1["tool_calls"]] == ["call-1"]
    assert [item["agent_id"] for item in s1["agents"]] == ["agent-s1"]
    assert [item["tool_name"] for item in s1["generated_tools"]] == ["report_tool"]
    assert s2["generated_tools"] == []

    edge_pairs = {(edge["from"], edge["to"]) for edge in journey["edges"]}
    assert ("plan_review", "s1") in edge_pairs
    assert ("s1", "s2") in edge_pairs
    assert ("s2", "aggregation") in edge_pairs
    assert ("result_review", "end") in edge_pairs

    messages = [item["message"] for item in snapshot["timeline"]]
    assert any("调用工具 read_file" in message for message in messages)
    assert any("创建 Agent：agent-s1" in message for message in messages)
    assert any("动态工具 report_tool" in message for message in messages)


def test_terminal_task_settles_unmatched_tool_call(tmp_path):
    classification = ClassificationResult(
        task_type=TaskType.DIRECT_ANSWER, reasoning="test"
    )
    manager, task, events = _make_task(tmp_path, classification)
    events.log_event(
        "tool_called",
        {
            "agent": "supervisor",
            "action": "invoke_started",
            "tool_name": "read_file",
            "call_id": "c-missing",
        },
    )

    calls = manager._build_tool_calls(task, {})

    assert calls[0]["status"] == "failed"
    assert calls[0]["result_ok"] is False
    assert "未收到工具输出" in calls[0]["output"]


def test_journey_subtask_separates_dispatch_and_executing_agent(tmp_path):
    classification = ClassificationResult(
        task_type=TaskType.FULL_WORKFLOW,
        reasoning="需要读取文件并写出报告",
        steps=[
            RouteStep(stage="rule_check", status="skipped", detail="存在验收标准"),
            RouteStep(
                stage="llm_classify",
                status="passed",
                detail="模型判定 full_workflow",
            ),
        ],
    )
    manager, task, events = _make_task(tmp_path, classification)
    task.engine = _EngineStub()
    dispatch_agent = "task-journey:s1:primary-1"
    dynamic_agent = "task-journey:s1:dynamic-2"

    events.log_event(
        "task_dispatched",
        {
            "task_id": "engine-task",
            "subtask_id": "s1",
            "agent_id": dispatch_agent,
            "attempt": 1,
            "max_attempts": 3,
        },
    )
    for agent_id in (dispatch_agent, dynamic_agent):
        events.log_event(
            "agent_created",
            {
                "task_id": "engine-task",
                "agent_id": agent_id,
                "agent_type": "dynamic",
                "subtask_id": "s1",
                "plan_version": 1,
                "allowed_tools": ["read_file"],
                "injected_tools": ["read_file"],
                "ignored_tools": [],
                "replacement_of": None,
            },
        )
    events.log_event(
        "tool_called",
        {
            "agent": "subagent",
            "agent_id": dynamic_agent,
            "subtask_id": "s1",
            "action": "invoke_started",
            "tool_name": "read_file",
            "call_id": "call-dyn",
        },
    )
    events.log_event(
        "tool_called",
        {
            "agent": "subagent",
            "agent_id": dynamic_agent,
            "subtask_id": "s1",
            "action": "invoke_completed",
            "tool_name": "read_file",
            "call_id": "call-dyn",
            "result_ok": True,
        },
    )

    snapshot = manager.snapshot(task.task_id)
    assert snapshot is not None
    s1 = next(
        node for node in snapshot["journey"]["nodes"] if node["id"] == "s1"
    )
    rows = {label: value for label, value in s1["rows"]}
    assert rows["派发 Agent"] == dispatch_agent
    assert rows["执行 Agent"].startswith(dynamic_agent)
    assert "最近执行 Agent" not in rows


@pytest.mark.parametrize(
    ("action", "status"),
    [("invoke_failed", "failed"), ("invoke_completed", "completed")],
)
def test_tool_call_status_follows_action(tmp_path, action, status):
    classification = ClassificationResult(
        task_type=TaskType.DIRECT_ANSWER, reasoning="test"
    )
    manager, task, events = _make_task(tmp_path, classification)
    events.log_event(
        "tool_called",
        {
            "agent": "supervisor",
            "action": "invoke_started",
            "tool_name": "read_file",
            "call_id": "c9",
        },
    )
    events.log_event(
        "tool_called",
        {
            "agent": "supervisor",
            "action": action,
            "tool_name": "read_file",
            "call_id": "c9",
            "duration": 0.1,
        },
    )

    calls = manager._build_tool_calls(task, {})
    assert calls[0]["status"] == status

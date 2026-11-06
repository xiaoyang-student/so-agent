"""子 Agent 上下文最小化回归测试。

约束：执行阶段下发给主管/子 Agent 的载荷只包含当前子任务的规格、
授权与尝试信息，不携带兄弟子任务的计划或结果；动态子 Agent 的系统
提示词只描述自己的规格。
"""

from __future__ import annotations

import json

from so_agent.config import Settings
from so_agent.models import ExecutionResult, SubtaskSpec, TaskStatus
from so_agent.orchestrator import OrchestratorOutput
from tests.integration.test_human_escalation import (
    ReviewScript,
    build_engine,
)

_EXPECTED_EXEC_KEYS = {
    "phase",
    "task_id",
    "task_request",
    "subtask",
    "agent",
    "attempt",
    "grant",
    "routing",
}


class TwoSubtaskSupervisor:
    """两子任务计划：记录全部阶段载荷并返回成功结果。"""

    def __init__(self) -> None:
        self.payloads: list[dict] = []

    @property
    def phases(self) -> list[str]:
        return [payload.get("phase") for payload in self.payloads]

    def __call__(self, payload: dict) -> OrchestratorOutput:
        self.payloads.append(payload)
        phase = payload.get("phase")
        task_id = str(payload.get("task_id") or "stub-task")
        if phase == "planning":
            return OrchestratorOutput(
                task_id=task_id,
                status=TaskStatus.PLANNING,
                plan_versions=[
                    [
                        SubtaskSpec(
                            subtask_id="s1",
                            title="采集原始数据",
                            instructions="只处理数据采集，不要生成报告",
                            role="code",
                            allowed_tools=["write_file"],
                            max_attempts=1,
                        ),
                        SubtaskSpec(
                            subtask_id="s2",
                            title="生成最终报告",
                            instructions="只处理报告生成，不要重复采集数据",
                            role="code",
                            dependencies=["s1"],
                            allowed_tools=["write_file"],
                            max_attempts=1,
                        ),
                    ]
                ],
            )
        if phase == "execute_subtask":
            subtask_id = payload["subtask"]["subtask_id"]
            result = ExecutionResult(
                subtask_id=subtask_id,
                success=True,
                output=f"{subtask_id} 完成：" + "结果明细" * 150,
            )
            return OrchestratorOutput(
                task_id=task_id,
                status=TaskStatus.EXECUTING,
                final_result=result.model_dump_json(),
            )
        if phase == "aggregating":
            return OrchestratorOutput(
                task_id=task_id,
                status=TaskStatus.AGGREGATING,
                final_result="两阶段任务完成",
            )
        raise AssertionError(f"未预期的阶段载荷：{phase}")


async def test_execution_payload_only_contains_current_subtask(
    tmp_path, monkeypatch
):
    supervisor = TwoSubtaskSupervisor()
    reviews = ReviewScript([True])
    _, engine, _ = build_engine(
        tmp_path,
        monkeypatch,
        supervisor=supervisor,
        reviews=reviews,
        settings=Settings(_env_file=None, max_replan_attempts=3),
    )

    output = await engine.run("完成两阶段任务")

    assert output.status == TaskStatus.COMPLETED
    exec_payloads = [
        payload
        for payload in supervisor.payloads
        if payload.get("phase") == "execute_subtask"
    ]
    assert [payload["subtask"]["subtask_id"] for payload in exec_payloads] == [
        "s1",
        "s2",
    ]

    titles = {"s1": "采集原始数据", "s2": "生成最终报告"}
    for payload in exec_payloads:
        assert set(payload) == _EXPECTED_EXEC_KEYS
        assert set(payload["task_request"]) == {
            "task_id",
            "objective",
        }
        current = payload["subtask"]["subtask_id"]
        other = "s2" if current == "s1" else "s1"
        serialized = json.dumps(payload, ensure_ascii=False)
        assert titles[other] not in serialized
        assert "execution_results" not in serialized
        assert "plan_versions" not in serialized

        grant = payload["grant"]
        if grant is not None:
            assert grant["task_id"] == payload["task_id"]
            assert set(grant["allowed_tools"]) <= set(
                payload["subtask"]["allowed_tools"]
            )
            # 授权槽位属于当前子任务（不与兄弟子任务共享）
            assert grant["agent_id"].startswith(
                f"{payload['task_id']}:{current}:"
            )


async def test_aggregation_receives_all_results_but_execution_does_not(
    tmp_path, monkeypatch
):
    supervisor = TwoSubtaskSupervisor()
    _, engine, _ = build_engine(
        tmp_path,
        monkeypatch,
        supervisor=supervisor,
        reviews=ReviewScript([True]),
    )

    await engine.run("完成两阶段任务")

    aggregate = next(
        payload
        for payload in supervisor.payloads
        if payload.get("phase") == "aggregating"
    )
    assert {
        item["subtask_id"] for item in aggregate["execution_results"]
    } == {"s1", "s2"}

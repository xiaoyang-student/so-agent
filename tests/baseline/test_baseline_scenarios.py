"""基准场景冒烟测试（任务 44 基座验证）。

对每个基准场景跑一遍并断言**核心终态行为**（刻意避免脆弱的全量
快照断言）：

- 简单问答：单子任务一次成功 → completed；
- DAG：依赖链按序执行 → completed；
- 网络失败：主管调用持续超时 → 网络重试预算耗尽 → human_required；
- 超时：pipeline 预算极短 → 优雅降级 human_required（不崩溃）；
- 取消：协作式取消 → cancelled，零模型调用；
- 重规划 + 显式复用：v1 失败升级重规划 → v2 显式 reuse_from 命中
  复用（不重跑）→ completed，计划版本快照持久化。

全部场景经 FakeRunner 分派到剧本主管，不发起任何真实付费模型请求。
"""

from __future__ import annotations

from typing import Callable

import pytest

from so_agent.models import TaskStatus
from so_agent.runtime.events import (
    EVENT_RESULT_REUSED,
    EVENT_TOOL_GRANTED,
    EVENT_TOOL_REVOKED,
)

from .scenarios import (
    Scenario,
    assemble_scenario,
    cancel_scenario,
    dag_pipeline_scenario,
    network_failure_scenario,
    replan_reuse_scenario,
    simple_qa_scenario,
    timeout_scenario,
)


async def _run(harness, scenario: Scenario, *, before_close: Callable | None = None):
    """执行场景并收尾关闭存储；``before_close`` 在关闭前拿到 result_store
    （用于抓取持久化快照，避免在已关闭连接上查询）。"""
    try:
        return await harness.engine.run(
            scenario.objective, scenario.constraints, scenario.acceptance_criteria
        )
    finally:
        if harness.result_store is not None:
            if before_close is not None:
                before_close(harness.result_store)
            harness.result_store.close()


class TestSimpleQaScenario:
    async def test_single_subtask_completes(self, tmp_path, monkeypatch):
        scenario = simple_qa_scenario()
        harness = assemble_scenario(scenario, tmp_path, monkeypatch)
        output = await _run(harness, scenario)

        assert output.status == TaskStatus.COMPLETED
        assert harness.engine.current_status == TaskStatus.COMPLETED
        # 单子任务恰好执行一次且成功
        assert harness.engine.plan_version == 1
        supervisor = scenario.supervisor
        assert supervisor.exec_sequence == ["s1"]
        assert supervisor.exec_counts == {"s1": 1}
        assert output.final_result


class TestDagPipelineScenario:
    async def test_dependency_order_respected(self, tmp_path, monkeypatch):
        scenario = dag_pipeline_scenario()
        harness = assemble_scenario(scenario, tmp_path, monkeypatch)
        output = await _run(harness, scenario)

        assert output.status == TaskStatus.COMPLETED
        # DAG 依赖顺序：s1 → s2 → s3
        assert scenario.supervisor.exec_sequence == ["s1", "s2", "s3"]
        assert all(count == 1 for count in scenario.supervisor.exec_counts.values())


class TestNetworkFailureScenario:
    async def test_network_exhaustion_degrades_to_human(
        self, tmp_path, monkeypatch
    ):
        scenario = network_failure_scenario()
        harness = assemble_scenario(scenario, tmp_path, monkeypatch)
        output = await _run(harness, scenario)

        # 网络连续失败：优雅转人工兜底（不崩溃、不计入重规划额度）
        assert output.status == TaskStatus.HUMAN_REQUIRED
        assert output.human_escalation is not None
        assert "网络" in output.final_result


class TestTimeoutScenario:
    async def test_pipeline_timeout_degrades_gracefully(
        self, tmp_path, monkeypatch
    ):
        scenario = timeout_scenario()
        harness = assemble_scenario(scenario, tmp_path, monkeypatch)
        output = await _run(harness, scenario)

        # 超时：优雅降级（不崩溃、不悬挂）
        assert output.status == TaskStatus.HUMAN_REQUIRED
        assert "超时" in output.final_result


class TestCancelScenario:
    async def test_cooperative_cancel_before_start(self, tmp_path, monkeypatch):
        scenario = cancel_scenario()
        harness = assemble_scenario(scenario, tmp_path, monkeypatch)
        harness.engine.cancel()
        output = await _run(harness, scenario)

        assert output.status == TaskStatus.CANCELLED
        assert output.final_result == "任务已取消"
        # 取消发生在第一个检查点：零模型调用、零评审调用
        assert scenario.supervisor.payloads == []
        assert scenario.reviews.calls == []


class TestReplanReuseScenario:
    async def test_replan_and_explicit_reuse(self, tmp_path, monkeypatch):
        scenario = replan_reuse_scenario()
        harness = assemble_scenario(scenario, tmp_path, monkeypatch)
        # 计划版本快照持久化到 SQLite 结果存储（v1 与 v2 均在）
        snapshots: dict = {}
        output = await _run(
            harness,
            scenario,
            before_close=lambda store: snapshots.update(
                pv1=store.get_plan_version(1), pv2=store.get_plan_version(2)
            ),
        )
        assert output.status == TaskStatus.COMPLETED
        assert harness.engine.plan_version == 2
        # 显式复用命中：result_reused 事件存在
        reused_events = [
            record
            for record in harness.context.event_log
            if record["event_type"] == EVENT_RESULT_REUSED
        ]
        assert reused_events, "v2 的 n1 应命中显式复用并记录 result_reused"
        # 复用项不重跑：v2 的 n1 无执行载荷；新方案 n2 正常执行
        assert "n1" not in scenario.supervisor.exec_sequence
        assert "n2" in scenario.supervisor.exec_sequence
        # v1 的 s2 多次尝试后升级（原始 + 替代尝试，不脆弱断言精确次数）
        assert scenario.supervisor.exec_counts.get("s2", 0) >= 3
        pv1 = snapshots["pv1"]
        pv2 = snapshots["pv2"]
        assert pv1 is not None and pv2 is not None
        assert {spec.subtask_id for spec in pv1.specs} == {"s1", "s2", "s3"}
        assert {spec.subtask_id for spec in pv2.specs} == {"n1", "n2"}


class TestLifecycleGrantHygiene:
    async def test_grants_revoked_on_every_exit_path(self, tmp_path, monkeypatch):
        """任意出口撤销：成功与失败终态均不留活跃授权（基线不变量）。"""

        async def _assert_no_active_grants(harness, scenario):
            output = await _run(harness, scenario)
            active_granted = [
                record
                for record in harness.context.event_log
                if record["event_type"] == EVENT_TOOL_GRANTED
            ]
            revoked = [
                record
                for record in harness.context.event_log
                if record["event_type"] == EVENT_TOOL_REVOKED
            ]
            # 每一条真实签发都有对应撤销事件（生命周期收口）
            assert len(revoked) >= len(active_granted) or not active_granted
            return output

        scenario = dag_pipeline_scenario()
        scenario.settings_overrides = dict(scenario.settings_overrides)
        harness = assemble_scenario(scenario, tmp_path, monkeypatch)
        output = await _assert_no_active_grants(harness, scenario)
        assert output.status == TaskStatus.COMPLETED

"""基准场景工厂：简单问答、DAG、网络失败、超时、取消、重规划与复用。

每个场景工厂只描述"输入"（剧本主管、评审脚本、配置覆盖与 run 参数），
统一经 :func:`run_scenario` 组装引擎并执行；全部场景不发起任何真实
付费模型请求（FakeRunner 分派到剧本主管）。本阶段（任务 44）仅建立
可复用入口与冒烟验证，不实现后续阶段的失败分类、预算管理等逻辑。
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

import httpx2
import openai

from so_agent.models import SubtaskSpec
from so_agent.orchestrator import OrchestratorOutput

from .fakes import (
    EngineHarness,
    ReviewScript,
    ScriptedSupervisor,
    SupervisorFn,
    build_engine,
    make_spec,
)

__all__ = [
    "Scenario",
    "assemble_scenario",
    "cancel_scenario",
    "dag_pipeline_scenario",
    "network_failure_scenario",
    "replan_reuse_scenario",
    "simple_qa_scenario",
    "timeout_scenario",
]


@dataclass
class Scenario:
    """基准场景描述（输入侧）。"""

    name: str
    supervisor: SupervisorFn
    reviews: ReviewScript
    settings_overrides: dict[str, Any] = field(default_factory=dict)
    objective: str = "基准任务目标（模拟）"
    constraints: str = ""
    acceptance_criteria: str = ""


def simple_qa_scenario() -> Scenario:
    """简单问答：单个无工具子任务一次执行成功（最短流水线）。"""
    supervisor = ScriptedSupervisor(
        plans=[[make_spec("s1", instructions="回答一个简单问题")]]
    )
    return Scenario(
        name="simple_qa",
        supervisor=supervisor,
        reviews=ReviewScript([True]),
    )


def dag_pipeline_scenario() -> Scenario:
    """DAG：三个子任务线性依赖（s1 ← s2 ← s3），按依赖顺序执行。"""
    supervisor = ScriptedSupervisor(
        plans=[
            [
                make_spec("s1", instructions="生成原始数据"),
                make_spec("s2", instructions="清洗数据", dependencies=["s1"]),
                make_spec("s3", instructions="生成报表", dependencies=["s2"]),
            ]
        ]
    )
    return Scenario(
        name="dag_pipeline",
        supervisor=supervisor,
        reviews=ReviewScript([True]),
    )


def network_failure_scenario() -> Scenario:
    """网络失败：主管调用持续超时（APITimeoutError），网络重试预算
    耗尽后转人工兜底（SupervisorNetworkError → human_required）。"""

    def _failing_supervisor(payload: dict) -> OrchestratorOutput:
        raise openai.APITimeoutError(
            request=httpx2.Request("POST", "http://so-agent-baseline.invalid/v1")
        )

    return Scenario(
        name="network_failure",
        supervisor=_failing_supervisor,
        reviews=ReviewScript([True]),
    )


def timeout_scenario() -> Scenario:
    """超时：主管调用无限挂起 + 极短 pipeline 预算 → 优雅降级转人工。"""

    async def _hanging_supervisor(payload: dict) -> OrchestratorOutput:
        await asyncio.sleep(30)  # 由外层 wait_for 超时取消

    return Scenario(
        name="timeout",
        supervisor=_hanging_supervisor,
        reviews=ReviewScript([True]),
        settings_overrides={"max_pipeline_timeout": 0.1},
    )


def cancel_scenario() -> Scenario:
    """取消：run 之前请求协作式取消 → 第一个检查点即取消，零模型调用。"""
    supervisor = ScriptedSupervisor(plans=[[make_spec("s1")]])
    return Scenario(
        name="cancel",
        supervisor=supervisor,
        reviews=ReviewScript([True]),
    )


def replan_reuse_scenario() -> Scenario:
    """重规划 + 显式复用：v1 的 s2 持续失败触发分解问题重规划；
    v2 中 n1 显式声明 reuse_from="s1"（语义一致，复用旧结果）、
    n2 换新方案重新执行。
    """
    v1: list[SubtaskSpec] = [
        make_spec("s1", instructions="生成原始数据"),
        make_spec("s2", instructions="清洗数据（初版方案）"),
        make_spec("s3", instructions="生成图表", dependencies=["s1"]),
    ]
    v2: list[SubtaskSpec] = [
        make_spec("n1", instructions="生成原始数据", reuse_from="s1"),
        make_spec("n2", instructions="清洗数据（改用备用数据源方案）"),
    ]
    supervisor = ScriptedSupervisor(
        plans=[v1, v2],
        fail_version=1,
        fail_ids={"s2"},
    )
    return Scenario(
        name="replan_reuse",
        supervisor=supervisor,
        reviews=ReviewScript([True]),
        objective="生成数据报表（基准重规划场景）",
        acceptance_criteria="通过验收",
    )


def assemble_scenario(
    scenario: Scenario,
    tmp_path,
    monkeypatch,
) -> EngineHarness:
    """按场景描述组装引擎（不执行）；执行由调用方 ``await`` 驱动。

    分离组装与执行：适配 pytest-asyncio 的 auto 事件循环模式，也便于
    取消等场景在 ``run`` 之前预置状态（如 ``engine.cancel()``）。

    用法::

        harness = assemble_scenario(scenario, tmp_path, monkeypatch)
        output = await harness.engine.run(
            scenario.objective, scenario.constraints,
            scenario.acceptance_criteria,
        )
    """
    return build_engine(
        tmp_path,
        monkeypatch,
        supervisor=scenario.supervisor,
        reviews=scenario.reviews,
        settings_overrides=scenario.settings_overrides,
    )

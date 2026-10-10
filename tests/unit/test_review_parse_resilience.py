"""单元测试：评审 Agent 解析 BUG 修复（Fix 2 系列回归）。

背景：评审 Agent 以纯文本返回 ReviewDecision JSON，LLM 输出可能带
轻微格式瑕疵（尾逗号、字段类型不符、stage 阶段值越界等）。修复前的
解析失败会被 ``_call_review_agent`` 的宽泛 ``except Exception`` 吞掉，
把实际"通过"的结论保守判定为"未通过"，引发无意义的返工循环。

本文件回归验证三层修复：

1. Fix 2.1（``state_machine._coerce_structured_output``）：ReviewDecision
   解析失败时降级抢救——尾逗号 JSON、passed 保留但其余字段不符；
   任务 32 起纯文本结论关键词不再用于推导结论（文本语气不能决定
   放行与否），缺失显式 passed 字段时抛 ``ReviewContractError``
   （fail closed），完全无法判定时才抛 WorkflowError；
2. Fix 2.3（``models.ReviewDecision.stage``）：放宽为 str，中文阶段名
   与授权/工具评审阶段值不再触发校验失败；
3. Fix 2.2（``workflow._call_review_agent``）：异常面收窄为
   WorkflowError + asyncio.TimeoutError（返回保守未通过并记录
   ``review_parse_error`` 事件），其余异常向上传播不再吞掉。
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

import so_agent.workflow as workflow_mod
from so_agent.config import Settings
from so_agent.context import ProjectContext
from so_agent.models import ReviewDecision, TaskRequest
from so_agent.runtime.events import (
    EVENT_REVIEW_AGENT_OUTPUT,
    EVENT_REVIEW_PARSE_ERROR,
)
from so_agent.state_machine import (
    WorkflowError,
    _coerce_structured_output,
    _rescue_review_decision,
)
from so_agent.tool_packages.review_agent.agent import ReviewInput, ReviewStage
from so_agent.workflow import WorkflowEngine


def make_review_input(
    stage: ReviewStage = ReviewStage.PLAN_REVIEW,
) -> ReviewInput:
    """构造最小合法评审输入。"""
    return ReviewInput(
        stage=stage,
        task_request=TaskRequest(task_id="t1", objective="演示目标"),
    )


def build_engine(tmp_path) -> WorkflowEngine:
    """构造仅用于直调 ``_call_review_agent`` 的引擎（占位 Agent 实例）。"""
    context = ProjectContext(sandbox_dir=tmp_path, project_name="review-parse")
    return WorkflowEngine(
        context,
        settings=Settings(_env_file=None),
        orchestrator=SimpleNamespace(name="fake-supervisor"),
        review_agent=SimpleNamespace(name="fake-reviewer"),
    )


def make_runner(*, output=None, exc: Exception | None = None) -> type:
    """生成一次性假 Runner 类（monkeypatch 替换 workflow_mod.Runner）。"""

    class _Runner:
        @classmethod
        async def run(cls, starting_agent, input=None, *, max_turns=None, **kwargs):
            if exc is not None:
                raise exc
            return SimpleNamespace(final_output=output, raw_responses=[])

    return _Runner


class TestRescueReviewDecision:
    """``_rescue_review_decision`` 的字段抢救规则（纯函数层）。"""

    def test_passed_field_rescued_as_true(self):
        rescued = _rescue_review_decision({"passed": True})
        assert rescued is not None
        assert rescued.passed is True
        assert rescued.stage == "plan_review"
        assert rescued.issues == []
        assert rescued.summary == ""

    def test_string_passed_coerced(self):
        # passed 为字符串 "true"（LLM 常见形态）也能抢救
        rescued = _rescue_review_decision({"passed": "true"})
        assert rescued is not None
        assert rescued.passed is True

    def test_list_fields_coerced_from_string(self):
        rescued = _rescue_review_decision(
            {"passed": False, "issues": "问题一", "required_fixes": "修复项"}
        )
        assert rescued is not None
        assert rescued.issues == ["问题一"]
        assert rescued.required_fixes == ["修复项"]

    def test_negative_keyword_wins(self):
        # 任务 32 反转："未通过"关键词不再推导结论——缺失显式 passed
        # 字段时无法构造合法评审决定，返回 None（调用方 fail closed）
        assert _rescue_review_decision(None, "评审结论：未通过，需返工") is None

    def test_positive_keyword_detected(self):
        # 任务 32 反转："通过"关键词不再推导结论——文本语气不能放行
        assert _rescue_review_decision(None, "评审结论：本次评审通过，无问题。") is None

    def test_text_keyword_not_used_even_with_candidate(self):
        # 即使提供文本，结论仍只来自显式 passed 字段
        rescued = _rescue_review_decision({"passed": False}, "评审结论：本次评审通过")
        assert rescued is not None
        assert rescued.passed is False

    def test_stage_preserved_when_string(self):
        rescued = _rescue_review_decision({"stage": "result_review", "passed": True})
        assert rescued is not None
        assert rescued.stage == "result_review"

    def test_no_signal_returns_none(self):
        # 完全无法判定结论：返回 None（调用方按原失败路径保守拦截）
        assert _rescue_review_decision(None, "无可奉告") is None
        assert _rescue_review_decision(None, "") is None


class TestCoerceReviewDecisionRescue:
    """``_coerce_structured_output`` 对 ReviewDecision 的专属降级路径。"""

    def test_trailing_comma_json_parses(self):
        # 场景 1：尾逗号 JSON → json-repair 修复后直接通过契约校验
        text = '{"stage": "plan_review", "passed": true, "issues": ["无"],}'
        decision = _coerce_structured_output(
            text, ReviewDecision, label="评审 Agent"
        )
        assert decision.passed is True
        assert decision.issues == ["无"]

    def test_passed_true_rescued_when_other_fields_fail(self):
        # 场景 2：passed=true 但 issues 为字符串、summary 为数字
        # → 契约校验失败后降级抢救，结论保持通过不被误判
        json_text = (
            '{"stage": "plan_review", "passed": true, '
            '"issues": "字段类型不符", "summary": 123}'
        )
        decision = _coerce_structured_output(
            json_text, ReviewDecision, label="评审 Agent"
        )
        assert decision.passed is True
        assert decision.issues == ["字段类型不符"]
        assert decision.summary == ""

    def test_dict_input_rescued(self):
        # 场景 2 的字典形态：Runner 可能直接返回 dict
        decision = _coerce_structured_output(
            {"passed": True, "summary": None, "retry_target": 7},
            ReviewDecision,
            label="评审 Agent",
        )
        assert decision.passed is True
        assert decision.summary == ""
        assert decision.retry_target is None

    def test_plain_text_pass_keyword_rejected(self):
        # 任务 32 反转：纯文本含"通过"关键词且无 JSON → 不再降级为
        # passed=True，直接抛 WorkflowError（文本语气不能推导放行）
        with pytest.raises(WorkflowError):
            _coerce_structured_output(
                "整体核对完成：本次评审通过，无必须修复项。",
                ReviewDecision,
                label="评审 Agent",
            )

    def test_plain_text_reject_keyword_rejected(self):
        # 任务 32 反转：纯文本含"未通过"关键词且无 JSON → 同样不推导，
        # 保守拦截（宁可拦截不可放行）
        with pytest.raises(WorkflowError):
            _coerce_structured_output(
                "核验后发现证据链不闭环，结论：未通过。",
                ReviewDecision,
                label="评审 Agent",
            )

    def test_completely_invalid_output_still_raises(self):
        # 场景 4：完全无法判定结论 → 保守拦截（抛 WorkflowError）
        with pytest.raises(WorkflowError):
            _coerce_structured_output(
                "无法给出结构化结论", ReviewDecision, label="评审 Agent"
            )

    def test_stage_chinese_value_accepted(self):
        # Fix 2.3：中文 stage 不再触发枚举校验失败
        text = '{"stage": "计划评审", "passed": true, "summary": "通过"}'
        decision = _coerce_structured_output(
            text, ReviewDecision, label="评审 Agent"
        )
        assert decision.passed is True
        assert decision.stage == "计划评审"


class TestCallReviewAgentResilience:
    """``_call_review_agent`` 的异常面收窄与事件留痕（Fix 2.2）。"""

    async def test_unparseable_output_conservative_fail_with_event(
        self, tmp_path
    ):
        # 场景 4 + 5：完全无效输出 → 保守 passed=False + review_parse_error 事件
        raw = "完全无法解析的评审自由文本"
        workflow_mod.Runner = make_runner(output=raw)
        engine = build_engine(tmp_path)

        decision = await engine._call_review_agent(make_review_input())

        assert decision.passed is False
        assert decision.stage == "plan_review"
        assert decision.summary == "评审不可用，保守判定为未通过"
        records = engine._events.get_events(event_type=EVENT_REVIEW_PARSE_ERROR)
        assert len(records) == 1
        payload = records[0]["payload"]
        assert payload["stage"] == "plan_review"
        assert payload["error"].startswith("WorkflowError:")
        assert payload["raw_output_preview"] == raw

    async def test_value_error_propagates(self, tmp_path):
        # 场景 6：意外异常（ValueError）不再被吞掉，向上传播
        workflow_mod.Runner = make_runner(exc=ValueError("意外网络错误"))
        engine = build_engine(tmp_path)

        with pytest.raises(ValueError, match="意外网络错误"):
            await engine._call_review_agent(make_review_input())

        # 意外异常不属于解析失败：不记录 review_parse_error
        assert engine._events.get_events(event_type=EVENT_REVIEW_PARSE_ERROR) == []

    async def test_timeout_error_conservative_fail_with_event(self, tmp_path):
        # 场景 6：调用超时 → 捕获并保守判定为未通过，事件留痕
        workflow_mod.Runner = make_runner(exc=asyncio.TimeoutError())
        engine = build_engine(tmp_path)

        decision = await engine._call_review_agent(make_review_input())

        assert decision.passed is False
        records = engine._events.get_events(event_type=EVENT_REVIEW_PARSE_ERROR)
        assert len(records) == 1
        payload = records[0]["payload"]
        assert "TimeoutError" in payload["error"]
        assert payload["raw_output_preview"] == "N/A"

    async def test_flawed_pass_json_not_misjudged(self, tmp_path):
        # 核心 BUG 回归：轻微格式瑕疵（尾逗号 + 中文 stage + 字符串 issues）
        # 的"通过"结论必须放行，而不是被保守判定为未通过
        raw = (
            '{"stage": "计划评审", "passed": true, '
            '"issues": ["小问题"], "required_fixes": [],}'
        )
        workflow_mod.Runner = make_runner(output=raw)
        engine = build_engine(tmp_path)

        decision = await engine._call_review_agent(make_review_input())

        assert decision.passed is True
        assert decision.stage == "plan_review"
        assert decision.issues == ["小问题"]

    async def test_success_path_records_agent_output_event(self, tmp_path):
        # 成功解析路径：同样记录 review_agent_output 事件（含原始输出预览）
        raw = '{"stage": "plan_review", "passed": true, "summary": "通过"}'
        workflow_mod.Runner = make_runner(output=raw)
        engine = build_engine(tmp_path)

        decision = await engine._call_review_agent(make_review_input())

        assert decision.passed is True
        records = engine._events.get_events(event_type=EVENT_REVIEW_AGENT_OUTPUT)
        assert len(records) == 1
        payload = records[0]["payload"]
        assert payload["stage"] == "plan_review"
        assert payload["passed"] is True
        assert payload["raw_output_preview"] == raw
        # 成功路径不产生解析错误事件
        assert engine._events.get_events(event_type=EVENT_REVIEW_PARSE_ERROR) == []

    async def test_long_output_truncated_to_1000_chars(self, tmp_path):
        # 原始输出超过 1000 字符时截断，避免事件日志膨胀
        raw = '{"passed": true, "summary": "' + "长" * 1200 + '"}'
        workflow_mod.Runner = make_runner(output=raw)
        engine = build_engine(tmp_path)

        await engine._call_review_agent(make_review_input())

        records = engine._events.get_events(event_type=EVENT_REVIEW_AGENT_OUTPUT)
        assert len(records) == 1
        assert len(records[0]["payload"]["raw_output_preview"]) == 1000

    async def test_result_review_success_event_stage_normalized(self, tmp_path):
        # 结果评审成功路径：事件中的 stage 同步归一化为 result_review
        raw = '{"passed": true, "summary": "结果通过"}'
        workflow_mod.Runner = make_runner(output=raw)
        engine = build_engine(tmp_path)

        await engine._call_review_agent(
            make_review_input(stage=ReviewStage.RESULT_REVIEW)
        )

        records = engine._events.get_events(event_type=EVENT_REVIEW_AGENT_OUTPUT)
        assert len(records) == 1
        payload = records[0]["payload"]
        assert payload["stage"] == "result_review"
        assert payload["passed"] is True

    async def test_result_review_stage_normalized(self, tmp_path):
        # 降级构造的 stage 回退值会被评审输入统一覆盖（结果评审）
        raw = '{"passed": true, "summary": "结果通过"}'
        workflow_mod.Runner = make_runner(output=raw)
        engine = build_engine(tmp_path)

        decision = await engine._call_review_agent(
            make_review_input(stage=ReviewStage.RESULT_REVIEW)
        )

        assert decision.passed is True
        assert decision.stage == "result_review"


class TestNarrowedExceptionFace:
    """``_call_review_agent`` 仅捕获 WorkflowError 与 TimeoutError。"""

    async def test_unexpected_exception_types_propagate(self, tmp_path):
        for exc in (RuntimeError("模型客户端崩溃"), KeyError("missing")):
            workflow_mod.Runner = make_runner(exc=exc)
            engine = build_engine(tmp_path)
            with pytest.raises(type(exc)):
                await engine._call_review_agent(make_review_input())

    async def test_workflow_error_still_conservative(self, tmp_path):
        # 任务 32 更新：有效 JSON 但完全缺 passed 字段 → ReviewContractError
        # → fail closed（保守未通过 + review_contract_error 事件）
        workflow_mod.Runner = make_runner(output='{"stage": "plan_review"}')
        engine = build_engine(tmp_path)

        decision = await engine._call_review_agent(make_review_input())

        assert decision.passed is False
        assert "评审契约无效" in decision.issues[0]
        assert decision.required_fixes
        assert all("契约无效" in fix for fix in decision.required_fixes)


class TestStageTypeWidened:
    """Fix 2.3：``ReviewDecision.stage`` 为 str，任意阶段值可接受。"""

    def test_authorization_and_tool_stage_values_accepted(self):
        for stage in ("authorization_review", "tool_review"):
            decision = ReviewDecision(stage=stage, passed=True)
            assert decision.stage == stage

    def test_workflow_module_reexports_coerce(self):
        # 历史 re-export 入口保持不变（workflow_mod._coerce_structured_output）
        assert workflow_mod._coerce_structured_output is _coerce_structured_output

"""单元测试：任务 32 —— 评审输出契约确定化。

核心不变量：

- 评审放行只依据合法 ReviewDecision 的显式 ``passed`` 布尔字段；
- 代码绝不从评审文本语气或"通过/未通过"等关键词推断结论；
- 字段契约：passed=true 时 required_fixes 必须为空；passed=false 时
  必须非空；suggestions 非空不单独阻断；
- 结构矛盾先做一次（且仅一次）契约修复调用（只修字段一致性，
  禁止重新评审）；修复后仍矛盾 / 缺失显式 passed / 解析失败 →
  fail closed（保守未通过）+ ``review_contract_error`` 审计事件。
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

import so_agent.workflow as workflow_mod
from so_agent.config import Settings
from so_agent.context import ProjectContext
from so_agent.models import ReviewDecision, TaskRequest
from so_agent.planning_coordinator import PlanningCoordinator
from so_agent.runtime.events import (
    EVENT_REVIEW_AGENT_OUTPUT,
    EVENT_REVIEW_CONTRACT_ERROR,
    EVENT_REVIEW_CONTRACT_REPAIR,
    EVENT_REVIEW_PARSE_ERROR,
)
from so_agent.state_machine import (
    ReviewContractError,
    WorkflowError,
    _coerce_structured_output,
    _rescue_review_decision,
    _review_contract_contradiction,
)
from so_agent.tool_packages.review_agent.agent import ReviewInput, ReviewStage
from so_agent.workflow import WorkflowEngine


# ---------------------------------------------------------------------------
# 测试基础设施
# ---------------------------------------------------------------------------
def make_review_input(
    stage: ReviewStage = ReviewStage.PLAN_REVIEW,
) -> ReviewInput:
    """构造最小合法评审输入。"""
    return ReviewInput(
        stage=stage,
        task_request=TaskRequest(task_id="t-contract", objective="契约测试目标"),
    )


def build_engine(tmp_path) -> WorkflowEngine:
    """构造仅用于直调 ``_call_review_agent`` 的引擎（占位 Agent 实例）。"""
    context = ProjectContext(sandbox_dir=tmp_path, project_name="review-contract")
    return WorkflowEngine(
        context,
        settings=Settings(_env_file=None),
        orchestrator=SimpleNamespace(name="fake-supervisor"),
        review_agent=SimpleNamespace(name="fake-reviewer"),
    )


def make_scripted_runner(*outputs):
    """生成按调用顺序弹出脚本的假 Runner。

    Returns:
        ``(runner_cls, inputs)``：runner_cls 为替换 workflow_mod.Runner
        的类；inputs 记录每次调用收到的 input（用于断言调用次数与
        修复消息内容）。
    """
    script = list(outputs)
    inputs: list[str] = []

    class _Runner:
        @classmethod
        async def run(cls, starting_agent, input=None, *, max_turns=None, **kwargs):
            inputs.append(input)
            index = min(len(inputs) - 1, len(script) - 1)
            output = script[index]
            if isinstance(output, Exception):
                raise output
            return SimpleNamespace(final_output=output, raw_responses=[])

    return _Runner, inputs


def install_runner(*outputs) -> tuple[type, list[str]]:
    """生成并直接安装脚本化假 Runner（与既有测试一致的赋值模式）。"""
    runner_cls, inputs = make_scripted_runner(*outputs)
    workflow_mod.Runner = runner_cls
    return runner_cls, inputs


# ---------------------------------------------------------------------------
# 一、纯函数层：_review_contract_contradiction 四种组合 + suggestions
# ---------------------------------------------------------------------------
class TestContractContradictionMatrix:
    """passed × required_fixes 四种组合的契约判定（纯函数层）。"""

    def test_passed_true_with_empty_fixes_is_valid(self):
        decision = ReviewDecision(
            stage="plan_review", passed=True, required_fixes=[]
        )
        assert _review_contract_contradiction(decision) is None

    def test_passed_true_with_fixes_contradicts(self):
        decision = ReviewDecision(
            stage="plan_review",
            passed=True,
            required_fixes=["必须补齐输入契约"],
        )
        contradiction = _review_contract_contradiction(decision)
        assert contradiction is not None
        assert "passed=true" in contradiction
        assert "required_fixes" in contradiction

    def test_passed_false_with_fixes_is_valid(self):
        decision = ReviewDecision(
            stage="plan_review",
            passed=False,
            required_fixes=["重新拆分子任务"],
        )
        assert _review_contract_contradiction(decision) is None

    def test_passed_false_without_fixes_contradicts(self):
        decision = ReviewDecision(stage="plan_review", passed=False)
        contradiction = _review_contract_contradiction(decision)
        assert contradiction is not None
        assert "passed=false" in contradiction

    def test_suggestions_never_block_pass(self):
        # suggestions 非空 + passed=true → 合法通过（不阻断）
        decision = ReviewDecision(
            stage="plan_review",
            passed=True,
            issues=["存在轻微表述问题"],
            suggestions=["建议补充边界用例", "建议补充回滚方案"],
            required_fixes=[],
        )
        assert _review_contract_contradiction(decision) is None

    def test_suggestions_do_not_rescue_missing_fixes(self):
        # suggestions 非空 + passed=false + required_fixes 空 → 仍矛盾
        decision = ReviewDecision(
            stage="plan_review",
            passed=False,
            suggestions=["建议补充边界用例"],
        )
        assert _review_contract_contradiction(decision) is not None


# ---------------------------------------------------------------------------
# 二、解析层：显式 passed 是唯一放行来源；文本关键词失效
# ---------------------------------------------------------------------------
class TestExplicitPassedOnly:
    """只有显式可解析 passed 字段才可构造合法评审决定。"""

    def test_missing_passed_raises_contract_error(self):
        # 提取出 JSON 但缺 passed → ReviewContractError（fail closed 依据）
        with pytest.raises(ReviewContractError):
            _coerce_structured_output(
                '{"stage": "plan_review", "summary": "整体尚可"}',
                ReviewDecision,
                label="评审 Agent",
            )

    def test_dict_missing_passed_raises_contract_error(self):
        with pytest.raises(ReviewContractError):
            _coerce_structured_output(
                {"stage": "result_review", "issues": ["问题"]},
                ReviewDecision,
                label="评审 Agent",
            )

    def test_chinese_pass_word_in_summary_cannot_infer(self):
        # summary 含"通过"但缺 passed 字段 → 仍拒绝（关键词不能推导结论）
        with pytest.raises(ReviewContractError):
            _coerce_structured_output(
                '{"stage": "plan_review", "summary": "本次评审通过"}',
                ReviewDecision,
                label="评审 Agent",
            )

    def test_plain_chinese_text_with_pass_word_not_parsed(self):
        # 纯中文"通过"文本无 JSON → WorkflowError 保守拦截（不降级放行）
        with pytest.raises(WorkflowError):
            _coerce_structured_output(
                "评审结论：通过，可以放行。",
                ReviewDecision,
                label="评审 Agent",
            )

    def test_rescue_ignores_text_keywords(self):
        # 抢救路径同样不使用文本关键词：只有显式 passed 字段生效
        assert _rescue_review_decision(None, "评审结论：通过") is None
        assert _rescue_review_decision(None, "评审结论：未通过") is None
        rescued = _rescue_review_decision(
            {"passed": True, "summary": "文字里写着未通过"},
        )
        assert rescued is not None
        assert rescued.passed is True

    def test_string_passed_still_accepted(self):
        # 显式 passed 字段的宽容布尔解释保留（"true"/"false" 字符串）
        rescued = _rescue_review_decision({"passed": "true"})
        assert rescued is not None and rescued.passed is True
        rescued = _rescue_review_decision({"passed": "false"})
        assert rescued is not None and rescued.passed is False


# ---------------------------------------------------------------------------
# 三、引擎层：_call_review_agent 的契约校验 / 修复 / fail closed
# ---------------------------------------------------------------------------
class TestCallReviewAgentContract:
    """``_call_review_agent`` 的契约执行路径（计划评审视角）。"""

    async def test_valid_pass_decision_no_repair_call(self, tmp_path):
        # 合法通过：直接放行，仅一次 Runner 调用，contract_valid=True
        install_runner(
            '{"stage": "plan_review", "passed": true, '
            '"required_fixes": [], "summary": "通过"}'
        )
        engine = build_engine(tmp_path)

        decision = await engine._call_review_agent(make_review_input())

        assert decision.passed is True
        assert decision.stage == "plan_review"
        assert decision.required_fixes == []
        records = engine._events.get_events(event_type=EVENT_REVIEW_AGENT_OUTPUT)
        assert len(records) == 1
        assert records[0]["payload"]["contract_valid"] is True
        assert records[0]["payload"]["passed"] is True
        # 无修复/契约错误/解析错误事件
        assert engine._events.get_events(event_type=EVENT_REVIEW_CONTRACT_REPAIR) == []
        assert engine._events.get_events(event_type=EVENT_REVIEW_CONTRACT_ERROR) == []
        assert engine._events.get_events(event_type=EVENT_REVIEW_PARSE_ERROR) == []

    async def test_valid_reject_decision_no_repair_call(self, tmp_path):
        # 合法拒绝（passed=false 且 required_fixes 非空）：直接生效
        install_runner(
            '{"stage": "plan_review", "passed": false, '
            '"required_fixes": ["重新拆分子任务"], "summary": "拒绝"}'
        )
        engine = build_engine(tmp_path)

        decision = await engine._call_review_agent(make_review_input())

        assert decision.passed is False
        assert decision.required_fixes == ["重新拆分子任务"]
        assert engine._events.get_events(event_type=EVENT_REVIEW_CONTRACT_REPAIR) == []

    async def test_suggestions_do_not_block_in_engine(self, tmp_path):
        # suggestions 非空不影响放行：passed=true + 建议 → 通过
        install_runner(
            '{"stage": "plan_review", "passed": true, '
            '"issues": ["轻微问题"], "suggestions": ["建议补充用例"], '
            '"required_fixes": [], "summary": "通过但带建议"}'
        )
        engine = build_engine(tmp_path)

        decision = await engine._call_review_agent(make_review_input())

        assert decision.passed is True
        assert decision.suggestions == ["建议补充用例"]
        assert engine._events.get_events(event_type=EVENT_REVIEW_CONTRACT_REPAIR) == []

    async def test_missing_passed_fails_closed_with_audit(self, tmp_path):
        # 缺显式 passed → fail closed + review_contract_error 审计 +
        # contract_valid=False 的 agent_output 事件
        install_runner('{"stage": "plan_review", "summary": "含糊结论"}')
        engine = build_engine(tmp_path)

        decision = await engine._call_review_agent(make_review_input())

        assert decision.passed is False
        assert "评审契约无效" in decision.issues[0]
        assert decision.required_fixes
        assert all("契约无效" in fix for fix in decision.required_fixes)
        errors = engine._events.get_events(event_type=EVENT_REVIEW_CONTRACT_ERROR)
        assert len(errors) == 1
        assert errors[0]["payload"]["stage"] == "plan_review"
        assert "passed" in errors[0]["payload"]["error"]
        outputs = engine._events.get_events(event_type=EVENT_REVIEW_AGENT_OUTPUT)
        assert len(outputs) == 1
        assert outputs[0]["payload"]["contract_valid"] is False
        assert outputs[0]["payload"]["passed"] is False
        # 缺 passed 不属于"矛盾"，不触发契约修复调用
        assert engine._events.get_events(event_type=EVENT_REVIEW_CONTRACT_REPAIR) == []

    async def test_repair_success_exactly_once(self, tmp_path):
        # passed=false + required_fixes 空 → 一次修复调用成功
        _, inputs = install_runner(
            '{"stage": "plan_review", "passed": false, '
            '"issues": ["拆分过粗"], "required_fixes": [], "summary": "拒绝"}',
            '{"stage": "plan_review", "passed": false, '
            '"issues": ["拆分过粗"], "required_fixes": ["重新拆分并补充依赖"], '
            '"summary": "拒绝（修复后）"}',
        )
        engine = build_engine(tmp_path)

        decision = await engine._call_review_agent(make_review_input())

        assert decision.passed is False
        assert decision.required_fixes == ["重新拆分并补充依赖"]
        # 恰好两次 Runner 调用：原始评审 + 一次修复
        assert len(inputs) == 2
        repairs = engine._events.get_events(event_type=EVENT_REVIEW_CONTRACT_REPAIR)
        assert len(repairs) == 1
        payload = repairs[0]["payload"]
        assert payload["attempt"] == 1
        assert payload["final_status"] == "repaired"
        assert "passed=false" in payload["contradiction"]
        assert payload["repair_error"] is None
        assert engine._events.get_events(event_type=EVENT_REVIEW_CONTRACT_ERROR) == []
        outputs = engine._events.get_events(event_type=EVENT_REVIEW_AGENT_OUTPUT)
        assert outputs[0]["payload"]["contract_valid"] is True

    async def test_repair_message_carries_contract_fields(self, tmp_path):
        # 修复消息 = 原始 ReviewInput JSON + contract_repair 附加字段
        # （含矛盾说明与严格模板；只修字段一致性，禁止重新评审）
        _, inputs = install_runner(
            '{"stage": "plan_review", "passed": true, '
            '"required_fixes": ["遗留修复项"], "summary": "通过"}',
            '{"stage": "plan_review", "passed": true, '
            '"required_fixes": [], "suggestions": ["遗留修复项"], "summary": "通过"}',
        )
        engine = build_engine(tmp_path)

        decision = await engine._call_review_agent(make_review_input())

        assert decision.passed is True
        assert decision.required_fixes == []
        assert len(inputs) == 2
        repair_payload = json.loads(inputs[1])
        # 原始评审输入字段保留（评审 Agent 据此定位上下文）
        assert repair_payload["stage"] == "plan_review"
        assert repair_payload["task_request"]["objective"] == "契约测试目标"
        # contract_repair 附加字段
        contract_repair = repair_payload["contract_repair"]
        assert "修复" in contract_repair["instruction"]
        assert "passed=true" in contract_repair["contradiction"]
        assert contract_repair["strict_template"]["passed"] in (True, False)
        assert "required_fixes" in contract_repair["strict_template"]

    async def test_repair_still_contradictory_fails_closed(self, tmp_path):
        # 修复输出仍矛盾（passed=false 仍无 required_fixes）→ fail closed
        _, inputs = install_runner(
            '{"stage": "plan_review", "passed": false, "required_fixes": []}',
            '{"stage": "plan_review", "passed": false, "required_fixes": []}',
        )
        engine = build_engine(tmp_path)

        decision = await engine._call_review_agent(make_review_input())

        assert decision.passed is False
        assert "评审契约无效" in decision.issues[0]
        assert len(inputs) == 2  # 不做第二次修复
        repairs = engine._events.get_events(event_type=EVENT_REVIEW_CONTRACT_REPAIR)
        assert repairs[0]["payload"]["final_status"] == "failed"
        assert repairs[0]["payload"]["repair_error"]
        errors = engine._events.get_events(event_type=EVENT_REVIEW_CONTRACT_ERROR)
        assert len(errors) == 1
        outputs = engine._events.get_events(event_type=EVENT_REVIEW_AGENT_OUTPUT)
        assert outputs[0]["payload"]["contract_valid"] is False

    async def test_repair_output_missing_passed_fails_closed(self, tmp_path):
        # 修复输出缺失显式 passed → fail closed
        install_runner(
            '{"stage": "plan_review", "passed": false, "required_fixes": []}',
            '{"stage": "plan_review", "summary": "修复失败无结论"}',
        )
        engine = build_engine(tmp_path)

        decision = await engine._call_review_agent(make_review_input())

        assert decision.passed is False
        assert "评审契约无效" in decision.issues[0]
        repairs = engine._events.get_events(event_type=EVENT_REVIEW_CONTRACT_REPAIR)
        assert repairs[0]["payload"]["final_status"] == "failed"
        assert "passed" in repairs[0]["payload"]["repair_error"]

    async def test_repair_call_unparseable_fails_closed(self, tmp_path):
        # 修复调用完全无法解析（WorkflowError）→ fail closed
        install_runner(
            '{"stage": "plan_review", "passed": false, "required_fixes": []}',
            "不是 JSON 的自由文本",
        )
        engine = build_engine(tmp_path)

        decision = await engine._call_review_agent(make_review_input())

        assert decision.passed is False
        assert decision.required_fixes
        assert engine._events.get_events(event_type=EVENT_REVIEW_CONTRACT_ERROR)

    async def test_repair_timeout_fails_closed(self, tmp_path):
        # 修复调用超时 → fail closed（TimeoutError 属可恢复失败）
        import asyncio

        install_runner(
            '{"stage": "plan_review", "passed": false, "required_fixes": []}',
            asyncio.TimeoutError(),
        )
        engine = build_engine(tmp_path)

        decision = await engine._call_review_agent(make_review_input())

        assert decision.passed is False
        repairs = engine._events.get_events(event_type=EVENT_REVIEW_CONTRACT_REPAIR)
        assert repairs[0]["payload"]["final_status"] == "failed"
        assert "TimeoutError" in repairs[0]["payload"]["repair_error"]


# ---------------------------------------------------------------------------
# 四、结果评审阶段同样适用
# ---------------------------------------------------------------------------
class TestResultReviewContractApplies:
    """契约校验对结果评审阶段同样生效。"""

    async def test_result_review_missing_passed_fails_closed(self, tmp_path):
        install_runner('{"stage": "result_review", "summary": "整体尚可"}')
        engine = build_engine(tmp_path)

        decision = await engine._call_review_agent(
            make_review_input(stage=ReviewStage.RESULT_REVIEW)
        )

        assert decision.passed is False
        assert decision.stage == "result_review"
        errors = engine._events.get_events(event_type=EVENT_REVIEW_CONTRACT_ERROR)
        assert errors[0]["payload"]["stage"] == "result_review"

    async def test_result_review_repair_success(self, tmp_path):
        # 结果评审矛盾 → 修复成功
        _, inputs = install_runner(
            '{"stage": "result_review", "passed": true, '
            '"required_fixes": ["证据链缺失"], "summary": "通过"}',
            '{"stage": "result_review", "passed": false, '
            '"required_fixes": ["补齐证据链"], "summary": "拒绝（修复后）"}',
        )
        engine = build_engine(tmp_path)

        decision = await engine._call_review_agent(
            make_review_input(stage=ReviewStage.RESULT_REVIEW)
        )

        assert decision.passed is False
        assert decision.required_fixes == ["补齐证据链"]
        assert decision.stage == "result_review"
        assert len(inputs) == 2
        repairs = engine._events.get_events(event_type=EVENT_REVIEW_CONTRACT_REPAIR)
        assert repairs[0]["payload"]["stage"] == "result_review"
        assert repairs[0]["payload"]["final_status"] == "repaired"

    async def test_result_review_valid_reject_no_repair(self, tmp_path):
        install_runner(
            '{"stage": "result_review", "passed": false, '
            '"required_fixes": ["重新生成输出"], "retry_target": "s1"}'
        )
        engine = build_engine(tmp_path)

        decision = await engine._call_review_agent(
            make_review_input(stage=ReviewStage.RESULT_REVIEW)
        )

        assert decision.passed is False
        assert decision.retry_target == "s1"
        assert engine._events.get_events(event_type=EVENT_REVIEW_CONTRACT_REPAIR) == []


# ---------------------------------------------------------------------------
# 五、中文"通过"文本在引擎层不能推导 passed
# ---------------------------------------------------------------------------
class TestChineseKeywordCannotInferInEngine:
    """引擎层：中文"通过"字样绝不构成放行依据。"""

    async def test_plain_text_with_pass_word_conservative_fail(self, tmp_path):
        install_runner("评审结论：通过，可以放行。")
        engine = build_engine(tmp_path)

        decision = await engine._call_review_agent(make_review_input())

        assert decision.passed is False
        # 纯文本无 JSON → review_parse_error 保守路径
        assert engine._events.get_events(event_type=EVENT_REVIEW_PARSE_ERROR)
        assert engine._events.get_events(event_type=EVENT_REVIEW_CONTRACT_ERROR) == []

    async def test_json_with_pass_word_but_no_passed_fails_closed(self, tmp_path):
        # JSON 里 summary 写"通过"但缺 passed 字段 → 契约无效，fail closed
        install_runner('{"stage": "plan_review", "summary": "本次评审通过"}')
        engine = build_engine(tmp_path)

        decision = await engine._call_review_agent(make_review_input())

        assert decision.passed is False
        assert engine._events.get_events(event_type=EVENT_REVIEW_CONTRACT_ERROR)


# ---------------------------------------------------------------------------
# 六、授权评审流程（任务 31）不回归
# ---------------------------------------------------------------------------
class TestAuthorizationReviewNoRegression:
    """授权评审输入映射为 plan_review；契约自洽结论不触发修复。"""

    async def test_authorization_valid_pass_no_repair(self, tmp_path):
        # 契约自洽的授权评审通过：单次调用、结论映射 plan_review
        _, inputs = install_runner(
            '{"stage": "plan_review", "passed": true, '
            '"issues": ["提醒：存在待补发授权"], "suggestions": ["执行前补发授权"], '
            '"required_fixes": [], "summary": "本次为授权最小化评审：通过"}'
        )
        engine = build_engine(tmp_path)

        decision = await engine._call_review_agent(
            make_review_input(stage=ReviewStage.AUTHORIZATION_REVIEW)
        )

        assert decision.passed is True
        assert decision.stage == "plan_review"  # 授权评审映射为 plan_review
        assert len(inputs) == 1  # 契约自洽 → 无修复调用
        assert engine._events.get_events(event_type=EVENT_REVIEW_CONTRACT_REPAIR) == []
        assert engine._events.get_events(event_type=EVENT_REVIEW_CONTRACT_ERROR) == []

    async def test_authorization_valid_reject_with_fixes(self, tmp_path):
        # 契约自洽的授权评审拒绝（超额授权 → required_fixes 非空）
        install_runner(
            '{"stage": "plan_review", "passed": false, '
            '"issues": ["存在超额授权"], '
            '"required_fixes": ["收紧授权至声明的最小集合"], '
            '"summary": "本次为授权最小化评审：拒绝"}'
        )
        engine = build_engine(tmp_path)

        decision = await engine._call_review_agent(
            make_review_input(stage=ReviewStage.AUTHORIZATION_REVIEW)
        )

        assert decision.passed is False
        assert decision.required_fixes == ["收紧授权至声明的最小集合"]
        assert engine._events.get_events(event_type=EVENT_REVIEW_CONTRACT_REPAIR) == []

    async def test_authorization_contradiction_repaired(self, tmp_path):
        # 授权评审矛盾（拒绝却无修复项）→ 修复后契约自洽
        install_runner(
            '{"stage": "plan_review", "passed": false, '
            '"issues": ["存在超额授权"], "required_fixes": [], '
            '"summary": "本次为授权最小化评审：拒绝"}',
            '{"stage": "plan_review", "passed": false, '
            '"issues": ["存在超额授权"], '
            '"required_fixes": ["收紧授权至声明的最小集合"], '
            '"summary": "本次为授权最小化评审：拒绝（修复后）"}',
        )
        engine = build_engine(tmp_path)

        decision = await engine._call_review_agent(
            make_review_input(stage=ReviewStage.AUTHORIZATION_REVIEW)
        )

        assert decision.passed is False
        assert decision.required_fixes
        repairs = engine._events.get_events(event_type=EVENT_REVIEW_CONTRACT_REPAIR)
        assert repairs[0]["payload"]["final_status"] == "repaired"


# ---------------------------------------------------------------------------
# 七、合并层：只读合法 decision.passed；suggestions 合并
# ---------------------------------------------------------------------------
class TestCombineReviewsContract:
    """``PlanningCoordinator._combine_reviews`` 的契约视角。"""

    @staticmethod
    def _decision(passed: bool, **overrides) -> ReviewDecision:
        payload = dict(
            stage="plan_review",
            passed=passed,
            issues=[] if passed else [f"问题（passed={passed}）"],
            suggestions=[],
            required_fixes=[] if passed else [f"修复项（passed={passed}）"],
            summary="合并测试",
        )
        payload.update(overrides)
        return ReviewDecision(**payload)

    def test_both_pass_combines_pass(self):
        combined = PlanningCoordinator._combine_reviews(
            self._decision(True), self._decision(True)
        )
        assert combined.passed is True
        assert combined.required_fixes == []

    def test_any_fail_combines_fail_with_fixes(self):
        combined = PlanningCoordinator._combine_reviews(
            self._decision(True), self._decision(False)
        )
        assert combined.passed is False
        assert combined.required_fixes == ["修复项（passed=False）"]

    def test_suggestions_merged_in_order(self):
        # 两路 suggestions 按序合并且不参与阻断
        combined = PlanningCoordinator._combine_reviews(
            self._decision(True, suggestions=["计划侧建议"]),
            self._decision(True, suggestions=["授权侧建议"]),
        )
        assert combined.passed is True
        assert combined.suggestions == ["计划侧建议", "授权侧建议"]

    def test_pass_with_issues_still_passes(self):
        # 合并只读 passed：issues 非空不构成二次拒绝依据
        combined = PlanningCoordinator._combine_reviews(
            self._decision(True, issues=["提醒：待补发授权"]),
            self._decision(True),
        )
        assert combined.passed is True
        assert combined.issues == ["提醒：待补发授权"]

    def test_fail_with_suggestions_only_from_one_side(self):
        # 一路拒绝（带修复项）+ 一路通过（仅建议）→ 拒绝且修复项保留
        combined = PlanningCoordinator._combine_reviews(
            self._decision(False),
            self._decision(True, suggestions=["授权侧建议"]),
        )
        assert combined.passed is False
        assert combined.required_fixes == ["修复项（passed=False）"]
        assert combined.suggestions == ["授权侧建议"]


# ---------------------------------------------------------------------------
# 八、事件契约字段完整性
# ---------------------------------------------------------------------------
class TestEventPayloadContract:
    """review_agent_output 保留既有字段并新增 contract_valid。"""

    async def test_agent_output_event_fields_preserved(self, tmp_path):
        raw = '{"stage": "plan_review", "passed": true, "summary": "通过"}'
        install_runner(raw)
        engine = build_engine(tmp_path)

        await engine._call_review_agent(make_review_input())

        records = engine._events.get_events(event_type=EVENT_REVIEW_AGENT_OUTPUT)
        assert len(records) == 1
        payload = records[0]["payload"]
        # 既有字段保留：stage / passed / raw_output_preview
        assert payload["stage"] == "plan_review"
        assert payload["passed"] is True
        assert payload["raw_output_preview"] == raw
        # 新增契约字段
        assert payload["contract_valid"] is True

    async def test_repair_event_records_attempt_and_final_status(self, tmp_path):
        install_runner(
            '{"stage": "plan_review", "passed": false, "required_fixes": []}',
            '{"stage": "plan_review", "passed": false, "required_fixes": ["补充修复项"]}',
        )
        engine = build_engine(tmp_path)

        await engine._call_review_agent(make_review_input())

        records = engine._events.get_events(event_type=EVENT_REVIEW_CONTRACT_REPAIR)
        assert len(records) == 1
        payload = records[0]["payload"]
        assert payload["attempt"] == 1
        assert payload["final_status"] == "repaired"
        assert "passed=false" in payload["contradiction"]
        # 修复输出预览有截断上限（避免敏感载荷无限写日志）
        assert len(payload["repair_output_preview"]) <= 500

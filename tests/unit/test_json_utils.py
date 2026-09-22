"""单元测试：围栏容错的纯文本 JSON 输出解析（千问兼容接口回归）。

背景：千问（Qwen）等 OpenAI 兼容接口忽略 ``response_format=json_schema``
约束，把内容正确的 JSON 用 Markdown 代码围栏（```json ... ```）包裹后
以纯文本返回。所有 Agent 的 ``output_type`` 均为 None（纯文本返回），
由调用方剥离围栏后按 Pydantic 契约手动校验。

本文件回归验证该容错链路的每一环：
- ``json_utils``：围栏剥离与 JSON 对象提取；
- ``workflow._coerce_structured_output``：四形态宽容解析；
- ``orchestrator`` 阶段性精简契约（PlanningOutput / ExecutionOutput /
  AggregationOutput）的宽容归一化；
- ``mcp.adapter._output_to_text``：MCP 适配层围栏剥离 + 契约校验；
- ``subagent_creator._coerce_execution_result``：动态子 Agent 输出兜底；
- 三个工具包的 ``as_tool`` 输出提取器。
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

import so_agent.workflow as workflow_mod
from so_agent.json_utils import extract_json_object, strip_code_fences
from so_agent.mcp import adapter as adapter_mod
from so_agent.models import ExecutionResult, ReviewDecision, SubtaskSpec, TaskStatus
from so_agent.orchestrator import (
    AggregationOutput,
    ExecutionOutput,
    OrchestratorOutput,
    PlanningOutput,
)
from so_agent.tool_packages.code_agent import agent as code_mod
from so_agent.tool_packages.review_agent import agent as review_mod
from so_agent.tool_packages.subagent_creator import agent as creator_mod
from so_agent.workflow import WorkflowError


def make_spec(subtask_id: str = "s1") -> SubtaskSpec:
    """构造最小合法 SubtaskSpec（与集成测试桩保持一致）。"""
    return SubtaskSpec(
        subtask_id=subtask_id,
        title=f"模拟子任务 {subtask_id}",
        instructions=f"完成子任务 {subtask_id}",
        role="code",
        allowed_tools=["write_file"],
        max_attempts=3,
    )


def make_decision(*, passed: bool = True) -> ReviewDecision:
    """构造最小合法 ReviewDecision。"""
    return ReviewDecision(
        stage="plan_review",
        passed=passed,
        issues=[] if passed else ["发现的问题"],
        summary="结论（模拟）",
    )


def fence(text: str) -> str:
    """用 ```json 围栏包裹文本（模拟千问兼容接口的返回形态）。"""
    return f"```json\n{text}\n```"


class TestStripCodeFences:
    """strip_code_fences：仅剥离以围栏起始的文本。"""

    def test_json_fenced_text(self):
        assert strip_code_fences('```json\n{"a": 1}\n```') == '{"a": 1}'

    def test_bare_fenced_text(self):
        assert strip_code_fences('```\n{"a": 1}\n```') == '{"a": 1}'

    def test_plain_text_unchanged(self):
        assert strip_code_fences('  {"a": 1}  ') == '{"a": 1}'

    def test_language_tagged_fence(self):
        assert strip_code_fences("```python\nprint(1)\n```") == "print(1)"

    def test_none_becomes_empty(self):
        assert strip_code_fences(None) == ""

    def test_non_string_coerced(self):
        assert strip_code_fences(123) == "123"


class TestExtractJsonObject:
    """extract_json_object：整体解析 → 贪心子串提取。"""

    def test_fenced_json(self):
        assert extract_json_object(fence('{"a": 1}')) == {"a": 1}

    def test_plain_json(self):
        assert extract_json_object('{"a": 1}') == {"a": 1}

    def test_json_with_surrounding_prose(self):
        # 前后带解释文字（非围栏起始）：走贪心 { ... } 提取
        text = '好的，以下是结果：\n{"a": 1}\n以上。'
        assert extract_json_object(text) == {"a": 1}

    def test_invalid_text_returns_none(self):
        assert extract_json_object("没有任何 JSON 的自由文本") is None

    def test_top_level_array_returns_none(self):
        assert extract_json_object("[1, 2, 3]") is None

    def test_empty_text_returns_none(self):
        assert extract_json_object("") is None

    def test_none_returns_none(self):
        assert extract_json_object(None) is None

    def test_unbalanced_braces_return_none(self):
        assert extract_json_object('{"a": 1') is None


class TestCoerceStructuredOutput:
    """控制层四形态宽容解析（含千问围栏行为回归）。"""

    def test_target_model_instance_passthrough(self):
        decision = make_decision()
        assert (
            workflow_mod._coerce_structured_output(
                decision, ReviewDecision, label="评审 Agent"
            )
            is decision
        )

    def test_other_base_model_revalidated(self):
        # 旧全量形态 OrchestratorOutput → PlanningOutput 宽容归一化
        spec = make_spec()
        legacy = OrchestratorOutput(
            task_id="t1", status=TaskStatus.PLANNING, plan_versions=[[spec]]
        )
        output = workflow_mod._coerce_structured_output(
            legacy, PlanningOutput, label="主管 Agent"
        )
        assert output.subtasks == [spec]

    def test_dict_validated_directly(self):
        spec = make_spec()
        output = workflow_mod._coerce_structured_output(
            {"subtasks": [spec.model_dump(mode="json")]},
            PlanningOutput,
            label="主管 Agent",
        )
        assert output.subtasks == [spec]

    def test_fenced_json_text_parsed(self):
        # 千问行为回归：```json 围栏包裹的纯文本 JSON 仍可解析
        spec = make_spec()
        text = fence(
            json.dumps({"subtasks": [spec.model_dump(mode="json")]}, ensure_ascii=False)
        )
        output = workflow_mod._coerce_structured_output(
            text, PlanningOutput, label="主管 Agent"
        )
        assert output.subtasks == [spec]

    def test_unparseable_text_raises_workflow_error(self):
        with pytest.raises(WorkflowError):
            workflow_mod._coerce_structured_output(
                "不是 JSON 的自由文本", PlanningOutput, label="主管 Agent"
            )

    def test_fenced_review_decision_parsed(self):
        # 评审 Agent 围栏输出 → ReviewDecision
        text = fence(make_decision().model_dump_json())
        decision = workflow_mod._coerce_structured_output(
            text, ReviewDecision, label="评审 Agent"
        )
        assert decision.passed is True
        assert decision.stage == "plan_review"


class TestPlanningOutputNormalization:
    """规划阶段精简契约的宽容归一化。"""

    def test_standard_form(self):
        spec = make_spec()
        output = PlanningOutput.model_validate(
            {"subtasks": [spec.model_dump(mode="json")]}
        )
        assert output.subtasks == [spec]

    def test_bare_array_form(self):
        spec = make_spec()
        output = PlanningOutput.model_validate([spec.model_dump(mode="json")])
        assert output.subtasks == [spec]

    def test_legacy_plan_versions_form(self):
        spec = make_spec()
        output = PlanningOutput.model_validate(
            {"plan_versions": [[spec.model_dump(mode="json")]]}
        )
        assert output.subtasks == [spec]

    def test_empty_payload_yields_empty_subtasks(self):
        assert PlanningOutput.model_validate({}).subtasks == []

    def test_empty_plan_versions_yields_empty_subtasks(self):
        assert PlanningOutput.model_validate({"plan_versions": []}).subtasks == []


class TestExecutionOutputNormalization:
    """执行阶段精简契约的宽容归一化。"""

    def test_standard_form(self):
        output = ExecutionOutput.model_validate({"final_result": "结果文本"})
        assert output.final_result == "结果文本"

    def test_unserialized_dict_final_result(self):
        output = ExecutionOutput.model_validate({"final_result": {"a": 1}})
        assert json.loads(output.final_result) == {"a": 1}

    def test_unwrapped_execution_result(self):
        # 模型直接返回 ExecutionResult 结构（未包裹 final_result 字段）
        output = ExecutionOutput.model_validate(
            {"subtask_id": "s1", "success": True, "output": "完成"}
        )
        payload = json.loads(output.final_result)
        assert payload["subtask_id"] == "s1"
        assert payload["success"] is True
        assert payload["output"] == "完成"

    def test_string_final_result_unchanged(self):
        output = ExecutionOutput.model_validate({"final_result": "已是文本"})
        assert output.final_result == "已是文本"


class TestAggregationOutputNormalization:
    """汇总阶段精简契约的宽容归一化。"""

    def test_standard_form(self):
        output = AggregationOutput.model_validate(
            {"final_result": "最终答案", "status": "aggregating"}
        )
        assert output.final_result == "最终答案"
        assert output.status == "aggregating"

    def test_unserialized_dict_final_result(self):
        output = AggregationOutput.model_validate({"final_result": {"a": 1}})
        assert json.loads(output.final_result) == {"a": 1}


class TestAdapterOutputToText:
    """MCP 适配层 _output_to_text：围栏剥离 + output_model 校验。"""

    def test_fenced_text_validated_against_model(self):
        text = fence(make_decision().model_dump_json())
        result = json.loads(adapter_mod._output_to_text(text, ReviewDecision))
        assert result == json.loads(make_decision().model_dump_json())

    def test_plain_text_falls_back_to_output_wrapper(self):
        result = json.loads(adapter_mod._output_to_text("自由文本", ReviewDecision))
        assert result == {"output": "自由文本"}

    def test_invalid_fenced_json_falls_back(self):
        raw = fence("{not valid json}")
        result = json.loads(adapter_mod._output_to_text(raw, ReviewDecision))
        assert result == {"output": raw}

    def test_base_model_serialized_directly(self):
        decision = make_decision()
        result = json.loads(adapter_mod._output_to_text(decision))
        assert result == json.loads(decision.model_dump_json())


class TestCoerceExecutionResult:
    """subagent_creator 动态子 Agent 输出的围栏容错。"""

    def test_fenced_json_text_parsed(self):
        result = ExecutionResult(subtask_id="s1", success=True, output="完成")
        coerced = creator_mod._coerce_execution_result(fence(result.model_dump_json()), "s9")
        assert coerced.subtask_id == "s1"
        assert coerced.success is True
        assert coerced.output == "完成"

    def test_plain_text_falls_back_to_failure(self):
        coerced = creator_mod._coerce_execution_result("自由文本", "s9")
        assert coerced.success is False
        assert coerced.subtask_id == "s9"
        assert coerced.output == "自由文本"
        assert "ExecutionResult" in coerced.error

    def test_instance_passthrough(self):
        result = ExecutionResult(subtask_id="s1", success=True, output="完成")
        assert creator_mod._coerce_execution_result(result, "s9") is result

    def test_empty_subtask_id_filled_with_fallback(self):
        result = ExecutionResult(subtask_id="", success=True, output="完成")
        coerced = creator_mod._coerce_execution_result(result, "s9")
        assert coerced.subtask_id == "s9"


class TestToolPackageExtractors:
    """三个工具包 as_tool 输出提取器的围栏容错。"""

    def _extractors(self):
        return (
            review_mod._extract_output_json,
            code_mod._extract_output_json,
            creator_mod._extract_output_json,
        )

    async def test_extractors_strip_fences(self):
        payload = {"success": True, "summary": "完成"}
        text = fence(json.dumps(payload, ensure_ascii=False))
        for extractor in self._extractors():
            fake = SimpleNamespace(final_output=text)
            assert json.loads(await extractor(fake)) == payload

    async def test_extractors_fall_back_to_output_wrapper(self):
        for extractor in self._extractors():
            fake = SimpleNamespace(final_output="自由文本")
            assert json.loads(await extractor(fake)) == {"output": "自由文本"}

    async def test_extractors_keep_model_dump_json(self):
        decision = make_decision()
        fake = SimpleNamespace(final_output=decision)
        assert json.loads(await review_mod._extract_output_json(fake)) == json.loads(
            decision.model_dump_json()
        )

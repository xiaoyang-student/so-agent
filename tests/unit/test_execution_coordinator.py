"""单元测试：ExecutionCoordinator._parse_execution_result 降级路径防御。

背景：主管 Agent 偶发返回内容为空壳或语义为"待执行/未完成"的 JSON，
降级抢救路径（快速契约校验失败后从内嵌 JSON 提取字段）可能把无效
结果误判为成功并混入最终交付物。两道防御：

- 防御一（output 有效性）：降级判定为成功但 output 为空/长度 < 20
  字符时，翻转为 ``success=False``（error=输出内容过短，疑似无效结果）；
- 防御二（关键词）：output/error 携带"待执行/未完成/pending/
  not started"（大小写不敏感）时强制 ``success=False``。

同时回归验证：完整合法的 ExecutionResult 仍走快速契约路径，不受
防御影响；显式失败结果原样保留错误信息。
"""

from __future__ import annotations

import json

from so_agent.execution_coordinator import ExecutionCoordinator
from so_agent.models import ExecutionResult, SubtaskSpec


def make_spec(subtask_id: str = "s1") -> SubtaskSpec:
    return SubtaskSpec(
        subtask_id=subtask_id,
        title=f"模拟子任务 {subtask_id}",
        instructions=f"完成子任务 {subtask_id}",
        role="code",
        allowed_tools=[],
        max_attempts=3,
    )


def parse(text: str, subtask_id: str = "s1") -> ExecutionResult:
    """直接调用被测静态方法（static，无需构造协调器）。"""
    return ExecutionCoordinator._parse_execution_result(text, make_spec(subtask_id))


# 长度 ≥ 20 的正常产出（不触发防御一，只用于隔离防御二）
_LONG_OUTPUT = "已完成数据分析并生成完整报告，包含结论与建议明细"


class TestPendingKeywordDefense:
    """防御二：output/error 携带未完成标记时强制判失败。"""

    def test_chinese_keyword_in_output_forces_failure(self):
        # 长产出 + "待执行"：只触发关键词防御
        text = json.dumps(
            {"success": True, "output": f"该子任务标记为待执行状态，{_LONG_OUTPUT}"},
            ensure_ascii=False,
        )
        result = parse(text)
        assert result.success is False
        assert "待执行" in result.error

    def test_pending_in_output_forces_failure(self):
        text = json.dumps(
            {
                "success": True,
                "output": f"The task is still pending, {_LONG_OUTPUT}",
            },
            ensure_ascii=False,
        )
        result = parse(text)
        assert result.success is False
        assert "pending" in result.error

    def test_keyword_in_error_forces_failure(self):
        # output 正常但 error 声明未完成：整体强制失败（保留原 error）
        text = json.dumps(
            {
                "success": True,
                "output": _LONG_OUTPUT,
                "error": "子任务未完成：外部依赖尚未就绪",
            },
            ensure_ascii=False,
        )
        result = parse(text)
        assert result.success is False
        assert result.error == "子任务未完成：外部依赖尚未就绪"

    def test_not_started_case_insensitive_forces_failure(self):
        # "Not Started" 大小写不敏感命中 "not started"
        text = json.dumps(
            {
                "success": True,
                "output": _LONG_OUTPUT,
                "error": "Execution Not Started yet",
            },
            ensure_ascii=False,
        )
        result = parse(text)
        assert result.success is False

    def test_inferred_success_with_keyword_forces_failure(self):
        # 无 success 字段（推断成功）+ 关键词：同样强制失败
        text = json.dumps(
            {"output": f"待执行清单如下：{_LONG_OUTPUT}"},
            ensure_ascii=False,
        )
        result = parse(text)
        assert result.success is False


class TestShortOutputDefense:
    """防御一：降级判定成功但产出为空/过短时翻转为失败。"""

    def test_short_output_flips_to_failure(self):
        text = json.dumps({"success": True, "output": "OK"}, ensure_ascii=False)
        result = parse(text)
        assert result.success is False
        assert result.error == "输出内容过短，疑似无效结果"

    def test_missing_output_flips_to_failure(self):
        # 无 output 字段：降级路径用整个候选 JSON 充当 output（空壳仍过短）
        text = json.dumps({"success": True}, ensure_ascii=False)
        result = parse(text)
        assert result.success is False
        assert "过短" in result.error

    def test_boundary_length_output_passes(self):
        # 恰好 20 字符（strip 后）：不触发防御一
        text = json.dumps(
            {"success": True, "output": "一二三四五六七八九十一二三四五六七八九十"},
            ensure_ascii=False,
        )
        result = parse(text)
        assert result.success is True


class TestValidResultsUnaffected:
    """回归：正常结果与显式失败不受防御影响。"""

    def test_valid_long_output_stays_success(self):
        text = json.dumps(
            {"success": True, "output": _LONG_OUTPUT},
            ensure_ascii=False,
        )
        result = parse(text)
        assert result.success is True
        assert result.output == _LONG_OUTPUT

    def test_inferred_success_long_output_stays_success(self):
        # 无 success 无 error：推断成功，产出充足时不误伤
        text = json.dumps({"output": _LONG_OUTPUT}, ensure_ascii=False)
        result = parse(text)
        assert result.success is True

    def test_explicit_failure_keeps_original_error(self):
        text = json.dumps(
            {"success": False, "error": "沙箱执行超时", "output": _LONG_OUTPUT},
            ensure_ascii=False,
        )
        result = parse(text)
        assert result.success is False
        assert result.error == "沙箱执行超时"

    def test_fast_path_contract_ignores_defenses(self):
        # 完整合法契约（含 subtask_id）：走快速路径，短 output 也不误伤
        payload = ExecutionResult(subtask_id="s1", success=True, output="42")
        result = parse(payload.model_dump_json())
        assert result.success is True
        assert result.subtask_id == "s1"
        assert result.output == "42"

    def test_unparseable_text_still_fails_gracefully(self):
        result = parse("这不是 JSON，也不是任何可抢救的结构")
        assert result.success is False
        assert "未能从主管 Agent 输出解析" in result.error

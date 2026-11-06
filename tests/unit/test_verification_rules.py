"""可机检验收规则：文件 / JSON / 数值 / 文本 / 退出码。"""

from __future__ import annotations

import json
from pathlib import Path

from so_agent.models import ExecutionResult, SubtaskSpec, VerificationRule
from so_agent.runtime.verification import (
    evaluate_rule,
    evaluate_verification_rules,
)


def make_result(**overrides) -> ExecutionResult:
    data = {
        "subtask_id": "s1",
        "success": True,
        "output": "",
        "artifacts": [],
        "metadata": {},
    }
    data.update(overrides)
    return ExecutionResult(**data)


def test_file_exists_and_non_empty(tmp_path: Path):
    (tmp_path / "a.txt").write_text("x", encoding="utf-8")
    (tmp_path / "empty.txt").write_text("", encoding="utf-8")
    result = make_result()
    assert evaluate_rule(VerificationRule(kind="file_exists", path="a.txt"), result, tmp_path) is None
    assert evaluate_rule(VerificationRule(kind="file_non_empty", path="a.txt"), result, tmp_path) is None
    assert "不存在" in evaluate_rule(
        VerificationRule(kind="file_exists", path="missing.txt"), result, tmp_path
    )
    assert "为空" in evaluate_rule(
        VerificationRule(kind="file_non_empty", path="empty.txt"), result, tmp_path
    )


def test_file_rules_reject_escape(tmp_path: Path):
    outside = tmp_path.parent / "outside.txt"
    outside.write_text("secret", encoding="utf-8")
    failure = evaluate_rule(
        VerificationRule(kind="file_exists", path=str(outside)), make_result(), tmp_path
    )
    assert "路径非法" in failure


def test_json_field_from_file_and_equals(tmp_path: Path):
    (tmp_path / "metrics.json").write_text(
        json.dumps({"totals": {"count": 3, "name": "demo"}}), encoding="utf-8"
    )
    result = make_result()
    assert evaluate_rule(
        VerificationRule(kind="json_field", path="metrics.json", field="totals.count"),
        result,
        tmp_path,
    ) is None
    failure = evaluate_rule(
        VerificationRule(
            kind="json_field", path="metrics.json", field="totals.count", equals=4
        ),
        result,
        tmp_path,
    )
    assert "期望 4" in failure
    assert "字段不存在" in evaluate_rule(
        VerificationRule(kind="json_field", path="metrics.json", field="totals.missing"),
        result,
        tmp_path,
    )


def test_json_field_from_output(tmp_path: Path):
    result = make_result(output=json.dumps({"status": "ok"}))
    assert evaluate_rule(
        VerificationRule(kind="json_field", field="status", equals="ok"),
        result,
        tmp_path,
    ) is None


def test_json_field_extracts_object_from_surrounding_text(tmp_path: Path):
    result = make_result(output="核对完成：{\"consistent\": true, \"n\": 2}")
    assert evaluate_rule(
        VerificationRule(kind="json_field", field="consistent", equals=True),
        result,
        tmp_path,
    ) is None


def test_json_field_falls_back_to_execution_output(tmp_path: Path):
    result = make_result(
        output="脚本已运行并给出 JSON 结果",
        metadata={"execution_output": json.dumps({"consistent": True})},
    )
    assert evaluate_rule(
        VerificationRule(kind="json_field", field="consistent", equals=True),
        result,
        tmp_path,
    ) is None


def test_json_field_rejects_non_json_text(tmp_path: Path):
    result = make_result(output="只有文字结论，没有结构化结果")
    assert "不是合法 JSON" in evaluate_rule(
        VerificationRule(kind="json_field", field="consistent"),
        result,
        tmp_path,
    )


def test_numeric_match_with_tolerance(tmp_path: Path):
    result = make_result(output=json.dumps({"revenue": 100.0000001}))
    assert evaluate_rule(
        VerificationRule(kind="numeric_match", field="revenue", value=100.0),
        result,
        tmp_path,
    ) is None
    failure = evaluate_rule(
        VerificationRule(kind="numeric_match", field="revenue", value=99.0),
        result,
        tmp_path,
    )
    assert "期望 99.0" in failure


def test_text_contains_requires_artifact_path(tmp_path: Path):
    (tmp_path / "report.md").write_text("销售总额：100", encoding="utf-8")
    result = make_result(output="结论：完成")
    # 无 path 的文本规则只可能匹配模型措辞：按失败关闭处理
    failure = evaluate_rule(
        VerificationRule(kind="text_contains", text="完成"), result, tmp_path
    )
    assert "必须指定 path" in failure
    assert evaluate_rule(
        VerificationRule(kind="text_contains", path="report.md", text="销售总额"),
        result,
        tmp_path,
    ) is None
    assert "未找到" in evaluate_rule(
        VerificationRule(kind="text_contains", path="report.md", text="失败"),
        result,
        tmp_path,
    )


def test_exit_code_from_metadata_and_output(tmp_path: Path):
    result = make_result(metadata={"exit_code": 0})
    assert evaluate_rule(VerificationRule(kind="exit_code"), result, tmp_path) is None
    assert "期望 1" in evaluate_rule(
        VerificationRule(kind="exit_code", value=1), result, tmp_path
    )
    nested = make_result(output=json.dumps({"execution_result": {"exit_code": 0}}))
    assert evaluate_rule(VerificationRule(kind="exit_code"), nested, tmp_path) is None
    assert "缺少退出码证据" in evaluate_rule(
        VerificationRule(kind="exit_code"), make_result(), tmp_path
    )


def test_exit_code_inside_execution_metadata(tmp_path: Path):
    """run_in_sandbox 把退出码放在 execution_result.metadata 内。"""
    result = make_result(
        metadata={"execution_result": {"metadata": {"exit_code": 0}}}
    )
    assert evaluate_rule(VerificationRule(kind="exit_code"), result, tmp_path) is None


def test_exit_code_unwraps_nested_output_json(tmp_path: Path):
    inner = json.dumps(
        {"execution_result": {"success": True, "metadata": {"exit_code": 0}}}
    )
    result = make_result(output=json.dumps({"output": inner}))
    assert evaluate_rule(VerificationRule(kind="exit_code"), result, tmp_path) is None


def test_evaluate_all_rules_collects_failures(tmp_path: Path):
    (tmp_path / "a.txt").write_text("hello", encoding="utf-8")
    spec = SubtaskSpec(
        subtask_id="s1",
        title="t",
        instructions="i",
        role="code",
        verification=[
            VerificationRule(kind="file_non_empty", path="a.txt"),
            VerificationRule(kind="text_contains", text="missing"),
        ],
    )
    failures = evaluate_verification_rules(spec, make_result(), tmp_path)
    assert len(failures) == 1
    assert "text_contains" in failures[0]

"""代码 Agent 后置校验：自述成功必须与文件系统/执行结果一致。"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from so_agent.tool_packages.code_agent.agent import (
    CodeAgentOutput,
    _apply_output_governance,
    _apply_postcondition,
    _load_prompt,
    _make_postcondition_validator,
)


BUDGETS = {
    "summary": 20,
    "action": 10,
    "evidence_item": 8,
    "evidence_items": 2,
}


def _success_output(**overrides) -> dict:
    data = {
        "success": True,
        "action_performed": "create file",
        "file_path": "out/result.txt",
        "code_diff": "diff",
        "execution_result": None,
        "error": None,
        "summary": "done",
    }
    data.update(overrides)
    return data


def test_create_success_requires_existing_non_empty_file(tmp_path: Path):
    target = tmp_path / "out" / "result.txt"
    target.parent.mkdir(parents=True)
    target.write_text("real content", encoding="utf-8")

    validated, failure = _apply_postcondition(
        _success_output(), {"action": "create", "file_path": "out/result.txt"}, tmp_path
    )
    assert failure is None
    assert validated["success"] is True


def test_create_claimed_file_missing_flips_to_failure(tmp_path: Path):
    validated, failure = _apply_postcondition(
        _success_output(), {"action": "create", "file_path": "out/result.txt"}, tmp_path
    )
    assert failure is not None
    assert "不存在" in failure
    assert validated["success"] is False
    assert "后置校验失败" in validated["error"]
    assert validated["summary"].startswith("done")


def test_create_empty_file_flips_to_failure(tmp_path: Path):
    target = tmp_path / "empty.txt"
    target.write_text("", encoding="utf-8")
    validated, failure = _apply_postcondition(
        _success_output(file_path="empty.txt"),
        {"action": "modify", "file_path": "empty.txt"},
        tmp_path,
    )
    assert failure is not None
    assert "为空" in failure
    assert validated["success"] is False


def test_create_without_file_path_flips_to_failure(tmp_path: Path):
    validated, failure = _apply_postcondition(
        _success_output(file_path=None),
        {"action": "create", "file_path": None},
        tmp_path,
    )
    assert failure is not None
    assert "未声明 file_path" in failure
    assert validated["success"] is False


def test_path_escape_is_rejected(tmp_path: Path):
    validated, failure = _apply_postcondition(
        _success_output(file_path="../outside.txt"),
        {"action": "create", "file_path": "../outside.txt"},
        tmp_path,
    )
    assert failure is not None
    assert "非法" in failure
    assert validated["success"] is False


def test_execute_requires_execution_result(tmp_path: Path):
    validated, failure = _apply_postcondition(
        _success_output(action_performed="run", execution_result=None),
        {"action": "execute"},
        tmp_path,
    )
    assert failure is not None
    assert "缺少 execution_result" in failure
    assert validated["success"] is False


def test_execute_with_failed_nested_result_flips(tmp_path: Path):
    validated, failure = _apply_postcondition(
        _success_output(
            execution_result={"success": False, "error": "exit code 1"}
        ),
        {"action": "execute"},
        tmp_path,
    )
    assert failure is not None
    assert "exit code 1" in failure
    assert validated["success"] is False


def test_execute_with_successful_nested_result_passes(tmp_path: Path):
    validated, failure = _apply_postcondition(
        _success_output(
            execution_result={"success": True, "output": "ok"}
        ),
        {"action": "execute"},
        tmp_path,
    )
    assert failure is None
    assert validated["success"] is True


def test_review_action_is_not_file_checked(tmp_path: Path):
    validated, failure = _apply_postcondition(
        _success_output(file_path="not-written.txt"),
        {"action": "review"},
        tmp_path,
    )
    assert failure is None
    assert validated["success"] is True


def test_auto_action_accepts_multiple_reviewed_files(tmp_path: Path):
    (tmp_path / "report.md").write_text("report", encoding="utf-8")
    (tmp_path / "summary.json").write_text("{}", encoding="utf-8")
    validated, failure = _apply_postcondition(
        _success_output(
            file_path="report.md, summary.json",
            action_performed="review",
        ),
        {"action": "auto"},
        tmp_path,
    )
    assert failure is None
    assert validated["success"] is True


def test_auto_action_still_flags_missing_file_in_list(tmp_path: Path):
    (tmp_path / "report.md").write_text("report", encoding="utf-8")
    validated, failure = _apply_postcondition(
        _success_output(file_path="report.md、missing.json"),
        {"action": "auto"},
        tmp_path,
    )
    assert failure is not None
    assert "missing.json" in failure
    assert validated["success"] is False


def test_base_model_output_is_flipped_via_dict(tmp_path: Path):
    output = CodeAgentOutput(
        success=True,
        action_performed="create",
        file_path="missing.txt",
        summary="claimed",
    )
    validated, failure = _apply_postcondition(
        output, {"action": "create", "file_path": "missing.txt"}, tmp_path
    )
    assert failure is not None
    assert isinstance(validated, dict)
    assert validated["success"] is False


def test_unparseable_output_is_left_untouched(tmp_path: Path):
    raw = "not json at all"
    validated, failure = _apply_postcondition(
        raw, {"action": "create", "file_path": "x.txt"}, tmp_path
    )
    assert validated == raw
    assert failure is None


def test_validator_disabled_returns_none(tmp_path: Path):
    context = SimpleNamespace(event_logger=None)
    settings = SimpleNamespace(
        code_agent_postcondition_enabled=False,
        code_agent_suppress_diff_echo=False,
    )
    assert _make_postcondition_validator(context, tmp_path, settings) is None


def test_validator_logs_event_on_flip(tmp_path: Path):
    events: list[tuple[str, dict]] = []
    context = SimpleNamespace(
        event_logger=SimpleNamespace(
            log_event=lambda event_type, payload: events.append(
                (event_type, payload)
            )
        )
    )
    settings = SimpleNamespace(code_agent_postcondition_enabled=True)
    validator = _make_postcondition_validator(context, tmp_path, settings)
    assert validator is not None
    validated = validator(
        _success_output(), {"action": "create", "file_path": "out.txt"}
    )
    assert validated["success"] is False
    assert events
    assert events[0][0] == "code_agent_postcondition_failed"
    assert events[0][1]["action"] == "create"


def test_output_governance_compacts_diff_echo(tmp_path: Path):
    target = tmp_path / "out.txt"
    target.write_text("hello world", encoding="utf-8")
    diff = "--- a/out.txt\n+++ b/out.txt\n@@ -1 +1 @@\n-old\n+hello world\n"
    governed = _apply_output_governance(
        _success_output(file_path="out.txt", code_diff=diff),
        {"action": "create", "file_path": "out.txt"},
        tmp_path,
        suppress_diff=True,
    )
    assert "[已省略 diff 回显]" in governed["code_diff"]
    assert "+1/-1" in governed["code_diff"]
    assert "11 字节" in governed["code_diff"]
    assert "hello world" not in governed["code_diff"]


def test_output_governance_keeps_diff_when_disabled(tmp_path: Path):
    output = _success_output(code_diff="--- a\n+++ b\n-old\n+new\n")
    governed = _apply_output_governance(
        output, {"action": "create", "file_path": "x.txt"}, tmp_path, suppress_diff=False
    )
    assert governed is output
    assert governed["code_diff"].startswith("--- a")


def test_output_governance_trims_narrative_fields(tmp_path: Path):
    output = _success_output(
        action_performed="执行了很长的动作描述" * 5,
        summary="完成情况" * 50,
        evidence=["证据一" * 10, "证据二", "证据三", "证据四"],
        code_diff="",
    )
    governed = _apply_output_governance(
        output,
        {"action": "execute"},
        tmp_path,
        suppress_diff=True,
        budgets=BUDGETS,
    )
    assert len(governed["summary"]) <= BUDGETS["summary"] + len("…（已截断）")
    assert governed["summary"].endswith("…（已截断）")
    assert len(governed["action_performed"]) <= BUDGETS["action"] + len(
        "…（已截断）"
    )
    assert governed["evidence"] == [
        ("证据一" * 10)[:8] + "…（已截断）",
        "证据二",
    ]
    assert output["summary"].startswith("完成情况")  # 原对象不被就地修改


def test_output_governance_keeps_identity_when_within_budget(tmp_path: Path):
    output = _success_output(summary="done", action_performed="create", code_diff="")
    governed = _apply_output_governance(
        output,
        {"action": "create", "file_path": "out/result.txt"},
        tmp_path,
        suppress_diff=True,
        budgets=BUDGETS,
    )
    assert governed is output


def test_prompt_contains_configured_output_budgets():
    settings = SimpleNamespace(
        code_agent_summary_max_chars=123,
        code_agent_action_max_chars=45,
        code_agent_evidence_item_max_chars=67,
        code_agent_evidence_max_items=3,
    )
    prompt = _load_prompt(settings)
    assert "`summary` ≤123 字符" in prompt
    assert "`action_performed` ≤45 字符" in prompt
    assert "最多 ≤3 条" in prompt
    assert "每条 ≤67 字符" in prompt


def test_validator_trims_output_and_reports_no_identity(tmp_path: Path):
    target = tmp_path / "out.txt"
    target.write_text("hello", encoding="utf-8")
    context = SimpleNamespace(event_logger=None)
    settings = SimpleNamespace(
        code_agent_postcondition_enabled=True,
        code_agent_suppress_diff_echo=True,
        code_agent_summary_max_chars=5,
        code_agent_action_max_chars=0,
        code_agent_evidence_item_max_chars=0,
        code_agent_evidence_max_items=0,
    )
    validator = _make_postcondition_validator(context, tmp_path, settings)
    validated = validator(
        _success_output(file_path="out.txt", summary="很长的总结内容"),
        {"action": "create", "file_path": "out.txt"},
    )
    assert validated["success"] is True
    assert validated["summary"].endswith("…（已截断）")

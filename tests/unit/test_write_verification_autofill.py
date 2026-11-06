"""控制层为写入子任务补齐最小机检验收规则。"""

from __future__ import annotations

from types import SimpleNamespace

from so_agent.config import Settings
from so_agent.context import ProjectContext
from so_agent.models import SubtaskSpec, TaskRequest, VerificationRule
from so_agent.planning_coordinator import PlanningCoordinator
from so_agent.runtime.events import EVENT_VERIFICATION_RULE_REJECTED, EventLogger
from so_agent.state_machine import WorkflowState


def _spec(write_scope: list[str], verification: list[VerificationRule] | None = None):
    return SubtaskSpec(
        subtask_id="s1",
        title="写文件",
        instructions="写入结果",
        role="code",
        write_scope=list(write_scope),
        verification=list(verification or []),
    )


def test_file_scope_gets_exists_and_non_empty_rules():
    spec = _spec(["out/a.md", "data/"])
    PlanningCoordinator._ensure_write_verification(spec)
    rules = {(rule.kind, rule.path) for rule in spec.verification}
    assert ("file_exists", "out/a.md") in rules
    assert ("file_non_empty", "out/a.md") in rules
    # 目录只要求存在，不要求是文件
    assert ("file_exists", "data/") in rules
    assert ("file_non_empty", "data/") not in rules


def test_glob_scope_is_not_expanded():
    spec = _spec(["out/*.md"])
    PlanningCoordinator._ensure_write_verification(spec)
    assert spec.verification == []


def test_read_only_subtask_is_untouched():
    spec = _spec([])
    PlanningCoordinator._ensure_write_verification(spec)
    assert spec.verification == []


def test_existing_rules_are_not_duplicated():
    spec = _spec(
        ["out/a.md"],
        verification=[VerificationRule(kind="file_exists", path="out/a.md")],
    )
    PlanningCoordinator._ensure_write_verification(spec)
    assert len(
        [rule for rule in spec.verification if rule.kind == "file_exists"]
    ) == 1
    assert any(rule.kind == "file_non_empty" for rule in spec.verification)


def _lint_coordinator(tmp_path):
    """构造仅承载事件与任务信息的最小协调器（不启动工作流）。"""
    context = ProjectContext(sandbox_dir=tmp_path, project_name="rule-lint")
    events = EventLogger(context=context)
    coordinator = object.__new__(PlanningCoordinator)
    coordinator._deps = SimpleNamespace(
        events=events, settings=Settings(_env_file=None)
    )
    coordinator._state = WorkflowState(events=events)
    coordinator._state.plan_version = 1
    coordinator._task_request = TaskRequest(task_id="t-lint", objective="规则 lint")
    return coordinator, events


def test_lint_drops_text_contains_without_path(tmp_path):
    coordinator, events = _lint_coordinator(tmp_path)
    spec = _spec(
        [],
        verification=[
            VerificationRule(kind="text_contains", text="execution_result"),
            VerificationRule(kind="text_contains", path="out/a.md", text="结论"),
        ],
    )
    coordinator._lint_verification_rules(spec)
    assert [rule.path for rule in spec.verification] == ["out/a.md"]
    rejected = [
        event
        for event in events.events
        if event["event_type"] == EVENT_VERIFICATION_RULE_REJECTED
    ]
    assert len(rejected) == 1
    assert rejected[0]["payload"]["rejected"][0]["reason"] == (
        "text_contains_without_path"
    )
    assert rejected[0]["payload"]["subtask_id"] == "s1"


def test_lint_drops_structurally_incomplete_rules_and_duplicates(tmp_path):
    coordinator, _ = _lint_coordinator(tmp_path)
    spec = _spec(
        [],
        verification=[
            VerificationRule(kind="file_exists"),
            VerificationRule(kind="json_field", field=None),
            VerificationRule(kind="numeric_match", field="rate"),
            VerificationRule(kind="file_non_empty", path="out/a.md"),
            VerificationRule(kind="file_non_empty", path="out/a.md"),
        ],
    )
    coordinator._lint_verification_rules(spec)
    assert [(rule.kind, rule.path) for rule in spec.verification] == [
        ("file_non_empty", "out/a.md")
    ]


def test_adopt_plan_applies_lint_before_write_autofill(tmp_path):
    coordinator, _ = _lint_coordinator(tmp_path)
    spec = _spec(
        ["out/a.md"],
        verification=[VerificationRule(kind="text_contains", text="完成")],
    )
    adopted = coordinator._adopt_plan([spec])
    kinds = {rule.kind for rule in adopted[0].verification}
    assert "text_contains" not in kinds
    assert {"file_exists", "file_non_empty"} <= kinds

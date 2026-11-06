"""简单计划确定性评审跳过条件。"""

from __future__ import annotations

from types import SimpleNamespace

from so_agent.models import SubtaskSpec, VerificationRule
from so_agent.planning_coordinator import PlanningCoordinator
from so_agent.runtime.tool_registry import ToolRegistry


def _spec(subtask_id: str, *, tools=None, write_scope=None, verification=None):
    return SubtaskSpec(
        subtask_id=subtask_id,
        title=subtask_id,
        instructions="执行",
        role="code",
        dependencies=[],
        allowed_tools=list(tools or []),
        write_scope=list(write_scope or []),
        verification=list(verification or []),
    )


def _coordinator(
    subtasks, *, plan_version=1, replan_count=0, enabled=True, steps_merged=0
):
    return SimpleNamespace(
        _settings=SimpleNamespace(skip_plan_review_for_simple=enabled),
        _replan_count=replan_count,
        _plan_version=plan_version,
        _subtasks=subtasks,
        _tool_registry=ToolRegistry(),
        _state=SimpleNamespace(plan_steps_merged=steps_merged),
    )


def test_simple_plan_with_verification_and_standard_tools_is_skipped():
    coordinator = _coordinator(
        [
            _spec(
                "s1",
                tools=["read_file", "write_file"],
                write_scope=["out/a.md"],
                verification=[VerificationRule(kind="file_exists", path="out/a.md")],
            )
        ]
    )
    assert PlanningCoordinator._should_skip_plan_review(coordinator) is True


def test_replanning_never_skips_review():
    coordinator = _coordinator([_spec("s1")], plan_version=2, replan_count=1)
    assert PlanningCoordinator._should_skip_plan_review(coordinator) is False


def test_dynamic_tool_gap_requires_review():
    coordinator = _coordinator([_spec("s1", tools=["custom_scraper"])])
    assert PlanningCoordinator._should_skip_plan_review(coordinator) is False


def test_write_without_verification_requires_review():
    coordinator = _coordinator([_spec("s1", write_scope=["out/a.md"])])
    assert PlanningCoordinator._should_skip_plan_review(coordinator) is False


def test_more_than_two_subtasks_requires_review():
    coordinator = _coordinator([_spec("s1"), _spec("s2"), _spec("s3")])
    assert PlanningCoordinator._should_skip_plan_review(coordinator) is False


def test_duplicate_write_scope_requires_review():
    coordinator = _coordinator(
        [
            _spec(
                "s1",
                write_scope=["out/a.md"],
                verification=[VerificationRule(kind="file_exists", path="out/a.md")],
            ),
            _spec(
                "s2",
                write_scope=["out/a.md"],
                verification=[VerificationRule(kind="file_exists", path="out/a.md")],
            ),
        ]
    )
    assert PlanningCoordinator._should_skip_plan_review(coordinator) is False


def test_disabled_by_configuration():
    coordinator = _coordinator([_spec("s1")], enabled=False)
    assert PlanningCoordinator._should_skip_plan_review(coordinator) is False


def test_merged_steps_never_skip_review():
    coordinator = _coordinator(
        [
            _spec(
                "s1",
                write_scope=["out/a.md"],
                verification=[VerificationRule(kind="file_exists", path="out/a.md")],
            )
        ],
        steps_merged=1,
    )
    assert PlanningCoordinator._should_skip_plan_review(coordinator) is False

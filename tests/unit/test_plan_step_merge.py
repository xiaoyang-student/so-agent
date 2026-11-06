"""计划内步骤合并：产生产物与写报告的两步合成一次执行。"""

from __future__ import annotations

from types import SimpleNamespace

from so_agent.config import Settings
from so_agent.context import ProjectContext
from so_agent.models import SubtaskSpec, TaskRequest, VerificationRule
from so_agent.planning_coordinator import PlanningCoordinator
from so_agent.runtime.events import EVENT_PLAN_STEPS_MERGED, EventLogger
from so_agent.state_machine import WorkflowState


def _spec(
    subtask_id: str,
    *,
    instructions: str,
    role: str = "data_engineer",
    dependencies: list[str] | None = None,
    write_scope: list[str] | None = None,
    allowed_tools: list[str] | None = None,
    expected_output: str = "",
    reuse_from: str | None = None,
) -> SubtaskSpec:
    return SubtaskSpec(
        subtask_id=subtask_id,
        title=f"{subtask_id} 标题",
        instructions=instructions,
        role=role,
        dependencies=list(dependencies or []),
        allowed_tools=list(allowed_tools if allowed_tools is not None else [
            "read_file",
            "write_file",
            "run_in_sandbox",
        ]),
        write_scope=list(write_scope or []),
        expected_output=expected_output,
        reuse_from=reuse_from,
    )


def _producer() -> SubtaskSpec:
    return _spec(
        "st1_analyze",
        instructions="生成 analyze_incident.py 并运行，输出 metrics.json",
        write_scope=["analyze_incident.py", "metrics.json"],
        expected_output="metrics.json 含 6 个指标字段",
    )


def _consumer() -> SubtaskSpec:
    return _spec(
        "st2_report",
        instructions="读取 metrics.json，撰写 incident_report.md 事故分析报告",
        role="sre_report_writer",
        dependencies=["st1_analyze"],
        write_scope=["incident_report.md"],
        expected_output="非空 incident_report.md，含关键指标与建议",
    )


def _coordinator(tmp_path, **settings) -> tuple[PlanningCoordinator, EventLogger]:
    events = EventLogger(
        context=ProjectContext(sandbox_dir=tmp_path, project_name="step-merge")
    )
    coordinator = object.__new__(PlanningCoordinator)
    coordinator._deps = SimpleNamespace(
        events=events,
        settings=Settings(_env_file=None, **settings),
        tool_registry=SimpleNamespace(get_tool=lambda name: None),
    )
    coordinator._state = WorkflowState(events=events)
    coordinator._state.plan_version = 1
    coordinator._task_request = TaskRequest(task_id="t-merge", objective="合并步骤")
    return coordinator, events


def _merge_events(events: EventLogger) -> list[dict]:
    return [
        event
        for event in events.events
        if event["event_type"] == EVENT_PLAN_STEPS_MERGED
    ]


class TestMergeEligibility:
    def test_producer_and_report_step_are_merged(self, tmp_path):
        coordinator, events = _coordinator(tmp_path)
        adopted = coordinator._adopt_plan([_producer(), _consumer()])

        assert [spec.subtask_id for spec in adopted] == ["st1_analyze"]
        merged = adopted[0]
        assert "撰写 incident_report.md" in merged.instructions
        assert "生成 analyze_incident.py" in merged.instructions
        assert "无需重新生成" in merged.instructions
        assert merged.write_scope == [
            "analyze_incident.py",
            "metrics.json",
            "incident_report.md",
        ]
        assert "关键指标与建议" in merged.expected_output
        assert coordinator._state.plan_steps_merged == 1
        assert coordinator._state.merged_subtask_ids == {"st1_analyze"}

        merged_events = _merge_events(events)
        assert len(merged_events) == 1
        payload = merged_events[0]["payload"]
        assert payload["merged_subtask_id"] == "st1_analyze"
        assert payload["absorbed_subtask_id"] == "st2_report"
        assert payload["artifacts"] == ["analyze_incident.py", "metrics.json"]
        assert payload["plan_version"] == 1

    def test_merged_step_keeps_write_verification_for_report(self, tmp_path):
        coordinator, _ = _coordinator(tmp_path)
        adopted = coordinator._adopt_plan([_producer(), _consumer()])

        rules = {(rule.kind, rule.path) for rule in adopted[0].verification}
        assert ("file_exists", "incident_report.md") in rules
        assert ("file_non_empty", "incident_report.md") in rules
        assert ("file_exists", "metrics.json") in rules

    def test_multiple_dependencies_are_not_merged(self, tmp_path):
        coordinator, events = _coordinator(tmp_path)
        other = _spec(
            "st0_source",
            instructions="准备 access.log",
            write_scope=["access.log"],
        )
        consumer = _consumer()
        consumer.dependencies = ["st1_analyze", "st0_source"]
        adopted = coordinator._adopt_plan([other, _producer(), consumer])

        assert [spec.subtask_id for spec in adopted] == [
            "st0_source",
            "st1_analyze",
            "st2_report",
        ]
        assert _merge_events(events) == []
        assert coordinator._state.plan_steps_merged == 0

    def test_non_report_write_scope_is_not_merged(self, tmp_path):
        coordinator, events = _coordinator(tmp_path)
        consumer = _consumer()
        consumer.write_scope = ["metrics_summary.json"]
        consumer.instructions = "读取 metrics.json 生成 metrics_summary.json"
        adopted = coordinator._adopt_plan([_producer(), consumer])

        assert len(adopted) == 2
        assert _merge_events(events) == []

    def test_unreferenced_artifact_is_not_merged(self, tmp_path):
        coordinator, events = _coordinator(tmp_path)
        consumer = _consumer()
        consumer.instructions = "撰写 incident_report.md 事故分析报告"
        consumer.expected_output = "非空 incident_report.md"
        adopted = coordinator._adopt_plan([_producer(), consumer])

        assert len(adopted) == 2
        assert _merge_events(events) == []

    def test_reuse_declaration_blocks_merge(self, tmp_path):
        coordinator, events = _coordinator(tmp_path)
        consumer = _consumer()
        consumer.reuse_from = "old_report"
        adopted = coordinator._adopt_plan([_producer(), consumer])

        assert len(adopted) == 2
        assert _merge_events(events) == []

    def test_reuse_target_is_not_merged(self, tmp_path):
        coordinator, events = _coordinator(tmp_path)
        downstream = _spec(
            "st3_index",
            instructions="汇总 incident_report.md 到 index.txt",
            dependencies=["st1_analyze"],
            write_scope=["index.txt"],
            reuse_from="st2_report",
        )
        adopted = coordinator._adopt_plan([_producer(), _consumer(), downstream])

        assert [spec.subtask_id for spec in adopted] == [
            "st1_analyze",
            "st2_report",
            "st3_index",
        ]
        assert _merge_events(events) == []

    def test_cross_executor_family_is_not_merged(self, tmp_path):
        coordinator, events = _coordinator(tmp_path)
        producer = _producer()
        producer.role = "code"
        adopted = coordinator._adopt_plan([producer, _consumer()])

        assert [spec.subtask_id for spec in adopted] == [
            "st1_analyze",
            "st2_report",
        ]
        assert _merge_events(events) == []

    def test_producer_without_write_scope_is_not_merged(self, tmp_path):
        coordinator, events = _coordinator(tmp_path)
        producer = _producer()
        producer.write_scope = []
        adopted = coordinator._adopt_plan([producer, _consumer()])

        assert len(adopted) == 2
        assert _merge_events(events) == []

    def test_pairs_cap_zero_disables_merge(self, tmp_path):
        coordinator, events = _coordinator(tmp_path, plan_merge_report_max_pairs=0)
        adopted = coordinator._adopt_plan([_producer(), _consumer()])

        assert len(adopted) == 2
        assert _merge_events(events) == []
        assert coordinator._state.plan_steps_merged == 0
        assert coordinator._state.merged_subtask_ids == set()

    def test_pairs_cap_limits_merges(self, tmp_path):
        coordinator, events = _coordinator(tmp_path, plan_merge_report_max_pairs=1)
        first = _producer()
        second = _spec(
            "st3_verify",
            instructions="运行 verify_metrics.py 校验 metrics.json",
            write_scope=["metrics.json", "verify_metrics.py"],
        )
        follow_up = _spec(
            "st4_summary",
            instructions="读取 metrics.json 写 summary.md",
            role="sre_report_writer",
            dependencies=["st3_verify"],
            write_scope=["summary.md"],
        )
        adopted = coordinator._adopt_plan([first, _consumer(), second, follow_up])

        assert [spec.subtask_id for spec in adopted] == [
            "st1_analyze",
            "st3_verify",
            "st4_summary",
        ]
        assert len(_merge_events(events)) == 1
        assert coordinator._state.plan_steps_merged == 1


class TestDependencyRewrite:
    def test_downstream_dependency_is_rewritten_to_producer(self, tmp_path):
        coordinator, _ = _coordinator(tmp_path)
        downstream = _spec(
            "st3_index",
            instructions="汇总 incident_report.md 到 index.txt",
            dependencies=["st2_report"],
            write_scope=["index.txt"],
        )
        adopted = coordinator._adopt_plan([_producer(), _consumer(), downstream])

        by_id = {spec.subtask_id: spec for spec in adopted}
        assert "st2_report" not in by_id
        assert by_id["st3_index"].dependencies == ["st1_analyze"]


class TestSimplePlanGuard:
    def test_merged_plan_is_not_simple(self, tmp_path):
        coordinator, _ = _coordinator(tmp_path)
        adopted = coordinator._adopt_plan([_producer(), _consumer()])
        coordinator._state.subtasks = adopted

        assert len(adopted) == 1
        assert coordinator._is_simple_plan() is False

    def test_unmerged_single_step_plan_is_still_simple(self, tmp_path):
        coordinator, _ = _coordinator(tmp_path)
        adopted = coordinator._adopt_plan(
            [_spec("st1", instructions="写 out/result.md", write_scope=["out/result.md"])]
        )
        coordinator._state.subtasks = adopted

        assert coordinator._is_simple_plan() is True


class TestPlanReviewGate:
    def test_merged_plan_still_requires_model_plan_review(self, tmp_path):
        coordinator, _ = _coordinator(tmp_path)
        adopted = coordinator._adopt_plan([_producer(), _consumer()])
        coordinator._state.subtasks = adopted
        coordinator._deps.tool_registry = SimpleNamespace(
            get_tool=lambda name: {"assignable": True},
            is_approved=lambda name: True,
            is_assignable=lambda name: True,
        )

        assert coordinator._state.plan_steps_merged == 1
        assert coordinator._should_skip_plan_review() is False

    def test_single_step_plan_without_merge_can_skip_review(self, tmp_path):
        coordinator, _ = _coordinator(tmp_path)
        adopted = coordinator._adopt_plan(
            [_spec("st1", instructions="写 out/result.md", write_scope=["out/result.md"])]
        )
        coordinator._state.subtasks = adopted
        coordinator._deps.tool_registry = SimpleNamespace(
            get_tool=lambda name: {"assignable": True},
            is_approved=lambda name: True,
            is_assignable=lambda name: True,
        )

        assert coordinator._state.plan_steps_merged == 0
        assert coordinator._should_skip_plan_review() is True


def test_verification_rules_are_kept_from_both_steps(tmp_path):
    coordinator, _ = _coordinator(tmp_path)
    producer = _producer()
    producer.verification = [
        VerificationRule(kind="text_contains", path="analyze_incident.py",
                         text="metrics.json")
    ]
    consumer = _consumer()
    consumer.verification = [
        VerificationRule(kind="text_contains", path="incident_report.md",
                         text="关键指标")
    ]
    adopted = coordinator._adopt_plan([producer, consumer])

    texts = {rule.text for rule in adopted[0].verification if rule.kind == "text_contains"}
    assert {"metrics.json", "关键指标"} <= texts

"""单元测试：领域数据模型（Pydantic v2）。"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from so_agent.models import (
    AgentRecord,
    ExecutionResult,
    GeneratedToolManifest,
    HumanEscalationRequest,
    ReplacementRequest,
    ReviewDecision,
    SubtaskSpec,
    TaskDecompositionIssue,
    TaskRequest,
    TaskStatus,
    ToolGrant,
)

UTC = timezone.utc


def _aware(dt: datetime) -> bool:
    return dt.tzinfo is not None and dt.utcoffset() is not None


class TestTaskStatus:
    """TaskStatus 枚举：14 个状态、str 枚举、值唯一。"""

    def test_has_exactly_fourteen_states(self):
        assert len(TaskStatus) == 14

    def test_state_values(self):
        expected = {
            "received",
            "planning",
            "plan_review",
            "executing",
            "replacement_requested",
            "replacement_executing",
            "decomposition_issue",
            "replanning",
            "aggregating",
            "result_review",
            "completed",
            "human_required",
            "failed",
            "cancelled",
        }
        assert {s.value for s in TaskStatus} == expected

    def test_is_str_enum(self):
        assert isinstance(TaskStatus.COMPLETED, str)
        assert TaskStatus("completed") is TaskStatus.COMPLETED
        assert TaskStatus.COMPLETED.value == "completed"

    def test_construct_by_value(self):
        for member in TaskStatus:
            assert TaskStatus(member.value) is member

    def test_invalid_value_raises(self):
        with pytest.raises(ValueError):
            TaskStatus("not-a-status")


class TestTaskRequest:
    def test_required_fields_only(self):
        req = TaskRequest(task_id="t1", objective="构建报表")
        assert req.task_id == "t1"
        assert req.objective == "构建报表"
        assert req.constraints == []
        assert req.acceptance_criteria == []
        assert req.original_input == ""

    def test_defaults_are_independent_lists(self):
        a = TaskRequest(task_id="t1", objective="a")
        b = TaskRequest(task_id="t2", objective="b")
        a.constraints.append("c")
        assert b.constraints == []

    def test_roundtrip_json(self):
        req = TaskRequest(
            task_id="t1",
            objective="目标",
            constraints=["离线运行"],
            acceptance_criteria=["有报告"],
            original_input="原始输入",
        )
        restored = TaskRequest.model_validate_json(req.model_dump_json())
        assert restored == req

    def test_roundtrip_dict_json_mode(self):
        req = TaskRequest(task_id="t1", objective="x")
        restored = TaskRequest.model_validate(req.model_dump(mode="json"))
        assert restored == req


class TestSubtaskSpec:
    def test_defaults(self):
        spec = SubtaskSpec(subtask_id="s1", title="t", instructions="i", role="code")
        assert spec.plan_version == 1
        assert spec.dependencies == []
        assert spec.allowed_tools == []
        assert spec.expected_output == ""
        assert spec.max_attempts == 3

    def test_full_roundtrip(self):
        spec = SubtaskSpec(
            subtask_id="s1",
            plan_version=2,
            title="t",
            instructions="i",
            role="code",
            dependencies=["s0"],
            allowed_tools=["write_file"],
            expected_output="out",
            max_attempts=5,
        )
        restored = SubtaskSpec.model_validate_json(spec.model_dump_json())
        assert restored == spec


class TestToolGrant:
    def test_defaults(self):
        grant = ToolGrant(
            grant_id="g1", task_id="t1", agent_id="a1", issued_by="supervisor"
        )
        assert grant.allowed_tools == []
        assert grant.expires_at is None
        assert grant.revoked is False
        assert _aware(grant.issued_at)
        assert grant.issued_at.utcoffset().total_seconds() == 0

    def test_expires_at_roundtrip(self):
        from datetime import timedelta

        expires = datetime(2030, 1, 1, tzinfo=UTC) + timedelta(days=1)
        grant = ToolGrant(
            grant_id="g1",
            task_id="t1",
            agent_id="a1",
            allowed_tools=["read_file"],
            issued_by="supervisor",
            expires_at=expires,
        )
        restored = ToolGrant.model_validate_json(grant.model_dump_json())
        assert restored.expires_at == expires
        assert restored.revoked is False


class TestAgentRecord:
    def test_defaults(self):
        rec = AgentRecord(agent_id="a1", agent_type="dynamic", parent_task_id="t1")
        assert rec.plan_version == 1
        assert rec.allowed_tools == []
        assert rec.attempt_count == 0
        assert rec.can_request_replacement is True
        assert rec.replacement_requested is False
        assert rec.replacement_of is None
        assert rec.status == "created"
        assert rec.expires_at is None
        assert _aware(rec.created_at)
        assert rec.created_at.utcoffset().total_seconds() == 0

    def test_replacement_record_roundtrip(self):
        rec = AgentRecord(
            agent_id="a2",
            agent_type="dynamic",
            parent_task_id="t1",
            can_request_replacement=False,
            replacement_of="a1",
            status="running",
        )
        restored = AgentRecord.model_validate_json(rec.model_dump_json())
        assert restored == rec

    def test_invalid_status_rejected(self):
        with pytest.raises(ValueError):
            AgentRecord(
                agent_id="a1",
                agent_type="dynamic",
                parent_task_id="t1",
                status="unknown-status",
            )

    def test_agent_status_values_accepted(self):
        for status in (
            "created",
            "running",
            "waiting",
            "completed",
            "failed",
            "replaced",
            "revoked",
        ):
            rec = AgentRecord(
                agent_id="a1",
                agent_type="dynamic",
                parent_task_id="t1",
                status=status,
            )
            assert rec.status == status


class TestReplacementRequest:
    def test_defaults(self):
        req = ReplacementRequest(
            request_id="r1",
            task_id="t1",
            subtask_id="s1",
            requester_agent_id="a1",
            failure_summary="失败原因",
        )
        assert req.attempt_evidence == []
        assert req.requested_tools == []
        assert req.request_count == 1

    def test_roundtrip(self):
        req = ReplacementRequest(
            request_id="r1",
            task_id="t1",
            subtask_id="s1",
            requester_agent_id="a1",
            failure_summary="f",
            attempt_evidence=["e1"],
            requested_tools=["write_file"],
            request_count=2,
        )
        restored = ReplacementRequest.model_validate_json(req.model_dump_json())
        assert restored == req


class TestTaskDecompositionIssue:
    def test_defaults(self):
        issue = TaskDecompositionIssue(task_id="t1", subtask_id="s1", plan_version=1)
        assert issue.failed_agent_ids == []
        assert issue.failure_reasons == []
        assert issue.attempted_approaches == []
        assert issue.recommendation == ""

    def test_roundtrip(self):
        issue = TaskDecompositionIssue(
            task_id="t1",
            subtask_id="s1",
            plan_version=3,
            failed_agent_ids=["a1"],
            failure_reasons=["超时"],
            attempted_approaches=["缩小子任务"],
            recommendation="换工具",
        )
        restored = TaskDecompositionIssue.model_validate_json(issue.model_dump_json())
        assert restored == issue


class TestExecutionResult:
    def test_success_required(self):
        with pytest.raises(ValueError):
            ExecutionResult(subtask_id="s1")  # type: ignore[call-arg]

    def test_defaults(self):
        result = ExecutionResult(subtask_id="s1", success=True)
        assert result.output == ""
        assert result.evidence == []
        assert result.artifacts == []
        assert result.error is None
        assert result.duration == 0.0

    def test_roundtrip(self):
        result = ExecutionResult(
            subtask_id="s1",
            success=False,
            output="o",
            evidence=["e"],
            artifacts=["/tmp/x"],
            error="boom",
            duration=1.5,
        )
        restored = ExecutionResult.model_validate_json(result.model_dump_json())
        assert restored == result


class TestReviewDecision:
    def test_stage_literal_accepted(self):
        for stage in ("plan_review", "result_review"):
            decision = ReviewDecision(stage=stage, passed=True)
            assert decision.stage == stage

    def test_stage_invalid_rejected(self):
        with pytest.raises(ValueError):
            ReviewDecision(stage="other_review", passed=True)  # type: ignore[arg-type]

    def test_defaults_and_roundtrip(self):
        decision = ReviewDecision(stage="plan_review", passed=False)
        assert decision.issues == []
        assert decision.required_fixes == []
        assert decision.retry_target is None
        assert decision.summary == ""
        restored = ReviewDecision.model_validate_json(decision.model_dump_json())
        assert restored == decision


class TestGeneratedToolManifest:
    def test_defaults(self):
        manifest = GeneratedToolManifest(
            tool_id="gt1",
            tool_name="my_tool",
            entry_file="my_tool.py",
            created_by_task="t1",
        )
        assert manifest.description == ""
        assert manifest.input_schema == {}
        assert manifest.output_schema == {}
        assert manifest.review_status == "pending"
        assert manifest.version == "0.1.0"

    def test_review_status_literal(self):
        for status in ("pending", "approved", "rejected"):
            manifest = GeneratedToolManifest(
                tool_id="gt1",
                tool_name="x",
                entry_file="x.py",
                created_by_task="t1",
                review_status=status,
            )
            assert manifest.review_status == status
        with pytest.raises(ValueError):
            GeneratedToolManifest(
                tool_id="gt1",
                tool_name="x",
                entry_file="x.py",
                created_by_task="t1",
                review_status="unknown",  # type: ignore[arg-type]
            )

    def test_roundtrip(self):
        manifest = GeneratedToolManifest(
            tool_id="gt1",
            tool_name="x",
            entry_file="x.py",
            description="d",
            input_schema={"type": "object"},
            output_schema={"type": "object"},
            created_by_task="t1",
            review_status="approved",
            version="1.2.0",
        )
        restored = GeneratedToolManifest.model_validate_json(manifest.model_dump_json())
        assert restored == manifest


class TestHumanEscalationRequest:
    def test_defaults(self):
        req = HumanEscalationRequest(
            task_id="t1",
            original_request=TaskRequest(task_id="t1", objective="o"),
        )
        assert req.plan_history == []
        assert req.agent_failure_history == []
        assert req.tool_call_evidence == []
        assert req.final_review is None
        assert _aware(req.created_at)

    def test_full_roundtrip_with_nested_models(self):
        req = HumanEscalationRequest(
            task_id="t1",
            original_request=TaskRequest(task_id="t1", objective="o"),
            plan_history=[
                [SubtaskSpec(subtask_id="s1", title="t", instructions="i", role="code")]
            ],
            agent_failure_history=["a1 failed"],
            tool_call_evidence=['{"tool": "x"}'],
            final_review=ReviewDecision(stage="result_review", passed=False),
        )
        restored = HumanEscalationRequest.model_validate_json(req.model_dump_json())
        assert restored == req
        assert restored.plan_history[0][0].subtask_id == "s1"
        assert restored.final_review is not None
        assert restored.final_review.passed is False

    def test_timestamps_are_json_serializable(self):
        req = HumanEscalationRequest(
            task_id="t1",
            original_request=TaskRequest(task_id="t1", objective="o"),
        )
        payload = req.model_dump(mode="json")
        assert isinstance(payload["created_at"], str)
        assert "T" in payload["created_at"]

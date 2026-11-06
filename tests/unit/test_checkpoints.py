"""检查点落盘、工作流状态恢复与 CLI 断点续跑。"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from so_agent import main as main_mod
from so_agent.config import Settings
from so_agent.context import ProjectContext
from so_agent.models import (
    ExecutionResult,
    HumanEscalationRequest,
    ReviewDecision,
    SubtaskSpec,
    TaskRequest,
    TaskStatus,
    ToolGrant,
)
from so_agent.orchestrator import OrchestratorOutput
from so_agent.runtime.checkpoints import (
    CheckpointError,
    delete_checkpoint,
    load_checkpoint,
    save_checkpoint,
)
from so_agent.workflow import WorkflowEngine


def _build_engine(tmp_path) -> WorkflowEngine:
    context = ProjectContext(sandbox_dir=tmp_path, project_name="checkpoint-test")
    return WorkflowEngine(
        context,
        settings=Settings(_env_file=None),
        orchestrator=SimpleNamespace(name="fake-supervisor"),
        review_agent=SimpleNamespace(name="fake-reviewer"),
    )


def _seed_engine(engine: WorkflowEngine) -> None:
    subtask = SubtaskSpec(
        subtask_id="s1",
        title="分析需求",
        instructions="分析用户目标",
        role="analysis",
        dependencies=[],
        allowed_tools=["read_file"],
    )
    request = TaskRequest(task_id="task-1", objective="分析仓库")
    grant = ToolGrant(
        grant_id="g1",
        task_id="task-1",
        agent_id="supervisor",
        allowed_tools=["read_file"],
        issued_by="supervisor",
    )
    engine._task_request = request
    engine._state.current_status = TaskStatus.HUMAN_REQUIRED
    engine._plan_version = 2
    engine._replan_count = 1
    engine._result_rework_count = 1
    engine._state.plan_steps_merged = 1
    engine._state.merged_subtask_ids = {"s1"}
    engine._subtasks = [subtask]
    engine._pending_specs = [subtask]
    engine._results = {
        "s1": ExecutionResult(subtask_id="s1", success=False, error="boom")
    }
    engine._plan_history = [[subtask]]
    engine._agent_failure_history = ["执行失败：boom"]
    engine._grants = [grant]
    engine._plan_grants = [grant]
    engine._plan_grant_map = {"s1": grant}
    engine._plan_grant_proposals = [grant]
    engine._last_review = ReviewDecision(
        stage="result_review", passed=False, required_fixes=["fix it"]
    )
    engine._human_escalation = HumanEscalationRequest(
        task_id="task-1",
        original_request=request,
        reason="连续失败",
        rollback_targets=["s1"],
    )
    engine._archive_results = {
        (1, "s0"): ExecutionResult(subtask_id="s0", success=True, output="old")
    }
    engine._seen_replacement_requests = {"req-1"}
    engine._seen_issue_fingerprints = {("result_review", 0, ("fp-1",))}


def test_checkpoint_roundtrip_and_validation(tmp_path):
    path = save_checkpoint(tmp_path, "task-1", {"status": "human_required"})
    assert path.is_file()
    payload = load_checkpoint(tmp_path, "task-1")
    assert payload["checkpoint_version"] == 1
    assert payload["status"] == "human_required"

    with pytest.raises(CheckpointError):
        save_checkpoint(tmp_path, "../escape", {})

    bad = tmp_path / "checkpoints" / "bad.json"
    bad.write_text(json.dumps({"checkpoint_version": 99}), encoding="utf-8")
    with pytest.raises(CheckpointError):
        load_checkpoint(tmp_path, "bad")

    broken = tmp_path / "checkpoints" / "broken.json"
    broken.write_text("{not json", encoding="utf-8")
    with pytest.raises(CheckpointError):
        load_checkpoint(tmp_path, "broken")

    assert delete_checkpoint(tmp_path, "task-1") is True
    assert delete_checkpoint(tmp_path, "task-1") is False
    with pytest.raises(CheckpointError):
        load_checkpoint(tmp_path, "task-1")


def test_engine_export_restore_preserves_state(tmp_path):
    engine = _build_engine(tmp_path)
    _seed_engine(engine)
    payload = engine.export_checkpoint()
    assert payload["task_id"] == "task-1"
    assert payload["status"] == "human_required"
    assert payload["archive_results"][0]["subtask_id"] == "s0"
    assert payload["plan_steps_merged"] == 1
    assert payload["merged_subtask_ids"] == ["s1"]

    restored = _build_engine(tmp_path / "restored")
    restored.restore_checkpoint(payload)
    assert restored.current_status is TaskStatus.HUMAN_REQUIRED
    assert restored.plan_version == 2
    assert restored.replan_count == 1
    assert restored._task_request.objective == "分析仓库"
    assert [item.subtask_id for item in restored._subtasks] == ["s1"]
    assert restored._results["s1"].error == "boom"
    assert restored._plan_history[0][0].title == "分析需求"
    assert restored._agent_failure_history == ["执行失败：boom"]
    assert restored._plan_grant_map["s1"].grant_id == "g1"
    assert restored._last_review.required_fixes == ["fix it"]
    assert restored._human_escalation.reason == "连续失败"
    assert restored._archive_results[(1, "s0")].output == "old"
    assert restored._state.plan_steps_merged == 1
    assert restored._state.merged_subtask_ids == {"s1"}
    assert restored._seen_replacement_requests == {"req-1"}
    assert restored._seen_issue_fingerprints == {
        ("result_review", 0, ("fp-1",))
    }
    assert any(
        record.get("event_type") == "checkpoint_resumed"
        for record in restored._context.event_log
    )


def test_engine_export_merges_scheduler_results_and_progress(tmp_path):
    engine = _build_engine(tmp_path)
    _seed_engine(engine)
    engine._scheduler.results["s2"] = ExecutionResult(
        subtask_id="s2", success=True, output="清理期落账的部分结果"
    )
    engine._gate.record_progress("batch_completed:1")

    payload = engine.export_checkpoint()

    results = payload["results"]
    # 调度器清理期落账的结果与引擎结果合并，续跑时不重做已完成工作
    assert results["s1"]["success"] is False
    assert results["s2"]["output"] == "清理期落账的部分结果"
    assert payload["progress"]["label"] == "batch_completed:1"
    assert payload["progress"]["count"] == 1


def test_escalation_persists_checkpoint(tmp_path):
    engine = _build_engine(tmp_path)
    _seed_engine(engine)
    engine._persist_checkpoint("连续失败")
    payload = load_checkpoint(tmp_path, "task-1")
    assert payload["status"] == "human_required"
    assert payload["human_escalation"]["reason"] == "连续失败"
    assert any(
        record.get("event_type") == "checkpoint_saved"
        for record in engine._context.event_log
    )


def _args(**overrides):
    values = {
        "resume": "task-1",
        "guidance_action": "abort",
        "rollback_to": ["s1"],
        "instructions": "补充边界条件",
        "additional_info": "参考 docs/spec.md",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class _FakeEngine:
    next_status = TaskStatus.HUMAN_REQUIRED

    def __init__(self, context, **kwargs) -> None:
        self.context = context
        self.restored = None
        self.guidance = None

    def restore_checkpoint(self, payload) -> None:
        self.restored = payload

    async def resume_with_guidance(self, guidance):
        self.guidance = guidance
        return OrchestratorOutput(
            task_id="task-1",
            status=self.next_status,
            final_result="resumed",
        )


def test_cli_resume_abort_keeps_checkpoint(tmp_path, monkeypatch, capsys):
    _FakeEngine.next_status = TaskStatus.HUMAN_REQUIRED
    save_checkpoint(tmp_path, "task-1", {"status": "human_required"})
    monkeypatch.setattr(
        "so_agent.workflow.WorkflowEngine", _FakeEngine
    )
    context = ProjectContext(sandbox_dir=tmp_path, project_name="cli")
    code = asyncio.run(
        main_mod._resume_from_checkpoint(
            _args(), context, Settings(_env_file=None), tmp_path / "run.log"
        )
    )
    assert code == 1
    assert load_checkpoint(tmp_path, "task-1")["status"] == "human_required"
    output = capsys.readouterr().out
    assert '"task_id": "task-1"' in output


def test_cli_resume_completed_deletes_checkpoint(tmp_path, monkeypatch):
    save_checkpoint(tmp_path, "task-1", {"status": "human_required"})
    _FakeEngine.next_status = TaskStatus.COMPLETED
    monkeypatch.setattr(
        "so_agent.workflow.WorkflowEngine", _FakeEngine
    )
    context = ProjectContext(sandbox_dir=tmp_path, project_name="cli")
    code = asyncio.run(
        main_mod._resume_from_checkpoint(
            _args(), context, Settings(_env_file=None), tmp_path / "run.log"
        )
    )
    assert code == 0
    assert delete_checkpoint(tmp_path, "task-1") is False


def test_cli_resume_rejects_objective_conflict():
    with pytest.raises(SystemExit) as excinfo:
        main_mod.main(["--resume", "task-1", "目标"])
    assert excinfo.value.code == 2

"""无进展看门狗底座：PipelineGate 进展记录与子任务级检查点触发。"""

from __future__ import annotations

import time
from types import SimpleNamespace

from so_agent.config import Settings
from so_agent.context import ProjectContext
from so_agent.models import TaskRequest
from so_agent.runtime.checkpoints import load_checkpoint
from so_agent.workflow import WorkflowEngine
from so_agent.workflow_support import PipelineGate


def _build_engine(tmp_path) -> WorkflowEngine:
    context = ProjectContext(
        sandbox_dir=tmp_path, project_name="progress-test"
    )
    return WorkflowEngine(
        context,
        settings=Settings(_env_file=None),
        orchestrator=SimpleNamespace(name="fake-supervisor"),
        review_agent=SimpleNamespace(name="fake-reviewer"),
    )


class TestPipelineGateProgress:
    def test_fresh_gate_has_no_progress(self):
        gate = PipelineGate()
        assert gate.progress_count == 0
        assert gate.seconds_since_progress() == 0.0
        snapshot = gate.progress_snapshot()
        assert snapshot["count"] == 0
        assert snapshot["label"] == ""

    def test_record_progress_updates_label_and_count(self):
        gate = PipelineGate()
        gate.record_progress("plan_accepted")
        gate.record_progress("subtask:s1")
        assert gate.progress_count == 2
        snapshot = gate.progress_snapshot()
        assert snapshot["label"] == "subtask:s1"
        assert snapshot["seconds_since_progress"] >= 0.0

    def test_seconds_since_progress_tracks_idle_window(self):
        gate = PipelineGate()
        gate.record_progress("batch_completed:1")
        # 回拨时间戳模拟长时间无进展（等价于真实等待窗口）
        gate._last_progress_at = time.monotonic() - 42.0
        assert gate.seconds_since_progress() >= 42.0
        gate.record_progress("subtask:s2")
        assert gate.seconds_since_progress() < 1.0

    def test_reset_clears_progress_but_keeps_generation_increasing(self):
        gate = PipelineGate()
        gate.record_progress("progress")
        generation_before = gate.generation
        gate.reset()
        assert gate.progress_count == 0
        assert gate.progress_snapshot()["label"] == ""
        assert gate.generation == generation_before + 1


class TestContextActivityPropagation:
    def test_engine_wires_context_sink_to_gate(self, tmp_path):
        engine = _build_engine(tmp_path)
        engine._context.report_activity("nested:stream:model")
        assert engine._gate.progress_count == 1
        assert (
            engine._gate.progress_snapshot()["label"] == "nested:stream:model"
        )

    def test_context_without_sink_is_noop(self, tmp_path):
        context = ProjectContext(sandbox_dir=tmp_path, project_name="no-sink")
        context.report_activity("nested:tool")  # 不抛错、不产生副作用
        assert context.activity_sink is None

    def test_sink_failure_does_not_break_caller(self, tmp_path):
        context = ProjectContext(sandbox_dir=tmp_path, project_name="bad-sink")

        def boom(label: str) -> None:
            raise RuntimeError("sink down")

        context.activity_sink = boom
        context.report_activity("nested:stream")


class TestSubtaskProgressCheckpoint:
    def test_subtask_completion_records_progress_and_checkpoint(self, tmp_path):
        engine = _build_engine(tmp_path)
        engine._task_request = TaskRequest(
            task_id="task-progress", objective="验证检查点"
        )

        engine._handle_subtask_progress("s1", None)

        assert engine._gate.progress_count == 1
        assert engine._gate.progress_snapshot()["label"] == "subtask:s1"
        payload = load_checkpoint(tmp_path, "task-progress")
        assert payload["progress"]["label"] == "subtask:s1"
        assert payload["progress"]["count"] == 1

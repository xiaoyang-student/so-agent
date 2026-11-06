"""CLI 断点续跑遇 provisional 时必须等待后台评审收口。"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import so_agent.main as main_mod
import so_agent.runtime.checkpoints as checkpoints_mod
import so_agent.workflow as workflow_mod
from so_agent.config import Settings
from so_agent.context import ProjectContext
from so_agent.models import TaskStatus
from so_agent.orchestrator import OrchestratorOutput


def test_resume_waits_for_provisional_review(monkeypatch, tmp_path, capsys):
    task_id = "c" * 32
    provisional = OrchestratorOutput(
        task_id=task_id,
        status=TaskStatus.COMPLETED_PROVISIONAL,
        final_result="初步结果",
        provisional=True,
    )
    final = OrchestratorOutput(
        task_id=task_id,
        status=TaskStatus.COMPLETED,
        final_result="评审后的结果",
    )
    waited: list[int] = []

    class FakeEngine:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def restore_checkpoint(self, payload) -> None:
            pass

        async def resume_with_guidance(self, guidance):
            return provisional

        async def wait_for_provisional(self):
            waited.append(1)
            return final

    monkeypatch.setattr(checkpoints_mod, "load_checkpoint", lambda *a, **k: {})
    monkeypatch.setattr(checkpoints_mod, "delete_checkpoint", lambda *a, **k: None)
    monkeypatch.setattr(workflow_mod, "WorkflowEngine", FakeEngine)

    args = SimpleNamespace(
        resume=task_id,
        guidance_action="resume",
        rollback_to=[],
        instructions="继续",
        additional_info="",
    )
    context = ProjectContext(
        sandbox_dir=tmp_path,
        project_name="resume-cli",
        config=Settings(_env_file=None),
    )
    code = asyncio.run(
        main_mod._resume_from_checkpoint(
            args, context, Settings(_env_file=None), tmp_path / "run.log"
        )
    )

    assert code == 0
    assert waited == [1]
    out = capsys.readouterr().out
    assert "评审后的结果" in out
    assert "completed_provisional" not in out

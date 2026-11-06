"""Web 产物扫描过滤内部目录与业务文件放行。"""

from __future__ import annotations

from pathlib import Path

from so_agent.web.manager import TaskManager


def test_internal_directories_are_hidden():
    for relative in (
        "checkpoints/task.json",
        "task_abc/run_1/main.py",
        "run_1/stdout.txt",
        "generated_tools/tool.py",
        "__pycache__/mod.pyc",
    ):
        assert TaskManager._is_hidden_or_internal(Path(relative)) is True


def test_business_outputs_are_exposed():
    for relative in (
        "report.md",
        "out/data.csv",
        "task_summary.md",
        "running_notes.txt",
    ):
        assert TaskManager._is_hidden_or_internal(Path(relative)) is False

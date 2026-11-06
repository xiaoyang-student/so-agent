"""Web 控制台（可选组件）。

``create_app`` 需要安装可选依赖：``pip install -e ".[web]"``。
任务执行本身仍由核心包完成，本包只负责提交、观察与产物浏览。
"""

from __future__ import annotations

__all__ = ["create_app", "TaskManager", "WebTask"]


def __getattr__(name: str):
    if name in {"create_app", "main"}:
        from so_agent.web.app import create_app, main

        return {"create_app": create_app, "main": main}[name]
    if name in {"TaskManager", "WebTask"}:
        from so_agent.web.manager import TaskManager, WebTask

        return {"TaskManager": TaskManager, "WebTask": WebTask}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

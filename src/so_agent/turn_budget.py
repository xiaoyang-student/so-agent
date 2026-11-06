"""单次模型回合上限的作用域覆盖。

合并后的子任务一次调用承担原两步工作量，固定回合上限可能在正常完成
前耗尽。控制层在派发该类子任务时抬高上限，嵌套 Agent 适配器读取当前
作用域生效；覆盖值不进入 MCP 参数契约，模型看不到也无法自行抬高。
"""

from __future__ import annotations

import contextvars
from contextlib import contextmanager
from typing import Iterator

_override: contextvars.ContextVar[int | None] = contextvars.ContextVar(
    "so_agent_model_turn_override", default=None
)


@contextmanager
def model_turn_override(turns: int) -> Iterator[None]:
    """在作用域内把嵌套 Agent 的回合上限至少提高到 ``turns``。"""
    token = _override.set(max(1, int(turns)))
    try:
        yield
    finally:
        _override.reset(token)


def current_model_turn_override() -> int | None:
    """返回当前作用域的回合上限覆盖值（未设置时为 None）。"""
    return _override.get()

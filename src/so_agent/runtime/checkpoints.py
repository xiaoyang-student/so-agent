"""工作流检查点：人工兜底时落盘，CLI 断点续跑时读取。

检查点只写入任务沙箱的 ``checkpoints`` 目录，文件名由任务 ID 严格校验，
拒绝目录穿越；内容使用原子写入，读取时校验顶层结构。
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from so_agent.runtime.tool_store import atomic_write_text

CHECKPOINT_DIRNAME = "checkpoints"
_CHECKPOINT_VERSION = 1
_TASK_ID_PATTERN = re.compile(r"^[A-Za-z0-9_.-]{1,128}$")


class CheckpointError(RuntimeError):
    """检查点缺失、损坏或任务 ID 非法。"""


def checkpoint_dir(sandbox_dir: str | Path) -> Path:
    return Path(sandbox_dir).expanduser().resolve() / CHECKPOINT_DIRNAME


def checkpoint_path(sandbox_dir: str | Path, task_id: str) -> Path:
    if not _TASK_ID_PATTERN.fullmatch(task_id or ""):
        raise CheckpointError(f"任务 ID 非法：{task_id!r}")
    return checkpoint_dir(sandbox_dir) / f"{task_id}.json"


def save_checkpoint(
    sandbox_dir: str | Path, task_id: str, payload: dict[str, Any]
) -> Path:
    """原子写入检查点；返回落盘路径。"""
    path = checkpoint_path(sandbox_dir, task_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    document = {"checkpoint_version": _CHECKPOINT_VERSION, **payload}
    atomic_write_text(
        path, json.dumps(document, ensure_ascii=False, indent=2) + "\n"
    )
    return path


def load_checkpoint(
    sandbox_dir: str | Path, task_id: str
) -> dict[str, Any]:
    """读取并校验检查点；不存在或损坏时抛 ``CheckpointError``。"""
    path = checkpoint_path(sandbox_dir, task_id)
    if not path.is_file():
        raise CheckpointError(f"检查点不存在：{path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CheckpointError(f"检查点读取失败：{path}（{exc}）") from exc
    if not isinstance(data, dict):
        raise CheckpointError(f"检查点顶层必须是对象：{path}")
    if int(data.get("checkpoint_version") or 0) != _CHECKPOINT_VERSION:
        raise CheckpointError(
            f"不支持的检查点版本：{data.get('checkpoint_version')!r}"
        )
    return data


def delete_checkpoint(sandbox_dir: str | Path, task_id: str) -> bool:
    path = checkpoint_path(sandbox_dir, task_id)
    try:
        path.unlink()
        return True
    except FileNotFoundError:
        return False


__all__ = [
    "CheckpointError",
    "checkpoint_dir",
    "checkpoint_path",
    "delete_checkpoint",
    "load_checkpoint",
    "save_checkpoint",
]

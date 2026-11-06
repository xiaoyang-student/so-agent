"""同族任务脚本复用缓存。

成功执行过的 ``.py`` 产物按任务目标哈希缓存在工具库；同一目标再次执行
时，控制层先校验缓存文件哈希，再复制进沙箱，由 code_agent 决定是否复用。
只复制哈希一致的缓存，且绝不覆盖沙箱内本次已存在的文件——脚本是否适用
仍由执行器按真实输入判断，代码不凭文件名自动替代执行。
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from so_agent.config import Settings
from so_agent.runtime.events import EVENT_SCRIPT_REUSED
from so_agent.runtime.logging_setup import get_logger
from so_agent.runtime.tool_store import (
    ToolStoreLockError,
    atomic_write_text,
    registry_write_lock,
    resolve_tool_store_dir,
)

logger = get_logger("script_cache")

SCRIPT_CACHE_SUBDIR = "verified_scripts"

# 单个脚本缓存上限：超过视为数据文件而非可复用脚本，不进入缓存
_MAX_SCRIPT_BYTES = 1_048_576

# 目标目录上限：缓存是加速手段而非归档，超出后删除最久未更新的目标
_MAX_FAMILIES = 50


def resolve_script_cache_dir(settings: Settings | None = None) -> Path:
    """返回脚本缓存根目录：``{tool_store}/verified_scripts``。"""
    return resolve_tool_store_dir(settings) / SCRIPT_CACHE_SUBDIR


def objective_key(objective: str) -> str:
    """任务目标的稳定键（空白归一化后取 sha256 前 16 位）。"""
    normalized = " ".join((objective or "").split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


def _load_index(family_dir: Path) -> dict[str, dict[str, Any]]:
    path = family_dir / "index.json"
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _save_index(family_dir: Path, index: dict[str, Any]) -> None:
    atomic_write_text(
        family_dir / "index.json",
        json.dumps(index, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    """按字节原子写入：脚本缓存必须逐字节保真（换行符也是哈希的一部分）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temp.write_bytes(data)
        os.replace(temp, path)
    finally:
        try:
            temp.unlink(missing_ok=True)
        except OSError:
            pass


def _safe_script_path(root: Path, relative: str) -> Path | None:
    """把相对路径解析为沙箱内的 .py 文件路径（越界/非脚本返回 None）。"""
    text = (relative or "").strip().replace("\\", "/")
    if (
        not text
        or text.startswith("/")
        or ".." in text.split("/")
        or not text.lower().endswith(".py")
    ):
        return None
    try:
        target = (root / text).resolve()
        target.relative_to(root.resolve())
    except (OSError, ValueError):
        return None
    return target


def _prune_families(root: Path, keep: Path) -> None:
    """目标目录超过上限时删除最久未更新的目录（保留当前目标）。"""
    try:
        families = [path for path in root.iterdir() if path.is_dir() and path != keep]
    except OSError:
        return
    if len(families) + 1 <= _MAX_FAMILIES:
        return
    families.sort(key=lambda path: path.stat().st_mtime)
    for old in families[: len(families) + 1 - _MAX_FAMILIES]:
        shutil.rmtree(old, ignore_errors=True)


def record_verified_scripts(
    sandbox_root: Path,
    objective: str,
    artifacts: Iterable[str],
    *,
    task_id: str,
    settings: Settings | None = None,
    events: Any = None,
) -> list[dict[str, str]]:
    """把成功子任务的 .py 产物登记到同目标缓存，返回已登记条目。"""
    if not bool(getattr(settings, "code_agent_script_reuse", True)):
        return []
    root = Path(sandbox_root).expanduser().resolve()
    family_dir = resolve_script_cache_dir(settings) / objective_key(objective)
    recorded: list[dict[str, str]] = []
    try:
        # 索引是读-改-写：并发任务会互相覆盖条目，必须与工具清单共用
        # 同一种跨进程锁；锁等待失败只放弃本次登记，不影响任务结果。
        with registry_write_lock(family_dir / "index.json"):
            index = _load_index(family_dir)
            seen: set[Path] = set()
            for relative in artifacts:
                target = _safe_script_path(root, str(relative))
                if target is None or target in seen or not target.is_file():
                    continue
                seen.add(target)
                try:
                    data = target.read_bytes()
                except OSError:
                    continue
                if not data or len(data) > _MAX_SCRIPT_BYTES:
                    continue
                digest = hashlib.sha256(data).hexdigest()
                cached_name = f"{digest[:12]}-{target.name}"
                cached_path = family_dir / cached_name
                if not cached_path.is_file():
                    _atomic_write_bytes(cached_path, data)
                index[target.name] = {
                    "file": cached_name,
                    "sha256": digest,
                    "bytes": len(data),
                    "task_id": task_id,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                }
                recorded.append(
                    {"path": target.relative_to(root).as_posix(), "sha256": digest}
                )
            if not recorded:
                return []
            _save_index(family_dir, index)
    except ToolStoreLockError as exc:
        logger.warning("脚本缓存索引加锁失败，本次不登记：%s", exc)
        return []
    _prune_families(family_dir.parent, family_dir)
    if events is not None:
        events.log_event(
            EVENT_SCRIPT_REUSED,
            {"task_id": task_id, "action": "recorded", "scripts": recorded},
        )
    logger.info("已登记 %d 个可复用脚本（任务 %s）", len(recorded), task_id)
    return recorded


def stage_cached_scripts(
    sandbox_root: Path,
    objective: str,
    *,
    task_id: str,
    settings: Settings | None = None,
    events: Any = None,
) -> list[dict[str, str]]:
    """把哈希校验通过的缓存脚本复制进沙箱，返回本次放入的条目。"""
    if not bool(getattr(settings, "code_agent_script_reuse", True)):
        return []
    root = Path(sandbox_root).expanduser().resolve()
    family_dir = resolve_script_cache_dir(settings) / objective_key(objective)
    index = _load_index(family_dir)
    staged: list[dict[str, str]] = []
    rejected: list[dict[str, str]] = []
    for name, entry in index.items():
        if not isinstance(entry, dict):
            continue
        target = _safe_script_path(root, name)
        if target is None:
            rejected.append({"name": name, "reason": "invalid_path"})
            continue
        if target.exists():
            continue  # 本次已产出：不覆盖进行中的文件
        cached_path = family_dir / str(entry.get("file") or "")
        if not cached_path.is_file():
            rejected.append({"name": name, "reason": "cache_missing"})
            continue
        try:
            data = cached_path.read_bytes()
        except OSError:
            rejected.append({"name": name, "reason": "cache_unreadable"})
            continue
        digest = hashlib.sha256(data).hexdigest()
        if digest != entry.get("sha256"):
            rejected.append({"name": name, "reason": "hash_mismatch"})
            continue
        _atomic_write_bytes(target, data)
        staged.append({"path": target.relative_to(root).as_posix(), "sha256": digest})
    if staged and events is not None:
        events.log_event(
            EVENT_SCRIPT_REUSED,
            {"task_id": task_id, "action": "staged", "scripts": staged},
        )
    if rejected and events is not None:
        events.log_event(
            EVENT_SCRIPT_REUSED,
            {"task_id": task_id, "action": "rejected", "scripts": rejected},
        )
    return staged

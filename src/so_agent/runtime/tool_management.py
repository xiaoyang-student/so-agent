"""生成工具库运维：清单读取与受控变更（固定、解除隔离、弃用）。

所有写入都在 ``registry.json`` 的跨进程写锁内完成，并保留无法解析的
历史条目（孤儿条目），避免运维操作与任务运行并发时丢更新。这里只做
清单层面的治理动作，不注册工具、不绕过评审或安全准入。
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from so_agent.config import Settings, get_settings
from so_agent.models import GeneratedToolManifest
from so_agent.runtime.tool_store import (
    ToolStoreLockError,
    atomic_write_text,
    registry_write_lock,
    resolve_generated_registry_path,
)

logger = logging.getLogger(__name__)

_QUARANTINE_ISSUE_PREFIX = "连续失败"


class ToolAdminError(RuntimeError):
    """工具运维操作被拒绝：清单缺失、工具不存在或状态不允许。"""


def _registry_path(settings: Settings | None) -> Path:
    return resolve_generated_registry_path(settings or get_settings())


def _read_entries(path: Path) -> list[Any]:
    if not path.is_file():
        raise ToolAdminError(f"生成工具清单不存在：{path}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ToolAdminError(f"生成工具清单读取失败：{path}（{exc}）") from exc
    if not isinstance(raw, list):
        raise ToolAdminError(f"生成工具清单顶层必须是数组：{path}")
    return raw


def _split_entries(entries: list[Any]) -> tuple[list[tuple[int, GeneratedToolManifest]], list[Any]]:
    valid: list[tuple[int, GeneratedToolManifest]] = []
    orphans: list[Any] = []
    for index, item in enumerate(entries):
        try:
            valid.append((index, GeneratedToolManifest.model_validate(item)))
        except Exception:
            orphans.append(item)
    return valid, orphans


def load_generated_manifests(
    settings: Settings | None = None,
) -> list[GeneratedToolManifest]:
    """读取工具库清单；文件不存在时返回空列表，非法条目跳过。"""
    path = _registry_path(settings)
    if not path.is_file():
        return []
    try:
        entries = _read_entries(path)
    except ToolAdminError:
        return []
    manifests, _ = _split_entries(entries)
    return [manifest for _, manifest in manifests]


def summarize_manifests(
    manifests: list[GeneratedToolManifest],
) -> dict[str, Any]:
    """汇总工具库状态（不携带源码与测试用例）。"""
    review = Counter(item.review_status for item in manifests)
    safety = Counter(item.safety_status for item in manifests)
    return {
        "generated_total": len(manifests),
        "by_review_status": dict(sorted(review.items())),
        "by_safety_status": dict(sorted(safety.items())),
        "verified": sum(1 for item in manifests if item.verified),
        "quarantined": safety.get("quarantined", 0),
        "pinned": sum(1 for item in manifests if item.pinned),
        "use_count": sum(item.use_count for item in manifests),
        "failure_count": sum(item.failure_count for item in manifests),
        "last_used_at": max(
            (item.last_used_at for item in manifests if item.last_used_at),
            default="",
        ),
        "checked_at": datetime.now(timezone.utc).isoformat(),
    }


def manifest_view(manifest: GeneratedToolManifest) -> dict[str, Any]:
    """单工具运维视图（不含输入输出契约与源码）。"""
    return {
        "tool_name": manifest.tool_name,
        "description": manifest.description,
        "review_status": manifest.review_status,
        "safety_status": manifest.safety_status,
        "safety_issues": list(manifest.safety_issues),
        "verified": manifest.verified,
        "version": manifest.version,
        "created_at": manifest.created_at,
        "last_used_at": manifest.last_used_at,
        "use_count": manifest.use_count,
        "failure_count": manifest.failure_count,
        "consecutive_failures": manifest.consecutive_failures,
        "pinned": manifest.pinned,
        "deprecated_at": manifest.deprecated_at,
        "created_by_task": manifest.created_by_task,
    }


def _update_manifest(
    settings: Settings | None,
    tool_name: str,
    updater: Callable[[GeneratedToolManifest], GeneratedToolManifest],
) -> GeneratedToolManifest:
    path = _registry_path(settings)
    try:
        with registry_write_lock(path):
            entries = _read_entries(path)
            valid, orphans = _split_entries(entries)
            target: GeneratedToolManifest | None = None
            rewritten: list[dict[str, Any]] = []
            for _, manifest in valid:
                if manifest.tool_name == tool_name:
                    target = updater(manifest.model_copy(deep=True))
                    rewritten.append(target.model_dump(mode="json"))
                else:
                    rewritten.append(manifest.model_dump(mode="json"))
            if target is None:
                raise ToolAdminError(f"生成工具不存在：{tool_name}")
            payload = rewritten + orphans
            atomic_write_text(
                path,
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            )
            logger.info("工具库运维：%s 已更新", tool_name)
            return target
    except ToolStoreLockError as exc:
        raise ToolAdminError(f"获取工具清单写锁失败：{exc}") from exc


def set_tool_pinned(
    tool_name: str, pinned: bool, settings: Settings | None = None
) -> GeneratedToolManifest:
    """固定/取消固定工具；固定工具不会被自动生命周期清理。"""

    def updater(manifest: GeneratedToolManifest) -> GeneratedToolManifest:
        return manifest.model_copy(update={"pinned": bool(pinned)})

    return _update_manifest(settings, tool_name, updater)


def clear_tool_quarantine(
    tool_name: str, settings: Settings | None = None
) -> GeneratedToolManifest:
    """解除连续失败隔离，恢复为通过状态并清零连续失败计数。"""

    def updater(manifest: GeneratedToolManifest) -> GeneratedToolManifest:
        if manifest.safety_status != "quarantined":
            raise ToolAdminError(f"工具未处于隔离状态：{tool_name}")
        issues = [
            item
            for item in manifest.safety_issues
            if not item.startswith(_QUARANTINE_ISSUE_PREFIX)
        ]
        return manifest.model_copy(
            update={
                "safety_status": "passed",
                "safety_issues": issues,
                "consecutive_failures": 0,
            }
        )

    return _update_manifest(settings, tool_name, updater)


def deprecate_tool(
    tool_name: str,
    settings: Settings | None = None,
    *,
    now: datetime | None = None,
) -> GeneratedToolManifest:
    """立即进入弃用期；宽限期后由既有生命周期维护归档。"""
    stamp = (now or datetime.now(timezone.utc)).isoformat()

    def updater(manifest: GeneratedToolManifest) -> GeneratedToolManifest:
        if manifest.review_status != "approved":
            raise ToolAdminError(f"仅已批准工具可弃用：{tool_name}")
        if manifest.deprecated_at:
            return manifest
        return manifest.model_copy(update={"deprecated_at": stamp})

    return _update_manifest(settings, tool_name, updater)


__all__ = [
    "ToolAdminError",
    "clear_tool_quarantine",
    "deprecate_tool",
    "load_generated_manifests",
    "manifest_view",
    "set_tool_pinned",
    "summarize_manifests",
]

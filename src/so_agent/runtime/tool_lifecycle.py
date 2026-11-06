"""生成工具生命周期维护：弃用、归档与清理。

规则（全部可通过 Settings 调整）：
- 未批准候选 / 已拒绝 / 安全检查未通过：超过候选 TTL 直接归档；
- 已批准工具：超过无使用 TTL 先进入弃用期，宽限期后再归档；
- ``pinned`` 与当前任务正在使用的工具永不自动清理；
- 归档把源码移入 ``{tool_store}/archive``，清单条目移除并写入归档索引；
- 归档目录超过保留期后删除文件，但索引记录保留供审计。

维护在任务开始时按最小间隔触发（默认每天一次），也可显式 ``force``。
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

from so_agent.config import Settings, get_settings
from so_agent.models import GeneratedToolManifest
from so_agent.runtime.tool_store import (
    ToolStoreLockError,
    atomic_write_text,
    registry_write_lock,
    resolve_archive_dir,
    resolve_generated_registry_path,
    resolve_generated_tools_dir,
)

_STATE_FILENAME = ".lifecycle_state.json"
_ARCHIVE_INDEX = "index.json"


@dataclass
class ToolLifecycleReport:
    """一次生命周期维护的结果摘要。"""

    swept: bool = False
    deprecated: list[str] = field(default_factory=list)
    archived: list[str] = field(default_factory=list)
    purged: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def _now(now: datetime | None) -> datetime:
    if now is None:
        return datetime.now(timezone.utc)
    return now if now.tzinfo is not None else now.replace(tzinfo=timezone.utc)


def _parse_time(value: str) -> datetime | None:
    text = (value or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)


def _age_base(manifest: GeneratedToolManifest, source_path: Path) -> datetime:
    for candidate in (manifest.last_used_at, manifest.created_at):
        parsed = _parse_time(candidate)
        if parsed is not None:
            return parsed
    try:
        return datetime.fromtimestamp(
            source_path.stat().st_mtime, tz=timezone.utc
        )
    except OSError:
        return datetime.now(timezone.utc)


def _archive_tool(
    manifest: GeneratedToolManifest,
    tools_dir: Path,
    archive_dir: Path,
    reason: str,
    now: datetime,
) -> None:
    stamp = now.strftime("%Y%m%d%H%M%S")
    destination = archive_dir / f"{manifest.tool_name}-{stamp}"
    counter = 1
    while destination.exists():
        counter += 1
        destination = archive_dir / f"{manifest.tool_name}-{stamp}-{counter}"
    destination.mkdir(parents=True, exist_ok=True)
    for filename in (
        manifest.entry_file,
        f"{manifest.tool_name}.mcp_schema.json",
    ):
        source = tools_dir / filename
        if source.is_file():
            shutil.move(str(source), str(destination / source.name))
    index_path = archive_dir / _ARCHIVE_INDEX
    records: list[dict] = []
    if index_path.is_file():
        try:
            raw = json.loads(index_path.read_text(encoding="utf-8"))
            if isinstance(raw, list):
                records = [item for item in raw if isinstance(item, dict)]
        except (OSError, json.JSONDecodeError):
            records = []
    records.append(
        {
            "archived_at": now.isoformat(),
            "reason": reason,
            "archive_dir": destination.name,
            "manifest": manifest.model_dump(mode="json"),
        }
    )
    atomic_write_text(
        index_path, json.dumps(records, ensure_ascii=False, indent=2) + "\n"
    )


def _read_state(path: Path) -> dict:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return raw if isinstance(raw, dict) else {}


def _purge_archive(
    archive_dir: Path, retention_days: int, now: datetime
) -> list[str]:
    purged: list[str] = []
    if not archive_dir.is_dir():
        return purged
    cutoff = now - timedelta(days=max(0, retention_days))
    for child in archive_dir.iterdir():
        if not child.is_dir():
            continue
        try:
            mtime = datetime.fromtimestamp(child.stat().st_mtime, tz=timezone.utc)
        except OSError:
            continue
        if mtime < cutoff:
            try:
                shutil.rmtree(child)
                purged.append(child.name)
            except OSError:
                continue
    return purged


def _sweep_locked(
    *,
    tools_dir: Path,
    registry_path: Path,
    archive_dir: Path,
    state_path: Path,
    current: datetime,
    report: ToolLifecycleReport,
    candidate_ttl_days: int,
    approved_ttl_days: int,
    deprecation_grace_days: int,
    archive_retention_days: int,
    active_tools: set[str] | frozenset[str],
) -> ToolLifecycleReport:
    """在工具清单写锁内执行维护（读-改-写不与其他进程交错）。"""
    try:
        raw = json.loads(registry_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        report.errors.append(f"读取工具清单失败：{exc}")
        return report
    if not isinstance(raw, list):
        report.errors.append("工具清单顶层不是数组，跳过维护")
        return report

    manifests: list[GeneratedToolManifest] = []
    orphans: list[dict] = []
    for index, item in enumerate(raw):
        try:
            manifests.append(GeneratedToolManifest.model_validate(item))
        except Exception as exc:
            report.errors.append(f"清单第 {index} 条非法：{exc}")
            if isinstance(item, dict):
                orphans.append(item)

    kept: list[GeneratedToolManifest] = []
    changed = False
    candidate_cutoff = current - timedelta(days=max(0, candidate_ttl_days))
    approved_cutoff = current - timedelta(days=max(0, approved_ttl_days))
    for manifest in manifests:
        if manifest.pinned or manifest.tool_name in active_tools:
            kept.append(manifest)
            continue
        source_path = tools_dir / manifest.entry_file
        base = _age_base(manifest, source_path)
        if not manifest.created_at:
            manifest = manifest.model_copy(update={"created_at": base.isoformat()})
            changed = True

        if manifest.deprecated_at:
            deprecated_at = _parse_time(manifest.deprecated_at) or base
            if current - deprecated_at >= timedelta(
                days=max(0, deprecation_grace_days)
            ):
                try:
                    _archive_tool(
                        manifest,
                        tools_dir,
                        archive_dir,
                        "deprecated_expired",
                        current,
                    )
                    report.archived.append(manifest.tool_name)
                    changed = True
                except OSError as exc:
                    report.errors.append(f"归档 {manifest.tool_name} 失败：{exc}")
                    kept.append(manifest)
                continue
            kept.append(manifest)
            continue

        if manifest.review_status != "approved" or manifest.safety_status in {
            "failed",
            "quarantined",
        }:
            if base <= candidate_cutoff:
                try:
                    _archive_tool(
                        manifest,
                        tools_dir,
                        archive_dir,
                        "candidate_expired",
                        current,
                    )
                    report.archived.append(manifest.tool_name)
                    changed = True
                except OSError as exc:
                    report.errors.append(f"归档 {manifest.tool_name} 失败：{exc}")
                    kept.append(manifest)
                continue
            kept.append(manifest)
            continue

        if base <= approved_cutoff:
            manifest = manifest.model_copy(
                update={"deprecated_at": current.isoformat()}
            )
            changed = True
            report.deprecated.append(manifest.tool_name)
        kept.append(manifest)

    if changed:
        try:
            atomic_write_text(
                registry_path,
                json.dumps(
                    [item.model_dump(mode="json") for item in kept] + orphans,
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
            )
            report.swept = True
        except OSError as exc:
            report.errors.append(f"写回工具清单失败：{exc}")
            return report

    report.purged = _purge_archive(archive_dir, archive_retention_days, current)
    try:
        atomic_write_text(
            state_path,
            json.dumps({"last_sweep_at": current.isoformat()}, indent=2) + "\n",
        )
    except OSError as exc:
        report.errors.append(f"写入维护状态失败：{exc}")
    return report


def sweep_generated_tools(
    *,
    tools_dir: str | Path | None = None,
    registry_path: str | Path | None = None,
    archive_dir: str | Path | None = None,
    settings: Settings | None = None,
    now: datetime | None = None,
    candidate_ttl_days: int = 7,
    approved_ttl_days: int = 30,
    deprecation_grace_days: int = 7,
    archive_retention_days: int = 90,
    active_tools: set[str] | frozenset[str] = frozenset(),
    force: bool = True,
    min_interval_minutes: int = 0,
) -> ToolLifecycleReport:
    """执行一次生成工具生命周期维护。

    Args:
        tools_dir: 生成工具目录；未提供时按默认设置解析。
        registry_path: registry.json 路径；未提供时取 ``tools_dir/registry.json``。
        archive_dir: 归档目录；未提供时取 ``{tool_store}/archive``。
        now: 当前时间（测试注入用）。
        candidate_ttl_days: 候选/失败工具保留天数。
        approved_ttl_days: 已批准工具无使用保留天数。
        deprecation_grace_days: 弃用宽限天数。
        archive_retention_days: 归档文件保留天数。
        active_tools: 当前任务正在使用的工具名（永不清理）。
        force: 忽略最小间隔立即执行。
        min_interval_minutes: 自动维护最小间隔（分钟）。
    """
    resolved_settings = settings if settings is not None else get_settings()
    resolved_tools = (
        Path(tools_dir)
        if tools_dir is not None
        else resolve_generated_tools_dir(resolved_settings)
    )
    resolved_registry = (
        Path(registry_path)
        if registry_path is not None
        else (
            Path(tools_dir) / "registry.json"
            if tools_dir is not None
            else resolve_generated_registry_path(resolved_settings)
        )
    )
    resolved_archive = (
        Path(archive_dir)
        if archive_dir is not None
        else resolve_archive_dir(resolved_settings)
    )
    current = _now(now)
    report = ToolLifecycleReport()

    if not resolved_registry.is_file():
        return report

    state_path = resolved_tools / _STATE_FILENAME
    if not force and min_interval_minutes > 0:
        last = _parse_time(str(_read_state(state_path).get("last_sweep_at", "")))
        if last is not None and current - last < timedelta(
            minutes=min_interval_minutes
        ):
            return report

    try:
        with registry_write_lock(resolved_registry):
            return _sweep_locked(
                tools_dir=resolved_tools,
                registry_path=resolved_registry,
                archive_dir=resolved_archive,
                state_path=state_path,
                current=current,
                report=report,
                candidate_ttl_days=candidate_ttl_days,
                approved_ttl_days=approved_ttl_days,
                deprecation_grace_days=deprecation_grace_days,
                archive_retention_days=archive_retention_days,
                active_tools=active_tools,
            )
    except ToolStoreLockError as exc:
        report.errors.append(f"获取工具清单写锁失败：{exc}")
        return report


__all__ = ["ToolLifecycleReport", "sweep_generated_tools"]

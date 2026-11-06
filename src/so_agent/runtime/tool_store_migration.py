"""旧工具库迁移：把沙箱内 ``generated_tools`` 复制到主控侧 ToolStore。

迁移只做一次性复制（不移动、不删除旧目录），以哈希保证源码一致，并且
幂等：目标清单已存在同名工具时跳过。所有写入在工具清单写锁内完成，
与运行中的注册表/生命周期维护互不覆盖。
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from so_agent.integrity import file_matches_hash, sha256_file
from so_agent.models import GeneratedToolManifest
from so_agent.runtime.tool_store import (
    ToolStoreLockError,
    atomic_write_text,
    registry_write_lock,
)


@dataclass
class ToolMigrationReport:
    """一次迁移的结果摘要。"""

    migrated: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    source_dir: str = ""
    target_dir: str = ""

    @property
    def changed(self) -> bool:
        return bool(self.migrated)


def _read_manifest_entries(path: Path) -> tuple[list[dict], list[str]]:
    if not path.is_file():
        return [], []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [], [f"读取清单失败：{path}（{exc}）"]
    if not isinstance(raw, list):
        return [], [f"清单顶层不是数组：{path}"]
    return [item for item in raw if isinstance(item, dict)], []


def migrate_legacy_generated_tools(
    legacy_dir: str | Path,
    target_dir: str | Path,
) -> ToolMigrationReport:
    """把 ``legacy_dir`` 中的生成工具复制进 ``target_dir``（幂等）。

    - 旧目录缺失或与目标目录相同：直接返回空报告；
    - 目标已存在同名工具：跳过（以目标为准）；
    - 源码缺失或哈希不匹配：跳过并记录错误（不迁移被篡改的工具）；
    - 旧清单缺 ``content_hash``：迁移时按源码计算并写入。
    """
    source = Path(legacy_dir).expanduser()
    target = Path(target_dir).expanduser()
    report = ToolMigrationReport(
        source_dir=str(source), target_dir=str(target)
    )
    if not source.is_dir():
        return report
    try:
        if source.resolve() == target.resolve():
            return report
    except OSError:
        pass

    legacy_registry = source / "registry.json"
    target_registry = target / "registry.json"
    legacy_entries, read_errors = _read_manifest_entries(legacy_registry)
    report.errors.extend(read_errors)
    if not legacy_entries:
        return report

    target.mkdir(parents=True, exist_ok=True)
    try:
        with registry_write_lock(target_registry):
            existing_entries, _ = _read_manifest_entries(target_registry)
            existing_names = {
                str(item.get("tool_name") or "")
                for item in existing_entries
            }
            migrated_entries: list[dict] = []
            for item in legacy_entries:
                try:
                    manifest = GeneratedToolManifest.model_validate(item)
                except Exception as exc:
                    report.errors.append(
                        f"旧清单条目非法（{item.get('tool_name', '?')}）：{exc}"
                    )
                    continue
                if manifest.tool_name in existing_names:
                    report.skipped.append(manifest.tool_name)
                    continue
                source_path = source / manifest.entry_file
                if not source_path.is_file():
                    report.errors.append(
                        f"源码缺失，跳过迁移：{manifest.tool_name}"
                    )
                    continue
                if manifest.content_hash and not file_matches_hash(
                    source_path, manifest.content_hash
                ):
                    report.errors.append(
                        f"源码哈希不匹配（可能被篡改），跳过迁移：{manifest.tool_name}"
                    )
                    continue
                content_hash = manifest.content_hash or sha256_file(source_path)
                destination = target / manifest.entry_file
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_path, destination)
                schema_name = f"{manifest.tool_name}.mcp_schema.json"
                schema_source = source / schema_name
                if schema_source.is_file():
                    shutil.copy2(schema_source, target / schema_name)
                migrated = manifest.model_copy(
                    update={"content_hash": content_hash}
                )
                migrated_entries.append(migrated.model_dump(mode="json"))
                existing_names.add(manifest.tool_name)
                report.migrated.append(manifest.tool_name)

            if migrated_entries:
                payload = existing_entries + migrated_entries
                atomic_write_text(
                    target_registry,
                    json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                )
    except ToolStoreLockError as exc:
        report.errors.append(f"获取工具清单写锁失败：{exc}")
    except OSError as exc:
        report.errors.append(f"迁移写入失败：{exc}")
    return report


__all__ = ["ToolMigrationReport", "migrate_legacy_generated_tools"]

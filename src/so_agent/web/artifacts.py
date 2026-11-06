"""产物扫描、路径校验与媒体类型判定。"""

from __future__ import annotations

from datetime import datetime
from datetime import timezone
from pathlib import Path
from so_agent.web.text import tr
from typing import Any
import mimetypes

from so_agent.web.common import (
    ArtifactError,
    MAX_ARTIFACTS,
    MAX_ARTIFACT_BYTES,
    WebTask,
    _INTERNAL_DIR_NAMES,
    _INTERNAL_DIR_PREFIXES,
    _INTERNAL_FILE_NAMES,
    _TEXT_EXTENSIONS,
)


class ArtifactMixin:

    def _scan_artifacts(self, task: WebTask) -> list[dict[str, Any]]:
        root = task.sandbox_dir
        if not root.is_dir():
            return []
        artifacts: list[dict[str, Any]] = []
        try:
            paths = sorted(root.rglob("*"))
        except OSError:
            return []
        for path in paths:
            if len(artifacts) >= MAX_ARTIFACTS:
                break
            try:
                if not path.is_file() or path.is_symlink():
                    continue
                relative = path.relative_to(root)
                if self._is_hidden_or_internal(relative):
                    continue
                stat = path.stat()
            except (OSError, ValueError):
                continue
            if stat.st_size > MAX_ARTIFACT_BYTES:
                continue
            artifacts.append(
                {
                    "path": relative.as_posix(),
                    "name": path.name,
                    "size": stat.st_size,
                    "modified_at": datetime.fromtimestamp(
                        stat.st_mtime, tz=timezone.utc
                    ).isoformat(timespec="seconds"),
                    "kind": self._artifact_kind(path),
                }
            )
        artifacts.sort(key=lambda item: item["modified_at"], reverse=True)
        return artifacts


    @staticmethod
    def resolve_artifact(
        task: WebTask, relative_path: str, *, lang: str = "zh"
    ) -> Path:
        """把请求路径解析为沙箱内真实文件（拒绝越界、内部目录与符号链接）。"""
        candidate_rel = Path(relative_path)
        if candidate_rel.is_absolute() or ".." in candidate_rel.parts:
            raise ArtifactError(tr(lang, "非法产物路径", "Invalid artifact path"))
        root = task.sandbox_dir.resolve()
        candidate = (root / candidate_rel).resolve()
        try:
            relative = candidate.relative_to(root)
        except ValueError as exc:
            raise ArtifactError(
                tr(lang, "产物路径越界", "Artifact path escapes the task sandbox")
            ) from exc
        if ArtifactMixin._is_hidden_or_internal(relative):
            raise ArtifactError(
                tr(lang, "该路径不对外提供", "This path is not exposed")
            )
        if candidate.is_symlink() or not candidate.is_file():
            raise ArtifactError(tr(lang, "产物不存在", "Artifact not found"))
        if candidate.stat().st_size > MAX_ARTIFACT_BYTES:
            raise ArtifactError(
                tr(lang, "产物超过下载大小上限", "Artifact exceeds download size limit")
            )
        return candidate


    @staticmethod
    def _is_hidden_or_internal(relative: Path) -> bool:
        parts = relative.parts
        if any(part.startswith(".") for part in parts):
            return True
        if any(
            part in _INTERNAL_DIR_NAMES
            or part.startswith(_INTERNAL_DIR_PREFIXES)
            for part in parts[:-1]
        ):
            return True
        return relative.name in _INTERNAL_FILE_NAMES


    @staticmethod
    def _artifact_kind(path: Path) -> str:
        suffix = path.suffix.lower()
        if suffix in {".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg"}:
            return "image"
        if suffix in {".py", ".js", ".ts", ".html", ".css", ".json", ".yaml", ".yml", ".md", ".txt", ".csv", ".sql"}:
            return "text"
        if suffix in {".zip", ".tar", ".gz", ".7z", ".rar"}:
            return "archive"
        if suffix in {".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx"}:
            return "document"
        guessed, _ = mimetypes.guess_type(path.name)
        return "text" if guessed and guessed.startswith("text/") else "other"


    def artifact_media_type(self, path: Path) -> str:
        override = _TEXT_EXTENSIONS.get(path.suffix.lower())
        if override is not None:
            return override
        guessed, _ = mimetypes.guess_type(path.name)
        if guessed and (
            guessed.startswith("text/")
            or guessed.startswith("image/")
            or guessed in {"application/pdf", "application/json"}
        ):
            return guessed
        return "application/octet-stream"


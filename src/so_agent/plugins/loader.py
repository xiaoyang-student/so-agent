"""外部插件发现与加载（目录约定 + fail-closed 安全校验）。

目录约定::

    plugins/
      <kind>/                 # sandbox / llm / mcp / rules / observability
        <name>/
          plugin.yaml         # 清单（id=<kind>.<name>）
          plugin.py           # 入口模块（entrypoint=plugin.py:<factory>）

安全边界：

- 只加载显式启用（``plugins_enabled``）的插件；未启用目录不会被导入；
- 清单额外字段、未知权限、明文密钥、API 主版本不符一律拒绝；
- entrypoint 必须是插件目录内的相对 ``.py`` 路径，``resolve`` 后仍须
  位于插件目录内（阻止 ``..`` 与符号链接越界）；
- 可选用入口文件 SHA-256 钉住（``sha256`` 字段或
  ``plugins_require_hash`` 强制），摘要不匹配拒绝加载；
- 导入异常封装为 PluginLoadError，不污染内核异常语义。

信任模型：当前为进程内加载（与内核同权限）。仅供受信任的私有插件
使用；来源不可信的插件必须在单独进程/沙箱中运行（后续 provider 层
扩展，不在本模块承诺）。
"""

from __future__ import annotations

import hashlib
import hmac
import importlib.util
import sys
from pathlib import Path

import yaml

from so_agent.plugins.contracts import (
    PLUGIN_KINDS,
    PluginError,
    PluginFactory,
    PluginLoadError,
    PluginManifest,
    PluginManifestError,
)


def sha256_file(path: Path) -> str:
    """计算文件 SHA-256（小写 hex）。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class PluginLoader:
    """从插件根目录发现、校验并导入插件工厂。"""

    def __init__(self, *, require_hash: bool = False) -> None:
        self._require_hash = bool(require_hash)

    def discover(
        self, root: str | Path
    ) -> list[tuple[PluginManifest, Path]]:
        """发现 ``root/<kind>/<name>/plugin.yaml`` 清单（顺序稳定）。"""
        base = Path(root)
        if not base.exists():
            return []
        if not base.is_dir():
            raise PluginLoadError(f"插件根路径不是目录：{base}")
        discovered: list[tuple[PluginManifest, Path]] = []
        seen: dict[str, Path] = {}
        for kind_dir in sorted(base.iterdir()):
            if not kind_dir.is_dir() or kind_dir.name not in PLUGIN_KINDS:
                continue
            for plugin_dir in sorted(kind_dir.iterdir()):
                if not plugin_dir.is_dir():
                    continue
                manifest_path = self._manifest_path(plugin_dir)
                if manifest_path is None:
                    continue
                manifest, plugin_root = self.load_manifest(manifest_path)
                if manifest.id in seen:
                    raise PluginLoadError(
                        f"插件 id 重复：{manifest.id}（{seen[manifest.id]} 与 "
                        f"{plugin_root}）"
                    )
                seen[manifest.id] = plugin_root
                discovered.append((manifest, plugin_root))
        return discovered

    def load_manifest(
        self, manifest_path: str | Path
    ) -> tuple[PluginManifest, Path]:
        """读取并校验清单；目录名与 kind/name 必须一致。"""
        path = Path(manifest_path)
        plugin_dir = path.parent.resolve()
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as exc:
            raise PluginManifestError(f"插件清单读取失败：{path}：{exc}") from exc
        manifest = PluginManifest.from_mapping(raw)
        if plugin_dir.parent.name != manifest.kind.value:
            raise PluginManifestError(
                f"插件目录 {plugin_dir} 与清单 kind {manifest.kind.value!r} 不一致"
            )
        if plugin_dir.name != manifest.name:
            raise PluginManifestError(
                f"插件目录名 {plugin_dir.name!r} 与清单 name "
                f"{manifest.name!r} 不一致"
            )
        return manifest, plugin_dir

    def load(self, manifest: PluginManifest, plugin_dir: Path) -> PluginFactory:
        """导入入口模块并返回工厂；失败一律抛 PluginLoadError。"""
        entry_path = self._resolve_entrypoint(manifest, plugin_dir)
        expected = manifest.sha256 or ""
        if expected or self._require_hash:
            actual = sha256_file(entry_path)
            if not expected:
                raise PluginLoadError(
                    f"插件 {manifest.id} 未声明 sha256，当前要求摘要钉住"
                )
            if not hmac.compare_digest(actual, expected):
                raise PluginLoadError(
                    f"插件 {manifest.id} 入口摘要不匹配：期望 {expected}，"
                    f"实际 {actual}"
                )
        module_name = (
            f"_so_agent_plugin_{manifest.kind.value}_{manifest.name}_"
            f"{sha256_file(entry_path)[:8]}"
        )
        spec = importlib.util.spec_from_file_location(module_name, entry_path)
        if spec is None or spec.loader is None:
            raise PluginLoadError(f"无法构造插件模块：{entry_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        except Exception as exc:
            sys.modules.pop(module_name, None)
            raise PluginLoadError(
                f"插件 {manifest.id} 导入失败：{type(exc).__name__}: {exc}"
            ) from exc
        _, attribute = manifest.entrypoint.rsplit(":", 1)
        factory = getattr(module, attribute, None)
        if not callable(factory):
            raise PluginLoadError(
                f"插件 {manifest.id} 入口 {attribute!r} 不存在或不可调用"
            )
        return factory

    @staticmethod
    def _manifest_path(plugin_dir: Path) -> Path | None:
        for name in ("plugin.yaml", "plugin.yml"):
            candidate = plugin_dir / name
            if candidate.is_file():
                return candidate
        return None

    @staticmethod
    def _resolve_entrypoint(manifest: PluginManifest, plugin_dir: Path) -> Path:
        relative, _ = manifest.entrypoint.rsplit(":", 1)
        root = plugin_dir.resolve()
        entry_path = (root / relative).resolve()
        if entry_path != root and root not in entry_path.parents:
            raise PluginLoadError(
                f"插件 {manifest.id} 入口越出插件目录：{relative!r}"
            )
        if not entry_path.is_file():
            raise PluginLoadError(
                f"插件 {manifest.id} 入口文件不存在：{entry_path}"
            )
        return entry_path


__all__ = ["PluginLoader", "PluginError", "sha256_file"]

"""插件管理器：内置目录 + 显式启用的外部插件，失败即拒绝启动。

启用规则（``plugins_enabled``）：

- 空字符串：只登记内置插件，不扫描外部目录；
- ``"*"``：加载 ``plugins_dir`` 下发现的全部外部插件；
- 逗号分隔 id：仅加载列出的插件；任一 id 既非内置也未发现时抛
  :class:`PluginNotFoundError`，不做静默跳过。

``plugins_dir`` 为相对路径时按当前工作目录解析。外部沙箱插件加载后
同时注册到 ``runtime.sandbox`` 的后端表，可用 ``sandbox_backend=<name>``
选择；其余类别登记在 :class:`PluginRegistry` 中，由消费方按需创建。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from so_agent.plugins.builtin import register_builtins
from so_agent.plugins.contracts import (
    PluginError,
    PluginKind,
    PluginManifest,
    PluginNotFoundError,
    PluginRegistry,
    validate_plugin_id,
)
from so_agent.plugins.loader import PluginLoader

_ALL = "*"


def parse_enabled(raw: str | None) -> tuple[str, ...] | None:
    """解析启用列表：``None`` 表示全部，空元组表示仅内置。"""
    if raw is None:
        return ()
    text = raw.strip()
    if not text:
        return ()
    tokens = [token.strip() for token in text.split(",") if token.strip()]
    if _ALL in tokens:
        if len(tokens) > 1:
            raise PluginError("plugins_enabled 中的 '*' 不能与其他插件 id 混用")
        return None
    result: list[str] = []
    for token in tokens:
        validate_plugin_id(token)
        if token not in result:
            result.append(token)
    return tuple(result)


class PluginManager:
    """按配置登记内置插件并加载显式启用的外部插件。"""

    def __init__(
        self,
        settings: Any = None,
        *,
        cwd: str | Path | None = None,
        registry: PluginRegistry | None = None,
        loader: PluginLoader | None = None,
    ) -> None:
        self.settings = settings
        self.cwd = Path(cwd) if cwd is not None else Path.cwd()
        self.registry = registry if registry is not None else PluginRegistry()
        require_hash = bool(getattr(settings, "plugins_require_hash", False))
        self.loader = loader or PluginLoader(require_hash=require_hash)
        self._loaded = False
        self._external: tuple[PluginManifest, ...] = ()

    @property
    def loaded(self) -> bool:
        """是否已完成一次加载。"""
        return self._loaded

    @property
    def external(self) -> tuple[PluginManifest, ...]:
        """本次加载的外部插件清单。"""
        return self._external

    def load(self) -> PluginRegistry:
        """执行登记与加载；重复调用直接返回已完成的注册表。"""
        if self._loaded:
            return self.registry
        register_builtins(self.registry)
        root = self.resolve_plugins_root()
        discovered = self.loader.discover(root)
        enabled = parse_enabled(getattr(self.settings, "plugins_enabled", ""))
        chosen = self._select(enabled, discovered)
        loaded: list[PluginManifest] = []
        for manifest, plugin_dir in chosen:
            factory = self.loader.load(manifest, plugin_dir)
            self.registry.register(manifest, factory)
            if manifest.kind is PluginKind.SANDBOX:
                from so_agent.runtime.sandbox import register_sandbox_backend

                register_sandbox_backend(manifest.name, factory)
            loaded.append(manifest)
        self._external = tuple(loaded)
        self._loaded = True
        return self.registry

    def resolve_plugins_root(self) -> Path:
        """返回插件根目录（相对路径按构造时的 cwd 解析）。"""
        raw = str(getattr(self.settings, "plugins_dir", "plugins") or "plugins")
        path = Path(raw).expanduser()
        if not path.is_absolute():
            path = self.cwd / path
        return path

    def _select(
        self,
        enabled: tuple[str, ...] | None,
        discovered: list[tuple[PluginManifest, Path]],
    ) -> list[tuple[PluginManifest, Path]]:
        if enabled is None:
            return list(discovered)
        if not enabled:
            return []
        by_id = {manifest.id: (manifest, plugin_dir) for manifest, plugin_dir in discovered}
        chosen: list[tuple[PluginManifest, Path]] = []
        for plugin_id in enabled:
            kind, name = plugin_id.split(".", 1)
            hit = by_id.get(plugin_id)
            if hit is None:
                # 仅内置插件可直接视为启用；若同 id 的外部插件存在，
                # 仍加载它，由注册表以重复注册拒绝覆盖内置实现。
                if name in self.registry.names(kind):
                    continue
                available = sorted(
                    set(by_id)
                    | {
                        f"{kind}.{name}"
                        for kind in (item.value for item in PluginKind)
                        for name in self.registry.names(kind)
                    }
                )
                raise PluginNotFoundError(
                    f"启用的插件不存在：{plugin_id}；可用插件：{', '.join(available)}"
                )
            chosen.append(hit)
        return chosen


def load_plugins(
    settings: Any = None,
    *,
    cwd: str | Path | None = None,
) -> PluginRegistry:
    """加载插件并返回注册表（启动期调用；失败向上抛出 PluginError）。"""
    return PluginManager(settings, cwd=cwd).load()

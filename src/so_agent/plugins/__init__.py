"""插件子系统：契约、内置目录、外部加载与管理。

控制面通过 :func:`load_plugins` 在启动期完成登记；外部插件仅在
``plugins_enabled`` 显式启用时导入，校验失败即拒绝启动。
"""

from so_agent.plugins.builtin import register_builtins
from so_agent.plugins.contracts import (
    KNOWN_PERMISSIONS,
    PLUGIN_API_VERSION,
    PLUGIN_KINDS,
    PluginEntry,
    PluginError,
    PluginKind,
    PluginLoadError,
    PluginManifest,
    PluginManifestError,
    PluginNotFoundError,
    PluginRegistry,
    validate_plugin_id,
)
from so_agent.plugins.loader import PluginLoader, sha256_file
from so_agent.plugins.manager import PluginManager, load_plugins, parse_enabled

__all__ = [
    "KNOWN_PERMISSIONS",
    "PLUGIN_API_VERSION",
    "PLUGIN_KINDS",
    "PluginEntry",
    "PluginError",
    "PluginKind",
    "PluginLoadError",
    "PluginLoader",
    "PluginManager",
    "PluginManifest",
    "PluginManifestError",
    "PluginNotFoundError",
    "PluginRegistry",
    "load_plugins",
    "parse_enabled",
    "register_builtins",
    "sha256_file",
    "validate_plugin_id",
]

"""内置能力目录：把现有实现登记为插件，供统一选择与外部插件对照。

内置插件只做薄适配，不复制实现：

- sandbox：沿用 ``runtime.sandbox`` 的后端注册表（local / docker /
  podman / auto / cloud）；
- llm：OpenAI 兼容客户端与多供应商故障转移（``runtime.model_gateway``）；
- mcp：外部 MCP 目录管理（``mcp.directory``）；
- rules：可机检验收规则评估（``runtime.verification``）；
- observability：事件日志（``runtime.events``；OTel 由 telemetry 模块
  在同一事件流上导出）。

沙箱类别同时接入 ``register_sandbox_backend``：外部插件启用后即可用
``sandbox_backend=<name>`` 选择；其余类别先进入目录，消费方按需通过
``PluginRegistry.create`` 取用（逐步迁移，不改变现有默认路径）。
"""

from __future__ import annotations

from typing import Any

from so_agent.plugins.contracts import PluginManifest, PluginRegistry

_BUILTIN_VERSION = "1.0.0"

_CAPABILITIES: dict[str, list[str]] = {
    "sandbox": ["execute"],
    "llm": ["chat", "failover"],
    "mcp": ["directory", "discovery"],
    "rules": ["verification"],
    "observability": ["events", "metrics", "audit"],
}


def _builtin_manifest(kind: str, name: str) -> PluginManifest:
    return PluginManifest(
        id=f"{kind}.{name}",
        kind=kind,  # type: ignore[arg-type]
        version=_BUILTIN_VERSION,
        entrypoint="builtin.py:factory",
        description=f"内置 {kind} 能力：{name}",
        capabilities=list(_CAPABILITIES.get(kind, ())),
    )


def _sandbox_factory(name: str):
    def factory(*, context: Any = None, sandbox_dir: Any = None,
                settings: Any = None, python_executable: str | None = None) -> Any:
        from so_agent.runtime.sandbox import create_sandbox

        return create_sandbox(
            context=context,
            sandbox_dir=sandbox_dir,
            settings=settings,
            backend=name,
            python_executable=python_executable,
        )

    return factory


def _llm_factory(*, config: Any, config_path: Any = None,
                 client_factory: Any = None, http_timeout: Any = None) -> Any:
    from so_agent.runtime.model_gateway import build_chat_model

    kwargs: dict[str, Any] = {"config": config, "config_path": config_path}
    if client_factory is not None:
        kwargs["client_factory"] = client_factory
    if http_timeout is not None:
        kwargs["http_timeout"] = http_timeout
    return build_chat_model(**kwargs)


def _register(
    registry: PluginRegistry, kind: str, name: str, factory: Any
) -> None:
    if name in registry.names(kind):
        return
    registry.register(_builtin_manifest(kind, name), factory, builtin=True)


def register_builtins(registry: PluginRegistry) -> None:
    """把内核现有能力登记为插件（幂等，内置项不覆盖外部同名项）。"""
    from so_agent.runtime.sandbox import available_sandbox_backends

    for name in available_sandbox_backends():
        _register(registry, "sandbox", name, _sandbox_factory(name))

    _register(registry, "llm", "openai_compatible", _llm_factory)

    from so_agent.mcp.directory import MCPDirectoryManager

    _register(registry, "mcp", "directory", MCPDirectoryManager)

    from so_agent.runtime.verification import evaluate_verification_rules

    _register(registry, "rules", "builtin", evaluate_verification_rules)

    from so_agent.runtime.events import EventLogger

    _register(registry, "observability", "events", EventLogger)

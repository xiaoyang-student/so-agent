"""模型接入网关：单供应商保持原行为，多供应商按顺序故障转移。

配置沿用各 Agent 的 ``api_config.yaml``，新增可选字段::

    fallbacks:
      - provider: backup
        model: backup-model
        base_url: https://backup.example.com/v1
        timeout: 120
        retry: 1
        api_key_env: SO_AGENT_BACKUP_API_KEY
    failover:
      failure_threshold: 3
      cooldown_seconds: 60

未配置 ``fallbacks`` 时不构造代理，客户端与旧行为完全一致。
故障转移只处理"换一个供应商可能成功"的错误（连接、超时、429、5xx、
认证/权限/模型不存在）；400 等请求本身非法的错误直接抛出，避免掩盖问题。
流式调用只在收到首个响应前切换；已开始输出后不重放，防止重复内容。
"""

from __future__ import annotations

import asyncio
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import yaml
from agents import OpenAIChatCompletionsModel
from openai import APIConnectionError, APITimeoutError, AsyncOpenAI
from pydantic import ValidationError

from so_agent.runtime.config_loader import (
    AgentAPIConfig,
    ConfigLoadError,
    resolve_model_http_timeout,
)
from so_agent.runtime.logging_setup import get_logger

logger = get_logger("runtime.model_gateway")

DEFAULT_FAILURE_THRESHOLD = 3
DEFAULT_COOLDOWN_SECONDS = 60.0

_FAILOVER_STATUS = frozenset({401, 403, 404, 408, 409, 429})


@dataclass(frozen=True)
class FailoverSettings:
    failure_threshold: int = DEFAULT_FAILURE_THRESHOLD
    cooldown_seconds: float = DEFAULT_COOLDOWN_SECONDS


@dataclass
class _ProviderHealth:
    consecutive_failures: int = 0
    cooldown_until: float = 0.0


@dataclass
class _Provider:
    config: AgentAPIConfig = field(repr=False)
    client: Any = field(repr=False)

    @property
    def signature(self) -> tuple[str, str, str]:
        return (
            self.config.provider,
            self.config.base_url or "",
            self.config.model,
        )

    @property
    def label(self) -> str:
        return f"{self.config.provider}/{self.config.model}"


_HEALTH: dict[tuple[str, str, str], _ProviderHealth] = {}
_HEALTH_LOCK = threading.Lock()
_FAILOVER_COUNT = 0


def reset_gateway_state() -> None:
    """清空健康状态与转移计数（测试与运维排查使用）。"""
    global _FAILOVER_COUNT
    with _HEALTH_LOCK:
        _HEALTH.clear()
        _FAILOVER_COUNT = 0


def gateway_snapshot() -> dict[str, Any]:
    """返回不含密钥的网关健康快照。"""
    now = time.monotonic()
    with _HEALTH_LOCK:
        providers = [
            {
                "provider": signature[0],
                "base_url": signature[1],
                "model": signature[2],
                "consecutive_failures": health.consecutive_failures,
                "cooldown_remaining_seconds": round(
                    max(0.0, health.cooldown_until - now), 1
                ),
            }
            for signature, health in _HEALTH.items()
        ]
        return {"failovers": _FAILOVER_COUNT, "providers": providers}


def is_failover_error(exc: BaseException) -> bool:
    """判断异常是否值得切换到下一个供应商。"""
    if isinstance(exc, (APIConnectionError, APITimeoutError)):
        return True
    status = getattr(exc, "status_code", None)
    if status is None:
        response = getattr(exc, "response", None)
        status = getattr(response, "status_code", None)
    if not isinstance(status, int):
        return False
    return status in _FAILOVER_STATUS or 500 <= status < 600


def _record_failure(provider: _Provider, settings: FailoverSettings) -> None:
    with _HEALTH_LOCK:
        health = _HEALTH.setdefault(provider.signature, _ProviderHealth())
        health.consecutive_failures += 1
        if health.consecutive_failures >= settings.failure_threshold:
            health.cooldown_until = time.monotonic() + settings.cooldown_seconds


def _record_success(provider: _Provider) -> None:
    with _HEALTH_LOCK:
        health = _HEALTH.setdefault(provider.signature, _ProviderHealth())
        health.consecutive_failures = 0
        health.cooldown_until = 0.0


def _in_cooldown(provider: _Provider) -> bool:
    with _HEALTH_LOCK:
        health = _HEALTH.get(provider.signature)
        return health is not None and health.cooldown_until > time.monotonic()


def _read_config_document(config_path: str | Path | None) -> dict[str, Any]:
    if config_path is None:
        return {}
    path = Path(config_path).expanduser()
    if not path.is_file():
        return {}
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ConfigLoadError(f"模型网关配置读取失败 {path}：{exc}") from exc
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ConfigLoadError(f"模型网关配置根节点必须是映射：{path}")
    return raw


def _parse_failover_settings(raw: Any) -> FailoverSettings:
    if raw is None:
        return FailoverSettings()
    if not isinstance(raw, dict):
        raise ConfigLoadError("failover 配置必须是映射")
    try:
        threshold = int(raw.get("failure_threshold", DEFAULT_FAILURE_THRESHOLD))
        cooldown = float(raw.get("cooldown_seconds", DEFAULT_COOLDOWN_SECONDS))
    except (TypeError, ValueError) as exc:
        raise ConfigLoadError(f"failover 配置数值非法：{exc}") from exc
    return FailoverSettings(
        failure_threshold=max(1, threshold),
        cooldown_seconds=max(0.0, cooldown),
    )


def load_fallback_configs(
    config_path: str | Path | None,
    *,
    primary: AgentAPIConfig | None = None,
) -> tuple[list[AgentAPIConfig], FailoverSettings]:
    """读取配置中的 fallbacks 链；未配置时返回空列表与默认设置。"""
    raw = _read_config_document(config_path)
    settings = _parse_failover_settings(raw.get("failover"))
    entries = raw.get("fallbacks")
    if entries is None:
        return [], settings
    if not isinstance(entries, list):
        raise ConfigLoadError("fallbacks 必须是列表")
    seen: set[tuple[str, str, str]] = set()
    if primary is not None:
        seen.add(
            (
                primary.provider,
                primary.base_url or "",
                primary.model,
            )
        )
    configs: list[AgentAPIConfig] = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ConfigLoadError(f"fallbacks[{index}] 必须是映射")
        try:
            config = AgentAPIConfig.model_validate(entry)
        except ValidationError as exc:
            raise ConfigLoadError(f"fallbacks[{index}] 配置非法：{exc}") from exc
        signature = (config.provider, config.base_url or "", config.model)
        if signature in seen:
            continue
        seen.add(signature)
        configs.append(config)
    return configs, settings


def _make_client(
    config: AgentAPIConfig,
    client_factory: Callable[..., Any],
    http_timeout: Any | None = None,
) -> Any:
    return client_factory(
        api_key=config.resolve_api_key(),
        base_url=config.base_url or None,
        timeout=(
            http_timeout
            if http_timeout is not None
            else resolve_model_http_timeout(config)
        ),
        max_retries=config.retry,
    )


class _Completions:
    def __init__(self, gateway: "FailoverOpenAI") -> None:
        self._gateway = gateway

    async def create(self, **kwargs: Any) -> Any:
        return await self._gateway._create(**kwargs)


class _ChatNamespace:
    def __init__(self, gateway: "FailoverOpenAI") -> None:
        self.completions = _Completions(gateway)


class FailoverOpenAI:
    """兼容 ``AsyncOpenAI.chat.completions.create`` 的故障转移代理。

    只实现当前代码实际使用的表面；未知属性透传给主供应商客户端。
    """

    def __init__(
        self,
        providers: list[_Provider],
        settings: FailoverSettings,
    ) -> None:
        if not providers:
            raise ValueError("至少需要一个模型供应商")
        self._providers = providers
        self._settings = settings
        self._chat = _ChatNamespace(self)

    @property
    def base_url(self) -> Any:
        return getattr(self._providers[0].client, "base_url", None)

    @property
    def chat(self) -> _ChatNamespace:
        return self._chat

    def with_options(self, **options: Any) -> "FailoverOpenAI":
        providers: list[_Provider] = []
        for provider in self._providers:
            with_options = getattr(provider.client, "with_options", None)
            client = (
                with_options(**options) if callable(with_options) else provider.client
            )
            providers.append(_Provider(config=provider.config, client=client))
        return FailoverOpenAI(providers, self._settings)

    async def close(self) -> None:
        closers = []
        for provider in self._providers:
            close = getattr(provider.client, "close", None)
            if callable(close):
                closers.append(close())
        if closers:
            await asyncio.gather(*closers, return_exceptions=True)

    async def aclose(self) -> None:
        await self.close()

    def _ordered_providers(self) -> list[_Provider]:
        available = [p for p in self._providers if not _in_cooldown(p)]
        return available or list(self._providers)

    async def _create(self, **kwargs: Any) -> Any:
        global _FAILOVER_COUNT
        providers = self._ordered_providers()
        last_error: BaseException | None = None
        for index, provider in enumerate(providers):
            call_kwargs = dict(kwargs)
            call_kwargs["model"] = provider.config.model
            if "timeout" in call_kwargs:
                call_kwargs["timeout"] = resolve_model_http_timeout(provider.config)
            try:
                result = await provider.client.chat.completions.create(**call_kwargs)
            except Exception as exc:
                if not is_failover_error(exc):
                    raise
                _record_failure(provider, self._settings)
                last_error = exc
                if index == len(providers) - 1:
                    raise
                with _HEALTH_LOCK:
                    _FAILOVER_COUNT += 1
                logger.warning(
                    "模型供应商 %s 不可用（%s），切换到下一个供应商",
                    provider.label,
                    type(exc).__name__,
                )
                continue
            _record_success(provider)
            return result
        if last_error is not None:
            raise last_error
        raise RuntimeError("模型网关没有可用供应商")

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        providers = object.__getattribute__(self, "_providers")
        return getattr(providers[0].client, name)


def build_openai_client(
    config: AgentAPIConfig,
    *,
    config_path: str | Path | None = None,
    client_factory: Callable[..., Any] = AsyncOpenAI,
    http_timeout: Any | None = None,
) -> Any:
    """构造客户端：无 fallbacks 时返回原生客户端，否则返回故障转移代理。"""
    fallbacks, settings = load_fallback_configs(config_path, primary=config)
    primary = _Provider(
        config=config,
        client=_make_client(config, client_factory, http_timeout),
    )
    if not fallbacks:
        return primary.client
    providers = [primary]
    for fallback in fallbacks:
        providers.append(
            _Provider(
                config=fallback,
                client=_make_client(fallback, client_factory),
            )
        )
    return FailoverOpenAI(providers, settings)


def build_chat_model(
    config: AgentAPIConfig,
    *,
    config_path: str | Path | None = None,
    client_factory: Callable[..., Any] = AsyncOpenAI,
    http_timeout: Any | None = None,
) -> OpenAIChatCompletionsModel:
    """构造 OpenAI 兼容聊天模型；多供应商时由网关代理故障转移。"""
    client = build_openai_client(
        config,
        config_path=config_path,
        client_factory=client_factory,
        http_timeout=http_timeout,
    )
    return OpenAIChatCompletionsModel(model=config.model, openai_client=client)

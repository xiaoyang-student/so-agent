"""可选 OpenTelemetry 追踪。

默认关闭；``SO_AGENT_OTEL_ENABLED=1`` 后为任务、分类、子任务、单次
路径与模型调用创建 span。``otel_exporter=none`` 时沿用宿主已配置的
全局 TracerProvider（未配置即为 no-op）；``console`` / ``otlp`` 时由
本进程自建 provider，``otlp`` 读取标准 ``OTEL_EXPORTER_OTLP_*`` 环境变量。

追踪失败绝不影响任务执行；缺少 opentelemetry 依赖时静默降级为 no-op。
"""

from __future__ import annotations

import atexit
import threading
from contextlib import asynccontextmanager, contextmanager
from typing import Any, AsyncIterator, Iterator

from so_agent.runtime.logging_setup import get_logger

logger = get_logger("runtime.telemetry")

_lock = threading.Lock()
_tracer: Any | None = None
_provider: Any | None = None
_initialized = False


def _settings() -> Any:
    from so_agent.config import get_settings

    return get_settings()


def _create_provider(service_name: str, exporter: str) -> Any:
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    provider = TracerProvider(
        resource=Resource.create({"service.name": service_name})
    )
    if exporter == "console":
        import sys

        from opentelemetry.sdk.trace.export import ConsoleSpanExporter

        # 写 stderr，避免污染 CLI 仅输出结构化 JSON 的 stdout 契约。
        provider.add_span_processor(
            BatchSpanProcessor(ConsoleSpanExporter(out=sys.stderr))
        )
    elif exporter == "otlp":
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter,
        )

        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    return provider


def _ensure_tracer() -> Any | None:
    global _tracer, _provider, _initialized
    if _initialized:
        return _tracer
    with _lock:
        if _initialized:
            return _tracer
        _initialized = True
        try:
            settings = _settings()
            if not bool(settings.otel_enabled):
                return None
            from opentelemetry import trace

            provider = trace.get_tracer_provider()
            exporter = str(settings.otel_exporter or "none")
            if exporter != "none":
                from opentelemetry.trace import ProxyTracerProvider

                if isinstance(provider, ProxyTracerProvider):
                    provider = _create_provider(
                        str(settings.otel_service_name or "so-agent"),
                        exporter,
                    )
                    trace.set_tracer_provider(provider)
                    _provider = provider
                    atexit.register(shutdown_telemetry)
            _tracer = trace.get_tracer("so_agent")
        except Exception:
            _tracer = None
            logger.warning(
                "OpenTelemetry 初始化失败，追踪降级为 no-op", exc_info=True
            )
        return _tracer


def telemetry_enabled() -> bool:
    return _ensure_tracer() is not None


def shutdown_telemetry() -> None:
    """刷新并关闭本进程自建的 provider（atexit 自动调用）。"""
    global _tracer, _provider, _initialized
    with _lock:
        provider = _provider
        _tracer = None
        _provider = None
        _initialized = False
    if provider is not None:
        try:
            provider.shutdown()
        except Exception:
            pass


def reset_telemetry_for_tests() -> None:
    """重置初始化状态（不关闭外部注入的全局 provider）。"""
    global _tracer, _provider, _initialized
    with _lock:
        _tracer = None
        _provider = None
        _initialized = False


def set_tracer_for_tests(tracer: Any | None) -> None:
    """注入 tracer（仅测试/嵌入方使用），避免依赖全局单例。"""
    global _tracer, _initialized, _provider
    with _lock:
        _tracer = tracer
        _initialized = True
        _provider = None


def _clean_attributes(attributes: dict[str, Any] | None) -> dict[str, Any]:
    if not attributes:
        return {}
    cleaned: dict[str, Any] = {}
    for key, value in attributes.items():
        if value is None:
            continue
        if isinstance(value, (str, bool, int, float)):
            cleaned[str(key)] = value
        else:
            cleaned[str(key)] = str(value)
    return cleaned


def set_span_attributes(span: Any, attributes: dict[str, Any] | None) -> None:
    if span is None:
        return
    for key, value in _clean_attributes(attributes).items():
        try:
            span.set_attribute(key, value)
        except Exception:
            continue


@contextmanager
def span_scope(
    name: str, attributes: dict[str, Any] | None = None
) -> Iterator[Any | None]:
    """同步 span 上下文；未启用时 yield None。"""
    tracer = _ensure_tracer()
    if tracer is None:
        yield None
        return
    try:
        scope = tracer.start_as_current_span(
            name,
            attributes=_clean_attributes(attributes),
            record_exception=True,
            set_status_on_exception=True,
        )
    except Exception:
        yield None
        return
    with scope as span:
        yield span


@asynccontextmanager
async def async_span_scope(
    name: str, attributes: dict[str, Any] | None = None
) -> AsyncIterator[Any | None]:
    """异步 span 上下文；未启用时 yield None。"""
    tracer = _ensure_tracer()
    if tracer is None:
        yield None
        return
    try:
        scope = tracer.start_as_current_span(
            name,
            attributes=_clean_attributes(attributes),
            record_exception=True,
            set_status_on_exception=True,
        )
    except Exception:
        yield None
        return
    with scope as span:
        yield span


def current_trace_context() -> dict[str, str]:
    """返回当前 trace/span id（十六进制），用于日志关联。"""
    tracer = _ensure_tracer()
    if tracer is None:
        return {}
    try:
        from opentelemetry import trace

        context = trace.get_current_span().get_span_context()
        if not context.is_valid:
            return {}
        return {
            "trace_id": format(context.trace_id, "032x"),
            "span_id": format(context.span_id, "016x"),
        }
    except Exception:
        return {}

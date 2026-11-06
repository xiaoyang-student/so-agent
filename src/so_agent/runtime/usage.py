"""模型用量归因（全链路 token 记账）。

主管 / 评审调用由控制层直接记录；``code_agent`` 与动态子 Agent 的
模型调用发生在 MCP 工具内部，控制层看不到其 ``RunResult``。本模块
提供统一提取与记录入口，使嵌套调用的 token 也进入同一事件流。
"""

from __future__ import annotations

import contextlib
import contextvars
from typing import Any, Callable, Iterator

from so_agent.runtime.budget import TokenBudgetTracker
from so_agent.runtime.events import (
    EVENT_TOKEN_USAGE,
    EventLogger,
    shared_event_logger,
)

# 当前执行作用域（task_id, subtask_id, attempt）：由执行协调器在单次
# 子任务尝试外层设置，主管与嵌套 code_agent 的模型调用共享同一异步
# 上下文，因此用量事件与预算都能精确归因到子任务（并发子任务互不串账）。
_usage_scope: contextvars.ContextVar[tuple[str | None, str | None, int | None]] = (
    contextvars.ContextVar("so_agent_usage_scope", default=(None, None, None))
)


@contextlib.contextmanager
def usage_scope(
    task_id: str | None,
    subtask_id: str | None,
    attempt: int | None = None,
) -> Iterator[None]:
    """在子任务尝试期间设置用量归因作用域。"""
    token = _usage_scope.set((task_id, subtask_id, attempt))
    try:
        yield
    finally:
        _usage_scope.reset(token)


def current_usage_scope() -> tuple[str | None, str | None, int | None]:
    """返回当前用量归因作用域（未设置时为 (None, None, None)）。"""
    return _usage_scope.get()


def _responses(result: Any) -> list[Any]:
    """取出运行结果中的模型响应列表（兼容流式与非流式结果）。"""
    responses = list(getattr(result, "raw_responses", None) or [])
    if responses:
        return responses
    # RunResultStreaming 不向 raw_responses 回填（SDK 只在 run_state 的
    # _model_responses 中累计），需回退到内部状态，否则流式嵌套 Agent
    # 的用量恒为 0，台账出现无解释的缺口。
    state = getattr(result, "_state", None)
    if state is None:
        to_state = getattr(result, "to_state", None)
        if callable(to_state):
            try:
                state = to_state()
            except Exception:
                state = None
    return list(getattr(state, "_model_responses", None) or [])


def _int_value(value: Any) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def _field(container: Any, name: str) -> Any:
    """读取字段，兼容字典、Pydantic 模型与供应商扩展字段。"""
    if container is None:
        return None
    if isinstance(container, dict):
        return container.get(name)
    value = getattr(container, name, None)
    if value is not None:
        return value
    extra = getattr(container, "model_extra", None)
    if isinstance(extra, dict):
        return extra.get(name)
    return None


def _cache_tokens(usage: Any) -> tuple[int, int]:
    """读取 prompt cache 命中与写入 token；各供应商报文位置不同。

    Agents SDK 统一放在 ``input_tokens_details``，Chat Completions 原生
    报文放在 ``prompt_tokens_details``，DeepSeek 另在顶层返回
    ``prompt_cache_hit_tokens``；Anthropic 风格用 ``cache_read_input_tokens``。
    """
    cached = written = 0
    for details_name in ("input_tokens_details", "prompt_tokens_details"):
        details = _field(usage, details_name)
        if details is None:
            continue
        cached = cached or _int_value(_field(details, "cached_tokens"))
        written = written or _int_value(_field(details, "cache_write_tokens"))
        cached = cached or _int_value(
            _field(details, "cache_read_input_tokens")
        )
        written = written or _int_value(
            _field(details, "cache_creation_input_tokens")
        )
    cached = cached or _int_value(_field(usage, "prompt_cache_hit_tokens"))
    cached = cached or _int_value(_field(usage, "cache_read_input_tokens"))
    written = written or _int_value(_field(usage, "cache_creation_input_tokens"))
    return cached, written


def _usage_record(
    *,
    requests: int,
    input_tokens: int,
    output_tokens: int,
    total_tokens: int,
    cached_tokens: int,
    cache_write_tokens: int,
) -> dict[str, int]:
    # 缓存命中是 input_tokens 的子集，钳制上限避免供应商报文异常时命中率 > 100%
    cached_tokens = min(cached_tokens, input_tokens)
    return {
        "requests": requests,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "cached_tokens": cached_tokens,
        "cache_write_tokens": cache_write_tokens,
    }


def extract_usage(result: Any) -> dict[str, int]:
    """从 SDK ``RunResult`` / ``RunResultStreaming`` 汇总 token 用量。"""
    input_tokens = output_tokens = total_tokens = 0
    cached_tokens = cache_write_tokens = 0
    requests = 0
    for response in _responses(result):
        usage = getattr(response, "usage", None)
        if usage is None:
            continue
        requests += 1
        input_tokens += getattr(usage, "input_tokens", 0) or 0
        output_tokens += getattr(usage, "output_tokens", 0) or 0
        total_tokens += getattr(usage, "total_tokens", 0) or 0
        cached, written = _cache_tokens(usage)
        cached_tokens += cached
        cache_write_tokens += written
    return _usage_record(
        requests=requests,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        cached_tokens=cached_tokens,
        cache_write_tokens=cache_write_tokens,
    )


def extract_completion_usage(response: Any) -> dict[str, int]:
    """从 Chat Completions 响应汇总 token 用量。"""
    usage = getattr(response, "usage", None)
    input_tokens = _int_value(getattr(usage, "prompt_tokens", 0))
    output_tokens = _int_value(getattr(usage, "completion_tokens", 0))
    total_tokens = _int_value(getattr(usage, "total_tokens", 0))
    if total_tokens == 0:
        total_tokens = input_tokens + output_tokens
    cached_tokens, cache_write_tokens = _cache_tokens(usage)
    return _usage_record(
        requests=1 if usage is not None else 0,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        cached_tokens=cached_tokens,
        cache_write_tokens=cache_write_tokens,
    )


def _log_usage(
    events: EventLogger | None,
    usage: dict[str, int],
    agent_label: str,
    budget: TokenBudgetTracker | None = None,
    **extra: Any,
) -> None:
    scope_task, scope_subtask, scope_attempt = _usage_scope.get()
    attribution: dict[str, Any] = {}
    if scope_task is not None:
        attribution.setdefault("task_id", scope_task)
    if scope_subtask is not None:
        attribution.setdefault("subtask_id", scope_subtask)
    if scope_attempt is not None:
        attribution.setdefault("attempt", scope_attempt)
    if events is not None:
        try:
            events.log_event(
                EVENT_TOKEN_USAGE,
                {"agent": agent_label, **usage, **attribution, **extra},
            )
        except Exception:
            pass
    if budget is not None:
        try:
            budget.observe(agent_label, usage)
            if scope_subtask is not None:
                budget.observe_subtask(scope_subtask, usage)
        except Exception:
            pass


def log_model_usage(
    events: EventLogger | None,
    result: Any,
    agent_label: str,
    *,
    budget: TokenBudgetTracker | None = None,
    **extra: Any,
) -> None:
    """记录一次模型调用的 token 用量；任何异常都不得影响主流程。

    ``budget`` 提供时同步累加到预算 tracker（默认只观察不拦截）。
    """
    try:
        usage = extract_usage(result)
    except Exception:
        return
    _log_usage(events, usage, agent_label, budget, **extra)


def record_completion_usage(
    context: Any,
    response: Any,
    agent_label: str,
    **extra: Any,
) -> None:
    """记录一次 Chat Completions 调用的用量（非 Agents SDK 路径）。

    直答 / 深度推理 / 分类 / 调度解析直接使用 OpenAI 客户端，没有
    ``RunResult``，需从响应的 ``usage`` 字段单独记账，否则这些调用
    在运维指标中恒为 0。
    """
    try:
        usage = extract_completion_usage(response)
        budget = getattr(context, "token_budget", None)
        _log_usage(
            shared_event_logger(context),
            usage,
            agent_label,
            budget if isinstance(budget, TokenBudgetTracker) else None,
            **extra,
        )
    except Exception:
        return


def make_usage_recorder(
    context: Any,
    agent_label: str,
    **extra: Any,
) -> Callable[[Any], None]:
    """构造绑定上下文的用量记录回调（供 MCP Server 在调用后上报）。"""
    events = shared_event_logger(context)

    def _record(result: Any) -> None:
        budget = getattr(context, "token_budget", None)
        log_model_usage(
            events,
            result,
            agent_label,
            budget=budget if isinstance(budget, TokenBudgetTracker) else None,
            **extra,
        )

    return _record

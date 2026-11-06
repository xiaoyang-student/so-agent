"""事件载荷脱敏：密钥值不得进入事件、日志文件与审计链。

按“键名 + 值形态”双重判断：键名命中敏感后缀时整体替换；字符串值中的
Bearer / sk- / key=value 形态做局部替换。计数类字段（如 tokens、
max_tokens）不在敏感键集合内，避免破坏用量统计。
"""

from __future__ import annotations

import re
from typing import Any

REDACTED = "***"

_SENSITIVE_KEYS = frozenset(
    {
        "authorization",
        "proxyauthorization",
        "cookie",
        "setcookie",
        "apikeys",
        "secrets",
    }
)

_SENSITIVE_KEY_SUFFIXES = (
    "apikey",
    "token",
    "password",
    "passwd",
    "secret",
    "cookie",
    "credential",
    "privatekey",
)

_MAX_DEPTH = 8

_VALUE_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"(?i)\b(bearer\s+)[A-Za-z0-9._~+/=-]{8,}"), r"\1" + REDACTED),
    (re.compile(r"\bsk-[A-Za-z0-9_-]{12,}"), "sk-" + REDACTED),
    (
        re.compile(
            r"(?i)\b((?:api[_-]?key|access[_-]?token|refresh[_-]?token|"
            r"password|secret)\s*[:=]\s*)['\"]?([^\s'\"&,;]{4,})"
        ),
        r"\1" + REDACTED,
    ),
    (
        re.compile(r"(?i)([?&](?:token|api_key|apikey|access_token)=)([^&\s#]+)"),
        r"\1" + REDACTED,
    ),
)

_QUICK_MARKERS = (
    "bearer",
    "sk-",
    "api_key",
    "apikey",
    "token",
    "password",
    "secret",
)


def _normalize_key(key: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(key).lower())


def is_sensitive_key(key: Any) -> bool:
    """键名是否表示凭据（``tokens`` / ``max_tokens`` 等计数不算）。"""
    normalized = _normalize_key(key)
    if not normalized:
        return False
    if normalized in _SENSITIVE_KEYS:
        return True
    return any(normalized.endswith(suffix) for suffix in _SENSITIVE_KEY_SUFFIXES)


def redact_text(text: str) -> str:
    """替换字符串中的常见凭据形态。"""
    if not text:
        return text
    lowered = text.lower()
    if not any(marker in lowered for marker in _QUICK_MARKERS):
        return text
    for pattern, replacement in _VALUE_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


def redact_payload(value: Any, *, depth: int = 0) -> Any:
    """递归脱敏事件载荷；超出深度上限时原样返回该子树。"""
    if depth > _MAX_DEPTH:
        return value
    if isinstance(value, dict):
        return {
            key: (
                REDACTED
                if is_sensitive_key(key)
                else redact_payload(item, depth=depth + 1)
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact_payload(item, depth=depth + 1) for item in value]
    if isinstance(value, tuple):
        return [redact_payload(item, depth=depth + 1) for item in value]
    if isinstance(value, str):
        return redact_text(value)
    return value


__all__ = ["REDACTED", "is_sensitive_key", "redact_payload", "redact_text"]

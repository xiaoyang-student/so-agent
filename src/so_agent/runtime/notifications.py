"""人工升级外部通知（安全 Webhook，默认关闭）。

安全边界与外部 MCP 一致：

- 默认关闭：未配置 URL 时不发起任何网络请求；
- 仅 HTTPS：不接受 http 或其它协议；
- 默认拒绝私有/回环/链路本地/保留地址，需显式放行内网部署；
- 不跟随重定向；
- 令牌与可选签名密钥只从环境变量读取，缺失时失败关闭；
- 超时与响应体上限固定，通知失败只记录，不阻断任务进入人工兜底；
- 有限次指数退避重试（仅网络错误、429 与 5xx），每次投递携带稳定的
  ``Idempotency-Key``，接收方据此去重；
- 可选持久化 outbox：通知先写 SQLite 再投递，进程崩溃或端点短暂不可用
  时由后台任务继续重试，同一 ``event_id`` 只入队一次。
- 可选 HMAC-SHA256 签名：``X-So-Agent-Signature: sha256=<hex>``，
  签名内容为 ``<timestamp>.<原始请求体>``。
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import ipaddress
import json
import os
import socket
import time
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable
from urllib.parse import urlparse

from so_agent.config import Settings, get_settings
from so_agent.models import HumanEscalationRequest
from so_agent.runtime.logging_setup import get_logger
from so_agent.runtime.notification_outbox import (
    NotificationOutbox,
    resolve_outbox_path,
)

logger = get_logger("runtime.notifications")

_MAX_RESPONSE_BYTES = 4096
_MAX_OBJECTIVE_CHARS = 500
_MAX_ERROR_CHARS = 300


class NotificationPolicyError(RuntimeError):
    """通知目标或凭据不符合安全策略。"""


@dataclass(frozen=True)
class NotificationResult:
    """一次通知投递的结果（失败不抛异常）。"""

    sent: bool
    skipped: bool = False
    status_code: int | None = None
    error: str = ""
    attempts: int = 0
    event_id: str = ""
    queued: bool = False
    duplicate: bool = False


NotificationSender = Callable[
    [str, dict[str, Any], dict[str, str], float], int
]


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: D102
        return None


def _payload_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(
        payload, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")


def _retryable_status(status: int) -> bool:
    return status == 429 or 500 <= status < 600


def _post_json(
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str],
    timeout: float,
) -> int:
    data = _payload_bytes(payload)
    request = urllib.request.Request(
        url, data=data, headers=headers, method="POST"
    )
    opener = urllib.request.build_opener(_NoRedirectHandler())
    with opener.open(request, timeout=timeout) as response:
        response.read(_MAX_RESPONSE_BYTES)
        return int(response.status)


class EscalationNotifier:
    """把人工升级材料包投递到配置的 Webhook。"""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        sender: NotificationSender | None = None,
        resolver: Callable[..., Any] | None = None,
        outbox: NotificationOutbox | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._sender = sender or _post_json
        self._resolver = resolver or socket.getaddrinfo
        self._outbox = outbox

    @property
    def configured(self) -> bool:
        return bool((self._settings.escalation_webhook_url or "").strip())

    async def notify(self, request: HumanEscalationRequest) -> NotificationResult:
        """投递通知；任何策略/网络失败都返回失败结果而不抛出。

        未启用 outbox 时仅对网络异常、HTTP 429 与 5xx 在进程内重试；
        启用后先持久化再尝试一次，失败条目留给后台任务退避重试。
        重试复用同一 ``event_id``，接收方可按 ``Idempotency-Key`` 去重。
        """
        payload = self._build_payload(request)
        return await self.notify_payload(
            payload, event_id=str(payload["event_id"])
        )

    async def notify_payload(
        self, payload: dict[str, Any], *, event_id: str | None = None
    ) -> NotificationResult:
        """投递自定义事件载荷（复用 outbox 与幂等键）。

        ``payload`` 必须包含 ``task_id``；``event_id`` 缺省时取载荷中的
        ``event_id``。未配置 Webhook 时返回 skipped，不发起网络请求。
        """
        if not self.configured:
            return NotificationResult(sent=False, skipped=True)
        event_id = str(event_id or payload.get("event_id") or "").strip()
        if not event_id:
            raise NotificationPolicyError("通知载荷缺少 event_id")
        if self._outbox is not None:
            return await self._notify_with_outbox(payload, event_id)

        attempts = max(
            1, min(10, int(self._settings.escalation_webhook_max_attempts))
        )
        try:
            url, headers, signing_secret = await self._prepare_target()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            message = str(exc)[:_MAX_ERROR_CHARS]
            logger.warning("人工升级通知失败：%s", message)
            return NotificationResult(
                sent=False, error=message, event_id=event_id
            )
        return await self._run_attempts(
            url,
            payload,
            event_id,
            headers,
            signing_secret,
            max_attempts=attempts,
        )

    async def deliver_once(
        self, payload: dict[str, Any], event_id: str
    ) -> NotificationResult:
        """单次投递（outbox 后台排空使用，不在内部重试）。"""
        try:
            url, headers, signing_secret = await self._prepare_target()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            message = str(exc)[:_MAX_ERROR_CHARS]
            logger.warning("人工升级通知失败：%s", message)
            return NotificationResult(
                sent=False, error=message, event_id=event_id
            )
        return await self._run_attempts(
            url,
            payload,
            event_id,
            headers,
            signing_secret,
            max_attempts=1,
        )

    async def _notify_with_outbox(
        self, payload: dict[str, Any], event_id: str
    ) -> NotificationResult:
        try:
            url, headers, signing_secret = await self._prepare_target()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            message = str(exc)[:_MAX_ERROR_CHARS]
            logger.warning("人工升级通知失败：%s", message)
            return NotificationResult(
                sent=False, error=message, event_id=event_id
            )
        inserted = self._outbox.enqueue(
            str(payload["task_id"]), event_id, payload
        )
        if not inserted:
            existing = self._outbox.get(event_id)
            if existing is not None and existing.status == "delivered":
                return NotificationResult(
                    sent=True, attempts=0, event_id=event_id, duplicate=True
                )
            return NotificationResult(
                sent=False,
                queued=True,
                attempts=existing.attempts if existing else 0,
                error="通知已在 outbox 等待重试",
                event_id=event_id,
            )
        recorded = self._outbox.record_attempt(event_id)
        if recorded is None:
            return NotificationResult(
                sent=False,
                queued=True,
                error="通知已在 outbox 等待重试",
                event_id=event_id,
            )
        result = await self._run_attempts(
            url,
            payload,
            event_id,
            headers,
            signing_secret,
            max_attempts=1,
        )
        if result.sent:
            self._outbox.mark_delivered(event_id)
            return result
        permanent = result.attempts == 0 or (
            result.status_code is not None
            and not _retryable_status(result.status_code)
        )
        if permanent:
            self._outbox.mark_failed(event_id, result.error)
            return result
        self._outbox.mark_retry(
            event_id,
            result.error,
            max_attempts=self._settings.notification_outbox_max_attempts,
            backoff_base_seconds=(
                self._settings.notification_outbox_retry_backoff_seconds
            ),
        )
        return NotificationResult(
            sent=False,
            queued=True,
            status_code=result.status_code,
            error=result.error,
            attempts=result.attempts,
            event_id=event_id,
        )

    async def _prepare_target(
        self,
    ) -> tuple[str, dict[str, str], str]:
        url = (self._settings.escalation_webhook_url or "").strip()
        if not url:
            raise NotificationPolicyError("未配置通知 Webhook")
        await self._validate_target(url)
        return url, self._auth_headers(), self._signing_secret()

    async def _run_attempts(
        self,
        url: str,
        payload: dict[str, Any],
        event_id: str,
        base_headers: dict[str, str],
        signing_secret: str,
        *,
        max_attempts: int,
    ) -> NotificationResult:
        attempts = max(1, min(10, int(max_attempts)))
        timeout = float(self._settings.escalation_webhook_timeout_seconds)
        last_error = ""
        for attempt in range(1, attempts + 1):
            headers = dict(base_headers)
            headers["Idempotency-Key"] = event_id
            headers["X-So-Agent-Delivery-Attempt"] = str(attempt)
            if signing_secret:
                timestamp = str(int(time.time()))
                body = _payload_bytes(payload)
                signature = hmac.new(
                    signing_secret.encode("utf-8"),
                    f"{timestamp}.".encode("utf-8") + body,
                    hashlib.sha256,
                ).hexdigest()
                headers["X-So-Agent-Timestamp"] = timestamp
                headers["X-So-Agent-Signature"] = f"sha256={signature}"
            try:
                status = int(
                    await asyncio.to_thread(
                        self._sender, url, payload, headers, timeout
                    )
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                code = getattr(exc, "code", None)
                last_error = str(exc)[:_MAX_ERROR_CHARS]
                if code is not None and not _retryable_status(int(code)):
                    logger.warning(
                        "人工升级通知失败（HTTP %s，不重试）：%s",
                        code,
                        last_error,
                    )
                    return NotificationResult(
                        sent=False,
                        status_code=int(code),
                        error=last_error,
                        attempts=attempt,
                        event_id=event_id,
                    )
                if attempt < attempts:
                    await self._sleep_before_retry(attempt)
                    continue
                logger.warning(
                    "人工升级通知失败（已尝试 %d 次）：%s", attempt, last_error
                )
                return NotificationResult(
                    sent=False,
                    status_code=int(code) if code is not None else None,
                    error=last_error,
                    attempts=attempt,
                    event_id=event_id,
                )
            if 200 <= status < 300:
                return NotificationResult(
                    sent=True,
                    status_code=status,
                    attempts=attempt,
                    event_id=event_id,
                )
            last_error = f"通知端点返回 HTTP {status}"
            if _retryable_status(status) and attempt < attempts:
                await self._sleep_before_retry(attempt)
                continue
            logger.warning("人工升级通知失败：%s", last_error)
            return NotificationResult(
                sent=False,
                status_code=status,
                error=last_error,
                attempts=attempt,
                event_id=event_id,
            )
        return NotificationResult(
            sent=False,
            error=last_error or "通知投递失败",
            attempts=attempts,
            event_id=event_id,
        )

    def _auth_headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json; charset=utf-8"}
        token_env = (self._settings.escalation_webhook_token_env or "").strip()
        if token_env:
            token = os.environ.get(token_env, "").strip()
            if not token:
                raise NotificationPolicyError(
                    f"通知令牌环境变量未设置：{token_env}"
                )
            headers["Authorization"] = f"Bearer {token}"
        return headers

    def _signing_secret(self) -> str:
        secret_env = (
            self._settings.escalation_webhook_signing_secret_env or ""
        ).strip()
        if not secret_env:
            return ""
        secret = os.environ.get(secret_env, "").strip()
        if not secret:
            raise NotificationPolicyError(
                f"通知签名密钥环境变量未设置：{secret_env}"
            )
        return secret

    async def _sleep_before_retry(self, attempt: int) -> None:
        base = max(
            0.0,
            float(self._settings.escalation_webhook_retry_backoff_seconds),
        )
        if base <= 0:
            return
        delay = min(5.0, base * (2 ** (attempt - 1)))
        await asyncio.sleep(delay)

    async def _validate_target(self, url: str) -> None:
        parts = urlparse(url)
        if parts.scheme != "https":
            raise NotificationPolicyError("通知 Webhook 必须使用 HTTPS")
        if not parts.hostname:
            raise NotificationPolicyError("通知 Webhook 缺少主机名")
        if parts.username or parts.password:
            raise NotificationPolicyError("通知 Webhook 不允许携带用户信息")
        if self._settings.escalation_webhook_allow_private_network:
            return
        host = parts.hostname
        try:
            direct = ipaddress.ip_address(host)
            addresses = [direct]
        except ValueError:
            infos = await asyncio.to_thread(
                self._resolver, host, parts.port or 443, 0, socket.SOCK_STREAM
            )
            addresses = []
            for info in infos:
                try:
                    addresses.append(ipaddress.ip_address(info[4][0]))
                except ValueError:
                    continue
        for address in addresses:
            if (
                address.is_private
                or address.is_loopback
                or address.is_link_local
                or address.is_reserved
                or address.is_multicast
                or address.is_unspecified
            ):
                raise NotificationPolicyError(
                    "通知 Webhook 解析到私有/保留地址；"
                    "内网部署需显式 allow_private_network"
                )

    @staticmethod
    def _build_payload(request: HumanEscalationRequest) -> dict[str, Any]:
        objective = (request.original_request.objective or "").strip()
        if len(objective) > _MAX_OBJECTIVE_CHARS:
            objective = objective[:_MAX_OBJECTIVE_CHARS] + "…"
        basis = "|".join(
            [
                request.task_id,
                request.reason,
                request.created_at.isoformat(),
                objective,
            ]
        )
        event_id = "so-agent-" + hashlib.sha256(
            basis.encode("utf-8")
        ).hexdigest()[:32]
        return {
            "event": "human_escalation",
            "event_id": event_id,
            "task_id": request.task_id,
            "reason": request.reason,
            "objective": objective,
            "status": "human_required",
            "rollback_targets": list(request.rollback_targets)[:50],
            "progress": [
                {
                    "subtask_id": item.subtask_id,
                    "title": item.title,
                    "status": item.status,
                    "error": (item.error or "")[:_MAX_ERROR_CHARS],
                }
                for item in request.progress
            ],
            "created_at": request.created_at.isoformat(),
        }


def build_escalation_notifier(settings: Settings) -> EscalationNotifier:
    """构造通知器；启用 outbox 时挂载持久化队列（写失败不阻断任务）。"""
    outbox = None
    if bool(settings.notification_outbox_enabled):
        path = resolve_outbox_path(settings)
        if path is not None:
            try:
                outbox = NotificationOutbox(path)
            except Exception:
                logger.warning(
                    "通知 outbox 初始化失败，回退为进程内通知", exc_info=True
                )
    return EscalationNotifier(settings, outbox=outbox)


__all__ = [
    "EscalationNotifier",
    "NotificationPolicyError",
    "NotificationResult",
    "build_escalation_notifier",
]

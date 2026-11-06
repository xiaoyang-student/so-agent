"""人工升级外部通知：安全策略、凭据与失败降级。"""

from __future__ import annotations

from typing import Any

from so_agent.config import Settings
from so_agent.models import (
    HumanEscalationRequest,
    SubtaskProgress,
    TaskRequest,
)
from so_agent.runtime.notifications import EscalationNotifier
from so_agent.workflow_support import HumanEscalationHandler


def _settings(**overrides: Any) -> Settings:
    values: dict[str, Any] = {
        "_env_file": None,
        "escalation_webhook_timeout_seconds": 3.0,
        "escalation_webhook_retry_backoff_seconds": 0.0,
    }
    values.update(overrides)
    return Settings(**values)


def _request(objective: str = "分析仓库并生成报告") -> HumanEscalationRequest:
    task = TaskRequest(task_id="task-1", objective=objective)
    return HumanEscalationRequest(
        task_id="task-1",
        original_request=task,
        reason="连续失败达到上限",
        progress=[
            SubtaskProgress(
                subtask_id="s1",
                title="分析",
                status="failed",
                error="boom" * 200,
            )
        ],
        rollback_targets=["s1", "s2"],
    )


async def test_notifier_skips_without_url():
    result = await EscalationNotifier(_settings()).notify(_request())
    assert result.skipped is True
    assert result.sent is False


async def test_notifier_posts_with_env_token():
    captured: dict[str, Any] = {}

    def sender(url, payload, headers, timeout):
        captured.update(
            url=url, payload=payload, headers=headers, timeout=timeout
        )
        return 204

    settings = _settings(
        escalation_webhook_url="https://93.184.216.34/hook",
        escalation_webhook_token_env="SO_AGENT_TEST_HOOK_TOKEN",
    )
    notifier = EscalationNotifier(settings, sender=sender)
    import os

    os.environ["SO_AGENT_TEST_HOOK_TOKEN"] = "hook-secret"
    try:
        result = await notifier.notify(_request("目标" * 400))
    finally:
        os.environ.pop("SO_AGENT_TEST_HOOK_TOKEN", None)

    assert result.sent is True
    assert result.status_code == 204
    assert captured["url"] == "https://93.184.216.34/hook"
    assert captured["headers"]["Authorization"] == "Bearer hook-secret"
    assert captured["timeout"] == 3.0
    payload = captured["payload"]
    assert payload["event"] == "human_escalation"
    assert payload["event_id"].startswith("so-agent-")
    assert payload["status"] == "human_required"
    assert captured["headers"]["Idempotency-Key"] == payload["event_id"]
    assert result.attempts == 1
    assert len(payload["objective"]) == 501
    assert "output" not in payload
    assert len(payload["progress"][0]["error"]) == 300


async def test_notifier_requires_token_env():
    called = False

    def sender(url, payload, headers, timeout):
        nonlocal called
        called = True
        return 200

    notifier = EscalationNotifier(
        _settings(
            escalation_webhook_url="https://93.184.216.34/hook",
            escalation_webhook_token_env="SO_AGENT_TEST_MISSING_TOKEN",
        ),
        sender=sender,
    )
    result = await notifier.notify(_request())
    assert result.sent is False
    assert "环境变量未设置" in result.error
    assert called is False


async def test_notifier_rejects_insecure_and_private_targets():
    insecure = await EscalationNotifier(
        _settings(escalation_webhook_url="http://93.184.216.34/hook")
    ).notify(_request())
    assert insecure.sent is False
    assert "HTTPS" in insecure.error

    private = await EscalationNotifier(
        _settings(escalation_webhook_url="https://127.0.0.1/hook")
    ).notify(_request())
    assert private.sent is False
    assert "私有" in private.error

    userinfo = await EscalationNotifier(
        _settings(escalation_webhook_url="https://user:pass@93.184.216.34/hook")
    ).notify(_request())
    assert userinfo.sent is False
    assert "用户信息" in userinfo.error


async def test_notifier_resolves_hostname_before_sending():
    called = False

    def sender(url, payload, headers, timeout):
        nonlocal called
        called = True
        return 200

    def resolver(host, port, family, socktype):
        return [(2, 1, 6, "", ("10.0.0.8", 443))]

    notifier = EscalationNotifier(
        _settings(escalation_webhook_url="https://internal.example/hook"),
        sender=sender,
        resolver=resolver,
    )
    result = await notifier.notify(_request())
    assert result.sent is False
    assert "私有" in result.error
    assert called is False


async def test_notifier_allows_private_network_when_explicit():
    def sender(url, payload, headers, timeout):
        return 202

    notifier = EscalationNotifier(
        _settings(
            escalation_webhook_url="https://127.0.0.1/hook",
            escalation_webhook_allow_private_network=True,
        ),
        sender=sender,
    )
    result = await notifier.notify(_request())
    assert result.sent is True
    assert result.status_code == 202


async def test_notifier_never_raises_on_sender_failure():
    calls = 0

    def sender(url, payload, headers, timeout):
        nonlocal calls
        calls += 1
        raise RuntimeError("connection reset")

    notifier = EscalationNotifier(
        _settings(escalation_webhook_url="https://93.184.216.34/hook"),
        sender=sender,
    )
    result = await notifier.notify(_request())
    assert result.sent is False
    assert "connection reset" in result.error
    assert result.attempts == 3
    assert calls == 3


async def test_notifier_retries_5xx_with_stable_idempotency_key():
    seen: list[dict[str, Any]] = []
    statuses = [503, 200]

    def sender(url, payload, headers, timeout):
        seen.append(dict(headers))
        return statuses.pop(0)

    notifier = EscalationNotifier(
        _settings(escalation_webhook_url="https://93.184.216.34/hook"),
        sender=sender,
    )
    result = await notifier.notify(_request())
    assert result.sent is True
    assert result.attempts == 2
    keys = {item["Idempotency-Key"] for item in seen}
    assert len(keys) == 1
    assert seen[1]["X-So-Agent-Delivery-Attempt"] == "2"


async def test_notifier_does_not_retry_4xx():
    calls = 0

    def sender(url, payload, headers, timeout):
        nonlocal calls
        calls += 1
        return 400

    notifier = EscalationNotifier(
        _settings(escalation_webhook_url="https://93.184.216.34/hook"),
        sender=sender,
    )
    result = await notifier.notify(_request())
    assert result.sent is False
    assert result.attempts == 1
    assert calls == 1


async def test_notifier_signs_payload_when_secret_configured():
    import hashlib
    import hmac
    import os

    from so_agent.runtime.notifications import _payload_bytes

    captured: dict[str, Any] = {}

    def sender(url, payload, headers, timeout):
        captured.update(payload=payload, headers=headers)
        return 200

    notifier = EscalationNotifier(
        _settings(
            escalation_webhook_url="https://93.184.216.34/hook",
            escalation_webhook_signing_secret_env="SO_AGENT_TEST_HOOK_SECRET",
        ),
        sender=sender,
    )
    os.environ["SO_AGENT_TEST_HOOK_SECRET"] = "sign-secret"
    try:
        result = await notifier.notify(_request())
    finally:
        os.environ.pop("SO_AGENT_TEST_HOOK_SECRET", None)

    assert result.sent is True
    timestamp = captured["headers"]["X-So-Agent-Timestamp"]
    expected = hmac.new(
        b"sign-secret",
        f"{timestamp}.".encode("utf-8") + _payload_bytes(captured["payload"]),
        hashlib.sha256,
    ).hexdigest()
    assert captured["headers"]["X-So-Agent-Signature"] == f"sha256={expected}"


async def test_notifier_requires_signing_secret_env():
    called = False

    def sender(url, payload, headers, timeout):
        nonlocal called
        called = True
        return 200

    notifier = EscalationNotifier(
        _settings(
            escalation_webhook_url="https://93.184.216.34/hook",
            escalation_webhook_signing_secret_env="SO_AGENT_TEST_MISSING_SECRET",
        ),
        sender=sender,
    )
    result = await notifier.notify(_request())
    assert result.sent is False
    assert "签名密钥环境变量未设置" in result.error
    assert called is False


async def test_escalation_handler_records_last_notification():
    def sender(url, payload, headers, timeout):
        return 200

    handler = HumanEscalationHandler(
        notifier=EscalationNotifier(
            _settings(escalation_webhook_url="https://93.184.216.34/hook"),
            sender=sender,
        )
    )
    await handler.escalate(_request())
    assert len(handler.pending_requests) == 1
    assert handler.last_notification is not None
    assert handler.last_notification.sent is True

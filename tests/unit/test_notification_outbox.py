"""单元测试：人工升级通知持久化 outbox。"""

from __future__ import annotations

import time
from typing import Any

from so_agent.config import Settings
from so_agent.models import HumanEscalationRequest, TaskRequest
from so_agent.runtime.notification_outbox import (
    NotificationOutbox,
    NotificationOutboxWorker,
)
from so_agent.runtime.notifications import (
    EscalationNotifier,
    build_escalation_notifier,
)


def _settings(tmp_path, **overrides: Any) -> Settings:
    values: dict[str, Any] = {
        "_env_file": None,
        "escalation_webhook_url": "https://93.184.216.34/hook",
        "escalation_webhook_timeout_seconds": 3.0,
        "escalation_webhook_retry_backoff_seconds": 0.0,
        "notification_outbox_enabled": True,
        "notification_outbox_path": str(tmp_path / "outbox.db"),
        "notification_outbox_retry_backoff_seconds": 0.0,
        "notification_outbox_drain_interval_seconds": 60.0,
    }
    values.update(overrides)
    return Settings(**values)


def _request(objective: str = "分析仓库并生成报告") -> HumanEscalationRequest:
    return HumanEscalationRequest(
        task_id="task-1",
        original_request=TaskRequest(task_id="task-1", objective=objective),
        reason="连续失败达到上限",
    )


class _HttpError(Exception):
    def __init__(self, code: int) -> None:
        super().__init__(f"HTTP {code}")
        self.code = code


class TestOutboxStore:
    def test_enqueue_is_idempotent(self, tmp_path):
        outbox = NotificationOutbox(tmp_path / "outbox.db")
        payload = {"event": "human_escalation", "event_id": "e1", "task_id": "t1"}
        assert outbox.enqueue("t1", "e1", payload) is True
        assert outbox.enqueue("t1", "e1", payload) is False
        entry = outbox.get("e1")
        assert entry is not None
        assert entry.status == "pending"
        assert entry.attempts == 0
        assert entry.payload == payload

    def test_claim_due_respects_lease_and_attempts(self, tmp_path):
        outbox = NotificationOutbox(tmp_path / "outbox.db")
        outbox.enqueue("t1", "e1", {"event_id": "e1"}, now=100.0)
        outbox.enqueue("t1", "e2", {"event_id": "e2"}, now=100.0)

        first = outbox.claim_due(limit=1, now=100.0, lease_seconds=60.0)
        assert [entry.event_id for entry in first] == ["e1"]
        assert first[0].attempts == 1

        again = outbox.claim_due(limit=10, now=101.0, lease_seconds=60.0)
        assert [entry.event_id for entry in again] == ["e2"]
        after_lease = outbox.claim_due(limit=10, now=200.0, lease_seconds=60.0)
        assert [entry.event_id for entry in after_lease] == ["e1", "e2"]
        assert [entry.attempts for entry in after_lease] == [2, 2]

    def test_retry_backoff_caps_at_max_attempts(self, tmp_path):
        outbox = NotificationOutbox(tmp_path / "outbox.db")
        outbox.enqueue("t1", "e1", {"event_id": "e1"}, now=0.0)
        outbox.claim_due(limit=1, now=0.0)

        queued = outbox.mark_retry(
            "e1", "boom", max_attempts=2, backoff_base_seconds=10.0, now=1.0
        )
        assert queued is True
        entry = outbox.get("e1")
        assert entry is not None and entry.status == "pending"
        assert entry.attempts == 1
        assert entry.next_attempt_at == 11.0

        outbox.claim_due(limit=1, now=100.0)
        queued = outbox.mark_retry(
            "e1", "still boom", max_attempts=2, backoff_base_seconds=10.0, now=101.0
        )
        assert queued is False
        failed = outbox.get("e1")
        assert failed is not None
        assert failed.status == "failed"
        assert failed.last_error == "still boom"

    def test_stats_reports_counts_and_due(self, tmp_path):
        outbox = NotificationOutbox(tmp_path / "outbox.db")
        outbox.enqueue("t1", "e1", {"event_id": "e1"}, now=0.0)
        outbox.enqueue("t1", "e2", {"event_id": "e2"}, now=0.0)
        outbox.mark_delivered("e2")
        stats = outbox.stats(now=10.0)
        assert stats["pending"] == 1
        assert stats["delivered"] == 1
        assert stats["failed"] == 0
        assert stats["due"] == 1
        assert stats["oldest_pending_age_seconds"] == 10


class TestNotifierOutbox:
    async def test_success_is_persisted_and_not_resent(self, tmp_path):
        calls: list[dict[str, Any]] = []

        def sender(url, payload, headers, timeout):
            calls.append({"url": url, "event_id": headers["Idempotency-Key"]})
            return 204

        outbox = NotificationOutbox(tmp_path / "outbox.db")
        notifier = EscalationNotifier(
            _settings(tmp_path), sender=sender, outbox=outbox
        )
        request = _request()
        result = await notifier.notify(request)
        assert result.sent is True
        assert result.event_id == calls[0]["event_id"]
        assert outbox.get(result.event_id).status == "delivered"

        duplicate = await notifier.notify(request)
        assert duplicate.sent is True
        assert duplicate.duplicate is True
        assert len(calls) == 1

    async def test_retryable_failure_stays_pending(self, tmp_path):
        calls: list[int] = []

        def sender(url, payload, headers, timeout):
            calls.append(1)
            raise OSError("network down")

        outbox = NotificationOutbox(tmp_path / "outbox.db")
        notifier = EscalationNotifier(
            _settings(tmp_path), sender=sender, outbox=outbox
        )
        request = _request()
        result = await notifier.notify(request)
        assert result.sent is False
        assert result.queued is True
        entry = outbox.get(result.event_id)
        assert entry is not None
        assert entry.status == "pending"
        assert entry.attempts == 1

        await notifier.notify(request)
        assert len(calls) == 1
        assert outbox.get(result.event_id).attempts == 1

    async def test_permanent_http_error_marks_failed(self, tmp_path):
        def sender(url, payload, headers, timeout):
            raise _HttpError(400)

        outbox = NotificationOutbox(tmp_path / "outbox.db")
        notifier = EscalationNotifier(
            _settings(tmp_path), sender=sender, outbox=outbox
        )
        result = await notifier.notify(_request())
        assert result.sent is False
        assert result.queued is False
        assert outbox.get(result.event_id).status == "failed"

    async def test_worker_drains_due_entry(self, tmp_path):
        state = {"fail": True, "calls": 0}

        def sender(url, payload, headers, timeout):
            state["calls"] += 1
            if state["fail"]:
                raise OSError("network down")
            return 200

        settings = _settings(tmp_path)
        outbox = NotificationOutbox(tmp_path / "outbox.db")
        notifier = EscalationNotifier(settings, sender=sender, outbox=outbox)
        first = await notifier.notify(_request())
        assert first.queued is True

        state["fail"] = False
        worker = NotificationOutboxWorker(
            settings, outbox=outbox, notifier=notifier
        )
        assert await worker.drain_once() == 1
        assert state["calls"] == 2
        assert outbox.get(first.event_id).status == "delivered"
        assert await worker.drain_once() == 0

    async def test_worker_disabled_without_webhook(self, tmp_path):
        settings = _settings(tmp_path, escalation_webhook_url="")
        outbox = NotificationOutbox(tmp_path / "outbox.db")
        worker = NotificationOutboxWorker(
            settings,
            outbox=outbox,
            notifier=EscalationNotifier(settings, outbox=outbox),
        )
        assert worker.enabled is False
        await worker.start()
        assert worker.running is False
        await worker.stop()

    def test_build_notifier_attaches_outbox_only_when_enabled(self, tmp_path):
        enabled = build_escalation_notifier(_settings(tmp_path))
        assert enabled._outbox is not None

        disabled = build_escalation_notifier(
            _settings(tmp_path, notification_outbox_enabled=False)
        )
        assert disabled._outbox is None

    def test_retry_backoff_delay_grows(self, tmp_path):
        outbox = NotificationOutbox(tmp_path / "outbox.db")
        outbox.enqueue("t1", "e1", {"event_id": "e1"}, now=0.0)
        outbox.claim_due(limit=1, now=0.0)
        outbox.mark_retry(
            "e1", "boom", max_attempts=5, backoff_base_seconds=5.0, now=0.0
        )
        first_due = None
        for candidate in range(0, 20):
            if outbox.claim_due(limit=1, now=float(candidate)):
                first_due = candidate
                break
        assert first_due == 5
        outbox.mark_retry(
            "e1", "boom", max_attempts=5, backoff_base_seconds=5.0, now=5.0
        )
        second_due = None
        for candidate in range(6, 40):
            if outbox.claim_due(limit=1, now=float(candidate)):
                second_due = candidate
                break
        assert second_due == 15

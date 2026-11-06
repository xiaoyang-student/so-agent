"""人工升级通知的持久化 outbox。

升级通知先写入独立 SQLite 再投递，进程崩溃或端点暂时不可用不会丢失；
后台任务按指数退避重试，同一 ``event_id`` 只入队一次，接收方可用
``Idempotency-Key`` 去重。outbox 默认关闭，启用后写失败只告警、
不阻断任务进入人工兜底。
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from so_agent.config import Settings, get_settings
from so_agent.runtime.logging_setup import get_logger

logger = get_logger("runtime.notification_outbox")

_LEASE_SECONDS = 300.0
_BACKOFF_CAP_SECONDS = 300.0

_SCHEMA = """
CREATE TABLE IF NOT EXISTS notification_outbox (
    event_id TEXT PRIMARY KEY,
    task_id TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    attempts INTEGER NOT NULL DEFAULT 0,
    next_attempt_at REAL NOT NULL,
    last_error TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    created_ts REAL NOT NULL,
    updated_at TEXT NOT NULL,
    delivered_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_notification_outbox_due
    ON notification_outbox(status, next_attempt_at);
"""


def _utc_iso(ts: float | None = None) -> str:
    moment = datetime.fromtimestamp(ts if ts is not None else time.time(), timezone.utc)
    return moment.isoformat()


def resolve_outbox_path(settings: Settings) -> Path | None:
    raw = (settings.notification_outbox_path or "").strip()
    if not raw:
        return None
    path = Path(raw)
    if not path.is_absolute():
        path = Path.cwd() / path
    return path


def _retryable_status(status: int | None) -> bool:
    return status is not None and (status == 429 or 500 <= status < 600)


@dataclass(frozen=True)
class OutboxEntry:
    event_id: str
    task_id: str
    payload: dict[str, Any]
    attempts: int
    status: str
    last_error: str = ""
    next_attempt_at: float = 0.0
    created_at: str = ""
    updated_at: str = ""
    delivered_at: str | None = None


class NotificationOutbox:
    """SQLite 通知队列（单条事件只入队一次）。"""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._connection() as conn:
            conn.executescript(_SCHEMA)
            conn.commit()

    @property
    def path(self) -> Path:
        return self._path

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self._path, timeout=10)
        try:
            conn.execute("PRAGMA busy_timeout=10000")
            conn.execute("PRAGMA journal_mode=WAL")
            conn.row_factory = sqlite3.Row
            yield conn
        finally:
            conn.close()

    def enqueue(
        self,
        task_id: str,
        event_id: str,
        payload: dict[str, Any],
        *,
        now: float | None = None,
    ) -> bool:
        """入队；返回 True 表示新建，False 表示同 event_id 已存在。"""
        current = time.time() if now is None else now
        timestamp = _utc_iso(current)
        with self._connection() as conn:
            cursor = conn.execute(
                "INSERT OR IGNORE INTO notification_outbox("
                "event_id, task_id, payload_json, status, attempts,"
                " next_attempt_at, last_error, created_at, created_ts, updated_at"
                ") VALUES (?, ?, ?, 'pending', 0, ?, '', ?, ?, ?)",
                (
                    event_id,
                    task_id,
                    json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
                    current,
                    timestamp,
                    current,
                    timestamp,
                ),
            )
            conn.commit()
            return cursor.rowcount > 0

    def get(self, event_id: str) -> OutboxEntry | None:
        with self._connection() as conn:
            row = conn.execute(
                "SELECT * FROM notification_outbox WHERE event_id = ?",
                (event_id,),
            ).fetchone()
        return self._entry(row) if row else None

    def claim_due(
        self,
        *,
        limit: int = 20,
        now: float | None = None,
        lease_seconds: float = _LEASE_SECONDS,
    ) -> list[OutboxEntry]:
        """原子领取到期条目并预扣一次尝试（崩溃后按租约重新到期）。"""
        current = time.time() if now is None else now
        claimed: list[OutboxEntry] = []
        with self._connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            rows = conn.execute(
                "SELECT * FROM notification_outbox"
                " WHERE status = 'pending' AND next_attempt_at <= ?"
                " ORDER BY next_attempt_at, created_ts LIMIT ?",
                (current, max(1, int(limit))),
            ).fetchall()
            for row in rows:
                attempts = int(row["attempts"]) + 1
                conn.execute(
                    "UPDATE notification_outbox"
                    " SET attempts = ?, next_attempt_at = ?, updated_at = ?"
                    " WHERE event_id = ? AND status = 'pending'",
                    (
                        attempts,
                        current + lease_seconds,
                        _utc_iso(current),
                        row["event_id"],
                    ),
                )
                claimed.append(
                    OutboxEntry(
                        event_id=str(row["event_id"]),
                        task_id=str(row["task_id"]),
                        payload=json.loads(row["payload_json"]),
                        attempts=attempts,
                        status="pending",
                        last_error=str(row["last_error"]),
                        next_attempt_at=current + lease_seconds,
                        created_at=str(row["created_at"]),
                        updated_at=_utc_iso(current),
                    )
                )
            conn.commit()
        return claimed

    def record_attempt(
        self,
        event_id: str,
        *,
        now: float | None = None,
        lease_seconds: float = _LEASE_SECONDS,
    ) -> OutboxEntry | None:
        """为单条通知预扣一次尝试；不存在或已非 pending 时返回 None。"""
        current = time.time() if now is None else now
        with self._connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT * FROM notification_outbox WHERE event_id = ?",
                (event_id,),
            ).fetchone()
            if row is None or str(row["status"]) != "pending":
                conn.rollback()
                return None
            attempts = int(row["attempts"]) + 1
            next_at = current + lease_seconds
            conn.execute(
                "UPDATE notification_outbox"
                " SET attempts = ?, next_attempt_at = ?, updated_at = ?"
                " WHERE event_id = ?",
                (attempts, next_at, _utc_iso(current), event_id),
            )
            conn.commit()
            return OutboxEntry(
                event_id=str(row["event_id"]),
                task_id=str(row["task_id"]),
                payload=json.loads(row["payload_json"]),
                attempts=attempts,
                status="pending",
                last_error=str(row["last_error"]),
                next_attempt_at=next_at,
                created_at=str(row["created_at"]),
                updated_at=_utc_iso(current),
            )

    def mark_delivered(self, event_id: str) -> None:
        now = time.time()
        with self._connection() as conn:
            conn.execute(
                "UPDATE notification_outbox"
                " SET status = 'delivered', last_error = '', delivered_at = ?,"
                " updated_at = ? WHERE event_id = ?",
                (_utc_iso(now), _utc_iso(now), event_id),
            )
            conn.commit()

    def mark_retry(
        self,
        event_id: str,
        error: str,
        *,
        max_attempts: int,
        backoff_base_seconds: float,
        now: float | None = None,
    ) -> bool:
        """记录可重试失败；返回 True 表示仍待重试。"""
        current = time.time() if now is None else now
        with self._connection() as conn:
            row = conn.execute(
                "SELECT attempts FROM notification_outbox WHERE event_id = ?",
                (event_id,),
            ).fetchone()
            if row is None:
                return False
            attempts = int(row["attempts"])
            message = (error or "")[:300]
            if attempts >= max(1, int(max_attempts)):
                conn.execute(
                    "UPDATE notification_outbox"
                    " SET status = 'failed', last_error = ?, updated_at = ?"
                    " WHERE event_id = ?",
                    (message, _utc_iso(current), event_id),
                )
                conn.commit()
                return False
            base = max(0.0, float(backoff_base_seconds))
            delay = min(_BACKOFF_CAP_SECONDS, base * (2 ** max(0, attempts - 1)))
            conn.execute(
                "UPDATE notification_outbox"
                " SET last_error = ?, next_attempt_at = ?, updated_at = ?"
                " WHERE event_id = ?",
                (message, current + delay, _utc_iso(current), event_id),
            )
            conn.commit()
            return True

    def mark_failed(self, event_id: str, error: str) -> None:
        now = time.time()
        with self._connection() as conn:
            conn.execute(
                "UPDATE notification_outbox"
                " SET status = 'failed', last_error = ?, updated_at = ?"
                " WHERE event_id = ?",
                ((error or "")[:300], _utc_iso(now), event_id),
            )
            conn.commit()

    def stats(self, *, now: float | None = None) -> dict[str, Any]:
        current = time.time() if now is None else now
        with self._connection() as conn:
            rows = conn.execute(
                "SELECT status, COUNT(*) AS n FROM notification_outbox GROUP BY status"
            ).fetchall()
            due = conn.execute(
                "SELECT COUNT(*) FROM notification_outbox"
                " WHERE status = 'pending' AND next_attempt_at <= ?",
                (current,),
            ).fetchone()[0]
            oldest = conn.execute(
                "SELECT MIN(created_ts) FROM notification_outbox WHERE status = 'pending'"
            ).fetchone()[0]
        counts = {str(row["status"]): int(row["n"]) for row in rows}
        return {
            "enabled": True,
            "path": str(self._path),
            "pending": counts.get("pending", 0),
            "delivered": counts.get("delivered", 0),
            "failed": counts.get("failed", 0),
            "due": int(due),
            "oldest_pending_age_seconds": (
                int(max(0.0, current - float(oldest)))
                if oldest is not None
                else 0
            ),
        }

    @staticmethod
    def _entry(row: sqlite3.Row) -> OutboxEntry:
        return OutboxEntry(
            event_id=str(row["event_id"]),
            task_id=str(row["task_id"]),
            payload=json.loads(row["payload_json"]),
            attempts=int(row["attempts"]),
            status=str(row["status"]),
            last_error=str(row["last_error"]),
            next_attempt_at=float(row["next_attempt_at"]),
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
            delivered_at=row["delivered_at"],
        )


class NotificationOutboxWorker:
    """后台排空 outbox；未配置 Webhook 或未启用时为空操作。"""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        outbox: NotificationOutbox | None = None,
        notifier: Any | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        path = resolve_outbox_path(self._settings)
        if path is None:
            raise ValueError("notification_outbox_path 不能为空")
        self._outbox = outbox or NotificationOutbox(path)
        if notifier is None:
            from so_agent.runtime.notifications import EscalationNotifier

            notifier = EscalationNotifier(self._settings)
        self._notifier = notifier
        self._task: asyncio.Task[None] | None = None
        self._stop_event: asyncio.Event | None = None

    @property
    def enabled(self) -> bool:
        return bool(self._settings.notification_outbox_enabled) and bool(
            self._notifier.configured
        )

    @property
    def outbox(self) -> NotificationOutbox:
        return self._outbox

    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()

    async def drain_once(self) -> int:
        entries = self._outbox.claim_due(
            limit=max(1, int(self._settings.notification_outbox_batch_size)),
            now=time.time(),
        )
        for entry in entries:
            try:
                result = await self._notifier.deliver_once(
                    entry.payload, entry.event_id
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # 单条失败不影响同批其它通知
                logger.warning("outbox 投递异常：%s", type(exc).__name__)
                self._outbox.mark_retry(
                    entry.event_id,
                    f"{type(exc).__name__}: {exc}",
                    max_attempts=self._settings.notification_outbox_max_attempts,
                    backoff_base_seconds=(
                        self._settings.notification_outbox_retry_backoff_seconds
                    ),
                )
                continue
            if result.sent:
                self._outbox.mark_delivered(entry.event_id)
                continue
            permanent = result.attempts == 0 or (
                result.status_code is not None
                and not _retryable_status(result.status_code)
            )
            if permanent:
                self._outbox.mark_failed(entry.event_id, result.error)
            else:
                self._outbox.mark_retry(
                    entry.event_id,
                    result.error,
                    max_attempts=self._settings.notification_outbox_max_attempts,
                    backoff_base_seconds=(
                        self._settings.notification_outbox_retry_backoff_seconds
                    ),
                )
        return len(entries)

    async def start(self) -> None:
        if not self.enabled:
            return
        if self._task is not None and not self._task.done():
            return
        self._stop_event = asyncio.Event()
        self._task = asyncio.create_task(
            self._run(self._stop_event), name="notification-outbox"
        )
        logger.info("通知 outbox 后台排空已启动：%s", self._outbox.path)

    async def stop(self) -> None:
        stop_event = self._stop_event
        task = self._task
        self._stop_event = None
        self._task = None
        if stop_event is not None:
            stop_event.set()
        if task is not None and not task.done():
            try:
                await asyncio.wait_for(task, timeout=5)
            except asyncio.TimeoutError:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
            except Exception:
                pass

    async def _run(self, stop_event: asyncio.Event) -> None:
        interval = max(
            0.05, float(self._settings.notification_outbox_drain_interval_seconds)
        )
        while not stop_event.is_set():
            try:
                await self.drain_once()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("通知 outbox 排空失败：%s", exc)
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=interval)
            except asyncio.TimeoutError:
                continue


__all__ = [
    "NotificationOutbox",
    "NotificationOutboxWorker",
    "OutboxEntry",
    "resolve_outbox_path",
]

"""高风险动作审批记录的持久化存储（SQLite/WAL）。

一次高风险工具调用生成一条审批记录，以 ``idempotency_key`` 唯一约束：
同一任务、同一工具、同一参数重试时复用原记录，不会重复通知或重复
弹审批。记录跨进程持久化，Web 控制台可在任务等待期间作出决定。
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

from so_agent.config import Settings
from so_agent.runtime.logging_setup import get_logger

logger = get_logger("runtime.approvals")

PENDING = "pending"
APPROVED = "approved"
DENIED = "denied"
EXPIRED = "expired"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS tool_approvals (
    id TEXT PRIMARY KEY,
    idempotency_key TEXT NOT NULL UNIQUE,
    task_id TEXT NOT NULL,
    tool_name TEXT NOT NULL,
    risk TEXT NOT NULL,
    summary TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    reason TEXT NOT NULL DEFAULT '',
    decided_by TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    created_ts REAL NOT NULL,
    expires_ts REAL NOT NULL,
    decided_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_tool_approvals_pending
    ON tool_approvals(status, created_ts);
CREATE INDEX IF NOT EXISTS idx_tool_approvals_task
    ON tool_approvals(task_id, created_ts);
"""


def _utc_iso(ts: float | None = None) -> str:
    moment = datetime.fromtimestamp(
        ts if ts is not None else time.time(), timezone.utc
    )
    return moment.isoformat()


def resolve_approval_store_path(
    settings: Settings, *, base_dir: str | Path | None = None
) -> Path:
    """解析审批库路径：显式配置 > ``base_dir`` > 当前目录。"""
    raw = (getattr(settings, "approval_store_path", "") or "").strip()
    root = Path(base_dir) if base_dir is not None else Path.cwd()
    if not raw:
        return root / "tool_approvals.db"
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = root / path
    return path


@dataclass(frozen=True)
class ApprovalRequest:
    """一条审批记录。"""

    id: str
    idempotency_key: str
    task_id: str
    tool_name: str
    risk: str
    summary: str
    status: str
    reason: str = ""
    decided_by: str = ""
    created_at: str = ""
    expires_ts: float = 0.0
    decided_at: str | None = None

    @property
    def pending(self) -> bool:
        return self.status == PENDING

    def to_dict(self) -> dict[str, Any]:
        return {
            "approval_id": self.id,
            "task_id": self.task_id,
            "tool_name": self.tool_name,
            "risk": self.risk,
            "summary": self.summary,
            "status": self.status,
            "reason": self.reason,
            "decided_by": self.decided_by,
            "created_at": self.created_at,
            "expires_at": _utc_iso(self.expires_ts) if self.expires_ts else "",
            "decided_at": self.decided_at,
        }


class ApprovalStore:
    """SQLite 审批记录存储（多进程安全，WAL + busy timeout）。"""

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

    def request(
        self,
        *,
        idempotency_key: str,
        task_id: str,
        tool_name: str,
        risk: str,
        summary: str,
        expires_ts: float,
    ) -> tuple[ApprovalRequest, bool]:
        """创建或复用审批记录；返回 ``(记录, 是否新建)``。"""
        current = time.time()
        request_id = "approval-" + idempotency_key[:24]
        with self._connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT * FROM tool_approvals WHERE idempotency_key = ?",
                (idempotency_key,),
            ).fetchone()
            if row is not None:
                conn.commit()
                return self._to_model(row), False
            conn.execute(
                "INSERT INTO tool_approvals("
                "id, idempotency_key, task_id, tool_name, risk, summary,"
                " status, created_at, created_ts, expires_ts"
                ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    request_id,
                    idempotency_key,
                    task_id,
                    tool_name,
                    risk,
                    summary,
                    PENDING,
                    _utc_iso(current),
                    current,
                    expires_ts,
                ),
            )
            conn.commit()
            row = conn.execute(
                "SELECT * FROM tool_approvals WHERE id = ?", (request_id,)
            ).fetchone()
        return self._to_model(row), True

    def get(self, request_id: str) -> ApprovalRequest | None:
        with self._connection() as conn:
            row = conn.execute(
                "SELECT * FROM tool_approvals WHERE id = ?", (request_id,)
            ).fetchone()
        return self._to_model(row) if row else None

    def list(
        self,
        *,
        status: str | None = None,
        task_id: str | None = None,
        limit: int = 100,
    ) -> list[ApprovalRequest]:
        clauses: list[str] = []
        params: list[Any] = []
        if status:
            clauses.append("status = ?")
            params.append(status)
        if task_id:
            clauses.append("task_id = ?")
            params.append(task_id)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(max(1, int(limit)))
        with self._connection() as conn:
            rows = conn.execute(
                "SELECT * FROM tool_approvals" + where
                + " ORDER BY created_ts DESC LIMIT ?",
                params,
            ).fetchall()
        return [self._to_model(row) for row in rows]

    def decide(
        self,
        request_id: str,
        *,
        approved: bool,
        decided_by: str = "",
        reason: str = "",
    ) -> ApprovalRequest:
        """对 pending 记录作出决定；非 pending 或已过期时抛 ValueError。"""
        current = time.time()
        target = APPROVED if approved else DENIED
        with self._connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT * FROM tool_approvals WHERE id = ?", (request_id,)
            ).fetchone()
            if row is None:
                conn.commit()
                raise ValueError(f"审批记录不存在：{request_id}")
            if row["status"] != PENDING:
                conn.commit()
                raise ValueError(f"审批记录已处理：{row['status']}")
            if float(row["expires_ts"]) <= current:
                conn.execute(
                    "UPDATE tool_approvals SET status = ?, reason = ?,"
                    " decided_at = ? WHERE id = ?",
                    (EXPIRED, "审批已过期", _utc_iso(current), request_id),
                )
                conn.commit()
                raise ValueError("审批已过期")
            conn.execute(
                "UPDATE tool_approvals SET status = ?, reason = ?,"
                " decided_by = ?, decided_at = ? WHERE id = ?",
                (
                    target,
                    reason,
                    decided_by,
                    _utc_iso(current),
                    request_id,
                ),
            )
            conn.commit()
            row = conn.execute(
                "SELECT * FROM tool_approvals WHERE id = ?", (request_id,)
            ).fetchone()
        return self._to_model(row)

    def expire(self, request_id: str, reason: str = "") -> ApprovalRequest | None:
        """把 pending 记录标记为过期（已决定的记录保持不变）。"""
        current = time.time()
        with self._connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                "UPDATE tool_approvals SET status = ?, reason = ?,"
                " decided_at = ? WHERE id = ? AND status = ?",
                (
                    EXPIRED,
                    reason or "审批超时",
                    _utc_iso(current),
                    request_id,
                    PENDING,
                ),
            )
            conn.commit()
        return self.get(request_id)

    async def wait(
        self,
        request_id: str,
        *,
        timeout: float,
        poll_seconds: float = 1.0,
    ) -> ApprovalRequest:
        """等待决定；超时后标记过期并返回过期记录。"""
        deadline = time.monotonic() + max(0.0, float(timeout))
        interval = max(0.05, float(poll_seconds))
        while True:
            current = self.get(request_id)
            if current is None:
                raise ValueError(f"审批记录不存在：{request_id}")
            if not current.pending:
                return current
            remaining = deadline - time.monotonic()
            if current.expires_ts <= time.time() or remaining <= 0:
                expired = self.expire(request_id)
                return expired or current
            await asyncio.sleep(min(interval, remaining))

    @staticmethod
    def _to_model(row: sqlite3.Row) -> ApprovalRequest:
        return ApprovalRequest(
            id=str(row["id"]),
            idempotency_key=str(row["idempotency_key"]),
            task_id=str(row["task_id"]),
            tool_name=str(row["tool_name"]),
            risk=str(row["risk"]),
            summary=str(row["summary"]),
            status=str(row["status"]),
            reason=str(row["reason"]),
            decided_by=str(row["decided_by"]),
            created_at=str(row["created_at"]),
            expires_ts=float(row["expires_ts"]),
            decided_at=row["decided_at"],
        )


def encode_summary(payload: dict[str, Any]) -> str:
    """把风险摘要编码为稳定 JSON（用于持久化展示）。"""
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)

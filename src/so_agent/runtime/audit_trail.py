"""跨任务防篡改审计链（可选能力）。

默认事件仍写入内存与任务日志文件；设置 ``SO_AGENT_AUDIT_TRAIL_PATH``
后，``EventLogger`` 的每条事件同步追加到独立 SQLite 审计库，并按
SHA-256 前向哈希链串联：

    entry_hash = sha256(prev_hash + "\\n" + 规范化事件字段)

可发现中间记录被修改、重排或删除。边界：没有外部锚点时无法发现
尾部整体截断；审计库只追加、不提供裁剪接口，因为裁剪会破坏链。

审计写入失败只记录警告，不阻断任务主流程；需要强一致审计的部署应
监控该警告并为审计库所在磁盘提供持久化保障。
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from so_agent.runtime.logging_setup import get_logger

logger = get_logger("runtime.audit_trail")

AUDIT_TRAIL_ENV = "SO_AGENT_AUDIT_TRAIL_PATH"
GENESIS_HASH = "0" * 64

_SCHEMA = """
CREATE TABLE IF NOT EXISTS audit_events (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id TEXT NOT NULL,
    event_type TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    task_id TEXT,
    agent_id TEXT,
    payload_json TEXT NOT NULL,
    prev_hash TEXT NOT NULL,
    entry_hash TEXT NOT NULL,
    recorded_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_audit_events_task
    ON audit_events(task_id, seq);
"""


@dataclass(frozen=True)
class AuditEntry:
    seq: int
    event_id: str
    entry_hash: str


@dataclass(frozen=True)
class AuditVerification:
    ok: bool
    entries: int
    first_bad_seq: int | None = None
    reason: str = ""


def _payload_json(payload: Any) -> str:
    if not isinstance(payload, dict):
        payload = {}
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def _entry_digest(
    prev_hash: str,
    *,
    event_id: str,
    event_type: str,
    timestamp: str,
    task_id: str | None,
    agent_id: str | None,
    payload_json: str,
) -> str:
    fields = [
        event_id or "",
        event_type or "",
        timestamp or "",
        task_id,
        agent_id,
        payload_json,
    ]
    material = prev_hash + "\n" + json.dumps(
        fields, ensure_ascii=False, separators=(",", ":")
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _connect(path: Path, *, read_only: bool = False) -> sqlite3.Connection:
    if read_only:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=10.0)
    else:
        conn = sqlite3.connect(path, timeout=10.0)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA busy_timeout=10000")
    return conn


class AuditTrail:
    """单文件 SQLite 审计链；追加时用 ``BEGIN IMMEDIATE`` 串行化跨进程写入。"""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path).expanduser()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        conn = _connect(self._path)
        try:
            conn.executescript(_SCHEMA)
            conn.commit()
        finally:
            conn.close()

    @property
    def path(self) -> Path:
        return self._path

    def append(self, record: dict[str, Any]) -> AuditEntry:
        payload_json = _payload_json(record.get("payload"))
        event_id = str(record.get("event_id") or "")
        event_type = str(record.get("event_type") or "")
        timestamp = str(record.get("timestamp") or "")
        raw_task_id = record.get("task_id")
        raw_agent_id = record.get("agent_id")
        task_id = None if raw_task_id is None else str(raw_task_id)
        agent_id = None if raw_agent_id is None else str(raw_agent_id)

        with self._lock:
            conn = _connect(self._path)
            try:
                conn.execute("BEGIN IMMEDIATE")
                row = conn.execute(
                    "SELECT entry_hash FROM audit_events ORDER BY seq DESC LIMIT 1"
                ).fetchone()
                prev_hash = str(row[0]) if row else GENESIS_HASH
                digest = _entry_digest(
                    prev_hash,
                    event_id=event_id,
                    event_type=event_type,
                    timestamp=timestamp,
                    task_id=task_id,
                    agent_id=agent_id,
                    payload_json=payload_json,
                )
                cursor = conn.execute(
                    """
                    INSERT INTO audit_events (
                        event_id, event_type, timestamp, task_id, agent_id,
                        payload_json, prev_hash, entry_hash, recorded_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
                    """,
                    (
                        event_id,
                        event_type,
                        timestamp,
                        task_id,
                        agent_id,
                        payload_json,
                        prev_hash,
                        digest,
                    ),
                )
                seq = int(cursor.lastrowid or 0)
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()
        return AuditEntry(seq=seq, event_id=event_id, entry_hash=digest)


def verify_audit_trail(path: str | Path) -> AuditVerification:
    """逐条复核哈希链；返回首个断点，不修改审计库。"""
    audit_path = Path(path).expanduser()
    if not audit_path.is_file():
        return AuditVerification(False, 0, None, "审计库不存在")
    conn = _connect(audit_path, read_only=True)
    try:
        rows = conn.execute(
            """
            SELECT seq, event_id, event_type, timestamp, task_id, agent_id,
                   payload_json, prev_hash, entry_hash
            FROM audit_events ORDER BY seq
            """
        ).fetchall()
    except sqlite3.DatabaseError as exc:
        return AuditVerification(False, 0, None, f"审计库无法读取：{exc}")
    finally:
        conn.close()

    expected_prev = GENESIS_HASH
    for index, row in enumerate(rows, start=1):
        (
            seq,
            event_id,
            event_type,
            timestamp,
            task_id,
            agent_id,
            payload_json,
            prev_hash,
            entry_hash,
        ) = row
        if int(seq) != index:
            return AuditVerification(
                False, len(rows), int(seq), "记录序号不连续（可能存在删除）"
            )
        if str(prev_hash) != expected_prev:
            return AuditVerification(
                False, len(rows), int(seq), "前向哈希不匹配"
            )
        digest = _entry_digest(
            str(prev_hash),
            event_id=str(event_id),
            event_type=str(event_type),
            timestamp=str(timestamp),
            task_id=task_id,
            agent_id=agent_id,
            payload_json=str(payload_json),
        )
        if digest != str(entry_hash):
            return AuditVerification(
                False, len(rows), int(seq), "记录哈希不匹配（内容被修改）"
            )
        expected_prev = str(entry_hash)
    return AuditVerification(True, len(rows))


def _configured_path() -> str:
    value = os.environ.get(AUDIT_TRAIL_ENV, "").strip()
    if value:
        return value
    try:
        from so_agent.config import get_settings

        return str(get_settings().audit_trail_path or "").strip()
    except Exception:
        return ""


_default_lock = threading.Lock()
_default_trail: AuditTrail | None = None
_default_path: str = ""


def get_default_audit_trail() -> AuditTrail | None:
    """按环境变量/配置返回进程级共享审计链；未配置时返回 None。"""
    global _default_trail, _default_path
    path = _configured_path()
    if not path:
        return None
    with _default_lock:
        if _default_trail is not None and _default_path == path:
            return _default_trail
        try:
            _default_trail = AuditTrail(path)
            _default_path = path
        except Exception:
            _default_trail = None
            _default_path = path
            logger.warning("审计链初始化失败，事件不会写入跨任务审计库", exc_info=True)
        return _default_trail


def reset_default_audit_trail() -> None:
    global _default_trail, _default_path
    with _default_lock:
        _default_trail = None
        _default_path = ""

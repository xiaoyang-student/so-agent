"""跨任务防篡改审计链测试。"""

from __future__ import annotations

import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor

import pytest

from so_agent.audit_cli import main as audit_main
from so_agent.runtime.audit_trail import (
    AUDIT_TRAIL_ENV,
    AuditTrail,
    get_default_audit_trail,
    reset_default_audit_trail,
    verify_audit_trail,
)
from so_agent.runtime.events import (
    EVENT_STATUS_TRANSITION,
    EVENT_TASK_CREATED,
    EventLogger,
)


def _record(index: int, task_id: str = "task-1") -> dict:
    return {
        "event_id": f"event-{index}",
        "event_type": EVENT_TASK_CREATED,
        "timestamp": f"2026-01-01T00:00:{index:02d}+00:00",
        "task_id": task_id,
        "agent_id": None,
        "payload": {
            "task_id": task_id,
            "index": index,
            "nested": {"beta": 2, "alpha": 1},
        },
    }


@pytest.fixture(autouse=True)
def _isolate_default_trail(monkeypatch):
    monkeypatch.delenv(AUDIT_TRAIL_ENV, raising=False)
    reset_default_audit_trail()
    yield
    reset_default_audit_trail()


def test_append_and_verify(tmp_path):
    path = tmp_path / "audit.db"
    trail = AuditTrail(path)
    entries = [trail.append(_record(i)) for i in range(3)]

    assert [entry.seq for entry in entries] == [1, 2, 3]
    result = verify_audit_trail(path)
    assert result.ok is True
    assert result.entries == 3
    assert result.first_bad_seq is None


def test_payload_tampering_is_detected(tmp_path):
    path = tmp_path / "audit.db"
    trail = AuditTrail(path)
    for i in range(3):
        trail.append(_record(i))
    with sqlite3.connect(path) as conn:
        conn.execute(
            "UPDATE audit_events SET payload_json = ? WHERE seq = 2",
            ('{"index":99}',),
        )

    result = verify_audit_trail(path)
    assert result.ok is False
    assert result.first_bad_seq == 2
    assert "修改" in result.reason


def test_middle_row_deletion_is_detected(tmp_path):
    path = tmp_path / "audit.db"
    trail = AuditTrail(path)
    for i in range(4):
        trail.append(_record(i))
    with sqlite3.connect(path) as conn:
        conn.execute("DELETE FROM audit_events WHERE seq = 2")

    result = verify_audit_trail(path)
    assert result.ok is False
    assert result.first_bad_seq == 3
    assert "不连续" in result.reason


def test_prev_hash_tampering_is_detected(tmp_path):
    path = tmp_path / "audit.db"
    trail = AuditTrail(path)
    for i in range(3):
        trail.append(_record(i))
    with sqlite3.connect(path) as conn:
        conn.execute(
            "UPDATE audit_events SET prev_hash = ? WHERE seq = 2",
            ("f" * 64,),
        )

    result = verify_audit_trail(path)
    assert result.ok is False
    assert result.first_bad_seq == 2
    assert "前向哈希" in result.reason


def test_missing_database_is_reported(tmp_path):
    result = verify_audit_trail(tmp_path / "missing.db")
    assert result.ok is False
    assert result.entries == 0
    assert "不存在" in result.reason


def test_event_logger_appends_every_event(tmp_path):
    path = tmp_path / "audit.db"
    logger = EventLogger(audit_trail=AuditTrail(path))
    logger.log_event(EVENT_TASK_CREATED, {"task_id": "task-1"})
    logger.log_event(
        EVENT_STATUS_TRANSITION,
        {"task_id": "task-1", "from_status": "queued", "to_status": "running"},
    )

    result = verify_audit_trail(path)
    assert result.ok is True
    assert result.entries == 2


def test_audit_write_failure_does_not_break_event_logging():
    class BrokenTrail:
        def append(self, record):
            raise RuntimeError("disk full")

    logger = EventLogger(audit_trail=BrokenTrail())
    record = logger.log_event(EVENT_TASK_CREATED, {"task_id": "task-1"})

    assert record["event_type"] == EVENT_TASK_CREATED
    assert len(logger) == 1


def test_default_trail_disabled_without_env():
    assert get_default_audit_trail() is None


def test_default_trail_uses_env_path(monkeypatch, tmp_path):
    path = tmp_path / "audit.db"
    monkeypatch.setenv(AUDIT_TRAIL_ENV, str(path))
    trail = get_default_audit_trail()
    assert trail is not None
    assert trail.path == path

    logger = EventLogger()
    logger.log_event(EVENT_TASK_CREATED, {"task_id": "task-1"})
    assert verify_audit_trail(path).entries == 1


def test_changing_env_path_reinitializes(monkeypatch, tmp_path):
    first = tmp_path / "first.db"
    second = tmp_path / "second.db"
    monkeypatch.setenv(AUDIT_TRAIL_ENV, str(first))
    assert get_default_audit_trail().path == first
    monkeypatch.setenv(AUDIT_TRAIL_ENV, str(second))
    assert get_default_audit_trail().path == second


def test_concurrent_appends_keep_chain_valid(tmp_path):
    path = tmp_path / "audit.db"
    trail = AuditTrail(path)

    def append(index: int) -> None:
        trail.append(_record(index, task_id=f"task-{index % 3}"))

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(append, range(40)))

    result = verify_audit_trail(path)
    assert result.ok is True
    assert result.entries == 40


def test_cli_verify_json_success(tmp_path, capsys):
    path = tmp_path / "audit.db"
    trail = AuditTrail(path)
    trail.append(_record(0))

    code = audit_main(["verify", str(path), "--json"])
    payload = json.loads(capsys.readouterr().out)

    assert code == 0
    assert payload["ok"] is True
    assert payload["entries"] == 1


def test_cli_verify_detects_tampering(tmp_path, capsys):
    path = tmp_path / "audit.db"
    trail = AuditTrail(path)
    trail.append(_record(0))
    with sqlite3.connect(path) as conn:
        conn.execute("UPDATE audit_events SET entry_hash = ? WHERE seq = 1", ("0" * 64,))

    code = audit_main(["verify", str(path), "--json"])
    payload = json.loads(capsys.readouterr().out)

    assert code == 1
    assert payload["ok"] is False
    assert payload["first_bad_seq"] == 1


def test_cli_requires_path(monkeypatch, capsys):
    monkeypatch.delenv(AUDIT_TRAIL_ENV, raising=False)
    code = audit_main(["verify", "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert code == 2
    assert payload["ok"] is False

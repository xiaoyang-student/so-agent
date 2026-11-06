"""SQLite WAL 参数与 Web 单实例锁测试。"""

from __future__ import annotations

import sqlite3

import pytest
from fastapi.testclient import TestClient

from so_agent.config import Settings
from so_agent.persistence import TaskStore
from so_agent.runtime.instance_lock import InstanceLock, InstanceLockError
from so_agent.web.app import create_app
from so_agent.web.store import WebTaskStore


class TestSqliteTuning:
    def test_web_store_enables_wal_and_busy_timeout(self, tmp_path):
        store = WebTaskStore(tmp_path / "web_tasks.db")
        mode = store._conn.execute("PRAGMA journal_mode").fetchone()[0]
        timeout = store._conn.execute("PRAGMA busy_timeout").fetchone()[0]
        assert str(mode).lower() == "wal"
        assert timeout == 10_000
        store.close()

    def test_task_store_enables_wal(self, tmp_path):
        store = TaskStore(tmp_path / "task_store.db")
        mode = store._conn.execute("PRAGMA journal_mode").fetchone()[0]
        assert str(mode).lower() == "wal"
        store.close()

    def test_wal_allows_concurrent_reader_while_writer_open(self, tmp_path):
        path = tmp_path / "shared.db"
        writer = WebTaskStore(path)
        reader = sqlite3.connect(path, timeout=1)
        try:
            assert reader.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
            writer._conn.execute(
                "INSERT INTO web_tasks (task_id, objective, status, created_at) "
                "VALUES ('t1', 'o', 'completed', '2026-01-01')"
            )
            writer._conn.commit()
            assert reader.execute(
                "SELECT COUNT(*) FROM web_tasks"
            ).fetchone()[0] == 1
        finally:
            reader.close()
            writer.close()


class TestInstanceLock:
    def test_second_holder_is_rejected(self, tmp_path):
        path = tmp_path / "instance.lock"
        first = InstanceLock(path)
        second = InstanceLock(path)
        first.acquire()
        try:
            assert first.held
            with pytest.raises(InstanceLockError):
                second.acquire()
        finally:
            first.release()
        assert not first.held
        second.acquire()
        second.release()

    def test_lock_file_records_owner(self, tmp_path):
        path = tmp_path / "instance.lock"
        with InstanceLock(path):
            assert path.exists()
            assert path.stat().st_size > 0
        content = path.read_text(encoding="ascii").strip()
        assert content.isdigit()

    def test_web_app_rejects_second_instance(self, tmp_path):
        settings = Settings(_env_file=None, web_scheduler_enabled=False)
        base = tmp_path / "web"
        first = create_app(settings=settings, base_dir=base)
        second = create_app(settings=settings, base_dir=base)

        with TestClient(first):
            with pytest.raises(InstanceLockError):
                with TestClient(second):
                    pass

        with TestClient(second):
            assert second.state.manager.base_dir == base

"""单元测试：schema 版本机制与存储兼容迁移（任务 44）。

覆盖基线冻结的兼容承诺：

- 全新空库一次建到当前基线版本并登记版本号；
- 隐式迁移时代的旧库（无版本表）自动升级：v1 ResultStore 补列且
  ``executions`` / ``plan_versions`` 既有数据完整保留；旧代码已隐式
  补列的库登记为 v2 不重复迁移；TaskStore 旧库原样登记 v1；
- 重复打开幂等：版本不变、数据完好、无重复迁移副作用；
- 迁移失败整体回滚：schema 与版本号保持原状，失败后库仍可正常
  重新迁移（不留下半迁移损坏状态）；
- 未知高版本 fail closed：拒绝打开且过程零写入（文件字节不变）；
- 版本机制单元语义：显式事务回滚、迁移链断链配置报错；
- 版本读取与 ``row_factory`` 无关（任务 56）：裸连接（默认 tuple
  row）可读合法版本并完成迁移；非法版本值在 tuple row 与
  ``sqlite3.Row`` 两种连接下都 fail closed 为 SchemaVersionError
  且零写入，不泄漏 TypeError。
"""

from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path

import pytest

import so_agent.result_store as result_store_mod
from so_agent.persistence import TASK_STORE_BASELINE_VERSION, TaskStore
from so_agent.result_store import (
    RESULT_STORE_BASELINE_VERSION,
    SQLiteResultStore,
)
from so_agent.schema_migrations import (
    SCHEMA_META_TABLE,
    MigrationStep,
    SchemaVersionError,
    ensure_schema,
    run_in_transaction,
    table_columns,
    table_exists,
)

# ---------------------------------------------------------------------------
# 旧库构造辅助（与历史实现 schema 逐字一致；不含 schema_meta 版本表）
# ---------------------------------------------------------------------------

_V1_EXECUTIONS_SQL = """
    CREATE TABLE executions (
        execution_id TEXT PRIMARY KEY,
        task_id TEXT NOT NULL,
        subtask_id TEXT NOT NULL,
        plan_version INTEGER NOT NULL,
        agent_id TEXT,
        attempt INTEGER DEFAULT 1,
        input_fingerprint TEXT DEFAULT '',
        output_fingerprint TEXT DEFAULT '',
        status TEXT DEFAULT 'pending',
        result_json TEXT,
        token_usage INTEGER DEFAULT 0,
        duration REAL DEFAULT 0.0,
        failure_type TEXT,
        recovery_action TEXT,
        started_at TEXT,
        completed_at TEXT
    )
"""

_V1_PLAN_VERSIONS_SQL = """
    CREATE TABLE plan_versions (
        version INTEGER PRIMARY KEY,
        specs_json TEXT NOT NULL,
        executions_json TEXT DEFAULT '{}',
        review_json TEXT,
        trigger_json TEXT,
        created_at TEXT DEFAULT ''
    )
"""

_V1_TASKS_SQL = """
    CREATE TABLE tasks (
        task_id TEXT PRIMARY KEY,
        objective TEXT NOT NULL,
        constraints TEXT DEFAULT '',
        acceptance_criteria TEXT DEFAULT '',
        task_type TEXT NOT NULL,
        status TEXT NOT NULL DEFAULT 'pending',
        schedule_spec TEXT,
        checkpoint TEXT,
        result TEXT,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        next_run_at TEXT
    )
"""

_V1_TASK_LOGS_SQL = """
    CREATE TABLE task_logs (
        log_id INTEGER PRIMARY KEY AUTOINCREMENT,
        task_id TEXT NOT NULL REFERENCES tasks(task_id),
        event_type TEXT NOT NULL,
        payload TEXT,
        created_at TEXT NOT NULL
    )
"""

_EXECUTION_ROW = (
    "e-legacy-1", "T-legacy", "s1", 1, "agent-1", 1,
    "fp-legacy", "out-fp", "completed",
    '{"subtask_id":"s1","success":true,"output":"旧产出"}',
    42, 1.25, None, None,
    "2026-01-01T00:00:00+00:00", "2026-01-01T00:00:01+00:00",
)


def _make_v1_result_db(path: Path, *, with_column: bool = False) -> None:
    """构造隐式迁移时代的 ResultStore 旧库（无版本表），并写入数据。"""
    conn = sqlite3.connect(str(path))
    try:
        conn.execute(_V1_EXECUTIONS_SQL)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_exec_fingerprint "
            "ON executions(input_fingerprint)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_exec_subtask "
            "ON executions(subtask_id, plan_version)"
        )
        conn.execute(_V1_PLAN_VERSIONS_SQL)
        if with_column:
            conn.execute(
                "ALTER TABLE plan_versions "
                "ADD COLUMN fingerprints_json TEXT DEFAULT '{}'"
            )
        conn.execute(
            "INSERT INTO executions VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            _EXECUTION_ROW,
        )
        conn.execute(
            "INSERT INTO plan_versions (version, specs_json, executions_json,"
            " review_json, trigger_json, created_at) VALUES (1,'[]','{}',"
            "NULL,NULL,'2026-01-01T00:00:00+00:00')"
        )
        conn.commit()
    finally:
        conn.close()


def _make_v1_task_db(path: Path) -> None:
    """构造隐式迁移时代的 TaskStore 旧库（无版本表），并写入数据。"""
    conn = sqlite3.connect(str(path))
    try:
        conn.execute(_V1_TASKS_SQL)
        conn.execute(_V1_TASK_LOGS_SQL)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_tasks_next_run ON tasks(next_run_at)"
        )
        conn.execute(
            "INSERT INTO tasks (task_id, objective, task_type, status,"
            " created_at, updated_at) VALUES"
            " ('t-legacy','旧目标','full_workflow','pending',"
            "'2026-01-01T00:00:00+00:00','2026-01-01T00:00:00+00:00')"
        )
        conn.execute(
            "INSERT INTO task_logs (task_id, event_type, created_at) VALUES"
            " ('t-legacy','error','2026-01-01T00:00:01+00:00')"
        )
        conn.commit()
    finally:
        conn.close()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stored_version(path: Path, key: str) -> int | None:
    conn = sqlite3.connect(str(path))
    try:
        if not table_exists(conn, SCHEMA_META_TABLE):
            return None
        row = conn.execute(
            f"SELECT value FROM {SCHEMA_META_TABLE} WHERE key = ?", (key,)
        ).fetchone()
        return int(row[0]) if row else None
    finally:
        conn.close()


def _set_stored_version(path: Path, key: str, value: str) -> None:
    conn = sqlite3.connect(str(path))
    try:
        conn.execute(
            f"UPDATE {SCHEMA_META_TABLE} SET value = ? WHERE key = ?",
            (value, key),
        )
        conn.commit()
    finally:
        conn.close()


def _raw_stored_version(path: Path, key: str) -> str | None:
    """读取版本记录的原始字符串值（不做 int 转换，供非法值断言用）。"""
    conn = sqlite3.connect(str(path))
    try:
        row = conn.execute(
            f"SELECT value FROM {SCHEMA_META_TABLE} WHERE key = ?", (key,)
        ).fetchone()
        return str(row[0]) if row else None
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# 全新空库：一次建到当前基线
# ---------------------------------------------------------------------------

class TestFreshDatabase:
    def test_result_store_fresh_baseline(self, tmp_path: Path) -> None:
        db = tmp_path / "rs.db"
        store = SQLiteResultStore(str(db))
        try:
            assert RESULT_STORE_BASELINE_VERSION == 2
            assert _stored_version(db, "result_store_schema_version") == 2
            conn = sqlite3.connect(str(db))
            try:
                assert table_exists(conn, "executions")
                assert table_exists(conn, "plan_versions")
                assert "fingerprints_json" in table_columns(conn, "plan_versions")
            finally:
                conn.close()
        finally:
            store.close()

    def test_task_store_fresh_baseline(self, tmp_path: Path) -> None:
        db = tmp_path / "ts.db"
        store = TaskStore(db)
        try:
            assert TASK_STORE_BASELINE_VERSION == 1
            assert _stored_version(db, "task_store_schema_version") == 1
            conn = sqlite3.connect(str(db))
            try:
                assert table_exists(conn, "tasks")
                assert table_exists(conn, "task_logs")
            finally:
                conn.close()
        finally:
            store.close()


# ---------------------------------------------------------------------------
# 旧库自动升级：数据完整保留
# ---------------------------------------------------------------------------

class TestLegacyUpgrade:
    def test_v1_result_store_auto_upgrade_keeps_data(self, tmp_path: Path) -> None:
        """v1 旧库（无 fingerprints_json）自动升级：补列 + 数据无损。"""
        db = tmp_path / "legacy.db"
        _make_v1_result_db(db, with_column=False)

        store = SQLiteResultStore(str(db))
        try:
            # 版本推进到 v2
            assert _stored_version(db, "result_store_schema_version") == 2
            # 既有数据完整保留
            loaded_pv = store.get_plan_version(1)
            assert loaded_pv is not None
            assert loaded_pv.version == 1
            rows = store._conn.execute(
                "SELECT * FROM executions WHERE execution_id = 'e-legacy-1'"
            ).fetchall()
            assert len(rows) == 1
            assert rows[0]["input_fingerprint"] == "fp-legacy"
            # plan_versions 旧行的补列取默认值（与隐式迁移时代行为一致）
            pv_rows = store._conn.execute(
                "SELECT fingerprints_json FROM plan_versions WHERE version = 1"
            ).fetchall()
            assert pv_rows[0]["fingerprints_json"] == "{}"
            # 升级后可正常读写（契约行为不受迁移影响）
            store.save_plan_version(loaded_pv)
            assert store.get_plan_version(1) is not None
        finally:
            store.close()

    def test_implicitly_migrated_store_registers_v2_without_remigration(
        self, tmp_path: Path
    ) -> None:
        """旧代码已隐式补列的库：直接登记 v2，不做二次迁移。"""
        db = tmp_path / "implicit.db"
        _make_v1_result_db(db, with_column=True)
        assert not table_exists(sqlite3.connect(str(db)), SCHEMA_META_TABLE)

        store = SQLiteResultStore(str(db))
        try:
            assert _stored_version(db, "result_store_schema_version") == 2
            rows = store._conn.execute(
                "SELECT fingerprints_json FROM plan_versions"
            ).fetchall()
            assert len(rows) == 1
        finally:
            store.close()

    def test_v1_task_store_registers_v1_keeps_data(self, tmp_path: Path) -> None:
        """TaskStore 旧库（无版本表）：登记 v1，tasks/task_logs 数据保留。"""
        db = tmp_path / "legacy.db"
        _make_v1_task_db(db)

        store = TaskStore(db)
        try:
            assert _stored_version(db, "task_store_schema_version") == 1
            record = store.get_task("t-legacy")
            assert record is not None
            assert record.objective == "旧目标"
            logs = store._conn.execute(
                "SELECT * FROM task_logs WHERE task_id = 't-legacy'"
            ).fetchall()
            assert len(logs) == 1
        finally:
            store.close()


# ---------------------------------------------------------------------------
# 重复打开幂等
# ---------------------------------------------------------------------------

class TestIdempotentReopen:
    @pytest.mark.parametrize(
        "open_store,version_key",
        [
            ("result", "result_store_schema_version"),
            ("task", "task_store_schema_version"),
        ],
    )
    def test_repeated_open_is_noop(self, tmp_path, open_store, version_key):
        """重复打开：版本不变、数据完好、无重复迁移副作用。"""
        db = tmp_path / "reopen.db"
        if open_store == "result":
            first = SQLiteResultStore(str(db))
        else:
            first = TaskStore(db)
        first.close()
        version_after_first = _stored_version(db, version_key)

        if open_store == "result":
            second = SQLiteResultStore(str(db))
        else:
            second = TaskStore(db)
        second.close()

        assert _stored_version(db, version_key) == version_after_first
        # 表集合不变（无重复建表/迁移产物）
        conn = sqlite3.connect(str(db))
        try:
            tables = {
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table'"
                )
            }
        finally:
            conn.close()
        assert SCHEMA_META_TABLE in tables

    def test_reopen_preserves_result_data(self, tmp_path: Path) -> None:
        """v1 升级后的库重复打开：数据与版本保持稳定。"""
        db = tmp_path / "stable.db"
        _make_v1_result_db(db)
        first = SQLiteResultStore(str(db))
        first.close()
        second = SQLiteResultStore(str(db))
        try:
            assert second.get_plan_version(1) is not None
            assert _stored_version(db, "result_store_schema_version") == 2
        finally:
            second.close()


# ---------------------------------------------------------------------------
# 迁移失败回滚
# ---------------------------------------------------------------------------

class TestMigrationRollback:
    def test_failed_step_rolls_back_schema_and_version(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """迁移步骤抛错：该步事务回滚，schema/数据/版本号保持原状。"""
        db = tmp_path / "rollback.db"
        _make_v1_result_db(db)
        before_sha = _file_sha256(db)

        def _boom(conn: sqlite3.Connection) -> None:
            raise RuntimeError("注入的迁移失败")

        monkeypatch.setattr(result_store_mod, "_migrate_v1_to_v2", _boom)
        with pytest.raises(RuntimeError, match="注入的迁移失败"):
            SQLiteResultStore(str(db))
        monkeypatch.undo()

        # schema 未变：无新列、无版本表（版本表创建被一并回滚）
        conn = sqlite3.connect(str(db))
        try:
            assert "fingerprints_json" not in table_columns(conn, "plan_versions")
            assert not table_exists(conn, SCHEMA_META_TABLE)
        finally:
            conn.close()
        # 数据完好
        assert _stored_version(db, "result_store_schema_version") is None

    def test_store_usable_after_failed_migration_recovers(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """迁移失败不留下损坏状态：恢复后重新打开即可正常完成迁移。"""
        db = tmp_path / "recover.db"
        _make_v1_result_db(db)

        def _boom(conn: sqlite3.Connection) -> None:
            raise RuntimeError("注入的迁移失败")

        monkeypatch.setattr(result_store_mod, "_migrate_v1_to_v2", _boom)
        with pytest.raises(RuntimeError):
            SQLiteResultStore(str(db))
        monkeypatch.undo()

        store = SQLiteResultStore(str(db))
        try:
            assert _stored_version(db, "result_store_schema_version") == 2
            assert store.get_plan_version(1) is not None
        finally:
            store.close()

    def test_unit_transaction_rollback_semantics(self, tmp_path: Path) -> None:
        """run_in_transaction：异常时 DDL 一并回滚。"""
        db = tmp_path / "unit.db"
        conn = sqlite3.connect(str(db))
        try:

            def _action(c: sqlite3.Connection) -> None:
                c.execute("CREATE TABLE should_vanish (id INTEGER)")
                raise ValueError("回滚触发")

            with pytest.raises(ValueError, match="回滚触发"):
                run_in_transaction(conn, _action)
            assert not table_exists(conn, "should_vanish")

            run_in_transaction(
                conn, lambda c: c.execute("CREATE TABLE stays (id INTEGER)")
            )
            assert table_exists(conn, "stays")
        finally:
            conn.close()


# ---------------------------------------------------------------------------
# 未知高版本 fail closed（零写入）
# ---------------------------------------------------------------------------

class TestFailClosedOnHigherVersion:
    def test_result_store_rejects_higher_version(self, tmp_path: Path) -> None:
        db = tmp_path / "future.db"
        SQLiteResultStore(str(db)).close()
        _set_stored_version(db, "result_store_schema_version", "99")
        before_sha = _file_sha256(db)

        with pytest.raises(SchemaVersionError, match="fail closed"):
            SQLiteResultStore(str(db))

        # 零写入：拒绝过程不改动数据库文件（字节级一致）
        assert _file_sha256(db) == before_sha
        # 版本记录未被降级覆盖
        assert _stored_version(db, "result_store_schema_version") == 99

    def test_task_store_rejects_higher_version(self, tmp_path: Path) -> None:
        db = tmp_path / "future.db"
        TaskStore(db).close()
        _set_stored_version(db, "task_store_schema_version", "2")
        before_sha = _file_sha256(db)

        with pytest.raises(SchemaVersionError, match="fail closed"):
            TaskStore(db)

        assert _file_sha256(db) == before_sha
        assert _stored_version(db, "task_store_schema_version") == 2

    def test_illegal_version_record_fails_closed(self, tmp_path: Path) -> None:
        db = tmp_path / "corrupt.db"
        SQLiteResultStore(str(db)).close()
        _set_stored_version(db, "result_store_schema_version", "not-a-number")

        with pytest.raises(SchemaVersionError, match="非法"):
            SQLiteResultStore(str(db))


# ---------------------------------------------------------------------------
# ensure_schema 单元语义
# ---------------------------------------------------------------------------

class TestEnsureSchemaUnit:
    def test_step_chain_validation_rejects_gap(self) -> None:
        steps = (
            MigrationStep(1, 2, "a", lambda conn: None),
            MigrationStep(1, 2, "断链", lambda conn: None),
        )
        with pytest.raises(ValueError, match="断链"):
            ensure_schema(
                sqlite3.connect(":memory:"),
                version_key="k",
                baseline_statements=(),
                baseline_version=2,
                steps=steps,
                detect_legacy_version=lambda conn: None,
            )

    def test_step_chain_validation_rejects_non_consecutive(self) -> None:
        steps = (MigrationStep(1, 3, "跳版本", lambda conn: None),)
        with pytest.raises(ValueError, match="不连续"):
            ensure_schema(
                sqlite3.connect(":memory:"),
                version_key="k",
                baseline_statements=(),
                baseline_version=3,
                steps=steps,
                detect_legacy_version=lambda conn: None,
            )

    def test_missing_step_for_stored_version_fails_closed(self, tmp_path: Path) -> None:
        """库版本低于链首且无覆盖步骤：fail closed 而非静默重建。"""
        db = tmp_path / "too_old.db"
        conn = sqlite3.connect(str(db))
        try:
            conn.execute("CREATE TABLE legacy_stuff (id INTEGER)")
            conn.commit()
            with pytest.raises(SchemaVersionError, match="拒绝静默处理"):
                ensure_schema(
                    conn,
                    version_key="k",
                    baseline_statements=(
                        "CREATE TABLE IF NOT EXISTS t (id INTEGER)",
                    ),
                    baseline_version=2,
                    steps=(),  # 空链无法从任何更低版本推进
                    detect_legacy_version=lambda c: 0,  # 探测出未知低版本
                )
        finally:
            conn.close()


# ---------------------------------------------------------------------------
# 版本读取与 row_factory 无关（任务 56：裸连接兼容）
# ---------------------------------------------------------------------------


class TestRowFactoryAgnosticVersionRead:
    """经 ``ensure_schema`` 公开路径验证版本读取不依赖 ``row_factory``：

    - 裸连接（默认 tuple row）可读取合法版本并完成 1 → 2 迁移；
    - 非法版本值在 tuple row 与 ``sqlite3.Row`` 两种连接下统一 fail
      closed 为 SchemaVersionError（不泄漏 TypeError），且零写入。
    """

    @staticmethod
    def _ensure(conn: sqlite3.Connection, *, baseline_version: int) -> int:
        steps = (
            MigrationStep(
                1,
                2,
                "加列",
                lambda c: c.execute(
                    "ALTER TABLE t ADD COLUMN extra TEXT DEFAULT ''"
                ),
            ),
        ) if baseline_version == 2 else ()
        return ensure_schema(
            conn,
            version_key="k",
            baseline_statements=("CREATE TABLE IF NOT EXISTS t (id INTEGER)",),
            baseline_version=baseline_version,
            steps=steps,
            detect_legacy_version=lambda c: None,
        )

    def test_bare_connection_reads_legal_version_and_migrates(
        self, tmp_path: Path
    ) -> None:
        """裸连接（默认 tuple row）读出合法版本 1 并推进 1 → 2 迁移。"""
        db = tmp_path / "bare.db"
        conn = sqlite3.connect(str(db))  # 不设 row_factory：默认 tuple row
        try:
            assert self._ensure(conn, baseline_version=1) == 1
        finally:
            conn.close()

        # 裸连接读取合法版本 1 → 执行 1 → 2 增量迁移
        conn = sqlite3.connect(str(db))
        try:
            assert self._ensure(conn, baseline_version=2) == 2
            assert "extra" in table_columns(conn, "t")
        finally:
            conn.close()

        # 再次裸连接打开：版本已处于基线，走幂等快路径原样返回
        conn = sqlite3.connect(str(db))
        try:
            assert self._ensure(conn, baseline_version=2) == 2
        finally:
            conn.close()
        assert _stored_version(db, "k") == 2

    @pytest.mark.parametrize(
        "row_factory",
        [None, sqlite3.Row],
        ids=["tuple-row-default", "sqlite3-row"],
    )
    def test_illegal_version_fails_closed_both_row_kinds(
        self, tmp_path: Path, row_factory
    ) -> None:
        """非法版本值：tuple row 与 Row 连接都抛 SchemaVersionError 且零写入。"""
        db = tmp_path / "corrupt.db"
        conn = sqlite3.connect(str(db))
        try:
            self._ensure(conn, baseline_version=1)
        finally:
            conn.close()
        _set_stored_version(db, "k", "not-a-number")
        before_sha = _file_sha256(db)

        conn = sqlite3.connect(str(db))
        try:
            if row_factory is not None:
                conn.row_factory = row_factory
            with pytest.raises(SchemaVersionError, match="非法"):
                self._ensure(conn, baseline_version=2)
        finally:
            conn.close()

        # 零写入：拒绝过程文件字节不变，非法版本记录原样保留
        assert _file_sha256(db) == before_sha
        assert _raw_stored_version(db, "k") == "not-a-number"

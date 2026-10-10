"""执行结果存储 — Protocol 接口 + InMemory / SQLite 两种实现。

- ResultStore：存储契约（save/get 执行记录与计划版本快照）；
- InMemoryResultStore：内存实现，用于测试和短期任务；
- SQLiteResultStore：持久化实现，跨任务复用既有成功结果。

get_latest_success 语义统一为：返回该指纹下最近一条
status=completed 且 result.success=True 的执行记录。
"""

from __future__ import annotations

import json
import os
import sqlite3
from typing import Any, Protocol

from so_agent.models import (
    Execution,
    ExecutionResult,
    PlanVersion,
    ReviewDecision,
    SubtaskSpec,
    TaskDecompositionIssue,
)
from so_agent.schema_migrations import (
    MigrationStep,
    ensure_schema,
    table_columns,
    table_exists,
)

__all__ = ["ResultStore", "InMemoryResultStore", "SQLiteResultStore"]

# 任务 44：ResultStore schema 基线版本。
# v1 = executions + plan_versions（无 fingerprints_json 列）——历史初始形态；
# v2 = plan_versions 补 fingerprints_json 列——原实现中的隐式迁移
#     （PRAGMA table_info 检查 + ALTER TABLE ADD COLUMN）已正式化为
#     版本化步骤 1→2；
# 后续变更在 ``_result_store_steps`` 顺序追加，禁止再做隐式结构变更。
RESULT_STORE_BASELINE_VERSION: int = 2

# 全新空库一次性建到基线版本（v2）的幂等 DDL。
_RESULT_STORE_BASELINE_STATEMENTS: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS executions (
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
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_exec_fingerprint
        ON executions(input_fingerprint)
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_exec_subtask
        ON executions(subtask_id, plan_version)
    """,
    """
    CREATE TABLE IF NOT EXISTS plan_versions (
        version INTEGER PRIMARY KEY,
        specs_json TEXT NOT NULL,
        fingerprints_json TEXT DEFAULT '{}',
        executions_json TEXT DEFAULT '{}',
        review_json TEXT,
        trigger_json TEXT,
        created_at TEXT DEFAULT ''
    )
    """,
)


def _migrate_v1_to_v2(conn: Any) -> None:
    """v1 → v2：plan_versions 补 fingerprints_json 列。

    正式化原先的隐式迁移；只增列、不改写既有数据（旧行取列默认
    '{}'，与旧行为一致）。SQLite 的 ALTER TABLE ADD COLUMN 事务化，
    与版本号写入同生共死。
    """
    conn.execute(
        "ALTER TABLE plan_versions "
        "ADD COLUMN fingerprints_json TEXT DEFAULT '{}'"
    )


def _result_store_steps() -> tuple[MigrationStep, ...]:
    """ResultStore 增量迁移步骤表（函数形式便于测试 monkeypatch）。"""
    return (
        MigrationStep(
            from_version=1,
            to_version=2,
            description="plan_versions 补 fingerprints_json 列",
            upgrade=_migrate_v1_to_v2,
        ),
    )


def _detect_result_store_legacy_version(conn: Any) -> int | None:
    """探测无版本表的旧库版本（只读）。

    - plan_versions 表不存在 → 空库/残缺库，返回 None（走基线建表，
      CREATE TABLE IF NOT EXISTS 对残缺库幂等补齐）；
    - 已含 fingerprints_json 列（旧代码隐式迁移过）→ 视为 v2，无需迁移；
    - 缺该列 → 视为 v1，需执行 1→2 增量迁移。
    """
    if not table_exists(conn, "plan_versions"):
        return None
    columns = table_columns(conn, "plan_versions")
    return 2 if "fingerprints_json" in columns else 1


class ResultStore(Protocol):
    """执行结果存储接口。"""

    def save_execution(self, execution: Execution) -> None: ...

    def get_executions(self, subtask_id: str, plan_version: int) -> list[Execution]: ...

    def get_latest_success(self, fingerprint: str) -> Execution | None: ...

    def save_plan_version(self, pv: PlanVersion) -> None: ...

    def get_plan_version(self, version: int) -> PlanVersion | None: ...


class InMemoryResultStore:
    """内存实现 — 用于测试和短期任务。"""

    def __init__(self) -> None:
        self._executions: list[Execution] = []
        self._plan_versions: dict[int, PlanVersion] = {}

    def save_execution(self, execution: Execution) -> None:
        self._executions.append(execution)

    def get_executions(self, subtask_id: str, plan_version: int) -> list[Execution]:
        return [
            e for e in self._executions
            if e.subtask_id == subtask_id and e.plan_version == plan_version
        ]

    def get_latest_success(self, fingerprint: str) -> Execution | None:
        candidates = [
            e for e in self._executions
            if e.input_fingerprint == fingerprint
            and e.status == "completed"
            and e.result is not None
            and e.result.success
        ]
        if not candidates:
            return None
        return candidates[-1]  # 最近的成功记录

    def save_plan_version(self, pv: PlanVersion) -> None:
        self._plan_versions[pv.version] = pv

    def get_plan_version(self, version: int) -> PlanVersion | None:
        return self._plan_versions.get(version)


class SQLiteResultStore:
    """SQLite 持久化实现 — 扩展现有 persistence.py 的持久化能力。"""

    def __init__(self, db_path: str = "sandbox/result_store.db"):
        parent = os.path.dirname(db_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        self._conn = sqlite3.connect(db_path)
        self._conn.row_factory = sqlite3.Row
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        """建表与 schema 版本推进（幂等；任务 44 版本化迁移）。

        - 空库：单事务建到 v2 并写入版本号；
        - v1 旧库（plan_versions 无 fingerprints_json）：执行 1→2
          增量迁移（只增列，数据保留）；
        - 旧代码已隐式补列的库：登记为 v2，不重复迁移；
        - 高版本库：抛 :class:`SchemaVersionError`（fail closed）。
        """
        ensure_schema(
            self._conn,
            version_key="result_store_schema_version",
            baseline_statements=_RESULT_STORE_BASELINE_STATEMENTS,
            baseline_version=RESULT_STORE_BASELINE_VERSION,
            steps=_result_store_steps(),
            detect_legacy_version=_detect_result_store_legacy_version,
        )

    def save_execution(self, execution: Execution) -> None:
        self._conn.execute(
            """INSERT OR REPLACE INTO executions
               (execution_id, task_id, subtask_id, plan_version, agent_id, attempt,
                input_fingerprint, output_fingerprint, status, result_json,
                token_usage, duration, failure_type, recovery_action, started_at, completed_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (execution.execution_id, execution.task_id, execution.subtask_id,
             execution.plan_version, execution.agent_id, execution.attempt,
             execution.input_fingerprint, execution.output_fingerprint, execution.status,
             execution.result.model_dump_json() if execution.result else None,
             execution.token_usage, execution.duration, execution.failure_type,
             execution.recovery_action, execution.started_at, execution.completed_at)
        )
        self._conn.commit()

    def get_executions(self, subtask_id: str, plan_version: int) -> list[Execution]:
        rows = self._conn.execute(
            "SELECT * FROM executions WHERE subtask_id = ? AND plan_version = ?",
            (subtask_id, plan_version)
        ).fetchall()
        return [self._row_to_execution(r) for r in rows]

    def get_latest_success(self, fingerprint: str) -> Execution | None:
        rows = self._conn.execute(
            """SELECT * FROM executions
               WHERE input_fingerprint = ? AND status = 'completed'
               ORDER BY rowid DESC""",
            (fingerprint,)
        ).fetchall()
        for row in rows:  # 从最近往回找第一条 success=True 的记录
            exec_ = self._row_to_execution(row)
            if exec_.result is not None and exec_.result.success:
                return exec_
        return None

    def save_plan_version(self, pv: PlanVersion) -> None:
        self._conn.execute(
            """INSERT OR REPLACE INTO plan_versions
               (version, specs_json, fingerprints_json, executions_json,
                review_json, trigger_json, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (pv.version,
             json.dumps([s.model_dump(mode="json") for s in pv.specs], ensure_ascii=False),
             json.dumps(pv.fingerprints, ensure_ascii=False),
             json.dumps(
                 {k: [e.model_dump(mode="json") for e in v] for k, v in pv.executions.items()},
                 ensure_ascii=False,
             ),
             pv.review_decision.model_dump_json() if pv.review_decision else None,
             pv.trigger_issue.model_dump_json() if pv.trigger_issue else None,
             pv.created_at)
        )
        self._conn.commit()

    def get_plan_version(self, version: int) -> PlanVersion | None:
        row = self._conn.execute(
            "SELECT * FROM plan_versions WHERE version = ?", (version,)
        ).fetchone()
        if not row:
            return None
        specs = [SubtaskSpec.model_validate(s) for s in json.loads(row["specs_json"])]
        fingerprints_raw = row["fingerprints_json"]
        fingerprints = json.loads(fingerprints_raw) if fingerprints_raw else {}
        execs_raw = json.loads(row["executions_json"])
        execs = {
            k: [Execution.model_validate(e) for e in v]
            for k, v in execs_raw.items()
        }
        review = (
            ReviewDecision.model_validate_json(row["review_json"])
            if row["review_json"] else None
        )
        trigger = (
            TaskDecompositionIssue.model_validate_json(row["trigger_json"])
            if row["trigger_json"] else None
        )
        return PlanVersion(
            version=row["version"],
            specs=specs,
            fingerprints=fingerprints,
            executions=execs,
            review_decision=review,
            trigger_issue=trigger,
            created_at=row["created_at"] or "",
        )

    def _row_to_execution(self, row: sqlite3.Row) -> Execution:
        result = None
        if row["result_json"]:
            result = ExecutionResult.model_validate_json(row["result_json"])
        return Execution(
            execution_id=row["execution_id"],
            task_id=row["task_id"],
            subtask_id=row["subtask_id"],
            plan_version=row["plan_version"],
            agent_id=row["agent_id"],
            attempt=row["attempt"],
            input_fingerprint=row["input_fingerprint"],
            output_fingerprint=row["output_fingerprint"],
            status=row["status"],
            result=result,
            token_usage=row["token_usage"],
            duration=row["duration"],
            failure_type=row["failure_type"],
            recovery_action=row["recovery_action"],
            started_at=row["started_at"],
            completed_at=row["completed_at"],
        )

    def close(self) -> None:
        self._conn.close()

"""任务状态持久化（SQLite）。

使用 SQLite 存储长任务的状态、调度规格和检查点，支持：
- 任务创建/查询/更新
- 到期任务查询（用于调度器轮询）
- 检查点保存/恢复（断点续跑）
- 任务执行日志

设计要点：
- 仅依赖标准库 ``sqlite3``，无需额外第三方依赖；
- 所有时间戳统一为 UTC ISO-8601 字符串（字典序与时间序一致，
  便于 ``next_run_at <= now`` 的字符串比较过滤）；
- ``TaskRecord`` 为面向调用方的 Pydantic 视图，落库时
  ``ScheduleSpec`` 序列化为 JSON 文本，读出时反序列化还原。
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from so_agent.models import ScheduleSpec, TaskType
from so_agent.schema_migrations import (
    MigrationStep,
    ensure_schema,
    table_exists,
)

# 任务 44：TaskStore schema 基线版本。v1 = tasks + task_logs + 索引；
# 后续 schema 变更在 ``_task_store_steps`` 登记为顺序增量步骤，
# 禁止在旧表上做隐式结构变更（必须经版本化迁移，兼容既有数据）。
TASK_STORE_BASELINE_VERSION: int = 1

# 全新空库一次性建到基线版本的幂等 DDL（与 v1 建表行为逐条一致）。
_TASK_STORE_BASELINE_STATEMENTS: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS tasks (
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
    """,
    """
    CREATE TABLE IF NOT EXISTS task_logs (
        log_id INTEGER PRIMARY KEY AUTOINCREMENT,
        task_id TEXT NOT NULL REFERENCES tasks(task_id),
        event_type TEXT NOT NULL,
        payload TEXT,
        created_at TEXT NOT NULL
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_tasks_next_run ON tasks(next_run_at)
        WHERE status IN ('scheduled', 'paused')
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_task_logs_task ON task_logs(task_id)
    """,
)


def _task_store_steps() -> tuple[MigrationStep, ...]:
    """TaskStore 增量迁移步骤表（函数形式便于测试 monkeypatch）。

    当前基线即 v1（无历史变更）；后续变更追加 MigrationStep(1→2, …)。
    """
    return ()


def _detect_task_store_legacy_version(conn: Any) -> int | None:
    """探测无版本表的旧库版本（只读）：tasks 表存在即视为 v1。"""
    return 1 if table_exists(conn, "tasks") else None


class TaskRecord(BaseModel):
    """持久化任务记录。"""

    task_id: str
    objective: str
    constraints: str = ""
    acceptance_criteria: str = ""
    task_type: TaskType
    status: str = "pending"  # pending/scheduled/running/paused/completed/failed
    schedule_spec: ScheduleSpec | None = None
    checkpoint: dict[str, Any] | None = None
    result: str | None = None  # JSON 化的结果
    created_at: str = ""
    updated_at: str = ""
    next_run_at: str | None = None


class TaskStore:
    """任务状态持久化（SQLite）。"""

    def __init__(self, db_path: Path | str):
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path))
        self._conn.row_factory = sqlite3.Row
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        """建表与 schema 版本推进（幂等；任务 44 版本化迁移）。

        - 空库：单事务执行基线 DDL 并写入版本号；
        - 旧库（无版本表）：探测为 v1 后原样登记版本号（结构与 v1
          基线一致，无需实际迁移）；
        - 高版本库：抛 :class:`SchemaVersionError`（fail closed）。

        兼容承诺：``tasks`` / ``task_logs`` 既有数据只增列、不改写。
        """
        ensure_schema(
            self._conn,
            version_key="task_store_schema_version",
            baseline_statements=_TASK_STORE_BASELINE_STATEMENTS,
            baseline_version=TASK_STORE_BASELINE_VERSION,
            steps=_task_store_steps(),
            detect_legacy_version=_detect_task_store_legacy_version,
        )

    def _now_iso(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def create_task(
        self,
        objective: str,
        task_type: TaskType,
        constraints: str = "",
        acceptance_criteria: str = "",
        schedule_spec: ScheduleSpec | None = None,
        status: str = "pending",
        next_run_at: str | None = None,
    ) -> TaskRecord:
        """创建并持久化一个新任务。"""
        now = self._now_iso()
        task_id = uuid.uuid4().hex
        spec_json = schedule_spec.model_dump_json() if schedule_spec else None
        self._conn.execute(
            """INSERT INTO tasks
               (task_id, objective, constraints, acceptance_criteria,
                task_type, status, schedule_spec, created_at, updated_at, next_run_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                task_id,
                objective,
                constraints,
                acceptance_criteria,
                task_type.value,
                status,
                spec_json,
                now,
                now,
                next_run_at,
            ),
        )
        self._conn.commit()
        return TaskRecord(
            task_id=task_id,
            objective=objective,
            constraints=constraints,
            acceptance_criteria=acceptance_criteria,
            task_type=task_type,
            status=status,
            schedule_spec=schedule_spec,
            created_at=now,
            updated_at=now,
            next_run_at=next_run_at,
        )

    def get_task(self, task_id: str) -> TaskRecord | None:
        """按 ID 查询任务。"""
        row = self._conn.execute(
            "SELECT * FROM tasks WHERE task_id = ?", (task_id,)
        ).fetchone()
        if not row:
            return None
        return self._row_to_record(row)

    def get_due_tasks(self) -> list[TaskRecord]:
        """查询所有到期待执行的任务。"""
        now = self._now_iso()
        rows = self._conn.execute(
            """SELECT * FROM tasks
               WHERE status IN ('scheduled', 'paused')
                 AND next_run_at IS NOT NULL
                 AND next_run_at <= ?
               ORDER BY next_run_at""",
            (now,),
        ).fetchall()
        return [self._row_to_record(r) for r in rows]

    def update_status(
        self,
        task_id: str,
        status: str,
        *,
        result: str | None = None,
        error: str | None = None,
    ) -> None:
        """更新任务状态。

        Args:
            task_id: 任务编号。
            status: 目标状态。
            result: 任务结果文本（可选；None 表示不更新 result 字段）。
            error: 失败信息（可选；非空时追加一条 error 事件日志）。
        """
        now = self._now_iso()
        if result is not None:
            self._conn.execute(
                "UPDATE tasks SET status=?, result=?, updated_at=? WHERE task_id=?",
                (status, result, now, task_id),
            )
        else:
            self._conn.execute(
                "UPDATE tasks SET status=?, updated_at=? WHERE task_id=?",
                (status, now, task_id),
            )
        if error:
            self.add_log(task_id, "error", {"message": error})
        self._conn.commit()

    def save_checkpoint(self, task_id: str, checkpoint: dict[str, Any]) -> None:
        """保存检查点（断点恢复用）。"""
        now = self._now_iso()
        self._conn.execute(
            "UPDATE tasks SET checkpoint=?, updated_at=? WHERE task_id=?",
            (json.dumps(checkpoint, ensure_ascii=False), now, task_id),
        )
        self._conn.commit()

    def load_checkpoint(self, task_id: str) -> dict[str, Any] | None:
        """加载检查点。"""
        row = self._conn.execute(
            "SELECT checkpoint FROM tasks WHERE task_id=?", (task_id,)
        ).fetchone()
        if row and row["checkpoint"]:
            return json.loads(row["checkpoint"])
        return None

    def reschedule(self, task_id: str, next_run_at: str) -> None:
        """重新调度周期任务的下次执行时间。"""
        now = self._now_iso()
        self._conn.execute(
            "UPDATE tasks SET status='scheduled', next_run_at=?, updated_at=? "
            "WHERE task_id=?",
            (next_run_at, now, task_id),
        )
        self._conn.commit()

    def add_log(self, task_id: str, event_type: str, payload: Any = None) -> None:
        """记录任务事件日志。"""
        self._conn.execute(
            "INSERT INTO task_logs (task_id, event_type, payload, created_at) "
            "VALUES (?, ?, ?, ?)",
            (
                task_id,
                event_type,
                json.dumps(payload, ensure_ascii=False) if payload else None,
                self._now_iso(),
            ),
        )
        self._conn.commit()

    def list_tasks(self, status: str | None = None) -> list[TaskRecord]:
        """列出任务（可按状态过滤）。"""
        if status:
            rows = self._conn.execute(
                "SELECT * FROM tasks WHERE status=? ORDER BY created_at DESC",
                (status,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM tasks ORDER BY created_at DESC"
            ).fetchall()
        return [self._row_to_record(r) for r in rows]

    def _row_to_record(self, row: sqlite3.Row) -> TaskRecord:
        d = dict(row)
        d["task_type"] = TaskType(d["task_type"])
        if d.get("schedule_spec"):
            d["schedule_spec"] = ScheduleSpec.model_validate_json(d["schedule_spec"])
        if d.get("checkpoint"):
            d["checkpoint"] = json.loads(d["checkpoint"])
        return TaskRecord(**d)

    def close(self) -> None:
        """关闭数据库连接。"""
        self._conn.close()

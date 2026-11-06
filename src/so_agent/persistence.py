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
import shutil
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from so_agent.config import Settings
from so_agent.context import ProjectContext
from so_agent.models import ScheduleSpec, TaskType
from so_agent.sqlite_tuning import configure_sqlite
from so_agent.schema_migrations import (
    MigrationStep,
    ensure_schema,
    table_columns,
    table_exists,
)

# 任务 44：TaskStore schema 基线版本。v1 = tasks + task_logs + 索引；
# v2 = tasks 增量列（last_run_at / consecutive_failures）+ task_runs 运行历史。
# 后续 schema 变更在 ``_task_store_steps`` 登记为顺序增量步骤，
# 禁止在旧表上做隐式结构变更（必须经版本化迁移，兼容既有数据）。
TASK_STORE_BASELINE_VERSION: int = 2

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
        next_run_at TEXT,
        last_run_at TEXT,
        consecutive_failures INTEGER NOT NULL DEFAULT 0
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
    CREATE TABLE IF NOT EXISTS task_runs (
        run_id TEXT PRIMARY KEY,
        task_id TEXT NOT NULL REFERENCES tasks(task_id),
        status TEXT NOT NULL,
        trigger_at TEXT NOT NULL,
        started_at TEXT NOT NULL,
        finished_at TEXT,
        next_run_at TEXT,
        error TEXT,
        result TEXT
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_tasks_next_run ON tasks(next_run_at)
        WHERE status IN ('scheduled', 'paused')
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_task_logs_task ON task_logs(task_id)
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_task_runs_task
        ON task_runs(task_id, started_at DESC)
    """,
)


def _migrate_task_store_v1_to_v2(conn: sqlite3.Connection) -> None:
    """v1 → v2：补齐运行状态列并创建 task_runs（仅增列/增表，不改写数据）。"""
    columns = table_columns(conn, "tasks")
    if "last_run_at" not in columns:
        conn.execute("ALTER TABLE tasks ADD COLUMN last_run_at TEXT")
    if "consecutive_failures" not in columns:
        conn.execute(
            "ALTER TABLE tasks ADD COLUMN consecutive_failures "
            "INTEGER NOT NULL DEFAULT 0"
        )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS task_runs (
            run_id TEXT PRIMARY KEY,
            task_id TEXT NOT NULL REFERENCES tasks(task_id),
            status TEXT NOT NULL,
            trigger_at TEXT NOT NULL,
            started_at TEXT NOT NULL,
            finished_at TEXT,
            next_run_at TEXT,
            error TEXT,
            result TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_task_runs_task
            ON task_runs(task_id, started_at DESC)
        """
    )


def _task_store_steps() -> tuple[MigrationStep, ...]:
    """TaskStore 增量迁移步骤表（函数形式便于测试 monkeypatch）。"""
    return (
        MigrationStep(
            from_version=1,
            to_version=2,
            description="调度运行历史：tasks 运行元数据 + task_runs 表",
            upgrade=_migrate_task_store_v1_to_v2,
        ),
    )


def _detect_task_store_legacy_version(conn: Any) -> int | None:
    """探测无版本表的旧库版本（只读）：tasks 表存在即视为 v1。"""
    return 1 if table_exists(conn, "tasks") else None


def resolve_task_store_path(context: ProjectContext, settings: Settings) -> Path:
    """解析任务库绝对路径，并迁移旧版 ``<sandbox>/sandbox/`` 下的数据。

    旧默认值为 ``sandbox/task_store.db``，叠加 ``sandbox_dir`` 后落盘到
    ``<sandbox>/sandbox/task_store.db``（双重 sandbox 缺陷）。新默认值为
    ``task_store.db``；若新位置尚无数据库而旧位置存在，则复制一份
    （保留旧文件，不破坏既有数据）。

    ``context.task_store_dir`` 非空时以它为根（Web 用它把所有任务的持久化
    任务收敛到同一个共享库），此时不执行旧目录迁移。
    """
    raw = Path(settings.task_store_path).expanduser()
    base = context.task_store_dir or context.sandbox_dir
    target = raw if raw.is_absolute() else base / raw
    if context.task_store_dir is not None:
        return target
    if raw.parts and raw.parts[0] == "sandbox":
        return target
    legacy = context.sandbox_dir / "sandbox" / raw.name
    if not target.exists() and legacy.is_file():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(legacy, target)
    return target


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
    last_run_at: str | None = None
    consecutive_failures: int = 0


class TaskStore:
    """任务状态持久化（SQLite）。"""

    def __init__(self, db_path: Path | str):
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path), timeout=10)
        self._conn.row_factory = sqlite3.Row
        configure_sqlite(self._conn)
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
        """查询所有到期待执行的任务。

        仅返回 ``scheduled``：``paused`` 表示执行失败或人工暂停，必须由
        人工恢复（reschedule）后才会再次到期，避免轮询风暴。
        """
        now = self._now_iso()
        rows = self._conn.execute(
            """SELECT * FROM tasks
               WHERE status = 'scheduled'
                 AND next_run_at IS NOT NULL
                 AND next_run_at <= ?
               ORDER BY next_run_at""",
            (now,),
        ).fetchall()
        return [self._row_to_record(r) for r in rows]

    def claim(self, task_id: str) -> bool:
        """原子地把到期任务置为 running；返回是否抢占成功。

        基于 ``UPDATE ... WHERE status='scheduled'`` 的条件更新，多个调度
        进程共享同一数据库时只有一个能抢到，避免重复执行。
        """
        cursor = self._conn.execute(
            "UPDATE tasks SET status='running', updated_at=? "
            "WHERE task_id=? AND status='scheduled'",
            (self._now_iso(), task_id),
        )
        self._conn.commit()
        return cursor.rowcount == 1

    def pause(self, task_id: str, *, error: str | None = None) -> None:
        """暂停任务并清空下次执行时间（需人工恢复）。"""
        self._conn.execute(
            "UPDATE tasks SET status='paused', next_run_at=NULL, updated_at=? "
            "WHERE task_id=?",
            (self._now_iso(), task_id),
        )
        if error:
            self.add_log(task_id, "error", {"message": error})
        self._conn.commit()

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

    def start_run(
        self, task_id: str, scheduled_for: str, *, now: str | None = None
    ) -> str:
        """登记一次调度执行（status=running），返回 run_id。"""
        run_id = uuid.uuid4().hex
        started = now or self._now_iso()
        self._conn.execute(
            """INSERT INTO task_runs
               (run_id, task_id, status, trigger_at, started_at)
               VALUES (?, ?, 'running', ?, ?)""",
            (run_id, task_id, scheduled_for, started),
        )
        self._conn.commit()
        return run_id

    def finish_run(
        self,
        run_id: str,
        status: str,
        *,
        result: str | None = None,
        error: str | None = None,
        next_run_at: str | None = None,
        result_limit: int = 4000,
    ) -> None:
        """收口一次运行记录（result 截断，避免历史表无限膨胀）。"""
        excerpt = result[:result_limit] if result else result
        self._conn.execute(
            """UPDATE task_runs
               SET status=?, finished_at=?, result=?, error=?, next_run_at=?
               WHERE run_id=?""",
            (status, self._now_iso(), excerpt, error, next_run_at, run_id),
        )
        self._conn.commit()

    def record_skipped_run(
        self,
        task_id: str,
        scheduled_for: str,
        reason: str,
        *,
        next_run_at: str | None = None,
    ) -> str:
        """记录一次因错过触发窗口而被跳过的运行（不计入执行次数）。"""
        return self.start_run_and_finish(
            task_id,
            scheduled_for,
            "skipped",
            error=reason,
            next_run_at=next_run_at,
        )

    def start_run_and_finish(
        self,
        task_id: str,
        scheduled_for: str,
        status: str,
        *,
        result: str | None = None,
        error: str | None = None,
        next_run_at: str | None = None,
    ) -> str:
        """创建并立即收口一条运行记录（用于跳过等瞬时事件）。"""
        now = self._now_iso()
        run_id = uuid.uuid4().hex
        excerpt = result[:4000] if result else result
        self._conn.execute(
            """INSERT INTO task_runs
               (run_id, task_id, status, trigger_at, started_at, finished_at,
                next_run_at, error, result)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                run_id,
                task_id,
                status,
                scheduled_for,
                now,
                now,
                next_run_at,
                error,
                excerpt,
            ),
        )
        self._conn.commit()
        return run_id

    def record_success(self, task_id: str, *, at: str | None = None) -> None:
        """标记一次成功执行：刷新 last_run_at 并清零连续失败计数。"""
        moment = at or self._now_iso()
        self._conn.execute(
            "UPDATE tasks SET last_run_at=?, consecutive_failures=0, updated_at=? "
            "WHERE task_id=?",
            (moment, moment, task_id),
        )
        self._conn.commit()

    def schedule_retry(
        self, task_id: str, next_run_at: str, *, error: str | None = None
    ) -> int:
        """失败后重排：递增连续失败计数并返回新计数。"""
        now = self._now_iso()
        self._conn.execute(
            """UPDATE tasks
               SET status='scheduled', next_run_at=?,
                   consecutive_failures=consecutive_failures+1, updated_at=?
               WHERE task_id=?""",
            (next_run_at, now, task_id),
        )
        row = self._conn.execute(
            "SELECT consecutive_failures FROM tasks WHERE task_id=?", (task_id,)
        ).fetchone()
        self._conn.commit()
        return int(row["consecutive_failures"]) if row else 0

    def list_runs(self, task_id: str, limit: int = 50) -> list[dict[str, Any]]:
        """返回任务最近的运行历史（新的在前）。"""
        rows = self._conn.execute(
            """SELECT run_id, task_id, status, trigger_at, started_at,
                      finished_at, next_run_at, error, result
               FROM task_runs WHERE task_id=?
               ORDER BY started_at DESC LIMIT ?""",
            (task_id, max(1, limit)),
        ).fetchall()
        return [dict(row) for row in rows]

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

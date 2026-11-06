"""Web 任务元数据的 SQLite 存储。

与 CLI 持久化任务的 ``TaskStore`` 分离：Web 任务只保存控制台展示所需的
元数据、终态快照与 token 汇总，不参与调度。数据库位于 Web 沙箱根目录
``web_tasks.db``，任务沙箱仍是 ``<base>/<task_id>``。

设计要点：
- 单连接 + 进程内锁；写操作发生在事件循环线程，读操作可能来自线程池；
- 终态快照按语言分别保存，重启后可直接回放，不依赖已丢失的
  ``WorkflowEngine`` 运行态；
- 保留策略只清理终态记录，绝不删除仍在执行的任务。
"""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from so_agent.sqlite_tuning import configure_sqlite

_TERMINAL_STATUSES: tuple[str, ...] = ("completed", "failed", "cancelled")
_SCHEMA_VERSION = 2


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class WebTaskStore:
    """Web 任务记录持久化（SQLite）。"""

    def __init__(self, path: Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(
            self._path, timeout=10, check_same_thread=False
        )
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            configure_sqlite(self._conn)
            self._ensure_schema()

    def _ensure_schema(self) -> None:
        version = int(self._conn.execute("PRAGMA user_version").fetchone()[0])
        if version > _SCHEMA_VERSION:
            raise RuntimeError(
                f"Web 任务库版本 {version} 高于当前支持的 {_SCHEMA_VERSION}"
            )
        if version < 1:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS web_tasks (
                    task_id TEXT PRIMARY KEY,
                    objective TEXT NOT NULL,
                    constraints TEXT NOT NULL DEFAULT '',
                    acceptance_criteria TEXT NOT NULL DEFAULT '',
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    started_at TEXT,
                    finished_at TEXT,
                    task_type TEXT,
                    classification_json TEXT,
                    output_json TEXT,
                    error TEXT,
                    cancel_requested INTEGER NOT NULL DEFAULT 0,
                    last_guidance_json TEXT,
                    snapshot_zh TEXT,
                    snapshot_en TEXT,
                    tokens_json TEXT,
                    resume_count INTEGER NOT NULL DEFAULT 0
                );
                CREATE INDEX IF NOT EXISTS idx_web_tasks_created
                    ON web_tasks(created_at);
                """
            )
            version = 1
        if version < 2:
            columns = {
                str(row["name"])
                for row in self._conn.execute(
                    "PRAGMA table_info(web_tasks)"
                ).fetchall()
            }
            if "resume_count" not in columns:
                self._conn.execute(
                    "ALTER TABLE web_tasks ADD COLUMN resume_count "
                    "INTEGER NOT NULL DEFAULT 0"
                )
            version = 2
        self._conn.execute(f"PRAGMA user_version={version}")
        self._conn.commit()

    # ------------------------------------------------------------------
    # 写入
    # ------------------------------------------------------------------
    def create(self, task: Any) -> None:
        """插入新任务记录（``task`` 为 WebTask 的鸭子类型）。"""
        with self._lock:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO web_tasks (
                    task_id, objective, constraints, acceptance_criteria,
                    status, created_at, started_at, finished_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    task.task_id,
                    task.objective,
                    task.constraints,
                    task.acceptance_criteria,
                    task.status,
                    task.created_at,
                    task.started_at,
                    task.finished_at,
                ),
            )
            self._conn.commit()

    def save(self, task: Any) -> int:
        """更新运行时字段（状态、分类、输出、错误、指导），返回更新行数。"""
        classification = (
            task.classification.model_dump_json()
            if task.classification is not None
            else None
        )
        output = (
            task.output.model_dump_json() if task.output is not None else None
        )
        guidance = (
            task.last_guidance.model_dump_json()
            if task.last_guidance is not None
            else None
        )
        task_type = (
            task.classification.task_type.value
            if task.classification is not None
            else None
        )
        with self._lock:
            cursor = self._conn.execute(
                """
                UPDATE web_tasks SET
                    status = ?, started_at = ?, finished_at = ?,
                    task_type = ?, classification_json = ?, output_json = ?,
                    error = ?, cancel_requested = ?, last_guidance_json = ?
                WHERE task_id = ?
                """,
                (
                    task.status,
                    task.started_at,
                    task.finished_at,
                    task_type,
                    classification,
                    output,
                    task.error,
                    1 if task.cancel_requested else 0,
                    guidance,
                    task.task_id,
                ),
            )
            self._conn.commit()
            return int(cursor.rowcount or 0)

    def save_snapshots(
        self,
        task_id: str,
        snapshots: dict[str, dict[str, Any]],
        tokens: dict[str, Any] | None = None,
    ) -> None:
        to_json = lambda value: json.dumps(  # noqa: E731
            value, ensure_ascii=False, default=str
        )
        with self._lock:
            self._conn.execute(
                """
                UPDATE web_tasks SET snapshot_zh = ?, snapshot_en = ?,
                    tokens_json = ?
                WHERE task_id = ?
                """,
                (
                    to_json(snapshots["zh"]) if "zh" in snapshots else None,
                    to_json(snapshots["en"]) if "en" in snapshots else None,
                    to_json(tokens or {}),
                    task_id,
                ),
            )
            self._conn.commit()

    def clear_snapshots(self, task_id: str) -> None:
        with self._lock:
            self._conn.execute(
                """
                UPDATE web_tasks SET snapshot_zh = NULL, snapshot_en = NULL
                WHERE task_id = ?
                """,
                (task_id,),
            )
            self._conn.commit()

    def recover_interrupted(self, *, max_resumes: int = 1) -> list[str]:
        """把未完成任务重新排队，返回可自动恢复的 task_id。

        ``resume_count`` 记录自动重跑次数；达到上限的任务保持原状态，
        由 :meth:`mark_interrupted` 标记失败，避免崩溃任务在每次重启时
        无限重跑。旧快照与部分输出会清空，防止前端把上一次运行的中间
        状态误认为当前进度。
        """
        limit = max(0, int(max_resumes))
        if limit <= 0:
            return []
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT task_id, resume_count FROM web_tasks
                WHERE status IN ('queued', 'running')
                  AND resume_count < ?
                ORDER BY created_at ASC
                """,
                (limit,),
            ).fetchall()
            task_ids = [str(row[0]) for row in rows]
            if task_ids:
                self._conn.executemany(
                    """
                    UPDATE web_tasks SET
                        status = 'queued', started_at = NULL,
                        finished_at = NULL, cancel_requested = 0,
                        output_json = NULL, snapshot_zh = NULL,
                        snapshot_en = NULL, tokens_json = NULL,
                        resume_count = resume_count + 1,
                        error = ?
                    WHERE task_id = ?
                    """,
                    [
                        (
                            "服务重启，任务已自动重新排队"
                            f"（自动恢复 {int(row[1]) + 1}/{limit}）",
                            str(row[0]),
                        )
                        for row in rows
                    ],
                )
                self._conn.commit()
        return task_ids

    def mark_interrupted(
        self,
        *,
        running_reason: str,
        human_reason: str,
        provisional_reason: str = "",
        max_resumes: int | None = None,
    ) -> int:
        """把重启时遗留的非终态记录标记为失败，返回受影响行数。

        ``max_resumes`` 不为 ``None`` 时只处理自动恢复次数已用尽的记录，
        留给 :meth:`recover_interrupted` 重排队其余任务。

        ``completed_provisional`` 表示结果已交付但后台评审未收口；重启后
        后台任务已丢失，提供 ``provisional_reason`` 时一并标记为失败，
        避免记录永久停留在非终态而不被保留策略清理。
        """
        now = _iso_now()
        running_where = "status IN ('queued', 'running')"
        running_params: tuple[Any, ...] = ()
        if max_resumes is not None:
            running_where += " AND resume_count >= ?"
            running_params = (int(max_resumes),)
        with self._lock:
            cursor = self._conn.execute(
                f"""
                UPDATE web_tasks SET status = 'failed', finished_at = ?,
                    error = ?, cancel_requested = 0
                WHERE {running_where}
                """,
                (now, running_reason, *running_params),
            )
            human = self._conn.execute(
                """
                UPDATE web_tasks SET status = 'failed', finished_at = ?,
                    error = ?, cancel_requested = 0
                WHERE status = 'human_required'
                """,
                (now, human_reason),
            )
            provisional = 0
            if provisional_reason:
                provisional_cursor = self._conn.execute(
                    """
                    UPDATE web_tasks SET status = 'failed', finished_at = ?,
                        error = ?, cancel_requested = 0
                    WHERE status = 'completed_provisional'
                    """,
                    (now, provisional_reason),
                )
                provisional = int(provisional_cursor.rowcount or 0)
            self._conn.commit()
            return (
                int(cursor.rowcount or 0)
                + int(human.rowcount or 0)
                + provisional
            )

    # ------------------------------------------------------------------
    # 读取
    # ------------------------------------------------------------------
    def get(self, task_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM web_tasks WHERE task_id = ?", (task_id,)
            ).fetchone()
        return dict(row) if row is not None else None

    def list_recent(self, limit: int) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM web_tasks ORDER BY created_at DESC LIMIT ?",
                (max(0, int(limit)),),
            ).fetchall()
        return [dict(row) for row in rows]

    def all_ids(self) -> set[str]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT task_id FROM web_tasks"
            ).fetchall()
        return {str(row[0]) for row in rows}

    def delete(self, task_id: str) -> bool:
        """删除单条任务记录，返回是否命中。"""
        with self._lock:
            cursor = self._conn.execute(
                "DELETE FROM web_tasks WHERE task_id = ?", (task_id,)
            )
            self._conn.commit()
            return bool(cursor.rowcount and cursor.rowcount > 0)

    # ------------------------------------------------------------------
    # 保留清理
    # ------------------------------------------------------------------
    def prune(
        self, *, retention_days: int, max_records: int
    ) -> list[str]:
        """删除过期与超量的终态记录，返回被删除的 task_id。"""
        deleted: set[str] = set()
        with self._lock:
            if retention_days > 0:
                cutoff = (
                    datetime.now(timezone.utc) - timedelta(days=retention_days)
                ).isoformat(timespec="seconds")
                rows = self._conn.execute(
                    f"""
                    SELECT task_id FROM web_tasks
                    WHERE status IN ({_placeholders()})
                      AND COALESCE(finished_at, created_at) < ?
                    """,
                    (*_TERMINAL_STATUSES, cutoff),
                ).fetchall()
                deleted.update(str(row[0]) for row in rows)
            if max_records > 0:
                rows = self._conn.execute(
                    f"""
                    SELECT task_id FROM web_tasks
                    WHERE status IN ({_placeholders()})
                    ORDER BY COALESCE(finished_at, created_at) DESC
                    LIMIT -1 OFFSET ?
                    """,
                    (*_TERMINAL_STATUSES, int(max_records)),
                ).fetchall()
                deleted.update(str(row[0]) for row in rows)
            if deleted:
                self._conn.executemany(
                    "DELETE FROM web_tasks WHERE task_id = ?",
                    [(task_id,) for task_id in deleted],
                )
                self._conn.commit()
        return sorted(deleted)

    def close(self) -> None:
        with self._lock:
            self._conn.close()


def _placeholders() -> str:
    return ", ".join("?" for _ in _TERMINAL_STATUSES)

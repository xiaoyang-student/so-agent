"""SQLite 运行参数统一设置。

单用户部署下仍存在多进程访问同一库的场景（CLI 守护进程与 Web 内置
调度器共享 ``task_store.db``），因此统一启用 WAL 与 busy_timeout：
读不阻塞写、写冲突时等待而不是立即抛 ``database is locked``。

WAL 依赖本地文件系统；网络盘等不支持 WAL 的环境会回退到原有日志
模式并记录警告，不阻断启动。
"""

from __future__ import annotations

import logging
import sqlite3

logger = logging.getLogger(__name__)


def configure_sqlite(
    conn: sqlite3.Connection,
    *,
    busy_timeout_ms: int = 10_000,
    wal: bool = True,
) -> str:
    """设置 busy_timeout 与 WAL，返回实际 journal_mode。"""
    conn.execute(f"PRAGMA busy_timeout={max(0, int(busy_timeout_ms))}")
    mode = ""
    if wal:
        row = conn.execute("PRAGMA journal_mode=WAL").fetchone()
        mode = str(row[0]).lower() if row else ""
        if mode != "wal":
            logger.warning(
                "SQLite WAL 不可用（实际 journal_mode=%s），回退为默认日志模式",
                mode or "unknown",
            )
    conn.execute("PRAGMA synchronous=NORMAL")
    return mode

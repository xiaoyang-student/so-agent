"""SQLite schema 版本机制与顺序增量迁移（任务 44 基线冻结与兼容护栏）。

为 ``TaskStore``（persistence.py）与 ``SQLiteResultStore``（result_store.py）
两个 SQLite 存储提供统一的最小 schema 版本机制：

- 每个数据库文件内置 ``schema_meta`` 版本表（键值对，键为各存储的
  版本键），记录当前 schema 版本；
- schema 变更以 :class:`MigrationStep`（from_version → to_version +
  升级回调）顺序登记，打开数据库时从已存版本逐版本推进到代码当前
  版本（顺序增量迁移）；
- 每一步迁移在单个 SQLite 事务内执行（DDL 在 SQLite 中事务化，失败
  整体回滚，不产生半迁移状态）；版本号与 schema 变更同事务写入；
- 全流程幂等：重复打开不重复迁移；已迁移过的旧库（隐式迁移时代
  产物，无版本表）经探测后只补齐缺失步骤；
- fail closed：数据库版本高于代码支持版本（未知高版本）时抛出
  :class:`SchemaVersionError` 拒绝打开，绝不静默降级覆盖，且该检查
  发生在任何写操作之前（高版本库打开过程零写入）。

第三方方案评估（任务 44 补充约束的记录）：
- Alembic：绑定 SQLAlchemy 生态；本项目持久化层为纯标准库
  ``sqlite3``、无 ORM，引入 Alembic 需连带引入 SQLAlchemy，迁移脚本
  的文件系统 revision 链机制对"两个小库、每库个位数迁移步骤"的
  规模过重，集成/运维/锁定成本大于收益，拒绝；
- yoyo-migrations 等独立迁移工具：仍需第三方依赖与脚本目录布局，
  其版本表能力与本模块等价，收益不明确，拒绝；
- 结论：采用标准库内的最小版本机制（本模块），事务化 / 幂等 /
  fail closed 语义在约 200 行内精确可控，测试面完全自主。

兼容承诺：``executions``、``plan_versions``、``tasks``、``task_logs``
既有数据在迁移中只增列、不改写、不重建，历史数据保持可读。
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable, Sequence
from dataclasses import dataclass

__all__ = [
    "MigrationStep",
    "SCHEMA_META_TABLE",
    "SchemaVersionError",
    "ensure_schema",
    "run_in_transaction",
    "table_columns",
    "table_exists",
]

# 版本表表名（每个数据库文件内一张；多个存储共库时以版本键区分）
SCHEMA_META_TABLE: str = "schema_meta"


class SchemaVersionError(RuntimeError):
    """数据库 schema 版本无法安全迁移时抛出（fail closed）。

    触发场景：

    - 数据库版本高于代码支持版本（未知高版本，通常是新版代码写入后
      又被旧版代码打开）——拒绝打开，绝不静默降级覆盖；
    - 版本记录存在但值非法（非整数）；
    - 迁移注册表无法覆盖数据库当前版本（过旧或断链）。
    """


@dataclass(frozen=True)
class MigrationStep:
    """一步增量 schema 迁移。

    Attributes:
        from_version: 迁移前 schema 版本。
        to_version: 迁移后 schema 版本（必须为 from_version + 1）。
        description: 人类可读描述（写入迁移日志/审计）。
        upgrade: 升级回调；在单事务内收到连接并执行 DDL/DML。
            回调抛出异常时整个事务回滚（schema 与版本号均不变）。
    """

    from_version: int
    to_version: int
    description: str
    upgrade: Callable[[sqlite3.Connection], None]


def run_in_transaction(conn: sqlite3.Connection, action: Callable[[sqlite3.Connection], None]) -> None:
    """在单个显式事务内执行 ``action``；成功提交、异常回滚并透传。

    使用显式 ``BEGIN IMMEDIATE`` 而非 sqlite3 隐式事务，保证 DDL 与
    版本号写入同生共死。进入前先提交任何遗留的隐式活动事务（正常
    打开路径上不存在；防御性归零避免"事务嵌套"错误）。
    """
    if conn.in_transaction:
        conn.commit()
    conn.execute("BEGIN IMMEDIATE")
    try:
        action(conn)
    except BaseException:
        try:
            conn.rollback()
        except sqlite3.Error:
            pass  # 事务已被 SQLite 自动回滚时放弃二次回滚
        raise
    conn.commit()


def table_exists(conn: sqlite3.Connection, name: str) -> bool:
    """判断表是否存在（只读查询 sqlite_master）。"""
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
        (name,),
    ).fetchone()
    return row is not None


def table_columns(conn: sqlite3.Connection, name: str) -> set[str]:
    """返回表的列名集合（表不存在时为空集合）。

    按索引取列（PRAGMA table_info 第 2 列为列名），不依赖调用方
    连接的 ``row_factory`` 设置。
    """
    return {
        str(row[1])
        for row in conn.execute(f"PRAGMA table_info({name})")
    }


def _ensure_meta_table(conn: sqlite3.Connection) -> None:
    """创建版本表（幂等；仅在确认要写入的事务内调用）。"""
    conn.execute(
        f"CREATE TABLE IF NOT EXISTS {SCHEMA_META_TABLE} ("
        "key TEXT PRIMARY KEY, value TEXT NOT NULL)"
    )


def _read_stored_version(conn: sqlite3.Connection, version_key: str) -> int | None:
    """读取已存 schema 版本；无版本表或无记录时返回 None（只读）。

    按列索引取值（查询只选 ``value`` 一列），与 :func:`table_columns`
    同理不依赖调用方连接的 ``row_factory``：裸连接（默认 tuple row）
    与 ``sqlite3.Row`` 连接均可安全读取；值非法时统一转换为
    :class:`SchemaVersionError`（fail closed），不向调用方泄漏底层
    ``TypeError``。

    Raises:
        SchemaVersionError: 版本记录存在但值非法时。
    """
    if not table_exists(conn, SCHEMA_META_TABLE):
        return None
    row = conn.execute(
        f"SELECT value FROM {SCHEMA_META_TABLE} WHERE key = ?", (version_key,)
    ).fetchone()
    if row is None:
        return None
    raw_value = row[0]
    try:
        return int(raw_value)
    except (TypeError, ValueError) as exc:
        raise SchemaVersionError(
            f"schema 版本记录非法：key={version_key!r} "
            f"value={raw_value!r}（{exc}）"
        ) from exc


def _write_version(conn: sqlite3.Connection, version_key: str, version: int) -> None:
    """写入 schema 版本记录（须在事务内调用）。"""
    conn.execute(
        f"INSERT OR REPLACE INTO {SCHEMA_META_TABLE} (key, value) VALUES (?, ?)",
        (version_key, str(version)),
    )


def _validate_step_chain(
    steps: Sequence[MigrationStep], baseline_version: int
) -> None:
    """校验迁移步骤构成 1 → … → baseline_version 的连续链（配置错误即抛）。

    空步骤表仅当 baseline_version 就是唯一已知形态时合法（如 TaskStore
    当前 v1）；此时任何更低版本的库都无法推进，会在应用步骤时报错。
    """
    if not steps:
        return
    if steps[0].from_version != 1:
        raise ValueError(
            f"迁移链必须从版本 1 开始，实际起始为 {steps[0].from_version}"
        )
    for index, step in enumerate(steps):
        if step.to_version != step.from_version + 1:
            raise ValueError(
                f"迁移步骤 {index}（{step.description}）版本不连续："
                f"{step.from_version} → {step.to_version}"
            )
        if index + 1 < len(steps) and steps[index + 1].from_version != step.to_version:
            raise ValueError(
                f"迁移步骤 {index} 与 {index + 1} 之间断链："
                f"{step.to_version} != {steps[index + 1].from_version}"
            )
    if steps[-1].to_version != baseline_version:
        raise ValueError(
            f"迁移链终点 {steps[-1].to_version} 与当前基线版本 "
            f"{baseline_version} 不一致"
        )


def _apply_steps(
    conn: sqlite3.Connection,
    *,
    version_key: str,
    steps: Sequence[MigrationStep],
    from_version: int,
    baseline_version: int,
) -> int:
    """从 ``from_version`` 起逐步执行迁移直至基线版本；每步单事务。

    每步事务内容：确保版本表存在 → 执行升级回调 → 写入新版本号。
    任一步失败：该事务回滚（schema 与版本号保持原状），异常向上传播
    由调用方（存储构造函数）决定是否终止打开。
    """
    version = from_version
    while version < baseline_version:
        candidates = [s for s in steps if s.from_version == version]
        if not candidates:
            raise SchemaVersionError(
                f"数据库 schema 版本为 {version}，但迁移注册表中没有 "
                f"{version} → {version + 1} 的步骤（库过旧或注册表断链）；"
                "拒绝静默处理，请人工确认后处理"
            )
        step = candidates[0]

        def _apply(conn: sqlite3.Connection, step: MigrationStep = step) -> None:
            _ensure_meta_table(conn)
            step.upgrade(conn)
            _write_version(conn, version_key, step.to_version)

        run_in_transaction(conn, _apply)
        version = step.to_version
    return version


def ensure_schema(
    conn: sqlite3.Connection,
    *,
    version_key: str,
    baseline_statements: Sequence[str],
    baseline_version: int,
    steps: Sequence[MigrationStep],
    detect_legacy_version: Callable[[sqlite3.Connection], int | None],
) -> int:
    """确保数据库 schema 处于当前基线版本；返回最终版本号。

    处理矩阵（按顺序判定，前两类为只读检查）：

    1. 版本表已有记录 ``stored``：
       - ``stored > baseline_version`` → :class:`SchemaVersionError`
         （未知高版本，fail closed，零写入）；
       - ``stored == baseline_version`` → 原样返回（幂等快路径）；
       - ``stored < baseline_version`` → 顺序执行缺失步骤；
    2. 无版本表：``detect_legacy_version`` 探测隐式迁移时代的旧库：
       - 探测到版本 → 同上校验并补齐缺失步骤（只补差异，不重建）；
       - 探测为 None（空库）→ 单事务执行全部基线语句并写入版本号。

    Args:
        conn: SQLite 连接（调用方持有；不负责关闭）。
        version_key: 本存储在 ``schema_meta`` 中的版本键（共库多存储时
            各自独立）。
        baseline_statements: 全新空库一次性建到基线版本的 DDL 语句
            （必须全部为幂等语句，CREATE ... IF NOT EXISTS 形态）。
        baseline_version: 代码当前支持的基线 schema 版本。
        steps: 增量迁移步骤（顺序链）；当前无变更的存储可为空元组。
        detect_legacy_version: 旧库版本探测器（无版本表时调用）；返回
            None 表示空库/未知库。探测器必须只读。

    Returns:
        最终 schema 版本号（== baseline_version）。

    Raises:
        SchemaVersionError: 高版本库、非法版本记录或无法覆盖的旧版本。
        ValueError: 迁移注册表配置错误（断链等，属代码缺陷）。
        sqlite3.Error: 迁移语句失败（该步事务已回滚）。
    """
    _validate_step_chain(steps, baseline_version)
    # 防御性归零：清掉调用方遗留的隐式活动事务（保证下方只读检查与
    # 后续显式事务不互相干扰）；无活动事务时为空操作。
    conn.commit()

    stored = _read_stored_version(conn, version_key)
    if stored is not None:
        if stored > baseline_version:
            raise SchemaVersionError(
                f"数据库 schema 版本（{stored}）高于代码支持版本"
                f"（{baseline_version}，key={version_key!r}）："
                "未知高版本库拒绝打开（fail closed），请升级程序或"
                "在确认兼容后人工处理该数据库"
            )
        if stored == baseline_version:
            return stored
        return _apply_steps(
            conn,
            version_key=version_key,
            steps=steps,
            from_version=stored,
            baseline_version=baseline_version,
        )

    legacy = detect_legacy_version(conn)
    if legacy is not None:
        if legacy > baseline_version:
            raise SchemaVersionError(
                f"旧库探测版本（{legacy}）高于代码支持版本"
                f"（{baseline_version}，key={version_key!r}）："
                "拒绝打开（fail closed）"
            )
        if legacy == baseline_version:
            # 隐式迁移时代的旧库已处于当前基线：仅补登记版本号
            # （单事务幂等写入），此后打开走版本快路径，不再重复探测。
            def _register(conn: sqlite3.Connection) -> None:
                _ensure_meta_table(conn)
                _write_version(conn, version_key, baseline_version)

            run_in_transaction(conn, _register)
            return baseline_version
        return _apply_steps(
            conn,
            version_key=version_key,
            steps=steps,
            from_version=legacy,
            baseline_version=baseline_version,
        )

    # 空库：基线语句一次到位（单事务；语句均要求幂等形态）
    def _baseline(conn: sqlite3.Connection) -> None:
        _ensure_meta_table(conn)
        for statement in baseline_statements:
            conn.execute(statement)
        _write_version(conn, version_key, baseline_version)

    run_in_transaction(conn, _baseline)
    return baseline_version

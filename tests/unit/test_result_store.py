"""单元测试：执行结果存储（ResultStore 契约 + 两种实现）。

- InMemoryResultStore / SQLiteResultStore 通过参数化 fixture 共享同一组
  契约测试：执行记录保存与查询、get_latest_success 仅返回 success=True
  的最近成功记录、计划版本快照无损往返；
- SQLite 实现额外验证跨实例持久化与父目录自动创建。
"""

from __future__ import annotations

import pytest

from so_agent.fingerprint import spec_fingerprint
from so_agent.models import (
    Execution,
    ExecutionResult,
    PlanVersion,
    ReviewDecision,
    SubtaskSpec,
    TaskDecompositionIssue,
)
from so_agent.result_store import InMemoryResultStore, SQLiteResultStore


def make_spec(subtask_id: str = "s1", plan_version: int = 1) -> SubtaskSpec:
    return SubtaskSpec(
        subtask_id=subtask_id,
        plan_version=plan_version,
        title=f"子任务 {subtask_id}",
        instructions=f"完成子任务 {subtask_id}",
        role="code",
        allowed_tools=["write_file"],
    )


def make_result(*, success: bool = True, output: str = "42") -> ExecutionResult:
    return ExecutionResult(
        subtask_id="s1",
        success=success,
        output=output,
        evidence=["命令输出片段"],
        duration=1.5,
        error=None if success else "模拟失败",
    )


def make_execution(
    *,
    execution_id: str = "e1",
    subtask_id: str = "s1",
    plan_version: int = 1,
    status: str = "completed",
    success: bool | None = True,
    fingerprint: str | None = None,
    attempt: int = 1,
    **overrides,
) -> Execution:
    result = None if success is None else make_result(success=success)
    base = dict(
        execution_id=execution_id,
        task_id="T-001",
        subtask_id=subtask_id,
        plan_version=plan_version,
        agent_id=f"agent-{execution_id}",
        attempt=attempt,
        input_fingerprint=fingerprint if fingerprint is not None else spec_fingerprint(make_spec(subtask_id)),
        output_fingerprint="deadbeefdeadbeef",
        status=status,
        result=result,
        token_usage=123,
        duration=1.5,
        failure_type=None if status == "completed" else "TIMEOUT",
        recovery_action=None if status == "completed" else "retry",
        started_at="2026-10-08T00:00:00+00:00",
        completed_at="2026-10-08T00:00:01+00:00" if status == "completed" else None,
    )
    base.update(overrides)
    return Execution(**base)


def make_plan_version(version: int = 1, *, with_review: bool = True) -> PlanVersion:
    execution = make_execution(plan_version=version)
    return PlanVersion(
        version=version,
        specs=[make_spec(plan_version=version), make_spec("s2", plan_version=version)],
        executions={"s1": [execution]},
        review_decision=(
            ReviewDecision(stage="plan_review", passed=True, summary="通过")
            if with_review else None
        ),
        trigger_issue=(
            TaskDecompositionIssue(
                task_id="T-001",
                subtask_id="s1",
                plan_version=version,
                recommendation="建议合并子任务",
            )
            if with_review else None
        ),
        created_at="2026-10-08T00:00:00+00:00",
    )


@pytest.fixture(params=["memory", "sqlite"])
def store(request, tmp_path):
    """两种实现共享同一契约测试。"""
    if request.param == "memory":
        return InMemoryResultStore()
    return SQLiteResultStore(str(tmp_path / "nested" / "result_store.db"))


class TestExecutionContract:
    """执行记录保存与查询契约（两种实现共用）。"""

    def test_save_and_get_roundtrip(self, store) -> None:
        execution = make_execution()
        store.save_execution(execution)
        loaded = store.get_executions("s1", 1)
        assert len(loaded) == 1
        assert loaded[0] == execution

    def test_get_executions_filters_by_subtask_and_version(self, store) -> None:
        store.save_execution(make_execution(execution_id="e1", subtask_id="s1", plan_version=1))
        store.save_execution(make_execution(execution_id="e2", subtask_id="s2", plan_version=1))
        store.save_execution(make_execution(execution_id="e3", subtask_id="s1", plan_version=2))
        assert [e.execution_id for e in store.get_executions("s1", 1)] == ["e1"]
        assert [e.execution_id for e in store.get_executions("s2", 1)] == ["e2"]
        assert [e.execution_id for e in store.get_executions("s1", 2)] == ["e3"]
        assert store.get_executions("s9", 1) == []

    def test_get_executions_preserves_all_fields(self, store) -> None:
        execution = make_execution(status="failed", success=None, attempt=2)
        store.save_execution(execution)
        loaded = store.get_executions("s1", 1)[0]
        assert loaded.status == "failed"
        assert loaded.attempt == 2
        assert loaded.failure_type == "TIMEOUT"
        assert loaded.recovery_action == "retry"
        assert loaded.result is None
        assert loaded.token_usage == 123


class TestLatestSuccessContract:
    """get_latest_success：仅返回 success=True 的最近成功记录。"""

    def test_no_match_returns_none(self, store) -> None:
        assert store.get_latest_success("nonexistent") is None

    def test_empty_store_returns_none(self, store) -> None:
        fp = spec_fingerprint(make_spec())
        assert store.get_latest_success(fp) is None

    def test_returns_success_record(self, store) -> None:
        fp = spec_fingerprint(make_spec())
        store.save_execution(make_execution(fingerprint=fp))
        loaded = store.get_latest_success(fp)
        assert loaded is not None
        assert loaded.execution_id == "e1"
        assert loaded.result is not None and loaded.result.success is True

    def test_failed_records_are_ignored(self, store) -> None:
        fp = spec_fingerprint(make_spec())
        # 已完成但业务失败（success=False）与 failed 状态均不可复用
        store.save_execution(make_execution(execution_id="ef1", fingerprint=fp, success=False))
        store.save_execution(make_execution(execution_id="ef2", fingerprint=fp, status="failed", success=None))
        assert store.get_latest_success(fp) is None

    def test_multiple_success_returns_latest(self, store) -> None:
        fp = spec_fingerprint(make_spec())
        store.save_execution(make_execution(execution_id="e-old", fingerprint=fp, output_fingerprint="aaaaaaaaaaaaaaaa"))
        store.save_execution(make_execution(execution_id="e-new", fingerprint=fp, output_fingerprint="bbbbbbbbbbbbbbbb"))
        loaded = store.get_latest_success(fp)
        assert loaded is not None
        assert loaded.execution_id == "e-new"

    def test_success_before_later_failure_still_returned(self, store) -> None:
        """最近记录失败时，回溯返回更早的成功记录。"""
        fp = spec_fingerprint(make_spec())
        store.save_execution(make_execution(execution_id="e-ok", fingerprint=fp))
        store.save_execution(make_execution(execution_id="e-bad", fingerprint=fp, success=False))
        loaded = store.get_latest_success(fp)
        assert loaded is not None
        assert loaded.execution_id == "e-ok"

    def test_fingerprint_mismatch_not_returned(self, store) -> None:
        store.save_execution(make_execution(fingerprint="aaaaaaaaaaaaaaaa"))
        assert store.get_latest_success("bbbbbbbbbbbbbbbb") is None


class TestPlanVersionContract:
    """计划版本快照保存与读取契约（两种实现共用）。"""

    def test_roundtrip_full_snapshot(self, store) -> None:
        pv = make_plan_version(version=1)
        store.save_plan_version(pv)
        loaded = store.get_plan_version(1)
        assert loaded == pv

    def test_missing_version_returns_none(self, store) -> None:
        assert store.get_plan_version(999) is None

    def test_empty_snapshot_roundtrip(self, store) -> None:
        pv = PlanVersion(version=2, created_at="")
        store.save_plan_version(pv)
        loaded = store.get_plan_version(2)
        assert loaded is not None
        assert loaded.version == 2
        assert loaded.specs == []
        assert loaded.executions == {}
        assert loaded.review_decision is None
        assert loaded.trigger_issue is None

    def test_save_overwrites_same_version(self, store) -> None:
        store.save_plan_version(make_plan_version(version=1))
        updated = make_plan_version(version=1).model_copy(
            update={"created_at": "2026-10-08T01:00:00+00:00"}
        )
        store.save_plan_version(updated)
        loaded = store.get_plan_version(1)
        assert loaded is not None
        assert loaded.created_at == "2026-10-08T01:00:00+00:00"


class TestSQLitePersistence:
    """SQLite 实现特有行为。"""

    def test_data_survives_reopen(self, tmp_path) -> None:
        db_path = str(tmp_path / "persist.db")
        fp = spec_fingerprint(make_spec())
        store_a = SQLiteResultStore(db_path)
        store_a.save_execution(make_execution(fingerprint=fp))
        store_a.save_plan_version(make_plan_version(version=3))
        store_a.close()

        store_b = SQLiteResultStore(db_path)
        loaded_exec = store_b.get_latest_success(fp)
        assert loaded_exec is not None
        assert loaded_exec.execution_id == "e1"
        loaded_pv = store_b.get_plan_version(3)
        assert loaded_pv is not None
        assert loaded_pv.version == 3
        store_b.close()

    def test_creates_parent_directory(self, tmp_path) -> None:
        db_path = str(tmp_path / "deep" / "deeper" / "store.db")
        store = SQLiteResultStore(db_path)
        try:
            store.save_execution(make_execution())
            assert len(store.get_executions("s1", 1)) == 1
        finally:
            store.close()

    def test_repeated_save_same_execution_id_is_replace(self, tmp_path) -> None:
        store = SQLiteResultStore(str(tmp_path / "replace.db"))
        try:
            store.save_execution(make_execution(execution_id="e1", token_usage=1))
            store.save_execution(make_execution(execution_id="e1", token_usage=99))
            loaded = store.get_executions("s1", 1)
            assert len(loaded) == 1
            assert loaded[0].token_usage == 99
        finally:
            store.close()

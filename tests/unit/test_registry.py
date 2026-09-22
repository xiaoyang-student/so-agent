"""单元测试：动态 Agent 注册表 AgentRegistry。"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from so_agent.config import Settings
from so_agent.context import ProjectContext
from so_agent.models import AgentRecord
from so_agent.runtime.registry import (
    DEFAULT_SUPERVISOR_ID,
    AgentRegistry,
    AgentRegistryError,
)

UTC = timezone.utc


def make_record(agent_id: str, task_id: str = "t1", **kwargs) -> AgentRecord:
    return AgentRecord(agent_id=agent_id, agent_type="dynamic", parent_task_id=task_id, **kwargs)


@pytest.fixture
def registry(tmp_path):
    context = ProjectContext(sandbox_dir=tmp_path, project_name="registry-test")
    return AgentRegistry(
        context=context, settings=Settings(max_dynamic_agents=2)
    )


class TestRegister:
    def test_register_and_get(self, registry):
        record = make_record("a1")
        registry.register(record)
        assert registry.get("a1") is record
        assert registry.active_count() == 1

    def test_get_missing_returns_none(self, registry):
        assert registry.get("nope") is None

    def test_duplicate_id_rejected(self, registry):
        registry.register(make_record("a1"))
        with pytest.raises(AgentRegistryError):
            registry.register(make_record("a1"))
        assert registry.active_count() == 1

    def test_empty_agent_id_rejected(self, registry):
        with pytest.raises(AgentRegistryError):
            registry.register(make_record(""))
        with pytest.raises(AgentRegistryError):
            registry.register(make_record("   "))

    def test_wrong_type_rejected(self, registry):
        with pytest.raises(AgentRegistryError):
            registry.register("not-a-record")  # type: ignore[arg-type]

    def test_mirrored_to_context(self, tmp_path):
        context = ProjectContext(sandbox_dir=tmp_path, project_name="p")
        registry = AgentRegistry(context=context, settings=Settings(max_dynamic_agents=4))
        record = make_record("a1")
        registry.register(record)
        assert context.created_agents["a1"] is record


class TestMaxDynamicAgents:
    def test_limit_enforced(self, registry):
        registry.register(make_record("a1"))
        registry.register(make_record("a2"))
        with pytest.raises(AgentRegistryError):
            registry.register(make_record("a3"))
        assert registry.active_count() == 2

    def test_limit_from_settings_value(self, tmp_path):
        registry = AgentRegistry(settings=Settings(max_dynamic_agents=1))
        registry.register(make_record("a1"))
        with pytest.raises(AgentRegistryError):
            registry.register(make_record("a2"))

    def test_rejected_record_not_stored(self, registry):
        registry.register(make_record("a1"))
        registry.register(make_record("a2"))
        with pytest.raises(AgentRegistryError):
            registry.register(make_record("a3"))
        assert registry.get("a3") is None


class TestRelease:
    def test_release_frees_quota(self, registry):
        registry.register(make_record("a1"))
        registry.register(make_record("a2"))
        registry.release("a1")
        assert registry.active_count() == 1
        registry.register(make_record("a3"))  # 配额已恢复，可继续注册
        assert registry.active_count() == 2

    def test_release_is_idempotent(self, registry):
        registry.register(make_record("a1"))
        registry.release("a1")
        registry.release("a1")  # 重复释放不抛异常
        assert registry.active_count() == 0

    def test_release_unknown_raises(self, registry):
        with pytest.raises(AgentRegistryError):
            registry.release("ghost")

    def test_record_kept_after_release(self, registry):
        record = make_record("a1")
        registry.register(record)
        registry.release("a1")
        assert registry.get("a1") is record

    def test_released_kept_in_context(self, tmp_path):
        context = ProjectContext(sandbox_dir=tmp_path, project_name="p")
        registry = AgentRegistry(context=context, settings=Settings(max_dynamic_agents=2))
        record = make_record("a1")
        registry.register(record)
        registry.release("a1")
        assert context.created_agents["a1"] is record


class TestUpdateStatus:
    def test_update_existing(self, registry):
        registry.register(make_record("a1"))
        registry.update_status("a1", "running")
        assert registry.get("a1").status == "running"

    def test_update_unknown_raises(self, registry):
        with pytest.raises(AgentRegistryError):
            registry.update_status("ghost", "running")


class TestPurgeExpired:
    def test_purge_marks_revoked_and_frees_quota(self, registry):
        expired = make_record("a1", expires_at=datetime(2025, 1, 1, tzinfo=UTC))
        future = make_record("a2", expires_at=datetime(2035, 1, 1, tzinfo=UTC))
        registry.register(expired)
        registry.register(future)
        purged = registry.purge_expired(now=datetime(2030, 1, 1, tzinfo=UTC))
        assert purged == ["a1"]
        assert expired.status == "revoked"
        assert future.status == "created"
        assert registry.active_count() == 1

    def test_purge_ignores_no_expiry(self, registry):
        registry.register(make_record("a1"))
        assert registry.purge_expired(now=datetime(2030, 1, 1, tzinfo=UTC)) == []
        assert registry.active_count() == 1

    def test_purge_not_yet_expired_kept(self, registry):
        registry.register(make_record("a1", expires_at=datetime(2030, 6, 1, tzinfo=UTC)))
        purged = registry.purge_expired(now=datetime(2030, 1, 1, tzinfo=UTC))
        assert purged == []
        assert registry.active_count() == 1

    def test_purge_boundary_equals_now(self, registry):
        moment = datetime(2030, 1, 1, tzinfo=UTC)
        registry.register(make_record("a1", expires_at=moment))
        assert registry.purge_expired(now=moment) == ["a1"]

    def test_purge_naive_expiry_treated_as_utc(self, registry):
        naive = datetime(2020, 1, 1)
        registry.register(make_record("a1", expires_at=naive))
        purged = registry.purge_expired(now=datetime(2025, 1, 1, tzinfo=UTC))
        assert purged == ["a1"]

    def test_purge_records_keep_in_registry(self, registry):
        record = make_record("a1", expires_at=datetime(2025, 1, 1, tzinfo=UTC))
        registry.register(record)
        registry.purge_expired(now=datetime(2030, 1, 1, tzinfo=UTC))
        assert registry.get("a1") is record


class TestListByTask:
    def test_lists_only_matching_task(self, registry):
        registry.register(make_record("a1", "t1"))
        registry.register(make_record("a2", "t2"))
        listed = registry.list_by_task("t1")
        assert [record.agent_id for record in listed] == ["a1"]

    def test_sorted_by_created_at(self, registry):
        later = make_record("a2", created_at=datetime(2025, 2, 1, tzinfo=UTC))
        earlier = make_record("a1", created_at=datetime(2025, 1, 1, tzinfo=UTC))
        registry.register(later)
        registry.register(earlier)
        listed = registry.list_by_task("t1")
        assert [record.agent_id for record in listed] == ["a1", "a2"]

    def test_active_only_filter(self, registry):
        registry.register(make_record("a1"))
        registry.register(make_record("a2"))
        registry.release("a1")
        active = registry.list_by_task("t1", active_only=True)
        assert [record.agent_id for record in active] == ["a2"]
        all_records = registry.list_by_task("t1")
        assert len(all_records) == 2

    def test_unknown_task_returns_empty(self, registry):
        assert registry.list_by_task("nope") == []


class TestSupervisorBoundary:
    def test_supervisor_id_default(self, registry):
        assert registry.supervisor_id == DEFAULT_SUPERVISOR_ID

    def test_assert_controller_allows_supervisor(self, registry):
        registry.assert_controller(DEFAULT_SUPERVISOR_ID)  # 不抛异常

    def test_assert_controller_rejects_others(self, registry):
        with pytest.raises(PermissionError):
            registry.assert_controller("dynamic-agent-1")

    def test_settings_precedence_context_over_global(self, tmp_path):
        context = ProjectContext(sandbox_dir=tmp_path, project_name="p")
        context.config = Settings(max_dynamic_agents=1)
        registry = AgentRegistry(context=context)
        registry.register(make_record("a1"))
        with pytest.raises(AgentRegistryError):
            registry.register(make_record("a2"))

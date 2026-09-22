"""单元测试：主管路由器 SupervisorRouter。

通信拓扑：唯一主管 + 子 Agent 星形结构（仅允许纵向通信）；
全局约束：所有 Agent 通信必须严格使用 JSON 对象。
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from so_agent.config import Settings
from so_agent.context import ProjectContext
from so_agent.models import TaskRequest
from so_agent.runtime.events import (
    EVENT_MESSAGE_REJECTED,
    EVENT_MESSAGE_ROUTED,
    EventLogger,
)
from so_agent.runtime.registry import AgentRegistry
from so_agent.runtime.supervisor_router import (
    CommunicationError,
    SupervisorRouter,
)

UTC = timezone.utc
SUPERVISOR = "supervisor"
CHILD_A = "t1:s1:dynamic-aaaa1111"
CHILD_B = "t1:s2:dynamic-bbbb2222"


@pytest.fixture
def events():
    return EventLogger()


@pytest.fixture
def router(events):
    return SupervisorRouter(events=events)


def routed_events(events: EventLogger) -> list[dict]:
    return events.get_events(event_type=EVENT_MESSAGE_ROUTED)


def rejected_events(events: EventLogger) -> list[dict]:
    return events.get_events(event_type=EVENT_MESSAGE_REJECTED)


class TestVerticalCommunicationAllowed:
    def test_supervisor_to_child(self, router, events):
        router.route_message(SUPERVISOR, CHILD_A, {"phase": "dispatch"})
        records = routed_events(events)
        assert len(records) == 1
        assert records[0]["payload"]["sender_id"] == SUPERVISOR
        assert records[0]["payload"]["recipient_id"] == CHILD_A
        assert rejected_events(events) == []

    def test_child_to_supervisor(self, router, events):
        router.route_message(CHILD_A, SUPERVISOR, {"phase": "result"})
        records = routed_events(events)
        assert len(records) == 1
        assert records[0]["payload"]["sender_id"] == CHILD_A

    def test_validate_communication(self, router):
        assert router.validate_communication(SUPERVISOR, CHILD_A) is True
        assert router.validate_communication(CHILD_A, SUPERVISOR) is True
        assert router.validate_communication(CHILD_A, CHILD_B) is False

    def test_supervisor_id_property(self, router):
        assert router.supervisor_id == SUPERVISOR


class TestHorizontalCommunicationRejected:
    def test_child_to_child_rejected(self, router, events):
        with pytest.raises(CommunicationError):
            router.route_message(CHILD_A, CHILD_B, {"phase": "chat"})
        records = rejected_events(events)
        assert len(records) == 1
        assert "横向通信" in records[0]["payload"]["reason"]
        assert routed_events(events) == []

    def test_self_loop_rejected(self, router, events):
        with pytest.raises(CommunicationError):
            router.route_message(SUPERVISOR, SUPERVISOR, {"a": 1})
        with pytest.raises(CommunicationError):
            router.route_message(CHILD_A, CHILD_A, {"a": 1})
        assert len(rejected_events(events)) == 2

    def test_blank_party_rejected(self, router, events):
        with pytest.raises(CommunicationError):
            router.route_message("", CHILD_A, {"a": 1})
        with pytest.raises(CommunicationError):
            router.route_message(SUPERVISOR, "  ", {"a": 1})
        assert len(rejected_events(events)) == 2


class TestMessageJsonStrictness:
    def test_plain_text_rejected(self, router, events):
        with pytest.raises(CommunicationError):
            router.route_message(SUPERVISOR, CHILD_A, "你好，请执行任务")
        assert len(rejected_events(events)) == 1
        assert "JSON" in rejected_events(events)[0]["payload"]["reason"]

    def test_invalid_json_string_rejected(self, router, events):
        with pytest.raises(CommunicationError):
            router.route_message(SUPERVISOR, CHILD_A, "{broken json")
        assert len(rejected_events(events)) == 1

    def test_json_array_rejected(self, router):
        with pytest.raises(CommunicationError):
            router.route_message(SUPERVISOR, CHILD_A, "[1, 2, 3]")

    def test_json_scalar_rejected(self, router):
        with pytest.raises(CommunicationError):
            router.route_message(SUPERVISOR, CHILD_A, "42")

    def test_non_str_dict_model_type_rejected(self, router):
        with pytest.raises(CommunicationError):
            router.route_message(SUPERVISOR, CHILD_A, 123)  # type: ignore[arg-type]

    def test_valid_json_string_accepted_and_normalized(self, router, events):
        router.route_message(SUPERVISOR, CHILD_A, '{"phase": "dispatch", "n": 1}')
        payload = routed_events(events)[0]["payload"]
        assert payload["message"]["phase"] == "dispatch"
        assert payload["message"]["n"] == 1

    def test_dict_message_setdefaults_identity(self, router, events):
        router.route_message(SUPERVISOR, CHILD_A, {"phase": "dispatch"})
        message = routed_events(events)[0]["payload"]["message"]
        assert message["sender_id"] == SUPERVISOR
        assert message["recipient_id"] == CHILD_A

    def test_pydantic_model_message(self, router, events):
        request = TaskRequest(task_id="t1", objective="目标")
        router.route_message(SUPERVISOR, CHILD_A, request)
        message = routed_events(events)[0]["payload"]["message"]
        assert message["task_id"] == "t1"
        assert message["objective"] == "目标"

    def test_datetime_in_message_converted(self, router, events):
        router.route_message(
            SUPERVISOR, CHILD_A, {"when": datetime(2025, 1, 1, tzinfo=UTC)}
        )
        message = routed_events(events)[0]["payload"]["message"]
        assert message["when"] == "2025-01-01T00:00:00+00:00"

    def test_unserializable_content_rejected(self, router, events):
        with pytest.raises(CommunicationError):
            router.route_message(SUPERVISOR, CHILD_A, {"obj": object()})
        assert len(rejected_events(events)) == 1


class TestIdentityAntiSpoofing:
    def test_mismatched_sender_id_rejected(self, router, events):
        with pytest.raises(CommunicationError):
            router.route_message(
                SUPERVISOR, CHILD_A, {"sender_id": "someone-else", "phase": "x"}
            )
        assert "冒充" in rejected_events(events)[0]["payload"]["reason"]

    def test_mismatched_recipient_id_rejected(self, router, events):
        with pytest.raises(CommunicationError):
            router.route_message(
                SUPERVISOR, CHILD_A, {"recipient_id": CHILD_B, "phase": "x"}
            )
        assert len(rejected_events(events)) == 1

    def test_matching_identity_fields_accepted(self, router, events):
        router.route_message(
            SUPERVISOR,
            CHILD_A,
            {"sender_id": SUPERVISOR, "recipient_id": CHILD_A, "phase": "x"},
        )
        assert len(routed_events(events)) == 1


class TestRegistryEnforcement:
    def _make_router(self, tmp_path, events):
        context = ProjectContext(sandbox_dir=tmp_path, project_name="router-test")
        registry = AgentRegistry(context=context, settings=Settings(max_dynamic_agents=4))
        from so_agent.models import AgentRecord

        registry.register(
            AgentRecord(
                agent_id=CHILD_A, agent_type="dynamic", parent_task_id="t1"
            )
        )
        return SupervisorRouter(
            registry=registry, events=events, extra_recipients=("code_agent",)
        )

    def test_registered_child_allowed(self, tmp_path, events):
        router = self._make_router(tmp_path, events)
        router.route_message(SUPERVISOR, CHILD_A, {"phase": "x"})
        assert len(routed_events(events)) == 1

    def test_unregistered_child_rejected(self, tmp_path, events):
        router = self._make_router(tmp_path, events)
        with pytest.raises(CommunicationError):
            router.route_message(SUPERVISOR, "ghost-agent", {"phase": "x"})
        assert "未登记" in rejected_events(events)[0]["payload"]["reason"]

    def test_extra_recipient_exempt(self, tmp_path, events):
        router = self._make_router(tmp_path, events)
        router.route_message(SUPERVISOR, "code_agent", {"phase": "x"})
        assert len(routed_events(events)) == 1

    def test_supervisor_always_allowed(self, tmp_path, events):
        router = self._make_router(tmp_path, events)
        router.route_message(CHILD_A, SUPERVISOR, {"phase": "done"})
        assert len(routed_events(events)) == 1


class TestEventPayloads:
    def test_routed_event_fields(self, router, events):
        router.route_message(SUPERVISOR, CHILD_A, {"phase": "dispatch"})
        record = routed_events(events)[0]
        assert record["agent_id"] == SUPERVISOR
        payload = record["payload"]
        assert set(payload) == {"sender_id", "recipient_id", "agent_id", "message"}
        json.dumps(payload, ensure_ascii=False)  # 可序列化

    def test_rejected_event_fields(self, router, events):
        with pytest.raises(CommunicationError):
            router.route_message(CHILD_A, CHILD_B, {"phase": "x"})
        payload = rejected_events(events)[0]["payload"]
        assert payload["sender_id"] == CHILD_A
        assert payload["recipient_id"] == CHILD_B
        assert payload["agent_id"] == CHILD_A
        assert payload["reason"]

    def test_rejected_message_not_logged_as_routed(self, router, events):
        with pytest.raises(CommunicationError):
            router.route_message(SUPERVISOR, CHILD_A, "自由文本")
        assert routed_events(events) == []
        assert len(rejected_events(events)) == 1

    def test_multiple_routes_recorded_in_order(self, router, events):
        router.route_message(SUPERVISOR, CHILD_A, {"i": 1})
        router.route_message(CHILD_A, SUPERVISOR, {"i": 2})
        messages = [r["payload"]["message"]["i"] for r in routed_events(events)]
        assert messages == [1, 2]

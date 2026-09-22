"""单元测试：事件记录器 EventLogger。"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from so_agent.context import ProjectContext
from so_agent.models import TaskRequest
from so_agent.runtime.events import (
    ALL_EVENT_TYPES,
    EVENT_TASK_CREATED,
    EVENT_TOOL_CALLED,
    EVENT_TOOL_GRANTED,
    EventLogger,
)

UTC = timezone.utc


@pytest.fixture
def context(tmp_path):
    return ProjectContext(sandbox_dir=tmp_path, project_name="events-test")


class TestEventTypesRegistry:
    def test_fourteen_registered_types(self):
        assert len(ALL_EVENT_TYPES) == 14

    def test_expected_type_names(self):
        expected = {
            "task_created",
            "status_transition",
            "task_dispatched",
            "tool_granted",
            "tool_revoked",
            "tool_called",
            "agent_created",
            "agent_released",
            "replacement_requested",
            "decomposition_issue",
            "replan",
            "human_escalation",
            "message_routed",
            "message_rejected",
        }
        assert set(ALL_EVENT_TYPES) == expected


class TestLogEvent:
    def test_record_structure(self):
        logger = EventLogger()
        record = logger.log_event(
            EVENT_TASK_CREATED,
            {"task_id": "t1", "agent_id": "a1", "objective": "o"},
        )
        assert set(record.keys()) == {
            "event_id",
            "event_type",
            "timestamp",
            "task_id",
            "agent_id",
            "payload",
        }
        assert record["event_type"] == EVENT_TASK_CREATED
        assert record["task_id"] == "t1"
        assert record["agent_id"] == "a1"
        assert record["payload"]["objective"] == "o"
        # timestamp 为 ISO 8601 字符串
        parsed = datetime.fromisoformat(record["timestamp"])
        assert parsed.tzinfo is not None

    def test_none_payload_becomes_empty_dict(self):
        logger = EventLogger()
        record = logger.log_event(EVENT_TASK_CREATED)
        assert record["payload"] == {}
        assert record["task_id"] is None
        assert record["agent_id"] is None

    def test_index_fields_extracted_from_payload(self):
        logger = EventLogger()
        record = logger.log_event(
            EVENT_TOOL_CALLED,
            {"task_id": "t9", "agent_id": "a9", "tool": "write_file"},
        )
        assert record["task_id"] == "t9"
        assert record["agent_id"] == "a9"

    def test_custom_timestamp_kept(self):
        logger = EventLogger()
        ts = datetime(2025, 5, 1, 12, 0, tzinfo=UTC)
        record = logger.log_event(EVENT_TASK_CREATED, {"task_id": "t1"}, timestamp=ts)
        assert record["timestamp"] == ts.isoformat()

    def test_naive_timestamp_treated_as_utc(self):
        logger = EventLogger()
        ts = datetime(2025, 5, 1, 12, 0)  # naive
        record = logger.log_event(EVENT_TASK_CREATED, {}, timestamp=ts)
        assert record["timestamp"] == ts.replace(tzinfo=UTC).isoformat()

    def test_event_id_unique(self):
        logger = EventLogger()
        ids = {logger.log_event(EVENT_TASK_CREATED, {})["event_id"] for _ in range(20)}
        assert len(ids) == 20


class TestInvalidInputs:
    def test_unregistered_event_type_rejected(self):
        logger = EventLogger()
        with pytest.raises(ValueError):
            logger.log_event("not_registered_event", {"a": 1})
        assert len(logger) == 0

    def test_non_string_event_type_rejected(self):
        logger = EventLogger()
        with pytest.raises(ValueError):
            logger.log_event(123, {})  # type: ignore[arg-type]

    def test_payload_wrong_type_rejected(self):
        logger = EventLogger()
        with pytest.raises(TypeError):
            logger.log_event(EVENT_TASK_CREATED, ["not", "a", "dict"])  # type: ignore[arg-type]
        with pytest.raises(TypeError):
            logger.log_event(EVENT_TASK_CREATED, "text")  # type: ignore[arg-type]

    def test_unserializable_payload_rejected(self):
        logger = EventLogger()
        with pytest.raises(ValueError):
            logger.log_event(EVENT_TASK_CREATED, {"obj": object()})
        assert len(logger) == 0


class TestJsonCompatibility:
    def test_pydantic_model_payload(self):
        logger = EventLogger()
        request = TaskRequest(task_id="t1", objective="目标")
        record = logger.log_event(EVENT_TASK_CREATED, request)
        assert record["payload"]["task_id"] == "t1"
        assert record["payload"]["objective"] == "目标"
        assert record["task_id"] == "t1"

    def test_datetime_inside_payload_converted(self):
        logger = EventLogger()
        ts = datetime(2025, 1, 1, tzinfo=UTC)
        record = logger.log_event(EVENT_TASK_CREATED, {"when": ts})
        assert record["payload"]["when"] == ts.isoformat()

    def test_set_and_tuple_converted_to_list(self):
        logger = EventLogger()
        record = logger.log_event(EVENT_TASK_CREATED, {"tags": ("a", "b"), "ids": {1, 2}})
        assert record["payload"]["tags"] == ["a", "b"]
        assert sorted(record["payload"]["ids"]) == [1, 2]

    def test_nested_struct_serializable(self):
        logger = EventLogger()
        record = logger.log_event(
            EVENT_TASK_CREATED,
            {"nested": {"items": [{"key": "值"}, (1, 2)]}},
        )
        json.dumps(record["payload"], ensure_ascii=False)  # 不抛异常即通过

    def test_payload_can_be_json_dumped(self):
        logger = EventLogger()
        record = logger.log_event(
            EVENT_TOOL_CALLED,
            {"task_id": "t1", "agent_id": "a1", "args": {"x": [1, 2, 3]}},
        )
        dumped = json.dumps(record["payload"], ensure_ascii=False)
        assert json.loads(dumped)["args"]["x"] == [1, 2, 3]


class TestQueryAndFilter:
    def _seed(self, logger: EventLogger) -> None:
        logger.log_event(EVENT_TASK_CREATED, {"task_id": "t1", "agent_id": "a1"})
        logger.log_event(EVENT_TOOL_GRANTED, {"task_id": "t1", "agent_id": "a2"})
        logger.log_event(EVENT_TOOL_CALLED, {"task_id": "t2", "agent_id": "a1"})
        logger.log_event(EVENT_TOOL_CALLED, {"task_id": "t1", "agent_id": "a1"})

    def test_no_filter_returns_all(self):
        logger = EventLogger()
        self._seed(logger)
        assert len(logger.get_events()) == 4

    def test_filter_by_task_id(self):
        logger = EventLogger()
        self._seed(logger)
        task1 = logger.get_events(task_id="t1")
        assert len(task1) == 3
        assert all(record["task_id"] == "t1" for record in task1)

    def test_filter_by_agent_id(self):
        logger = EventLogger()
        self._seed(logger)
        agent1 = logger.get_events(agent_id="a1")
        assert len(agent1) == 3
        assert all(record["agent_id"] == "a1" for record in agent1)

    def test_filter_by_event_type(self):
        logger = EventLogger()
        self._seed(logger)
        called = logger.get_events(event_type=EVENT_TOOL_CALLED)
        assert len(called) == 2
        assert all(record["event_type"] == EVENT_TOOL_CALLED for record in called)

    def test_combined_filters(self):
        logger = EventLogger()
        self._seed(logger)
        matched = logger.get_events(task_id="t1", event_type=EVENT_TOOL_CALLED)
        assert len(matched) == 1
        assert matched[0]["agent_id"] == "a1"

    def test_returned_list_is_copy(self):
        logger = EventLogger()
        self._seed(logger)
        result = logger.get_events()
        result.clear()
        assert len(logger.get_events()) == 4


class TestSubscription:
    def test_subscribe_receives_records(self):
        logger = EventLogger()
        received: list[dict] = []
        logger.subscribe(received.append)
        logger.log_event(EVENT_TASK_CREATED, {"task_id": "t1"})
        assert len(received) == 1
        assert received[0]["task_id"] == "t1"

    def test_duplicate_subscribe_ignored(self):
        logger = EventLogger()
        received: list[dict] = []
        logger.subscribe(received.append)
        logger.subscribe(received.append)
        logger.log_event(EVENT_TASK_CREATED, {})
        assert len(received) == 1

    def test_unsubscribe_stops_delivery(self):
        logger = EventLogger()
        received: list[dict] = []
        logger.subscribe(received.append)
        logger.log_event(EVENT_TASK_CREATED, {})
        logger.unsubscribe(received.append)
        logger.log_event(EVENT_TASK_CREATED, {})
        assert len(received) == 1

    def test_unsubscribe_missing_callback_silent(self):
        logger = EventLogger()
        logger.unsubscribe(lambda record: None)  # 不抛异常

    def test_subscriber_exception_does_not_break_logging(self):
        logger = EventLogger()

        def bad_callback(record):
            raise RuntimeError("订阅者故障")

        logger.subscribe(bad_callback)
        record = logger.log_event(EVENT_TASK_CREATED, {"task_id": "t1"})
        assert record["task_id"] == "t1"
        assert len(logger) == 1


class TestContextMirroring:
    def test_events_appended_to_context(self, context):
        logger = EventLogger(context=context)
        logger.log_event(EVENT_TASK_CREATED, {"task_id": "t1"})
        logger.log_event(EVENT_TOOL_GRANTED, {"task_id": "t1"})
        assert len(context.event_log) == 2
        assert context.event_log[0]["event_type"] == EVENT_TASK_CREATED

    def test_clear_does_not_touch_context(self, context):
        logger = EventLogger(context=context)
        logger.log_event(EVENT_TASK_CREATED, {"task_id": "t1"})
        logger.clear()
        assert len(logger) == 0
        assert len(context.event_log) == 1


class TestMaxEvents:
    def test_truncates_oldest_records(self):
        logger = EventLogger(max_events=3)
        for index in range(5):
            logger.log_event(EVENT_TASK_CREATED, {"index": index})
        events = logger.events
        assert len(events) == 3
        assert [record["payload"]["index"] for record in events] == [2, 3, 4]

    def test_no_limit_keeps_all(self):
        logger = EventLogger()
        for _ in range(10):
            logger.log_event(EVENT_TASK_CREATED, {})
        assert len(logger) == 10


class TestEventsProperty:
    def test_events_returns_copy(self):
        logger = EventLogger()
        logger.log_event(EVENT_TASK_CREATED, {})
        snapshot = logger.events
        snapshot.clear()
        assert len(logger) == 1

    def test_clear_empties_internal_list(self):
        logger = EventLogger()
        logger.log_event(EVENT_TASK_CREATED, {})
        logger.clear()
        assert len(logger) == 0
        assert logger.events == []


class TestTimestampPersistence:
    def test_out_of_order_timestamps_preserved(self):
        logger = EventLogger()
        early = datetime(2025, 1, 1, tzinfo=UTC)
        late = early + timedelta(hours=2)
        logger.log_event(EVENT_TASK_CREATED, {}, timestamp=late)
        logger.log_event(EVENT_TOOL_GRANTED, {}, timestamp=early)
        events = logger.events
        assert events[0]["timestamp"] == late.isoformat()
        assert events[1]["timestamp"] == early.isoformat()

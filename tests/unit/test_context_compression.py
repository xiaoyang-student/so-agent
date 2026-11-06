"""确定性上下文压缩：预算、标记、幂等与事件。"""

from __future__ import annotations

from types import SimpleNamespace

from so_agent.config import Settings
from so_agent.runtime.context_compression import (
    DEFAULT_PROTECTED_KEYS,
    compress_payload,
    compress_text,
)
from so_agent.workflow_support import BaseCoordinator


class _Recorder:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict]] = []

    def log_event(self, event_type: str, payload: dict) -> None:
        self.events.append((event_type, payload))


def test_compress_text_keeps_head_and_tail():
    text = "HEAD" + "x" * 1000 + "TAIL"
    compressed = compress_text(text, 120)
    assert len(compressed) == 120
    assert compressed.startswith("HEAD")
    assert compressed.endswith("TAIL")
    assert "上下文压缩" in compressed


def test_compress_text_noop_when_within_budget():
    assert compress_text("short", 100) == "short"
    assert compress_text("short", 0) == "short"


def test_compress_payload_limits_each_string_and_total():
    payload = {
        "task": "t1",
        "results": [{"output": "a" * 9000}],
        "nested": {"deep": ["b" * 9000, "c" * 9000]},
        "count": 3,
        "ok": True,
        "empty": None,
    }
    compressed, report = compress_payload(
        payload, max_total_chars=600, per_string_chars=400
    )
    flattened = [
        compressed["results"][0]["output"],
        compressed["nested"]["deep"][0],
        compressed["nested"]["deep"][1],
    ]
    assert all(len(item) <= 400 for item in flattened)
    assert compressed["count"] == 3
    assert compressed["ok"] is True
    assert compressed["empty"] is None
    assert report.changed is True
    assert report.compressed_chars < report.original_chars
    assert report.paths


def test_compress_payload_is_deterministic_and_non_mutating():
    payload = {"a": "x" * 5000, "b": {"c": "y" * 5000}}
    first, _ = compress_payload(payload, max_total_chars=400, per_string_chars=300)
    second, _ = compress_payload(payload, max_total_chars=400, per_string_chars=300)
    assert first == second
    assert payload["a"] == "x" * 5000
    assert payload["b"]["c"] == "y" * 5000


def test_compress_payload_handles_root_string():
    compressed, report = compress_payload(
        "z" * 5000, max_total_chars=300, per_string_chars=300
    )
    assert isinstance(compressed, str)
    assert len(compressed) <= 300
    assert report.paths == ["$"]


def test_compress_context_emits_event_only_when_changed():
    recorder = _Recorder()
    coordinator = SimpleNamespace(
        _settings=Settings(
            _env_file=None,
            context_compression_max_chars=200,
            context_compression_result_chars=50,
        ),
        _events=recorder,
        _task_request=None,
    )
    payload = {"result": "r" * 2000}
    compressed = BaseCoordinator._compress_context(coordinator, payload)
    assert len(compressed["result"]) <= 50
    assert recorder.events
    event_type, event_payload = recorder.events[0]
    assert event_type == "context_compressed"
    assert event_payload["original_chars"] > event_payload["compressed_chars"]
    assert event_payload["task_id"] is None

    recorder.events.clear()
    coordinator._settings = Settings(
        _env_file=None, context_compression_enabled=False
    )
    same = BaseCoordinator._compress_context(coordinator, payload)
    assert same is payload
    assert recorder.events == []


def test_protected_keys_survive_per_string_and_total_budget():
    long_path = "deep/" + "p" * 500 + ".txt"
    payload = {
        "task_id": "t-" + "1" * 500,
        "subtask_id": "s-" + "2" * 500,
        "write_scope": [long_path],
        "acceptance_criteria": ["a" * 600],
        "result": "r" * 9000,
    }
    compressed, _ = compress_payload(
        payload,
        max_total_chars=300,
        per_string_chars=100,
        protected_keys=DEFAULT_PROTECTED_KEYS,
    )
    assert compressed["task_id"] == payload["task_id"]
    assert compressed["subtask_id"] == payload["subtask_id"]
    assert compressed["write_scope"] == payload["write_scope"]
    assert compressed["acceptance_criteria"] == payload["acceptance_criteria"]
    assert len(compressed["result"]) < len(payload["result"])


def test_without_protected_keys_legacy_behaviour_truncates_everything():
    payload = {"task_id": "t-" + "1" * 500, "result": "r" * 5000}
    compressed, _ = compress_payload(
        payload, max_total_chars=300, per_string_chars=100
    )
    assert len(compressed["task_id"]) <= 100
    assert len(compressed["result"]) <= 100


def test_compress_context_defaults_protect_critical_fields():
    recorder = _Recorder()
    coordinator = SimpleNamespace(
        _settings=Settings(
            _env_file=None,
            context_compression_max_chars=200,
            context_compression_result_chars=50,
        ),
        _events=recorder,
        _task_request=None,
    )
    payload = {"subtask_id": "s-" + "9" * 500, "result": "r" * 2000}
    compressed = BaseCoordinator._compress_context(coordinator, payload)
    assert compressed["subtask_id"] == payload["subtask_id"]
    assert len(compressed["result"]) <= 50

    coordinator._settings = Settings(
        _env_file=None,
        context_compression_max_chars=200,
        context_compression_result_chars=50,
        context_compression_protect_critical=False,
    )
    unprotected = BaseCoordinator._compress_context(coordinator, payload)
    assert len(unprotected["subtask_id"]) <= 50

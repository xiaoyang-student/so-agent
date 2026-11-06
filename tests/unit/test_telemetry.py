"""单元测试：可选 OpenTelemetry 追踪。"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("opentelemetry.sdk.trace")

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.trace import StatusCode

from so_agent.config import Settings
from so_agent.models import AgentRecord, ExecutionResult, SubtaskSpec
from so_agent.runtime import telemetry
from so_agent.runtime.scheduler import SubtaskDAGScheduler


@pytest.fixture(autouse=True)
def _reset_telemetry():
    telemetry.reset_telemetry_for_tests()
    yield
    telemetry.reset_telemetry_for_tests()


@pytest.fixture
def exporter():
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    telemetry.set_tracer_for_tests(provider.get_tracer("test"))
    yield exporter
    provider.shutdown()


def _settings(**overrides):
    values = {
        "otel_enabled": True,
        "otel_service_name": "so-agent-test",
        "otel_exporter": "none",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class TestDisabled:
    def test_default_settings_keep_tracing_off(self):
        settings = Settings()
        assert settings.otel_enabled is False
        assert settings.otel_exporter == "none"
        assert settings.otel_service_name == "so-agent"

    def test_disabled_scope_yields_none(self, monkeypatch):
        monkeypatch.setattr(
            telemetry, "_settings", lambda: _settings(otel_enabled=False)
        )
        assert telemetry.telemetry_enabled() is False
        with telemetry.span_scope("x", {"a": 1}) as span:
            assert span is None
        assert telemetry.current_trace_context() == {}

    async def test_disabled_async_scope_yields_none(self, monkeypatch):
        monkeypatch.setattr(
            telemetry, "_settings", lambda: _settings(otel_enabled=False)
        )
        async with telemetry.async_span_scope("x") as span:
            assert span is None

    def test_injected_none_tracer_yields_none(self):
        telemetry.set_tracer_for_tests(None)
        assert telemetry.telemetry_enabled() is False
        with telemetry.span_scope("x") as span:
            assert span is None


class TestSpanScope:
    def test_span_created_with_clean_attributes(self, exporter):
        with telemetry.span_scope(
            "so_agent.test", {"a": 1, "b": None, "c": object()}
        ) as span:
            assert span is not None
        spans = exporter.get_finished_spans()
        assert [s.name for s in spans] == ["so_agent.test"]
        attrs = dict(spans[0].attributes)
        assert attrs["a"] == 1
        assert "b" not in attrs
        assert isinstance(attrs["c"], str)

    def test_exception_recorded_and_propagated(self, exporter):
        with pytest.raises(ValueError):
            with telemetry.span_scope("so_agent.fail") as span:
                assert span is not None
                raise ValueError("boom")
        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].status.status_code is StatusCode.ERROR
        assert any(event.name == "exception" for event in spans[0].events)

    async def test_async_scope_nests_and_exposes_context(self, exporter):
        async with telemetry.async_span_scope("outer") as outer:
            async with telemetry.async_span_scope("inner") as inner:
                context = telemetry.current_trace_context()
                assert len(context["trace_id"]) == 32
                assert len(context["span_id"]) == 16
            assert outer is not None and inner is not None
        spans = {s.name: s for s in exporter.get_finished_spans()}
        assert set(spans) == {"outer", "inner"}
        assert spans["inner"].parent.span_id == spans["outer"].context.span_id

    def test_set_span_attributes_tolerates_none(self):
        telemetry.set_span_attributes(None, {"a": 1})
        with telemetry.span_scope("attrs") as span:
            telemetry.set_span_attributes(span, {"n": None, "ok": "v"})


class TestEnabledProvider:
    def test_enabled_with_global_provider(self, monkeypatch):
        monkeypatch.setattr(telemetry, "_settings", lambda: _settings())
        assert telemetry.telemetry_enabled() is True


class TestSchedulerIntegration:
    async def test_subtask_span_records_result(self, exporter):
        async def executor(
            spec: SubtaskSpec, agent: AgentRecord
        ) -> ExecutionResult:
            return ExecutionResult(subtask_id=spec.subtask_id, success=True)

        scheduler = SubtaskDAGScheduler(executor)
        spec = SubtaskSpec(
            subtask_id="s1",
            title="s1",
            instructions="do s1",
            role="code",
        )
        results = await scheduler.schedule("task-1", [spec])
        assert results["s1"].success

        subtask_spans = [
            s for s in exporter.get_finished_spans() if s.name == "so_agent.subtask"
        ]
        assert len(subtask_spans) == 1
        attrs = dict(subtask_spans[0].attributes)
        assert attrs["task.id"] == "task-1"
        assert attrs["subtask.id"] == "s1"
        assert attrs["subtask.success"] is True

"""单元测试：确定性执行器路由、动态子 Agent 路由守卫与 token 归因。"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from agents.tool_context import ToolContext

from so_agent.config import Settings
from so_agent.context import ProjectContext
from so_agent.models import SubtaskSpec, ToolGrant
from so_agent.runtime.events import EventLogger, shared_event_logger
from so_agent.runtime.registry import AgentRegistry, DEFAULT_SUPERVISOR_ID
from so_agent.runtime.routing import (
    EXECUTOR_CODE_AGENT,
    EXECUTOR_SUBAGENT_CREATOR,
    REASON_NON_STANDARD_TOOLS,
    REASON_STANDARD_TOOLS_ONLY,
    is_standard_only,
    resolve_executor,
)
from so_agent.runtime.usage import (
    extract_completion_usage,
    extract_usage,
    make_usage_recorder,
)
from so_agent.tool_packages.subagent_creator import agent as creator_mod


def make_ctx(tool_name: str) -> ToolContext:
    return ToolContext(
        context=None,
        tool_name=tool_name,
        tool_call_id=f"call-{tool_name}-1",
        tool_arguments="{}",
    )


class TestResolveExecutor:
    def test_empty_tools_route_to_code_agent(self):
        assert resolve_executor([]) == {
            "tool": EXECUTOR_CODE_AGENT,
            "reason": REASON_STANDARD_TOOLS_ONLY,
        }
        assert resolve_executor(None)["tool"] == EXECUTOR_CODE_AGENT

    def test_standard_tools_route_to_code_agent(self):
        decision = resolve_executor(["read_file", "write_file", "run_in_sandbox"])
        assert decision["tool"] == EXECUTOR_CODE_AGENT
        assert decision["reason"] == REASON_STANDARD_TOOLS_ONLY

    def test_whitespace_and_blank_entries_are_ignored(self):
        assert is_standard_only([" read_file ", "", "  "])

    def test_non_standard_tool_routes_to_subagent_creator(self):
        decision = resolve_executor(["excel_export"])
        assert decision["tool"] == EXECUTOR_SUBAGENT_CREATOR
        assert decision["reason"] == REASON_NON_STANDARD_TOOLS

    def test_mixed_tools_route_to_subagent_creator(self):
        assert (
            resolve_executor(["read_file", "domain_tool"])["tool"]
            == EXECUTOR_SUBAGENT_CREATOR
        )


class TestStreamedUsageAttribution:
    @staticmethod
    def _usage(total: int) -> SimpleNamespace:
        return SimpleNamespace(
            input_tokens=total - 2, output_tokens=2, total_tokens=total
        )

    def test_raw_responses_take_priority(self):
        usage = self._usage(10)
        result = SimpleNamespace(
            raw_responses=[SimpleNamespace(usage=usage)],
            _state=SimpleNamespace(_model_responses=[SimpleNamespace(usage=usage)]),
        )
        assert extract_usage(result)["requests"] == 1
        assert extract_usage(result)["total_tokens"] == 10

    def test_streamed_result_falls_back_to_run_state(self):
        """流式结果 raw_responses 为空时，用量从 run_state 内部响应累计。"""
        usage = self._usage(42)
        result = SimpleNamespace(
            raw_responses=[],
            _state=SimpleNamespace(
                _model_responses=[
                    SimpleNamespace(usage=usage),
                    SimpleNamespace(usage=usage),
                ]
            ),
        )
        assert extract_usage(result) == {
            "requests": 2,
            "input_tokens": 80,
            "output_tokens": 4,
            "total_tokens": 84,
            "cached_tokens": 0,
            "cache_write_tokens": 0,
        }

    def test_to_state_fallback(self):
        usage = self._usage(7)
        result = SimpleNamespace(
            raw_responses=[],
            to_state=lambda: SimpleNamespace(
                _model_responses=[SimpleNamespace(usage=usage)]
            ),
        )
        assert extract_usage(result)["total_tokens"] == 7

    def test_no_responses_yields_zero(self):
        assert extract_usage(SimpleNamespace(raw_responses=[])) == {
            "requests": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "cached_tokens": 0,
            "cache_write_tokens": 0,
        }


class TestCacheUsageExtraction:
    def test_sdk_details_sum_cached_and_written(self):
        usage = SimpleNamespace(
            input_tokens=1000,
            output_tokens=50,
            total_tokens=1050,
            input_tokens_details=SimpleNamespace(
                cached_tokens=768, cache_write_tokens=64
            ),
        )
        result = SimpleNamespace(
            raw_responses=[SimpleNamespace(usage=usage), SimpleNamespace(usage=usage)]
        )
        report = extract_usage(result)
        assert report["cached_tokens"] == 1536
        assert report["cache_write_tokens"] == 128

    def test_deepseek_top_level_cache_hit_is_read(self):
        usage = SimpleNamespace(
            prompt_tokens=500,
            completion_tokens=20,
            total_tokens=520,
            prompt_cache_hit_tokens=384,
        )
        report = extract_completion_usage(SimpleNamespace(usage=usage))
        assert report["cached_tokens"] == 384
        assert report["requests"] == 1

    def test_anthropic_style_cache_fields_are_read(self):
        usage = SimpleNamespace(
            input_tokens=900,
            output_tokens=10,
            prompt_tokens_details=None,
            cache_read_input_tokens=600,
            cache_creation_input_tokens=100,
        )
        result = SimpleNamespace(raw_responses=[SimpleNamespace(usage=usage)])
        report = extract_usage(result)
        assert report["cached_tokens"] == 600
        assert report["cache_write_tokens"] == 100

    def test_cached_tokens_clamped_to_input(self):
        usage = SimpleNamespace(
            prompt_tokens=100,
            completion_tokens=5,
            total_tokens=105,
            prompt_cache_hit_tokens=400,
        )
        report = extract_completion_usage(SimpleNamespace(usage=usage))
        assert report["cached_tokens"] == 100


class TestSharedEventLogger:
    def test_reuses_context_logger(self, tmp_path):
        context = ProjectContext(sandbox_dir=tmp_path, project_name="shared")
        first = EventLogger(context=context)
        context.event_logger = first
        assert shared_event_logger(context) is first

    def test_creates_and_backfills_when_missing(self, tmp_path):
        context = ProjectContext(sandbox_dir=tmp_path, project_name="shared")
        logger = shared_event_logger(context)
        assert isinstance(logger, EventLogger)
        assert context.event_logger is logger
        assert shared_event_logger(context) is logger

    def test_none_context_returns_new_logger(self):
        logger = shared_event_logger(None)
        assert isinstance(logger, EventLogger)

    def test_recorder_writes_to_shared_logger(self, tmp_path):
        context = ProjectContext(sandbox_dir=tmp_path, project_name="shared")
        events = EventLogger(context=context)
        context.event_logger = events
        usage = SimpleNamespace(input_tokens=3, output_tokens=1, total_tokens=4)
        recorder = make_usage_recorder(context, "subagent_creator")
        recorder(
            SimpleNamespace(
                raw_responses=[SimpleNamespace(usage=usage)],
            )
        )
        records = events.get_events(event_type="token_usage")
        assert len(records) == 1
        assert records[0]["payload"]["agent"] == "subagent_creator"
        assert records[0]["payload"]["total_tokens"] == 4
        assert context.event_log[-1]["event_type"] == "token_usage"


class TestCreatorStandardRoutingGuard:
    @pytest.fixture
    def context(self, tmp_path):
        return ProjectContext(sandbox_dir=tmp_path, project_name="routing")

    @staticmethod
    def _payload(spec: SubtaskSpec, grant: ToolGrant) -> str:
        return json.dumps(
            {
                "subtask_spec_json": spec.model_dump_json(),
                "tool_grant_json": grant.model_dump_json(),
            }
        )

    @staticmethod
    def _spec(allowed_tools: list[str]) -> SubtaskSpec:
        return SubtaskSpec(
            subtask_id="s1",
            title="演示子任务",
            instructions="读取沙箱文件",
            role="code",
            allowed_tools=allowed_tools,
        )

    @staticmethod
    def _grant(allowed_tools: list[str]) -> ToolGrant:
        return ToolGrant(
            grant_id="g1",
            task_id="t1",
            agent_id="t1:s1:grant-slot",
            allowed_tools=allowed_tools,
            issued_by=DEFAULT_SUPERVISOR_ID,
        )

    @pytest.fixture
    def strict_env(self, context):
        settings = Settings(max_dynamic_agents=8, _env_file=None)
        registry = AgentRegistry(context=context, settings=settings)
        tools = creator_mod.build_creator_tools(
            context,
            settings=settings,
            registry=registry,
            subagent_model="test-model",
            enforce_standard_routing=True,
        )
        return tools, registry

    @pytest.fixture
    def lenient_env(self, context):
        settings = Settings(max_dynamic_agents=8, _env_file=None)
        registry = AgentRegistry(context=context, settings=settings)
        tools = creator_mod.build_creator_tools(
            context,
            settings=settings,
            registry=registry,
            subagent_model="test-model",
        )
        return tools, registry

    async def test_strict_guard_rejects_standard_tools(self, strict_env):
        tools, registry = strict_env
        raw = await tools["create_subagent"].on_invoke_tool(
            make_ctx("create_subagent"),
            self._payload(
                self._spec(["read_file", "write_file"]),
                self._grant(["read_file", "write_file"]),
            ),
        )
        payload = json.loads(raw)
        assert payload["status"] == "error"
        assert payload["code"] == REASON_STANDARD_TOOLS_ONLY
        assert payload["subtask_id"] == "s1"
        assert "code_agent" in payload["error"]
        assert registry.list_by_task("t1") == []

    async def test_strict_guard_rejects_empty_tools(self, strict_env):
        tools, registry = strict_env
        raw = await tools["create_subagent"].on_invoke_tool(
            make_ctx("create_subagent"),
            self._payload(self._spec([]), self._grant([])),
        )
        assert json.loads(raw)["code"] == REASON_STANDARD_TOOLS_ONLY
        assert registry.list_by_task("t1") == []

    async def test_strict_guard_allows_non_standard_tools(self, strict_env):
        tools, registry = strict_env
        raw = await tools["create_subagent"].on_invoke_tool(
            make_ctx("create_subagent"),
            self._payload(
                self._spec(["domain_tool"]), self._grant(["domain_tool"])
            ),
        )
        payload = json.loads(raw)
        assert payload.get("status") != "error"
        assert payload["ignored_tools"] == ["domain_tool"]
        assert len(registry.list_by_task("t1")) == 1

    async def test_lenient_default_still_creates_for_standard_tools(
        self, lenient_env
    ):
        tools, registry = lenient_env
        raw = await tools["create_subagent"].on_invoke_tool(
            make_ctx("create_subagent"),
            self._payload(self._spec(["read_file"]), self._grant(["read_file"])),
        )
        payload = json.loads(raw)
        assert payload.get("status") != "error"
        assert payload["injected_tools"] == ["read_file"]
        assert len(registry.list_by_task("t1")) == 1

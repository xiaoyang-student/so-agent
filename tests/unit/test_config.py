"""单元测试：全局运行配置（pydantic-settings）。"""

from __future__ import annotations

import pytest

from so_agent.config import Settings, get_settings

SO_AGENT_ENV_KEYS = [
    "SO_AGENT_MAX_DYNAMIC_AGENTS",
    "SO_AGENT_MAX_AGENT_ATTEMPTS",
    "SO_AGENT_MAX_REPLACEMENT_REQUESTS",
    "SO_AGENT_MAX_CONCURRENCY",
    "SO_AGENT_MAX_REPLAN_ATTEMPTS",
    "SO_AGENT_MAX_MODEL_TURNS",
    "SO_AGENT_MODEL_TIMEOUT",
    "SO_AGENT_SANDBOX_TIMEOUT",
    "SO_AGENT_SANDBOX_OUTPUT_LIMIT",
    "SO_AGENT_MAX_PIPELINE_TIMEOUT",
    "SO_AGENT_MAX_SUPERVISOR_CALL_TIMEOUT",
    "SO_AGENT_MAX_IDLE_TIMEOUT",
    "SO_AGENT_FIRST_EVENT_TIMEOUT",
    "SO_AGENT_MODEL_IDLE_TIMEOUT",
    "SO_AGENT_TOOL_EXECUTION_TIMEOUT",
    "SO_AGENT_AGENT_TOTAL_TIMEOUT",
    "SO_AGENT_SUBAGENT_TOOL_EXECUTION_TIMEOUT",
    "SO_AGENT_SUPERVISOR_TOTAL_TIMEOUT",
    "SO_AGENT_CLASSIFIER_MAX_TOKENS",
    "SO_AGENT_ENABLE_ROUTER",
    "SO_AGENT_TASK_STORE_PATH",
    "SO_AGENT_SCHEDULER_POLL_INTERVAL",
]


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch, tmp_path):
    """隔离环境：清空 SO_AGENT_* 变量、切换到无 .env 的临时目录、清缓存。"""
    for key in SO_AGENT_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.chdir(tmp_path)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


class TestDefaults:
    """默认值必须与架构治理参数完全一致。"""

    def test_dynamic_agent_governance_defaults(self):
        settings = Settings()
        assert settings.max_dynamic_agents == 8
        assert settings.max_agent_attempts == 3
        assert settings.max_replacement_requests == 1

    def test_concurrency_and_replan_defaults(self):
        settings = Settings()
        assert settings.max_concurrency == 3
        assert settings.max_replan_attempts == 3
        assert settings.max_model_turns == 8

    def test_timeout_and_sandbox_defaults(self):
        settings = Settings()
        assert settings.model_timeout == 45.0
        assert settings.sandbox_timeout == 30.0
        assert settings.sandbox_output_limit == 1_048_576

    def test_agent_level_timeout_defaults(self):
        """Agent 级超时控制：流水线 40 分钟、单次主管调用 10 分钟、

        流式空闲超时 60 秒（三级关系 idle < supervisor < pipeline）。
        """
        settings = Settings()
        assert settings.max_idle_timeout == 60.0
        assert settings.max_supervisor_call_timeout == 600.0
        assert settings.max_pipeline_timeout == 2400.0
        assert (
            settings.max_idle_timeout
            < settings.max_supervisor_call_timeout
            < settings.max_pipeline_timeout
        )

    def test_staged_stream_timeout_defaults(self):
        """分阶段流式超时预算默认值（任务 33 含嵌套修复）。

        阶段窗口（首事件 / 模型空闲 / 工具执行）均小于 Agent 总预算；
        嵌套子 Agent 工具窗口覆盖子 Agent 总预算与清理宽限
        （600 + 10 = 610）；主管总预算保底不短于该窗口（900），
        且整体不超过流水线上限（2400）。
        """
        settings = Settings()
        assert settings.first_event_timeout == 180.0
        assert settings.model_idle_timeout == 90.0
        assert settings.tool_execution_timeout == 180.0
        assert settings.agent_total_timeout == 600.0
        assert settings.subagent_tool_execution_timeout == 610.0
        assert settings.supervisor_total_timeout == 900.0
        assert settings.max_pipeline_timeout == 2400.0
        assert settings.first_event_timeout < settings.agent_total_timeout
        assert settings.model_idle_timeout < settings.agent_total_timeout
        assert settings.tool_execution_timeout < settings.agent_total_timeout
        assert settings.agent_total_timeout < settings.subagent_tool_execution_timeout
        assert (
            settings.subagent_tool_execution_timeout
            <= settings.supervisor_total_timeout
        )
        assert settings.supervisor_total_timeout < settings.max_pipeline_timeout

    def test_legacy_timeout_fields_keep_legacy_defaults(self):
        """旧字段保留原默认值（deprecated 兼容，不再驱动运行时窗口）。"""
        settings = Settings()
        assert settings.max_idle_timeout == 60.0
        assert settings.max_supervisor_call_timeout == 600.0

    def test_router_and_classifier_defaults(self):
        """\u56db\u8c61\u9650\u8def\u7531\u67b6\u6784\u65b0\u589e\u914d\u7f6e\u5b57\u6bb5\u9ed8\u8ba4\u503c\u3002"""
        s = Settings()
        assert s.classifier_max_tokens == 512
        assert s.enable_router is True
        assert s.task_store_path == "sandbox/task_store.db"
        assert s.scheduler_poll_interval == 60

    def test_field_types(self):
        settings = Settings()
        assert isinstance(settings.max_concurrency, int)
        assert isinstance(settings.model_timeout, float)
        assert isinstance(settings.sandbox_output_limit, int)


class TestEnvOverride:
    """环境变量（前缀 SO_AGENT_）覆盖默认值。"""

    def test_int_override(self, monkeypatch):
        monkeypatch.setenv("SO_AGENT_MAX_CONCURRENCY", "5")
        assert Settings().max_concurrency == 5

    def test_float_override(self, monkeypatch):
        monkeypatch.setenv("SO_AGENT_MODEL_TIMEOUT", "12.5")
        assert Settings().model_timeout == 12.5

    def test_multiple_overrides(self, monkeypatch):
        monkeypatch.setenv("SO_AGENT_MAX_DYNAMIC_AGENTS", "2")
        monkeypatch.setenv("SO_AGENT_MAX_REPLAN_ATTEMPTS", "1")
        monkeypatch.setenv("SO_AGENT_SANDBOX_OUTPUT_LIMIT", "4096")
        settings = Settings()
        assert settings.max_dynamic_agents == 2
        assert settings.max_replan_attempts == 1
        assert settings.sandbox_output_limit == 4096

    def test_unknown_env_key_ignored(self, monkeypatch):
        monkeypatch.setenv("SO_AGENT_UNKNOWN_OPTION", "whatever")
        assert Settings().max_concurrency == 3

    def test_invalid_value_raises(self, monkeypatch):
        monkeypatch.setenv("SO_AGENT_MAX_CONCURRENCY", "not-a-number")
        with pytest.raises(ValueError):
            Settings()

    def test_staged_timeout_env_override(self, monkeypatch):
        monkeypatch.setenv("SO_AGENT_FIRST_EVENT_TIMEOUT", "150")
        monkeypatch.setenv("SO_AGENT_MODEL_IDLE_TIMEOUT", "70")
        monkeypatch.setenv("SO_AGENT_TOOL_EXECUTION_TIMEOUT", "200")
        monkeypatch.setenv("SO_AGENT_AGENT_TOTAL_TIMEOUT", "500")
        settings = Settings()
        assert settings.first_event_timeout == 150.0
        assert settings.model_idle_timeout == 70.0
        assert settings.tool_execution_timeout == 200.0
        assert settings.agent_total_timeout == 500.0

    def test_nested_subagent_timeout_env_override(self, monkeypatch):
        """嵌套修复：子 Agent 工具窗口与主管总预算可经环境变量覆盖。"""
        monkeypatch.setenv("SO_AGENT_SUBAGENT_TOOL_EXECUTION_TIMEOUT", "700")
        monkeypatch.setenv("SO_AGENT_SUPERVISOR_TOTAL_TIMEOUT", "1000")
        settings = Settings()
        assert settings.subagent_tool_execution_timeout == 700.0
        assert settings.supervisor_total_timeout == 1000.0

    def test_supervisor_env_below_subagent_window_lifted(self, monkeypatch):
        """嵌套修复：显式主管总预算低于子 Agent 工具窗口时被保底提升。"""
        monkeypatch.setenv("SO_AGENT_SUPERVISOR_TOTAL_TIMEOUT", "400")
        settings = Settings()
        # 子 Agent 工具窗口按默认 agent_total(600) 派生为 610
        assert settings.subagent_tool_execution_timeout == 610.0
        assert settings.supervisor_total_timeout == 610.0


class TestDeprecatedTimeoutAliases:
    """旧超时字段显式设置时映射到新字段（deprecated 兼容，任务 33）。"""

    def test_legacy_idle_maps_to_first_event_and_model(self):
        s = Settings(max_idle_timeout=45.0)
        assert s.first_event_timeout == 45.0
        assert s.model_idle_timeout == 45.0
        # 工具执行阶段不沿用旧窗口（修复旧的固定空闲超时误杀工具执行）
        assert s.tool_execution_timeout == 180.0

    def test_legacy_supervisor_call_maps_to_agent_total(self):
        s = Settings(max_supervisor_call_timeout=700.0)
        assert s.agent_total_timeout == 700.0

    def test_explicit_new_field_beats_legacy(self):
        s = Settings(max_idle_timeout=45.0, first_event_timeout=30.0)
        assert s.first_event_timeout == 30.0
        assert s.model_idle_timeout == 45.0  # 未显式设置的新字段仍接受映射

        s2 = Settings(
            max_supervisor_call_timeout=700.0, agent_total_timeout=800.0
        )
        assert s2.agent_total_timeout == 800.0

    def test_defaults_not_affected_by_unset_legacy(self):
        # 未显式设置旧字段时不得触发映射（默认值不覆盖新默认值）
        s = Settings()
        assert s.first_event_timeout == 180.0
        assert s.model_idle_timeout == 90.0
        assert s.agent_total_timeout == 600.0

    def test_env_legacy_alias_applies(self, monkeypatch):
        monkeypatch.setenv("SO_AGENT_MAX_IDLE_TIMEOUT", "30")
        s = Settings()
        assert s.first_event_timeout == 30.0
        assert s.model_idle_timeout == 30.0
        assert s.tool_execution_timeout == 180.0

    def test_env_new_field_beats_legacy_env(self, monkeypatch):
        monkeypatch.setenv("SO_AGENT_MAX_IDLE_TIMEOUT", "30")
        monkeypatch.setenv("SO_AGENT_MODEL_IDLE_TIMEOUT", "70")
        s = Settings()
        assert s.first_event_timeout == 30.0
        assert s.model_idle_timeout == 70.0

    def test_env_legacy_supervisor_call_alias(self, monkeypatch):
        monkeypatch.setenv("SO_AGENT_MAX_SUPERVISOR_CALL_TIMEOUT", "720")
        s = Settings()
        assert s.agent_total_timeout == 720.0

    def test_subagent_window_derives_from_agent_total(self):
        """嵌套修复：子 Agent 工具窗口未显式设置时按 agent_total + 10 秒派生。"""
        s = Settings(agent_total_timeout=300.0)
        assert s.subagent_tool_execution_timeout == 310.0
        # 主管总预算默认 900 不短于派生窗口，无需提升
        assert s.supervisor_total_timeout == 900.0

    def test_legacy_supervisor_alias_derives_subagent_window(self):
        """旧字段映射后，子 Agent 工具窗口基于映射结果派生。"""
        s = Settings(max_supervisor_call_timeout=300.0)
        assert s.agent_total_timeout == 300.0
        assert s.subagent_tool_execution_timeout == 310.0
        assert s.supervisor_total_timeout == 900.0

    def test_explicit_subagent_window_respected(self):
        """显式设置子 Agent 工具窗口时不被派生逻辑覆盖。"""
        s = Settings(subagent_tool_execution_timeout=750.0)
        assert s.subagent_tool_execution_timeout == 750.0
        assert s.supervisor_total_timeout == 900.0

    def test_supervisor_budget_floor_not_short_of_subagent_window(self):
        """嵌套修复：主管总预算保底不低于子 Agent 工具窗口。"""
        # 显式主管预算低于默认派生窗口 → 提升到 610
        s = Settings(supervisor_total_timeout=500.0)
        assert s.subagent_tool_execution_timeout == 610.0
        assert s.supervisor_total_timeout == 610.0

        # 大子 Agent 预算（1200）→ 派生 1210，主管预算提升到 1210
        s2 = Settings(agent_total_timeout=1200.0)
        assert s2.subagent_tool_execution_timeout == 1210.0
        assert s2.supervisor_total_timeout == 1210.0
        # 流水线总上限不变
        assert s2.max_pipeline_timeout == 2400.0


class TestGetSettingsSingleton:
    """get_settings() 为进程内单例，且返回 Settings 实例。"""

    def test_returns_settings_instance(self):
        assert isinstance(get_settings(), Settings)

    def test_cached_singleton(self):
        assert get_settings() is get_settings()

    def test_cache_clear_recreates(self):
        first = get_settings()
        get_settings.cache_clear()
        second = get_settings()
        assert first is not second

    def test_env_change_after_cache_not_visible(self, monkeypatch):
        first = get_settings()
        monkeypatch.setenv("SO_AGENT_MAX_CONCURRENCY", "9")
        assert get_settings() is first
        assert get_settings().max_concurrency == 3

    def test_cache_clear_picks_up_env(self, monkeypatch):
        monkeypatch.setenv("SO_AGENT_MAX_CONCURRENCY", "9")
        get_settings.cache_clear()
        assert get_settings().max_concurrency == 9

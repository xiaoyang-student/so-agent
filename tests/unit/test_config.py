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
        assert settings.max_model_turns == 10

    def test_timeout_and_sandbox_defaults(self):
        settings = Settings()
        assert settings.model_timeout == 60.0
        assert settings.sandbox_timeout == 30.0
        assert settings.sandbox_output_limit == 1_048_576

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

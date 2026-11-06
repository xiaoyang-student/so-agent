"""ModelGateway 故障转移测试（假客户端，不发起真实请求）。"""

from __future__ import annotations

from types import SimpleNamespace

import httpx2
import pytest
import yaml
from openai import APIConnectionError

from so_agent.runtime import model_gateway as gw
from so_agent.runtime.config_loader import AgentAPIConfig, ConfigLoadError


class FakeCompletions:
    def __init__(self, client: "FakeClient") -> None:
        self._client = client

    async def create(self, **kwargs):
        self._client.calls.append(kwargs)
        if not self._client.outcomes:
            raise AssertionError("FakeClient 没有剩余结果")
        outcome = self._client.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


class FakeClient:
    def __init__(self, base_url: str, outcomes: list) -> None:
        self.base_url = base_url
        self.outcomes = list(outcomes)
        self.calls: list[dict] = []
        self.closed = False
        self.chat = SimpleNamespace(completions=FakeCompletions(self))

    async def close(self) -> None:
        self.closed = True

    def with_options(self, **options):
        return self


class FakeBadRequest(Exception):
    status_code = 400


class FakeServerError(Exception):
    status_code = 503


def _connection_error() -> APIConnectionError:
    return APIConnectionError(
        request=httpx2.Request("POST", "https://primary.example/v1/chat/completions")
    )


def _primary_config() -> AgentAPIConfig:
    return AgentAPIConfig(
        provider="primary",
        model="primary-model",
        base_url="https://primary.example/v1",
        api_key="primary-key",
        timeout=5,
        retry=0,
    )


def _write_chain_config(tmp_path, *, duplicate_primary: bool = False) -> tuple:
    path = tmp_path / "chain.yaml"
    fallbacks = [
        {
            "provider": "backup",
            "model": "backup-model",
            "base_url": "https://backup.example/v1",
            "api_key": "backup-key",
            "timeout": 7,
            "retry": 0,
        }
    ]
    if duplicate_primary:
        fallbacks.append(
            {
                "provider": "primary",
                "model": "primary-model",
                "base_url": "https://primary.example/v1",
                "api_key": "primary-key",
            }
        )
    path.write_text(
        yaml.safe_dump(
            {
                "provider": "primary",
                "model": "primary-model",
                "base_url": "https://primary.example/v1",
                "api_key": "primary-key",
                "timeout": 5,
                "retry": 0,
                "fallbacks": fallbacks,
                "failover": {
                    "failure_threshold": 2,
                    "cooldown_seconds": 30,
                },
            },
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    return path, AgentAPIConfig.model_validate(yaml.safe_load(path.read_text("utf-8")))


def _factory(clients: dict[str, FakeClient]):
    def factory(**kwargs):
        base_url = kwargs["base_url"]
        return clients[base_url]

    return factory


@pytest.fixture(autouse=True)
def _reset_gateway():
    gw.reset_gateway_state()
    yield
    gw.reset_gateway_state()


def test_single_provider_returns_plain_client():
    captured = {}

    def factory(**kwargs):
        captured.update(kwargs)
        return FakeClient(kwargs["base_url"], [SimpleNamespace(ok=True)])

    client = gw.build_openai_client(_primary_config(), client_factory=factory)
    assert isinstance(client, FakeClient)
    assert captured["api_key"] == "primary-key"
    assert captured["max_retries"] == 0


def test_single_provider_keeps_shared_http_timeout_object():
    marker = object()
    captured = {}

    def factory(**kwargs):
        captured.update(kwargs)
        return FakeClient(kwargs["base_url"], [SimpleNamespace(ok=True)])

    gw.build_openai_client(
        _primary_config(), client_factory=factory, http_timeout=marker
    )
    assert captured["timeout"] is marker


async def test_failover_to_backup_on_connection_error(tmp_path):
    path, primary = _write_chain_config(tmp_path)
    primary_client = FakeClient(
        "https://primary.example/v1", [_connection_error()]
    )
    backup_client = FakeClient("https://backup.example/v1", [SimpleNamespace(ok="backup")])
    client = gw.build_openai_client(
        primary,
        config_path=path,
        client_factory=_factory(
            {
                "https://primary.example/v1": primary_client,
                "https://backup.example/v1": backup_client,
            }
        ),
    )
    result = await client.chat.completions.create(messages=[], model="ignored")
    assert result.ok == "backup"
    assert backup_client.calls[0]["model"] == "backup-model"
    assert gw.gateway_snapshot()["failovers"] == 1


async def test_non_failover_error_is_raised_without_switch(tmp_path):
    path, primary = _write_chain_config(tmp_path)
    primary_client = FakeClient(
        "https://primary.example/v1", [FakeBadRequest("bad")]
    )
    backup_client = FakeClient("https://backup.example/v1", [SimpleNamespace(ok="backup")])
    client = gw.build_openai_client(
        primary,
        config_path=path,
        client_factory=_factory(
            {
                "https://primary.example/v1": primary_client,
                "https://backup.example/v1": backup_client,
            }
        ),
    )
    with pytest.raises(FakeBadRequest):
        await client.chat.completions.create(messages=[])
    assert backup_client.calls == []
    assert gw.gateway_snapshot()["providers"] == []


async def test_streaming_call_fails_over_before_first_event(tmp_path):
    path, primary = _write_chain_config(tmp_path)
    primary_client = FakeClient(
        "https://primary.example/v1", [_connection_error()]
    )
    backup_stream = SimpleNamespace(stream=True)
    backup_client = FakeClient("https://backup.example/v1", [backup_stream])
    client = gw.build_openai_client(
        primary,
        config_path=path,
        client_factory=_factory(
            {
                "https://primary.example/v1": primary_client,
                "https://backup.example/v1": backup_client,
            }
        ),
    )
    result = await client.chat.completions.create(stream=True, messages=[])
    assert result is backup_stream


async def test_request_timeout_is_rewritten_per_provider(tmp_path):
    path, primary = _write_chain_config(tmp_path)
    primary_client = FakeClient(
        "https://primary.example/v1", [FakeServerError("down")]
    )
    backup_client = FakeClient("https://backup.example/v1", [SimpleNamespace(ok=True)])
    client = gw.build_openai_client(
        primary,
        config_path=path,
        client_factory=_factory(
            {
                "https://primary.example/v1": primary_client,
                "https://backup.example/v1": backup_client,
            }
        ),
    )
    await client.chat.completions.create(messages=[], timeout=1.0)
    assert backup_client.calls[0]["timeout"] == 7.0


async def test_cooldown_skips_failing_primary(tmp_path, monkeypatch):
    path, primary = _write_chain_config(tmp_path)
    raw = yaml.safe_load(path.read_text("utf-8"))
    raw["failover"] = {"failure_threshold": 1, "cooldown_seconds": 60}
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    primary_client = FakeClient(
        "https://primary.example/v1",
        [_connection_error(), SimpleNamespace(ok="primary")],
    )
    backup_client = FakeClient(
        "https://backup.example/v1",
        [SimpleNamespace(ok="backup-1"), SimpleNamespace(ok="backup-2")],
    )
    client = gw.build_openai_client(
        primary,
        config_path=path,
        client_factory=_factory(
            {
                "https://primary.example/v1": primary_client,
                "https://backup.example/v1": backup_client,
            }
        ),
    )
    first = await client.chat.completions.create(messages=[])
    second = await client.chat.completions.create(messages=[])
    assert first.ok == "backup-1"
    assert second.ok == "backup-2"
    assert len(primary_client.calls) == 1


async def test_cooldown_expires_and_primary_is_retried(tmp_path, monkeypatch):
    path, primary = _write_chain_config(tmp_path)
    raw = yaml.safe_load(path.read_text("utf-8"))
    raw["failover"] = {"failure_threshold": 1, "cooldown_seconds": 10}
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    primary_client = FakeClient(
        "https://primary.example/v1",
        [_connection_error(), SimpleNamespace(ok="primary-again")],
    )
    backup_client = FakeClient("https://backup.example/v1", [SimpleNamespace(ok="backup")])
    client = gw.build_openai_client(
        primary,
        config_path=path,
        client_factory=_factory(
            {
                "https://primary.example/v1": primary_client,
                "https://backup.example/v1": backup_client,
            }
        ),
    )
    clock = [100.0]
    monkeypatch.setattr(gw.time, "monotonic", lambda: clock[0])
    await client.chat.completions.create(messages=[])
    clock[0] += 11
    result = await client.chat.completions.create(messages=[])
    assert result.ok == "primary-again"
    assert len(primary_client.calls) == 2


def test_duplicate_primary_fallback_is_deduped(tmp_path):
    path, primary = _write_chain_config(tmp_path, duplicate_primary=True)
    configs, settings = gw.load_fallback_configs(path, primary=primary)
    assert [c.model for c in configs] == ["backup-model"]
    assert settings.failure_threshold == 2
    assert settings.cooldown_seconds == 30.0


def test_invalid_fallback_entry_fails_closed(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "fallbacks": [
                    {
                        "provider": "backup",
                        "base_url": "https://backup.example/v1",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ConfigLoadError):
        gw.load_fallback_configs(path)


def test_missing_fallback_key_fails_closed(tmp_path, monkeypatch):
    path = tmp_path / "keyless.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "fallbacks": [
                    {
                        "provider": "backup",
                        "model": "backup-model",
                        "base_url": "https://backup.example/v1",
                        "api_key_env": "SO_AGENT_MISSING_BACKUP_KEY",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.delenv("SO_AGENT_MISSING_BACKUP_KEY", raising=False)
    with pytest.raises(ConfigLoadError):
        gw.build_openai_client(
            _primary_config(),
            config_path=path,
            client_factory=lambda **kwargs: FakeClient(kwargs["base_url"], []),
        )


async def test_with_options_and_close_delegate_to_all_providers(tmp_path):
    path, primary = _write_chain_config(tmp_path)
    primary_client = FakeClient("https://primary.example/v1", [])
    backup_client = FakeClient("https://backup.example/v1", [])
    client = gw.build_openai_client(
        primary,
        config_path=path,
        client_factory=_factory(
            {
                "https://primary.example/v1": primary_client,
                "https://backup.example/v1": backup_client,
            }
        ),
    )
    assert client.with_options(max_retries=0) is not client
    await client.close()
    assert primary_client.closed and backup_client.closed


def test_gateway_snapshot_excludes_keys(tmp_path):
    path, primary = _write_chain_config(tmp_path)
    gw.build_openai_client(
        primary,
        config_path=path,
        client_factory=lambda **kwargs: FakeClient(kwargs["base_url"], []),
    )
    snapshot = gw.gateway_snapshot()
    assert "primary-key" not in str(snapshot)
    assert "backup-key" not in str(snapshot)

"""单元测试：api_config.yaml 配置加载器。"""

from __future__ import annotations

import importlib
import json
from importlib.resources import files
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from agents import OpenAIChatCompletionsModel

import so_agent.orchestrator as orchestrator_mod
from so_agent.config import Settings
from so_agent.context import ProjectContext
from so_agent.mcp.client import MCPClient
from so_agent.runtime.config_loader import (
    REQUIRED_FIELDS,
    AgentAPIConfig,
    ConfigLoadError,
    discover_tool_package_configs,
    load_agent_api_config,
)
from so_agent.runtime.registry import AgentRegistry
from so_agent.runtime.tool_registry import ToolRegistry

VALID_YAML = """
provider: openai
model: gpt-5
base_url: https://api.openai.com/v1
timeout: 120
retry: 2
api_key_env: TEST_AGENT_API_KEY
"""

MINIMAL_YAML = """
provider: openai
model: gpt-5
api_key_env: TEST_AGENT_API_KEY
"""

DIRECT_KEY_YAML = """
provider: openai
model: gpt-5
api_key: sk-direct-value
"""


def write_yaml(tmp_path: Path, text: str, name: str = "api_config.yaml") -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


class TestLoadValidConfig:
    def test_full_config(self, tmp_path):
        config = load_agent_api_config(write_yaml(tmp_path, VALID_YAML))
        assert isinstance(config, AgentAPIConfig)
        assert config.provider == "openai"
        assert config.model == "gpt-5"
        assert config.base_url == "https://api.openai.com/v1"
        assert config.timeout == 120.0
        assert config.retry == 2
        assert config.api_key_env == "TEST_AGENT_API_KEY"

    def test_minimal_config_defaults(self, tmp_path):
        config = load_agent_api_config(write_yaml(tmp_path, MINIMAL_YAML))
        assert config.base_url is None
        assert config.timeout == 60.0
        assert config.retry == 2
        assert config.api_key is None

    def test_no_secret_resolution_by_default(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TEST_AGENT_API_KEY", "sk-super-secret")
        config = load_agent_api_config(write_yaml(tmp_path, VALID_YAML))
        assert config.api_key is None  # 默认不解析密钥


class TestLoadErrors:
    def test_missing_file(self, tmp_path):
        with pytest.raises(ConfigLoadError):
            load_agent_api_config(tmp_path / "nope.yaml")

    def test_invalid_yaml(self, tmp_path):
        with pytest.raises(ConfigLoadError):
            load_agent_api_config(write_yaml(tmp_path, "provider: [unclosed"))

    def test_top_level_not_mapping(self, tmp_path):
        with pytest.raises(ConfigLoadError):
            load_agent_api_config(write_yaml(tmp_path, "- a\n- b\n"))

    def test_empty_file_rejected(self, tmp_path):
        with pytest.raises(ConfigLoadError):
            load_agent_api_config(write_yaml(tmp_path, ""))

    @pytest.mark.parametrize("field", REQUIRED_FIELDS)
    def test_each_required_field_missing(self, tmp_path, field):
        lines = [
            line
            for line in MINIMAL_YAML.strip().splitlines()
            if not line.startswith(f"{field}:")
        ]
        path = write_yaml(tmp_path, "\n".join(lines) + "\n")
        with pytest.raises(ConfigLoadError):
            load_agent_api_config(path)

    def test_blank_required_value_rejected(self, tmp_path):
        text = MINIMAL_YAML.replace("model: gpt-5", 'model: "   "')
        with pytest.raises(ConfigLoadError):
            load_agent_api_config(write_yaml(tmp_path, text))

    def test_invalid_field_type_rejected(self, tmp_path):
        text = MINIMAL_YAML.replace("model: gpt-5", "model: [1, 2]")
        with pytest.raises(ConfigLoadError):
            load_agent_api_config(write_yaml(tmp_path, text))

    def test_error_message_contains_path_and_missing_fields(self, tmp_path):
        path = write_yaml(tmp_path, "provider: openai\n")
        with pytest.raises(ConfigLoadError) as excinfo:
            load_agent_api_config(path)
        message = str(excinfo.value)
        assert str(path) in message
        assert "model" in message
        assert "必填字段" in message

    def test_missing_secret_pair_rejected(self, tmp_path):
        """api_key（直接值）与 api_key_env（环境变量名）均缺失时应拒绝加载。"""
        path = write_yaml(tmp_path, "provider: openai\nmodel: gpt-5\n")
        with pytest.raises(ConfigLoadError) as excinfo:
            load_agent_api_config(path)
        message = str(excinfo.value)
        assert str(path) in message
        assert "api_key" in message
        assert "api_key_env" in message


class TestResolveApiKey:
    def test_resolve_from_env(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TEST_AGENT_API_KEY", "sk-abc123")
        config = load_agent_api_config(write_yaml(tmp_path, VALID_YAML))
        assert config.resolve_api_key() == "sk-abc123"

    def test_resolve_from_direct_key(self, tmp_path):
        config = load_agent_api_config(write_yaml(tmp_path, DIRECT_KEY_YAML))
        assert config.api_key == "sk-direct-value"
        assert config.resolve_api_key() == "sk-direct-value"

    def test_direct_key_priority_over_env(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TEST_AGENT_API_KEY", "sk-from-env")
        text = DIRECT_KEY_YAML + "api_key_env: TEST_AGENT_API_KEY\n"
        config = load_agent_api_config(write_yaml(tmp_path, text))
        assert config.resolve_api_key() == "sk-direct-value"

    def test_resolve_secrets_prefers_direct_key(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TEST_AGENT_API_KEY", "sk-from-env")
        text = DIRECT_KEY_YAML + "api_key_env: TEST_AGENT_API_KEY\n"
        config = load_agent_api_config(
            write_yaml(tmp_path, text), resolve_secrets=True
        )
        assert config.api_key == "sk-direct-value"

    def test_resolve_strips_whitespace(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TEST_AGENT_API_KEY", "  sk-padded  ")
        config = load_agent_api_config(write_yaml(tmp_path, VALID_YAML))
        assert config.resolve_api_key() == "sk-padded"

    def test_missing_env_raises(self, tmp_path, monkeypatch):
        monkeypatch.delenv("TEST_AGENT_API_KEY", raising=False)
        config = load_agent_api_config(write_yaml(tmp_path, VALID_YAML))
        with pytest.raises(ConfigLoadError):
            config.resolve_api_key()

    def test_blank_env_raises(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TEST_AGENT_API_KEY", "   ")
        config = load_agent_api_config(write_yaml(tmp_path, VALID_YAML))
        with pytest.raises(ConfigLoadError):
            config.resolve_api_key()

    def test_resolve_secrets_flag(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TEST_AGENT_API_KEY", "sk-flag")
        config = load_agent_api_config(
            write_yaml(tmp_path, VALID_YAML), resolve_secrets=True
        )
        assert config.api_key == "sk-flag"

    def test_resolve_secrets_missing_env_raises(self, tmp_path, monkeypatch):
        monkeypatch.delenv("TEST_AGENT_API_KEY", raising=False)
        with pytest.raises(ConfigLoadError):
            load_agent_api_config(
                write_yaml(tmp_path, VALID_YAML), resolve_secrets=True
            )

    def test_error_message_mentions_env_name_not_secret(self, tmp_path, monkeypatch):
        monkeypatch.delenv("TEST_AGENT_API_KEY", raising=False)
        config = load_agent_api_config(write_yaml(tmp_path, VALID_YAML))
        with pytest.raises(ConfigLoadError) as excinfo:
            config.resolve_api_key()
        assert "TEST_AGENT_API_KEY" in str(excinfo.value)


class TestSecretNeverSerialized:
    def test_api_key_excluded_from_model_dump(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TEST_AGENT_API_KEY", "sk-top-secret-value")
        config = load_agent_api_config(
            write_yaml(tmp_path, VALID_YAML), resolve_secrets=True
        )
        assert config.api_key == "sk-top-secret-value"

        dumped = config.model_dump()
        assert "api_key" not in dumped
        assert "api_key_env" in dumped  # 仅保存变量名

    def test_api_key_excluded_from_json(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TEST_AGENT_API_KEY", "sk-top-secret-value")
        config = load_agent_api_config(
            write_yaml(tmp_path, VALID_YAML), resolve_secrets=True
        )
        payload = config.model_dump_json()
        assert "sk-top-secret-value" not in payload
        assert "api_key_env" in payload

    def test_literal_api_key_excluded_from_dump_and_json(self, tmp_path):
        config = load_agent_api_config(write_yaml(tmp_path, DIRECT_KEY_YAML))
        assert config.api_key == "sk-direct-value"
        assert "api_key" not in config.model_dump()
        assert "sk-direct-value" not in config.model_dump_json()

    def test_yaml_file_contains_only_env_name(self, tmp_path):
        text = VALID_YAML
        assert "api_key_env" in text
        assert "sk-" not in text  # 配置文件中不出现任何密钥值

    def test_dump_json_roundtrip_loses_api_key(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TEST_AGENT_API_KEY", "sk-roundtrip")
        config = load_agent_api_config(
            write_yaml(tmp_path, VALID_YAML), resolve_secrets=True
        )
        restored = AgentAPIConfig.model_validate_json(config.model_dump_json())
        assert restored.api_key is None


class TestDiscoverToolPackageConfigs:
    def test_discovers_all_packages(self, tmp_path):
        for name in ("code_agent", "subagent_creator", "review_agent"):
            package = tmp_path / name
            package.mkdir()
            (package / "api_config.yaml").write_text(MINIMAL_YAML, encoding="utf-8")
        # 无配置文件的目录（如 generated_tools）应被跳过
        (tmp_path / "generated_tools").mkdir()

        configs = discover_tool_package_configs(tmp_path)
        assert set(configs) == {"code_agent", "subagent_creator", "review_agent"}
        assert all(isinstance(c, AgentAPIConfig) for c in configs.values())

    def test_missing_root_raises(self, tmp_path):
        with pytest.raises(ConfigLoadError):
            discover_tool_package_configs(tmp_path / "missing")

    def test_invalid_package_config_raises(self, tmp_path):
        package = tmp_path / "bad_package"
        package.mkdir()
        (package / "api_config.yaml").write_text("provider: openai\n", encoding="utf-8")
        with pytest.raises(ConfigLoadError):
            discover_tool_package_configs(tmp_path)

    def test_resolve_secrets_forwarded(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TEST_AGENT_API_KEY", "sk-discovered")
        package = tmp_path / "code_agent"
        package.mkdir()
        (package / "api_config.yaml").write_text(VALID_YAML, encoding="utf-8")
        configs = discover_tool_package_configs(tmp_path, resolve_secrets=True)
        assert configs["code_agent"].api_key == "sk-discovered"


class TestRealProjectConfigs:
    """真实项目中的三个工具包配置应可被发现并加载。"""

    def test_discover_real_tool_packages(self):
        packages_dir = (
            Path(__file__).resolve().parents[2]
            / "src"
            / "so_agent"
            / "tool_packages"
        )
        configs = discover_tool_package_configs(packages_dir)
        assert set(configs) == {"code_agent", "subagent_creator", "review_agent"}
        for config in configs.values():
            assert config.provider == "dashscope"
            assert config.model
            assert config.api_key  # api_key 直读方式：密钥值随配置读取
            assert config.api_key_env is None  # 不再依赖环境变量名

    def test_real_config_json_has_no_secret(self):
        config_path = (
            Path(__file__).resolve().parents[2]
            / "src"
            / "so_agent"
            / "tool_packages"
            / "code_agent"
            / "api_config.yaml"
        )
        config = load_agent_api_config(config_path)
        payload = json.loads(config.model_dump_json())
        assert "api_key" not in payload

    def test_supervisor_config_is_independent_package_resource(self):
        resource = files("so_agent").joinpath("api_config.yaml")
        assert resource.is_file()
        assert orchestrator_mod.API_CONFIG_PATH == (
            Path(orchestrator_mod.__file__).resolve().parent / "api_config.yaml"
        )
        config = load_agent_api_config(orchestrator_mod.API_CONFIG_PATH)
        dumped = config.model_dump()
        assert "api_key" not in dumped  # 密钥值始终排除在序列化之外
        assert dumped == {
            "provider": "dashscope",
            "model": "qwen3.8-max",
            "base_url": "https://maas.qianwenaiapi.com/compatible-mode/v1",
            "timeout": 300.0,
            "retry": 1,
            "api_key_env": None,
        }
        assert config.api_key  # api_key 直读方式：密钥值随配置读取
        packages = orchestrator_mod.API_CONFIG_PATH.parent / "tool_packages"
        assert {
            path.name for path in packages.iterdir()
            if path.is_dir() and not path.name.startswith(("_", "."))
        } == {"code_agent", "subagent_creator", "review_agent", "generated_tools"}


@pytest.fixture
def supervisor_config_setup(tmp_path, monkeypatch):
    """使用临时配置、假密钥与假客户端，保留真实模型及 MCP 装配逻辑。"""
    text = (
        VALID_YAML.replace("gpt-5", "supervisor-only")
        .replace("https://api.openai.com/v1", "https://supervisor.example/v1")
        .replace("timeout: 120", "timeout: 17")
        .replace("retry: 2", "retry: 4")
        .replace("TEST_AGENT_API_KEY", "TEST_SUPERVISOR_API_KEY")
    )
    path = write_yaml(tmp_path, text)
    monkeypatch.setattr(orchestrator_mod, "API_CONFIG_PATH", path)
    monkeypatch.setenv("TEST_SUPERVISOR_API_KEY", "supervisor-test-key")
    monkeypatch.setenv("TEST_AGENT_API_KEY", "tool-test-key")
    clients = {"supervisor": Mock()}
    monkeypatch.setattr(orchestrator_mod, "AsyncOpenAI", clients["supervisor"])
    modules = {}
    builders = {}
    for name in orchestrator_mod.SUPERVISOR_TOOL_NAMES:
        module = importlib.import_module(f"so_agent.tool_packages.{name}.agent")
        modules[name] = module
        tool_path = write_yaml(
            tmp_path, VALID_YAML.replace("gpt-5", f"{name}-model"), f"{name}.yaml"
        )
        monkeypatch.setattr(module, "API_CONFIG_PATH", tool_path)
        clients[name] = Mock()
        monkeypatch.setattr(module, "AsyncOpenAI", clients[name])
        builders[name] = Mock(wraps=module._build_chat_model)
        monkeypatch.setattr(module, "_build_chat_model", builders[name])
    context = ProjectContext(
        sandbox_dir=tmp_path,
        project_name="supervisor-config",
        config=Settings(_env_file=None),
    )
    return SimpleNamespace(
        context=context, path=path, clients=clients, modules=modules, builders=builders
    )


class TestSupervisorAPIConfig:
    @pytest.mark.parametrize("factory_name", ["create_orchestrator", "get_orchestrator"])
    def test_uses_own_model_url_and_secret_env(
        self, supervisor_config_setup, factory_name
    ):
        setup = supervisor_config_setup
        agent = getattr(orchestrator_mod, factory_name)(setup.context)

        assert isinstance(agent.model, OpenAIChatCompletionsModel)
        assert agent.model.model == "supervisor-only"
        setup.clients["supervisor"].assert_called_once_with(
            api_key="supervisor-test-key",
            base_url="https://supervisor.example/v1",
            timeout=17.0,
            max_retries=4,
        )
        for name, builder in setup.builders.items():
            assert builder.call_args.args[0].model == f"{name}-model"
            setup.clients[name].assert_called_once_with(
                api_key="tool-test-key",
                base_url="https://api.openai.com/v1",
                timeout=120.0,
                max_retries=2,
            )

    def test_code_agent_config_changes_do_not_affect_supervisor(
        self, supervisor_config_setup, monkeypatch
    ):
        setup = supervisor_config_setup
        before = orchestrator_mod.create_orchestrator(setup.context)
        supervisor_call = setup.clients["supervisor"].call_args
        changed = (
            VALID_YAML.replace("gpt-5", "code-reconfigured")
            .replace("https://api.openai.com/v1", "https://code.example/v1")
            .replace("TEST_AGENT_API_KEY", "TEST_CHANGED_CODE_API_KEY")
        )
        setup.modules["code_agent"].API_CONFIG_PATH.write_text(changed, encoding="utf-8")
        monkeypatch.setenv("TEST_CHANGED_CODE_API_KEY", "changed-code-test-key")

        after = orchestrator_mod.create_orchestrator(setup.context)

        assert before.model.model == after.model.model == "supervisor-only"
        assert setup.clients["supervisor"].call_args_list == [supervisor_call] * 2
        assert setup.builders["code_agent"].call_args.args[0].model == "code-reconfigured"
        setup.clients["code_agent"].assert_called_with(
            api_key="changed-code-test-key",
            base_url="https://code.example/v1",
            timeout=120.0,
            max_retries=2,
        )

    def test_missing_supervisor_config_does_not_fall_back(
        self, supervisor_config_setup, monkeypatch
    ):
        setup = supervisor_config_setup
        missing = setup.path.parent / "missing-supervisor.yaml"
        monkeypatch.setattr(orchestrator_mod, "API_CONFIG_PATH", missing)

        with pytest.raises(ConfigLoadError, match="不存在") as excinfo:
            orchestrator_mod.create_orchestrator(setup.context)

        assert str(missing) in str(excinfo.value)
        for client in setup.clients.values():
            client.assert_not_called()

    @pytest.mark.parametrize("field", REQUIRED_FIELDS)
    def test_missing_supervisor_required_field_does_not_fall_back(
        self, supervisor_config_setup, field
    ):
        setup = supervisor_config_setup
        text = "\n".join(
            line for line in setup.path.read_text(encoding="utf-8").splitlines()
            if not line.startswith(f"{field}:")
        )
        setup.path.write_text(text, encoding="utf-8")

        with pytest.raises(ConfigLoadError, match="缺少必填字段") as excinfo:
            orchestrator_mod.create_orchestrator(setup.context)

        assert field in str(excinfo.value)
        assert str(setup.path) in str(excinfo.value)
        for client in setup.clients.values():
            client.assert_not_called()

    def test_missing_supervisor_secret_does_not_use_tool_secret(
        self, supervisor_config_setup, monkeypatch
    ):
        setup = supervisor_config_setup
        monkeypatch.delenv("TEST_SUPERVISOR_API_KEY")

        with pytest.raises(ConfigLoadError, match="TEST_SUPERVISOR_API_KEY"):
            orchestrator_mod.create_orchestrator(setup.context)

        for client in setup.clients.values():
            client.assert_not_called()

    @pytest.mark.parametrize("factory_name", ["create_orchestrator", "get_orchestrator"])
    @pytest.mark.parametrize("model_kind", ["name", "instance"])
    def test_explicit_model_override_skips_supervisor_config_only(
        self, supervisor_config_setup, monkeypatch, factory_name, model_kind
    ):
        setup = supervisor_config_setup
        monkeypatch.setattr(
            orchestrator_mod, "API_CONFIG_PATH", setup.path.parent / "missing.yaml"
        )
        override = (
            "explicit-model"
            if model_kind == "name"
            else OpenAIChatCompletionsModel(model="explicit-model", openai_client=Mock())
        )

        agent = getattr(orchestrator_mod, factory_name)(setup.context, model=override)

        assert agent.model is override
        setup.clients["supervisor"].assert_not_called()
        for name, builder in setup.builders.items():
            assert builder.call_args.args[0].model == f"{name}-model"
            setup.clients[name].assert_called_once()

    @pytest.mark.parametrize("settings_source", ["explicit", "context", "global"])
    def test_fixed_tools_and_injected_dependencies_unchanged(
        self, supervisor_config_setup, monkeypatch, settings_source
    ):
        setup = supervisor_config_setup
        explicit = Settings(_env_file=None, max_model_turns=31)
        global_settings = Settings(_env_file=None, max_model_turns=41)
        monkeypatch.setattr(orchestrator_mod, "get_settings", lambda: global_settings)
        settings = explicit if settings_source == "explicit" else None
        expected = explicit if settings is not None else setup.context.config
        if settings_source == "global":
            monkeypatch.setattr(setup.context, "config", None)
            expected = global_settings
        registry = AgentRegistry(context=setup.context, settings=expected)
        tool_registry = ToolRegistry(generated_tools_dir=setup.path.parent)
        client = MCPClient()
        factories = {}
        for name in orchestrator_mod.SUPERVISOR_TOOL_NAMES:
            attribute = f"get_{name}_mcp_server"
            factories[name] = Mock(wraps=getattr(orchestrator_mod, attribute))
            monkeypatch.setattr(orchestrator_mod, attribute, factories[name])

        agent = orchestrator_mod.create_orchestrator(
            setup.context,
            settings=settings,
            registry=registry,
            tool_registry=tool_registry,
            mcp_client=client,
        )

        assert agent.name == "supervisor_agent"
        assert agent.output_type is None  # 纯文本返回：由控制层手动解析（兼容千问围栏行为）
        assert [tool.name for tool in agent.tools] == [
            "code_agent",
            "subagent_creator",
            "review_agent",
        ]
        assert client.list_servers() == sorted(orchestrator_mod.SUPERVISOR_TOOL_NAMES)
        for name in orchestrator_mod.SUPERVISOR_TOOL_NAMES:
            server = tool_registry.get_mcp_server(name)
            assert server is not None
            assert client.get_server(name) is server
            assert server.get_tool_schema(name) == tool_registry.get_mcp_schema(name)
            kwargs = {"settings": expected}
            if name == "subagent_creator":
                kwargs.update(registry=registry, tool_registry=tool_registry)
            factories[name].assert_called_once_with(setup.context, **kwargs)

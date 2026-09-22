"""单元测试：工具包目录结构与静态资产（prompt.md / api_config.yaml / registry.json）。"""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest

from so_agent.models import GeneratedToolManifest
from so_agent.runtime.config_loader import (
    REQUIRED_FIELDS,
    SECRET_FIELDS,
    load_agent_api_config,
)
from so_agent.runtime.tool_registry import DEFAULT_BUILTIN_AGENTS

PACKAGES_DIR = (
    Path(__file__).resolve().parents[2] / "src" / "so_agent" / "tool_packages"
)
AGENT_PACKAGES = ("code_agent", "subagent_creator", "review_agent")
BUILD_FUNCTIONS = {
    "code_agent": "build_code_tools",
    "subagent_creator": "build_creator_tools",
    "review_agent": "build_review_tools",
}


class TestPackageDirectories:
    def test_tool_packages_root_exists(self):
        assert PACKAGES_DIR.is_dir()
        assert (PACKAGES_DIR / "__init__.py").is_file()

    @pytest.mark.parametrize("package", AGENT_PACKAGES)
    def test_agent_package_exists(self, package):
        package_dir = PACKAGES_DIR / package
        assert package_dir.is_dir()
        assert (package_dir / "__init__.py").is_file()
        assert (package_dir / "agent.py").is_file()

    def test_generated_tools_dir_exists(self):
        assert (PACKAGES_DIR / "generated_tools").is_dir()

    def test_registry_json_exists(self):
        assert (PACKAGES_DIR / "generated_tools" / "registry.json").is_file()

    def test_packages_match_builtin_registry_entries(self):
        assert set(DEFAULT_BUILTIN_AGENTS) == set(AGENT_PACKAGES)


class TestPromptFiles:
    @pytest.mark.parametrize("package", AGENT_PACKAGES)
    def test_prompt_md_exists_and_non_empty(self, package):
        prompt = PACKAGES_DIR / package / "prompt.md"
        assert prompt.is_file()
        text = prompt.read_text(encoding="utf-8")
        assert text.strip()

    @pytest.mark.parametrize("package", AGENT_PACKAGES)
    def test_prompt_utf8_no_bom(self, package):
        raw = (PACKAGES_DIR / package / "prompt.md").read_bytes()
        assert not raw.startswith(b"\xef\xbb\xbf")


class TestApiConfigFiles:
    @pytest.mark.parametrize("package", AGENT_PACKAGES)
    def test_config_loadable(self, package):
        config_path = PACKAGES_DIR / package / "api_config.yaml"
        assert config_path.is_file()
        config = load_agent_api_config(config_path)
        assert config.provider
        assert config.model
        # api_key（直接值）与 api_key_env（环境变量名）至少配置一个
        assert config.api_key or config.api_key_env

    @pytest.mark.parametrize("package", AGENT_PACKAGES)
    def test_required_fields_present_in_raw_yaml(self, package):
        import yaml

        raw = yaml.safe_load(
            (PACKAGES_DIR / package / "api_config.yaml").read_text(encoding="utf-8")
        )
        for field in REQUIRED_FIELDS:
            assert str(raw.get(field) or "").strip(), f"{package} 缺少 {field}"

    @pytest.mark.parametrize("package", AGENT_PACKAGES)
    def test_secret_field_present_in_raw_yaml(self, package):
        import yaml

        raw = yaml.safe_load(
            (PACKAGES_DIR / package / "api_config.yaml").read_text(encoding="utf-8")
        )
        # api_key（直接密钥值）为正式字段，与 api_key_env 至少配置其一
        assert any(
            str(raw.get(field) or "").strip() for field in SECRET_FIELDS
        ), f"{package} 缺少 {list(SECRET_FIELDS)} 中任一密钥配置字段"

    @pytest.mark.parametrize("package", AGENT_PACKAGES)
    def test_secret_configured_directly(self, package):
        config = load_agent_api_config(PACKAGES_DIR / package / "api_config.yaml")
        # 真实配置采用 api_key 直读方式：密钥值已随配置读取，且可正常解析
        assert config.api_key
        assert config.api_key_env is None
        assert config.resolve_api_key() == config.api_key


class TestAgentModules:
    @pytest.mark.parametrize("package", AGENT_PACKAGES)
    def test_module_importable(self, package):
        module = importlib.import_module(f"so_agent.tool_packages.{package}.agent")
        assert module.AGENT_NAME == package

    @pytest.mark.parametrize("package", AGENT_PACKAGES)
    def test_build_function_present(self, package):
        module = importlib.import_module(f"so_agent.tool_packages.{package}.agent")
        build = getattr(module, BUILD_FUNCTIONS[package])
        assert callable(build)

    @pytest.mark.parametrize("package", AGENT_PACKAGES)
    def test_package_path_constants_point_inside_package(self, package):
        module = importlib.import_module(f"so_agent.tool_packages.{package}.agent")
        assert module.PACKAGE_DIR == (PACKAGES_DIR / package).resolve()
        assert module.PROMPT_PATH.is_file()
        assert module.API_CONFIG_PATH.is_file()

    def test_code_agent_generated_tools_constants(self):
        module = importlib.import_module("so_agent.tool_packages.code_agent.agent")
        assert module.GENERATED_TOOLS_DIR == (PACKAGES_DIR / "generated_tools").resolve()
        assert module.GENERATED_REGISTRY_PATH == module.GENERATED_TOOLS_DIR / "registry.json"


class TestGeneratedRegistryJson:
    def test_registry_json_is_valid_array(self):
        raw = json.loads(
            (PACKAGES_DIR / "generated_tools" / "registry.json").read_text(
                encoding="utf-8"
            )
        )
        assert isinstance(raw, list)

    def test_registry_entries_parse_as_manifests(self):
        raw = json.loads(
            (PACKAGES_DIR / "generated_tools" / "registry.json").read_text(
                encoding="utf-8"
            )
        )
        for item in raw:
            manifest = GeneratedToolManifest.model_validate(item)
            assert manifest.tool_name

    def test_generated_tools_init_exists(self):
        assert (PACKAGES_DIR / "generated_tools" / "__init__.py").is_file()


class TestSupervisorFixedEntries:
    def test_three_builtin_entries(self):
        assert set(DEFAULT_BUILTIN_AGENTS) == {
            "code_agent",
            "subagent_creator",
            "review_agent",
        }

    def test_builtin_refs_are_package_paths(self):
        for name, (_description, agent_ref) in DEFAULT_BUILTIN_AGENTS.items():
            assert agent_ref == f"so_agent.tool_packages.{name}"

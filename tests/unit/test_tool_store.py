"""工具库路径解析：默认迁出沙箱，旧配置保持兼容。"""

from __future__ import annotations

from pathlib import Path

from so_agent.config import Settings
from so_agent.context import ProjectContext
from so_agent.runtime.tool_store import (
    resolve_generated_registry_path,
    resolve_generated_tools_dir,
    resolve_tool_store_dir,
)
from so_agent.tool_packages.code_agent import agent as code_agent_agent_mod


def test_default_store_is_outside_sandbox(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    settings = Settings()
    store = resolve_tool_store_dir(settings)
    assert store == (tmp_path / "tools").resolve()
    assert resolve_generated_tools_dir(settings) == store / "generated"
    assert resolve_generated_registry_path(settings) == store / "generated" / "registry.json"


def test_legacy_generated_tools_dir_stays_in_sandbox(tmp_path, monkeypatch):
    monkeypatch.setattr(
        code_agent_agent_mod,
        "GENERATED_TOOLS_DIR",
        code_agent_agent_mod._PACKAGE_GENERATED_TOOLS_DIR,
    )
    monkeypatch.setattr(
        code_agent_agent_mod,
        "GENERATED_REGISTRY_PATH",
        code_agent_agent_mod._PACKAGE_GENERATED_TOOLS_DIR / "registry.json",
    )
    context = ProjectContext(sandbox_dir=tmp_path / "sandbox", project_name="t")
    settings = Settings(generated_tools_dir="custom_tools")
    assert code_agent_agent_mod.resolve_generated_tools_dir(
        context, settings
    ) == (tmp_path / "sandbox" / "custom_tools")
    assert code_agent_agent_mod.resolve_generated_registry_path(
        context, settings
    ) == (tmp_path / "sandbox" / "custom_tools" / "registry.json")


def test_module_override_wins(tmp_path, monkeypatch):
    override = tmp_path / "override"
    monkeypatch.setattr(code_agent_agent_mod, "GENERATED_TOOLS_DIR", override)
    context = ProjectContext(sandbox_dir=tmp_path / "sandbox", project_name="t")
    assert code_agent_agent_mod.resolve_generated_tools_dir(
        context, Settings(tool_store_dir=str(tmp_path / "store"))
    ) == override

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from so_agent.config import Settings
from so_agent.plugins import (
    PluginError,
    PluginLoadError,
    PluginManifest,
    PluginManifestError,
    PluginNotFoundError,
    PluginRegistry,
    load_plugins,
    parse_enabled,
    register_builtins,
    sha256_file,
)
from so_agent.runtime.sandbox import (
    _BACKEND_FACTORIES,
    create_sandbox,
)


@pytest.fixture
def backend_snapshot():
    snapshot = dict(_BACKEND_FACTORIES)
    try:
        yield
    finally:
        _BACKEND_FACTORIES.clear()
        _BACKEND_FACTORIES.update(snapshot)


def _settings(tmp_path: Path, **kwargs) -> Settings:
    values = {
        "plugins_dir": str(tmp_path / "plugins"),
        "plugins_enabled": "",
        "plugins_require_hash": False,
    }
    values.update(kwargs)
    return Settings(_env_file=None, **values)


def _write_plugin(
    root: Path,
    kind: str,
    name: str,
    *,
    entrypoint: str = "plugin.py:factory",
    body: str | None = None,
    manifest_extra: str = "",
) -> Path:
    plugin_dir = root / kind / name
    plugin_dir.mkdir(parents=True)
    code = body or textwrap.dedent(
        """
        def factory(*, context=None, sandbox_dir=None, settings=None,
                    python_executable=None):
            return {"plugin": "__NAME__"}
        """
    ).replace("__NAME__", name)
    (plugin_dir / "plugin.py").write_text(code, encoding="utf-8")
    (plugin_dir / "plugin.yaml").write_text(
        textwrap.dedent(
            f"""
            id: {kind}.{name}
            kind: {kind}
            version: 1.0.0
            entrypoint: {entrypoint}
            capabilities: [execute]
            """
        ).strip()
        + manifest_extra,
        encoding="utf-8",
    )
    return plugin_dir


def test_parse_enabled_variants() -> None:
    assert parse_enabled("") == ()
    assert parse_enabled(None) == ()
    assert parse_enabled(" * ") is None
    assert parse_enabled("sandbox.a, llm.b,sandbox.a") == ("sandbox.a", "llm.b")
    with pytest.raises(PluginError):
        parse_enabled("*,sandbox.a")
    with pytest.raises(PluginManifestError):
        parse_enabled("not-a-plugin")


def test_loads_enabled_sandbox_plugin(tmp_path: Path, backend_snapshot) -> None:
    root = tmp_path / "plugins"
    _write_plugin(root, "sandbox", "fake")
    registry = load_plugins(
        _settings(tmp_path, plugins_enabled="sandbox.fake"), cwd=tmp_path
    )

    assert "fake" in registry.names("sandbox")
    provider = registry.create("sandbox", "fake")
    assert provider == {"plugin": "fake"}
    assert create_sandbox(backend="fake") == {"plugin": "fake"}


def test_disabled_plugins_are_not_imported(tmp_path: Path, backend_snapshot) -> None:
    root = tmp_path / "plugins"
    plugin_dir = _write_plugin(
        root,
        "sandbox",
        "lazy",
        body=textwrap.dedent(
            """
            from pathlib import Path

            Path(__file__).with_name("imported.marker").write_text("1")

            def factory(**kwargs):
                return object()
            """
        ),
    )
    registry = load_plugins(_settings(tmp_path), cwd=tmp_path)

    assert "lazy" not in registry.names("sandbox")
    assert not (plugin_dir / "imported.marker").exists()


def test_unknown_enabled_plugin_fails(tmp_path: Path, backend_snapshot) -> None:
    with pytest.raises(PluginNotFoundError):
        load_plugins(
            _settings(tmp_path, plugins_enabled="sandbox.missing"), cwd=tmp_path
        )


def test_wildcard_loads_all_discovered(tmp_path: Path, backend_snapshot) -> None:
    root = tmp_path / "plugins"
    _write_plugin(root, "sandbox", "one")
    _write_plugin(root, "rules", "two")
    registry = load_plugins(
        _settings(tmp_path, plugins_enabled="*"), cwd=tmp_path
    )

    assert "one" in registry.names("sandbox")
    assert "two" in registry.names("rules")
    assert {m.id for m in registry.manifests() if not m.id.startswith("sandbox.local")}


def test_entrypoint_traversal_rejected(tmp_path: Path, backend_snapshot) -> None:
    root = tmp_path / "plugins"
    plugin_dir = root / "sandbox" / "evil"
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "plugin.yaml").write_text(
        textwrap.dedent(
            """
            id: sandbox.evil
            kind: sandbox
            version: 1.0.0
            entrypoint: ../outside.py:factory
            """
        ),
        encoding="utf-8",
    )
    with pytest.raises(PluginManifestError):
        load_plugins(
            _settings(tmp_path, plugins_enabled="sandbox.evil"), cwd=tmp_path
        )


def test_directory_kind_mismatch_rejected(tmp_path: Path, backend_snapshot) -> None:
    root = tmp_path / "plugins"
    plugin_dir = root / "sandbox" / "wrong"
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "plugin.py").write_text("def factory(**kwargs):\n    return 1\n")
    (plugin_dir / "plugin.yaml").write_text(
        textwrap.dedent(
            """
            id: rules.wrong
            kind: rules
            version: 1.0.0
            entrypoint: plugin.py:factory
            """
        ),
        encoding="utf-8",
    )
    with pytest.raises(PluginError):
        load_plugins(
            _settings(tmp_path, plugins_enabled="rules.wrong"), cwd=tmp_path
        )


def test_sha256_match_and_mismatch(tmp_path: Path, backend_snapshot) -> None:
    root = tmp_path / "plugins"
    plugin_dir = _write_plugin(root, "sandbox", "hashed")
    digest = sha256_file(plugin_dir / "plugin.py")
    manifest_path = plugin_dir / "plugin.yaml"
    manifest_path.write_text(
        manifest_path.read_text(encoding="utf-8") + f'\nsha256: "{digest}"\n',
        encoding="utf-8",
    )

    registry = load_plugins(
        _settings(
            tmp_path,
            plugins_enabled="sandbox.hashed",
            plugins_require_hash=True,
        ),
        cwd=tmp_path,
    )
    assert "hashed" in registry.names("sandbox")

    manifest_path.write_text(
        manifest_path.read_text(encoding="utf-8").replace(digest, "0" * 64),
        encoding="utf-8",
    )
    with pytest.raises(PluginLoadError):
        load_plugins(
            _settings(tmp_path, plugins_enabled="sandbox.hashed"), cwd=tmp_path
        )


def test_require_hash_rejects_missing_digest(tmp_path: Path, backend_snapshot) -> None:
    root = tmp_path / "plugins"
    _write_plugin(root, "sandbox", "plain")
    with pytest.raises(PluginLoadError):
        load_plugins(
            _settings(
                tmp_path,
                plugins_enabled="sandbox.plain",
                plugins_require_hash=True,
            ),
            cwd=tmp_path,
        )


@pytest.mark.parametrize(
    "extra",
    [
        "secrets: ['sk-plaintext-value']\n",
        "api_version: '2'\n",
        "permissions: [root_all]\n",
        "unexpected: true\n",
    ],
)
def test_invalid_manifest_fields_rejected(tmp_path: Path, extra: str) -> None:
    root = tmp_path / "plugins"
    plugin_dir = root / "sandbox" / "bad"
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "plugin.py").write_text("def factory(**kwargs):\n    return 1\n")
    (plugin_dir / "plugin.yaml").write_text(
        textwrap.dedent(
            f"""
            id: sandbox.bad
            kind: sandbox
            version: 1.0.0
            entrypoint: plugin.py:factory
            {extra}
            """
        ),
        encoding="utf-8",
    )
    with pytest.raises((PluginManifestError, PluginError)):
        load_plugins(_settings(tmp_path, plugins_enabled="*"), cwd=tmp_path)


def test_registry_rejects_duplicate_registration() -> None:
    registry = PluginRegistry()
    manifest = PluginManifest(
        id="sandbox.same",
        kind="sandbox",
        version="1.0.0",
        entrypoint="plugin.py:factory",
    )
    registry.register(manifest, lambda **_: None)
    with pytest.raises(PluginError):
        registry.register(manifest, lambda **_: None)


def test_external_plugin_cannot_shadow_builtin(tmp_path: Path, backend_snapshot) -> None:
    root = tmp_path / "plugins"
    _write_plugin(root, "sandbox", "local")
    with pytest.raises(PluginError):
        load_plugins(
            _settings(tmp_path, plugins_enabled="sandbox.local"), cwd=tmp_path
        )


def test_builtin_catalog_covers_all_kinds() -> None:
    registry = PluginRegistry()
    register_builtins(registry)

    assert "local" in registry.names("sandbox")
    assert "openai_compatible" in registry.names("llm")
    assert "directory" in registry.names("mcp")
    assert "builtin" in registry.names("rules")
    assert "events" in registry.names("observability")
    assert all(entry.builtin for entry in registry.entries())

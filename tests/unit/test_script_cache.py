"""同族任务脚本复用：登记、哈希校验、不覆盖与失败留痕。"""

from __future__ import annotations

from pathlib import Path

from so_agent.config import Settings
from so_agent.runtime.events import EVENT_SCRIPT_REUSED, EventLogger
from so_agent.runtime.script_cache import (
    objective_key,
    record_verified_scripts,
    resolve_script_cache_dir,
    stage_cached_scripts,
)

OBJECTIVE = "分析 access.log 并生成 metrics.json"


def _settings(tmp_path: Path, **overrides) -> Settings:
    return Settings(
        _env_file=None,
        tool_store_dir=str(tmp_path / "tools"),
        **overrides,
    )


def _record(sandbox: Path, settings: Settings, events=None, artifacts=None):
    return record_verified_scripts(
        sandbox,
        OBJECTIVE,
        artifacts if artifacts is not None else ["analyze.py"],
        task_id="t-record",
        settings=settings,
        events=events,
    )


class TestObjectiveKey:
    def test_whitespace_is_normalized(self):
        assert objective_key("a  b\nc") == objective_key("a b c")

    def test_different_objectives_differ(self):
        assert objective_key("a") != objective_key("b")


class TestRecord:
    def test_records_python_artifact(self, tmp_path):
        sandbox = tmp_path / "sandbox"
        sandbox.mkdir()
        (sandbox / "analyze.py").write_text("print(1)\n", encoding="utf-8")
        settings = _settings(tmp_path)
        events = EventLogger()

        recorded = _record(sandbox, settings, events)
        assert [entry["path"] for entry in recorded] == ["analyze.py"]

        family = resolve_script_cache_dir(settings) / objective_key(OBJECTIVE)
        index_files = list(family.glob("*.py"))
        assert len(index_files) == 1
        assert index_files[0].read_text(encoding="utf-8") == "print(1)\n"
        assert family.joinpath("index.json").is_file()
        assert events.events[-1]["event_type"] == EVENT_SCRIPT_REUSED
        assert events.events[-1]["payload"]["action"] == "recorded"

    def test_duplicate_artifacts_recorded_once(self, tmp_path):
        sandbox = tmp_path / "sandbox"
        sandbox.mkdir()
        (sandbox / "analyze.py").write_text("print(1)\n", encoding="utf-8")
        settings = _settings(tmp_path)
        events = EventLogger()
        recorded = _record(
            sandbox,
            settings,
            events,
            artifacts=["analyze.py", "analyze.py"],
        )
        assert [entry["path"] for entry in recorded] == ["analyze.py"]
        assert len(events.events[-1]["payload"]["scripts"]) == 1

    def test_ignores_non_python_and_escaping_paths(self, tmp_path):
        sandbox = tmp_path / "sandbox"
        sandbox.mkdir()
        (sandbox / "data.txt").write_text("x", encoding="utf-8")
        settings = _settings(tmp_path)
        recorded = _record(
            sandbox,
            settings,
            artifacts=["data.txt", "../escape.py", "missing.py"],
        )
        assert recorded == []

    def test_disabled_returns_empty(self, tmp_path):
        sandbox = tmp_path / "sandbox"
        sandbox.mkdir()
        (sandbox / "analyze.py").write_text("print(1)\n", encoding="utf-8")
        settings = _settings(tmp_path, code_agent_script_reuse=False)
        assert _record(sandbox, settings) == []
        assert not resolve_script_cache_dir(settings).exists()

    def test_lock_contention_skips_recording(self, tmp_path, monkeypatch):
        from functools import partial

        from so_agent.runtime import script_cache, tool_store

        sandbox = tmp_path / "sandbox"
        sandbox.mkdir()
        (sandbox / "analyze.py").write_text("print(1)\n", encoding="utf-8")
        settings = _settings(tmp_path)
        family = resolve_script_cache_dir(settings) / objective_key(OBJECTIVE)
        family.mkdir(parents=True)
        lock_path = family / "index.json.lock"
        lock_path.write_text("held", encoding="utf-8")
        monkeypatch.setattr(
            script_cache,
            "registry_write_lock",
            partial(tool_store.registry_write_lock, timeout=0.2),
        )

        assert _record(sandbox, settings) == []
        assert not (family / "index.json").exists()
        assert list(family.glob("*.py")) == []


class TestStage:
    def test_stages_verified_script_into_fresh_sandbox(self, tmp_path):
        source = tmp_path / "source"
        source.mkdir()
        (source / "analyze.py").write_text("print(1)\n", encoding="utf-8")
        settings = _settings(tmp_path)
        _record(source, settings)

        target = tmp_path / "target"
        target.mkdir()
        events = EventLogger()
        staged = stage_cached_scripts(
            target, OBJECTIVE, task_id="t-stage", settings=settings, events=events
        )
        assert [entry["path"] for entry in staged] == ["analyze.py"]
        assert (target / "analyze.py").read_text(encoding="utf-8") == "print(1)\n"
        assert events.events[-1]["payload"]["action"] == "staged"

    def test_does_not_overwrite_existing_sandbox_file(self, tmp_path):
        source = tmp_path / "source"
        source.mkdir()
        (source / "analyze.py").write_text("print(1)\n", encoding="utf-8")
        settings = _settings(tmp_path)
        _record(source, settings)

        target = tmp_path / "target"
        target.mkdir()
        (target / "analyze.py").write_text("print(2)\n", encoding="utf-8")
        stage_cached_scripts(target, OBJECTIVE, task_id="t-stage", settings=settings)
        assert (target / "analyze.py").read_text(encoding="utf-8") == "print(2)\n"

    def test_hash_mismatch_is_rejected(self, tmp_path):
        source = tmp_path / "source"
        source.mkdir()
        (source / "analyze.py").write_text("print(1)\n", encoding="utf-8")
        settings = _settings(tmp_path)
        _record(source, settings)

        family = resolve_script_cache_dir(settings) / objective_key(OBJECTIVE)
        cached = next(family.glob("*.py"))
        cached.write_text("print('tampered')\n", encoding="utf-8")

        target = tmp_path / "target"
        target.mkdir()
        events = EventLogger()
        staged = stage_cached_scripts(
            target, OBJECTIVE, task_id="t-stage", settings=settings, events=events
        )
        assert staged == []
        assert not (target / "analyze.py").exists()
        rejected = [
            event
            for event in events.events
            if event["event_type"] == EVENT_SCRIPT_REUSED
            and event["payload"]["action"] == "rejected"
        ]
        assert rejected[0]["payload"]["scripts"][0]["reason"] == "hash_mismatch"

    def test_disabled_returns_empty(self, tmp_path):
        target = tmp_path / "target"
        target.mkdir()
        settings = _settings(tmp_path, code_agent_script_reuse=False)
        assert (
            stage_cached_scripts(target, OBJECTIVE, task_id="t", settings=settings)
            == []
        )

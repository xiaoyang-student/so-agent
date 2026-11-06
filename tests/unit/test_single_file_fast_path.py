"""单元测试：单文件轻量快通道（P3）。

锁定四件事：
- 判定必须保守：多文件、路径越界、运行/网络等动作、否定写入意图
  一律拒绝（返回 None），由调用方回退完整工作流；
- 默认关闭：配置未开启时不构造任何模型客户端；
- 成功路径：一次模型调用生成正文，原子写入沙箱并完成确定性校验，
  模型请求受 ``fast_path_max_output_tokens`` 限制；
- 失败路径：空正文、工具标记、缺关键内容、超限、写入失败均返回
  ``applied=False``，绝不留下半成品或伪造成功。
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import so_agent.paths.single_file as single_file_mod
import so_agent.router as router_mod
from so_agent.config import Settings
from so_agent.context import ProjectContext
from so_agent.models import ClassificationResult, TaskStatus, TaskType
from so_agent.orchestrator import OrchestratorOutput
from so_agent.paths.single_file import (
    FastPathOutcome,
    detect_single_file_task,
    maybe_run_single_file_fast_path,
)
from so_agent.router import TaskRouter
from so_agent.runtime.config_loader import AgentAPIConfig


def make_config() -> AgentAPIConfig:
    return AgentAPIConfig(
        provider="openai",
        model="fake-model",
        base_url="http://localhost/v1",
        timeout=12.0,
        retry=0,
        api_key="sk-test",
    )


def install_fake_client(
    monkeypatch: pytest.MonkeyPatch, content: str
) -> dict:
    captured: dict = {}

    class _Completions:
        async def create(self, **kwargs):
            captured["request"] = kwargs
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content=content)
                    )
                ],
                usage=SimpleNamespace(
                    prompt_tokens=3,
                    completion_tokens=5,
                    total_tokens=8,
                ),
            )

    class _Client:
        def __init__(self) -> None:
            self.chat = SimpleNamespace(completions=_Completions())

        async def close(self) -> None:
            captured["closed"] = True

    monkeypatch.setattr(
        single_file_mod, "build_openai_client", lambda *a, **kw: _Client()
    )
    monkeypatch.setattr(
        single_file_mod, "load_agent_api_config", lambda *a, **kw: make_config()
    )
    return captured


def enabled_settings(**overrides) -> Settings:
    return Settings(
        _env_file=None, fast_path_single_file_enabled=True, **overrides
    )


# ---------------------------------------------------------------------------
# 判定
# ---------------------------------------------------------------------------


class TestDetection:
    def test_single_text_write_is_detected(self):
        task = detect_single_file_task("创建 note.md，写入 FAST_PATH_OK")
        assert task is not None
        assert task.path == "note.md"
        assert task.required_contains == ("FAST_PATH_OK",)

    def test_nested_safe_path_is_detected(self):
        task = detect_single_file_task("在 docs 中新建 docs/guide.md")
        assert task is not None
        assert task.path == "docs/guide.md"

    def test_longest_extension_wins(self):
        task = detect_single_file_task("创建 data.jsonl")
        assert task is not None
        assert task.path == "data.jsonl"

    @pytest.mark.parametrize(
        "objective",
        [
            "创建 a.md 和 b.md",                    # 多个文件
            "把所有文件写入 notes.md",              # 多文件提示
            "创建 ../escape.md",                    # 路径越界
            "创建 C:/tmp/escape.md",                # 绝对路径
            "创建 note.exe",                        # 非文本扩展名
            "创建 .hidden.md",                      # 隐藏文件
            "创建 note.md 并运行它",                # 禁止动作
            "创建 note.md 后上传到服务器",           # 网络动作
            "不要创建 note.md",                     # 否定写入意图
            "解释 note.md 的内容",                  # 无写入意图
            "创建 报告.md 并发送邮件",               # 非文件副作用
        ],
    )
    def test_unsafe_or_ambiguous_is_rejected(self, objective):
        assert detect_single_file_task(objective) is None

    def test_extra_path_in_constraints_is_rejected(self):
        assert (
            detect_single_file_task(
                "创建 a.md", constraints="同时参考 b.md"
            )
            is None
        )


# ---------------------------------------------------------------------------
# 执行
# ---------------------------------------------------------------------------


class TestFastPathExecution:
    async def test_disabled_by_default(self, tmp_path: Path, monkeypatch):
        captured = install_fake_client(monkeypatch, '{"content": "x"}')
        context = ProjectContext(sandbox_dir=tmp_path, project_name="fp-off")
        outcome = await maybe_run_single_file_fast_path(
            context, Settings(_env_file=None), "创建 note.md", "", ""
        )
        assert outcome.applied is False
        assert outcome.reason == "disabled"
        assert "request" not in captured
        assert not (tmp_path / "note.md").exists()

    async def test_not_applicable_skips_model(
        self, tmp_path: Path, monkeypatch
    ):
        captured = install_fake_client(monkeypatch, '{"content": "x"}')
        context = ProjectContext(sandbox_dir=tmp_path, project_name="fp-na")
        outcome = await maybe_run_single_file_fast_path(
            context, enabled_settings(), "解释一下量子计算", "", ""
        )
        assert outcome.applied is False
        assert outcome.reason == "not_applicable"
        assert "request" not in captured

    async def test_success_writes_file_and_emits_events(
        self, tmp_path: Path, monkeypatch
    ):
        captured = install_fake_client(
            monkeypatch, '{"content": "hello\\nFAST_PATH_OK\\n"}'
        )
        context = ProjectContext(
            sandbox_dir=tmp_path, project_name="fp-ok"
        )
        outcome = await maybe_run_single_file_fast_path(
            context,
            enabled_settings(),
            "创建 note.md，内容必须包含 FAST_PATH_OK",
            "",
            "文件内容包含 FAST_PATH_OK",
            task_id="t-fast",
        )
        assert outcome.applied is True
        assert outcome.output is not None
        assert outcome.output.status.value == "completed"
        assert outcome.output.task_id == "t-fast"
        assert (tmp_path / "note.md").read_text(encoding="utf-8") == (
            "hello\nFAST_PATH_OK\n"
        )
        assert captured["request"]["max_tokens"] == 2048
        assert captured["closed"] is True

        events = context.event_logger.get_events(event_type="fast_path")
        statuses = [item["payload"]["status"] for item in events]
        assert statuses == ["hit", "completed"]

    async def test_max_tokens_can_be_disabled(
        self, tmp_path: Path, monkeypatch
    ):
        captured = install_fake_client(monkeypatch, '{"content": "x"}')
        context = ProjectContext(sandbox_dir=tmp_path, project_name="fp-tok")
        await maybe_run_single_file_fast_path(
            context,
            enabled_settings(fast_path_max_output_tokens=0),
            "创建 note.md",
            "",
            "",
        )
        assert "max_tokens" not in captured["request"]

    @pytest.mark.parametrize(
        ("content", "reason"),
        [
            ('{"content": ""}', "empty_content"),
            ("not json at all", "empty_content"),
            ('{"content": "<tool_call>bad</tool_call>"}', "tool_markup"),
        ],
    )
    async def test_rejects_bad_content(
        self, tmp_path: Path, monkeypatch, content, reason
    ):
        install_fake_client(monkeypatch, content)
        context = ProjectContext(sandbox_dir=tmp_path, project_name="fp-bad")
        outcome = await maybe_run_single_file_fast_path(
            context, enabled_settings(), "创建 note.md", "", ""
        )
        assert outcome.applied is False
        assert outcome.reason == reason
        assert not (tmp_path / "note.md").exists()

    async def test_missing_required_text_falls_back(
        self, tmp_path: Path, monkeypatch
    ):
        install_fake_client(monkeypatch, '{"content": "hello"}')
        context = ProjectContext(sandbox_dir=tmp_path, project_name="fp-req")
        outcome = await maybe_run_single_file_fast_path(
            context,
            enabled_settings(),
            "创建 note.md",
            "",
            "内容包含 FAST_PATH_OK",
        )
        assert outcome.applied is False
        assert outcome.reason == "required_text_missing"
        assert not (tmp_path / "note.md").exists()

    async def test_too_large_falls_back(self, tmp_path: Path, monkeypatch):
        install_fake_client(monkeypatch, '{"content": "0123456789"}')
        context = ProjectContext(sandbox_dir=tmp_path, project_name="fp-big")
        outcome = await maybe_run_single_file_fast_path(
            context,
            enabled_settings(fast_path_max_file_bytes=5),
            "创建 note.md",
            "",
            "",
        )
        assert outcome.applied is False
        assert outcome.reason == "too_large"
        assert not (tmp_path / "note.md").exists()

    async def test_model_error_falls_back(self, tmp_path: Path, monkeypatch):
        class _Completions:
            async def create(self, **kwargs):
                raise RuntimeError("boom")

        class _Client:
            def __init__(self) -> None:
                self.chat = SimpleNamespace(completions=_Completions())

            async def close(self) -> None:
                pass

        monkeypatch.setattr(
            single_file_mod, "build_openai_client", lambda *a, **kw: _Client()
        )
        monkeypatch.setattr(
            single_file_mod,
            "load_agent_api_config",
            lambda *a, **kw: make_config(),
        )
        context = ProjectContext(sandbox_dir=tmp_path, project_name="fp-err")
        outcome = await maybe_run_single_file_fast_path(
            context, enabled_settings(), "创建 note.md", "", ""
        )
        assert outcome.applied is False
        assert outcome.reason == "model_error"

    async def test_write_error_falls_back(self, tmp_path: Path, monkeypatch):
        install_fake_client(monkeypatch, '{"content": "hello"}')
        context = ProjectContext(sandbox_dir=tmp_path, project_name="fp-wr")

        def fail_write(root, relative, content):
            raise OSError("disk full")

        monkeypatch.setattr(single_file_mod, "_write_text_atomic", fail_write)
        outcome = await maybe_run_single_file_fast_path(
            context, enabled_settings(), "创建 note.md", "", ""
        )
        assert outcome.applied is False
        assert outcome.reason == "write_error"
        assert not (tmp_path / "note.md").exists()


class TestRouterWiring:
    """四象限路由：FULL_WORKFLOW 分支先试快通道，失败回退且留痕。"""

    def make_router(self, tmp_path, **settings) -> TaskRouter:
        context = ProjectContext(
            sandbox_dir=tmp_path, project_name="router-fast-path"
        )
        return TaskRouter(context, Settings(_env_file=None, **settings))

    async def test_applied_fast_path_short_circuits_workflow(
        self, tmp_path, monkeypatch
    ):
        router = self.make_router(tmp_path)
        output = OrchestratorOutput(
            task_id="t1",
            status=TaskStatus.COMPLETED,
            final_result="done",
        )

        async def fake_fast(*args, **kwargs):
            return FastPathOutcome(
                applied=True,
                output=output,
                reason="completed",
                target="note.md",
            )

        async def should_not_run(*args, **kwargs):
            raise AssertionError("快通道命中后不应进入完整工作流")

        monkeypatch.setattr(
            router_mod, "maybe_run_single_file_fast_path", fake_fast
        )
        monkeypatch.setattr(router, "_run_full_workflow", should_not_run)

        classification = ClassificationResult(
            task_type=TaskType.FULL_WORKFLOW, reasoning="文件写入"
        )
        result = await router.route(
            "创建 note.md", "", "", classification=classification
        )
        assert result is output
        assert any(
            step.stage == "fast_path" and step.status == "hit"
            for step in classification.steps
        )

    async def test_failed_fast_path_falls_back_with_step(
        self, tmp_path, monkeypatch
    ):
        router = self.make_router(tmp_path)

        async def fake_fast(*args, **kwargs):
            return FastPathOutcome(
                applied=False, reason="model_error", target="note.md"
            )

        called: list = []

        async def fake_workflow(objective, constraints, criteria):
            called.append(objective)
            return OrchestratorOutput(
                task_id="t2",
                status=TaskStatus.COMPLETED,
                final_result="workflow",
            )

        monkeypatch.setattr(
            router_mod, "maybe_run_single_file_fast_path", fake_fast
        )
        monkeypatch.setattr(router, "_run_full_workflow", fake_workflow)

        classification = ClassificationResult(
            task_type=TaskType.FULL_WORKFLOW, reasoning="文件写入"
        )
        result = await router.route(
            "创建 note.md", "", "", classification=classification
        )
        assert result.final_result == "workflow"
        assert called == ["创建 note.md"]
        assert any(
            step.stage == "fast_path" and step.status == "fallback"
            for step in classification.steps
        )


class TestAtomicWrite:
    def test_rejects_symlink_ancestor(self, tmp_path: Path):
        outside = tmp_path / "outside"
        outside.mkdir()
        sandbox = tmp_path / "sandbox"
        sandbox.mkdir()
        link = sandbox / "docs"
        try:
            link.symlink_to(outside, target_is_directory=True)
        except (OSError, NotImplementedError):
            pytest.skip("当前环境不支持创建符号链接")
        with pytest.raises((OSError, ValueError)):
            single_file_mod._write_text_atomic(sandbox, "docs/x.md", "x")
        assert not (outside / "x.md").exists()

    def test_creates_parent_directories(self, tmp_path: Path):
        target = single_file_mod._write_text_atomic(
            tmp_path, "a/b/c.md", "content"
        )
        assert target == (tmp_path / "a/b/c.md").resolve()
        assert target.read_text(encoding="utf-8") == "content"

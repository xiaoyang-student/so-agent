"""Web 控制台中英双语测试：语言归一化、快照本地化与前端资源标记。

后端本地化只覆盖界面文案（路由、流程节点、时间线），
用户目标、模型输出与工具描述保持原文。
"""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from so_agent.config import Settings
from so_agent.context import ProjectContext
from so_agent.models import (
    ClassificationResult,
    RouteStep,
    TaskStatus,
    TaskType,
)
from so_agent.orchestrator import OrchestratorOutput
from so_agent.web import manager as web_manager
from so_agent.web.app import create_app
from so_agent.web.manager import TaskManager, WebTask
from so_agent.web.text import normalize_lang, tr

_STATIC = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "so_agent"
    / "web"
    / "static"
    / "index.html"
)


def _make_task(tmp_path: Path) -> tuple[TaskManager, WebTask]:
    manager = TaskManager(tmp_path, max_concurrent=1)
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir(exist_ok=True)
    context = ProjectContext(
        sandbox_dir=sandbox,
        project_name="i18n-test",
        config=Settings(),
    )
    task = WebTask(
        task_id="task-i18n",
        objective="3 + 7",
        constraints="",
        acceptance_criteria="",
        sandbox_dir=sandbox,
        context=context,
        status="completed",
    )
    task.classification = ClassificationResult(
        task_type=TaskType.DIRECT_ANSWER,
        reasoning="纯算术直接回答",
        steps=[RouteStep(stage="rule_check", status="hit", detail="命中纯数学表达式规则")],
    )
    task.output = OrchestratorOutput(
        task_id="engine-task", status=TaskStatus.COMPLETED, final_result="10"
    )
    manager._tasks[task.task_id] = task
    return manager, task


def test_normalize_lang_and_tr():
    assert normalize_lang(None) == "zh"
    assert normalize_lang("") == "zh"
    assert normalize_lang("zh-CN") == "zh"
    assert normalize_lang("en") == "en"
    assert normalize_lang("EN-us") == "en"
    assert tr("zh", "中文", "English") == "中文"
    assert tr("en", "中文", "English") == "English"


def test_snapshot_english_interface_labels(tmp_path):
    manager, task = _make_task(tmp_path)

    zh = manager.snapshot(task.task_id)
    en = manager.snapshot(task.task_id, "en")

    assert zh is not None and en is not None
    assert zh["timeline"][0]["message"] == "任务已提交到 Web 控制台"
    assert en["timeline"][0]["message"] == "Task submitted to the web console"
    assert en["timeline"][1]["message"].startswith("Classification: Direct answer")
    assert zh["route"]["method_label"] == "规则预判"
    assert en["route"]["method_label"] == "Rule pre-check"
    assert en["route"]["effective_label"] == "Direct answer"

    zh_start = next(node for node in zh["journey"]["nodes"] if node["id"] == "start")
    en_start = next(node for node in en["journey"]["nodes"] if node["id"] == "start")
    assert zh_start["title"] == "任务提交"
    assert en_start["title"] == "Task submitted"

    en_rule = next(node for node in en["journey"]["nodes"] if node["id"] == "rule")
    assert en_rule["rows"][0][0] == "Decision method"
    en_end = next(node for node in en["journey"]["nodes"] if node["id"] == "end")
    assert en_end["title"] == "Final answer"


def test_snapshot_language_falls_back_to_chinese(tmp_path):
    manager, task = _make_task(tmp_path)

    snapshot = manager.snapshot(task.task_id, "fr")
    assert snapshot is not None
    assert snapshot["timeline"][0]["message"] == "任务已提交到 Web 控制台"


def test_web_api_lang_query(tmp_path, monkeypatch):
    classification = ClassificationResult(
        task_type=TaskType.DIRECT_ANSWER, reasoning="纯算术直接回答"
    )

    class FakeRouter:
        def __init__(self, *args, **kwargs) -> None:
            pass

        async def classify(self, objective, constraints, acceptance_criteria):
            return classification

        async def route(
            self, objective, constraints, acceptance_criteria, *, classification=None
        ):
            return OrchestratorOutput(
                task_id="engine-task",
                status=TaskStatus.COMPLETED,
                final_result="10",
            )

    monkeypatch.setattr(web_manager, "TaskRouter", FakeRouter)

    app = create_app(settings=Settings(), base_dir=tmp_path)
    with TestClient(app) as client:
        created = client.post("/api/tasks", json={"objective": "3 + 7"})
        assert created.status_code == 202
        task_id = created.json()["task_id"]

        zh = client.get(f"/api/tasks/{task_id}").json()
        en = client.get(f"/api/tasks/{task_id}?lang=en").json()

        assert zh["timeline"][0]["message"] == "任务已提交到 Web 控制台"
        assert en["timeline"][0]["message"] == "Task submitted to the web console"

        missing = client.get("/api/tasks/not-exists?lang=en")
        assert missing.status_code == 404
        assert missing.json()["detail"] == "Task not found"


def test_frontend_bilingual_assets():
    html = _STATIC.read_text(encoding="utf-8")

    assert 'id="lang-btn"' in html
    assert "var I18N = {" in html
    assert '"lang.button"' in html
    assert 'data-i18n="app.title"' in html
    assert 'data-i18n-placeholder="ask.objectivePlaceholder"' in html
    assert "function setLang(" in html
    assert "function applyStaticI18n(" in html
    assert "so-agent-lang" in html
    assert '"lang=" + state.lang' in html
    # 两套字典必须键一致
    assert html.count('"app.title"') >= 2
    assert '"ask.submit": "Start"' in html
    assert '"ask.submit": "开始执行"' in html


def test_frontend_self_hosted_fonts(tmp_path: Path):
    html = _STATIC.read_text(encoding="utf-8")

    assert 'url("static/fonts/host-grotesk-400.woff2")' in html
    assert 'url("static/fonts/atkinson-mono-400.woff2")' in html
    assert "fonts.googleapis.com" not in html

    app = create_app(settings=Settings(), base_dir=tmp_path)
    with TestClient(app) as client:
        served = client.get("/static/fonts/host-grotesk-400.woff2")
        assert served.status_code == 200
        assert served.headers["content-type"] == "font/woff2"
        assert "max-age=86400" in served.headers["cache-control"]

        assert client.get("/static/fonts/index.html").status_code == 404
        assert client.get("/static/fonts/missing.woff2").status_code == 404
        assert client.get("/static/fonts/../../app.py").status_code == 404

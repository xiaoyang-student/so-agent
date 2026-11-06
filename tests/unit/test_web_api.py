"""Web 控制台测试：任务管理、流程图数据、产物安全与 HTTP 接口。

全部使用假路由器 / 假工作流引擎，不触发任何真实模型调用。
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from so_agent.config import Settings
from so_agent.models import (
    ClassificationResult,
    ExecutionResult,
    SubtaskSpec,
    TaskStatus,
    TaskType,
)
from so_agent.orchestrator import OrchestratorOutput
from so_agent.state_machine import HumanGuidanceError
from so_agent.web import manager as web_manager
from so_agent.web.app import create_app
from so_agent.web.manager import ArtifactError, TaskManager


def _fake_router(classification: ClassificationResult, output: OrchestratorOutput):
    class FakeRouter:
        def __init__(self, *args, **kwargs) -> None:
            pass

        async def classify(self, objective, constraints, acceptance_criteria):
            return classification

        async def route(
            self,
            objective,
            constraints,
            acceptance_criteria,
            *,
            classification=None,
        ):
            return output

    return FakeRouter


def _fake_engine_class():
    plan = [
        SubtaskSpec(
            subtask_id="s1",
            title="分析需求",
            instructions="分析用户目标",
            role="analysis",
            dependencies=[],
            allowed_tools=["read_file"],
        ),
        SubtaskSpec(
            subtask_id="s2",
            title="生成报告",
            instructions="根据分析结果生成报告",
            role="code",
            dependencies=["s1"],
            allowed_tools=["write_file"],
        ),
    ]
    results = {
        "s1": ExecutionResult(
            subtask_id="s1", success=True, output="分析完成", duration=1.5
        ),
        "s2": ExecutionResult(
            subtask_id="s2",
            success=True,
            output="报告已生成",
            artifacts=["report.md"],
            duration=2.5,
        ),
    }

    class FakeEngine:
        def __init__(self, context, **kwargs) -> None:
            self.context = context
            self.plan_history = [[spec.model_copy(deep=True) for spec in plan]]
            self.results = {
                key: value.model_copy(deep=True) for key, value in results.items()
            }
            self.plan_version = 1
            self.replan_count = 0
            self.current_status = TaskStatus.EXECUTING
            self.cancelled = False

        async def run(self, objective, constraints, acceptance_criteria):
            sandbox = Path(self.context.sandbox_dir)
            (sandbox / "report.md").write_text("hello report", encoding="utf-8")
            generated = sandbox / "generated_tools"
            generated.mkdir(parents=True, exist_ok=True)
            (generated / "tool.py").write_text("print('x')", encoding="utf-8")
            (sandbox / "task_store.db").write_text("db", encoding="utf-8")
            (sandbox / ".hidden").write_text("secret", encoding="utf-8")
            (sandbox / "page.html").write_text("<script>alert(1)</script>", encoding="utf-8")
            self.current_status = TaskStatus.COMPLETED
            return OrchestratorOutput(
                task_id="engine-task",
                status=TaskStatus.COMPLETED,
                final_result="## 报告\n\n- 子任务 s1 完成\n- 子任务 s2 完成",
                plan_versions=self.plan_history,
            )

        def cancel(self) -> None:
            self.cancelled = True

    return FakeEngine


def _guidance_engine_class():
    """工作流引擎假实现：目标含“人工”时进入人工兜底，否则直接完成。"""
    plan = [
        SubtaskSpec(
            subtask_id="s1",
            title="分析需求",
            instructions="分析用户目标",
            role="analysis",
            dependencies=[],
            allowed_tools=["read_file"],
        ),
        SubtaskSpec(
            subtask_id="s2",
            title="生成报告",
            instructions="根据分析结果生成报告",
            role="code",
            dependencies=["s1"],
            allowed_tools=["write_file"],
        ),
    ]

    class FakeEngine:
        def __init__(self, context, **kwargs) -> None:
            self.context = context
            self.plan_history = [[spec.model_copy(deep=True) for spec in plan]]
            self.results = {}
            self.plan_version = 1
            self.replan_count = 0
            self.current_status = TaskStatus.PLANNING
            self.human_escalation = None
            self.guidance = []
            self.cancelled = False

        async def run(self, objective, constraints, acceptance_criteria):
            if "人工" in objective:
                self.current_status = TaskStatus.HUMAN_REQUIRED
                return OrchestratorOutput(
                    task_id="guidance-task",
                    status=TaskStatus.HUMAN_REQUIRED,
                    final_result="任务需要人工介入",
                    plan_versions=self.plan_history,
                )
            self.current_status = TaskStatus.COMPLETED
            return OrchestratorOutput(
                task_id="guidance-task",
                status=TaskStatus.COMPLETED,
                final_result="直接完成",
                plan_versions=self.plan_history,
            )

        async def resume_with_guidance(self, guidance):
            self.guidance.append(guidance)
            if guidance.instructions == "invalid-target":
                raise HumanGuidanceError("回退目标不在当前计划中")
            self.current_status = TaskStatus.COMPLETED
            return OrchestratorOutput(
                task_id="guidance-task",
                status=TaskStatus.COMPLETED,
                final_result=f"指导后完成：{guidance.action}",
                plan_versions=self.plan_history,
                review_summary="人工指导后恢复",
            )

        def cancel(self) -> None:
            self.cancelled = True

    return FakeEngine


@pytest.fixture()
def fake_guidance_workflow(monkeypatch):
    classification = ClassificationResult(
        task_type=TaskType.FULL_WORKFLOW, reasoning="测试人工指导"
    )
    monkeypatch.setattr(
        web_manager,
        "TaskRouter",
        _fake_router(
            classification,
            OrchestratorOutput(task_id="unused", status=TaskStatus.COMPLETED),
        ),
    )
    monkeypatch.setattr(web_manager, "WorkflowEngine", _guidance_engine_class())
    return classification


@pytest.fixture()
def fake_full_workflow(monkeypatch):
    classification = ClassificationResult(
        task_type=TaskType.FULL_WORKFLOW, reasoning="需要多步骤协作"
    )
    monkeypatch.setattr(
        web_manager, "TaskRouter", _fake_router(classification, OrchestratorOutput(
            task_id="unused", status=TaskStatus.COMPLETED
        ))
    )
    monkeypatch.setattr(web_manager, "WorkflowEngine", _fake_engine_class())
    return classification


async def test_manager_non_workflow_task(tmp_path, monkeypatch):
    output = OrchestratorOutput(
        task_id="engine-1", status=TaskStatus.COMPLETED, final_result="42"
    )
    monkeypatch.setattr(
        web_manager,
        "TaskRouter",
        _fake_router(
            ClassificationResult(
                task_type=TaskType.DIRECT_ANSWER, reasoning="纯算术直接回答"
            ),
            output,
        ),
    )
    manager = TaskManager(tmp_path, max_concurrent=1)
    task = manager.submit("3+7")
    await task.runner

    snapshot = manager.snapshot(task.task_id)
    assert snapshot is not None
    assert snapshot["status"] == "completed"
    assert snapshot["final_result"] == "42"
    assert snapshot["classification"]["task_type"] == "direct_answer"
    versions = snapshot["plan"]["versions"]
    assert len(versions) == 1
    assert versions[0]["nodes"][0]["status"] == "completed"
    assert versions[0]["edges"] == []
    assert any(item["type"] == "classification" for item in snapshot["timeline"])
    await manager.shutdown()


async def test_manager_full_workflow_snapshot_and_artifacts(
    tmp_path, fake_full_workflow
):
    manager = TaskManager(tmp_path, max_concurrent=1)
    task = manager.submit("分析仓库并生成报告")
    await task.runner

    snapshot = manager.snapshot(task.task_id)
    assert snapshot is not None
    assert snapshot["status"] == "completed"
    assert snapshot["phase"] == "completed"
    version = snapshot["plan"]["versions"][0]
    assert [node["id"] for node in version["nodes"]] == ["s1", "s2"]
    assert all(node["status"] == "completed" for node in version["nodes"])
    assert version["edges"] == [{"from": "s1", "to": "s2"}]
    assert version["nodes"][1]["dependencies"] == ["s1"]

    paths = [item["path"] for item in snapshot["artifacts"]]
    assert "report.md" in paths
    assert "page.html" in paths
    assert "generated_tools/tool.py" not in paths
    assert "task_store.db" not in paths
    assert ".hidden" not in paths

    artifact = manager.resolve_artifact(task, "report.md")
    assert artifact.read_text(encoding="utf-8") == "hello report"
    with pytest.raises(ArtifactError):
        manager.resolve_artifact(task, "../secret.txt")
    with pytest.raises(ArtifactError):
        manager.resolve_artifact(task, "generated_tools/tool.py")
    with pytest.raises(ArtifactError):
        manager.resolve_artifact(task, ".hidden")
    await manager.shutdown()


async def test_manager_cancel_marks_task_cancelled(tmp_path, monkeypatch):
    release = asyncio.Event()

    class SlowEngine:
        def __init__(self, context, **kwargs) -> None:
            self.context = context
            self.plan_history = []
            self.results = {}
            self.plan_version = 0
            self.replan_count = 0
            self.current_status = TaskStatus.PLANNING

        async def run(self, objective, constraints, acceptance_criteria):
            await release.wait()
            return OrchestratorOutput(
                task_id="slow", status=TaskStatus.COMPLETED, final_result="done"
            )

        def cancel(self) -> None:
            release.set()

    monkeypatch.setattr(
        web_manager,
        "TaskRouter",
        _fake_router(
            ClassificationResult(
                task_type=TaskType.FULL_WORKFLOW, reasoning="测试取消"
            ),
            OrchestratorOutput(task_id="unused", status=TaskStatus.COMPLETED),
        ),
    )
    monkeypatch.setattr(web_manager, "WorkflowEngine", SlowEngine)
    manager = TaskManager(tmp_path, max_concurrent=1)
    task = manager.submit("慢任务")
    await asyncio.sleep(0)
    assert manager.cancel(task.task_id) is True
    await asyncio.gather(task.runner, return_exceptions=True)
    assert task.status == "cancelled"
    assert manager.cancel(task.task_id) is False
    await manager.shutdown()


def _wait_terminal(client: TestClient, task_id: str, timeout: float = 5.0) -> dict:
    deadline = time.time() + timeout
    detail: dict = {}
    while time.time() < deadline:
        detail = client.get(f"/api/tasks/{task_id}").json()
        if detail.get("status") in {"completed", "failed", "cancelled", "human_required"}:
            return detail
        time.sleep(0.02)
    return detail


def test_web_api_end_to_end(tmp_path, fake_full_workflow):
    app = create_app(settings=Settings(), base_dir=tmp_path)
    with TestClient(app) as client:
        health = client.get("/api/health")
        assert health.status_code == 200
        assert health.json()["status"] == "ok"

        page = client.get("/")
        assert page.status_code == 200
        assert "SO-Agent 控制台" in page.text
        assert "Content-Security-Policy" in page.headers

        blank = client.post("/api/tasks", json={"objective": "   "})
        assert blank.status_code == 400

        created = client.post(
            "/api/tasks",
            json={
                "objective": "分析仓库并生成报告",
                "constraints": "仅使用标准库",
                "acceptance_criteria": "包含报告文件",
            },
        )
        assert created.status_code == 202
        task_id = created.json()["task_id"]

        detail = _wait_terminal(client, task_id)
        assert detail["status"] == "completed"
        assert detail["final_result"].startswith("## 报告")
        assert detail["plan"]["versions"][0]["edges"] == [
            {"from": "s1", "to": "s2"}
        ]
        assert detail["artifacts"]

        listing = client.get("/api/tasks").json()["tasks"]
        assert any(item["task_id"] == task_id for item in listing)

        downloaded = client.get(
            f"/api/tasks/{task_id}/artifacts/report.md"
        )
        assert downloaded.status_code == 200
        assert downloaded.text == "hello report"
        assert downloaded.headers["x-content-type-options"] == "nosniff"
        assert "attachment" in downloaded.headers["content-disposition"]

        inline_html = client.get(
            f"/api/tasks/{task_id}/artifacts/page.html?inline=1"
        )
        # HTML 允许预览，但必须降级为 text/plain，避免同源脚本执行
        assert inline_html.status_code == 200
        assert inline_html.headers["content-type"].startswith("text/plain")

        inline_markdown = client.get(
            f"/api/tasks/{task_id}/artifacts/report.md?inline=1"
        )
        assert inline_markdown.status_code == 200
        assert inline_markdown.headers["content-type"].startswith("text/plain")
        assert "inline" in inline_markdown.headers["content-disposition"]

        missing = client.get("/api/tasks/not-exists")
        assert missing.status_code == 404
        traversal = client.get(
            f"/api/tasks/{task_id}/artifacts/%2e%2e%2fsecret.txt"
        )
        assert traversal.status_code == 404


def test_web_api_errors_honor_language(tmp_path, fake_full_workflow):
    app = create_app(settings=Settings(), base_dir=tmp_path)
    with TestClient(app) as client:
        blank_zh = client.post("/api/tasks", json={"objective": "  "})
        assert blank_zh.status_code == 400
        assert "不能为空" in blank_zh.json()["detail"]

        blank_en = client.post("/api/tasks?lang=en", json={"objective": "  "})
        assert blank_en.status_code == 400
        assert "must not be empty" in blank_en.json()["detail"]

        for response in (
            client.get("/api/tasks/not-exists?lang=en"),
            client.post("/api/tasks/not-exists/cancel?lang=en"),
            client.post(
                "/api/tasks/not-exists/guidance?lang=en",
                json={"action": "resume"},
            ),
        ):
            assert response.status_code == 404
            assert "not found" in response.json()["detail"].lower()

        created = client.post(
            "/api/tasks", json={"objective": "分析仓库并生成报告"}
        )
        task_id = created.json()["task_id"]
        _wait_terminal(client, task_id)

        conflict = client.post(
            f"/api/tasks/{task_id}/guidance?lang=en",
            json={"action": "resume"},
        )
        assert conflict.status_code == 409
        assert "human_required" in conflict.json()["detail"]

        missing_artifact = client.get(
            f"/api/tasks/{task_id}/artifacts/not-exists.txt?lang=en"
        )
        assert missing_artifact.status_code == 404
        assert "not found" in missing_artifact.json()["detail"].lower()

        traversal = client.get(
            f"/api/tasks/{task_id}/artifacts/%2e%2e%2fsecret.txt?lang=en"
        )
        assert traversal.status_code == 404

        internal = client.get(
            f"/api/tasks/{task_id}/artifacts/task_store.db?lang=en"
        )
        assert internal.status_code == 404
        assert "not exposed" in internal.json()["detail"].lower()

        sandbox = app.state.manager.get_task(task_id).sandbox_dir
        (sandbox / "archive.zip").write_bytes(b"PK\x03\x04")
        inline = client.get(
            f"/api/tasks/{task_id}/artifacts/archive.zip?inline=1&lang=en"
        )
        assert inline.status_code == 400
        assert "attachment" in inline.json()["detail"].lower()


def test_web_api_requires_token_when_configured(tmp_path, fake_full_workflow):
    settings = Settings(web_auth_token="secret-token")
    app = create_app(settings=settings, base_dir=tmp_path)
    with TestClient(app) as client:
        assert client.get("/api/tasks").status_code == 401
        assert (
            client.get("/api/tasks", headers={"x-web-token": "wrong"}).status_code
            == 401
        )
        assert (
            client.get(
                "/api/tasks", headers={"x-web-token": "secret-token"}
            ).status_code
            == 200
        )
        # 查询串令牌会泄露到访问日志，API 不再接受
        assert client.get("/api/tasks?token=secret-token").status_code == 401

        client.get("/?token=secret-token")
        assert client.get("/api/tasks").status_code == 200
        cookie = client.cookies.get("so_agent_web_token")
        assert cookie == "secret-token"


def test_web_cookie_secure_only_over_https(tmp_path):
    app = create_app(
        settings=Settings(web_auth_token="secret-token"), base_dir=tmp_path
    )
    with TestClient(app) as client:
        response = client.get("/?token=secret-token")
        assert "secure" not in response.headers["set-cookie"].lower()

        response = client.get(
            "/?token=secret-token", headers={"x-forwarded-proto": "https"}
        )
        assert "secure" in response.headers["set-cookie"].lower()


def test_web_non_loopback_bind_requires_token():
    from so_agent.web import app as web_app

    assert web_app._is_loopback_host("127.0.0.1")
    assert web_app._is_loopback_host("::1")
    assert web_app._is_loopback_host("localhost")
    assert not web_app._is_loopback_host("0.0.0.0")
    assert not web_app._is_loopback_host("192.168.1.10")

    assert web_app.main(["--host", "0.0.0.0", "--token", ""]) == 2


def test_web_api_guidance_resume_flow(tmp_path, fake_guidance_workflow):
    app = create_app(settings=Settings(), base_dir=tmp_path)
    with TestClient(app) as client:
        created = client.post("/api/tasks", json={"objective": "需要人工处理的任务"})
        assert created.status_code == 202
        task_id = created.json()["task_id"]
        assert _wait_terminal(client, task_id)["status"] == "human_required"

        accepted = client.post(
            f"/api/tasks/{task_id}/guidance",
            json={
                "action": "rollback",
                "rollback_to": ["s1"],
                "instructions": "s1 分析遗漏了边界条件，请补充后重跑",
                "additional_info": "参考 docs/spec.md",
            },
        )
        assert accepted.status_code == 202
        assert accepted.json() == {
            "accepted": True,
            "task_id": task_id,
            "action": "rollback",
        }

        resumed = _wait_terminal(client, task_id)
        assert resumed["status"] == "completed"
        assert resumed["final_result"] == "指导后完成：rollback"
        assert resumed["human_guidance"]["rollback_to"] == ["s1"]
        engine = app.state.manager._tasks[task_id].engine
        assert engine.guidance[-1].instructions == "s1 分析遗漏了边界条件，请补充后重跑"

        missing = client.post(
            "/api/tasks/not-exists/guidance", json={"action": "resume"}
        )
        assert missing.status_code == 404

        direct_id = client.post(
            "/api/tasks", json={"objective": "直接完成"}
        ).json()["task_id"]
        assert _wait_terminal(client, direct_id)["status"] == "completed"
        conflict = client.post(
            f"/api/tasks/{direct_id}/guidance", json={"action": "resume"}
        )
        assert conflict.status_code == 409

        too_long = client.post(
            f"/api/tasks/{task_id}/guidance",
            json={"action": "resume", "instructions": "x" * 8001},
        )
        assert too_long.status_code == 422


def test_web_api_invalid_guidance_keeps_human_required(
    tmp_path, fake_guidance_workflow
):
    app = create_app(settings=Settings(), base_dir=tmp_path)
    with TestClient(app) as client:
        task_id = client.post(
            "/api/tasks", json={"objective": "需要人工处理的任务"}
        ).json()["task_id"]
        assert _wait_terminal(client, task_id)["status"] == "human_required"

        accepted = client.post(
            f"/api/tasks/{task_id}/guidance",
            json={"action": "rollback", "instructions": "invalid-target"},
        )
        assert accepted.status_code == 202

        detail = _wait_terminal(client, task_id)
        assert detail["status"] == "human_required"
        assert detail["error"].startswith("人工指导无效")
        assert detail["human_guidance"]["instructions"] == "invalid-target"


def test_web_api_lists_mcp_directory(tmp_path):
    store = tmp_path / "store"
    server_dir = store / "mcp_servers" / "demo"
    server_dir.mkdir(parents=True)
    (server_dir / "server.yaml").write_text(
        "\n".join(
            [
                "name: demo",
                "enabled: false",
                "url: https://mcp.example.com/rpc",
                "allowed_hosts: [mcp.example.com]",
                "allowed_tools: [echo]",
            ]
        ),
        encoding="utf-8",
    )
    app = create_app(
        settings=Settings(tool_store_dir=str(store)), base_dir=tmp_path / "web"
    )
    with TestClient(app) as client:
        payload = client.get("/api/tools/catalog").json()
        assert payload["server_dir"] == str(server_dir.parent)
        assert payload["entries"][0]["name"] == "demo"
        assert payload["entries"][0]["enabled"] is False
        assert payload["entries"][0]["tools"] == ["echo"]

        reloaded = client.post("/api/tools/reload").json()
        assert reloaded["entries"][0]["name"] == "demo"

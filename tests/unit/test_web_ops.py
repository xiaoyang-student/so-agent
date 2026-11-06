"""Web 运维能力：指标导出、工具库管理与前端入口。

全部使用假路由器 / 假工作流引擎与临时工具库，不触发模型调用。
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from so_agent.config import Settings
from so_agent.models import (
    ClassificationResult,
    GeneratedToolManifest,
    TaskStatus,
    TaskType,
)
from so_agent.orchestrator import OrchestratorOutput
from so_agent.runtime.tool_store import resolve_generated_registry_path
from so_agent.web import manager as web_manager
from so_agent.web.app import create_app
from so_agent.web.manager import TaskManager


def _settings(tmp_path: Path) -> Settings:
    # 固定 local：本文件的断言针对验证型沙箱的运维字段，不随宿主是否
    # 安装 Docker 变化（生产默认 auto 由 sandbox 专项测试覆盖）。
    return Settings(
        _env_file=None,
        tool_store_dir=str(tmp_path / "store"),
        sandbox_backend="local",
    )


def _manifest(
    name: str,
    *,
    review_status: str = "approved",
    safety_status: str = "passed",
    consecutive_failures: int = 0,
    safety_issues: list[str] | None = None,
) -> GeneratedToolManifest:
    return GeneratedToolManifest(
        tool_id=f"id-{name}",
        tool_name=name,
        entry_file=f"{name}.py",
        description=f"{name} 工具",
        created_by_task="task-seed",
        review_status=review_status,
        safety_status=safety_status,
        verified=True,
        consecutive_failures=consecutive_failures,
        safety_issues=safety_issues or [],
    )


def _write_registry(
    settings: Settings, manifests: list[GeneratedToolManifest], orphans: list[dict] | None = None
) -> Path:
    path = resolve_generated_registry_path(settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [item.model_dump(mode="json") for item in manifests]
    payload.extend(orphans or [])
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


async def test_metrics_snapshot_aggregates_runtime_facts(tmp_path, monkeypatch):
    class FakeRouter:
        def __init__(self, *args, **kwargs) -> None:
            pass

        async def classify(self, objective, constraints, acceptance_criteria):
            return ClassificationResult(
                task_type=TaskType.FULL_WORKFLOW, reasoning="指标测试"
            )

        async def route(self, *args, **kwargs):
            raise AssertionError("full workflow should not call route")

    class FakeEngine:
        def __init__(self, context, **kwargs) -> None:
            self.context = context
            self.plan_history = []
            self.results = {}
            self.plan_version = 0
            self.replan_count = 0
            self.current_status = TaskStatus.EXECUTING
            self.human_escalation = None

        async def run(self, objective, constraints, acceptance_criteria):
            self.context.event_log.append(
                {"event_type": "tool_called", "payload": {"tool": "read_file"}}
            )
            self.context.event_log.append(
                {"event_type": "human_escalation", "payload": {"reason": "测试"}}
            )
            self.current_status = TaskStatus.COMPLETED
            return OrchestratorOutput(
                task_id="metrics-task",
                status=TaskStatus.COMPLETED,
                final_result="done",
                plan_versions=[],
            )

        def token_budget_snapshot(self):
            return {
                "total_tokens": 120,
                "per_agent": {"supervisor": 100, "coder": 20},
                "violations": [],
            }

        def cancel(self) -> None:
            pass

    monkeypatch.setattr(web_manager, "TaskRouter", FakeRouter)
    monkeypatch.setattr(web_manager, "WorkflowEngine", FakeEngine)
    settings = _settings(tmp_path)
    manager = TaskManager(tmp_path / "web", settings, max_concurrent=1)
    task = manager.submit("统计指标")
    await task.runner

    metrics = manager.metrics_snapshot()
    assert metrics["tasks"]["total"] == 1
    assert metrics["tasks"]["by_status"]["completed"] == 1
    assert metrics["tokens"]["total_tokens"] == 120
    assert metrics["tokens"]["by_agent"]["supervisor"] == 100
    assert metrics["events"]["tool_calls"] == 1
    assert metrics["events"]["human_escalations"] == 1
    assert metrics["sandbox"]["backend"] == "local"
    assert "cloud" in metrics["sandbox"]["available_backends"]
    assert metrics["tools"]["generated_total"] == 0
    assert metrics["escalation"]["webhook_configured"] is False
    await manager.shutdown()


async def test_metrics_snapshot_prefers_token_usage_events(tmp_path, monkeypatch):
    """非工作流模型调用（直答/分类）经事件记账，指标不应回退为 0。"""

    class FakeRouter:
        def __init__(self, *args, **kwargs) -> None:
            pass

        async def classify(self, objective, constraints, acceptance_criteria):
            return ClassificationResult(
                task_type=TaskType.FULL_WORKFLOW, reasoning="事件口径测试"
            )

        async def route(self, *args, **kwargs):
            raise AssertionError("full workflow should not call route")

    class FakeEngine:
        def __init__(self, context, **kwargs) -> None:
            self.context = context
            self.plan_history = []
            self.results = {}
            self.plan_version = 0
            self.replan_count = 0
            self.current_status = TaskStatus.EXECUTING
            self.human_escalation = None

        async def run(self, objective, constraints, acceptance_criteria):
            self.context.event_log.append(
                {
                    "event_type": "token_usage",
                    "payload": {
                        "agent": "router_classifier",
                        "total_tokens": 30,
                    },
                }
            )
            self.context.event_log.append(
                {
                    "event_type": "token_usage",
                    "payload": {"agent": "supervisor", "total_tokens": 90},
                }
            )
            self.current_status = TaskStatus.COMPLETED
            return OrchestratorOutput(
                task_id="metrics-task",
                status=TaskStatus.COMPLETED,
                final_result="done",
                plan_versions=[],
            )

        def token_budget_snapshot(self):
            return {
                "total_tokens": 90,
                "per_agent": {"supervisor": 90},
                "violations": [],
            }

        def cancel(self) -> None:
            pass

    monkeypatch.setattr(web_manager, "TaskRouter", FakeRouter)
    monkeypatch.setattr(web_manager, "WorkflowEngine", FakeEngine)
    manager = TaskManager(tmp_path / "web", _settings(tmp_path), max_concurrent=1)
    task = manager.submit("事件口径")
    await task.runner

    metrics = manager.metrics_snapshot()
    assert metrics["tokens"]["total_tokens"] == 120
    assert metrics["tokens"]["by_agent"] == {
        "router_classifier": 30,
        "supervisor": 90,
    }
    await manager.shutdown()


def test_metrics_endpoints_and_auth(tmp_path):
    settings = _settings(tmp_path)
    app = create_app(settings=settings, base_dir=tmp_path / "web")
    with TestClient(app) as client:
        payload = client.get("/api/metrics").json()
        assert payload["tasks"]["total"] == 0
        assert payload["sandbox"]["isolation"] == "local-validation-only"
        assert "cloud" in payload["sandbox"]["available_backends"]

        text = client.get("/metrics")
        assert text.status_code == 200
        assert text.headers["content-type"].startswith("text/plain")
        assert "so_agent_tasks_total 0" in text.text
        assert "so_agent_generated_tools_total 0" in text.text
        assert "so_agent_mcp_servers 0" in text.text

        health = client.get("/api/health").json()
        assert health["status"] == "ok"
        # 未配置令牌时仅允许回环访问：健康探针保留旧版运维字段
        assert health["auth_required"] is False
        assert health["tasks"]["total"] == 0
        assert health["sandbox_backend"] == "local"
        assert "total_tokens" in health["token_budget"]
        metrics = client.get("/api/metrics").json()
        assert metrics["sandbox"]["backend"] == "local"
        assert "cloud" in metrics["sandbox"]["available_backends"]

    locked = create_app(
        settings=settings.model_copy(update={"web_auth_token": "secret"}),
        base_dir=tmp_path / "web-locked",
    )
    with TestClient(locked) as client:
        assert client.get("/api/metrics").status_code == 401
        assert client.get("/metrics").status_code == 401
        assert client.get("/metrics?token=secret").status_code == 401
        assert (
            client.get(
                "/metrics", headers={"x-web-token": "secret"}
            ).status_code
            == 200
        )
        # 健康探针：未鉴权最小载荷，鉴权后返回兼容字段
        minimal = client.get("/api/health").json()
        assert minimal["status"] == "ok"
        assert minimal["auth_required"] is True
        assert "tasks" not in minimal
        detail = client.get(
            "/api/health", headers={"x-web-token": "secret"}
        ).json()
        assert detail["tasks"]["total"] == 0
        assert detail["sandbox"]["backend"] == "local"
        assert detail["sandbox_backend"] == "local"
        assert "cloud" in detail["available_sandbox_backends"]
        assert "total_tokens" in detail["token_budget"]


def test_ops_paths_render_relative_inside_project(tmp_path, monkeypatch):
    """运维面板路径提示：项目内显示相对路径，便于复制到命令行。"""
    monkeypatch.chdir(tmp_path)
    settings = Settings(_env_file=None, tool_store_dir="store")
    app = create_app(settings=settings, base_dir=tmp_path / "web")
    with TestClient(app) as client:
        generated = client.get("/api/tools/generated").json()
        assert generated["registry"] == "store/generated/registry.json"
        catalog = client.get("/api/tools/catalog").json()
        assert catalog["server_dir"] == "store/mcp_servers"


def test_generated_tool_admin_api_preserves_orphans(tmp_path):
    settings = _settings(tmp_path)
    alpha = _manifest("alpha")
    beta = _manifest(
        "beta",
        safety_status="quarantined",
        consecutive_failures=3,
        safety_issues=["连续失败 3 次，已自动隔离"],
    )
    gamma = _manifest("gamma", review_status="pending", safety_status="unchecked")
    path = _write_registry(
        settings, [alpha, beta, gamma], orphans=[{"broken": True}]
    )

    app = create_app(settings=settings, base_dir=tmp_path / "web")
    with TestClient(app) as client:
        listing = client.get("/api/tools/generated").json()
        assert listing["summary"]["generated_total"] == 3
        assert listing["summary"]["quarantined"] == 1
        assert {item["tool_name"] for item in listing["tools"]} == {
            "alpha",
            "beta",
            "gamma",
        }

        pinned = client.post(
            "/api/tools/generated/alpha/pin", json={"pinned": True}
        )
        assert pinned.status_code == 200
        assert pinned.json()["tool"]["pinned"] is True

        cleared = client.post("/api/tools/generated/beta/clear-quarantine")
        assert cleared.status_code == 200
        assert cleared.json()["tool"]["safety_status"] == "passed"
        assert cleared.json()["tool"]["consecutive_failures"] == 0
        assert cleared.json()["tool"]["safety_issues"] == []

        deprecated = client.post("/api/tools/generated/alpha/deprecate")
        assert deprecated.status_code == 200
        assert deprecated.json()["tool"]["deprecated_at"]

        assert client.post("/api/tools/generated/gamma/deprecate").status_code == 409
        assert (
            client.post("/api/tools/generated/beta/clear-quarantine").status_code
            == 409
        )
        assert (
            client.post("/api/tools/generated/missing/pin", json={"pinned": True})
            .status_code
            == 404
        )

    entries = json.loads(path.read_text(encoding="utf-8"))
    assert {"broken": True} in entries
    alpha_entry = next(item for item in entries if item.get("tool_name") == "alpha")
    assert alpha_entry["pinned"] is True
    assert alpha_entry["deprecated_at"]


def test_frontend_page_exposes_ops_and_guidance(tmp_path):
    app = create_app(settings=_settings(tmp_path), base_dir=tmp_path / "web")
    with TestClient(app) as client:
        page = client.get("/").text
        assert 'id="ops-btn"' in page
        assert 'data-ops-tab="metrics"' in page
        assert 'data-ops-tab="tools"' in page
        assert 'data-ops-tab="mcp"' in page
        assert "/api/metrics" in page
        assert "/api/tools/generated" in page
        assert "renderGuidance" in page
        assert 'id="guidance-submit"' in page
        assert "data-guidance-action" in page
